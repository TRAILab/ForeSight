"""Feature and target builders that bridge NavSim's AgentInput / Scene to the
tensor dict that SparseDrive's head expects.

Camera mapping (8 → 6, nuScenes-aligned):
    cam_f0 -> CAM_FRONT
    cam_l0 -> CAM_FRONT_LEFT
    cam_l2 -> CAM_BACK_LEFT
    cam_r0 -> CAM_FRONT_RIGHT
    cam_r2 -> CAM_BACK_RIGHT
    cam_b0 -> CAM_BACK
    cam_l1, cam_r1: dropped (pure-side cameras have no nuScenes counterpart)
"""

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import torch
from shapely import affinity
from shapely.geometry import LineString

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer

from navsim.agents.abstract_agent import AbstractAgent  # noqa: F401  (for type hints)
from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
from navsim.common.dataclasses import AgentInput, Annotations, Cameras, Scene
from navsim.common.enums import BoundingBoxIndex
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)


_NUSC_TO_NAVSIM = (
    "cam_f0",
    "cam_l0",
    "cam_l2",
    "cam_r0",
    "cam_r2",
    "cam_b0",
)

# Spacing between consecutive frames in NavSim/openscene logs.
_NAVSIM_INTERVAL = 0.5  # seconds

# Driving command is a 4-dim onehot (left=0, straight=1, right=2, unknown=3).
# See tools/gen_navsim_kmeans.py: cmd_idx = argmax(driving_command).
_CMD_LEFT, _CMD_RIGHT = 0, 2

# 50% probability for hflip augmentation. Cache-time only — applied
# identically in compute_features and compute_targets via a shared
# per-token seed so the input mirror matches the supervision mirror.
_HFLIP_PROB = 0.5


def _hflip_seed(timestamp_us) -> int:
    """Stable per-token seed shared between feature + target builders.

    Uses Frame.timestamp (microseconds), which is unique per token and
    populated identically into AgentInput.EgoStatus.timestamp (the
    feature-builder path) and Scene.frames[i].ego_status.timestamp /
    Frame.timestamp (the target-builder path). Falls back to 0 — that
    path produces no flip diversity but stays safe.
    """
    if timestamp_us is None:
        return 0
    # 32-bit mask keeps numpy's seed accept happy across versions.
    return int(timestamp_us) & 0xFFFFFFFF


def _decide_hflip(timestamp_us) -> bool:
    """Deterministic per-token hflip choice (must match across builders)."""
    return bool(np.random.default_rng(_hflip_seed(timestamp_us)).random() < _HFLIP_PROB)

# ImageNet stats in [0, 255] uint8 scale — same constants the nuScenes
# pipeline uses (NormalizeMultiviewImage). The SparseDrive ResNet50 backbone
# was pretrained against this distribution; without it inputs land 3-5σ off
# the training distribution and FPN/depth_branch features blow up.
_IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_IMG_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


# SparseDrive map class indices — same as nuScenes config (3 classes).
# `_MAP_LAYERS` maps each class to the nuPlan `SemanticMapLayer` enum
# values it pulls from. CROSSWALK + LANE_CONNECTOR are missing from
# nuScenes but useful in nuPlan; coalesced into the closest SparseDrive
# class so the (3-class) anchors stay reusable.
_MAP_CLASS_PED_CROSSING = 0
_MAP_CLASS_DIVIDER = 1
_MAP_CLASS_BOUNDARY = 2
_MAP_LAYERS: Dict[int, List[SemanticMapLayer]] = {
    _MAP_CLASS_PED_CROSSING: [SemanticMapLayer.CROSSWALK],
    # Lane and lane_connector centerlines act as dividers (gives the planner
    # a sense of which way each lane goes).
    _MAP_CLASS_DIVIDER: [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
    # Walkways approximate sidewalk boundaries; close enough for the
    # nuScenes "boundary" semantic.
    _MAP_CLASS_BOUNDARY: [SemanticMapLayer.WALKWAYS],
}


def _line_to_local(
    line: LineString, origin: StateSE2,
) -> LineString:
    """Transfuser pattern: translate then rotate a shapely geometry into
    the local frame of `origin` (a global SE2)."""
    a = np.cos(origin.heading)
    b = np.sin(origin.heading)
    d = -np.sin(origin.heading)
    e = np.cos(origin.heading)
    translated = affinity.affine_transform(line, [1, 0, 0, 1, -origin.x, -origin.y])
    rotated = affinity.affine_transform(translated, [a, b, d, e, 0, 0])
    return rotated


def _resample_line(line: LineString, num_sample: int) -> npt.NDArray[np.float32]:
    """Sample `num_sample` evenly-spaced points along a shapely LineString.

    Returns (num_sample, 2) float32 in the line's coordinate frame.
    """
    if line.length <= 0 or len(line.coords) < 2:
        return np.zeros((num_sample, 2), dtype=np.float32)
    distances = np.linspace(0.0, line.length, num_sample)
    return np.array(
        [list(line.interpolate(d).coords)[0] for d in distances],
        dtype=np.float32,
    )


def _permute_line(
    line: npt.NDArray[np.float32], padding: float = 1e5,
) -> npt.NDArray[np.float32]:
    """Expand a `(num_pts, 2)` polyline into `(2 * (num_pts - 1), num_pts, 2)`.

    Matches `VectorizeMap.permute_line` in
    `projects/mmdet3d_plugin/datasets/pipelines/vectorize.py`. The
    SparseDrive map head's loss compares predictions against this stacked
    permutation set — orderings the model can equivalently produce —
    using a Hungarian matcher. Closed lines (e.g. crosswalk perimeters)
    use `2 * (num_pts - 1)` rotational + flipped rotations; open lines
    pad with a sentinel `padding` value so the head ignores them.
    """
    is_closed = np.allclose(line[0], line[-1], atol=1e-3)
    num_points = len(line)
    permute_num = num_points - 1
    coords_dim = line.shape[-1]
    out: List[npt.NDArray[np.float32]] = []
    if is_closed:
        pts = line[:-1]
        for shift in range(permute_num):
            out.append(np.roll(pts, shift, axis=0))
        flipped = np.flip(pts, axis=0)
        for shift in range(permute_num):
            out.append(np.roll(flipped, shift, axis=0))
        arr = np.stack(out, axis=0)
        # Re-attach the closing point as a copy of the first point.
        full = np.zeros((permute_num * 2, num_points, coords_dim), dtype=np.float32)
        full[:, :-1] = arr
        full[:, -1] = arr[:, 0]
        return full
    else:
        out.append(line)
        out.append(np.flip(line, axis=0))
        arr = np.stack(out, axis=0)
        # Pad with sentinel — the head's matcher will skip these slots.
        pad = np.full(
            (permute_num * 2 - 2, num_points, coords_dim),
            padding, dtype=np.float32,
        )
        return np.concatenate([arr, pad], axis=0)


def _extract_map_polylines(
    map_api: AbstractMap,
    global_ego_pose: npt.NDArray,
    roi: Tuple[float, float],
    num_sample: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pull lanes / crosswalks / walkways within `roi` of the current ego,
    transform to local frame, vectorize to fixed-length point lists.

    Returns:
        gt_map_labels: (N,) int64 — class index per polyline (0/1/2)
        gt_map_pts:    (N, 2*(num_sample-1), num_sample, 2) float32 —
                       local-frame xy with all permutations the head's
                       Hungarian matcher considers equivalent.
    """
    # Treat global_ego_pose as a 3-vec [x, y, heading] in nuPlan global SE2.
    origin = StateSE2(
        x=float(global_ego_pose[0]),
        y=float(global_ego_pose[1]),
        heading=float(global_ego_pose[2]),
    )
    radius = max(roi)  # one circular query covers the rectangular ROI
    all_layers = sorted(
        {layer for layers in _MAP_LAYERS.values() for layer in layers},
        key=lambda l: l.value,
    )
    obj_dict = map_api.get_proximal_map_objects(
        point=origin.point, radius=radius, layers=all_layers
    )

    labels: List[int] = []
    pts: List[npt.NDArray[np.float32]] = []
    lat_lim, fwd_lim = roi
    for cls, layers in _MAP_LAYERS.items():
        for layer in layers:
            for map_object in obj_dict.get(layer, []):
                # CROSSWALK / WALKWAYS expose `.polygon` (use exterior as
                # the polyline). LANE / LANE_CONNECTOR expose
                # `.baseline_path.linestring` (centerline).
                if layer == SemanticMapLayer.CROSSWALK or layer == SemanticMapLayer.WALKWAYS:
                    line = LineString(map_object.polygon.exterior.coords)
                else:
                    line = map_object.baseline_path.linestring
                local = _line_to_local(line, origin)
                xy = np.asarray(local.coords, dtype=np.float32)
                # Quick ROI cull: drop polylines fully outside the local
                # rectangle. SparseDrive's `CircleObjectRangeFilter` does a
                # circular cull at 55 m for boxes; for map we use the
                # configured rectangle.
                if not (
                    (np.abs(xy[:, 0]) <= lat_lim).any()
                    and (np.abs(xy[:, 1]) <= fwd_lim).any()
                ):
                    continue
                resampled = _resample_line(local, num_sample)  # (num_sample, 2)
                pts.append(_permute_line(resampled))           # (38, 20, 2)
                labels.append(cls)

    permute_num = 2 * (num_sample - 1)
    if not labels:
        return (
            torch.zeros((0,), dtype=torch.long),
            torch.zeros((0, permute_num, num_sample, 2), dtype=torch.float32),
        )
    return (
        torch.tensor(labels, dtype=torch.long),
        torch.from_numpy(np.stack(pts, axis=0)),
    )


def _photo_metric_distortion(
    img: npt.NDArray[np.float32], rng: np.random.Generator
) -> npt.NDArray[np.float32]:
    """Random brightness / contrast / saturation / hue jitter.

    Mirrors `PhotoMetricDistortionMultiViewImage` in
    `projects/mmdet3d_plugin/datasets/pipelines/augment.py`. Each transform
    fires with probability 0.5; constants match the nuScenes pipeline.
    Input/output: float32 RGB in [0, 255], not clipped.

    Cache-time augmentation: each token gets one persisted draw, so this
    gives ~1× extra diversity per cache build (not per-epoch). Acceptable
    given the navsim Lightning DataLoader has no on-the-fly transform hook.
    """
    if rng.random() < 0.5:
        img = img + np.float32(rng.uniform(-32.0, 32.0))

    # contrast either before or after the HSV jitter
    contrast_first = bool(rng.integers(2))
    if not contrast_first and rng.random() < 0.5:
        img = img * np.float32(rng.uniform(0.5, 1.5))

    # HSV ops via cv2 uint8 path (most numerically stable). H ∈ [0, 180],
    # S/V ∈ [0, 255] in cv2's uint8 convention; mmcv's float-HSV uses
    # H ∈ [0, 360], so the nuScenes hue_delta=18 halves to 9 here.
    img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    img_hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    if rng.random() < 0.5:
        img_hsv[..., 1] *= np.float32(rng.uniform(0.5, 1.5))
    if rng.random() < 0.5:
        img_hsv[..., 0] += np.float32(rng.uniform(-9.0, 9.0))
        img_hsv[..., 0] = img_hsv[..., 0] % 180
    img_hsv = np.clip(img_hsv, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB).astype(np.float32)

    if contrast_first and rng.random() < 0.5:
        img = img * np.float32(rng.uniform(0.5, 1.5))

    return img


def _normalize_image(img: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """ImageNet (mean, std) normalization. Input float32 RGB in [0, 255]."""
    return (img - _IMG_MEAN) / _IMG_STD


def _se2_to_se3(pose: npt.NDArray) -> npt.NDArray[np.float64]:
    """Promote a 3-dim SE2 [x, y, heading] to a 4×4 SE3 with z=0 (flat world).

    Returns float64 because nuPlan's absolute global poses are UTM-style
    coordinates with magnitudes ~3.6e5 m; storing T_global as float32 would
    leave only ~1 m of resolution in the translation block. instance_bank
    computes relative transforms (T_global_inv(curr) @ T_global(temp)) where
    the translation difference is small, so the downcast to float32 at the
    GPU boundary loses precision only on the small residual — safe.
    """
    x = float(pose[0])
    y = float(pose[1])
    h = float(pose[2])
    cos_h = np.cos(h)
    sin_h = np.sin(h)
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = cos_h
    T[0, 1] = -sin_h
    T[1, 0] = sin_h
    T[1, 1] = cos_h
    T[0, 3] = x
    T[1, 3] = y
    return T


# nuPlan tracked-object name -> SparseDrive (nuScenes-style) class index. The
# nuScenes class list is fixed at 10 in projects/configs/...; nuPlan ships a
# coarser set so we coalesce. -1 means drop the object entirely.
_NUSC_CLASS_INDEX = {
    "car": 0, "truck": 1, "construction_vehicle": 2, "bus": 3, "trailer": 4,
    "barrier": 5, "motorcycle": 6, "bicycle": 7, "pedestrian": 8, "traffic_cone": 9,
}
_NAVSIM_TO_NUSC_CLASS = {
    "vehicle": _NUSC_CLASS_INDEX["car"],
    "pedestrian": _NUSC_CLASS_INDEX["pedestrian"],
    "bicycle": _NUSC_CLASS_INDEX["bicycle"],
    "traffic_cone": _NUSC_CLASS_INDEX["traffic_cone"],
    "barrier": _NUSC_CLASS_INDEX["barrier"],
    # czone_sign + generic_object don't map cleanly; bucket into barrier so
    # they still contribute to obstacle awareness in the planning-only first
    # pass. Drop ego (it's the agent itself).
    "czone_sign": _NUSC_CLASS_INDEX["barrier"],
    "generic_object": _NUSC_CLASS_INDEX["barrier"],
    "ego": -1,
}


def navsim_boxes_to_sparsedrive(
    annotations: Annotations,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int_], List[int]]:
    """nuPlan Annotations -> SparseDrive raw GT (N, 9) + class indices (N,)
    + the indices into `annotations.*` that survived the class filter.

    The third return value lets callers align motion GT (cross-frame walks
    by `track_tokens`) to the same set of agents as detection GT.

    SparseDrive's training-mode loss path expects RAW boxes in nuScenes layout
    (the head's encode_reg_target handles log-WLH + sin/cos-yaw):
        [X, Y, Z, W, L, H, YAW, VX, VY]
    nuPlan box (BoundingBoxIndex):
        [X, Y, Z, LENGTH, WIDTH, HEIGHT, HEADING]
    Velocity comes from Annotations.velocity_3d (m/s, lidar frame).
    """
    if len(annotations.boxes) == 0:
        return np.zeros((0, 9), dtype=np.float32), np.zeros((0,), dtype=np.int64), []

    out_boxes: List[npt.NDArray[np.float32]] = []
    out_labels: List[int] = []
    keep_idx: List[int] = []
    for i, name in enumerate(annotations.names):
        cls = _NAVSIM_TO_NUSC_CLASS.get(name, -1)
        if cls < 0:
            continue
        b = annotations.boxes[i]
        x = float(b[BoundingBoxIndex.X])
        y = float(b[BoundingBoxIndex.Y])
        z = float(b[BoundingBoxIndex.Z])
        length = max(float(b[BoundingBoxIndex.LENGTH]), 1e-3)
        width = max(float(b[BoundingBoxIndex.WIDTH]), 1e-3)
        height = max(float(b[BoundingBoxIndex.HEIGHT]), 1e-3)
        heading = float(b[BoundingBoxIndex.HEADING])
        vx, vy, _vz = annotations.velocity_3d[i].astype(np.float32).tolist()
        out_boxes.append(
            np.array([x, y, z, width, length, height, heading, vx, vy],
                     dtype=np.float32)
        )
        out_labels.append(cls)
        keep_idx.append(i)

    if not out_boxes:
        return np.zeros((0, 9), dtype=np.float32), np.zeros((0,), dtype=np.int64), []
    return np.stack(out_boxes, axis=0), np.array(out_labels, dtype=np.int64), keep_idx


def _extract_motion_gt(
    scene: Scene,
    keep_idx: List[int],
    fut_ts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-agent future trajectory deltas in the current ego frame.

    For each box that survived the det class filter at the current frame,
    walk `scene.frames[curr+1 : curr+1+fut_ts]` looking up the same
    `track_token`. When found, project that frame's box center into the
    current ego frame (via global SE3) and accumulate deltas. When not
    found (occluded, exited scene, etc.), mask = 0 for that timestep.

    Returns:
        gt_agent_fut_trajs: (N, fut_ts, 2) float32 — per-step xy deltas
        gt_agent_fut_masks: (N, fut_ts) float32 — 1 = valid, 0 = missing
    """
    history = scene.scene_metadata.num_history_frames
    curr_frame = scene.frames[history - 1]
    curr_anns = curr_frame.annotations
    curr_tracks = [curr_anns.track_tokens[i] for i in keep_idx]
    N = len(curr_tracks)

    if N == 0:
        return (
            torch.zeros((0, fut_ts, 2), dtype=torch.float32),
            torch.zeros((0, fut_ts), dtype=torch.float32),
        )

    # T_t→curr in absolute global frame (same convention used by T_global
    # plumbing — global SE2 promoted to SE3 with z=0). Using float64
    # internally to avoid UTM-translation precision loss; convert to
    # float32 only for the final delta tensor.
    curr_pose = np.asarray(curr_frame.ego_status.ego_pose, dtype=np.float64)
    T_curr_global_inv = np.linalg.inv(_se2_to_se3(curr_pose))

    abs_xy = np.zeros((N, fut_ts, 2), dtype=np.float32)
    masks = np.zeros((N, fut_ts), dtype=np.float32)

    for t in range(fut_ts):
        f_idx = history + t
        if f_idx >= len(scene.frames):
            break
        f = scene.frames[f_idx]
        f_pose = np.asarray(f.ego_status.ego_pose, dtype=np.float64)
        T_t2curr = T_curr_global_inv @ _se2_to_se3(f_pose)
        f_tracks = list(f.annotations.track_tokens)
        for i, track in enumerate(curr_tracks):
            if track not in f_tracks:
                continue
            j = f_tracks.index(track)
            b = f.annotations.boxes[j]
            pt = np.array(
                [
                    float(b[BoundingBoxIndex.X]),
                    float(b[BoundingBoxIndex.Y]),
                    float(b[BoundingBoxIndex.Z]),
                    1.0,
                ],
                dtype=np.float64,
            )
            pt_curr = T_t2curr @ pt
            abs_xy[i, t, 0] = float(pt_curr[0])
            abs_xy[i, t, 1] = float(pt_curr[1])
            masks[i, t] = 1.0

    # Deltas in current ego frame; first delta is from origin (current ego)
    # to the first future xy. Carry forward the last valid abs xy through
    # missing timesteps so the delta is 0 (mask=0 anyway, won't backprop).
    for i in range(N):
        last_valid = np.zeros(2, dtype=np.float32)
        for t in range(fut_ts):
            if masks[i, t] == 0:
                abs_xy[i, t] = last_valid
            else:
                last_valid = abs_xy[i, t]

    deltas = np.zeros_like(abs_xy)
    deltas[:, 0] = abs_xy[:, 0]
    deltas[:, 1:] = abs_xy[:, 1:] - abs_xy[:, :-1]

    return torch.from_numpy(deltas), torch.from_numpy(masks)


def _select_cams(cameras: Cameras, cam_names) -> List:
    return [getattr(cameras, n) for n in cam_names]


def _resize_image(image: np.ndarray, target_hw) -> np.ndarray:
    target_h, target_w = target_hw
    h, w = image.shape[:2]
    scale = max(target_h / h, target_w / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (new_w, new_h))
    # center-crop to target
    top = (new_h - target_h) // 2
    left = (new_w - target_w) // 2
    return resized[top : top + target_h, left : left + target_w], scale, top, left


def _build_lidar2img(camera, scale: float, top: int, left: int) -> np.ndarray:
    """Camera intrinsic + sensor2lidar extrinsic → 4x4 lidar→image matrix in
    SparseDrive's convention (lidar2img @ [x, y, z, 1] = [u*z, v*z, z, 1])."""
    K = np.eye(4, dtype=np.float32)
    K[:3, :3] = camera.intrinsics

    # adjust for the resize + crop
    K[0, 0] *= scale
    K[1, 1] *= scale
    K[0, 2] = K[0, 2] * scale - left
    K[1, 2] = K[1, 2] * scale - top

    # sensor2lidar -> lidar2sensor
    R = camera.sensor2lidar_rotation
    t = camera.sensor2lidar_translation
    sensor2lidar = np.eye(4, dtype=np.float32)
    sensor2lidar[:3, :3] = R
    sensor2lidar[:3, 3] = t
    lidar2sensor = np.linalg.inv(sensor2lidar)

    return K @ lidar2sensor


class SparseDriveFeatureBuilder(AbstractFeatureBuilder):
    """Builds the input tensor dict consumed by SparseDrive.forward."""

    def __init__(self, config: SparseDriveConfig):
        self._config = config

    def get_unique_name(self) -> str:
        return "sparsedrive_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        cfg = self._config
        # NavSim convention: ego_statuses[-1] is the "current" frame
        current_cameras = agent_input.cameras[-1]
        current_status = agent_input.ego_statuses[-1]

        cams = _select_cams(current_cameras, _NUSC_TO_NAVSIM)

        # Per-token RNG for photo-metric jitter. Numpy default_rng() draws a
        # fresh seed from OS entropy on each call, so different tokens get
        # different aug — stable within one cache build, varies across builds.
        rng = np.random.default_rng()

        imgs, lidar2img_mats, image_wh = [], [], []
        for cam in cams:
            assert cam.image is not None, (
                "SparseDriveFeatureBuilder requires all 6 selected cams loaded; "
                "make sure SensorConfig includes them at the current iteration."
            )
            img, scale, top, left = _resize_image(cam.image, cfg.image_target_size)
            # img comes from PIL.Image.open → uint8 RGB. Promote to float32
            # in [0, 255] (matches the scale photo-metric and Normalize use)
            # before per-image jitter and ImageNet normalization.
            img = img.astype(np.float32)
            img = _photo_metric_distortion(img, rng)
            img = _normalize_image(img)
            # HWC → CHW torch tensor (no extra division — already normalized).
            imgs.append(torch.from_numpy(np.ascontiguousarray(
                img.transpose(2, 0, 1)
            )))
            lidar2img_mats.append(_build_lidar2img(cam, scale, top, left))
            image_wh.append([cfg.image_target_size[1], cfg.image_target_size[0]])

        img_tensor = torch.stack(imgs, dim=0)  # (num_cams, 3, H, W)

        # Per-history-frame T_global in nuPlan's absolute global frame
        # (z=0 flat-world promotion of the SE2 ego pose). The vendored
        # navsim EgoStatus retains the absolute pose alongside the local
        # one (`global_ego_pose`), so warping in instance_bank /
        # InstanceQueue is consistent across sequential calls (e.g. PDM
        # eval): both calls reference the same global frame, and the
        # cached `T_global` from a previous call lines up with the current
        # one via T_temp2cur = T_global_inv(curr) @ T_global(temp).
        # Fallback: if global_ego_pose is missing (legacy data), use the
        # local pose — that's the current-frame-as-origin convention,
        # functionally equivalent to identity for the current frame.
        n_hist = len(agent_input.ego_statuses)
        T_global = np.empty((n_hist, 4, 4), dtype=np.float64)
        T_global_inv = np.empty((n_hist, 4, 4), dtype=np.float64)
        for i, status in enumerate(agent_input.ego_statuses):
            pose = (
                status.global_ego_pose
                if status.global_ego_pose is not None
                else status.ego_pose
            )
            T_global[i] = _se2_to_se3(pose)
            T_global_inv[i] = np.linalg.inv(T_global[i])

        # rot_rate_z from consecutive local headings: Δheading / 0.5 s.
        # Local poses are fine here — the heading *delta* is invariant to
        # the choice of frame origin. Wrap into (-π, π] for safety.
        if n_hist >= 2:
            h_prev = float(agent_input.ego_statuses[-2].ego_pose[2])
            h_curr = float(agent_input.ego_statuses[-1].ego_pose[2])
            d_h = (h_curr - h_prev + np.pi) % (2 * np.pi) - np.pi
            rot_rate_z = d_h / _NAVSIM_INTERVAL
        else:
            rot_rate_z = 0.0

        # SparseDrive expects a 10-dim ego_status (nuScenes layout):
        #   [acc_x, acc_y, acc_z, rot_rate_xyz(3), vel_x, vel_y, vel_z, steer]
        # NavSim ships 2D acc + 2D vel; rot_rate_z comes from the heading
        # delta above. tire_steering_angle is not exposed via AgentInput
        # (only nuPlan's full EgoState carries it, behind the scene
        # loader), so steer stays zero.
        ego_status = np.zeros(10, dtype=np.float32)
        ego_status[0:2] = current_status.ego_acceleration[:2]
        ego_status[5] = rot_rate_z
        ego_status[6:8] = current_status.ego_velocity[:2]

        # Frame timestamp in seconds (Frame.timestamp is microseconds).
        # instance_bank uses (current.timestamp - cached.timestamp) to gate
        # the temporal warp via max_time_interval; a zero placeholder makes
        # the gate always pass and warps stale anchors across scene
        # boundaries. Float64 because absolute UTC microseconds → seconds
        # is ~1.7e9; the small per-step delta downstream stays well within
        # float32 once the cast happens.
        ts_us = current_status.timestamp if current_status.timestamp is not None else 0
        timestamp = torch.tensor(float(ts_us) / 1e6, dtype=torch.float64)

        return {
            "img": img_tensor,
            "projection_mat": torch.tensor(np.stack(lidar2img_mats, axis=0)),
            "image_wh": torch.tensor(image_wh, dtype=torch.float32),
            "ego_status": torch.tensor(ego_status),
            "gt_ego_fut_cmd": torch.tensor(
                current_status.driving_command.astype(np.float32)
            ),
            "T_global": torch.tensor(T_global),          # (num_history, 4, 4)
            "T_global_inv": torch.tensor(T_global_inv),  # (num_history, 4, 4)
            "timestamp": timestamp,                       # scalar, seconds
        }


class SparseDriveTargetBuilder(AbstractTargetBuilder):
    """Builds the supervision tensor dict consumed by SparseDrive.head.loss."""

    def __init__(self, config: SparseDriveConfig):
        self._config = config

    def get_unique_name(self) -> str:
        return "sparsedrive_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        cfg = self._config
        # ego future trajectory: (num_future_poses, 3) in ego frame, (x, y, heading)
        future = scene.get_future_trajectory(num_trajectory_frames=cfg.num_future_poses)
        poses = torch.tensor(future.poses, dtype=torch.float32)  # (T, 3)

        # SparseDrive expects deltas in (x, y) per step (not absolute), see
        # projects/mmdet3d_plugin/datasets/pipelines/augment.py:traj_rotate.
        deltas = torch.zeros_like(poses[:, :2])
        deltas[0] = poses[0, :2]
        deltas[1:] = poses[1:, :2] - poses[:-1, :2]


        # Real detection GT for stage-1 supervision.
        # nuPlan Annotations (variable N per scene) → SparseDrive 9-dim raw
        # boxes [X, Y, Z, W, L, H, YAW, VX, VY] + class indices, dropping
        # ego + classes that don't map to nuScenes. The variable-length
        # (N_i, 9) tensors are passed through to the model as Python lists
        # by `sparsedrive_collate` — the head's loss path already iterates
        # the per-batch list (see SparseDriveHead.loss → det_head.loss).
        current_frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
        boxes_np, labels_np, keep_idx = navsim_boxes_to_sparsedrive(
            current_frame.annotations
        )
        gt_bboxes_3d = torch.from_numpy(boxes_np)             # (N, 9) float32
        gt_labels_3d = torch.from_numpy(labels_np).long()     # (N,)   int64

        # Per-agent motion GT — cross-frame walk over `track_tokens`,
        # aligned to the same N as gt_bboxes_3d via `keep_idx`.
        gt_agent_fut_trajs, gt_agent_fut_masks = _extract_motion_gt(
            scene, keep_idx, fut_ts=cfg.motion_fut_ts,
        )

        # Map GT — see `_extract_map_polylines` below; cribs the
        # transfuser pattern of `map_api.get_proximal_map_objects` +
        # `.baseline_path.discrete_path`. Returns variable-N polylines as
        # lists, also handled by sparsedrive_collate.
        gt_map_labels, gt_map_pts = _extract_map_polylines(
            scene.map_api,
            current_frame.ego_status.ego_pose,  # global SE2 [x, y, heading]
            roi=cfg.map_roi,
            num_sample=cfg.map_num_sample,
        )

        return {
            "gt_ego_fut_trajs": deltas,
            "gt_ego_fut_masks": torch.ones(cfg.num_future_poses, dtype=torch.float32),
            "gt_bboxes_3d": gt_bboxes_3d,
            "gt_labels_3d": gt_labels_3d,
            "gt_map_labels": gt_map_labels,
            "gt_map_pts": gt_map_pts,
            "gt_agent_fut_trajs": gt_agent_fut_trajs,
            "gt_agent_fut_masks": gt_agent_fut_masks,
        }
