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

from typing import Dict, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import torch

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
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int_]]:
    """nuPlan Annotations -> SparseDrive raw GT (N, 9) + class indices (N,).

    SparseDrive's training-mode loss path expects RAW boxes in nuScenes layout
    (the head's encode_reg_target handles log-WLH + sin/cos-yaw):
        [X, Y, Z, W, L, H, YAW, VX, VY]
    nuPlan box (BoundingBoxIndex):
        [X, Y, Z, LENGTH, WIDTH, HEIGHT, HEADING]
    Velocity comes from Annotations.velocity_3d (m/s, lidar frame).
    """
    if len(annotations.boxes) == 0:
        return np.zeros((0, 9), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    out_boxes: List[npt.NDArray[np.float32]] = []
    out_labels: List[int] = []
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

    if not out_boxes:
        return np.zeros((0, 9), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(out_boxes, axis=0), np.array(out_labels, dtype=np.int64)


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


        # NOTE on det / map / motion targets:
        # PyTorch's default_collate can't stack variable-length per-scene
        # tensors (gt_bboxes_3d shapes like (N_i, 9) where N_i varies). The
        # navsim Lightning runner uses default_collate, so emitting any
        # variable-length target here breaks the dataloader. The agent's
        # forward injects empty placeholders for gt_bboxes_3d / gt_labels_3d
        # / gt_map_* / gt_agent_fut_* and the corresponding losses get
        # filtered out for the planning-only first pass.
        # When we wire real detection supervision, this builder will need to
        # pad to fixed N (or we'll need a custom collate_fn) — left for the
        # stage-1 follow-up.
        return {
            "gt_ego_fut_trajs": deltas,
            "gt_ego_fut_masks": torch.ones(cfg.num_future_poses, dtype=torch.float32),
        }
