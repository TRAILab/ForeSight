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
from torchvision import transforms

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

        imgs, lidar2img_mats, image_wh = [], [], []
        for cam in cams:
            assert cam.image is not None, (
                "SparseDriveFeatureBuilder requires all 6 selected cams loaded; "
                "make sure SensorConfig includes them at the current iteration."
            )
            img, scale, top, left = _resize_image(cam.image, cfg.image_target_size)
            imgs.append(transforms.ToTensor()(img))
            lidar2img_mats.append(_build_lidar2img(cam, scale, top, left))
            image_wh.append([cfg.image_target_size[1], cfg.image_target_size[0]])

        img_tensor = torch.stack(imgs, dim=0)  # (num_cams, 3, H, W)

        # SparseDrive expects a 10-dim ego_status (nuScenes layout):
        #   [acc_x, acc_y, acc_z, rot_rate_xyz(3), vel_x, vel_y, vel_z, steer]
        # NavSim ships 2D acc + 2D vel only; pad the unavailable channels with 0.
        ego_status = np.zeros(10, dtype=np.float32)
        ego_status[0:2] = current_status.ego_acceleration[:2]
        ego_status[6:8] = current_status.ego_velocity[:2]
        # ego_status[2] (acc_z), [3:6] (rot_rate), [8] (vel_z), [9] (steer) -> 0

        return {
            "img": img_tensor,
            "projection_mat": torch.tensor(np.stack(lidar2img_mats, axis=0)),
            "image_wh": torch.tensor(image_wh, dtype=torch.float32),
            "ego_status": torch.tensor(ego_status),
            "gt_ego_fut_cmd": torch.tensor(
                current_status.driving_command.astype(np.float32)
            ),
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

        # Det/map GT from the current frame's annotations, mapped into the
        # SparseDrive 11-dim anchor format. The det loss is filtered out of
        # the planning-only total in compute_loss for the first pass; these
        # targets keep the head's training-mode sampler from KeyError-ing.
        frame_idx = scene.scene_metadata.num_history_frames - 1
        annotations = scene.frames[frame_idx].annotations
        boxes_11d, labels = navsim_boxes_to_sparsedrive(annotations)

        return {
            "gt_ego_fut_trajs": deltas,
            "gt_ego_fut_masks": torch.ones(cfg.num_future_poses, dtype=torch.float32),
            "gt_bboxes_3d": torch.from_numpy(boxes_11d),
            "gt_labels_3d": torch.from_numpy(labels),
        }
