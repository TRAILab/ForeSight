from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SparseDriveConfig:
    """Config for the SparseDrive NavSim agent."""

    # Which ForeSight mmcv config to load (relative to the ForeSight repo root).
    # The agent will read this file via mmcv.Config.fromfile to build the head.
    foresight_config: str = "projects/configs/sparsedrive_r50_stage2_4gpu_bs24.py"
    # Optional path to a stage-2 checkpoint to warm-start from. None to train from scratch.
    foresight_pretrained: str = "ckpt/sparsedrive_stage2.pth"

    # NavSim-side data shape
    num_history_frames: int = 4         # NavSim default
    num_future_poses: int = 8           # 4 s @ 0.5 s

    # 8 → 6 camera reduction. Drop pure-side cams; mapping mirrors nuScenes.
    # Order matches projects/mmdet3d_plugin/core/box3d.py CAM order.
    cam_names: Tuple[str, ...] = (
        "cam_f0",  # CAM_FRONT
        "cam_l0",  # CAM_FRONT_LEFT
        "cam_l2",  # CAM_BACK_LEFT
        "cam_r0",  # CAM_FRONT_RIGHT
        "cam_r2",  # CAM_BACK_RIGHT
        "cam_b0",  # CAM_BACK
    )

    # Image resize / crop (matches stage-2 nuScenes preprocessing)
    image_target_size: Tuple[int, int] = (256, 704)  # (H, W) after resize+crop

    # Driving command: NavSim ships a 4-dim onehot (left/straight/right/unknown)
    driving_command_dim: int = 4

    # Loss flags. Stage-2 planning-only first; flip these on once the stage-1 path
    # for nuPlan boxes is implemented.
    use_planning_loss: bool = True
    use_detection_loss: bool = False
    use_motion_loss: bool = False
    use_map_loss: bool = False

    # Optimizer (mirrors transfuser_agent defaults)
    lr: float = 1e-4
    weight_decay: float = 0.0
    lr_steps: List[int] = field(default_factory=lambda: [50, 75])
    optimizer_type: str = "AdamW"
    scheduler_type: str = "MultiStepLR"
