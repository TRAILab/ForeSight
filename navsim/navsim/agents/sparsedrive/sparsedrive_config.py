from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SparseDriveConfig:
    """Config for the SparseDrive NavSim agent."""

    # Which ForeSight mmcv config to load (relative to the ForeSight repo root).
    # The agent will read this file via mmcv.Config.fromfile to build the head.
    foresight_config: str = "projects/configs/sparsedrive_r50_stage2_4gpu_bs24.py"
    # Optional path to a stage-1/stage-2 checkpoint to warm-start from. Empty string = random init.
    foresight_pretrained: str = ""

    # NavSim-side data shape
    num_history_frames: int = 4         # NavSim default
    num_future_poses: int = 8           # 4 s @ 0.5 s — ego planning horizon
    # Motion head (per-agent) uses a 2× longer horizon than ego planning
    # in the SparseDrive minS2 / nuScenes configs (fut_ts=16 motion vs
    # ego_fut_ts=8). Capped at SceneFilter.num_future_frames if smaller —
    # missing tail timesteps mask=0 in gt_agent_fut_masks.
    motion_fut_ts: int = 16

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

    # Map GT extraction (stage-1 perception). Polylines from
    # `Scene.map_api.get_proximal_map_objects` within `map_roi` metres of
    # the current ego, transformed to ego-local frame, sub-sampled to
    # `map_num_sample` evenly spaced points each. Mirrors the nuScenes
    # `roi_size=(30, 60)` and `num_sample=20` defaults from the SparseDrive
    # configs (60 m forward × 30 m lateral, 20 sample points per line).
    map_roi: Tuple[float, float] = (30.0, 60.0)  # (lateral, forward) meters
    map_num_sample: int = 20

    # Loss flags. Stage-2 planning-only first; flip these on once the stage-1 path
    # for nuPlan boxes is implemented.
    use_planning_loss: bool = True
    use_detection_loss: bool = False
    use_motion_loss: bool = False
    use_map_loss: bool = False

    # Perception freeze. Default True keeps the current planning-only minS2
    # behavior (det/map heads loaded from stage-1 nuScenes ckpt and frozen
    # via requires_grad=False; their losses filtered out of compute_loss).
    # Set False for stage-1 NavSim training where we have real det+map+motion
    # GT and want the heads to actually learn.
    freeze_perception: bool = True

    # Optimizer (mirrors transfuser_agent defaults)
    lr: float = 1e-4
    weight_decay: float = 0.0
    lr_steps: List[int] = field(default_factory=lambda: [50, 75])
    optimizer_type: str = "AdamW"
    scheduler_type: str = "MultiStepLR"
