"""SparseDrive wrapped behind NavSim's AbstractAgent interface.

This agent is intentionally light: it loads ForeSight's mmcv config, builds the
SparseDrive model via mmcv's registry, and forwards the NavSim feature dict
through its head. The output trajectory is read back into NavSim's Trajectory
dataclass for PDM scoring.

NOTE: This is a Phase 1 scaffold. Stage-2 planning-only is the first target.
The detection / motion / map heads are constructed but their losses are masked
off via SparseDriveConfig flags until the nuPlan annotation conversion lands.
"""

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
from navsim.agents.sparsedrive.sparsedrive_features import (
    SparseDriveFeatureBuilder,
    SparseDriveTargetBuilder,
)
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)


class SparseDriveAgent(AbstractAgent):
    """NavSim-side wrapper around the ForeSight SparseDrive head."""

    def __init__(
        self,
        config: SparseDriveConfig,
        lr: Optional[float] = None,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        self._config = config
        self._lr = lr if lr is not None else config.lr
        self._checkpoint_path = checkpoint_path

        self._sparsedrive_model: nn.Module = self._build_model_from_foresight_config()

        if checkpoint_path:
            self._load_pretrained(checkpoint_path)

    # ------------------------------------------------------------------ build
    def _build_model_from_foresight_config(self) -> nn.Module:
        """Construct the SparseDrive nn.Module via mmcv's registry."""
        import os
        import sys

        from mmcv import Config
        from mmdet.models import build_detector  # type: ignore

        # Hydra cd's into its output dir at training time, so the foresight root
        # is no longer in cwd. Add it explicitly so the plugin import resolves.
        foresight_root = os.environ.get("FORESIGHT_ROOT", "/workspace/ForeSight")
        if foresight_root not in sys.path:
            sys.path.insert(0, foresight_root)

        # Importing the plugin registers SparseDriveHead, MotionPlanningHead, etc.
        import projects.mmdet3d_plugin  # noqa: F401

        cfg = Config.fromfile(self._config.foresight_config)
        model = build_detector(cfg.model)
        # Mute ForeSight head losses we don't want at this phase.
        # The actual masking lives inside loss flags read on the head; for now we
        # only call self._sparsedrive_model.head.planning_head subpath at loss time.
        return model

    def _load_pretrained(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        # ForeSight stores under top-level keys; navsim agent prefix is added by
        # AgentLightningModule, so we strip both possible prefixes.
        cleaned = {k.replace("agent.", "").replace("_sparsedrive_model.", ""): v
                   for k, v in state_dict.items()}
        missing, unexpected = self._sparsedrive_model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[SparseDriveAgent] missing keys: {len(missing)} (e.g. {missing[:3]})")
        if unexpected:
            print(f"[SparseDriveAgent] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")

    # ------------------------------------------------------- AbstractAgent API
    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if self._checkpoint_path:
            self._load_pretrained(self._checkpoint_path)

    def get_sensor_config(self) -> SensorConfig:
        # Only need the current frame's selected 6 cams; LiDAR off (camera-only
        # SparseDrive variant). Switch to include=True if we ever add lidar.
        cfg = SensorConfig.build_no_sensors()
        for cam_name in self._config.cam_names:
            setattr(cfg, cam_name, [self._config.num_history_frames - 1])
        return cfg

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [SparseDriveFeatureBuilder(self._config)]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [SparseDriveTargetBuilder(self._config)]

    # ----------------------------------------------------- forward + loss
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # SparseDrive expects (img, **data); features dict already has 'img'.
        img = features["img"]
        data = {k: v for k, v in features.items() if k != "img"}
        # Add a minimal projection_mat / image_wh wrapping if needed
        outputs = self._sparsedrive_model(img, **data)
        # `outputs` is a dict from the SparseDrive head. The trajectory lives
        # under outputs["planning"]["trajectory"] in stage-2 inference;
        # see motion_planning_head.post_process. Until that wiring is finalized
        # we expose a canonical "trajectory" key to PDM scoring.
        if "trajectory" in outputs:
            traj = outputs["trajectory"]
        elif "planning" in outputs and isinstance(outputs["planning"], dict):
            traj = outputs["planning"]["trajectory"]
        else:
            raise KeyError(
                "SparseDrive forward did not return a planning trajectory. "
                f"Got top-level keys: {list(outputs.keys())}"
            )
        return {"trajectory": traj}

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # Stage-2 planning-only: simple L2 / smooth-L1 between predicted traj
        # cumulative-sum and the GT delta-cumulative-sum.
        pred = predictions["trajectory"]  # (B, T, 3) or (B, T, 2)
        gt_deltas = targets["gt_ego_fut_trajs"]
        if pred.shape[-1] == 3:
            pred_xy = pred[..., :2]
        else:
            pred_xy = pred
        gt_xy = torch.cumsum(gt_deltas, dim=-2)
        masks = targets.get("gt_ego_fut_masks", torch.ones_like(gt_xy[..., 0]))
        loss = nn.functional.smooth_l1_loss(pred_xy, gt_xy, reduction="none")
        loss = (loss.mean(-1) * masks).sum() / masks.sum().clamp(min=1.0)
        return loss

    # ------------------------------------------------------- optim
    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        optim = torch.optim.AdamW(
            self._sparsedrive_model.parameters(),
            lr=self._lr,
            weight_decay=self._config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim, milestones=self._config.lr_steps, gamma=0.1
        )
        return {"optimizer": optim, "lr_scheduler": scheduler}

    def get_training_callbacks(self) -> List[pl.Callback]:
        return []
