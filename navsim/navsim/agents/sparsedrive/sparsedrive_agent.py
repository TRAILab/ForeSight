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

        # Planning-only first pass: det/map heads run forward (motion_plan_head
        # still consumes their outputs via cross-attn) but are FROZEN so their
        # weights don't drift under gradients from planning_loss. With empty
        # det/map GT, their unfrozen weights diverge to NaN within ~500 steps.
        # When real det/map GT is wired through, drop this freeze.
        head = self._sparsedrive_model.head
        for name in ("det_head", "map_head"):
            if hasattr(head, name):
                for p in getattr(head, name).parameters():
                    p.requires_grad = False

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
    #
    # NavSim's AgentLightningModule._step calls:
    #     prediction = self.agent.forward(features, targets)
    #     loss_dict  = self.agent.compute_loss(features, targets, prediction)
    #     return loss_dict['loss']     # backprop scalar
    # Each k in loss_dict is logged as f"{train|val}/{k}".
    #
    # SparseDrive (mmcv-style) bakes its losses into forward_train, expecting
    # gt_* keys inside the data dict. We bridge by:
    #   - in training: merging features+targets into data, calling the model,
    #     and returning the resulting loss dict as "prediction"
    #   - in eval:    calling simple_test, pulling the trajectory out of the
    #     planning_decoder result for PDM scoring
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        img = features["img"]
        data = {k: v for k, v in features.items() if k != "img"}
        if targets is not None:
            data.update(targets)

        # SparseDrive expects mmcv-style metas. Until the feature builder ships
        # T_global, timestamps, and detection GT from the scene, inject
        # placeholders so the head's training-mode forward runs end-to-end with
        # task_config.with_det=False / with_motion=False / planning-only loss.
        bs = img.shape[0]
        if "img_metas" not in data:
            # SparseDrive's instance_bank stacks T_global / T_global_inv via
            # numpy.stack, so they must be CPU numpy arrays, not torch tensors.
            import numpy as np
            eye = np.eye(4, dtype=np.float32)
            data["img_metas"] = [
                {"T_global": eye.copy(), "T_global_inv": eye.copy()}
                for _ in range(bs)
            ]
        if "timestamp" not in data:
            data["timestamp"] = img.new_zeros(bs)
        # Empty per-batch GT placeholders for all three heads. Real nuPlan
        # box / map polyline conversion is a Phase-2 stage-1 follow-up; here
        # they only keep the heads' sampling path from KeyError-ing during
        # training. The corresponding losses are filtered to zero in
        # compute_loss for the planning-only first pass.
        if "gt_labels_3d" not in data:
            data["gt_labels_3d"] = [
                img.new_zeros(0, dtype=torch.long) for _ in range(bs)
            ]
        if "gt_bboxes_3d" not in data:
            data["gt_bboxes_3d"] = [img.new_zeros(0, 9) for _ in range(bs)]
        if "gt_map_labels" not in data:
            data["gt_map_labels"] = [
                img.new_zeros(0, dtype=torch.long) for _ in range(bs)
            ]
        if "gt_map_pts" not in data:
            data["gt_map_pts"] = [img.new_zeros(0, 20, 2) for _ in range(bs)]
        if "map_instance_id" not in data:
            data["map_instance_id"] = [
                img.new_zeros(0, dtype=torch.long) for _ in range(bs)
            ]
        # Motion head needs per-agent future trajectories. Real per-agent
        # tracking through time isn't yet wired in the target builder, so for
        # the planning-only first pass we feed empty agent lists. Their motion
        # losses become zero contributions on the all-zero numerator.
        # Note: motion head's fut_ts uses the head's own constant (16 for
        # navsim variant) so we infer from the model.
        fut_ts = self._sparsedrive_model.head.motion_plan_head.fut_ts
        if "gt_agent_fut_trajs" not in data:
            data["gt_agent_fut_trajs"] = [
                img.new_zeros(0, fut_ts, 2) for _ in range(bs)
            ]
        if "gt_agent_fut_masks" not in data:
            data["gt_agent_fut_masks"] = [
                img.new_zeros(0, fut_ts) for _ in range(bs)
            ]

        if self.training:
            # forward_train returns a dict of named loss tensors.
            return self._sparsedrive_model(img, **data)

        # eval: simple_test -> List[{"img_bbox": merged_result_dict}] per batch
        # item. The merged dict has {"planning": (num_cmds, modes, ts, 2),
        # "final_planning": (ts, 2), "planning_score": ..., ...}.
        # PDM scoring needs the (ts, 3) trajectory — final_planning gives
        # (ts, 2) and we'll synthesize heading from the trajectory tangent
        # downstream once we wire the eval pipeline.
        outputs = self._sparsedrive_model(img, **data)
        trajs = []
        for sample in outputs:
            res = sample.get("img_bbox", sample)
            if isinstance(res, dict) and "final_planning" in res:
                trajs.append(res["final_planning"])
            elif isinstance(res, dict) and "planning" in res:
                trajs.append(res["planning"])
            else:
                raise KeyError(
                    f"SparseDrive post_process did not produce a planning key; "
                    f"got top-level keys: "
                    f"{list(res.keys()) if isinstance(res, dict) else type(res)}"
                )
        return {"trajectory": torch.stack(trajs, dim=0) if trajs else torch.zeros(0)}

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # In training, predictions IS the SparseDrive loss dict. Sum scalars
        # into the canonical 'loss' key the lightning module backprops on.
        # NOTE on det/map filtering: SparseDrive's det/map heads compute losses
        # against per-batch GT lists that the navsim path currently injects as
        # empty placeholders. Empty GT + 900 anchors makes the head try to
        # classify everything as background, producing massive (~8.5e3 per
        # decoder layer) loss that overflows fp16 / diverges in fp32. Until
        # real det/map GT is wired through, exclude their losses from the
        # backprop total. det/map heads still get gradients via the path
        # planning_loss -> planning_head -> cross-attn over det features, so
        # they're learned (just not directly supervised by classifying empty
        # anchor sets).
        if isinstance(predictions, dict):
            loss_terms = {
                k: v
                for k, v in predictions.items()
                if torch.is_tensor(v)
                and v.dim() == 0
                and v.requires_grad
                and not k.startswith("det_loss")
                and not k.startswith("map_loss")
            }
            if loss_terms:
                total = sum(loss_terms.values())
                return {**{k: v.detach() for k, v in predictions.items()
                           if torch.is_tensor(v)}, "loss": total}

        # Eval / fallback: simple smooth-L1 between predicted cumulative xy and
        # the GT cumulative-sum of delta trajectories.
        pred = predictions.get("trajectory") if isinstance(predictions, dict) else None
        if pred is None or not torch.is_tensor(pred):
            return {"loss": torch.zeros((), requires_grad=False)}
        if pred.shape[-1] == 3:
            pred_xy = pred[..., :2]
        else:
            pred_xy = pred
        gt_deltas = targets["gt_ego_fut_trajs"]
        gt_xy = torch.cumsum(gt_deltas, dim=-2)
        masks = targets.get("gt_ego_fut_masks", torch.ones_like(gt_xy[..., 0]))
        loss = nn.functional.smooth_l1_loss(pred_xy, gt_xy, reduction="none")
        loss = (loss.mean(-1) * masks).sum() / masks.sum().clamp(min=1.0)
        return {"loss": loss}

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
