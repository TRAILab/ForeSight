"""SparseDrive wrapped behind NavSim's AbstractAgent interface.

This agent is intentionally light: it loads ForeSight's mmcv config, builds the
SparseDrive model via mmcv's registry, and forwards the NavSim feature dict
through its head. The output trajectory is read back into NavSim's Trajectory
dataclass for PDM scoring.

NOTE: This is a Phase 1 scaffold. Stage-2 planning-only is the first target.
The detection / motion / map heads are constructed but their losses are masked
off via SparseDriveConfig flags until the nuPlan annotation conversion lands.
"""

import threading
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

# mmcv's Config.fromfile uses sys.path.insert/pop which is not thread-safe when
# run_pdm_score dispatches 64 worker threads each instantiating SparseDriveAgent
# concurrently. Serialise model construction globally.
_MODEL_BUILD_LOCK = threading.Lock()

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

        # Load stage-1 / pretrained backbone+det+map weights first so those
        # heads start from a meaningful state. A Lightning checkpoint_path
        # (resume) overwrites this, so order matters: pretrained → checkpoint.
        if config.foresight_pretrained:
            self._load_pretrained(config.foresight_pretrained)
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

        # mmcv Config.fromfile uses sys.path.insert/pop which races when 64 eval
        # worker threads each build the model concurrently. Serialise here.
        with _MODEL_BUILD_LOCK:
            # Hydra cd's into its output dir at training time, so the foresight
            # root is no longer in cwd. Add it explicitly so the plugin resolves.
            foresight_root = os.environ.get("FORESIGHT_ROOT", "/workspace/ForeSight")
            if foresight_root not in sys.path:
                sys.path.insert(0, foresight_root)

            # Importing the plugin registers SparseDriveHead, MotionPlanningHead, etc.
            import projects.mmdet3d_plugin  # noqa: F401

            cfg = Config.fromfile(self._config.foresight_config)
            model = build_detector(cfg.model)

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
        # PDM-Score eval invokes compute_trajectory directly (no Lightning
        # wrapper), so the model would otherwise stay on CPU. Move to CUDA
        # if available — the deformable_aggregation_ext op is CUDA-only and
        # falls over with `t == DeviceType::CUDA INTERNAL ASSERT FAILED`
        # when fed CPU tensors.
        if torch.cuda.is_available():
            self.cuda()

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
        # Derive device from the model's params, not from img — eval-mode
        # `compute_trajectory` feeds CPU features but the model lives on
        # CUDA. Lightning's training path moves both consistently, so this
        # also works during training (model+features both on the trainer's
        # device by then).
        device = next(self._sparsedrive_model.parameters()).device
        img = features["img"].to(device)
        data = {k: v for k, v in features.items() if k != "img"}
        if targets is not None:
            data.update(targets)
        # Move every tensor (features + targets) to the model's device.
        for k, v in list(data.items()):
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device)
            elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                data[k] = [x.to(device) for x in v]

        # SparseDrive expects mmcv-style metas. The feature builder now ships
        # per-history-frame T_global / T_global_inv (current-frame as origin,
        # so [-1] is identity by construction); instance_bank only consumes
        # the current frame's per call, so peel that off into img_metas.
        # Detection GT and timestamps are still placeholders — see below.
        bs = img.shape[0]
        if "img_metas" not in data:
            # SparseDrive's instance_bank stacks T_global / T_global_inv via
            # numpy.stack, so they must be CPU numpy arrays, not torch
            # tensors. Keep float64 here: nuPlan global poses have UTM-scale
            # translations (~3.6e5 m); the matmul T_global_inv(curr) @
            # T_global(temp) inside instance_bank needs the precision so the
            # cancellation produces a correct small relative transform.
            # cached_anchor.new_tensor downcasts the result to the GPU dtype.
            import numpy as np
            if "T_global" in data and "T_global_inv" in data:
                # Pop so they don't get forwarded as model kwargs.
                Tg_full = data.pop("T_global")           # (B, n_hist, 4, 4)
                Tg_inv_full = data.pop("T_global_inv")
                Tg_curr = (
                    Tg_full[:, -1].detach().cpu().numpy().astype(np.float64)
                )
                Tg_inv_curr = (
                    Tg_inv_full[:, -1].detach().cpu().numpy().astype(np.float64)
                )
            else:
                eye = np.eye(4, dtype=np.float64)
                Tg_curr = np.broadcast_to(eye, (bs, 4, 4)).copy()
                Tg_inv_curr = Tg_curr.copy()
            data["img_metas"] = [
                {"T_global": Tg_curr[i], "T_global_inv": Tg_inv_curr[i]}
                for i in range(bs)
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
        # PDM scoring needs (ts, 3) trajectories with (x, y, heading); we
        # synthesize heading from the cumulative xy tangent below.
        # Wrap the forward in fp16 autocast at eval time. Weights stay fp32
        # (no model conversion); ops cast to fp16 where safe. ~1.5-2× faster
        # on A100 and halves activation memory, letting more workers fit per
        # GPU. Off when self._eval_fp16=False (default True).
        if getattr(self, "_eval_fp16", True) and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = self._sparsedrive_model(img, **data)
        else:
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
        if not trajs:
            return {"trajectory": torch.zeros(0)}
        trajs_xy = torch.stack(trajs, dim=0)  # (B, ts, 2) cumulative ego-frame xy
        # Heading from tangent. Pad with the current ego pose at the origin
        # so heading[0] points from (0,0) to the first predicted xy — gives
        # SparseDrive's planning a sensible direction at t=0 instead of NaN.
        origin = torch.zeros_like(trajs_xy[..., :1, :])
        xy_with_origin = torch.cat([origin, trajs_xy], dim=-2)        # (B, ts+1, 2)
        delta = xy_with_origin[..., 1:, :] - xy_with_origin[..., :-1, :]  # (B, ts, 2)
        heading = torch.atan2(delta[..., 1], delta[..., 0])           # (B, ts)
        trajs_xyh = torch.cat([trajs_xy, heading.unsqueeze(-1)], dim=-1)  # (B, ts, 3)
        return {"trajectory": trajs_xyh}

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
        gt_deltas = targets["gt_ego_fut_trajs"].to(pred_xy.device)
        gt_xy = torch.cumsum(gt_deltas, dim=-2)
        masks = targets.get("gt_ego_fut_masks", torch.ones_like(gt_xy[..., 0])).to(pred_xy.device)
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
        import os
        # Hydra cds into the experiment output dir before training; checkpoints
        # go there so each run's ckpts are co-located with its logs/config.
        ckpt_dir = os.path.join(os.getcwd(), "checkpoints")
        return [
            pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_dir,
                every_n_epochs=1,
                save_top_k=-1,
                filename="sparsedrive-{epoch:02d}-{step:08d}",
                # Lightning defaults save_on_train_epoch_end to None which
                # resolves to "not _has_val()". The navsim DataModule defines
                # a val dataloader so _has_val()=True; we set
                # limit_val_batches=0 at the trainer level which skips val
                # actually firing — net effect: ModelCheckpoint waits
                # forever for val_epoch_end and never saves. Force on-train
                # save so every epoch's weights land regardless.
                save_on_train_epoch_end=True,
            ),
        ]
