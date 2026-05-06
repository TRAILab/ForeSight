"""
Bare-minimum ego-only planner that mirrors the components of SparseDrive's
MotionPlanningHead but strips out everything that depends on perception:

  * No detection / map cross-attention (no det K/V, no map K/V).
  * No motion decoder, no agent-token cross-attention, no instance bank.
  * No temporal cross-attention, no instance queue.
  * No deformable image attention.
  * No collision rescore (no detection output to rescore against).

What stays (verbatim from SparseDrive's planner path):

  * Plan-mode anchor parameter (3 cmds x M modes x T steps x 2).
  * sineembed -> MLP plan-anchor encoder.
  * Ego-status MLP that broadcast-adds to plan_mode_query (egostatus lever).
  * Self-attention over plan modes (mirrors the planner's `gnn` op).
  * FFN + norm interleaved (mirrors the planner's `ffn` + `norm` ops).
  * Per-mode trajectory regression and classification heads (plan_reg_branch,
    plan_cls_branch, plan_status_branch from MotionPlanningRefinementModule).

Loss path mirrors MotionPlanningHead.loss_planning at the bare minimum:
plan_loss_cls (FocalLoss on per-cmd mode picks), plan_loss_reg (L1 on the
best-matched mode trajectory), and plan_loss_status (L1 on next-frame ego
state). PlanningTarget is reused unchanged for cmd-conditioned mode sampling.

This is the lower-bound floor for the Decoupling Supervision vs. Planning
Context table. It establishes what plain ego-state planning achieves on the
SparseDrive codebase before any perception interface or supervision is
introduced.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import BaseModule, force_fp32
from mmdet.core import reduce_mean
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.models import HEADS, build_loss

from .attention import gen_sineembed_for_position
from .blocks import linear_relu_ln


@HEADS.register_module()
class EgoPlannerHead(BaseModule):
    """Minimal ego-only planner head. See module docstring for scope."""

    def __init__(
        self,
        ego_fut_ts: int = 6,
        ego_fut_mode: int = 6,
        num_driving_cmds: int = 3,
        plan_anchor: Optional[str] = None,
        embed_dims: int = 256,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        ffn_feedforward_channels: int = 512,
        ffn_drop: float = 0.1,
        ego_status_dim: int = 5,
        ego_status_indices: Optional[List[int]] = None,
        planning_sampler: Optional[Dict] = None,
        plan_loss_cls: Optional[Dict] = None,
        plan_loss_reg: Optional[Dict] = None,
        plan_loss_status: Optional[Dict] = None,
        init_cfg: Optional[Dict] = None,
    ):
        super().__init__(init_cfg)
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.num_driving_cmds = num_driving_cmds
        self.embed_dims = embed_dims
        self.num_decoder_layers = num_decoder_layers
        self.ego_status_dim = ego_status_dim
        # Defaults to MotionPlanningHead's egostatus lever:
        #   ego_status[:, [0, 1, 5, 6, 7]] = planar accel (x, y), yaw rate,
        #   planar velocity (x, y). Same 5-dim slice.
        if ego_status_indices is None:
            ego_status_indices = [0, 1, 5, 6, 7]
        self.ego_status_indices = list(ego_status_indices)
        assert len(self.ego_status_indices) == ego_status_dim, (
            f"ego_status_indices length {len(self.ego_status_indices)} "
            f"must equal ego_status_dim {ego_status_dim}"
        )

        # Plan anchors: (num_driving_cmds, ego_fut_mode, ego_fut_ts, 2). Same
        # data file as the baseline planner.
        if plan_anchor is None:
            raise ValueError(
                "EgoPlannerHead requires plan_anchor (path to the kmeans .npy)"
            )
        anchor = np.load(plan_anchor)
        anchor = torch.tensor(anchor, dtype=torch.float32)
        assert anchor.shape == (num_driving_cmds, ego_fut_mode, ego_fut_ts, 2), (
            f"plan_anchor shape {tuple(anchor.shape)} != expected "
            f"({num_driving_cmds}, {ego_fut_mode}, {ego_fut_ts}, 2)"
        )
        self.plan_anchor = nn.Parameter(anchor, requires_grad=True)

        # Sine-embed the endpoint of each plan anchor, then MLP -> embed_dims.
        # Mirrors MotionPlanningHead.plan_anchor_encoder.
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        # Ego-status MLP. Mirrors MotionPlanningHead.plan_ego_status_encoder
        # (egostatus lever): 5-d ego status -> embed_dims, broadcast-added to
        # every plan_mode_query at init and after each refine.
        self.plan_ego_status_encoder = nn.Sequential(
            nn.Linear(ego_status_dim, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
        )

        # Self-attention layers over plan modes. No cross-attention to anything
        # external; this is the bare-minimum decoder stack.
        self.self_attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.attn_norms = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        for _ in range(num_decoder_layers):
            self.self_attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=embed_dims,
                    num_heads=num_heads,
                    dropout=ffn_drop,
                    batch_first=True,
                )
            )
            self.attn_norms.append(nn.LayerNorm(embed_dims))
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(embed_dims, ffn_feedforward_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(ffn_drop),
                    nn.Linear(ffn_feedforward_channels, embed_dims),
                    nn.Dropout(ffn_drop),
                )
            )
            self.ffn_norms.append(nn.LayerNorm(embed_dims))

        # Refinement heads. Same shapes as MotionPlanningRefinementModule's
        # plan-side branches.
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 2),
        )
        self.plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10),
        )

        # PlanningTarget for cmd-conditioned mode sampling.
        if planning_sampler is None:
            planning_sampler = dict(
                type="PlanningTarget",
                ego_fut_ts=ego_fut_ts,
                ego_fut_mode=ego_fut_mode,
            )
        self.planning_sampler = BBOX_SAMPLERS.build(planning_sampler)

        # Losses.
        if plan_loss_cls is None:
            plan_loss_cls = dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.5,
            )
        if plan_loss_reg is None:
            plan_loss_reg = dict(type="L1Loss", loss_weight=1.0)
        if plan_loss_status is None:
            plan_loss_status = dict(type="L1Loss", loss_weight=1.0)
        self.plan_loss_cls = build_loss(plan_loss_cls)
        self.plan_loss_reg = build_loss(plan_loss_reg)
        self.plan_loss_status = build_loss(plan_loss_status)

    def init_weights(self):
        # Bias toward "no-collide" prior, same as MotionPlanningRefinementModule.
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    # ------------------------------------------------------------------ build
    def _build_plan_query(self, bs: int, metas: Dict, device, dtype) -> torch.Tensor:
        """Construct the per-batch plan-mode queries.

        Returns: (bs, num_driving_cmds * ego_fut_mode, embed_dims).
        """
        # plan_anchor: (cmds, M, T, 2) -> per-batch tile.
        plan_anchor = self.plan_anchor[None].expand(bs, -1, -1, -1, -1)
        # Sine-embed the endpoint of each mode (mirrors MotionPlanningHead).
        endpoint = plan_anchor[..., -1, :]  # (bs, cmds, M, 2)
        plan_pos = gen_sineembed_for_position(endpoint, hidden_dim=self.embed_dims)
        plan_mode_query = self.plan_anchor_encoder(plan_pos)  # (bs, cmds, M, D)

        # Ego status broadcast-add (egostatus lever).
        ego_status_full = metas["ego_status"].to(dtype)
        es = ego_status_full[:, self.ego_status_indices]
        ego_embed = self.plan_ego_status_encoder(es)  # (bs, D)
        plan_mode_query = plan_mode_query + ego_embed[:, None, None, :]

        # Flatten cmd x mode -> single sequence dim.
        plan_mode_query = plan_mode_query.flatten(1, 2)
        return plan_mode_query

    # ---------------------------------------------------------------- forward
    def forward(self, feature_maps, metas: Dict):
        """feature_maps is unused; planner is ego-only by construction."""
        ego_status = metas["ego_status"]
        bs = ego_status.shape[0]
        device = ego_status.device
        # Use plan_anchor's dtype so we stay consistent with FP32 anchors even
        # if the rest of the model is in fp16.
        dtype = self.plan_anchor.dtype

        plan_mode_query = self._build_plan_query(bs, metas, device, dtype)

        # Self-attn + FFN stack.
        for layer_idx in range(self.num_decoder_layers):
            attn_out, _ = self.self_attn_layers[layer_idx](
                plan_mode_query, plan_mode_query, plan_mode_query
            )
            plan_mode_query = self.attn_norms[layer_idx](plan_mode_query + attn_out)
            ffn_out = self.ffn_layers[layer_idx](plan_mode_query)
            plan_mode_query = self.ffn_norms[layer_idx](plan_mode_query + ffn_out)

        # Refinement. plan_mode_query: (bs, cmds*M, D).
        plan_query = plan_mode_query.unsqueeze(1)  # (bs, 1, cmds*M, D)
        plan_cls = self.plan_cls_branch(plan_query).squeeze(-1)  # (bs, 1, cmds*M)
        plan_reg = self.plan_reg_branch(plan_query).reshape(
            bs, 1, self.num_driving_cmds * self.ego_fut_mode, self.ego_fut_ts, 2
        )

        # Status branch: take the mean over modes as the ego summary.
        ego_summary = plan_mode_query.mean(dim=1)  # (bs, D)
        planning_status = self.plan_status_branch(ego_summary)  # (bs, 10)

        planning_output = {
            "classification": [plan_cls],
            "prediction": [plan_reg],
            "status": [planning_status],
        }
        return planning_output

    # ------------------------------------------------------------------- loss
    @force_fp32(apply_to=("planning_output",))
    def loss(self, planning_output, data):
        # Mirrors MotionPlanningHead.loss_planning at minimum: per-stage cls +
        # best-mode reg + per-stage status, summed across decoder stages (we
        # only emit one stage, so this is a single application).
        plan_cls_list = planning_output["classification"]
        plan_reg_list = planning_output["prediction"]
        plan_status_list = planning_output["status"]
        gt_reg_target = data["gt_ego_fut_trajs"]
        gt_reg_mask = data["gt_ego_fut_masks"]
        ego_status_target = data["ego_status"]

        losses: Dict[str, torch.Tensor] = {}
        for stage, (plan_cls, plan_reg, plan_status) in enumerate(
            zip(plan_cls_list, plan_reg_list, plan_status_list)
        ):
            (
                cls_pred,
                cls_target,
                cls_weight,
                best_reg,
                gt_reg_target_s,
                gt_reg_mask_s,
            ) = self.planning_sampler.sample(
                plan_cls, plan_reg, gt_reg_target, gt_reg_mask, data
            )
            # Cls loss.
            cls_pred = cls_pred.flatten(end_dim=-2)
            cls_target = cls_target.flatten(end_dim=-1)
            cls_weight = cls_weight.flatten(end_dim=-1)
            num_pos = max(cls_weight.sum().item(), 1.0)
            num_pos = reduce_mean(plan_cls.new_tensor([num_pos])).clamp_min_(1.0).item()
            losses[f"planning_loss_cls_{stage}"] = self.plan_loss_cls(
                cls_pred, cls_target, weight=cls_weight, avg_factor=num_pos
            )
            # Reg loss (best-matched mode only).
            reg_weights = gt_reg_mask_s.unsqueeze(-1).expand_as(best_reg)
            best_reg = best_reg.flatten(end_dim=-2)
            target_reg = gt_reg_target_s.flatten(end_dim=-2)
            reg_weights = reg_weights.flatten(end_dim=-2)
            losses[f"planning_loss_reg_{stage}"] = self.plan_loss_reg(
                best_reg, target_reg, weight=reg_weights, avg_factor=num_pos
            )
            # Status loss: regress current ego state from per-stage summary.
            losses[f"planning_loss_status_{stage}"] = self.plan_loss_status(
                plan_status, ego_status_target.to(plan_status.dtype)
            )

        return losses

    # ----------------------------------------------------------- post_process
    def post_process(self, planning_output, data) -> List[Dict]:
        """Return a planning_result list compatible with NuScenes planning eval.

        Eval consumes only ``final_planning``; we still populate a couple of
        baseline-shaped fields (``planning_score``, ``planning``) so the
        downstream dataset hook doesn't choke on missing keys, plus empty
        ``ego_anchor_queue``/``ego_period`` placeholders since this head
        carries no temporal queue.
        """
        plan_cls = planning_output["classification"][-1]  # (bs, 1, cmds*M)
        plan_reg = planning_output["prediction"][-1]  # (bs, 1, cmds*M, T, 2)
        bs = plan_cls.shape[0]
        plan_cls = plan_cls.reshape(bs, self.num_driving_cmds, self.ego_fut_mode)
        plan_reg = plan_reg.reshape(
            bs, self.num_driving_cmds, self.ego_fut_mode, self.ego_fut_ts, 2
        ).cumsum(dim=-2)
        # Per-batch: pick the gt cmd's argmax mode as the final plan.
        gt_cmd = data["gt_ego_fut_cmd"].argmax(dim=-1)  # (bs,)
        bs_idx = torch.arange(bs, device=plan_cls.device)
        plan_cls_cmd = plan_cls[bs_idx, gt_cmd]  # (bs, M)
        plan_reg_cmd = plan_reg[bs_idx, gt_cmd]  # (bs, M, T, 2)
        best_mode = plan_cls_cmd.argmax(dim=-1)  # (bs,)
        final_planning = plan_reg_cmd[bs_idx, best_mode]  # (bs, T, 2)

        empty_queue = plan_cls.new_zeros(0, 11)
        empty_period = plan_cls.new_zeros(1)

        result = []
        for i in range(bs):
            result.append(
                {
                    "planning_score": plan_cls[i].sigmoid().cpu(),
                    "planning": plan_reg[i].cpu(),
                    "final_planning": final_planning[i].cpu(),
                    "ego_period": empty_period.cpu(),
                    "ego_anchor_queue": empty_queue.cpu(),
                }
            )
        return result


@HEADS.register_module()
class EgoPlannerSparseDriveHead(BaseModule):
    """Drop-in replacement for SparseDriveHead that holds only an
    ``EgoPlannerHead``. No det / map / motion heads. Provides the
    ``forward`` / ``loss`` / ``post_process`` interface the
    ``SparseDrive`` model expects.
    """

    def __init__(self, planner: Dict, init_cfg=None, **kwargs):
        super().__init__(init_cfg)
        from mmdet.models import build_head as _build_head
        self.planner = _build_head(planner)

    def init_weights(self):
        self.planner.init_weights()

    def forward(self, feature_maps, metas):
        planning_output = self.planner(feature_maps, metas)
        return planning_output

    def loss(self, model_outs, data):
        return self.planner.loss(model_outs, data)

    def post_process(self, model_outs, data):
        # Return list of inner-result dicts (unwrapped). `SparseDrive.simple_test`
        # already wraps each result in {"img_bbox": ...}; double-wrapping here
        # produces {"img_bbox": {"img_bbox": {...}}} and breaks planning_eval
        # which does `res['img_bbox']['final_planning']`.
        return self.planner.post_process(model_outs, data)
