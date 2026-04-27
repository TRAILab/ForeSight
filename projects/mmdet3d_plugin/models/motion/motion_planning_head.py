from typing import List, Optional, Tuple, Union
import warnings
import copy

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.utils import build_from_cfg
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmdet.core.bbox.builder import BBOX_SAMPLERS, BBOX_CODERS
from mmdet.models import build_loss

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
from projects.mmdet3d_plugin.core.box3d import *

from ..attention import gen_sineembed_for_position
from ..blocks import linear_relu_ln
from ..instance_bank import topk


def _detach_perception_output(output):
    """Stop-gradient over a head output dict (tensors + per-stage lists)."""
    if output is None:
        return None
    result = {}
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.detach()
        elif isinstance(v, list):
            result[k] = [x.detach() if isinstance(x, torch.Tensor) else x for x in v]
        else:
            result[k] = v
    return result


class _MotionPlanningAdapter(nn.Module):
    """Owns width conversion between perception and planning spaces."""

    def __init__(self, input_embed_dims, planning_embed_dims, enabled):
        super().__init__()
        self.input_embed_dims = input_embed_dims
        self.planning_embed_dims = planning_embed_dims
        self.enabled = enabled and input_embed_dims != planning_embed_dims

        if self.enabled:
            self.feature_to_planning = nn.Linear(
                input_embed_dims, planning_embed_dims, bias=False
            )
            self.anchor_to_planning = nn.Linear(
                input_embed_dims, planning_embed_dims, bias=False
            )
            self.planning_to_deformable = nn.Linear(
                planning_embed_dims, input_embed_dims, bias=False
            )
            self.deformable_to_planning = nn.Linear(
                input_embed_dims, planning_embed_dims, bias=False
            )
        else:
            self.feature_to_planning = nn.Identity()
            self.anchor_to_planning = nn.Identity()
            self.planning_to_deformable = nn.Identity()
            self.deformable_to_planning = nn.Identity()

    def feature(self, tensor):
        return self.feature_to_planning(tensor)

    def anchor(self, tensor):
        return self.anchor_to_planning(tensor)

    def deformable_query(self, tensor):
        return self.planning_to_deformable(tensor)

    def deformable_output(self, tensor):
        return self.deformable_to_planning(tensor)

    def cache(self, tensor):
        if not self.enabled:
            return tensor
        # Queue state is detached and reused on later timesteps, so keep the
        # cache path parameter-free.
        return tensor[..., : self.input_embed_dims]


@HEADS.register_module()
class MotionPlanningHead(BaseModule):
    def __init__(
        self,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
        motion_anchor=None,
        plan_anchor=None,
        embed_dims=256,
        decouple_attn=False,
        instance_queue=None,
        operation_order=None,
        temp_graph_model=None,
        graph_model=None,
        cross_graph_model=None,
        deformable_model=None,
        norm_layer=None,
        ffn=None,
        refine_layer=None,
        motion_sampler=None,
        motion_loss_cls=None,
        motion_loss_reg=None,
        planning_sampler=None,
        plan_loss_cls=None,
        plan_loss_reg=None,
        plan_loss_status=None,
        motion_decoder=None,
        planning_decoder=None,
        num_det=50,
        num_map=10,
        use_planning_input_proj=False,
        input_embed_dims=None,
        planning_cumulative_refinement=False,
        motion_cumulative_refinement=False,
        planning_deformable=False,
        planning_deformable_instfeat=False,
        planning_deformable_instfeat_laststage=False,
        motion_deformable=False,
        motion_deformable_multimode=False,
        motion_deformable_multimode_uniform=False,
        motion_deformable_modeproj=False,
        planning_deformable_instfeat_additive=False,
        deformable_waypoint=-1,
        planning_deformable_waypoints=None,
        num_dn_pred_groups=0,
        dn_pred_noise_scale=0.5,
        dn_pred_loss_weight=1.0,
        num_dn_plan_groups=0,
        dn_plan_noise_scale=0.5,
        dn_plan_loss_weight=1.0,
        use_alldet_kv=False,
        bidir_planning=False,
        rev_graph_model=None,
        with_da_head=False,
        with_conflict_head=False,
        da_loss_weight=0.2,
        conflict_loss_weight=0.2,
        conflict_threshold=2.0,
        conflict_pos_weight=10.0,
        roi_size=(30, 60),
        plan_mode_softtgt=False,
        plan_mode_softtgt_tau=1.0,
        plan_diversity_reg=False,
        plan_diversity_loss_weight=0.05,
        plan_diversity_sigma=5.0,
        mode_no_agg=False,
        plan_mode_time_queries=False,
        plan_time_attn=False,
        plan_time_attn_heads=8,
        plan_time_attn_dropout=0.1,
        plan_softcost_collision_enable=False,
        plan_softcost_collision_weight=0.2,
        plan_softcost_collision_sigma=2.0,
        plan_softcost_geometry='gaussian',
        plan_softcost_collision_tau=0.5,
        detach_perception=False,
        relevance_selection=False,
        relevance_corridor=2.0,
        relevance_horizon=6,
        skip_perception_kv=False,
    ):
        super(MotionPlanningHead, self).__init__()
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.decouple_attn = decouple_attn
        self.operation_order = operation_order
        self.input_embed_dims = (
            embed_dims if input_embed_dims is None else input_embed_dims
        )
        self.use_planning_input_proj = (
            use_planning_input_proj and self.input_embed_dims != embed_dims
        )
        self.planning_cumulative_refinement = planning_cumulative_refinement
        self.motion_cumulative_refinement = motion_cumulative_refinement
        self.planning_deformable = planning_deformable
        self.planning_deformable_instfeat = planning_deformable_instfeat and planning_deformable
        self.planning_deformable_instfeat_laststage = planning_deformable_instfeat_laststage
        self._n_deformable_stages = sum(1 for op in operation_order if op == "deformable")
        self.motion_deformable = motion_deformable
        self.motion_deformable_multimode = motion_deformable_multimode
        self.motion_deformable_multimode_uniform = (
            motion_deformable_multimode_uniform and motion_deformable_multimode
        )
        self.motion_deformable_modeproj = motion_deformable_modeproj and motion_deformable_multimode
        self.planning_deformable_instfeat_additive = (
            planning_deformable_instfeat_additive and self.planning_deformable_instfeat
        )
        self.plan_mode_softtgt = plan_mode_softtgt
        self.plan_mode_softtgt_tau = plan_mode_softtgt_tau
        self.plan_diversity_reg = plan_diversity_reg
        self.plan_diversity_loss_weight = plan_diversity_loss_weight
        self.plan_diversity_sigma = plan_diversity_sigma
        self.plan_softcost_collision_enable = plan_softcost_collision_enable
        self.plan_softcost_collision_weight = plan_softcost_collision_weight
        self.plan_softcost_collision_sigma = plan_softcost_collision_sigma
        self.plan_softcost_geometry = plan_softcost_geometry
        self.plan_softcost_collision_tau = plan_softcost_collision_tau
        self.detach_perception = detach_perception
        self.relevance_selection = relevance_selection
        self.relevance_corridor = float(relevance_corridor)
        self.relevance_horizon = int(relevance_horizon)
        self.skip_perception_kv = skip_perception_kv
        self.mode_no_agg = mode_no_agg
        self.plan_mode_time_queries = plan_mode_time_queries
        self.plan_time_attn = plan_time_attn
        if self.plan_time_attn:
            assert self.plan_mode_time_queries, (
                "plan_time_attn=True requires plan_mode_time_queries=True"
            )
        self.deformable_waypoint = deformable_waypoint
        self.planning_deformable_waypoints = planning_deformable_waypoints
        self.use_alldet_kv = use_alldet_kv
        self.bidir_planning = bidir_planning
        self.with_da_head = with_da_head
        self.with_conflict_head = with_conflict_head
        self.da_loss_weight = da_loss_weight
        self.conflict_loss_weight = conflict_loss_weight
        self.conflict_threshold = conflict_threshold
        self.conflict_pos_weight = conflict_pos_weight
        self.roi_size = roi_size

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)
        
        self.instance_queue = build(instance_queue, PLUGIN_LAYERS)
        self.motion_sampler = build(motion_sampler, BBOX_SAMPLERS)
        self.planning_sampler = build(planning_sampler, BBOX_SAMPLERS)
        self.motion_decoder = build(motion_decoder, BBOX_CODERS)
        self.planning_decoder = build(planning_decoder, BBOX_CODERS)
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "cross_gnn": [cross_graph_model, ATTENTION],
            "rev_gnn": [rev_graph_model, ATTENTION],
            "deformable": [deformable_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "refine": [refine_layer, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        self.embed_dims = embed_dims
        self.adapter = _MotionPlanningAdapter(
            input_embed_dims=self.input_embed_dims,
            planning_embed_dims=self.embed_dims,
            enabled=self.use_planning_input_proj,
        )

        if self.motion_deformable_modeproj:
            self.motion_mode_projs = nn.ModuleList(
                [nn.Linear(embed_dims, embed_dims) for _ in range(fut_mode)]
            )

        if self.plan_time_attn:
            # Per-mode time-axis self-attention applied after each planning
            # deformable stage. Restores intra-mode temporal coupling that
            # plan_mode_time_queries removes by giving each (mode, ts) query
            # its own MLP output.
            self.plan_time_pos_embed = nn.Parameter(
                torch.zeros(1, ego_fut_ts, embed_dims)
            )
            nn.init.trunc_normal_(self.plan_time_pos_embed, std=0.02)
            self.plan_time_attn_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dims,
                    num_heads=plan_time_attn_heads,
                    dropout=plan_time_attn_dropout,
                    batch_first=True,
                )
                for _ in range(self._n_deformable_stages)
            ])
            self.plan_time_attn_norms = nn.ModuleList([
                nn.LayerNorm(embed_dims)
                for _ in range(self._n_deformable_stages)
            ])

        if self.decouple_attn:
            self.fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        self.motion_loss_cls = build_loss(motion_loss_cls)
        self.motion_loss_reg = build_loss(motion_loss_reg)
        self.plan_loss_cls = build_loss(plan_loss_cls)
        self.plan_loss_reg = build_loss(plan_loss_reg)
        self.plan_loss_status = build_loss(plan_loss_status)

        # motion init
        motion_anchor = np.load(motion_anchor)
        self.motion_anchor = nn.Parameter(
            torch.tensor(motion_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.motion_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        # plan anchor init
        # plan_anchor may be a single path or a list of paths whose per-cmd
        # mode tensors are concatenated along the mode axis. Each file holds
        # an array shaped (3, K_i, ego_fut_ts, 2); the concatenated result is
        # (3, sum(K_i), ego_fut_ts, 2). All sum(K_i) must equal ego_fut_mode.
        if isinstance(plan_anchor, (list, tuple)):
            anchors = [np.load(p) for p in plan_anchor]
            plan_anchor = np.concatenate(anchors, axis=1)
        else:
            plan_anchor = np.load(plan_anchor)
        assert plan_anchor.shape[1] == ego_fut_mode, (
            f"plan_anchor mode dim {plan_anchor.shape[1]} != ego_fut_mode "
            f"{ego_fut_mode}"
        )
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        self.num_det = num_det
        self.num_map = num_map

        self.num_dn_pred_groups = num_dn_pred_groups
        self.dn_pred_noise_scale = dn_pred_noise_scale
        self.dn_pred_loss_weight = dn_pred_loss_weight
        self.num_dn_plan_groups = num_dn_plan_groups
        self.dn_plan_noise_scale = dn_plan_noise_scale
        self.dn_plan_loss_weight = dn_plan_loss_weight

        # Index of the refine layer (last refine op) — used for DN forward.
        refine_indices = [i for i, op in enumerate(operation_order) if op == 'refine']
        self._dn_refine_idx = refine_indices[-1] if refine_indices else None

    def _build_motion_endpoint_anchors(self, motion_anchor_upd, det_anchors, motion_cls):
        """Build 3D box anchors at the best-mode predicted endpoint for each agent.

        motion_anchor_upd: (bs, num_det, fut_mode, fut_ts, 2) cumulative XY
            displacements in lidar-frame orientation (relative to agent position).
        det_anchors: (bs, num_det, 11) current detection boxes.
        motion_cls: (bs, num_det, fut_mode) classification logits.

        Returns (bs, num_det, 11) anchor boxes placed at the predicted endpoints.
        """
        best_mode = motion_cls.argmax(dim=-1)  # (bs, num_det)
        idx = best_mode[..., None, None, None].expand(
            -1, -1, 1, motion_anchor_upd.shape[-2], 2
        )
        best_traj = motion_anchor_upd.gather(2, idx).squeeze(2)  # (bs, num_det, fut_ts, 2)

        w = self.deformable_waypoint
        endpoint = best_traj[..., w, :]  # (bs, num_det, 2) — displacement from agent
        if best_traj.shape[-2] > 1:
            if w == 0:
                heading_vec = best_traj[..., 1, :] - best_traj[..., 0, :]
            else:
                heading_vec = best_traj[..., w, :] - best_traj[..., w - 1, :]
        else:
            heading_vec = endpoint

        anchor = det_anchors.clone()
        ego_yaw = torch.atan2(det_anchors[..., SIN_YAW], det_anchors[..., COS_YAW])
        heading_yaw = torch.atan2(heading_vec[..., 1], heading_vec[..., 0])
        static_mask = torch.linalg.norm(heading_vec, dim=-1) < 1e-3
        heading_yaw = torch.where(static_mask, ego_yaw, heading_yaw)

        # endpoint is a displacement — add to current agent position for absolute XY
        anchor[..., X] = det_anchors[..., X] + endpoint[..., 0]
        anchor[..., Y] = det_anchors[..., Y] + endpoint[..., 1]
        anchor[..., SIN_YAW] = torch.sin(heading_yaw)
        anchor[..., COS_YAW] = torch.cos(heading_yaw)
        return anchor

    def _build_motion_endpoint_anchors_all_modes(self, motion_anchor_upd, det_anchors):
        """Build 3D box anchors at the predicted endpoint for every mode of each agent.

        motion_anchor_upd: (bs, num_det, fut_mode, fut_ts, 2) cumulative XY
            displacements in lidar-frame orientation (relative to agent position).
        det_anchors: (bs, num_det, 11) current detection boxes.

        Returns (bs, num_det, fut_mode, 11) anchor boxes.
        """
        bs, num_det, fut_mode, fut_ts, _ = motion_anchor_upd.shape
        w = self.deformable_waypoint
        endpoint = motion_anchor_upd[..., w, :]  # (bs, num_det, fut_mode, 2)
        if fut_ts > 1:
            if w == 0:
                heading_vec = motion_anchor_upd[..., 1, :] - motion_anchor_upd[..., 0, :]
            else:
                heading_vec = motion_anchor_upd[..., w, :] - motion_anchor_upd[..., w - 1, :]
        else:
            heading_vec = endpoint

        anchor = det_anchors[:, :, None, :].expand(-1, -1, fut_mode, -1).clone()
        ego_yaw = torch.atan2(
            det_anchors[..., SIN_YAW], det_anchors[..., COS_YAW]
        ).unsqueeze(-1)  # (bs, num_det, 1)
        heading_yaw = torch.atan2(heading_vec[..., 1], heading_vec[..., 0])
        static_mask = torch.linalg.norm(heading_vec, dim=-1) < 1e-3
        heading_yaw = torch.where(static_mask, ego_yaw.expand_as(heading_yaw), heading_yaw)

        anchor[..., X] = det_anchors[..., X].unsqueeze(-1) + endpoint[..., 0]
        anchor[..., Y] = det_anchors[..., Y].unsqueeze(-1) + endpoint[..., 1]
        anchor[..., SIN_YAW] = torch.sin(heading_yaw)
        anchor[..., COS_YAW] = torch.cos(heading_yaw)
        return anchor

    def _build_planning_anchor_boxes(self, plan_anchor, ego_anchor):
        num_mode = plan_anchor.shape[1]
        w = self.deformable_waypoint
        plan_endpoint = plan_anchor[..., w, :]
        if plan_anchor.shape[-2] > 1:
            if w == 0:
                heading_vec = plan_anchor[..., 1, :] - plan_anchor[..., 0, :]
            else:
                heading_vec = plan_anchor[..., w, :] - plan_anchor[..., w - 1, :]
        else:
            heading_vec = plan_endpoint

        anchor = ego_anchor.expand(-1, num_mode, -1).clone()
        ego_yaw = torch.atan2(anchor[..., SIN_YAW], anchor[..., COS_YAW])
        heading_yaw = torch.atan2(heading_vec[..., 1], heading_vec[..., 0])
        static_mask = torch.linalg.norm(heading_vec, dim=-1) < 1e-3
        heading_yaw = torch.where(static_mask, ego_yaw, heading_yaw)

        anchor[..., X] = plan_endpoint[..., 0]
        anchor[..., Y] = plan_endpoint[..., 1]
        anchor[..., SIN_YAW] = torch.sin(heading_yaw)
        anchor[..., COS_YAW] = torch.cos(heading_yaw)
        return anchor

    def _build_planning_anchor_boxes_multi(self, plan_anchor, ego_anchor, waypoints):
        """Build 3D anchor boxes at multiple waypoints for each planning mode.

        plan_anchor: (bs, num_mode, ego_fut_ts, 2) cumulative XY in lidar frame.
        ego_anchor: (bs, 1, 11) ego box.
        waypoints: list[int] of timestep indices.

        Returns (bs, num_mode, K, 11) where K = len(waypoints).
        """
        num_mode = plan_anchor.shape[1]
        ego_fut_ts = plan_anchor.shape[-2]
        anchor_base = ego_anchor.expand(-1, num_mode, -1).clone()  # (bs, num_mode, 11)

        boxes = []
        for w in waypoints:
            endpoint = plan_anchor[..., w, :]  # (bs, num_mode, 2)
            if ego_fut_ts > 1:
                if w == 0:
                    heading_vec = plan_anchor[..., 1, :] - plan_anchor[..., 0, :]
                else:
                    heading_vec = plan_anchor[..., w, :] - plan_anchor[..., w - 1, :]
            else:
                heading_vec = endpoint

            anchor = anchor_base.clone()
            ego_yaw = torch.atan2(anchor[..., SIN_YAW], anchor[..., COS_YAW])
            heading_yaw = torch.atan2(heading_vec[..., 1], heading_vec[..., 0])
            static_mask = torch.linalg.norm(heading_vec, dim=-1) < 1e-3
            heading_yaw = torch.where(static_mask, ego_yaw, heading_yaw)

            anchor[..., X] = endpoint[..., 0]
            anchor[..., Y] = endpoint[..., 1]
            anchor[..., SIN_YAW] = torch.sin(heading_yaw)
            anchor[..., COS_YAW] = torch.cos(heading_yaw)
            boxes.append(anchor)

        return torch.stack(boxes, dim=2)  # (bs, num_mode, K, 11)

    def _build_dn_pred_queries(self, metas):
        """Build DN agent tokens for full-decoder prediction denoising.

        Returns (dn_agent_feat, dn_agent_anchor_embed, dn_motion_mode_query,
                 dn_reg_target, dn_valid) or None.
        - dn_agent_feat: (bs, num_dn_agents, embed_dims)  zero-init
        - dn_agent_anchor_embed: (bs, num_dn_agents, embed_dims)  zero (no positional bias)
        - dn_motion_mode_query: (bs, num_dn_agents, 1, embed_dims)  from noisy GT endpoint
        - dn_reg_target: (bs, num_dn_agents, fut_ts, 2)
        - dn_valid: (bs, num_dn_agents) bool
        """
        gt_trajs = metas.get('gt_agent_fut_trajs')
        if gt_trajs is None:
            return None

        bs = len(gt_trajs)
        max_gt = max((len(x) for x in gt_trajs), default=0)
        if max_gt == 0:
            return None

        device, dtype = None, None
        for t in gt_trajs:
            if len(t) > 0:
                device, dtype = t.device, t.dtype
                break
        if device is None:
            return None

        # Pad GT trajectories: (bs, max_gt, fut_ts, 2)
        gt_traj_padded = torch.stack([
            torch.cat([x, x.new_zeros(max_gt - len(x), self.fut_ts, 2)], dim=0)
            if len(x) < max_gt else x[:max_gt]
            for x in gt_trajs
        ])
        valid = torch.stack([
            torch.cat([x.new_ones(min(len(x), max_gt)),
                       x.new_zeros(max(0, max_gt - len(x)))])
            for x in gt_trajs
        ]).bool()  # (bs, max_gt)

        # Add noise: (num_dn, bs, max_gt, fut_ts, 2)
        noise = (
            torch.rand(self.num_dn_pred_groups, bs, max_gt, self.fut_ts, 2,
                       device=device, dtype=dtype) * 2 - 1
        ) * self.dn_pred_noise_scale
        dn_traj = (gt_traj_padded.unsqueeze(0) + noise
                   ).permute(1, 0, 2, 3, 4).flatten(1, 2)  # (bs, num_dn*max_gt, fut_ts, 2)

        # Mode query from noisy cumulative endpoint: (bs, num_dn*max_gt, 1, embed_dims)
        dn_endpoint = dn_traj.cumsum(dim=-2)[..., -1, :]
        dn_motion_mode_query = self.motion_anchor_encoder(
            gen_sineembed_for_position(dn_endpoint, hidden_dim=self.embed_dims)
        ).unsqueeze(2)

        num_dn_agents = self.num_dn_pred_groups * max_gt
        dn_agent_feat = torch.zeros(bs, num_dn_agents, self.embed_dims, device=device, dtype=dtype)
        dn_agent_anchor_embed = torch.zeros(bs, num_dn_agents, self.embed_dims, device=device, dtype=dtype)

        dn_valid = valid.unsqueeze(1).expand(-1, self.num_dn_pred_groups, -1).flatten(1)
        dn_reg_target = (gt_traj_padded.unsqueeze(1)
                         .expand(-1, self.num_dn_pred_groups, -1, -1, -1)
                         .flatten(1, 2))

        return dn_agent_feat, dn_agent_anchor_embed, dn_motion_mode_query, dn_reg_target, dn_valid

    def _build_dn_plan_queries(self, metas, ego_anchor_embed):
        """Build DN ego tokens for full-decoder planning denoising.

        Returns (dn_ego_feat, dn_ego_anchor_embed, dn_plan_mode_query,
                 dn_reg_target, dn_valid) or None.
        - dn_ego_feat: (bs, num_dn_ego, embed_dims)  zero-init
        - dn_ego_anchor_embed: (bs, num_dn_ego, embed_dims)  tiled from ego_anchor_embed
        - dn_plan_mode_query: (bs, num_dn_ego, 1, embed_dims)  from noisy GT ego endpoint
        - dn_reg_target: (bs, num_dn_ego, ego_fut_ts, 2)
        - dn_valid: (bs, num_dn_ego) bool
        """
        gt_ego_traj = metas.get('gt_ego_fut_trajs')  # (bs, ego_fut_ts, 2)
        if gt_ego_traj is None:
            return None

        bs = gt_ego_traj.shape[0]
        device, dtype = gt_ego_traj.device, gt_ego_traj.dtype

        noise = (
            torch.rand(bs, self.num_dn_plan_groups, self.ego_fut_ts, 2,
                       device=device, dtype=dtype) * 2 - 1
        ) * self.dn_plan_noise_scale
        dn_traj = gt_ego_traj.unsqueeze(1).expand(-1, self.num_dn_plan_groups, -1, -1) + noise

        # Mode query from noisy cumulative endpoint: (bs, num_dn_ego, 1, embed_dims)
        dn_endpoint = dn_traj.cumsum(dim=-2)[..., -1, :]
        dn_plan_mode_query = self.plan_anchor_encoder(
            gen_sineembed_for_position(dn_endpoint, hidden_dim=self.embed_dims)
        ).unsqueeze(2)

        dn_ego_feat = torch.zeros(bs, self.num_dn_plan_groups, self.embed_dims, device=device, dtype=dtype)
        # Tile ego anchor embed so DN ego tokens share the ego positional embedding
        dn_ego_anchor_embed = ego_anchor_embed.expand(-1, self.num_dn_plan_groups, -1).reshape(
            bs, self.num_dn_plan_groups, self.embed_dims
        )

        gt_ego_mask = metas.get('gt_ego_fut_masks')
        if gt_ego_mask is not None:
            dn_valid = gt_ego_mask.any(dim=-1, keepdim=True).expand(-1, self.num_dn_plan_groups)
        else:
            dn_valid = torch.ones(bs, self.num_dn_plan_groups, dtype=torch.bool, device=device)

        dn_reg_target = gt_ego_traj.unsqueeze(1).expand(-1, self.num_dn_plan_groups, -1, -1)

        return dn_ego_feat, dn_ego_anchor_embed, dn_plan_mode_query, dn_reg_target, dn_valid

    def _compute_det_corridor_score(self, metas, det_anchors, det_confidence):
        """Per-anchor planning-relevance score for K/V selection.

        Score = 1e3 * relevant + det_confidence (confidence is the tiebreaker).
        An anchor is relevant if its nearest GT agent (by BEV centre at t=0)
        comes within ``relevance_corridor`` metres of the GT ego BEV path under
        cross-time min: ``min_{t1,t2} dist(ego(t1), agent(t2))`` over the first
        ``relevance_horizon`` planning steps.

        Falls back to ``det_confidence`` when GT keys are missing.
        """
        gt_ego = metas.get('gt_ego_fut_trajs')
        gt_agent = metas.get('gt_agent_fut_trajs')
        gt_boxes = metas.get('gt_bboxes_3d')
        if gt_ego is None or gt_agent is None or gt_boxes is None:
            return det_confidence

        gt_ego_mask = metas.get('gt_ego_fut_masks')
        gt_agent_mask = metas.get('gt_agent_fut_masks')

        bs, num_anchor = det_anchors.shape[:2]
        device = det_anchors.device
        H = self.relevance_horizon
        thr2 = self.relevance_corridor ** 2
        score = det_confidence.clone()

        for b in range(bs):
            ego_b = gt_ego[b]
            if not torch.is_tensor(ego_b):
                continue
            ego_b = ego_b.to(device).float()
            if ego_b.numel() == 0:
                continue
            T_ego = min(ego_b.shape[0], H)
            ego_abs = ego_b[:T_ego].cumsum(dim=0)
            if gt_ego_mask is not None:
                em = gt_ego_mask[b].to(device)[:T_ego].bool()
            else:
                em = torch.ones(T_ego, dtype=torch.bool, device=device)

            agents_b = gt_agent[b].to(device).float()
            boxes_b = gt_boxes[b].to(device).float()
            if agents_b.shape[0] == 0:
                continue
            Nb, agent_fut_ts = agents_b.shape[:2]
            T_agt = min(agent_fut_ts, H)
            if T_agt == 0 or T_ego == 0:
                continue
            agents_abs = boxes_b[:, :2].unsqueeze(1) + agents_b[:, :T_agt].cumsum(dim=1)

            diff = ego_abs.view(1, T_ego, 1, 2) - agents_abs.view(Nb, 1, T_agt, 2)
            d2 = (diff ** 2).sum(dim=-1)  # (Nb, T_ego, T_agt)

            valid = em.view(1, T_ego, 1).expand(Nb, T_ego, T_agt)
            if gt_agent_mask is not None:
                am = gt_agent_mask[b].to(device)[:, :T_agt].bool()
                valid = valid & am.view(Nb, 1, T_agt).expand(-1, T_ego, -1)
            d2 = torch.where(valid, d2, d2.new_full((), float('inf')))
            min_d2_per_agent = d2.flatten(1).min(dim=-1).values  # (Nb,)
            relevant_per_agent = (min_d2_per_agent < thr2).to(score.dtype)

            det_xy = det_anchors[b, :, :2]
            gt_xy = boxes_b[:, :2]
            d_match = ((det_xy.unsqueeze(1) - gt_xy.unsqueeze(0)) ** 2).sum(dim=-1)
            nearest_gt = d_match.argmin(dim=-1)
            anchor_relevance = relevant_per_agent[nearest_gt]

            score[b] = anchor_relevance * 1e3 + det_confidence[b]

        return score

    def _compute_map_corridor_score(self, metas, map_anchors, map_confidence):
        """Per-map-anchor planning-relevance score for K/V selection.

        Score = -min_{t,p} dist(ego_gt(t), map_pt[m, p])^2 (closer is higher).
        Falls back to ``map_confidence`` when GT ego trajectory is missing.
        """
        gt_ego = metas.get('gt_ego_fut_trajs')
        if gt_ego is None:
            return map_confidence

        bs, num_map_anchor = map_anchors.shape[:2]
        device = map_anchors.device
        H = self.relevance_horizon
        map_pts = map_anchors.view(bs, num_map_anchor, -1, 2)
        score = map_confidence.clone()

        for b in range(bs):
            ego_b = gt_ego[b]
            if not torch.is_tensor(ego_b):
                continue
            ego_b = ego_b.to(device).float()
            if ego_b.numel() == 0:
                continue
            T_ego = min(ego_b.shape[0], H)
            ego_abs = ego_b[:T_ego].cumsum(dim=0)

            diff = map_pts[b].unsqueeze(2) - ego_abs.view(1, 1, T_ego, 2)
            d2 = (diff ** 2).sum(dim=-1)  # (M, P, T)
            min_d2_per_map = d2.flatten(1).min(dim=-1).values  # (M,)

            score[b] = -min_d2_per_map

        return score

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def _project_instance_feature(self, feature):
        return self.adapter.feature(feature)

    def _project_anchor_embed(self, anchor_embed):
        return self.adapter.anchor(anchor_embed)

    def _project_cache_feature(self, feature):
        return self.adapter.cache(feature)

    def _project_deformable_feature(self, feature):
        return self.adapter.deformable_query(feature)

    def _project_deformable_output(self, feature):
        return self.adapter.deformable_output(feature)

    def get_motion_anchor(
        self, 
        classification, 
        prediction,
    ):
        cls_ids = classification.argmax(dim=-1)
        motion_anchor = self.motion_anchor[cls_ids]
        prediction = prediction.detach()
        return self._agent2lidar(motion_anchor, prediction)

    def _agent2lidar(self, trajs, boxes):
        yaw = torch.atan2(boxes[..., SIN_YAW], boxes[..., COS_YAW])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot_mat_T = torch.stack(
            [
                torch.stack([cos_yaw, sin_yaw]),
                torch.stack([-sin_yaw, cos_yaw]),
            ]
        )

        trajs_lidar = torch.einsum('abcij,jkab->abcik', trajs, rot_mat_T)
        return trajs_lidar

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def forward(
        self, 
        det_output,
        map_output,
        feature_maps,
        metas,
        anchor_encoder,
        mask,
        anchor_handler,
    ):
        # =========== det/map feature/anchor ===========
        if self.detach_perception:
            det_output = _detach_perception_output(det_output)
            map_output = _detach_perception_output(map_output)
        raw_instance_feature = det_output["instance_feature"]
        raw_anchor_embed = det_output["anchor_embed"]
        instance_feature = self._project_instance_feature(raw_instance_feature)
        anchor_embed = self._project_anchor_embed(raw_anchor_embed)
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        bs_d = instance_feature.shape[0]
        if self.num_det <= 0:
            instance_feature_selected = instance_feature.new_zeros(
                bs_d, 0, instance_feature.shape[-1]
            )
            anchor_embed_selected = anchor_embed.new_zeros(
                bs_d, 0, anchor_embed.shape[-1]
            )
        else:
            if self.relevance_selection:
                det_select_score = self._compute_det_corridor_score(
                    metas, det_anchors, det_confidence
                )
            else:
                det_select_score = det_confidence
            _, (instance_feature_selected, anchor_embed_selected) = topk(
                det_select_score, self.num_det, instance_feature, anchor_embed
            )

        if map_output is not None:
            map_instance_feature = self._project_instance_feature(
                map_output["instance_feature"]
            )
            map_anchor_embed = self._project_anchor_embed(
                map_output["anchor_embed"]
            )
            map_classification = map_output["classification"][-1].sigmoid()
            map_anchors = map_output["prediction"][-1]
            map_confidence = map_classification.max(dim=-1).values
            if self.num_map <= 0:
                map_instance_feature_selected = map_instance_feature.new_zeros(
                    bs_d, 0, map_instance_feature.shape[-1]
                )
                map_anchor_embed_selected = map_anchor_embed.new_zeros(
                    bs_d, 0, map_anchor_embed.shape[-1]
                )
            else:
                if self.relevance_selection:
                    map_select_score = self._compute_map_corridor_score(
                        metas, map_anchors, map_confidence
                    )
                else:
                    map_select_score = map_confidence
                _, (map_instance_feature_selected, map_anchor_embed_selected) = topk(
                    map_select_score, self.num_map, map_instance_feature, map_anchor_embed
                )

        # =========== get ego/temporal feature/anchor ===========
        bs, num_anchor, _ = instance_feature.shape
        (
            ego_feature,
            ego_anchor,
            temp_instance_feature,
            temp_anchor,
            temp_mask,
        ) = self.instance_queue.get(
            det_output,
            feature_maps,
            metas,
            bs,
            mask,
            anchor_handler,
        )
        ego_feature_raw = ego_feature
        ego_feature = self._project_instance_feature(ego_feature)
        ego_anchor_embed = self._project_anchor_embed(anchor_encoder(ego_anchor))
        temp_instance_feature = self._project_instance_feature(temp_instance_feature)
        temp_anchor_embed = self._project_anchor_embed(anchor_encoder(temp_anchor))
        temp_instance_feature = temp_instance_feature.flatten(0, 1)
        temp_anchor_embed = temp_anchor_embed.flatten(0, 1)
        temp_mask = temp_mask.flatten(0, 1)
        dim = self.embed_dims

        # =========== mode anchor init ===========
        motion_anchor = self.get_motion_anchor(det_classification, det_anchors)
        plan_anchor = torch.tile(
            self.plan_anchor[None], (bs, 1, 1, 1, 1)
        ).reshape(bs, -1, self.ego_fut_ts, 2)

        # =========== mode query init ===========
        motion_mode_query = self.motion_anchor_encoder(
            gen_sineembed_for_position(
                motion_anchor[..., -1, :], hidden_dim=self.embed_dims
            )
        )
        if self.plan_mode_time_queries:
            # Per-(mode, ts) query, flattened along (mode, ts): (bs, 3*M*T, D).
            plan_pos = gen_sineembed_for_position(
                plan_anchor, hidden_dim=self.embed_dims
            )
            plan_mode_query = self.plan_anchor_encoder(plan_pos).flatten(1, 2)
        else:
            plan_pos = gen_sineembed_for_position(
                plan_anchor[..., -1, :], hidden_dim=self.embed_dims
            )
            plan_mode_query = self.plan_anchor_encoder(plan_pos)

        # =========== cat instance and ego ===========
        instance_feature_selected = torch.cat([instance_feature_selected, ego_feature], dim=1)
        anchor_embed_selected = torch.cat([anchor_embed_selected, ego_anchor_embed], dim=1)

        instance_feature = torch.cat([instance_feature, ego_feature], dim=1)
        anchor_embed = torch.cat([anchor_embed, ego_anchor_embed], dim=1)

        # =========== DN token init (training only) ===========
        # Layout: [0:num_anchor] agents | [num_anchor:num_anchor+1] ego
        #         | [num_anchor+1:num_anchor+1+num_dn_agents] DN agents
        #         | [num_anchor+1+num_dn_agents:] DN egos
        num_dn_agents = 0
        num_dn_ego = 0
        dn_motion_mode_query = None
        dn_plan_mode_query_var = None
        dn_reg_target_pred = None
        dn_valid_pred = None
        dn_reg_target_plan = None
        dn_valid_plan = None

        if self.training:
            if self.num_dn_pred_groups > 0:
                dn_pred_data = self._build_dn_pred_queries(metas)
                if dn_pred_data is not None:
                    (dn_agent_feat, dn_agent_anchor_embed,
                     dn_motion_mode_query, dn_reg_target_pred, dn_valid_pred) = dn_pred_data
                    num_dn_agents = dn_agent_feat.shape[1]
                    instance_feature = torch.cat([instance_feature, dn_agent_feat], dim=1)
                    anchor_embed = torch.cat([anchor_embed, dn_agent_anchor_embed], dim=1)
            if self.num_dn_plan_groups > 0:
                dn_plan_data = self._build_dn_plan_queries(metas, ego_anchor_embed)
                if dn_plan_data is not None:
                    (dn_ego_feat, dn_ego_anchor_embed,
                     dn_plan_mode_query_var, dn_reg_target_plan, dn_valid_plan) = dn_plan_data
                    num_dn_ego = dn_ego_feat.shape[1]
                    instance_feature = torch.cat([instance_feature, dn_ego_feat], dim=1)
                    anchor_embed = torch.cat([anchor_embed, dn_ego_anchor_embed], dim=1)

        # =================== forward the layers ====================
        motion_classification = []
        motion_prediction = []
        planning_classification = []
        planning_prediction = []
        planning_status = []
        planning_da_logits = []
        planning_conflict_logits = []
        # Initialize motion endpoint anchors from k-means prior for first decoder
        # deformable stage — a future position rather than the current det box.
        if self.motion_deformable_multimode:
            motion_endpoint_anchor_all = self._build_motion_endpoint_anchors_all_modes(
                motion_anchor, det_anchors
            )
            motion_endpoint_anchor = None
        elif self.motion_deformable:
            init_cls = torch.zeros(
                bs, num_anchor, motion_anchor.shape[2], device=det_anchors.device
            )
            motion_endpoint_anchor = self._build_motion_endpoint_anchors(
                motion_anchor, det_anchors, init_cls
            )
            motion_endpoint_anchor_all = None
        else:
            motion_endpoint_anchor = None
            motion_endpoint_anchor_all = None
        _deformable_stage_idx = 0
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                # DN tokens have no temporal history — run only normal tokens
                # through temp_gnn to avoid key/batch-size mismatch in MHA.
                num_normal = num_anchor + 1
                normal_feat = self.graph_model(
                    i,
                    instance_feature[:, :num_normal].flatten(0, 1).unsqueeze(1),
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed[:, :num_normal].flatten(0, 1).unsqueeze(1),
                    key_pos=temp_anchor_embed,
                    key_padding_mask=temp_mask,
                ).reshape(bs, num_normal, dim)
                if num_dn_agents + num_dn_ego > 0:
                    instance_feature = torch.cat(
                        [normal_feat, instance_feature[:, num_normal:]], dim=1
                    )
                else:
                    instance_feature = normal_feat
            elif op == "gnn":
                if self.skip_perception_kv:
                    continue
                if self.use_alldet_kv:
                    # Skip top-k bottleneck: planning queries cross-attend to
                    # all detection tokens (real agents + ego, no DN).
                    kv_inst = instance_feature[:, :num_anchor + 1]
                    kv_anchor = anchor_embed[:, :num_anchor + 1]
                else:
                    kv_inst = instance_feature_selected
                    kv_anchor = anchor_embed_selected
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    kv_inst,
                    kv_inst,
                    query_pos=anchor_embed,
                    key_pos=kv_anchor,
                )
            elif op == "rev_gnn":
                # Reverse cross-attention: detection (agent) features attend
                # to current planning query state, allowing planning to
                # reshape detection features within the same forward pass.
                agent_feat = instance_feature[:, :num_anchor]
                plan_state_kv = (
                    plan_mode_query
                    + instance_feature[:, num_anchor:num_anchor + 1]
                    + anchor_embed[:, num_anchor:num_anchor + 1]
                )
                agent_feat = self.layers[i](
                    agent_feat,
                    key=plan_state_kv,
                    query_pos=anchor_embed[:, :num_anchor],
                )
                instance_feature = torch.cat(
                    [agent_feat, instance_feature[:, num_anchor:]], dim=1
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "cross_gnn":
                if map_output is None or self.skip_perception_kv:
                    continue
                instance_feature = self.layers[i](
                    instance_feature,
                    key=map_instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=map_anchor_embed_selected,
                )
            elif op == "deformable":
                # Apply deformable cross-attention to sensor features for
                # agent instances only (ego token has no well-defined 3D box).
                _deformable_stage_idx += 1
                if self.motion_deformable_multimode:
                    # Attend at every mode's endpoint, aggregate by mode confidence.
                    # motion_endpoint_anchor_all: (bs, num_det, fut_mode, 11)
                    all_anchors_flat = motion_endpoint_anchor_all.reshape(
                        bs, num_anchor * self.fut_mode, 11
                    )
                    all_anchors_embed = anchor_encoder(all_anchors_flat)
                    agent_feat_exp = (
                        instance_feature[:, :num_anchor]
                        .unsqueeze(2)
                        .expand(-1, -1, self.fut_mode, -1)
                        .reshape(bs, num_anchor * self.fut_mode, self.embed_dims)
                    )
                    attended = self.layers[i](
                        self._project_deformable_feature(agent_feat_exp),
                        all_anchors_flat,
                        all_anchors_embed,
                        feature_maps, metas,
                    )  # (bs, num_det * fut_mode, embed_dims), includes residual
                    attended = self._project_deformable_output(attended).reshape(
                        bs, num_anchor, self.fut_mode, self.embed_dims
                    )
                    if self.motion_deformable_modeproj:
                        attended = torch.stack(
                            [self.motion_mode_projs[m](attended[:, :, m, :])
                             for m in range(self.fut_mode)],
                            dim=2,
                        )
                    if self.motion_deformable_multimode_uniform or self.mode_no_agg:
                        mode_weights = torch.full(
                            (bs, num_anchor, self.fut_mode), 1.0 / self.fut_mode,
                            device=det_anchors.device, dtype=det_anchors.dtype,
                        )
                    elif motion_classification:
                        mode_weights = motion_classification[-1].detach().softmax(dim=-1)
                    else:
                        mode_weights = torch.full(
                            (bs, num_anchor, self.fut_mode), 1.0 / self.fut_mode,
                            device=det_anchors.device, dtype=det_anchors.dtype,
                        )
                    agent_feature = (attended * mode_weights.unsqueeze(-1)).sum(dim=2)
                    if self.mode_no_agg:
                        # Bypass softmax aggregation for the mode_query path.
                        # Additive blend keeps motion_anchor_encoder gradient
                        # alive while injecting the per-mode attended features
                        # into refine.
                        motion_mode_query = motion_mode_query + attended
                else:
                    motion_anchor_ref = (
                        motion_endpoint_anchor if self.motion_deformable else det_anchors
                    )
                    agent_feature = self._project_deformable_output(self.layers[i](
                        self._project_deformable_feature(
                            instance_feature[:, :num_anchor]
                        ),
                        motion_anchor_ref,
                        anchor_encoder(motion_anchor_ref),
                        feature_maps,
                        metas,
                    ))
                if self.planning_deformable:
                    if self.plan_mode_time_queries:
                        # Per-(mode, ts) anchor box, flattened along (mode, ts).
                        T = self.ego_fut_ts
                        plan_anchor_box_mt = self._build_planning_anchor_boxes_multi(
                            plan_anchor.detach(), ego_anchor, list(range(T)),
                        )  # (bs, M_total, T, 11)
                        plan_anchor_box = plan_anchor_box_mt.flatten(1, 2)  # (bs, M_total*T, 11)
                    else:
                        plan_anchor_box = self._build_planning_anchor_boxes(
                            plan_anchor.detach(),
                            ego_anchor,
                        )
                    plan_anchor_embed = anchor_encoder(plan_anchor_box)
                    _use_instfeat = (
                        self.planning_deformable_instfeat and (
                            not self.planning_deformable_instfeat_laststage
                            or _deformable_stage_idx == self._n_deformable_stages
                        )
                    )
                    if _use_instfeat:
                        # Direction B: use ego instance_feature as DAF query,
                        # mirroring motion_deformable_multimode. Expand ego
                        # feature to all planning modes, attend at each mode's
                        # endpoint box, then aggregate back by mode confidence.
                        num_plan_modes = plan_anchor_box.shape[1]
                        # Use only the ego token (not DN tokens) as the DAF query.
                        ego_feat_exp = instance_feature[:, num_anchor:num_anchor+1].expand(
                            -1, num_plan_modes, -1
                        )  # (bs, num_plan_modes, embed_dims)
                        if self.planning_deformable_instfeat_additive:
                            # Additive variant: query = plan_mode_query + ego_feat_exp.
                            # Preserves per-mode specificity (plan_mode_query is
                            # mode-distinct) while still injecting ego instance
                            # feature. Update plan_mode_query like standard branch;
                            # leave ego token in instance_feature unchanged.
                            daf_query = plan_mode_query + ego_feat_exp
                            attended_plan = self.layers[i](
                                self._project_deformable_feature(daf_query),
                                plan_anchor_box,
                                plan_anchor_embed,
                                feature_maps,
                                metas,
                            )
                            plan_mode_query = self._project_deformable_output(
                                attended_plan
                            )
                            instance_feature = torch.cat(
                                [agent_feature, instance_feature[:, num_anchor:]], dim=1
                            )
                        else:
                            # Match deformable query shape to anchor box shape:
                            # ego_feat broadcast to all plan-mode (or mode-time) slots.
                            num_plan_queries = plan_anchor_box.shape[1]
                            ego_feat_exp_q = instance_feature[:, num_anchor:num_anchor+1].expand(
                                -1, num_plan_queries, -1
                            )
                            attended_plan = self.layers[i](
                                self._project_deformable_feature(ego_feat_exp_q),
                                plan_anchor_box,
                                plan_anchor_embed,
                                feature_maps,
                                metas,
                            )
                            attended_plan = self._project_deformable_output(
                                attended_plan
                            )  # (bs, num_plan_queries, embed_dims)
                            if self.mode_no_agg:
                                # Bypass softmax aggregation: additively blend per-query
                                # attended features into plan_mode_query (preserves
                                # plan_anchor_encoder gradient + adds per-mode signal).
                                # Use uniform mean for ego_new (shape preservation).
                                plan_weights = torch.full(
                                    (bs, num_plan_queries), 1.0 / num_plan_queries,
                                    device=det_anchors.device, dtype=det_anchors.dtype,
                                )
                                plan_mode_query = plan_mode_query + attended_plan
                            elif planning_classification and not self.plan_mode_time_queries:
                                plan_weights = (
                                    planning_classification[-1].detach()
                                    .squeeze(1).softmax(dim=-1)
                                )  # (bs, num_plan_modes)
                            else:
                                plan_weights = torch.full(
                                    (bs, num_plan_queries), 1.0 / num_plan_queries,
                                    device=det_anchors.device, dtype=det_anchors.dtype,
                                )
                            ego_new = (attended_plan * plan_weights.unsqueeze(-1)).sum(
                                dim=1, keepdim=True
                            )  # (bs, 1, embed_dims)
                            # Preserve DN tokens after the updated ego.
                            instance_feature = torch.cat(
                                [agent_feature, ego_new, instance_feature[:, num_anchor+1:]], dim=1
                            )
                    else:
                        if self.planning_deformable_waypoints is not None:
                            K = len(self.planning_deformable_waypoints)
                            num_mode = plan_mode_query.shape[1]
                            boxes_multi = self._build_planning_anchor_boxes_multi(
                                plan_anchor.detach(), ego_anchor,
                                self.planning_deformable_waypoints,
                            )  # (bs, num_mode, K, 11)
                            boxes_flat = boxes_multi.reshape(bs, num_mode * K, 11)
                            embed_flat = anchor_encoder(boxes_flat)
                            query_exp = (
                                plan_mode_query.unsqueeze(2)
                                .expand(-1, -1, K, -1)
                                .reshape(bs, num_mode * K, self.embed_dims)
                            )
                            attended = self.layers[i](
                                self._project_deformable_feature(query_exp),
                                boxes_flat,
                                embed_flat,
                                feature_maps, metas,
                            )
                            plan_mode_query = self._project_deformable_output(
                                attended
                            ).reshape(
                                bs, num_mode, K, self.embed_dims
                            ).mean(dim=2)
                        else:
                            plan_mode_query = self._project_deformable_output(
                                self.layers[i](
                                self._project_deformable_feature(plan_mode_query),
                                plan_anchor_box,
                                plan_anchor_embed,
                                feature_maps,
                                metas,
                            ))
                        if self.plan_time_attn and self.plan_mode_time_queries:
                            # Per-mode time-axis self-attention to restore
                            # intra-mode temporal coupling. Reshape
                            # (bs, M*T, D) -> (bs*M, T, D), add learnable
                            # time pos embed, MHA self-attn with residual + LN.
                            M_total = 3 * self.ego_fut_mode
                            T_q = self.ego_fut_ts
                            pq_mt = plan_mode_query.reshape(
                                bs, M_total, T_q, self.embed_dims
                            )
                            pq_seq = (pq_mt + self.plan_time_pos_embed).reshape(
                                bs * M_total, T_q, self.embed_dims
                            )
                            ta_idx = _deformable_stage_idx - 1
                            ta_out, _ = self.plan_time_attn_layers[ta_idx](
                                pq_seq, pq_seq, pq_seq, need_weights=False
                            )
                            pq_seq = self.plan_time_attn_norms[ta_idx](
                                pq_seq + ta_out
                            )
                            plan_mode_query = pq_seq.reshape(
                                bs, M_total * T_q, self.embed_dims
                            )
                        instance_feature = torch.cat(
                            [agent_feature, instance_feature[:, num_anchor:]], dim=1
                        )
                else:
                    instance_feature = torch.cat(
                        [agent_feature, instance_feature[:, num_anchor:]], dim=1
                    )
            elif op == "refine":
                motion_query = motion_mode_query + (instance_feature + anchor_embed)[:, :num_anchor].unsqueeze(2)
                # Use only the ego token (index num_anchor), not DN tokens that follow it.
                plan_query = plan_mode_query.unsqueeze(1) + (instance_feature + anchor_embed)[:, num_anchor:num_anchor+1].unsqueeze(2)
                agent_features_for_refine = (
                    instance_feature[:, :num_anchor] if self.with_conflict_head else None
                )
                (
                    motion_cls,
                    motion_reg,
                    plan_cls,
                    plan_reg,
                    plan_status,
                    plan_da,
                    plan_conflict,
                ) = self.layers[i](
                    motion_query,
                    plan_query,
                    instance_feature[:, num_anchor:num_anchor+1],
                    anchor_embed[:, num_anchor:num_anchor+1],
                    agent_features=agent_features_for_refine,
                )
                if plan_da is not None:
                    planning_da_logits.append(plan_da)
                if plan_conflict is not None:
                    planning_conflict_logits.append(plan_conflict)
                if self.motion_cumulative_refinement and motion_prediction:
                    motion_reg = motion_reg + motion_prediction[-1].detach()
                if self.planning_cumulative_refinement and planning_prediction:
                    plan_reg = plan_reg + planning_prediction[-1].detach()
                motion_classification.append(motion_cls)
                motion_prediction.append(motion_reg)
                planning_classification.append(plan_cls)
                planning_prediction.append(plan_reg)
                planning_status.append(plan_status)
                # Update mode anchor queries for the next decoder iteration.
                # cumsum converts delta trajectories to absolute endpoints.
                motion_anchor_upd = motion_reg.detach().cumsum(dim=-2)
                plan_anchor_upd = plan_reg.detach().squeeze(1).cumsum(dim=-2)
                motion_mode_query = self.motion_anchor_encoder(
                    gen_sineembed_for_position(
                        motion_anchor_upd[..., -1, :], hidden_dim=self.embed_dims
                    )
                )
                if self.motion_deformable_multimode:
                    motion_endpoint_anchor_all = self._build_motion_endpoint_anchors_all_modes(
                        motion_anchor_upd, det_anchors
                    )
                elif self.motion_deformable:
                    motion_endpoint_anchor = self._build_motion_endpoint_anchors(
                        motion_anchor_upd,
                        det_anchors,
                        motion_classification[-1].detach(),
                    )
                plan_anchor = plan_anchor_upd
                if self.plan_mode_time_queries:
                    plan_mode_query = self.plan_anchor_encoder(
                        gen_sineembed_for_position(
                            plan_anchor, hidden_dim=self.embed_dims
                        )
                    ).flatten(1, 2)
                else:
                    plan_mode_query = self.plan_anchor_encoder(
                        gen_sineembed_for_position(
                            plan_anchor[..., -1, :], hidden_dim=self.embed_dims
                        )
                    )
        
        cache_motion_feature = (
            self._project_cache_feature(instance_feature[:, :num_anchor])
            if self.use_planning_input_proj
            else instance_feature[:, :num_anchor]
        )
        cache_ego_feature = (
            self._project_cache_feature(instance_feature[:, num_anchor:num_anchor+1])
            if self.use_planning_input_proj
            else instance_feature[:, num_anchor:num_anchor+1]
        )
        self.instance_queue.cache_motion(cache_motion_feature, det_output, metas)
        # Cache only the real ego token, not DN ego tokens.
        self.instance_queue.cache_planning(cache_ego_feature, plan_status)

        motion_output = {
            "classification": motion_classification,
            "prediction": motion_prediction,
            "period": self.instance_queue.period,
            "anchor_queue": self.instance_queue.anchor_queue,
        }
        planning_output = {
            "classification": planning_classification,
            "prediction": planning_prediction,
            "status": planning_status,
            "period": self.instance_queue.ego_period,
            "anchor_queue": self.instance_queue.ego_anchor_queue,
        }
        if planning_da_logits:
            planning_output["da_logits"] = planning_da_logits
        if planning_conflict_logits:
            planning_output["conflict_logits"] = planning_conflict_logits

        if self.training:
            refine_module = self.layers[self._dn_refine_idx]
            if num_dn_agents > 0 and dn_reg_target_pred is not None:
                dn_a_start = num_anchor + 1
                dn_a_end = num_anchor + 1 + num_dn_agents
                dn_inst = instance_feature[:, dn_a_start:dn_a_end]
                dn_inst_embed = anchor_embed[:, dn_a_start:dn_a_end]
                # dn_motion_mode_query: (bs, num_dn_agents, 1, embed_dims)
                dn_mq = dn_motion_mode_query + (dn_inst + dn_inst_embed).unsqueeze(2)
                dn_motion_reg = refine_module.motion_reg_branch(dn_mq).reshape(
                    bs, num_dn_agents, self.fut_ts, 2
                )
                motion_output['dn_motion_reg'] = dn_motion_reg
                motion_output['dn_motion_reg_target'] = dn_reg_target_pred
                motion_output['dn_motion_valid'] = dn_valid_pred

            if num_dn_ego > 0 and dn_reg_target_plan is not None:
                dn_e_start = num_anchor + 1 + num_dn_agents
                dn_e_end = num_anchor + 1 + num_dn_agents + num_dn_ego
                dn_ego_inst = instance_feature[:, dn_e_start:dn_e_end]
                dn_ego_embed = anchor_embed[:, dn_e_start:dn_e_end]
                # dn_plan_mode_query_var: (bs, num_dn_ego, 1, embed_dims)
                dn_pq = dn_plan_mode_query_var + (dn_ego_inst + dn_ego_embed).unsqueeze(2)
                dn_plan_reg = refine_module.plan_reg_branch(dn_pq).reshape(
                    bs, num_dn_ego, self.ego_fut_ts, 2
                )
                planning_output['dn_plan_reg'] = dn_plan_reg
                planning_output['dn_plan_reg_target'] = dn_reg_target_plan
                planning_output['dn_plan_valid'] = dn_valid_plan

        return motion_output, planning_output
    
    def loss(self,
        motion_model_outs, 
        planning_model_outs,
        data, 
        motion_loss_cache
    ):
        loss = {}
        motion_loss = self.loss_motion(motion_model_outs, data, motion_loss_cache)
        loss.update(motion_loss)
        planning_loss = self.loss_planning(planning_model_outs, data, motion_loss_cache)
        loss.update(planning_loss)
        return loss

    @force_fp32(apply_to=("model_outs"))
    def loss_motion(self, model_outs, data, motion_loss_cache):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        output = {}
        for decoder_idx, (cls, reg) in enumerate(
            zip(cls_scores, reg_preds)
        ):
            (
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
                num_pos
            ) = self.motion_sampler.sample(
                reg,
                data["gt_agent_fut_trajs"],
                data["gt_agent_fut_masks"],
                motion_loss_cache,
            )
            num_pos = max(reduce_mean(num_pos), 1.0)

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.motion_loss_cls(cls, cls_target, weight=cls_weight, avg_factor=num_pos)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_pred = reg_pred.cumsum(dim=-2)
            reg_target = reg_target.cumsum(dim=-2)
            reg_loss = self.motion_loss_reg(
                reg_pred, reg_target, weight=reg_weight, avg_factor=num_pos
            )

            output.update(
                {
                    f"motion_loss_cls_{decoder_idx}": cls_loss,
                    f"motion_loss_reg_{decoder_idx}": reg_loss,
                }
            )

        if 'dn_motion_reg' in model_outs:
            dn_reg = model_outs['dn_motion_reg']          # (bs, N, fut_ts, 2)
            dn_target = model_outs['dn_motion_reg_target'] # (bs, N, fut_ts, 2)
            dn_valid = model_outs['dn_motion_valid']        # (bs, N) bool
            dn_valid_flat = dn_valid.flatten()
            dn_reg_flat = dn_reg.flatten(0, 1)[dn_valid_flat].cumsum(dim=-2)
            dn_tgt_flat = dn_target.flatten(0, 1)[dn_valid_flat].cumsum(dim=-2)
            dn_num_pos = max(reduce_mean(dn_valid.sum().to(dn_reg.dtype)), 1.0)
            output['motion_loss_dn_reg'] = (
                self.motion_loss_reg(dn_reg_flat, dn_tgt_flat, avg_factor=dn_num_pos)
                * self.dn_pred_loss_weight
            )

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_planning(self, model_outs, data, motion_loss_cache=None):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        status_preds = model_outs["status"]
        da_logits_list = model_outs.get("da_logits", [])
        conf_logits_list = model_outs.get("conflict_logits", [])
        output = {}
        for decoder_idx, (cls, reg, status) in enumerate(
            zip(cls_scores, reg_preds, status_preds)
        ):
            (
                cls,
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
            ) = self.planning_sampler.sample(
                cls,
                reg,
                data['gt_ego_fut_trajs'],
                data['gt_ego_fut_masks'],
                data,
            )
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)

            if self.plan_mode_softtgt:
                # Distance-softmax weighted L1 across all 6 modes for the
                # cmd-indexed slice. Replaces winner-takes-all L1 to densify
                # mode gradients and mitigate mode collapse.
                bs_st = reg.shape[0]
                M = self.ego_fut_mode
                cmd_idx = data['gt_ego_fut_cmd'].argmax(dim=-1)
                bs_idx = torch.arange(bs_st, device=reg.device)
                reg_modes = reg.reshape(
                    bs_st, 3, M, self.ego_fut_ts, 2
                )[bs_idx, cmd_idx]  # (bs, M, T, 2)
                gt_traj_exp = data['gt_ego_fut_trajs'].unsqueeze(1)
                gt_mask_exp = data['gt_ego_fut_masks'].unsqueeze(1)

                reg_cum = reg_modes.cumsum(dim=-2)
                gt_cum = gt_traj_exp.cumsum(dim=-2)
                dist = torch.linalg.norm(reg_cum - gt_cum, dim=-1)
                denom = gt_mask_exp.sum(dim=-1).clamp(min=1.0)
                mean_dist = (dist * gt_mask_exp).sum(dim=-1) / denom
                # Detach so weight gradients don't amplify mode collapse.
                weights_st = torch.softmax(
                    -mean_dist.detach() / self.plan_mode_softtgt_tau, dim=-1
                )

                diff = (reg_modes - gt_traj_exp).abs()
                masked_diff = diff * gt_mask_exp.unsqueeze(-1)
                per_mode_l1 = (
                    masked_diff.sum(dim=(-1, -2)) / float(self.ego_fut_ts * 2)
                )
                sample_loss = (per_mode_l1 * weights_st).sum(dim=-1).mean()
                reg_loss_weight = getattr(self.plan_loss_reg, 'loss_weight', 1.0)
                reg_loss = sample_loss * reg_loss_weight
            else:
                reg_loss = self.plan_loss_reg(
                    reg_pred, reg_target, weight=reg_weight
                )
            status_loss = self.plan_loss_status(status.squeeze(1), data['ego_status'])

            output.update(
                {
                    f"planning_loss_cls_{decoder_idx}": cls_loss,
                    f"planning_loss_reg_{decoder_idx}": reg_loss,
                    f"planning_loss_status_{decoder_idx}": status_loss,
                }
            )

            if self.plan_diversity_reg:
                # Pairwise-similarity penalty across cmd-indexed plan modes.
                # Encourages spread between mode trajectories at scale set by sigma.
                bs_d = reg.shape[0]
                M = self.ego_fut_mode
                cmd_idx_d = data['gt_ego_fut_cmd'].argmax(dim=-1)
                bs_idx_d = torch.arange(bs_d, device=reg.device)
                reg_modes_d = reg.reshape(
                    bs_d, 3, M, self.ego_fut_ts, 2
                )[bs_idx_d, cmd_idx_d]
                reg_cum_d = reg_modes_d.cumsum(dim=-2)
                flat_d = reg_cum_d.reshape(bs_d, M, -1)
                pdist = torch.cdist(flat_d, flat_d)
                eye_mask = 1.0 - torch.eye(M, device=pdist.device).unsqueeze(0)
                sigma2 = float(self.plan_diversity_sigma) ** 2
                sim = torch.exp(-pdist ** 2 / (2.0 * sigma2)) * eye_mask
                div_loss = sim.sum(dim=(1, 2)).mean() / float(M * (M - 1))
                output[f"planning_loss_div_{decoder_idx}"] = (
                    div_loss * self.plan_diversity_loss_weight
                )

            if decoder_idx < len(da_logits_list):
                da_loss = self._loss_planning_da(
                    da_logits_list[decoder_idx], reg, data
                )
                if da_loss is not None:
                    output[f"planning_loss_da_{decoder_idx}"] = (
                        da_loss * self.da_loss_weight
                    )

            if decoder_idx < len(conf_logits_list):
                conf_loss = self._loss_planning_conflict(
                    conf_logits_list[decoder_idx], reg, data, motion_loss_cache
                )
                if conf_loss is not None:
                    output[f"planning_loss_conf_{decoder_idx}"] = (
                        conf_loss * self.conflict_loss_weight
                    )

            if self.plan_softcost_collision_enable:
                softcost_col = self._loss_planning_softcost_collision(reg, data)
                if softcost_col is not None:
                    output[f"planning_loss_softcost_col_{decoder_idx}"] = (
                        softcost_col * self.plan_softcost_collision_weight
                    )

        if 'dn_plan_reg' in model_outs:
            dn_reg = model_outs['dn_plan_reg']          # (bs, num_dn, ego_fut_ts, 2)
            dn_target = model_outs['dn_plan_reg_target'] # (bs, num_dn, ego_fut_ts, 2)
            dn_valid = model_outs['dn_plan_valid']        # (bs, num_dn) bool
            dn_valid_flat = dn_valid.flatten()
            dn_reg_flat = dn_reg.flatten(0, 1)[dn_valid_flat].cumsum(dim=-2)
            dn_tgt_flat = dn_target.flatten(0, 1)[dn_valid_flat].cumsum(dim=-2)
            dn_num_pos = max(reduce_mean(dn_valid.sum().to(dn_reg.dtype)), 1.0)
            output['planning_loss_dn_reg'] = (
                self.plan_loss_reg(dn_reg_flat, dn_tgt_flat, avg_factor=dn_num_pos)
                * self.dn_plan_loss_weight
            )

        return output

    def _loss_planning_da(self, da_logits, reg, data):
        """Drivable-area compliance BCE on per-mode per-waypoint predictions.

        da_logits: (bs, 1, M, ego_fut_ts) raw logits.
        reg: (bs, 1, M, ego_fut_ts, 2) delta-XY plan predictions in lidar frame.
        data['gt_drivable_mask']: (bs, H_pix, W_pix) uint8/long, 1=drivable.
        """
        gt_mask = data.get('gt_drivable_mask')
        if gt_mask is None:
            return None
        if isinstance(gt_mask, list):
            gt_mask = torch.stack([m for m in gt_mask], dim=0)
        gt_mask = gt_mask.to(device=da_logits.device, dtype=da_logits.dtype)
        bs = da_logits.shape[0]
        M = da_logits.shape[2]
        T = da_logits.shape[3]

        pred_xy = reg.detach().squeeze(1).cumsum(dim=-2)  # (bs, M, T, 2)
        # Normalize lidar XY to grid_sample range [-1, 1]:
        # roi_size = (W_m, H_m). x is lateral (col), y is longitudinal (row).
        nx = pred_xy[..., 0] / (self.roi_size[0] / 2.0)
        ny = pred_xy[..., 1] / (self.roi_size[1] / 2.0)
        in_roi = (nx.abs() <= 1.0) & (ny.abs() <= 1.0)

        grid = torch.stack([nx, ny], dim=-1).reshape(bs, 1, M * T, 2)
        sampled = F.grid_sample(
            gt_mask.unsqueeze(1),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )
        target = (sampled.squeeze(1).squeeze(1).reshape(bs, M, T) > 0.5).float()

        logits = da_logits.squeeze(1)  # (bs, M, T)
        weight = in_roi.to(logits.dtype)
        loss = F.binary_cross_entropy_with_logits(
            logits, target, reduction='none'
        )
        denom = weight.sum().clamp(min=1.0)
        return (loss * weight).sum() / denom

    def _loss_planning_conflict(self, conf_logits, reg, data, motion_loss_cache):
        """Object-conflict BCE on (matched-anchor, plan-mode) pairs.

        conf_logits: (bs, num_anchor, M) raw logits.
        reg: (bs, 1, M, ego_fut_ts, 2) delta plan predictions.
        Hungarian indices from motion_loss_cache pair pred_idx <-> GT idx.
        """
        if motion_loss_cache is None:
            return None
        gt_boxes = data.get('gt_bboxes_3d')
        gt_traj = data.get('gt_agent_fut_trajs')
        gt_traj_mask = data.get('gt_agent_fut_masks')
        if gt_boxes is None or gt_traj is None:
            return None

        bs, num_anchor, M = conf_logits.shape
        device = conf_logits.device
        T_ego = reg.shape[-2]

        ego_pred_xy = reg.detach().squeeze(1).cumsum(dim=-2)  # (bs, M, T_ego, 2)

        target = conf_logits.new_zeros(bs, num_anchor, M)
        weight = conf_logits.new_zeros(bs, num_anchor, M)

        thr2 = float(self.conflict_threshold) ** 2
        for b in range(bs):
            pred_idx, target_idx = motion_loss_cache['indices'][b]
            if pred_idx is None or len(pred_idx) == 0:
                continue
            boxes_b = gt_boxes[b].to(device)
            trajs_b = gt_traj[b].to(device)
            if trajs_b.shape[0] == 0:
                continue
            T = min(trajs_b.shape[1], T_ego)
            if T == 0:
                continue
            agent_xy0 = boxes_b[target_idx, :2]  # (n_pos, 2)
            agent_traj = trajs_b[target_idx, :T]  # (n_pos, T, 2)
            agent_pos = agent_xy0.unsqueeze(1) + agent_traj.cumsum(dim=1)  # (n_pos, T, 2)

            ego_pos = ego_pred_xy[b, :, :T, :]  # (M, T, 2)

            diff = agent_pos.unsqueeze(1) - ego_pos.unsqueeze(0)  # (n_pos, M, T, 2)
            d2 = (diff ** 2).sum(dim=-1)  # (n_pos, M, T)

            if gt_traj_mask is not None:
                tmask = gt_traj_mask[b].to(device)[target_idx, :T].bool()
                d2 = torch.where(
                    tmask.unsqueeze(1).expand(-1, M, -1),
                    d2,
                    d2.new_full((), float('inf')),
                )
            min_d2 = d2.min(dim=-1).values  # (n_pos, M)
            label = (min_d2 < thr2).to(target.dtype)  # (n_pos, M)

            if isinstance(pred_idx, torch.Tensor):
                pidx = pred_idx.to(device).long()
            else:
                pidx = torch.as_tensor(pred_idx, device=device, dtype=torch.long)
            target[b, pidx] = label
            weight[b, pidx] = 1.0

        denom = weight.sum().clamp(min=1.0)
        pos_w = conf_logits.new_tensor([float(self.conflict_pos_weight)])
        loss = F.binary_cross_entropy_with_logits(
            conf_logits, target, reduction='none', pos_weight=pos_w
        )
        return (loss * weight).sum() / denom

    def _loss_planning_softcost_collision(self, reg, data):
        """Soft collision cost on plan_reg as a training-loss term.

        Dispatches to one of two geometries:
          - 'gaussian': v1 — center-to-center exp(-d²/σ²), mean over agents.
          - 'sdf_corners': v2 — min-of-4-ego-corners SDF to agent oriented bbox,
            softplus(-min_sdf/τ), closest-agent-only per (mode, t), mean over (M, T).
        """
        if self.plan_softcost_geometry == 'sdf_corners':
            return self._loss_planning_softcost_collision_sdf(reg, data)

        gt_boxes = data.get('gt_bboxes_3d')
        gt_traj = data.get('gt_agent_fut_trajs')
        gt_traj_mask = data.get('gt_agent_fut_masks')
        if gt_boxes is None or gt_traj is None:
            return None

        bs = reg.shape[0]
        device = reg.device
        T_ego = reg.shape[-2]
        sigma2 = float(self.plan_softcost_collision_sigma) ** 2

        ego_pred_xy = reg.squeeze(1).cumsum(dim=-2)  # (bs, M_total, T_ego, 2)

        sample_costs = []
        for b in range(bs):
            boxes_b = gt_boxes[b].to(device=device, dtype=ego_pred_xy.dtype)
            trajs_b = gt_traj[b].to(device=device, dtype=ego_pred_xy.dtype)
            if trajs_b.shape[0] == 0:
                continue
            T = min(trajs_b.shape[1], T_ego)
            if T == 0:
                continue
            agent_xy0 = boxes_b[:, :2]
            agent_traj = trajs_b[:, :T]
            agent_pos = agent_xy0.unsqueeze(1) + agent_traj.cumsum(dim=1)

            ego_pos = ego_pred_xy[b, :, :T, :]

            diff = agent_pos.unsqueeze(0) - ego_pos.unsqueeze(1)
            d2 = (diff ** 2).sum(dim=-1)

            cost_per = torch.exp(-d2 / sigma2)
            if gt_traj_mask is not None:
                tmask = gt_traj_mask[b].to(device=device).bool()[:, :T]
                tmask_e = tmask.unsqueeze(0).expand(d2.shape[0], -1, -1).to(cost_per.dtype)
                cost_per = cost_per * tmask_e
                denom = tmask_e.sum().clamp(min=1.0)
            else:
                denom = cost_per.new_tensor(float(cost_per.numel())).clamp(min=1.0)
            sample_costs.append(cost_per.sum() / denom)

        if not sample_costs:
            return reg.sum() * 0.0
        return torch.stack(sample_costs).mean()

    def _loss_planning_softcost_collision_sdf(self, reg, data):
        """Geometry-aware soft collision cost via 4-ego-corner SDF.

        Per (mode, t):
          - Ego footprint matches HierarchicalPlanningDecoder.rescore() constants
            (4.08 × 1.73 × 1.1 with +0.5 m forward offset along heading).
          - Ego heading from trajectory tangent (atan2 central diff on cumxy).
          - For each of 4 ego corners, compute SDF to each agent's oriented bbox
            (rotate corner into agent local frame, axis-aligned-rect SDF).
          - min over 4 corners → per-agent SDF.
          - Closest-agent-only: min over agents → per-(mode, t) min SDF.
          - Penalty = softplus(-min_sdf / τ); mean over (M_total, T_ego).
        Mean across batch.
        """
        gt_boxes = data.get('gt_bboxes_3d')
        gt_traj = data.get('gt_agent_fut_trajs')
        gt_traj_mask = data.get('gt_agent_fut_masks')
        if gt_boxes is None or gt_traj is None:
            return None

        bs = reg.shape[0]
        device = reg.device
        dtype = reg.dtype
        T_ego = reg.shape[-2]
        tau = float(self.plan_softcost_collision_tau)

        # Ego footprint matches rescore() constants (length × width × dim_scale).
        # Box-frame index 3 ('W') is along-heading; index 4 ('L') is lateral.
        ego_W = 4.08 * 1.1
        ego_L = 1.73 * 1.1
        ego_offset = 0.5
        half_W_e = ego_W * 0.5
        half_L_e = ego_L * 0.5

        # ego_pred_xy: (bs, M_total, T_ego, 2). reg is (bs, 1, M_total, T_ego, 2).
        ego_pred_xy = reg.squeeze(1).cumsum(dim=-2)
        bs_, M, T, _ = ego_pred_xy.shape

        # Ego heading from trajectory tangent (central diff on extended cumxy).
        zeros = ego_pred_xy.new_zeros(bs_, M, 1, 2)
        cum_ext = torch.cat([zeros, ego_pred_xy], dim=-2)  # (bs, M, T+1, 2)
        if T >= 2:
            central = cum_ext[..., 2:T + 1, :] - cum_ext[..., 0:T - 1, :]  # (bs, M, T-1, 2)
            last_diff = cum_ext[..., T:T + 1, :] - cum_ext[..., T - 1:T, :]  # (bs, M, 1, 2)
            diffs = torch.cat([central, last_diff], dim=-2)  # (bs, M, T, 2)
        else:
            diffs = cum_ext[..., 1:, :] - cum_ext[..., :-1, :]
        yaw = torch.atan2(diffs[..., 1], diffs[..., 0])  # (bs, M, T)

        # Static guard mirroring rescore.get_yaw: when total displacement is small,
        # heading is unstable; pin to lidar +y (np.pi/2).
        total_disp = torch.linalg.norm(
            ego_pred_xy[..., -1, :] - ego_pred_xy[..., 0, :], dim=-1, keepdim=True
        )
        static_mask = total_disp < 0.5  # (bs, M, 1)
        yaw = torch.where(
            static_mask.expand_as(yaw),
            torch.full_like(yaw, float(np.pi / 2)),
            yaw,
        )
        cos_y = torch.cos(yaw)  # (bs, M, T)
        sin_y = torch.sin(yaw)

        # Apply forward offset along heading to ego center.
        ego_cx = ego_pred_xy[..., 0] + ego_offset * cos_y
        ego_cy = ego_pred_xy[..., 1] + ego_offset * sin_y

        # 4 ego corners in box-local frame: (±half_W_e, ±half_L_e).
        sign_w = ego_pred_xy.new_tensor([1.0, 1.0, -1.0, -1.0])
        sign_l = ego_pred_xy.new_tensor([1.0, -1.0, -1.0, 1.0])
        cx_local = (sign_w * half_W_e).reshape(1, 1, 1, 4)
        cy_local = (sign_l * half_L_e).reshape(1, 1, 1, 4)
        cos_y_b = cos_y.unsqueeze(-1)  # (bs, M, T, 1)
        sin_y_b = sin_y.unsqueeze(-1)
        corners_world_x = cx_local * cos_y_b - cy_local * sin_y_b + ego_cx.unsqueeze(-1)
        corners_world_y = cx_local * sin_y_b + cy_local * cos_y_b + ego_cy.unsqueeze(-1)

        sample_costs = []
        for b in range(bs):
            boxes_b = gt_boxes[b].to(device=device, dtype=dtype)
            trajs_b = gt_traj[b].to(device=device, dtype=dtype)
            if trajs_b.shape[0] == 0:
                continue
            T_b = min(trajs_b.shape[1], T_ego)
            if T_b == 0:
                continue
            n_a = boxes_b.shape[0]
            agent_xy0 = boxes_b[:, :2]
            agent_traj = trajs_b[:, :T_b]
            agent_pos = agent_xy0.unsqueeze(1) + agent_traj.cumsum(dim=1)  # (n_a, T_b, 2)
            agent_yaw = boxes_b[:, 6]  # (n_a,)
            agent_W = boxes_b[:, 3]
            agent_L = boxes_b[:, 4]

            ex = corners_world_x[b, :, :T_b, :]  # (M, T_b, 4)
            ey = corners_world_y[b, :, :T_b, :]
            agent_pos_x = agent_pos[..., 0].reshape(n_a, 1, T_b, 1)
            agent_pos_y = agent_pos[..., 1].reshape(n_a, 1, T_b, 1)
            rel_x = ex.unsqueeze(0) - agent_pos_x  # (n_a, M, T_b, 4)
            rel_y = ey.unsqueeze(0) - agent_pos_y

            cos_a = torch.cos(agent_yaw).reshape(n_a, 1, 1, 1)
            sin_a = torch.sin(agent_yaw).reshape(n_a, 1, 1, 1)
            # Rotate by -agent_yaw into agent-local frame.
            local_x = rel_x * cos_a + rel_y * sin_a
            local_y = -rel_x * sin_a + rel_y * cos_a

            half_w_a = (agent_W * 0.5).reshape(n_a, 1, 1, 1)
            half_l_a = (agent_L * 0.5).reshape(n_a, 1, 1, 1)
            qx = local_x.abs() - half_w_a
            qy = local_y.abs() - half_l_a
            outside = torch.sqrt(
                torch.clamp(qx, min=0) ** 2 + torch.clamp(qy, min=0) ** 2 + 1e-12
            )
            inside = torch.clamp(torch.maximum(qx, qy), max=0)
            sdf = outside + inside  # (n_a, M, T_b, 4)
            min_sdf_corners = sdf.min(dim=-1).values  # (n_a, M, T_b)

            if gt_traj_mask is not None:
                tmask = gt_traj_mask[b].to(device=device).bool()[:, :T_b]  # (n_a, T_b)
                min_sdf_corners = torch.where(
                    tmask.unsqueeze(1).expand(-1, M, -1),
                    min_sdf_corners,
                    min_sdf_corners.new_full((), float('inf')),
                )

            min_sdf_agent = min_sdf_corners.min(dim=0).values  # (M, T_b)
            finite = torch.isfinite(min_sdf_agent)
            if not finite.any():
                continue
            penalty = F.softplus(
                -torch.where(finite, min_sdf_agent, min_sdf_agent.new_zeros(())) / tau
            )
            penalty = torch.where(finite, penalty, penalty.new_zeros(()))
            denom = finite.to(penalty.dtype).sum().clamp(min=1.0)
            sample_costs.append(penalty.sum() / denom)

        if not sample_costs:
            return reg.sum() * 0.0
        return torch.stack(sample_costs).mean()

    @force_fp32(apply_to=("model_outs"))
    def post_process(
        self,
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        motion_result = self.motion_decoder.decode(
            det_output["classification"],
            det_output["prediction"],
            det_output.get("instance_id"),
            det_output.get("quality"),
            motion_output,
        )
        planning_result = self.planning_decoder.decode(
            det_output,
            motion_output,
            planning_output, 
            data,
        )

        return motion_result, planning_result
