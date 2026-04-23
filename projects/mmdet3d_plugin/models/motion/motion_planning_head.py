from typing import List, Optional, Tuple, Union
import warnings
import copy

import numpy as np
import cv2
import torch
import torch.nn as nn

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
        motion_deformable_modeproj=False,
        deformable_waypoint=-1,
        planning_deformable_waypoints=None,
        num_dn_pred_groups=0,
        dn_pred_noise_scale=0.5,
        dn_pred_loss_weight=1.0,
        num_dn_plan_groups=0,
        dn_plan_noise_scale=0.5,
        dn_plan_loss_weight=1.0,
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
        self.motion_deformable_modeproj = motion_deformable_modeproj and motion_deformable_multimode
        self.deformable_waypoint = deformable_waypoint
        self.planning_deformable_waypoints = planning_deformable_waypoints

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
        plan_anchor = np.load(plan_anchor)
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
        raw_instance_feature = det_output["instance_feature"]
        raw_anchor_embed = det_output["anchor_embed"]
        instance_feature = self._project_instance_feature(raw_instance_feature)
        anchor_embed = self._project_anchor_embed(raw_anchor_embed)
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        _, (instance_feature_selected, anchor_embed_selected) = topk(
            det_confidence, self.num_det, instance_feature, anchor_embed
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
            _, (map_instance_feature_selected, map_anchor_embed_selected) = topk(
                map_confidence, self.num_map, map_instance_feature, map_anchor_embed
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
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    instance_feature_selected,
                    instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=anchor_embed_selected,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "cross_gnn":
                if map_output is None:
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
                    if motion_classification:
                        mode_weights = motion_classification[-1].detach().softmax(dim=-1)
                    else:
                        mode_weights = torch.full(
                            (bs, num_anchor, self.fut_mode), 1.0 / self.fut_mode,
                            device=det_anchors.device, dtype=det_anchors.dtype,
                        )
                    agent_feature = (attended * mode_weights.unsqueeze(-1)).sum(dim=2)
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
                        attended_plan = self.layers[i](
                            self._project_deformable_feature(ego_feat_exp),
                            plan_anchor_box,
                            plan_anchor_embed,
                            feature_maps,
                            metas,
                        )
                        attended_plan = self._project_deformable_output(
                            attended_plan
                        )  # (bs, num_plan_modes, embed_dims)
                        if planning_classification:
                            plan_weights = (
                                planning_classification[-1].detach()
                                .squeeze(1).softmax(dim=-1)
                            )  # (bs, num_plan_modes)
                        else:
                            plan_weights = torch.full(
                                (bs, num_plan_modes), 1.0 / num_plan_modes,
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
                (
                    motion_cls,
                    motion_reg,
                    plan_cls,
                    plan_reg,
                    plan_status,
                ) = self.layers[i](
                    motion_query,
                    plan_query,
                    instance_feature[:, num_anchor:num_anchor+1],
                    anchor_embed[:, num_anchor:num_anchor+1],
                )
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
        planning_loss = self.loss_planning(planning_model_outs, data)
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
    def loss_planning(self, model_outs, data):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        status_preds = model_outs["status"]
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
