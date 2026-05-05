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


def _eval_get_yaw(traj_abs, start_yaw_default=np.pi / 2, static_dis_thresh=0.5):
    """Per-timestep yaw via trajectory tangent — verbatim port of
    ``projects/mmdet3d_plugin/datasets/evaluation/planning/planning_eval.py``
    ``get_yaw``. ``traj_abs`` is *absolute* XY (post-cumsum), shape ``(*, T, 2)``.
    Returns ``(*, T)`` per-timestep yaw. Static fallback (total displacement
    < ``static_dis_thresh``) returns a constant ``start_yaw_default`` over T.
    """
    T = traj_abs.shape[-2]
    start = traj_abs[..., 0, :]
    end = traj_abs[..., -1, :]
    dist = torch.linalg.norm(end - start, dim=-1)  # (*,)
    static_mask = dist < static_dis_thresh

    zeros = traj_abs.new_zeros(traj_abs.shape[:-2] + (1, 2))
    traj_cat = torch.cat([zeros, traj_abs], dim=-2)  # (*, T+1, 2)
    yaw_padded = traj_abs.new_zeros(traj_abs.shape[:-2] + (T + 1,))
    if T >= 2:
        yaw_padded[..., 1:-1] = torch.atan2(
            traj_cat[..., 2:, 1] - traj_cat[..., :-2, 1],
            traj_cat[..., 2:, 0] - traj_cat[..., :-2, 0],
        )
    yaw_padded[..., -1] = torch.atan2(
        traj_cat[..., -1, 1] - traj_cat[..., -2, 1],
        traj_cat[..., -1, 0] - traj_cat[..., -2, 0],
    )
    yaw = yaw_padded[..., 1:]  # (*, T)
    static_yaw = traj_abs.new_full(yaw.shape, float(start_yaw_default))
    yaw = torch.where(static_mask.unsqueeze(-1).expand_as(yaw), static_yaw, yaw)
    return yaw


def _agent_get_yaw(traj_abs, start_yaw_t0, static_dis_thresh=0.5):
    """Per-timestep yaw for agent boxes. Same tangent recipe as ``_eval_get_yaw``
    but with a per-agent ``start_yaw_t0`` (from ``gt_bboxes_3d[..., YAW]``) used
    for the static-fallback constant. Shapes broadcast: ``traj_abs`` is
    ``(*, T, 2)`` and ``start_yaw_t0`` is ``(*,)``.
    """
    T = traj_abs.shape[-2]
    start = traj_abs[..., 0, :]
    end = traj_abs[..., -1, :]
    dist = torch.linalg.norm(end - start, dim=-1)
    static_mask = dist < static_dis_thresh

    zeros = traj_abs.new_zeros(traj_abs.shape[:-2] + (1, 2))
    traj_cat = torch.cat([zeros, traj_abs], dim=-2)
    yaw_padded = traj_abs.new_zeros(traj_abs.shape[:-2] + (T + 1,))
    if T >= 2:
        yaw_padded[..., 1:-1] = torch.atan2(
            traj_cat[..., 2:, 1] - traj_cat[..., :-2, 1],
            traj_cat[..., 2:, 0] - traj_cat[..., :-2, 0],
        )
    yaw_padded[..., -1] = torch.atan2(
        traj_cat[..., -1, 1] - traj_cat[..., -2, 1],
        traj_cat[..., -1, 0] - traj_cat[..., -2, 0],
    )
    yaw = yaw_padded[..., 1:]
    static_yaw = start_yaw_t0.unsqueeze(-1).expand_as(yaw).to(yaw.dtype)
    yaw = torch.where(static_mask.unsqueeze(-1).expand_as(yaw), static_yaw, yaw)
    return yaw


def _make_rect_corners_topdown(cx, cy, box_W, box_L, yaw):
    """Build the 4 base (z=z0) corners of a rotated 3D box in world XY.

    Matches ``box3d_to_corners(box).[..., [0, 3, 7, 4], :2]``: indices
    ``[0, 3, 7, 4]`` are the bottom-plane corners with local-frame
    coordinates ``(±W/2, ±L/2)``, ordered (-W/2,-L/2), (-W/2,+L/2),
    (+W/2,+L/2), (+W/2,-L/2). Rotation is around z by ``yaw``. ``box_W``
    is the local-x extent (vehicle length, since the dataset assigns
    length to the W slot), ``box_L`` is the local-y extent (width).
    All inputs broadcast; outputs ``(*, 4, 2)``.
    """
    half_W = box_W * 0.5
    half_L = box_L * 0.5
    sign_w = cx.new_tensor([-1.0, -1.0, 1.0, 1.0])
    sign_l = cx.new_tensor([-1.0, 1.0, 1.0, -1.0])
    half_W_e = half_W.unsqueeze(-1)
    half_L_e = half_L.unsqueeze(-1)
    lx = sign_w * half_W_e  # (*, 4)
    ly = sign_l * half_L_e
    cos_y = torch.cos(yaw).unsqueeze(-1)
    sin_y = torch.sin(yaw).unsqueeze(-1)
    wx = lx * cos_y - ly * sin_y + cx.unsqueeze(-1)
    wy = lx * sin_y + ly * cos_y + cy.unsqueeze(-1)
    return torch.stack([wx, wy], dim=-1)  # (*, 4, 2)


def _rect_intersects_sat(corners_a, corners_b):
    """SAT (Separating Axis Theorem) intersection test for two convex
    quadrilaterals expressed by their 4 corners in world XY.

    Inputs are ``(*, 4, 2)`` with matching leading dims. Output is ``(*,)``
    boolean. For convex polygons, two polygons intersect iff their
    projections overlap on every separating axis candidate; for
    rectangles only 2 unique axes per polygon are needed.
    """
    def edge_normals(corners):
        # Two edges sharing corner 0: c1-c0 and c3-c0 give the rectangle's
        # local axes; the perpendiculars are the SAT axes.
        e1 = corners[..., 1, :] - corners[..., 0, :]  # (*, 2)
        e2 = corners[..., 3, :] - corners[..., 0, :]
        n1 = torch.stack([-e1[..., 1], e1[..., 0]], dim=-1)
        n2 = torch.stack([-e2[..., 1], e2[..., 0]], dim=-1)
        return torch.stack([n1, n2], dim=-2)  # (*, 2, 2)

    axes = torch.cat([edge_normals(corners_a), edge_normals(corners_b)], dim=-2)  # (*, 4, 2)
    # Project: corners (*, 4, 2) on axes (*, 4, 2) -> (*, 4_axes, 4_corners).
    proj_a = (corners_a.unsqueeze(-3) * axes.unsqueeze(-2)).sum(dim=-1)
    proj_b = (corners_b.unsqueeze(-3) * axes.unsqueeze(-2)).sum(dim=-1)
    min_a, max_a = proj_a.min(dim=-1).values, proj_a.max(dim=-1).values
    min_b, max_b = proj_b.min(dim=-1).values, proj_b.max(dim=-1).values
    overlap = (max_a >= min_b) & (max_b >= min_a)  # (*, 4)
    return overlap.all(dim=-1)


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
        num_driving_cmds=3,
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
        plan_distill_rescore_enable=False,
        plan_distill_rescore_weight=0.05,
        conflict_label_source='predicted',
        conflict_smooth_max_tau=5.0,
        detach_perception=False,
        relevance_selection=False,
        relevance_corridor=2.0,
        relevance_horizon=6,
        skip_perception_kv=False,
        ego_only_planning=False,
        conflict_input='agent_token',
        conflict_image_sampler=None,
        # Detection/map-free conflict-sampler variants (B1.5/B1.6/B1.7).
        # init_anchor_path: path to a fixed anchor file (e.g. kmeans det 900);
        # init_topk: per-scene top-K-nearest-to-ego selection from that file.
        conflict_init_anchor_path=None,
        conflict_init_topk=50,
        # Option 2a: learned scene-query decoder (image_at_scene_query). Refines
        # K BEV anchors via image features; conflict head reads at refined positions.
        conflict_scene_query_decoder=None,
        # Optional aux loss on scene queries (Option 2a-aux): Hungarian match
        # against ego-interacting GT subset (within X m of ego in next Y s),
        # focal cls + L1 box on (x, y).
        scene_query_aux_loss_enable=False,
        scene_query_aux_loss_weight=1.0,
        scene_query_aux_dist_thresh=20.0,
        scene_query_aux_time_steps=6,
        scene_query_aux_cls_weight=1.0,
        scene_query_aux_box_weight=2.0,
        planning_temporal_stack=0,
        planning_temporal_egocomp=True,
        plan_anchor_norm_mode='none',
        plan_anchor_refmag=None,
        plan_anchor_velnorm_eps=1.0,
        plan_magnitude_head_enable=False,
        plan_magnitude_loss_weight=1.0,
        plan_ego_status_encode_enable=False,
        plan_ego_status_indices=(0, 1, 5, 6, 7),
        motion_target_in_agent_frame=False,
        ego_state_estimator=None,
        use_predicted_ego_status=False,
    ):
        super(MotionPlanningHead, self).__init__()
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.num_driving_cmds = num_driving_cmds

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
        self.plan_distill_rescore_enable = plan_distill_rescore_enable
        self.plan_distill_rescore_weight = plan_distill_rescore_weight
        self.conflict_label_source = conflict_label_source
        self.conflict_smooth_max_tau = float(conflict_smooth_max_tau)
        self.detach_perception = detach_perception
        self.relevance_selection = relevance_selection
        self.relevance_corridor = float(relevance_corridor)
        self.relevance_horizon = int(relevance_horizon)
        self.skip_perception_kv = skip_perception_kv
        self.ego_only_planning = ego_only_planning
        self.conflict_input = conflict_input
        _valid_conflict_inputs = (
            'agent_token', 'image_at_det', 'image_at_plan', 'image_at_init_topk',
            'image_at_init_topk_and_plan', 'image_at_ego_grid',
            'image_at_scene_query',
        )
        if conflict_input not in _valid_conflict_inputs:
            raise ValueError(
                f"conflict_input must be one of {_valid_conflict_inputs}, got {conflict_input!r}"
            )
        # conflict_image_sampler is only needed for det/plan/init_topk modes;
        # scene_query mode uses the decoder's own deformable attention.
        _needs_image_sampler = conflict_input in (
            'image_at_det', 'image_at_plan', 'image_at_init_topk',
            'image_at_init_topk_and_plan', 'image_at_ego_grid',
        )
        if _needs_image_sampler:
            assert conflict_image_sampler is not None, (
                f"conflict_input={conflict_input!r} requires conflict_image_sampler config"
            )
            self.conflict_image_sampler = build_from_cfg(conflict_image_sampler, ATTENTION)
        else:
            self.conflict_image_sampler = None
        if conflict_input == 'image_at_scene_query':
            assert conflict_scene_query_decoder is not None, (
                "conflict_input='image_at_scene_query' requires conflict_scene_query_decoder"
            )
            self.scene_query_decoder = build_from_cfg(
                conflict_scene_query_decoder, ATTENTION
            )
        else:
            self.scene_query_decoder = None
        # Aux loss config (active only when scene_query_aux_loss_enable=True
        # AND conflict_input == 'image_at_scene_query').
        self.scene_query_aux_loss_enable = bool(scene_query_aux_loss_enable)
        self.scene_query_aux_loss_weight = float(scene_query_aux_loss_weight)
        self.scene_query_aux_dist_thresh = float(scene_query_aux_dist_thresh)
        self.scene_query_aux_time_steps = int(scene_query_aux_time_steps)
        self.scene_query_aux_cls_weight = float(scene_query_aux_cls_weight)
        self.scene_query_aux_box_weight = float(scene_query_aux_box_weight)
        if self.scene_query_aux_loss_enable:
            assert conflict_input == 'image_at_scene_query', (
                "scene_query_aux_loss_enable requires conflict_input='image_at_scene_query'"
            )
            self.scene_query_cls_head = nn.Linear(embed_dims, 1)
            nn.init.constant_(self.scene_query_cls_head.bias, -2.0)  # rare-positive prior
        else:
            self.scene_query_cls_head = None
        # Defensive: initialise scene-query cache attrs so loss helpers can
        # safely check `is None` even before any forward pass has run.
        self._scene_query_features = None
        self._scene_query_anchors = None
        # Per-mode-restricted aggregation: image_at_plan / image_at_init_topk
        # build a per-(plan-mode, K) anchor pool, and per-mode collision logits
        # should read only from that mode's K samples (smooth-max over K), not
        # over all M*K. Track sample count K so the loss can reshape correctly.
        self._conflict_per_mode_K = None  # set in forward when applicable
        self.conflict_init_topk = int(conflict_init_topk)
        if conflict_input in ('image_at_init_topk', 'image_at_init_topk_and_plan'):
            assert conflict_init_anchor_path is not None, (
                f"conflict_input={conflict_input!r} requires conflict_init_anchor_path"
            )
            init_anchors = np.load(conflict_init_anchor_path)
            # Expect shape (N, 11) — full SparseDrive anchor format.
            assert init_anchors.shape[-1] == 11, (
                f"init anchors must be 11-dim, got shape {init_anchors.shape}"
            )
            self.register_buffer(
                'conflict_init_anchors',
                torch.from_numpy(init_anchors).float(),
                persistent=False,
            )
        else:
            self.conflict_init_anchors = None
        # B1.9: ego-anchored fixed BEV grid (5x5 by default). Shared across
        # plan modes; loss uses global-pool path (per_mode_K = grid size,
        # total = K, smooth-max over all grid points per output mode).
        if conflict_input == 'image_at_ego_grid':
            grid_x = torch.tensor([0.0, 6.0, 12.0, 18.0, 24.0])  # forward (m)
            grid_y = torch.tensor([-8.0, -4.0, 0.0, 4.0, 8.0])  # lateral (m)
            xs, ys = torch.meshgrid(grid_x, grid_y, indexing='ij')
            xy = torch.stack([xs.flatten(), ys.flatten()], dim=-1)  # (25, 2)
            ego_grid = xy.new_zeros(xy.shape[0], 11)
            ego_grid[:, 0:2] = xy
            ego_grid[:, 7] = 1.0  # cos(yaw=0)
            self.register_buffer(
                'conflict_ego_grid_anchors', ego_grid, persistent=False
            )
        else:
            self.conflict_ego_grid_anchors = None
        self.planning_temporal_stack = int(planning_temporal_stack)
        self.planning_temporal_egocomp = bool(planning_temporal_egocomp)
        # Anchor normalization variants. 'none' is the default static-meter
        # behavior. 'endptnorm' loads endpoint-normalized shape anchors and a
        # per-cluster reference-magnitude buffer (refmag); the metric anchor =
        # shape * refmag, fixed per (cmd, mode). 'velnorm' loads velocity-
        # normalized shape anchors (units = seconds); the metric anchor =
        # shape * ||v_0|| with ||v_0|| read per-batch from metas['ego_status'].
        if plan_anchor_norm_mode not in ('none', 'endptnorm', 'velnorm'):
            raise ValueError(
                f"plan_anchor_norm_mode must be one of "
                f"'none'/'endptnorm'/'velnorm'; got {plan_anchor_norm_mode!r}"
            )
        self.plan_anchor_norm_mode = plan_anchor_norm_mode
        self.plan_anchor_velnorm_eps = float(plan_anchor_velnorm_eps)
        self._plan_anchor_refmag_path = plan_anchor_refmag
        self.plan_magnitude_head_enable = bool(plan_magnitude_head_enable)
        self.plan_magnitude_loss_weight = float(plan_magnitude_loss_weight)
        self.plan_ego_status_encode_enable = bool(plan_ego_status_encode_enable)
        self.plan_ego_status_indices = list(plan_ego_status_indices)
        # Predicted ego_status pipeline: when use_predicted_ego_status=True the
        # estimator's output replaces metas['ego_status'] in all downstream
        # consumers (anchor velnorm scaling, plan_ego_status_encoder). The
        # true ego_status is only used as the auxiliary L1 supervision target
        # and to populate the history queue.
        self.use_predicted_ego_status = bool(use_predicted_ego_status)
        if ego_state_estimator is not None:
            self.ego_state_estimator = build_from_cfg(
                ego_state_estimator, PLUGIN_LAYERS
            )
        else:
            self.ego_state_estimator = None
        if self.use_predicted_ego_status:
            assert self.ego_state_estimator is not None, (
                "use_predicted_ego_status=True requires ego_state_estimator config"
            )
        self._predicted_ego_status = None  # populated each forward
        # When True, motion regression operates in each agent's heading-aligned
        # frame: the k-means motion anchor stays in agent frame (no
        # _agent2lidar rotation), the model predicts agent-frame deltas, the
        # GT target is rotated lidar→agent at loss time, and predictions are
        # rotated agent→lidar wherever they're consumed downstream
        # (endpoint-anchor builders, mode-query rebuild, post-process). The
        # rotation uses the predicted yaw from det_anchors at the matched
        # token, so target/prediction live in the same frame at inference too.
        self.motion_target_in_agent_frame = bool(motion_target_in_agent_frame)
        # Cache for temporal feature stacking. Set to None at construction;
        # populated each forward (current frame) and consumed on the next.
        self._temp_col_feats = None
        self._temp_spatial_shape = None
        self._temp_scale_start_index = None
        self._temp_projection_mat = None
        self._temp_T_global = None
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
        # Running stats for `conflict_pos_weight='auto'`. After
        # `_auto_pw_calib_iters` iterations of accumulating (pos, total)
        # over labelled entries, `_auto_pw_value` is frozen to the
        # estimator (1 - p) / p; subsequent calls use the frozen value.
        self._auto_pw_calib_iters = 200
        self._auto_pw_pos = 0.0
        self._auto_pw_total = 0.0
        self._auto_pw_iter = 0
        self._auto_pw_value = None
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
        if self.skip_perception_kv:
            # `skip_perception_kv` accepts True/'both' (null both gnn+cross_gnn),
            # 'det' (null only gnn — det K/V), or 'map' (null only cross_gnn —
            # map K/V). Layers are set to None so DDP doesn't see their params
            # as unused; forward already skips None layers.
            skip_det = self.skip_perception_kv in (True, 'both', 'det')
            skip_map = self.skip_perception_kv in (True, 'both', 'map')
            for i, op in enumerate(self.operation_order):
                if (op == "gnn" and skip_det) or (op == "cross_gnn" and skip_map):
                    self.layers[i] = None

        # Separate planning deformable layers with extended num_cams when
        # temporal stacking is enabled. Detection / motion deformables stay
        # on self.layers[i] with the original num_cams.
        self.planning_temporal_layers = None
        if self.planning_temporal_stack > 0:
            assert deformable_model is not None
            base_num_cams = deformable_model.get("num_cams", 6)
            ext_num_cams = base_num_cams * (self.planning_temporal_stack + 1)
            temporal_cfg = copy.deepcopy(deformable_model)
            temporal_cfg["num_cams"] = ext_num_cams
            self.planning_temporal_layers = nn.ModuleList(
                [
                    build_from_cfg(temporal_cfg, ATTENTION)
                    for _ in range(self._n_deformable_stages)
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

        # Endpoint-normalization variant: load the per-cluster reference
        # magnitude (median ||end|| of the cluster's training trajectories).
        # Shape (num_driving_cmds, ego_fut_mode); used to recover metric
        # sampling positions as shape_anchor * refmag.
        if self.plan_anchor_norm_mode == 'endptnorm':
            if self._plan_anchor_refmag_path is None:
                raise ValueError(
                    "plan_anchor_norm_mode='endptnorm' requires "
                    "plan_anchor_refmag (path to per-cluster refmag .npy)"
                )
            refmag_arr = np.load(self._plan_anchor_refmag_path)
            assert refmag_arr.shape == self.plan_anchor.shape[:2], (
                f"plan_anchor_refmag shape {refmag_arr.shape} does not match "
                f"plan_anchor leading shape {tuple(self.plan_anchor.shape[:2])}"
            )
            self.plan_anchor_refmag = nn.Parameter(
                torch.tensor(refmag_arr, dtype=torch.float32),
                requires_grad=False,
            )

        # Per-decoder-stage scalar magnitude head. One Linear-MLP per refine
        # stage; output shape (bs, num_driving_cmds * ego_fut_mode) at each
        # stage. Supervised below by smooth-L1 against ||gt_end|| for the
        # cmd-indexed best-mode (matches planning_sampler structure).
        if self.plan_magnitude_head_enable:
            n_refine = sum(1 for op in operation_order if op == 'refine')
            self.plan_magnitude_branches = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dims, embed_dims),
                    nn.ReLU(),
                    nn.Linear(embed_dims, embed_dims),
                    nn.ReLU(),
                    nn.Linear(embed_dims, 1),
                )
                for _ in range(n_refine)
            ])

        # ego_status injection. MLP consuming a fixed slice of CAN-frame
        # ego_status (default planar accel + yaw rate + planar velocity);
        # output broadcast-added to plan_mode_query at init and at every
        # per-stage rebuild after refine. The MLP learns the CAN→LiDAR
        # rotation implicitly; no pre-rotation is applied (rotation about z
        # is consistent across both frames under aug).
        if self.plan_ego_status_encode_enable:
            in_dim = len(self.plan_ego_status_indices)
            self.plan_ego_status_encoder = nn.Sequential(
                nn.Linear(in_dim, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, embed_dims),
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

    def _effective_ego_status(self, metas):
        """Return the ego_status tensor consumers should use this forward.

        When ``use_predicted_ego_status`` is enabled, returns the estimator's
        prediction stored in ``self._predicted_ego_status``; otherwise returns
        the true ``metas['ego_status']``.
        """
        if self.use_predicted_ego_status and self._predicted_ego_status is not None:
            return self._predicted_ego_status
        return metas['ego_status']

    def _get_initial_plan_anchor(self, bs, metas):
        """Per-batch metric plan_anchor at decoder init.

        Returns a tensor of shape (bs, num_driving_cmds * ego_fut_mode,
        ego_fut_ts, 2) in metric LiDAR-frame meters. The default 'none'
        path tiles the static (3, M, T, 2) buffer; the 'endptnorm' path
        rescales each (cmd, mode) anchor by the per-cluster reference
        magnitude; the 'velnorm' path rescales by per-batch ||v_0|| read
        from metas['ego_status'][:, 6:8].
        """
        base = self.plan_anchor[None]  # (1, 3, M, T, 2)
        if self.plan_anchor_norm_mode == 'endptnorm':
            scaled = base * self.plan_anchor_refmag[None, ..., None, None]
            plan_anchor = scaled.expand(bs, -1, -1, -1, -1)
        elif self.plan_anchor_norm_mode == 'velnorm':
            ego_status = self._effective_ego_status(metas).to(base.dtype)
            v_xy = ego_status[..., 6:8]
            v_0 = torch.linalg.norm(v_xy, dim=-1)  # (bs,)
            v_0 = v_0.clamp_min(self.plan_anchor_velnorm_eps)
            plan_anchor = base.expand(bs, -1, -1, -1, -1) * v_0[
                :, None, None, None, None
            ]
        else:
            plan_anchor = base.expand(bs, -1, -1, -1, -1)
        return plan_anchor.reshape(bs, -1, self.ego_fut_ts, 2).contiguous()

    def _build_motion_endpoint_anchors(self, motion_anchor_upd, det_anchors, motion_cls):
        """Build 3D box anchors at the best-mode predicted endpoint for each agent.

        motion_anchor_upd: (bs, num_det, fut_mode, fut_ts, 2) cumulative XY
            displacements in lidar-frame orientation (relative to agent position).
        det_anchors: (bs, num_det, 11) current detection boxes.
        motion_cls: (bs, num_det, fut_mode) classification logits.

        Returns (bs, num_det, 11) anchor boxes placed at the predicted endpoints.
        """
        if self.motion_target_in_agent_frame:
            # The model predicts agent-frame deltas; rotate to lidar before
            # treating them as offsets from det_anchors XY (which are lidar).
            motion_anchor_upd = self._agent2lidar(motion_anchor_upd, det_anchors)
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
        if self.motion_target_in_agent_frame:
            motion_anchor_upd = self._agent2lidar(motion_anchor_upd, det_anchors)
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
        if self.motion_target_in_agent_frame:
            # Agent-frame regression: keep the anchor in agent frame so the
            # model predicts agent-frame deltas. Downstream consumers rotate
            # to lidar where needed.
            return motion_anchor
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

    def _rotate_trajs(self, trajs, boxes, agent_to_lidar=True):
        """Rotate trajectories by box yaw, broadcasting to any traj rank.

        trajs: (..., T, 2) with arbitrary leading dims (bs, num_anchor, ...).
        boxes: shape compatible with trajs[..., 0, 0] when expanded — i.e.
            the leading dims of boxes must match the leading dims of trajs
            *up to T*. Boxes carries SIN_YAW / COS_YAW in the last axis.
        agent_to_lidar=True: agent → lidar (R · v).
        agent_to_lidar=False: lidar → agent (R⁻¹ · v); same as negating sin.

        Used as a shape-flexible companion to `_agent2lidar` (which is locked
        to a fixed rank via einsum). Suitable for the (bs, num_anchor, T, 2)
        traj shapes seen in the loss path.
        """
        sin_yaw = boxes[..., SIN_YAW]
        cos_yaw = boxes[..., COS_YAW]
        norm = torch.sqrt(sin_yaw ** 2 + cos_yaw ** 2).clamp_min(1e-6)
        sin_yaw = sin_yaw / norm
        cos_yaw = cos_yaw / norm
        if not agent_to_lidar:
            sin_yaw = -sin_yaw
        # Broadcast (..., 1) so the time axis aligns.
        sin_yaw = sin_yaw.unsqueeze(-1)
        cos_yaw = cos_yaw.unsqueeze(-1)
        x = trajs[..., 0]
        y = trajs[..., 1]
        x_new = cos_yaw * x - sin_yaw * y
        y_new = sin_yaw * x + cos_yaw * y
        return torch.stack([x_new, y_new], dim=-1)

    def _build_temporal_planning_inputs(self, feature_maps, metas):
        """
        Build extended (feature_maps, projection_mat) for the planning
        deformable when planning_temporal_stack > 0.

        Concatenates the cached prev N frames along the camera axis. For each
        past frame, the projection matrix is recomposed so that points in the
        *current* ego frame project into past camera images:
            proj_t-k = proj_past @ T_ego_past_from_global @ T_global_from_ego_curr
        which is equivalent to mapping `p_curr -> p_past -> pix_past`.

        Returns (feature_maps_t, projection_mat_t). If no cache yet (first
        frame in this run), the current frame is replicated for past slots,
        which is harmless for the deformable.
        """
        col_feats, spatial_shape, scale_start_index = feature_maps
        bs = col_feats.shape[0]
        proj_curr = metas["projection_mat"]
        # proj_curr: (bs, n_cam_base, 4, 4)
        n_cam_base = proj_curr.shape[1]
        N = self.planning_temporal_stack
        device = col_feats.device
        dtype = col_feats.dtype

        # Build past col_feats / spatial_shape / scale_start_index lists.
        # Cache stores past frames in order [t-1, t-2, ...], i.e. most recent
        # first. If cache is missing or batch size mismatches (eval-vs-train),
        # replicate current.
        cache_ok = (
            self._temp_col_feats is not None
            and isinstance(self._temp_col_feats, list)
            and len(self._temp_col_feats) > 0
            and self._temp_col_feats[0].shape[0] == bs
        )

        col_list = [col_feats]
        ss_list = [spatial_shape]
        ssi_list = [scale_start_index]
        proj_list = [proj_curr]

        for k in range(N):
            if cache_ok and k < len(self._temp_col_feats):
                past_col = self._temp_col_feats[k].to(device=device, dtype=dtype)
                past_ss = self._temp_spatial_shape[k].to(device=device)
                past_ssi = self._temp_scale_start_index[k].to(device=device)
                past_proj = self._temp_projection_mat[k].to(
                    device=device, dtype=proj_curr.dtype
                )
                if self.planning_temporal_egocomp:
                    # T_ego_past_from_ego_curr = T_global_from_ego_past_inv
                    #     @ T_global_from_ego_curr
                    # But projection_mat[..., :, :] already encodes
                    # (K @ T_cam<-ego_past), so to get a current-ego-input
                    # projection: past_proj @ T_ego_past_from_ego_curr
                    T_global_curr = proj_curr.new_tensor(
                        np.stack([m["T_global"] for m in metas["img_metas"]])
                    )  # (bs, 4, 4)
                    past_T_global = self._temp_T_global[k].to(
                        device=device, dtype=proj_curr.dtype
                    )  # (bs, 4, 4) ego_past -> global
                    T_global_inv_past = torch.linalg.inv(past_T_global)
                    T_egop_from_egoc = torch.matmul(T_global_inv_past, T_global_curr)
                    # (bs, 1, 4, 4)
                    T_egop_from_egoc = T_egop_from_egoc.unsqueeze(1)
                    past_proj = torch.matmul(past_proj, T_egop_from_egoc)
            else:
                # No cache → replicate current. T_temp2cur is identity.
                past_col = col_feats
                past_ss = spatial_shape
                past_ssi = scale_start_index
                past_proj = proj_curr

            col_list.append(past_col)
            ss_list.append(past_ss)
            ssi_list.append(past_ssi)
            proj_list.append(past_proj)

        # Concat col_feats along the (cam*spatial) axis (dim=1).
        col_feats_t = torch.cat(col_list, dim=1)
        # Stack spatial_shape along the cam axis (dim=0). Same H/W per level
        # so concatenation is safe.
        spatial_shape_t = torch.cat(ss_list, dim=0)
        # scale_start_index for past frames must be offset by the cumulative
        # length of preceding entries.
        offsets = [0]
        for col in col_list[:-1]:
            offsets.append(offsets[-1] + col.shape[1])
        ssi_t = torch.cat(
            [s + o for s, o in zip(ssi_list, offsets)], dim=0
        )

        feature_maps_t = [col_feats_t, spatial_shape_t, ssi_t]
        projection_mat_t = torch.cat(proj_list, dim=1)  # (bs, n_cam_base*(N+1), 4, 4)
        return feature_maps_t, projection_mat_t

    def _cache_planning_temporal(self, feature_maps, metas):
        """
        After the forward, push the current frame onto the temporal cache.
        Cache is kept on CPU to avoid GPU memory pressure across iters.
        """
        if self.planning_temporal_stack <= 0:
            return
        col_feats, spatial_shape, scale_start_index = feature_maps
        # Detach + move to CPU; the next forward will move back to device.
        col_cpu = col_feats.detach().to("cpu")
        ss_cpu = spatial_shape.detach().to("cpu")
        ssi_cpu = scale_start_index.detach().to("cpu")
        proj_cpu = metas["projection_mat"].detach().to("cpu")
        T_global = col_feats.new_tensor(
            np.stack([m["T_global"] for m in metas["img_metas"]])
        ).detach().to("cpu")

        if self._temp_col_feats is None:
            self._temp_col_feats = []
            self._temp_spatial_shape = []
            self._temp_scale_start_index = []
            self._temp_projection_mat = []
            self._temp_T_global = []

        # Insert at front (most-recent first), trim to stack length.
        self._temp_col_feats.insert(0, col_cpu)
        self._temp_spatial_shape.insert(0, ss_cpu)
        self._temp_scale_start_index.insert(0, ssi_cpu)
        self._temp_projection_mat.insert(0, proj_cpu)
        self._temp_T_global.insert(0, T_global)
        N = self.planning_temporal_stack
        self._temp_col_feats = self._temp_col_feats[:N]
        self._temp_spatial_shape = self._temp_spatial_shape[:N]
        self._temp_scale_start_index = self._temp_scale_start_index[:N]
        self._temp_projection_mat = self._temp_projection_mat[:N]
        self._temp_T_global = self._temp_T_global[:N]

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
        # Predict current-frame ego_status from history (and optionally visual
        # features), to be used in place of metas['ego_status'] downstream.
        # The true ego_status is still consumed for the auxiliary L1 loss and
        # for populating the history queue (see cache_planning).
        if self.use_predicted_ego_status:
            hist = self.instance_queue.get_ego_status_history(
                K=self.ego_state_estimator.history_K,
                batch_size=bs,
                mask=mask,
                device=ego_feature.device,
                dtype=ego_feature.dtype,
            )
            self._predicted_ego_status = self.ego_state_estimator(
                hist,
                ego_feature=ego_feature_raw[:, 0]
                if self.ego_state_estimator.variant == 'full' else None,
            )
        else:
            self._predicted_ego_status = None
        ego_feature = self._project_instance_feature(ego_feature)
        ego_anchor_embed = self._project_anchor_embed(anchor_encoder(ego_anchor))
        temp_instance_feature = self._project_instance_feature(temp_instance_feature)
        temp_anchor_embed = self._project_anchor_embed(anchor_encoder(temp_anchor))
        if self.ego_only_planning:
            # Drop agent temporal history; keep only the ego entry (queue.get
            # concatenates ego at the end, line 106-107 of instance_queue.py).
            temp_instance_feature = temp_instance_feature[:, -1:]
            temp_anchor_embed = temp_anchor_embed[:, -1:]
            temp_mask = temp_mask[:, -1:]
        temp_instance_feature = temp_instance_feature.flatten(0, 1)
        temp_anchor_embed = temp_anchor_embed.flatten(0, 1)
        temp_mask = temp_mask.flatten(0, 1)
        dim = self.embed_dims

        # =========== mode anchor init ===========
        motion_anchor = self.get_motion_anchor(det_classification, det_anchors)
        plan_anchor = self._get_initial_plan_anchor(bs, metas)

        # =========== mode query init ===========
        # When motion_target_in_agent_frame, motion_anchor is in agent frame;
        # rotate to lidar before sineembed so the positional code matches the
        # baseline distribution.
        motion_anchor_for_query = (
            self._agent2lidar(motion_anchor, det_anchors)
            if self.motion_target_in_agent_frame
            else motion_anchor
        )
        motion_mode_query = self.motion_anchor_encoder(
            gen_sineembed_for_position(
                motion_anchor_for_query[..., -1, :], hidden_dim=self.embed_dims
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
        # ego_status injection (variants 2/3): broadcast a per-batch encoded
        # vector to all plan modes. Same encoded vector is re-applied at every
        # per-stage rebuild after refine, so ego state stays in scope across
        # decoder layers.
        if self.plan_ego_status_encode_enable:
            es_in = self._effective_ego_status(metas)[:, self.plan_ego_status_indices].to(
                plan_mode_query.dtype
            )
            ego_status_embed = self.plan_ego_status_encoder(es_in)  # (bs, D)
            if self.plan_mode_time_queries:
                plan_mode_query = (
                    plan_mode_query + ego_status_embed[:, None, :]
                )
            else:
                plan_mode_query = (
                    plan_mode_query + ego_status_embed[:, None, :]
                )
        else:
            ego_status_embed = None

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
        planning_magnitude = []
        planning_da_logits = []
        planning_conflict_logits = []
        _refine_count = 0
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

        if self.ego_only_planning:
            # Strip the agent slots before the decoder loop. With
            # `skip_perception_kv='both'` and `with_conflict_head=False`, the
            # agents are already informationally isolated from ego (no cross-
            # token op writes them into ego); slicing them out just saves the
            # dead-end forward through temp_gnn/deformable/refine.
            instance_feature = instance_feature[:, num_anchor:]
            anchor_embed = anchor_embed[:, num_anchor:]
            instance_feature_selected = instance_feature_selected[:, -1:]
            anchor_embed_selected = anchor_embed_selected[:, -1:]
            motion_anchor = motion_anchor[:, :0]
            motion_mode_query = motion_mode_query[:, :0]
            if motion_endpoint_anchor_all is not None:
                motion_endpoint_anchor_all = motion_endpoint_anchor_all[:, :0]
            if motion_endpoint_anchor is not None:
                motion_endpoint_anchor = motion_endpoint_anchor[:, :0]
            num_anchor = 0
        # Temporal image-feature stacking for planning-deformable calls only.
        # Builds an extended (feature_maps, projection_mat) once; each
        # planning deformable call routes through self.planning_temporal_layers
        # with this stacked input.
        if self.planning_temporal_stack > 0 and self.planning_deformable:
            # If InstanceQueue just reset (eval-vs-train batch swap, scene
            # boundary at the start of a run), drop the temporal cache too —
            # otherwise we'd splice features from the previous run.
            if (
                self.instance_queue is not None
                and self.instance_queue.metas is None
            ):
                self._temp_col_feats = None
                self._temp_spatial_shape = None
                self._temp_scale_start_index = None
                self._temp_projection_mat = None
                self._temp_T_global = None
            feature_maps_p, projection_mat_p = self._build_temporal_planning_inputs(
                feature_maps, metas
            )
            metas_p = dict(metas)
            metas_p["projection_mat"] = projection_mat_p
            # image_wh is also per-cam; replicate it to match the extended cam
            # axis. All past frames share the same image dimensions.
            if metas.get("image_wh") is not None:
                metas_p["image_wh"] = metas["image_wh"].repeat(
                    1, self.planning_temporal_stack + 1, 1
                )
        else:
            feature_maps_p = feature_maps
            metas_p = metas
        # Precompute conflict-head image-feature samples once (shared across refines).
        # Path: zero queries + det_anchor positions → deformable → image features at
        # det BEV cells. Tests whether the conflict head needs detection's learned
        # semantic abstraction (`agent_token`) or raw image content at the agent's
        # spatial location (`image_at_det`).
        if self.with_conflict_head and self.conflict_input.startswith('image_at_'):
            if self.conflict_input == 'image_at_det':
                sampler_anchors = det_anchors
                # No per-mode partition; flat pool. K = num_det_anchor.
                self._conflict_per_mode_K = None
            elif self.conflict_input == 'image_at_plan':
                # Plan-trajectory waypoints in lidar frame as the spatial query.
                # plan_anchor: (num_cmd, ego_fut_mode, ego_fut_ts, 2). Flatten
                # cmd × mode → M; cumsum over time gives BEV positions per
                # waypoint in lidar frame.
                plan_xy = self.plan_anchor.detach()  # (cmd, mode, T, 2)
                M_total = plan_xy.shape[0] * plan_xy.shape[1]
                T = plan_xy.shape[2]
                plan_xy = plan_xy.reshape(M_total, T, 2)
                plan_xy = plan_xy.cumsum(dim=-2)  # waypoint absolute positions
                # Build full 11-dim anchors: [X,Y,Z=0, log_W=0, log_L=0, log_H=0,
                # SIN_YAW=0, COS_YAW=1, VX=0, VY=0, VZ=0]
                K = M_total * T
                anc = plan_xy.new_zeros(K, 11)
                anc[:, 0:2] = plan_xy.reshape(K, 2)
                anc[:, 7] = 1.0  # cos(yaw=0)
                bs_local = det_anchors.shape[0]
                sampler_anchors = anc.unsqueeze(0).expand(bs_local, -1, -1).contiguous().to(
                    det_anchors.device, dtype=det_anchors.dtype
                )
                self._conflict_per_mode_K = T  # K samples per plan mode (= waypoints)
            elif self.conflict_input == 'image_at_scene_query':
                # Run the scene-query decoder (image-feature-refined K BEV anchors).
                # Cache refined anchors + features for downstream conflict aggregation
                # and (optional) aux loss.
                sqd_out = self.scene_query_decoder(feature_maps, metas)
                sampler_anchors = sqd_out['anchors']  # (B, K, 11)
                self._scene_query_features = sqd_out['features']  # (B, K, D)
                self._scene_query_anchors = sampler_anchors
                # Global K queries: per-mode-restricted aggregation reuses the
                # diagonal mode read with K = num_queries; conf_logits shape
                # will be (B, K, M_pred). To keep the per-mode-restricted path
                # working, we set _conflict_per_mode_K=K and treat each mode's
                # logit as reading from all K queries — _loss_planning_conflict
                # routes scene_query through a special path that aggregates
                # per-mode without diagonal restriction (queries are global).
                self._conflict_per_mode_K = sampler_anchors.shape[1]
            elif self.conflict_input == 'image_at_init_topk':
                # Per-(scene, plan-mode) top-K nearest fixed init anchors.
                # plan_anchor cumulative XY → (M_total, T, 2). For each plan
                # mode m and scene b, distance(anchor_n) = min_t ||a_n.xy - traj_m_t||,
                # then top-K smallest distances → K anchors per (b, m).
                init_anchors = self.conflict_init_anchors  # (N, 11)
                N_init = init_anchors.shape[0]
                K_topk = min(self.conflict_init_topk, N_init)
                plan_xy = self.plan_anchor.detach()  # (cmd, mode, T, 2)
                M_total = plan_xy.shape[0] * plan_xy.shape[1]
                plan_xy = plan_xy.reshape(M_total, -1, 2).cumsum(dim=-2)  # (M, T, 2)
                anc_xy = init_anchors[:, 0:2]  # (N, 2)
                # diff: (M, T, N, 2) → d2: (M, T, N) → min over T: (M, N)
                d2 = ((plan_xy.to(anc_xy.device).unsqueeze(2) - anc_xy.unsqueeze(0).unsqueeze(0)) ** 2).sum(-1)
                d2_min = d2.min(dim=1).values  # (M, N)
                topk_idx = d2_min.topk(K_topk, dim=-1, largest=False).indices  # (M, K)
                topk_anchors = init_anchors[topk_idx]  # (M, K, 11)
                topk_anchors = topk_anchors.reshape(M_total * K_topk, 11)
                bs_local = det_anchors.shape[0]
                sampler_anchors = topk_anchors.unsqueeze(0).expand(bs_local, -1, -1).contiguous().to(
                    det_anchors.device, dtype=det_anchors.dtype
                )
                self._conflict_per_mode_K = K_topk
            elif self.conflict_input == 'image_at_init_topk_and_plan':
                # B1.8: union of B1.6 (init_topk) and B1.7 (plan waypoints).
                # Per mode K = K_topk + T. Anchors laid out as
                # [topk_0..topk_{K-1}, plan_t0..plan_{T-1}] for each mode.
                init_anchors = self.conflict_init_anchors  # (N, 11)
                N_init = init_anchors.shape[0]
                K_topk = min(self.conflict_init_topk, N_init)
                plan_xy_full = self.plan_anchor.detach()  # (cmd, mode, T, 2)
                M_total = plan_xy_full.shape[0] * plan_xy_full.shape[1]
                T = plan_xy_full.shape[2]
                plan_xy = plan_xy_full.reshape(M_total, T, 2).cumsum(dim=-2)
                anc_xy = init_anchors[:, 0:2]
                d2 = ((plan_xy.to(anc_xy.device).unsqueeze(2) - anc_xy.unsqueeze(0).unsqueeze(0)) ** 2).sum(-1)
                d2_min = d2.min(dim=1).values
                topk_idx = d2_min.topk(K_topk, dim=-1, largest=False).indices  # (M, K)
                topk_anchors = init_anchors[topk_idx]  # (M, K, 11)
                # Plan-waypoint anchors per mode (M, T, 11).
                plan_anc = plan_xy.new_zeros(M_total, T, 11)
                plan_anc[..., 0:2] = plan_xy
                plan_anc[..., 7] = 1.0  # cos(yaw=0)
                plan_anc = plan_anc.to(topk_anchors.device, dtype=topk_anchors.dtype)
                # Concatenate per-mode keypoints: [topk(K), plan(T)] → (M, K+T, 11)
                K_total_per_mode = K_topk + T
                combined = torch.cat([topk_anchors, plan_anc], dim=1)
                combined = combined.reshape(M_total * K_total_per_mode, 11)
                bs_local = det_anchors.shape[0]
                sampler_anchors = combined.unsqueeze(0).expand(bs_local, -1, -1).contiguous().to(
                    det_anchors.device, dtype=det_anchors.dtype
                )
                self._conflict_per_mode_K = K_total_per_mode
            elif self.conflict_input == 'image_at_ego_grid':
                # B1.9: fixed BEV grid centered on ego (lidar frame). 25 points
                # by default (5 forward × 5 lateral). Shared across plan modes;
                # loss uses global-pool path (total = K, K queries per mode).
                ego_grid = self.conflict_ego_grid_anchors  # (K, 11)
                K_grid = ego_grid.shape[0]
                bs_local = det_anchors.shape[0]
                sampler_anchors = ego_grid.unsqueeze(0).expand(bs_local, -1, -1).contiguous().to(
                    det_anchors.device, dtype=det_anchors.dtype
                )
                self._conflict_per_mode_K = K_grid  # global pool: total == K
            else:
                raise NotImplementedError(self.conflict_input)
            if self.conflict_input == 'image_at_scene_query':
                # Decoder already produced features via internal deformable
                # attention; skip the standalone conflict_image_sampler pass.
                conflict_image_features = self._scene_query_features
            else:
                sampler_query = sampler_anchors.new_zeros(
                    sampler_anchors.shape[0], sampler_anchors.shape[1], self.embed_dims
                )
                sampler_anchor_embed = anchor_encoder(sampler_anchors)
                conflict_image_features = self.conflict_image_sampler(
                    sampler_query,
                    sampler_anchors,
                    sampler_anchor_embed,
                    feature_maps,
                    metas,
                )
        else:
            conflict_image_features = None
            self._conflict_per_mode_K = None
            self._scene_query_features = None
            self._scene_query_anchors = None
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
                if self.ego_only_planning:
                    # No agents present; skip motion-deformable to avoid
                    # invoking the DAF kernel with zero-length queries.
                    agent_feature = instance_feature[:, :num_anchor]
                elif self.motion_deformable_multimode:
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
                    # Pick layer + temporal feature_maps/metas for planning
                    # deformable calls. When planning_temporal_stack > 0 we
                    # use a separate layer with extended num_cams and the
                    # past-frame-stacked feature maps built above.
                    if self.planning_temporal_layers is not None:
                        plan_layer = self.planning_temporal_layers[
                            _deformable_stage_idx - 1
                        ]
                        plan_fmaps = feature_maps_p
                        plan_metas = metas_p
                    else:
                        plan_layer = self.layers[i]
                        plan_fmaps = feature_maps
                        plan_metas = metas
                    multi_wp = (
                        self.planning_deformable_waypoints is not None
                        and not self.plan_mode_time_queries
                    )
                    if self.plan_mode_time_queries:
                        # Per-(mode, ts) anchor box, flattened along (mode, ts).
                        T = self.ego_fut_ts
                        plan_anchor_box_mt = self._build_planning_anchor_boxes_multi(
                            plan_anchor.detach(), ego_anchor, list(range(T)),
                        )  # (bs, M_total, T, 11)
                        plan_anchor_box = plan_anchor_box_mt.flatten(1, 2)  # (bs, M_total*T, 11)
                    elif multi_wp:
                        # Per-(mode, K) anchor boxes flattened to (bs, num_modes*K, 11).
                        # Used by both the non-instfeat else-branch (existing) and
                        # the _use_instfeat branch (K-pool after deformable).
                        plan_anchor_box_mk = self._build_planning_anchor_boxes_multi(
                            plan_anchor.detach(), ego_anchor,
                            self.planning_deformable_waypoints,
                        )  # (bs, num_modes, K, 11)
                        plan_anchor_box = plan_anchor_box_mk.flatten(1, 2)
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
                            if multi_wp:
                                # plan_mode_query is (bs, num_modes, D); expand to
                                # (bs, num_modes*K, D) to match per-(mode, K) anchors.
                                K = len(self.planning_deformable_waypoints)
                                num_modes_real = plan_mode_query.shape[1]
                                pmq_exp = (
                                    plan_mode_query.unsqueeze(2)
                                    .expand(-1, -1, K, -1)
                                    .reshape(bs, num_modes_real * K, self.embed_dims)
                                )
                                daf_query = pmq_exp + ego_feat_exp
                            else:
                                daf_query = plan_mode_query + ego_feat_exp
                            attended_plan = plan_layer(
                                self._project_deformable_feature(daf_query),
                                plan_anchor_box,
                                plan_anchor_embed,
                                plan_fmaps,
                                plan_metas,
                            )
                            attended_plan = self._project_deformable_output(
                                attended_plan
                            )
                            if multi_wp:
                                K = len(self.planning_deformable_waypoints)
                                num_modes_real = attended_plan.shape[1] // K
                                attended_plan = attended_plan.reshape(
                                    bs, num_modes_real, K, self.embed_dims
                                ).mean(dim=2)
                            plan_mode_query = attended_plan
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
                            attended_plan = plan_layer(
                                self._project_deformable_feature(ego_feat_exp_q),
                                plan_anchor_box,
                                plan_anchor_embed,
                                plan_fmaps,
                                plan_metas,
                            )
                            attended_plan = self._project_deformable_output(
                                attended_plan
                            )  # (bs, num_plan_queries, embed_dims)
                            if multi_wp:
                                # K-pool back to per-mode so classification-weighted
                                # aggregation matches plan_weights' (bs, num_modes) shape.
                                K = len(self.planning_deformable_waypoints)
                                num_modes_real = num_plan_queries // K
                                attended_plan = attended_plan.reshape(
                                    bs, num_modes_real, K, self.embed_dims
                                ).mean(dim=2)
                                num_plan_queries = num_modes_real
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
                            attended = plan_layer(
                                self._project_deformable_feature(query_exp),
                                boxes_flat,
                                embed_flat,
                                plan_fmaps, plan_metas,
                            )
                            plan_mode_query = self._project_deformable_output(
                                attended
                            ).reshape(
                                bs, num_mode, K, self.embed_dims
                            ).mean(dim=2)
                        else:
                            plan_mode_query = self._project_deformable_output(
                                plan_layer(
                                self._project_deformable_feature(plan_mode_query),
                                plan_anchor_box,
                                plan_anchor_embed,
                                plan_fmaps,
                                plan_metas,
                            ))
                        if self.plan_time_attn and self.plan_mode_time_queries:
                            # Per-mode time-axis self-attention to restore
                            # intra-mode temporal coupling. Reshape
                            # (bs, M*T, D) -> (bs*M, T, D), add learnable
                            # time pos embed, MHA self-attn with residual + LN.
                            M_total = self.num_driving_cmds * self.ego_fut_mode
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
                if not self.with_conflict_head:
                    agent_features_for_refine = None
                elif self.conflict_input.startswith('image_at_'):
                    agent_features_for_refine = conflict_image_features
                else:
                    agent_features_for_refine = instance_feature[:, :num_anchor]
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
                # Per-stage scalar magnitude head (variants 1a / 3). Reads the
                # refined plan_query for this stage and emits one scalar per
                # plan mode; supervised in loss_planning by smooth-L1 vs the
                # cmd-indexed best-mode's ||gt_end||.
                if self.plan_magnitude_head_enable:
                    mag_pred = self.plan_magnitude_branches[_refine_count](
                        plan_query.squeeze(1)
                    ).squeeze(-1)
                    planning_magnitude.append(mag_pred)
                _refine_count += 1
                # Update mode anchor queries for the next decoder iteration.
                # cumsum converts delta trajectories to absolute endpoints.
                motion_anchor_upd = motion_reg.detach().cumsum(dim=-2)
                plan_anchor_upd = plan_reg.detach().squeeze(1).cumsum(dim=-2)
                # Sineembed expects lidar-frame endpoint coordinates (matches
                # the static-anchor init at line ~1378). Rotate the agent-frame
                # cumsum into lidar before encoding so the positional code's
                # distribution stays the same as the baseline path.
                motion_anchor_upd_for_query = (
                    self._agent2lidar(motion_anchor_upd, det_anchors)
                    if self.motion_target_in_agent_frame
                    else motion_anchor_upd
                )
                motion_mode_query = self.motion_anchor_encoder(
                    gen_sineembed_for_position(
                        motion_anchor_upd_for_query[..., -1, :], hidden_dim=self.embed_dims
                    )
                )
                if self.ego_only_planning:
                    # No agents — keep motion_endpoint buffers empty.
                    pass
                elif self.motion_deformable_multimode:
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
                if ego_status_embed is not None:
                    plan_mode_query = (
                        plan_mode_query + ego_status_embed[:, None, :]
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
        # Append the true current-frame ego_status to the history queue used
        # by the ego-state estimator on the next forward. We cache the TRUE
        # value (not the prediction) so the queue remains a faithful
        # operational history; the supervision target also reads from CAN.
        if self.use_predicted_ego_status:
            self.instance_queue.cache_ego_status(
                metas['ego_status'],
                history_K=self.ego_state_estimator.history_K,
            )
        # Push current frame's image features onto the planning temporal cache
        # for the next forward.
        self._cache_planning_temporal(feature_maps, metas)

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
        if planning_magnitude:
            planning_output["magnitude"] = planning_magnitude

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
                if refine_module.plan_reg_branch is None:
                    raise RuntimeError(
                        "DN-plan groups are not supported when "
                        "plan_per_bucket_reg=True (no shared plan_reg_branch). "
                        "Set num_dn_plan_groups=0."
                    )
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
        motion_loss_cache,
        det_output=None,
    ):
        loss = {}
        motion_loss = self.loss_motion(
            motion_model_outs, data, motion_loss_cache, det_output=det_output
        )
        loss.update(motion_loss)
        planning_loss = self.loss_planning(
            planning_model_outs, data, motion_loss_cache,
            motion_model_outs=motion_model_outs, det_output=det_output,
        )
        loss.update(planning_loss)
        if (
            self.use_predicted_ego_status
            and self._predicted_ego_status is not None
            and 'ego_status' in data
        ):
            target = data['ego_status'].to(self._predicted_ego_status.dtype)
            loss['ego_state_aux_l1'] = self.ego_state_estimator.loss(
                self._predicted_ego_status, target
            )
        return loss

    @force_fp32(apply_to=("model_outs"))
    def loss_motion(self, model_outs, data, motion_loss_cache, det_output=None):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        output = {}
        if self.ego_only_planning:
            # Motion buffers are empty (num_anchor=0). Skip the sampler (which
            # would index into the empty pred buffer with detection's matched
            # indices and crash) but emit a zero-valued keepalive loss that
            # still references the motion-only branch outputs, so motion params
            # show up as "ready" in DDP without find_unused_parameters=True.
            keepalive = reg_preds[0].new_zeros(())
            for cls, reg in zip(cls_scores, reg_preds):
                keepalive = keepalive + cls.sum() * 0.0 + reg.sum() * 0.0
            output['motion_loss_keepalive'] = keepalive
            return output
        # For agent-frame regression we need each detection token's predicted
        # yaw to rotate `reg_target` (lidar-frame deltas from the dataset) into
        # the same agent frame the model is predicting in.
        det_anchors_for_rot = None
        if self.motion_target_in_agent_frame:
            assert det_output is not None, (
                "loss_motion requires det_output when motion_target_in_agent_frame=True"
            )
            det_anchors_for_rot = det_output["prediction"][-1].detach()
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

            if self.motion_target_in_agent_frame:
                # reg_target is (bs, num_anchor, fut_ts, 2) lidar-frame deltas
                # from the matched GT future. Rotate to agent frame using the
                # predicted yaw at each detection token. Unmatched positions
                # are zero — rotating zeros yields zeros, no harm.
                reg_target = self._rotate_trajs(
                    reg_target, det_anchors_for_rot, agent_to_lidar=False
                )

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
            assert not self.motion_target_in_agent_frame, (
                "DN motion path is not adapted for agent-frame regression yet. "
                "Set num_dn_pred_groups=0 or motion_target_in_agent_frame=False."
            )
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
    def loss_planning(
        self, model_outs, data, motion_loss_cache=None,
        motion_model_outs=None, det_output=None,
    ):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        status_preds = model_outs["status"]
        da_logits_list = model_outs.get("da_logits", [])
        conf_logits_list = model_outs.get("conflict_logits", [])
        magnitude_preds = model_outs.get("magnitude", [])
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

            if decoder_idx < len(magnitude_preds):
                # Smooth-L1 between mag_pred for the cmd-indexed best-mode and
                # ||gt_end||. cls_target (from planning_sampler) is the
                # winning-mode index within the 6-mode cmd slice, so we gather
                # mag for that mode only — non-matched modes are not penalized.
                mag_all = magnitude_preds[decoder_idx]  # (bs, 3*M)
                bs_m = mag_all.shape[0]
                M = self.ego_fut_mode
                cmd_idx_m = data['gt_ego_fut_cmd'].argmax(dim=-1)  # (bs,)
                bs_arr = torch.arange(bs_m, device=mag_all.device)
                mag_cmd = mag_all.reshape(bs_m, 3, M)[bs_arr, cmd_idx_m]  # (bs, M)
                best_mode = cls_target.reshape(bs_m).long()
                mag_best = mag_cmd[bs_arr, best_mode]  # (bs,)
                gt_cum = data['gt_ego_fut_trajs'].cumsum(dim=-2)  # (bs, T, 2)
                gt_end_norm = torch.linalg.norm(gt_cum[:, -1, :], dim=-1)  # (bs,)
                mag_loss = F.smooth_l1_loss(
                    mag_best, gt_end_norm.to(mag_best.dtype), beta=1.0
                ) * self.plan_magnitude_loss_weight
                output[f"planning_loss_mag_{decoder_idx}"] = mag_loss

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
                conf_ret = self._loss_planning_conflict(
                    conf_logits_list[decoder_idx], reg, data, motion_loss_cache
                )
                if conf_ret is not None:
                    conf_loss, conf_diag = conf_ret
                    if conf_loss is not None:
                        output[f"planning_loss_conf_{decoder_idx}"] = (
                            conf_loss * self.conflict_loss_weight
                        )
                    # Log classifier diagnostics from the last decoder stage only,
                    # to keep the log compact.
                    if (
                        conf_diag is not None
                        and decoder_idx == len(conf_logits_list) - 1
                    ):
                        for k, v in conf_diag.items():
                            output[f"plan_conf_{k}"] = v

            if (
                self.scene_query_aux_loss_enable
                and decoder_idx == len(reg_preds) - 1
            ):
                aux_ret = self._loss_scene_query_aux(data)
                if aux_ret is not None:
                    aux_loss, aux_diag = aux_ret
                    if aux_loss is not None:
                        output['planning_loss_scene_query_aux'] = aux_loss
                    if aux_diag is not None:
                        for k, v in aux_diag.items():
                            output[k] = v

            if self.plan_softcost_collision_enable:
                softcost_col = self._loss_planning_softcost_collision(reg, data)
                if softcost_col is not None:
                    output[f"planning_loss_softcost_col_{decoder_idx}"] = (
                        softcost_col * self.plan_softcost_collision_weight
                    )

            if (
                self.plan_distill_rescore_enable
                and motion_model_outs is not None
                and det_output is not None
                and decoder_idx == len(reg_preds) - 1
            ):
                # Distill only at the last decoder stage. Inference rescore
                # consumes the last stage's `plan_cls` exclusively, so earlier
                # stages don't need the constraint baked in. Skipping them
                # cuts `compute_rescore_collision_mask` calls 6× and recovers
                # the per-iter cost that timed out Killarney 3366621.
                distill_ret = self._loss_planning_distill_rescore(
                    cls, reg, data, motion_model_outs, det_output,
                )
                if distill_ret is not None:
                    distill_loss, distill_diag = distill_ret
                    output[f"planning_loss_distill_rescore_{decoder_idx}"] = (
                        distill_loss * self.plan_distill_rescore_weight
                    )
                    if distill_diag is not None:
                        for k, v in distill_diag.items():
                            output[f"plan_distill_{k}"] = v

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

    def _resolve_conflict_pos_weight(self, target, weight):
        """Resolve pos_weight: 'auto' calibrates from running pos_rate over
        ``self._auto_pw_calib_iters`` iterations, then freezes. Numeric
        values pass through unchanged. Called every iter of training; only
        accumulates while still calibrating.
        """
        cfg = self.conflict_pos_weight
        if not isinstance(cfg, str):
            return float(cfg)
        if cfg != 'auto':
            return float(cfg)
        if self._auto_pw_value is not None:
            return self._auto_pw_value
        with torch.no_grad():
            mask = weight.bool()
            if mask.any():
                t = target[mask].float()
                self._auto_pw_pos += t.sum().item()
                self._auto_pw_total += t.numel()
            self._auto_pw_iter += 1
            if self._auto_pw_iter >= self._auto_pw_calib_iters and self._auto_pw_total > 0:
                p = max(self._auto_pw_pos / self._auto_pw_total, 1e-6)
                self._auto_pw_value = (1.0 - p) / p
                print(
                    f"[planaux_conf auto pos_weight] frozen after "
                    f"{self._auto_pw_iter} iters: pos_rate={p:.6f}, "
                    f"pos_weight={self._auto_pw_value:.2f}",
                    flush=True,
                )
                return self._auto_pw_value
        # Default during calibration window: use a moderate prior so loss
        # doesn't collapse to all-negative in early iters. 50.0 matches the
        # mid of the expected (1 - p) / p range for p in [1%, 2%].
        return 50.0

    def _fill_evalmatch_labels(self, target, weight, reg, data, motion_loss_cache):
        """Replicate ``planning_eval.py``'s ``obj_box_col`` per-(mode, anchor)
        and write into ``target`` / ``weight`` for the BCE loss.

        Geometry matches the eval verbatim:
          - ego dims H=4.084, W=1.85, h=1.56 (planning_eval.py:61-62, 81)
          - 0.5m forward offset along yaw (planning_eval.py:22-23)
          - yaw via _eval_get_yaw (port of planning_eval.get_yaw)
          - GT agent box: GT WLH from gt_bboxes_3d, t=0 yaw + tangent fallback
          - polygon-on-polygon SAT intersection per (m, a, t)
          - any(t) reduction; box_coll AND NOT gt_box_coll mask (line 103)

        Restricted to the t=0-detected agent set (Hungarian-matched).
        Post-t=0-appearing agents are inaccessible to per-anchor labels —
        documented gap, same blind spot as rescore().
        """
        gt_boxes = data['gt_bboxes_3d']
        gt_traj = data['gt_agent_fut_trajs']
        gt_traj_mask = data.get('gt_agent_fut_masks')
        gt_ego_traj = data.get('gt_ego_fut_trajs')
        gt_ego_mask = data.get('gt_ego_fut_masks')
        if gt_ego_traj is None:
            return

        bs, num_anchor, M = target.shape
        device = target.device
        T_ego = reg.shape[-2]

        # Predicted ego absolute XY per (m, t) — detached to keep label
        # generation off the gradient graph.
        ego_xy_full = reg.detach().squeeze(1).cumsum(dim=-2)  # (bs, M, T_ego, 2)

        # Eval ego dims (planning_eval.py:61-62, 81). Eval assigns H to W slot
        # (length, 4.084) and W to L slot (width, 1.85); the local-x extent of
        # the box is the vehicle length.
        ego_box_W = ego_xy_full.new_tensor(4.084)
        ego_box_L = ego_xy_full.new_tensor(1.85)
        forward_offset = 0.5

        for b in range(bs):
            pred_idx, target_idx = motion_loss_cache['indices'][b]
            if pred_idx is None or len(pred_idx) == 0:
                continue
            boxes_b = gt_boxes[b].to(device).float()
            trajs_b = gt_traj[b].to(device).float()
            if trajs_b.shape[0] == 0:
                continue

            ego_traj_b = gt_ego_traj[b].to(device).float()  # (T, 2) deltas
            T_gt_ego = ego_traj_b.shape[0]
            T = min(trajs_b.shape[1], T_ego, T_gt_ego)
            if T == 0:
                continue

            # === Predicted ego boxes per (m, t) with eval geometry ===
            pred_ego_xy = ego_xy_full[b, :, :T, :]  # (M, T, 2)
            pred_ego_yaw = _eval_get_yaw(pred_ego_xy)  # (M, T)
            # Apply 0.5m forward offset to box CENTER along yaw.
            offset_x = forward_offset * torch.cos(pred_ego_yaw)
            offset_y = forward_offset * torch.sin(pred_ego_yaw)
            pred_cx = pred_ego_xy[..., 0] + offset_x  # (M, T)
            pred_cy = pred_ego_xy[..., 1] + offset_y
            pred_corners = _make_rect_corners_topdown(
                pred_cx, pred_cy,
                ego_box_W.expand_as(pred_cx),
                ego_box_L.expand_as(pred_cx),
                pred_ego_yaw,
            )  # (M, T, 4, 2)

            # === GT ego boxes per t (for the AND NOT gt_box_coll mask) ===
            gt_ego_xy = ego_traj_b[:T].cumsum(dim=0)  # (T, 2)
            gt_ego_yaw = _eval_get_yaw(gt_ego_xy)  # (T,)
            gt_ego_cx = gt_ego_xy[..., 0] + forward_offset * torch.cos(gt_ego_yaw)
            gt_ego_cy = gt_ego_xy[..., 1] + forward_offset * torch.sin(gt_ego_yaw)
            gt_ego_corners = _make_rect_corners_topdown(
                gt_ego_cx, gt_ego_cy,
                ego_box_W.expand_as(gt_ego_cx),
                ego_box_L.expand_as(gt_ego_cx),
                gt_ego_yaw,
            )  # (T, 4, 2)

            # === GT agent boxes per (a, t) for matched anchors ===
            n_pos = len(target_idx)
            agent_xy0 = boxes_b[target_idx, :2]  # (n_pos, 2)
            agent_traj = trajs_b[target_idx, :T]  # (n_pos, T, 2)
            agent_pos = agent_xy0.unsqueeze(1) + agent_traj.cumsum(dim=1)  # (n_pos, T, 2)

            agent_W = boxes_b[target_idx, W]  # (n_pos,) — index W=3
            agent_L = boxes_b[target_idx, L]  # (n_pos,) — index L=4
            agent_yaw_t0 = boxes_b[target_idx, YAW]  # (n_pos,) — decoded yaw
            agent_yaw_t = _agent_get_yaw(
                agent_pos, agent_yaw_t0
            )  # (n_pos, T)

            agent_corners = _make_rect_corners_topdown(
                agent_pos[..., 0], agent_pos[..., 1],
                agent_W.unsqueeze(-1).expand(-1, T),
                agent_L.unsqueeze(-1).expand(-1, T),
                agent_yaw_t,
            )  # (n_pos, T, 4, 2)

            # === Intersection per (m, a, t) and per (a, t) for GT ego ===
            # Broadcast: pred_corners (M, 1, T, 4, 2) vs agent_corners (1, n_pos, T, 4, 2)
            pred_b = pred_corners.unsqueeze(1).expand(M, n_pos, T, 4, 2)
            agent_b = agent_corners.unsqueeze(0).expand(M, n_pos, T, 4, 2)
            pred_coll = _rect_intersects_sat(pred_b, agent_b)  # (M, n_pos, T)

            # GT ego vs each agent per t — broadcast (1, n_pos, T, 4, 2)
            gt_ego_b = gt_ego_corners.unsqueeze(0).expand(n_pos, T, 4, 2)
            gt_coll = _rect_intersects_sat(gt_ego_b, agent_corners)  # (n_pos, T)

            # Validity mask: agent's GT future visible at t.
            if gt_traj_mask is not None:
                tmask = gt_traj_mask[b].to(device)[target_idx, :T].bool()  # (n_pos, T)
                pred_coll = pred_coll & tmask.unsqueeze(0)
                gt_coll = gt_coll & tmask
            # Also gate by ego's GT future validity at t.
            if gt_ego_mask is not None:
                em = gt_ego_mask[b].to(device)[:T].bool()  # (T,)
                pred_coll = pred_coll & em.view(1, 1, T)
                gt_coll = gt_coll & em.view(1, T)

            # box_coll AND NOT gt_box_coll (planning_eval.py:103).
            attributable = pred_coll & ~gt_coll.unsqueeze(0)  # (M, n_pos, T)
            label_per_pair = attributable.any(dim=-1)  # (M, n_pos) per (mode, agent)

            # Write into target/weight at matched anchor indices, per mode.
            if isinstance(pred_idx, torch.Tensor):
                pidx = pred_idx.to(device).long()
            else:
                pidx = torch.as_tensor(pred_idx, device=device, dtype=torch.long)
            # target shape (bs, num_anchor, M); transpose label to (n_pos, M)
            target[b, pidx] = label_per_pair.t().to(target.dtype)
            weight[b, pidx] = 1.0

    def _loss_scene_query_aux(self, data):
        """Detection-like aux loss on scene queries against ego-interacting GT.

        Filter: GT agents whose absolute position comes within
        ``self.scene_query_aux_dist_thresh`` meters of ego at any timestep
        within ``self.scene_query_aux_time_steps`` future steps.

        Hungarian-match scene queries to filtered GT by L2 distance on (x, y);
        focal cls (1=matched, 0=unmatched) on a per-query head, plus L1 on
        (x, y) for matched queries.
        """
        from scipy.optimize import linear_sum_assignment
        if self.scene_query_cls_head is None:
            return None, None
        feats = self._scene_query_features
        anchors = self._scene_query_anchors
        if feats is None or anchors is None:
            return None, None
        gt_boxes = data.get('gt_bboxes_3d')
        gt_traj = data.get('gt_agent_fut_trajs')
        gt_traj_mask = data.get('gt_agent_fut_masks')
        gt_ego_traj = data.get('gt_ego_fut_trajs')
        if gt_boxes is None or gt_traj is None or gt_ego_traj is None:
            return None, None

        bs, K, D = feats.shape
        device = feats.device
        cls_logits = self.scene_query_cls_head(feats).squeeze(-1)  # (bs, K)
        target = cls_logits.new_zeros(bs, K)
        weight = cls_logits.new_ones(bs, K)  # all queries get cls signal
        box_target = anchors.new_zeros(bs, K, 2)
        box_weight = anchors.new_zeros(bs, K)

        thr = self.scene_query_aux_dist_thresh
        T_h = self.scene_query_aux_time_steps

        for b in range(bs):
            boxes_b = gt_boxes[b].to(device).float()
            trajs_b = gt_traj[b].to(device).float()
            ego_b = gt_ego_traj[b].to(device).float()
            n_agents = boxes_b.shape[0]
            if n_agents == 0 or trajs_b.shape[0] == 0 or ego_b.shape[0] == 0:
                continue
            T = min(T_h, trajs_b.shape[1], ego_b.shape[0])
            if T == 0:
                continue
            # Ego absolute pos at t in [0..T]
            ego_pos = torch.cat(
                [ego_b.new_zeros(1, 2), ego_b[:T].cumsum(dim=0)], dim=0
            )  # (T+1, 2)
            # Agent absolute pos at t in [0..T]
            agent_xy0 = boxes_b[:, :2]
            agent_cum = torch.cat(
                [trajs_b.new_zeros(n_agents, 1, 2), trajs_b[:, :T].cumsum(dim=1)],
                dim=1,
            )
            agent_pos = agent_xy0[:, None, :] + agent_cum  # (N, T+1, 2)
            d = torch.norm(agent_pos - ego_pos[None, :, :], dim=-1)  # (N, T+1)
            valid_t = torch.cat(
                [torch.ones(n_agents, 1, dtype=torch.bool, device=device),
                 (gt_traj_mask[b].to(device)[:, :T].bool() if gt_traj_mask is not None else torch.ones(n_agents, T, dtype=torch.bool, device=device))],
                dim=1,
            )
            d_masked = torch.where(valid_t, d, d.new_full((), float('inf')))
            keep = (d_masked < thr).any(dim=1)
            if not keep.any():
                continue
            kept_xy = agent_xy0[keep]  # (n_kept, 2)
            # Hungarian match scene query positions ↔ kept GT positions by L2
            q_xy = anchors[b, :, :2]  # (K, 2)
            cost = torch.cdist(q_xy, kept_xy)  # (K, n_kept)
            cost_np = cost.detach().cpu().numpy()
            q_idx, g_idx = linear_sum_assignment(cost_np)
            q_idx = torch.as_tensor(q_idx, device=device, dtype=torch.long)
            g_idx = torch.as_tensor(g_idx, device=device, dtype=torch.long)
            target[b, q_idx] = 1.0
            box_target[b, q_idx] = kept_xy[g_idx]
            box_weight[b, q_idx] = 1.0

        # Focal-style cls loss (use BCE with positive bias prior)
        valid = weight.sum() > 0
        if not valid:
            return None, None
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_logits, target, weight=weight, reduction='sum'
        ) / weight.sum().clamp_min(1.0)
        cls_loss = cls_loss * self.scene_query_aux_cls_weight

        if box_weight.sum() > 0:
            box_loss = F.l1_loss(
                anchors[..., :2] * box_weight.unsqueeze(-1),
                box_target * box_weight.unsqueeze(-1),
                reduction='sum',
            ) / (box_weight.sum() * 2).clamp_min(1.0)
            box_loss = box_loss * self.scene_query_aux_box_weight
        else:
            box_loss = cls_loss.new_zeros(())

        loss = (cls_loss + box_loss) * self.scene_query_aux_loss_weight
        with torch.no_grad():
            n_pos = (target > 0.5).float().sum()
            n_total = weight.sum()
        diag = dict(
            scene_query_aux_pos_rate=(n_pos / n_total.clamp_min(1.0)).detach(),
            scene_query_aux_cls=cls_loss.detach(),
            scene_query_aux_box=box_loss.detach(),
        )
        return loss, diag

    def _loss_planning_conflict_per_mode(self, conf_logits, reg, data):
        """Per-mode aggregation for det-free conflict samplers.

        Two layouts supported:
          - Per-mode pool (image_at_plan / image_at_init_topk):
            conf_logits.shape == (bs, M_anc * K, M_pred), where M_anc = M_pred.
            For each output mode m, smooth-max over the K samples that BELONG
            to mode m (anchor indices [m*K : (m+1)*K]) — diagonal mode read.
          - Global pool (image_at_scene_query):
            conf_logits.shape == (bs, K, M_pred), K queries shared across modes.
            For each mode m, smooth-max over all K queries.
        Per-mode label via evalmatch geometry, no Hungarian needed.
        """
        bs, total, M_pred = conf_logits.shape
        K = self._conflict_per_mode_K
        if K is None:
            return None, None
        device = conf_logits.device
        tau = self.conflict_smooth_max_tau
        if total == M_pred * K:
            # Per-mode pool: diagonal mode read
            logits_per_mode = conf_logits.reshape(bs, M_pred, K, M_pred)
            diag_logits = logits_per_mode.diagonal(dim1=1, dim2=3)  # (bs, K, M)
            per_mode_logit = (1.0 / tau) * torch.logsumexp(tau * diag_logits, dim=1)
        elif total == K:
            # Global pool: smooth-max over all K queries per mode
            per_mode_logit = (1.0 / tau) * torch.logsumexp(tau * conf_logits, dim=1)
        else:
            return None, None

        # Per-mode label via evalmatch geometry: for each m, did predicted ego
        # trajectory reg[m] collide with any visible GT agent at any timestep?
        gt_boxes = data.get('gt_bboxes_3d')
        gt_traj = data.get('gt_agent_fut_trajs')
        gt_traj_mask = data.get('gt_agent_fut_masks')
        gt_ego_traj = data.get('gt_ego_fut_trajs')
        gt_ego_mask = data.get('gt_ego_fut_masks')
        if gt_boxes is None or gt_traj is None or gt_ego_traj is None:
            return None, None

        T_ego = reg.shape[-2]
        ego_xy_full = reg.detach().squeeze(1).cumsum(dim=-2)  # (bs, M, T_ego, 2)
        ego_box_W = ego_xy_full.new_tensor(4.084)
        ego_box_L = ego_xy_full.new_tensor(1.85)
        forward_offset = 0.5

        target = per_mode_logit.new_zeros(bs, M_pred)
        weight = per_mode_logit.new_zeros(bs, M_pred)

        for b in range(bs):
            boxes_b = gt_boxes[b].to(device).float()
            trajs_b = gt_traj[b].to(device).float()
            ego_traj_b = gt_ego_traj[b].to(device).float()
            if trajs_b.shape[0] == 0 or ego_traj_b.shape[0] == 0:
                continue
            T = min(trajs_b.shape[1], T_ego, ego_traj_b.shape[0])
            if T == 0:
                continue
            n_agents = trajs_b.shape[0]

            # Predicted ego corners (M, T, 4, 2)
            pred_ego_xy = ego_xy_full[b, :, :T, :]
            pred_ego_yaw = _eval_get_yaw(pred_ego_xy)
            pred_cx = pred_ego_xy[..., 0] + forward_offset * torch.cos(pred_ego_yaw)
            pred_cy = pred_ego_xy[..., 1] + forward_offset * torch.sin(pred_ego_yaw)
            pred_corners = _make_rect_corners_topdown(
                pred_cx, pred_cy,
                ego_box_W.expand_as(pred_cx),
                ego_box_L.expand_as(pred_cx),
                pred_ego_yaw,
            )

            # GT ego corners (T, 4, 2)
            gt_ego_xy = ego_traj_b[:T].cumsum(dim=0)
            gt_ego_yaw = _eval_get_yaw(gt_ego_xy)
            gt_ego_cx = gt_ego_xy[..., 0] + forward_offset * torch.cos(gt_ego_yaw)
            gt_ego_cy = gt_ego_xy[..., 1] + forward_offset * torch.sin(gt_ego_yaw)
            gt_ego_corners = _make_rect_corners_topdown(
                gt_ego_cx, gt_ego_cy,
                ego_box_W.expand_as(gt_ego_cx),
                ego_box_L.expand_as(gt_ego_cx),
                gt_ego_yaw,
            )

            # GT agent corners (n_agents, T, 4, 2) — full agent set, no Hungarian
            agent_xy0 = boxes_b[:, :2]
            agent_traj = trajs_b[:, :T]
            agent_pos = agent_xy0.unsqueeze(1) + agent_traj.cumsum(dim=1)
            agent_W = boxes_b[:, W]
            agent_L = boxes_b[:, L]
            agent_yaw_t0 = boxes_b[:, YAW]
            agent_yaw_t = _agent_get_yaw(agent_pos, agent_yaw_t0)
            agent_corners = _make_rect_corners_topdown(
                agent_pos[..., 0], agent_pos[..., 1],
                agent_W.unsqueeze(-1).expand(-1, T),
                agent_L.unsqueeze(-1).expand(-1, T),
                agent_yaw_t,
            )

            # Pred ego (M, n_agents, T) collisions
            pred_b = pred_corners.unsqueeze(1).expand(M_pred, n_agents, T, 4, 2)
            agent_b = agent_corners.unsqueeze(0).expand(M_pred, n_agents, T, 4, 2)
            pred_coll = _rect_intersects_sat(pred_b, agent_b)
            gt_ego_b = gt_ego_corners.unsqueeze(0).expand(n_agents, T, 4, 2)
            gt_coll = _rect_intersects_sat(gt_ego_b, agent_corners)
            if gt_traj_mask is not None:
                tmask = gt_traj_mask[b].to(device)[:, :T].bool()
                pred_coll = pred_coll & tmask.unsqueeze(0)
                gt_coll = gt_coll & tmask
            if gt_ego_mask is not None:
                em = gt_ego_mask[b].to(device)[:T].bool()
                pred_coll = pred_coll & em.view(1, 1, T)
                gt_coll = gt_coll & em.view(1, T)
            attributable = pred_coll & ~gt_coll.unsqueeze(0)  # (M, n_agents, T)
            target[b] = attributable.any(dim=-1).any(dim=-1).to(target.dtype)
            weight[b] = 1.0

        valid = weight.sum() > 0
        if not valid:
            return None, None
        loss = F.binary_cross_entropy_with_logits(
            per_mode_logit, target, weight=weight, reduction='sum'
        ) / weight.sum().clamp_min(1.0)
        # Note: caller multiplies by self.conflict_loss_weight; do NOT scale here.

        with torch.no_grad():
            wm = weight > 0
            pos = wm & (target > 0.5)
            neg = wm & (target < 0.5)
            pred = (per_mode_logit > 0).float()
            pos_logit_mean = per_mode_logit[pos].mean() if pos.any() else per_mode_logit.new_zeros(())
            neg_logit_mean = per_mode_logit[neg].mean() if neg.any() else per_mode_logit.new_zeros(())
            acc_05 = (pred[wm] == target[wm]).float().mean() if wm.any() else per_mode_logit.new_zeros(())
            pos_rate = (target[wm] > 0.5).float().mean() if wm.any() else per_mode_logit.new_zeros(())
        diag = dict(
            plan_conf_pos_logit_mean=pos_logit_mean.detach(),
            plan_conf_neg_logit_mean=neg_logit_mean.detach(),
            plan_conf_acc_05=acc_05.detach(),
            plan_conf_pos_rate=pos_rate.detach(),
        )
        return loss, diag

    def _loss_planning_conflict(self, conf_logits, reg, data, motion_loss_cache):
        """Object-conflict BCE on (matched-anchor, plan-mode) pairs.

        conf_logits: (bs, num_anchor, M) raw logits.
        reg: (bs, 1, M, ego_fut_ts, 2) delta plan predictions.
        Hungarian indices from motion_loss_cache pair pred_idx <-> GT idx.

        Label source is selected by `self.conflict_label_source`:
          - 'predicted' (v1): label = min_t ||cumsum(reg) − agent_pos|| < thr.
            Self-referential — head is supervised on its own current trajectory.
          - 'anchor' (v2): label = min_t ||cumsum(plan_anchor) − agent_pos|| < thr.
            Stationary, exogenous to the planner's current iteration.
          - 'evalmatch' (v3): label is the per-(mode, anchor) collision flag
            computed by replicating ``planning_eval.py`` geometry — predicted
            ego rotated bbox (4.084 × 1.85, 0.5m forward offset, yaw via
            trajectory tangent) vs GT agent rotated bbox (GT WLH, GT t=0 yaw +
            tangent), polygon-on-polygon SAT intersection per timestep,
            ``any(t)`` reduction, ``box_coll AND NOT gt_box_coll`` mask.
            This is the exact eval criterion the head should learn to predict.
          - 'evalmatch_mode' (v4): same evalmatch label generation, then
            aggregated to per-mode via ``any(matched_anchor)``. The training
            objective matches the inference-time selector
            (``rescore_learned_hard``'s ``any(anchor)`` reduction): BCE is
            applied on a smooth-max-over-matched-anchors of the head logits
            against the per-mode label. Fixes train/eval aggregation mismatch
            that v3 had — v3 supervised per-(anchor, mode) but eval reduces
            with ``any(anchor)``, causing per-anchor FPs to cascade.

        Returns (loss, diag) where diag is a dict with `pos_logit_mean`,
        `neg_logit_mean`, `acc_05`, `pos_rate` over the labelled (anchor, mode)
        entries, or None if no valid samples in the batch.
        """
        # Per-mode aggregation path for det-free conflict samplers
        # (image_at_plan, image_at_init_topk, image_at_scene_query). Bypasses
        # Hungarian matching and uses per-mode evalmatch labels directly.
        if self.conflict_input in (
            'image_at_plan', 'image_at_init_topk', 'image_at_init_topk_and_plan',
            'image_at_ego_grid', 'image_at_scene_query',
        ):
            return self._loss_planning_conflict_per_mode(conf_logits, reg, data)
        if motion_loss_cache is None:
            return None, None
        gt_boxes = data.get('gt_bboxes_3d')
        gt_traj = data.get('gt_agent_fut_trajs')
        gt_traj_mask = data.get('gt_agent_fut_masks')
        if gt_boxes is None or gt_traj is None:
            return None, None

        bs, num_anchor, M = conf_logits.shape
        device = conf_logits.device
        T_ego = reg.shape[-2]

        target = conf_logits.new_zeros(bs, num_anchor, M)
        weight = conf_logits.new_zeros(bs, num_anchor, M)

        if self.conflict_label_source in ('evalmatch', 'evalmatch_mode'):
            self._fill_evalmatch_labels(
                target, weight, reg, data, motion_loss_cache
            )
        else:
            if self.conflict_label_source == 'anchor':
                # plan_anchor: (num_cmd, ego_fut_mode, ego_fut_ts, 2)
                # cumulative XY in lidar frame. Flatten cmd × mode → M.
                anchor_xy = self.plan_anchor.detach()
                anchor_xy = anchor_xy.reshape(M, anchor_xy.shape[-2], 2)
                T_anchor = anchor_xy.shape[-2]
                ego_xy_full = anchor_xy.unsqueeze(0).expand(bs, M, T_anchor, 2)
                ego_xy_full = ego_xy_full.to(device=device, dtype=conf_logits.dtype)
                T_ego_eff = min(T_anchor, T_ego)
            else:
                ego_xy_full = reg.detach().squeeze(1).cumsum(dim=-2)
                T_ego_eff = T_ego

            thr2 = float(self.conflict_threshold) ** 2
            for b in range(bs):
                pred_idx, target_idx = motion_loss_cache['indices'][b]
                if pred_idx is None or len(pred_idx) == 0:
                    continue
                boxes_b = gt_boxes[b].to(device)
                trajs_b = gt_traj[b].to(device)
                if trajs_b.shape[0] == 0:
                    continue
                T = min(trajs_b.shape[1], T_ego_eff)
                if T == 0:
                    continue
                agent_xy0 = boxes_b[target_idx, :2]
                agent_traj = trajs_b[target_idx, :T]
                agent_pos = agent_xy0.unsqueeze(1) + agent_traj.cumsum(dim=1)

                ego_pos = ego_xy_full[b, :, :T, :]

                diff = agent_pos.unsqueeze(1) - ego_pos.unsqueeze(0)
                d2 = (diff ** 2).sum(dim=-1)

                if gt_traj_mask is not None:
                    tmask = gt_traj_mask[b].to(device)[target_idx, :T].bool()
                    d2 = torch.where(
                        tmask.unsqueeze(1).expand(-1, M, -1),
                        d2,
                        d2.new_full((), float('inf')),
                    )
                min_d2 = d2.min(dim=-1).values
                label = (min_d2 < thr2).to(target.dtype)

                if isinstance(pred_idx, torch.Tensor):
                    pidx = pred_idx.to(device).long()
                else:
                    pidx = torch.as_tensor(pred_idx, device=device, dtype=torch.long)
                target[b, pidx] = label
                weight[b, pidx] = 1.0

        # `evalmatch_mode`: collapse per-(anchor, mode) tensors to per-mode by
        # any(matched_anchor) on the label and smooth-max(matched_anchor) on
        # the logit. Done by substituting `conf_logits`, `target`, `weight`
        # before the shared BCE+diag block below so diagnostics report
        # per-mode classifier quality instead of per-(anchor, mode).
        if self.conflict_label_source == 'evalmatch_mode':
            matched_anchor_mask = weight.any(dim=-1)  # (bs, A) — anchor matched
            anchor_mask_expanded = matched_anchor_mask.unsqueeze(-1)  # (bs, A, 1)
            # Per-mode label: any matched anchor labelled positive.
            target_per_mode = (target * weight).max(dim=1).values  # (bs, M)
            # Per-mode weight: 1 if at least one matched anchor exists this batch item.
            wb = matched_anchor_mask.any(dim=1).float()  # (bs,)
            weight_per_mode = wb.unsqueeze(-1).expand(bs, M).contiguous()
            # Smooth-max logit over matched anchors only.
            tau = float(self.conflict_smooth_max_tau)
            masked = conf_logits.masked_fill(~anchor_mask_expanded, -1e9)
            # (1/τ)·logsumexp(τ·x, dim) → max as τ → ∞; τ=5 already very tight.
            pred_logit_per_mode = torch.logsumexp(tau * masked, dim=1) / tau  # (bs, M)
            # Substitute for the shared BCE+diag below.
            conf_logits = pred_logit_per_mode
            target = target_per_mode
            weight = weight_per_mode

        denom = weight.sum().clamp(min=1.0)
        pos_w_val = self._resolve_conflict_pos_weight(target, weight)
        pos_w = conf_logits.new_tensor([float(pos_w_val)])
        loss_per = F.binary_cross_entropy_with_logits(
            conf_logits, target, reduction='none', pos_weight=pos_w
        )
        loss = (loss_per * weight).sum() / denom

        # Diagnostics over labelled entries only.
        with torch.no_grad():
            mask = weight.bool()
            if mask.any():
                probs = torch.sigmoid(conf_logits[mask].float())
                tgt = target[mask].float()
                pos_n = tgt.sum().clamp(min=1.0)
                neg_n = (1.0 - tgt).sum().clamp(min=1.0)
                pos_logit_mean = (probs * tgt).sum() / pos_n
                neg_logit_mean = (probs * (1.0 - tgt)).sum() / neg_n
                pred_pos = (probs > 0.5).float()
                acc_05 = (pred_pos == tgt).float().mean()
                pos_rate = tgt.mean()
                diag = {
                    'pos_logit_mean': pos_logit_mean.detach(),
                    'neg_logit_mean': neg_logit_mean.detach(),
                    'acc_05': acc_05.detach(),
                    'pos_rate': pos_rate.detach(),
                }
            else:
                diag = None
        return loss, diag

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

    def _loss_planning_distill_rescore(self, cls, reg, data, motion_model_outs, det_output):
        """Distill `HierarchicalPlanningDecoder.rescore()`'s collide flag into
        `plan_cls`: pushes plan_cls of rescore-flagged modes toward 0 via
        BCE-with-logits. Geometry inputs all detached so no gradient flows
        back through plan_reg/motion/det.

        At inference (`use_rescore=False`), the planner's own logits naturally
        encode the rescore constraint without the explicit -999 mask.

        cls: (bs, M=ego_fut_mode) raw logits — already the cmd-indexed slice
            after `planning_sampler.sample` + `flatten(end_dim=1)`.
        reg: (bs, 1, 3*ego_fut_mode, ego_fut_ts, 2) delta plan predictions —
            the full pre-sample tensor; cmd slicing is done here.

        Returns (loss, diag_dict) where diag_dict reports the cmd-mode
        collision rate and the average sigmoid(plan_cls) on flagged modes.
        """
        decoder = self.planning_decoder
        if decoder is None or not hasattr(decoder, 'compute_rescore_collision_mask'):
            return None
        bs = reg.shape[0]
        device = reg.device

        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)
        bs_idx = torch.arange(bs, device=device)
        M = self.ego_fut_mode

        plan_cls_cmd = cls  # (bs, M) — already cmd-indexed + flattened

        plan_reg_modes = reg.squeeze(1).reshape(bs, 3, M, self.ego_fut_ts, 2)
        plan_reg_cmd_cum = plan_reg_modes[bs_idx, cmd].cumsum(dim=-2).detach()  # (bs, M, T, 2)

        det_anchors = det_output["prediction"][-1].detach()
        det_classification = det_output["classification"][-1].sigmoid().detach()
        det_confidence = det_classification.max(dim=-1).values

        motion_cls = motion_model_outs["classification"][-1].sigmoid().detach()
        motion_reg = motion_model_outs["prediction"][-1].detach()
        if self.motion_target_in_agent_frame:
            # Rescore collision check is geometric and operates in lidar
            # frame against det_anchors; rotate agent-frame motion predictions
            # back to lidar before consuming.
            motion_reg = self._agent2lidar(motion_reg, det_anchors)

        with torch.no_grad():
            collide = decoder.compute_rescore_collision_mask(
                plan_reg_cmd_cum, motion_cls, motion_reg, det_anchors, det_confidence,
            )  # bool (bs, M)

        weight = collide.to(plan_cls_cmd.dtype)
        denom = weight.sum().clamp(min=1.0)
        target = torch.zeros_like(plan_cls_cmd)
        loss_per = F.binary_cross_entropy_with_logits(
            plan_cls_cmd, target, reduction='none'
        )
        loss = (loss_per * weight).sum() / denom

        with torch.no_grad():
            probs = torch.sigmoid(plan_cls_cmd.detach().float())
            collide_rate = weight.mean()
            if weight.sum() > 0:
                flagged_prob = (probs * weight).sum() / weight.sum().clamp(min=1.0)
            else:
                flagged_prob = probs.new_zeros(())
            diag = {
                'collide_rate': collide_rate.detach(),
                'flagged_prob_mean': flagged_prob.detach(),
            }
        return loss, diag

    @force_fp32(apply_to=("model_outs"))
    def post_process(
        self,
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        # Agent-frame regression: motion_output["prediction"][-1] is in each
        # agent's heading-aligned frame internally. Rotate the last-stage
        # prediction to lidar before the decoders consume it (motion ADE/FDE,
        # planning rescore collision check). Build a shallow-copied dict so
        # we don't mutate the caller's reference.
        if self.motion_target_in_agent_frame and not self.ego_only_planning:
            det_anchors_last = det_output["prediction"][-1]
            preds = list(motion_output["prediction"])
            preds[-1] = self._agent2lidar(preds[-1], det_anchors_last)
            motion_output = {**motion_output, "prediction": preds}
        if self.ego_only_planning:
            bs = det_output["classification"][-1].shape[0]
            motion_result = [dict() for _ in range(bs)]
        else:
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
