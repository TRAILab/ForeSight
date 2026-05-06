import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.utils import build_from_cfg
from mmcv.cnn import Linear
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmdet.models import HEADS
from mmdet.core.bbox.builder import BBOX_SAMPLERS, BBOX_CODERS
from mmdet.models import build_loss

from projects.mmdet3d_plugin.core.box3d import *

from ..attention import gen_sineembed_for_position
from ..blocks import linear_relu_ln
from .motion_planning_head import (
    _eval_get_yaw,
    _agent_get_yaw,
    _make_rect_corners_topdown,
    _rect_intersects_sat,
)


@HEADS.register_module()
class PlanningOnlyHead(BaseModule):
    """Clean ego-only planning head forked from MotionPlanningHead.

    Step 1 of the planner redesign: same per-stage skeleton as the minS2 ego-
    only baseline (temp_gnn / deformable / ffn / refine × 6), with all motion,
    detection, and map paths stripped. Conflict head reads image features at
    static plan-anchor BEV positions (image_at_plan).
    """

    def __init__(
        self,
        ego_fut_ts=6,
        ego_fut_mode=6,
        num_driving_cmds=3,
        plan_anchor=None,
        embed_dims=256,
        decouple_attn=True,
        instance_queue=None,
        operation_order=None,
        temp_graph_model=None,
        deformable_model=None,
        ffn=None,
        norm_layer=None,
        refine_layer=None,
        planning_sampler=None,
        plan_loss_cls=None,
        plan_loss_reg=None,
        plan_loss_status=None,
        planning_decoder=None,
        planning_deformable_waypoints=(0, 1, 2, 3, 4, 5),
        plan_ego_status_encode_enable=False,
        plan_ego_status_indices=(0, 1, 5, 6, 7),
        with_conflict_head=False,
        conflict_image_sampler=None,
        conflict_loss_weight=0.1,
        conflict_smooth_max_tau=5.0,
        # `fut_ts` / `fut_mode` are unused but accepted so configs that
        # historically pointed motion settings at the head don't error.
        fut_ts=None,
        fut_mode=None,
    ):
        super().__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.num_driving_cmds = num_driving_cmds
        self.embed_dims = embed_dims
        self.decouple_attn = decouple_attn
        self.operation_order = list(operation_order)
        self.planning_deformable_waypoints = list(planning_deformable_waypoints)
        self.plan_ego_status_encode_enable = bool(plan_ego_status_encode_enable)
        self.plan_ego_status_indices = list(plan_ego_status_indices)
        self.with_conflict_head = bool(with_conflict_head)
        self.conflict_loss_weight = float(conflict_loss_weight)
        self.conflict_smooth_max_tau = float(conflict_smooth_max_tau)

        self._n_deformable_stages = sum(
            1 for op in self.operation_order if op == "deformable"
        )

        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.instance_queue = build(instance_queue, PLUGIN_LAYERS)
        self.planning_sampler = build(planning_sampler, BBOX_SAMPLERS)
        self.planning_decoder = build(planning_decoder, BBOX_CODERS)

        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
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

        if self.with_conflict_head:
            assert conflict_image_sampler is not None, (
                "with_conflict_head=True requires conflict_image_sampler config"
            )
            self.conflict_image_sampler = build_from_cfg(
                conflict_image_sampler, ATTENTION
            )
        else:
            self.conflict_image_sampler = None

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

        self.plan_loss_cls = build_loss(plan_loss_cls)
        self.plan_loss_reg = build_loss(plan_loss_reg)
        self.plan_loss_status = build_loss(plan_loss_status)

        # plan anchor (k-means buffer)
        plan_anchor = np.load(plan_anchor)
        assert plan_anchor.shape[1] == ego_fut_mode, (
            f"plan_anchor mode dim {plan_anchor.shape[1]} != ego_fut_mode {ego_fut_mode}"
        )
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        if self.plan_ego_status_encode_enable:
            in_dim = len(self.plan_ego_status_indices)
            self.plan_ego_status_encoder = nn.Sequential(
                nn.Linear(in_dim, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, embed_dims),
            )

        # K samples per plan mode for the per-mode conflict aggregation; set
        # only when conflict head is active so loss helpers can reshape.
        self._conflict_per_mode_K = None

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            if op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    # =========== anchor box construction ===========
    def _build_planning_anchor_boxes_multi(self, plan_anchor, ego_anchor, waypoints):
        num_mode = plan_anchor.shape[1]
        ego_fut_ts = plan_anchor.shape[-2]
        anchor_base = ego_anchor.expand(-1, num_mode, -1).clone()

        boxes = []
        for w in waypoints:
            endpoint = plan_anchor[..., w, :]
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

        return torch.stack(boxes, dim=2)

    def graph_model(self, index, query, key=None, value=None,
                    query_pos=None, key_pos=None, **kwargs):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query, key, value,
                query_pos=query_pos, key_pos=key_pos,
                **kwargs,
            )
        )

    # =========== forward ===========
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
        # det_output is consumed only to drive instance_queue temporal state
        # (and to provide a batch-size reference). The agent path itself is
        # discarded — only the ego token flows through this head.
        bs = det_output["instance_feature"].shape[0]

        # =========== get ego/temporal feature/anchor ===========
        (
            ego_feature,
            ego_anchor,
            temp_instance_feature,
            temp_anchor,
            temp_mask,
        ) = self.instance_queue.get(
            det_output, feature_maps, metas, bs, mask, anchor_handler,
        )
        ego_anchor_embed = anchor_encoder(ego_anchor)

        # Drop the agent temporal slots; keep only the ego entry. (queue.get
        # concatenates ego at the tail along dim=1.)
        temp_instance_feature = temp_instance_feature[:, -1:]
        temp_anchor_embed = anchor_encoder(temp_anchor)[:, -1:]
        temp_mask = temp_mask[:, -1:]
        temp_instance_feature = temp_instance_feature.flatten(0, 1)
        temp_anchor_embed = temp_anchor_embed.flatten(0, 1)
        temp_mask = temp_mask.flatten(0, 1)

        # =========== plan mode anchor / query init ===========
        plan_anchor = self.plan_anchor[None].expand(bs, -1, -1, -1, -1)
        plan_anchor = plan_anchor.reshape(
            bs, self.num_driving_cmds * self.ego_fut_mode, self.ego_fut_ts, 2
        ).contiguous()

        plan_pos = gen_sineembed_for_position(
            plan_anchor[..., -1, :], hidden_dim=self.embed_dims
        )
        plan_mode_query = self.plan_anchor_encoder(plan_pos)

        if self.plan_ego_status_encode_enable:
            es_in = metas['ego_status'][:, self.plan_ego_status_indices].to(
                plan_mode_query.dtype
            )
            ego_status_embed = self.plan_ego_status_encoder(es_in)
            plan_mode_query = plan_mode_query + ego_status_embed[:, None, :]
        else:
            ego_status_embed = None

        # Single ego token in instance_feature; no agents, no DN.
        instance_feature = ego_feature  # (bs, 1, D)
        anchor_embed = ego_anchor_embed  # (bs, 1, D)

        # =========== conflict-head image features (image_at_plan) ===========
        # Sample image features once at the static plan_anchor BEV positions
        # (cumsum of k-means deltas → per-mode per-waypoint XY in lidar). Same
        # features are reused at every refine stage.
        if self.with_conflict_head:
            plan_xy = self.plan_anchor.detach()  # (cmd, mode, T, 2)
            M_total = plan_xy.shape[0] * plan_xy.shape[1]
            T = plan_xy.shape[2]
            plan_xy = plan_xy.reshape(M_total, T, 2).cumsum(dim=-2)
            K = M_total * T
            anc = plan_xy.new_zeros(K, 11)
            anc[:, 0:2] = plan_xy.reshape(K, 2)
            anc[:, 7] = 1.0  # cos(yaw=0)
            sampler_anchors = (
                anc.unsqueeze(0).expand(bs, -1, -1).contiguous()
                .to(ego_feature.device, dtype=ego_feature.dtype)
            )
            sampler_query = sampler_anchors.new_zeros(bs, K, self.embed_dims)
            sampler_anchor_embed = anchor_encoder(sampler_anchors)
            conflict_image_features = self.conflict_image_sampler(
                sampler_query, sampler_anchors, sampler_anchor_embed,
                feature_maps, metas,
            )
            self._conflict_per_mode_K = T
        else:
            conflict_image_features = None
            self._conflict_per_mode_K = None

        # =========== decoder loop ===========
        planning_classification = []
        planning_prediction = []
        planning_status = []
        planning_conflict_logits = []

        _deformable_stage_idx = 0
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                # Ego token attends to its own temporal queue. Single-token
                # query; flatten/unflatten matches the minS2 layout.
                num_normal = 1
                instance_feature = self.graph_model(
                    i,
                    instance_feature[:, :num_normal].flatten(0, 1).unsqueeze(1),
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed[:, :num_normal].flatten(0, 1).unsqueeze(1),
                    key_pos=temp_anchor_embed,
                    key_padding_mask=temp_mask,
                ).reshape(bs, num_normal, self.embed_dims)
            elif op == "norm":
                instance_feature = self.layers[i](instance_feature)
            elif op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                _deformable_stage_idx += 1
                is_last = _deformable_stage_idx == self._n_deformable_stages

                # Multi-waypoint anchor boxes shared across stages.
                boxes_multi = self._build_planning_anchor_boxes_multi(
                    plan_anchor.detach(), ego_anchor,
                    self.planning_deformable_waypoints,
                )  # (bs, num_mode, K, 11)
                num_mode = boxes_multi.shape[1]
                K_wp = boxes_multi.shape[2]
                boxes_flat = boxes_multi.reshape(bs, num_mode * K_wp, 11)
                embed_flat = anchor_encoder(boxes_flat)

                if is_last:
                    # Stage 6: ego instance_feature replaces plan_mode_query as
                    # the deformable query. Outputs are mode-aggregated and
                    # written back into instance_feature.
                    ego_feat_exp = instance_feature[:, 0:1].expand(
                        -1, num_mode * K_wp, -1
                    )
                    attended = self.layers[i](
                        ego_feat_exp, boxes_flat, embed_flat,
                        feature_maps, metas,
                    )
                    attended = attended.reshape(
                        bs, num_mode, K_wp, self.embed_dims
                    ).mean(dim=2)  # (bs, num_mode, D)
                    if planning_classification:
                        plan_weights = (
                            planning_classification[-1].detach()
                            .squeeze(1).softmax(dim=-1)
                        )
                    else:
                        plan_weights = torch.full(
                            (bs, num_mode), 1.0 / num_mode,
                            device=instance_feature.device,
                            dtype=instance_feature.dtype,
                        )
                    ego_new = (attended * plan_weights.unsqueeze(-1)).sum(
                        dim=1, keepdim=True
                    )
                    instance_feature = ego_new
                else:
                    # Stages 1-5: plan_mode_query is the deformable query;
                    # multi-waypoint K-pool back to per-mode mean.
                    query_exp = (
                        plan_mode_query.unsqueeze(2)
                        .expand(-1, -1, K_wp, -1)
                        .reshape(bs, num_mode * K_wp, self.embed_dims)
                    )
                    attended = self.layers[i](
                        query_exp, boxes_flat, embed_flat,
                        feature_maps, metas,
                    )
                    plan_mode_query = attended.reshape(
                        bs, num_mode, K_wp, self.embed_dims
                    ).mean(dim=2)
            elif op == "refine":
                plan_query = (
                    plan_mode_query.unsqueeze(1)
                    + (instance_feature + anchor_embed)[:, 0:1].unsqueeze(2)
                )
                plan_cls, plan_reg, plan_status, plan_conflict = self.layers[i](
                    plan_query,
                    instance_feature[:, 0:1],
                    anchor_embed[:, 0:1],
                    agent_features=conflict_image_features,
                )
                if plan_conflict is not None:
                    planning_conflict_logits.append(plan_conflict)
                if planning_prediction:
                    plan_reg = plan_reg + planning_prediction[-1].detach()
                planning_classification.append(plan_cls)
                planning_prediction.append(plan_reg)
                planning_status.append(plan_status)

                # Update plan_anchor (cumsum of latest reg) for next stage.
                plan_anchor = plan_reg.detach().squeeze(1).cumsum(dim=-2)
                plan_mode_query = self.plan_anchor_encoder(
                    gen_sineembed_for_position(
                        plan_anchor[..., -1, :], hidden_dim=self.embed_dims
                    )
                )
                if ego_status_embed is not None:
                    plan_mode_query = (
                        plan_mode_query + ego_status_embed[:, None, :]
                    )

        # =========== queue caches ===========
        # cache_motion is called with empty agent buffer to keep det matching
        # state in the queue (prev_confidence / prev_instance_id) consistent.
        empty_motion = instance_feature.new_zeros(bs, 0, self.embed_dims)
        self.instance_queue.cache_motion(empty_motion, det_output, metas)
        self.instance_queue.cache_planning(
            instance_feature[:, 0:1], planning_status[-1]
        )

        # =========== outputs ===========
        # motion_output is empty but populated with zero-anchor tensors so the
        # planning decoder's signature (motion_cls / motion_reg) is satisfied.
        motion_cls_empty = instance_feature.new_zeros(bs, 0, self.ego_fut_mode)
        motion_reg_empty = instance_feature.new_zeros(
            bs, 0, self.ego_fut_mode, self.ego_fut_ts, 2
        )
        motion_output = {
            "classification": [motion_cls_empty],
            "prediction": [motion_reg_empty],
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
        if planning_conflict_logits:
            planning_output["conflict_logits"] = planning_conflict_logits
        return motion_output, planning_output

    # =========== loss ===========
    def loss(
        self,
        motion_model_outs,
        planning_model_outs,
        data,
        motion_loss_cache,
        det_output=None,
    ):
        return self.loss_planning(planning_model_outs, data)

    @force_fp32(apply_to=("model_outs",))
    def loss_planning(self, model_outs, data):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        status_preds = model_outs["status"]
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
                cls, reg,
                data["gt_ego_fut_trajs"], data["gt_ego_fut_masks"],
                data,
            )
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)

            reg_weight = reg_weight.flatten(end_dim=1).unsqueeze(-1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_loss = self.plan_loss_reg(reg_pred, reg_target, weight=reg_weight)

            status_loss = self.plan_loss_status(
                status.squeeze(1), data["ego_status"]
            )

            output[f"planning_loss_cls_{decoder_idx}"] = cls_loss
            output[f"planning_loss_reg_{decoder_idx}"] = reg_loss
            output[f"planning_loss_status_{decoder_idx}"] = status_loss

            if decoder_idx < len(conf_logits_list):
                conf_ret = self._loss_planning_conflict_per_mode(
                    conf_logits_list[decoder_idx], reg, data
                )
                if conf_ret is not None:
                    conf_loss, conf_diag = conf_ret
                    if conf_loss is not None:
                        output[f"planning_loss_conf_{decoder_idx}"] = (
                            conf_loss * self.conflict_loss_weight
                        )
                    if (
                        conf_diag is not None
                        and decoder_idx == len(conf_logits_list) - 1
                    ):
                        for k, v in conf_diag.items():
                            output[f"plan_conf_{k}"] = v

        return output

    def _loss_planning_conflict_per_mode(self, conf_logits, reg, data):
        """Per-mode conflict BCE for image_at_plan layout.

        conf_logits: (bs, M*K, M_pred) where M = M_pred = num_driving_cmds *
        ego_fut_mode and K = ego_fut_ts. Smooth-max over the K samples that
        belong to each output mode (diagonal mode read), BCE against the
        per-mode evalmatch label.
        """
        bs, total, M_pred = conf_logits.shape
        K = self._conflict_per_mode_K
        if K is None or total != M_pred * K:
            return None, None
        device = conf_logits.device
        tau = self.conflict_smooth_max_tau
        logits_per_mode = conf_logits.reshape(bs, M_pred, K, M_pred)
        diag_logits = logits_per_mode.diagonal(dim1=1, dim2=3)  # (bs, K, M)
        per_mode_logit = (1.0 / tau) * torch.logsumexp(tau * diag_logits, dim=1)

        gt_boxes = data.get("gt_bboxes_3d")
        gt_traj = data.get("gt_agent_fut_trajs")
        gt_traj_mask = data.get("gt_agent_fut_masks")
        gt_ego_traj = data.get("gt_ego_fut_trajs")
        gt_ego_mask = data.get("gt_ego_fut_masks")
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
            attributable = pred_coll & ~gt_coll.unsqueeze(0)
            target[b] = attributable.any(dim=-1).any(dim=-1).to(target.dtype)
            weight[b] = 1.0

        if weight.sum() == 0:
            return None, None
        loss = F.binary_cross_entropy_with_logits(
            per_mode_logit, target, weight=weight, reduction="sum"
        ) / weight.sum().clamp_min(1.0)

        with torch.no_grad():
            wm = weight > 0
            pos = wm & (target > 0.5)
            neg = wm & (target < 0.5)
            pred = (per_mode_logit > 0).float()
            zero = per_mode_logit.new_zeros(())
            pos_logit_mean = per_mode_logit[pos].mean() if pos.any() else zero
            neg_logit_mean = per_mode_logit[neg].mean() if neg.any() else zero
            acc_05 = (pred[wm] == target[wm]).float().mean() if wm.any() else zero
            pos_rate = (target[wm] > 0.5).float().mean() if wm.any() else zero
        diag = dict(
            plan_conf_pos_logit_mean=pos_logit_mean.detach(),
            plan_conf_neg_logit_mean=neg_logit_mean.detach(),
            plan_conf_acc_05=acc_05.detach(),
            plan_conf_pos_rate=pos_rate.detach(),
        )
        return loss, diag

    # =========== post-process ===========
    @force_fp32(apply_to=("motion_output", "planning_output"))
    def post_process(self, det_output, motion_output, planning_output, data):
        bs = det_output["classification"][-1].shape[0]
        motion_result = [dict() for _ in range(bs)]
        planning_result = self.planning_decoder.decode(
            det_output, motion_output, planning_output, data,
        )
        return motion_result, planning_result
