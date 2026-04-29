from typing import Optional

import numpy as np
import torch

from mmdet.core.bbox.builder import BBOX_CODERS

from projects.mmdet3d_plugin.core.box3d import *
from projects.mmdet3d_plugin.models.detection3d.decoder import *
from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners


@BBOX_CODERS.register_module()
class SparseBox3DMotionDecoder(SparseBox3DDecoder):
    def __init__(self):
        super(SparseBox3DMotionDecoder, self).__init__()

    def decode(
        self,
        cls_scores,
        box_preds,
        instance_id=None,
        quality=None,
        motion_output=None,
        output_idx=-1,
    ):
        squeeze_cls = instance_id is not None

        cls_scores = cls_scores[output_idx].sigmoid()

        if squeeze_cls:
            cls_scores, cls_ids = cls_scores.max(dim=-1)
            cls_scores = cls_scores.unsqueeze(dim=-1)

        box_preds = box_preds[output_idx]
        bs, num_pred, num_cls = cls_scores.shape
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            self.num_output, dim=1, sorted=self.sorted
        )
        if not squeeze_cls:
            cls_ids = indices % num_cls
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold

        if quality[output_idx] is None:
            quality = None
        if quality is not None:
            centerness = quality[output_idx][..., CNS]
            centerness = torch.gather(centerness, 1, indices // num_cls)
            cls_scores_origin = cls_scores.clone()
            cls_scores *= centerness.sigmoid()
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)
            if not squeeze_cls:
                cls_ids = torch.gather(cls_ids, 1, idx)
            if self.score_threshold is not None:
                mask = torch.gather(mask, 1, idx)
            indices = torch.gather(indices, 1, idx)

        output = []
        anchor_queue = motion_output["anchor_queue"]
        anchor_queue = torch.stack(anchor_queue, dim=2)
        period = motion_output["period"]

        for i in range(bs):
            category_ids = cls_ids[i]
            if squeeze_cls:
                category_ids = category_ids[indices[i]]
            scores = cls_scores[i]
            box = box_preds[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]
            if quality is not None:
                scores_origin = cls_scores_origin[i]
                if self.score_threshold is not None:
                    scores_origin = scores_origin[mask[i]]

            box = decode_box(box)
            trajs = motion_output["prediction"][-1]
            traj_cls = motion_output["classification"][-1].sigmoid()
            traj = trajs[i, indices[i] // num_cls]
            traj_cls = traj_cls[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                traj = traj[mask[i]]
                traj_cls = traj_cls[mask[i]]
            traj = traj.cumsum(dim=-2) + box[:, None, None, :2]
            output.append(
                {
                    "trajs_3d": traj.cpu(),
                    "trajs_score": traj_cls.cpu()
                }
            )

            temp_anchor = anchor_queue[i, indices[i] // num_cls]
            temp_period = period[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                temp_anchor = temp_anchor[mask[i]]
                temp_period = temp_period[mask[i]]
            num_pred, queue_len = temp_anchor.shape[:2]
            temp_anchor = temp_anchor.flatten(0, 1)
            temp_anchor = decode_box(temp_anchor)
            temp_anchor = temp_anchor.reshape([num_pred, queue_len, box.shape[-1]])
            output[-1]['anchor_queue'] = temp_anchor.cpu()
            output[-1]['period'] = temp_period.cpu()
        
        return output


@BBOX_CODERS.register_module()
class HierarchicalPlanningDecoder(object):
    def __init__(
        self,
        ego_fut_ts,
        ego_fut_mode,
        num_driving_cmds=3,
        use_rescore=False,
        use_rescore_soft=False,
        rescore_soft_w_col=10.0,
        rescore_soft_sigma=2.0,
        use_rescore_learned=False,
        rescore_learned_w_col=10.0,
        rescore_learned_score_thresh=0.5,
        use_rescore_learned_hard=False,
        rescore_learned_hard_score_thresh=0.5,
        rescore_learned_hard_prob_thresh=0.5,
        rescore_learned_hard_aggregation='any',
        rescore_learned_hard_topk_k=2,
        use_rescore_hybrid_or=False,
        rescore_confidence_source='det',
        rescore_score_thresh=0.5,
    ):
        super(HierarchicalPlanningDecoder, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.num_driving_cmds = num_driving_cmds
        self.use_rescore = use_rescore
        self.use_rescore_soft = use_rescore_soft
        self.rescore_soft_w_col = rescore_soft_w_col
        self.rescore_soft_sigma = rescore_soft_sigma
        self.use_rescore_learned = use_rescore_learned
        self.rescore_learned_w_col = rescore_learned_w_col
        self.rescore_learned_score_thresh = rescore_learned_score_thresh
        self.use_rescore_learned_hard = use_rescore_learned_hard
        self.rescore_learned_hard_score_thresh = rescore_learned_hard_score_thresh
        self.rescore_learned_hard_prob_thresh = rescore_learned_hard_prob_thresh
        # `rescore_learned_hard_aggregation`: how per-(anchor, mode) collision
        # probabilities collapse to per-mode binary flags.
        #   'any'         : (prob > thr).any(anchor)         — original behavior
        #   'topk'        : (prob > thr).sum(anchor) >= k    — robust to single-anchor FPs
        #   'detweighted' : (det_conf * prob > thr).any(...) — soft det-conf weighting,
        #                   replaces hard det_conf cutoff entirely.
        valid_agg = {'any', 'topk', 'detweighted'}
        if rescore_learned_hard_aggregation not in valid_agg:
            raise ValueError(
                f"rescore_learned_hard_aggregation must be in {valid_agg}, "
                f"got {rescore_learned_hard_aggregation!r}"
            )
        self.rescore_learned_hard_aggregation = rescore_learned_hard_aggregation
        self.rescore_learned_hard_topk_k = int(rescore_learned_hard_topk_k)
        # `use_rescore_hybrid_or`: applies hard rescore THEN learned-hard
        # rescore in sequence on the same plan_cls. -999 masks accumulate, so
        # the per-mode collide flag is the OR of both selectors. Tests whether
        # the learned head adds any signal hard rescore misses.
        self.use_rescore_hybrid_or = use_rescore_hybrid_or
        # `rescore_confidence_source`: which per-agent score gates the
        # rescore.filter_mask. 'det' (default) uses det_confidence (max class
        # prob from Sparse4DHead). 'motion' uses motion top-mode probability
        # (max over fut_mode of softmax(motion_cls)). Diagnostic for whether
        # rescore needs det specifically vs any agent-confidence channel.
        self.rescore_confidence_source = rescore_confidence_source
        self.rescore_score_thresh = rescore_score_thresh
    
    def decode(
        self, 
        det_output,
        motion_output,
        planning_output, 
        data,
    ):
        classification = planning_output['classification'][-1]
        prediction = planning_output['prediction'][-1]
        bs = classification.shape[0]
        classification = classification.reshape(bs, self.num_driving_cmds, self.ego_fut_mode)
        prediction = prediction.reshape(bs, self.num_driving_cmds, self.ego_fut_mode, self.ego_fut_ts, 2).cumsum(dim=-2)
        classification, final_planning = self.select(
            det_output, motion_output, classification, prediction, data, planning_output
        )
        anchor_queue = planning_output["anchor_queue"]
        anchor_queue = torch.stack(anchor_queue, dim=2)
        period = planning_output["period"]
        output = []
        for i, (cls, pred) in enumerate(zip(classification, prediction)):
            output.append(
                {
                    "planning_score": cls.sigmoid().cpu(),
                    "planning": pred.cpu(),
                    "final_planning": final_planning[i].cpu(),
                    "ego_period": period[i].cpu(),
                    "ego_anchor_queue": decode_box(anchor_queue[i]).cpu(),
                }
            )

        return output

    def select(
        self,
        det_output,
        motion_output,
        plan_cls,
        plan_reg,
        data,
        planning_output=None,
    ):
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        motion_cls = motion_output["classification"][-1].sigmoid()
        motion_reg = motion_output["prediction"][-1]

        # cmd select
        bs = motion_cls.shape[0]
        bs_indices = torch.arange(bs, device=motion_cls.device)
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)
        plan_cls_full = plan_cls.detach().clone()
        plan_cls = plan_cls[bs_indices, cmd]
        plan_reg = plan_reg[bs_indices, cmd]

        # rescore filter score: 'det' (default) uses Sparse4DHead's max class
        # prob; 'motion' uses motion's top-mode probability per agent. Same
        # tensor shape (bs, num_anchor); the threshold semantics stay
        # consistent with the original rescore signature.
        if self.rescore_confidence_source == 'motion':
            filter_score = motion_cls.max(dim=-1).values
        else:
            filter_score = det_confidence

        # rescore
        if self.use_rescore:
            plan_cls = self.rescore(
                plan_cls,
                plan_reg,
                motion_cls,
                motion_reg,
                det_anchors,
                filter_score,
                score_thresh=self.rescore_score_thresh,
            )
        elif self.use_rescore_soft:
            plan_cls = self.rescore_soft(
                plan_cls,
                plan_reg,
                motion_cls,
                motion_reg,
                det_anchors,
                det_confidence,
            )
        elif self.use_rescore_learned:
            plan_cls = self.rescore_learned(
                plan_cls,
                planning_output,
                det_confidence,
                cmd,
            )
        elif self.use_rescore_learned_hard:
            plan_cls = self.rescore_learned_hard(
                plan_cls,
                planning_output,
                det_confidence,
                cmd,
            )
        elif self.use_rescore_hybrid_or:
            # Hard rescore first, then learned-hard. Each writes -999 to
            # colliding modes; the union is the OR of both selectors' flags.
            plan_cls = self.rescore(
                plan_cls,
                plan_reg,
                motion_cls,
                motion_reg,
                det_anchors,
                filter_score,
                score_thresh=self.rescore_score_thresh,
            )
            plan_cls = self.rescore_learned_hard(
                plan_cls,
                planning_output,
                det_confidence,
                cmd,
            )
        plan_cls_full[bs_indices, cmd] = plan_cls
        mode_idx = plan_cls.argmax(dim=-1)
        final_planning = plan_reg[bs_indices, mode_idx]
        return plan_cls_full, final_planning

    def rescore(
        self, 
        plan_cls,
        plan_reg, 
        motion_cls,
        motion_reg, 
        det_anchors,
        det_confidence,
        score_thresh=0.5,
        static_dis_thresh=0.5,
        dim_scale=1.1,
        num_motion_mode=1,
        offset=0.5,
    ):
        
        def cat_with_zero(traj):
            zeros = traj.new_zeros(traj.shape[:-2] + (1, 2))
            traj_cat = torch.cat([zeros, traj], dim=-2)
            return traj_cat
        
        def get_yaw(traj, start_yaw=np.pi/2):
            yaw = traj.new_zeros(traj.shape[:-1])
            yaw[..., 1:-1] = torch.atan2(
                traj[..., 2:, 1] - traj[..., :-2, 1],
                traj[..., 2:, 0] - traj[..., :-2, 0],
            )
            yaw[..., -1] = torch.atan2(
                traj[..., -1, 1] - traj[..., -2, 1],
                traj[..., -1, 0] - traj[..., -2, 0],
            )
            yaw[..., 0] = start_yaw
            # for static object, estimated future yaw would be unstable
            start = traj[..., 0, :]
            end = traj[..., -1, :]
            dist = torch.linalg.norm(end - start, dim=-1)
            mask = dist < static_dis_thresh
            start_yaw = yaw[..., 0].unsqueeze(-1)
            yaw = torch.where(
                mask.unsqueeze(-1),
                start_yaw,
                yaw,
            )
            return yaw.unsqueeze(-1)
        
        ## ego
        bs = plan_reg.shape[0]
        plan_reg_cat = cat_with_zero(plan_reg)
        ego_box = det_anchors.new_zeros(bs, self.ego_fut_mode, self.ego_fut_ts + 1, 7)
        ego_box[..., [X, Y]] = plan_reg_cat
        ego_box[..., [W, L, H]] = ego_box.new_tensor([4.08, 1.73, 1.56]) * dim_scale
        ego_box[..., [YAW]] = get_yaw(plan_reg_cat)

        ## motion
        motion_reg = motion_reg[..., :self.ego_fut_ts, :].cumsum(-2)
        motion_reg = cat_with_zero(motion_reg) + det_anchors[:, :, None, None, :2]
        _, motion_mode_idx = torch.topk(motion_cls, num_motion_mode, dim=-1)
        motion_mode_idx = motion_mode_idx[..., None, None].repeat(1, 1, 1, self.ego_fut_ts + 1, 2)
        motion_reg = torch.gather(motion_reg, 2, motion_mode_idx)

        motion_box = motion_reg.new_zeros(motion_reg.shape[:-1] + (7,))
        motion_box[..., [X, Y]] = motion_reg
        motion_box[..., [W, L, H]] = det_anchors[..., None, None, [W, L, H]].exp()
        box_yaw = torch.atan2(
            det_anchors[..., SIN_YAW],
            det_anchors[..., COS_YAW],
        )
        motion_box[..., [YAW]] = get_yaw(motion_reg, box_yaw.unsqueeze(-1))

        filter_mask = det_confidence < score_thresh
        motion_box[filter_mask] = 1e6

        ego_box = ego_box[..., 1:, :]
        motion_box = motion_box[..., 1:, :]

        bs, num_ego_mode, ts, _ = ego_box.shape
        bs, num_anchor, num_motion_mode, ts, _ = motion_box.shape
        ego_box = ego_box[:, None, None].repeat(1, num_anchor, num_motion_mode, 1, 1, 1).flatten(0, -2)
        motion_box = motion_box.unsqueeze(3).repeat(1, 1, 1, num_ego_mode, 1, 1).flatten(0, -2)

        ego_box[0] += offset * torch.cos(ego_box[6])
        ego_box[1] += offset * torch.sin(ego_box[6])
        col = check_collision(ego_box, motion_box)
        col = col.reshape(bs, num_anchor, num_motion_mode, num_ego_mode, ts).permute(0, 3, 1, 2, 4)
        col = col.flatten(2, -1).any(dim=-1)
        all_col = col.all(dim=-1)
        col[all_col] = False # for case that all modes collide, no need to rescore
        score_offset = col.float() * -999
        plan_cls = plan_cls + score_offset
        return plan_cls

    def rescore_soft(
        self,
        plan_cls,
        plan_reg,
        motion_cls,
        motion_reg,
        det_anchors,
        det_confidence,
        score_thresh=0.5,
        static_dis_thresh=0.5,
        dim_scale=1.1,
        num_motion_mode=1,
        offset=0.5,
    ):
        """Soft variant of rescore.

        Replaces the hard `-999` mask with a continuous score offset
        `−w_col · sum_t exp(-clamp(min_sdf, 0)² / σ²)`, where `min_sdf` is the
        SDF from the closest of 4 ego corners to the closest agent's oriented
        bbox at each waypoint. Reuses the same ego/motion box construction as
        `rescore`. No fallback for "all-collide" — soft costs stay bounded so
        relative differences still discriminate modes.
        """

        def cat_with_zero(traj):
            zeros = traj.new_zeros(traj.shape[:-2] + (1, 2))
            return torch.cat([zeros, traj], dim=-2)

        def get_yaw(traj, start_yaw=np.pi / 2):
            yaw = traj.new_zeros(traj.shape[:-1])
            yaw[..., 1:-1] = torch.atan2(
                traj[..., 2:, 1] - traj[..., :-2, 1],
                traj[..., 2:, 0] - traj[..., :-2, 0],
            )
            yaw[..., -1] = torch.atan2(
                traj[..., -1, 1] - traj[..., -2, 1],
                traj[..., -1, 0] - traj[..., -2, 0],
            )
            yaw[..., 0] = start_yaw
            start = traj[..., 0, :]
            end = traj[..., -1, :]
            dist = torch.linalg.norm(end - start, dim=-1)
            mask = dist < static_dis_thresh
            sy = yaw[..., 0].unsqueeze(-1)
            yaw = torch.where(mask.unsqueeze(-1), sy, yaw)
            return yaw.unsqueeze(-1)

        bs = plan_reg.shape[0]
        plan_reg_cat = cat_with_zero(plan_reg)
        ego_box = det_anchors.new_zeros(
            bs, self.ego_fut_mode, self.ego_fut_ts + 1, 7
        )
        ego_box[..., [X, Y]] = plan_reg_cat
        ego_box[..., [W, L, H]] = ego_box.new_tensor([4.08, 1.73, 1.56]) * dim_scale
        ego_box[..., [YAW]] = get_yaw(plan_reg_cat)

        motion_reg_t = motion_reg[..., :self.ego_fut_ts, :].cumsum(-2)
        motion_reg_t = cat_with_zero(motion_reg_t) + det_anchors[:, :, None, None, :2]
        _, motion_mode_idx = torch.topk(motion_cls, num_motion_mode, dim=-1)
        motion_mode_idx = motion_mode_idx[..., None, None].repeat(
            1, 1, 1, self.ego_fut_ts + 1, 2
        )
        motion_reg_t = torch.gather(motion_reg_t, 2, motion_mode_idx)

        motion_box = motion_reg_t.new_zeros(motion_reg_t.shape[:-1] + (7,))
        motion_box[..., [X, Y]] = motion_reg_t
        motion_box[..., [W, L, H]] = det_anchors[..., None, None, [W, L, H]].exp()
        box_yaw = torch.atan2(
            det_anchors[..., SIN_YAW],
            det_anchors[..., COS_YAW],
        )
        motion_box[..., [YAW]] = get_yaw(motion_reg_t, box_yaw.unsqueeze(-1))

        # Drop t=0 (anchor frame, no displacement yet) to align with ego.
        ego_box = ego_box[..., 1:, :]
        motion_box = motion_box[..., 1:, :]

        # Apply forward offset to ego center along heading.
        ego_box = ego_box.clone()
        ego_yaw = ego_box[..., 6]
        ego_box[..., 0] = ego_box[..., 0] + offset * torch.cos(ego_yaw)
        ego_box[..., 1] = ego_box[..., 1] + offset * torch.sin(ego_yaw)

        bs_, num_ego_mode, ts, _ = ego_box.shape
        _, num_anchor, n_mm, _, _ = motion_box.shape

        # Build 4 ego corners in world frame: (..., 4, 2).
        half_W_e = ego_box[..., 3:4] * 0.5  # (bs, M_ego, ts, 1)
        half_L_e = ego_box[..., 4:5] * 0.5
        sign_w = ego_box.new_tensor([1.0, 1.0, -1.0, -1.0]).reshape(1, 1, 1, 4)
        sign_l = ego_box.new_tensor([1.0, -1.0, -1.0, 1.0]).reshape(1, 1, 1, 4)
        cx_local = sign_w * half_W_e
        cy_local = sign_l * half_L_e
        cos_e = torch.cos(ego_yaw).unsqueeze(-1)
        sin_e = torch.sin(ego_yaw).unsqueeze(-1)
        ego_corners_x = cx_local * cos_e - cy_local * sin_e + ego_box[..., 0:1]
        ego_corners_y = cx_local * sin_e + cy_local * cos_e + ego_box[..., 1:2]

        # Broadcast: ego (bs, 1, 1, M_ego, ts, 4); motion (bs, A, MM, 1, ts, 1)
        ex = ego_corners_x[:, None, None, :, :, :]  # (bs, 1, 1, M_ego, ts, 4)
        ey = ego_corners_y[:, None, None, :, :, :]
        m_cx = motion_box[..., 0:1].unsqueeze(3)  # (bs, A, MM, 1, ts, 1)
        m_cy = motion_box[..., 1:2].unsqueeze(3)
        m_yaw = motion_box[..., 6:7].unsqueeze(3)  # (bs, A, MM, 1, ts, 1)
        half_w_a = (motion_box[..., 3:4] * 0.5).unsqueeze(3)
        half_l_a = (motion_box[..., 4:5] * 0.5).unsqueeze(3)

        rel_x = ex - m_cx  # (bs, A, MM, M_ego, ts, 4)
        rel_y = ey - m_cy
        cos_a = torch.cos(m_yaw)
        sin_a = torch.sin(m_yaw)
        local_x = rel_x * cos_a + rel_y * sin_a
        local_y = -rel_x * sin_a + rel_y * cos_a
        qx = local_x.abs() - half_w_a
        qy = local_y.abs() - half_l_a
        outside = torch.sqrt(
            torch.clamp(qx, min=0) ** 2 + torch.clamp(qy, min=0) ** 2 + 1e-12
        )
        inside = torch.clamp(torch.maximum(qx, qy), max=0)
        sdf = outside + inside  # (bs, A, MM, M_ego, ts, 4)
        min_sdf_corners = sdf.min(dim=-1).values  # (bs, A, MM, M_ego, ts)

        # Filter low-conf agents: push their SDF to large positive (negligible cost).
        filter_mask = (det_confidence < score_thresh)  # (bs, A)
        if filter_mask.any():
            big = min_sdf_corners.new_tensor(1e6)
            fm = filter_mask[:, :, None, None, None].expand_as(min_sdf_corners)
            min_sdf_corners = torch.where(fm, big.expand_as(min_sdf_corners), min_sdf_corners)

        # Closest agent (and motion mode) per (ego_mode, t): min over (A, MM).
        min_sdf_per_t = min_sdf_corners.amin(dim=(1, 2))  # (bs, M_ego, ts)
        # Saturating Gaussian on positive distance: 1 inside the box, decays outside.
        sigma2 = float(self.rescore_soft_sigma) ** 2
        pos_dist = torch.clamp(min_sdf_per_t, min=0)
        c_col_t = torch.exp(-(pos_dist ** 2) / sigma2)  # (bs, M_ego, ts)
        c_col = c_col_t.sum(dim=-1)  # (bs, M_ego)

        score_offset = -float(self.rescore_soft_w_col) * c_col
        plan_cls = plan_cls + score_offset
        return plan_cls


    def rescore_learned(
        self,
        plan_cls,
        planning_output,
        det_confidence,
        cmd,
    ):
        """Inference selector backed by a trained per-(anchor, mode) collision
        classifier head (`conflict_logits`).

        For each ego mode, aggregate sigmoid(conflict_logit) across confidence-
        thresholded detection anchors via `max` (closest predicted-collision
        agent). Apply `plan_cls -= w_col · max_agg` per cmd-indexed mode.

        Requires `planning_output['conflict_logits']` to be populated by the
        motion-planning head; falls back to a no-op if absent.
        """
        if planning_output is None:
            return plan_cls
        conflict_logits = planning_output.get("conflict_logits")
        if not conflict_logits:
            return plan_cls

        logits = conflict_logits[-1]  # (bs, num_anchor, M_total = 3 * ego_fut_mode)
        bs, num_anchor, M_total = logits.shape
        M_per_cmd = self.ego_fut_mode
        # Reshape to (bs, A, num_cmd, M_per_cmd) and select cmd-conditional slice.
        logits = logits.reshape(bs, num_anchor, -1, M_per_cmd)
        bs_indices = torch.arange(bs, device=logits.device)
        logits_cmd = logits[bs_indices, :, cmd, :]  # (bs, A, M_per_cmd)

        # Drop low-confidence anchors by setting their logits to a very negative
        # value so sigmoid → 0 and max-aggregation skips them.
        thr = float(self.rescore_learned_score_thresh)
        det_mask = (det_confidence < thr)  # (bs, A)
        if det_mask.any():
            very_neg = logits_cmd.new_tensor(-1e6)
            logits_cmd = torch.where(
                det_mask.unsqueeze(-1).expand_as(logits_cmd),
                very_neg.expand_as(logits_cmd),
                logits_cmd,
            )

        # max over agents per (bs, M_per_cmd), then sigmoid → bounded penalty.
        max_logit_per_mode = logits_cmd.max(dim=1).values  # (bs, M_per_cmd)
        score = torch.sigmoid(max_logit_per_mode)
        offset = -float(self.rescore_learned_w_col) * score
        return plan_cls + offset

    def rescore_learned_hard(
        self,
        plan_cls,
        planning_output,
        det_confidence,
        cmd,
    ):
        """Hard binary rescore backed by the learned conflict head.

        Mirrors ``rescore()`` (line 221) structurally so the learned head can
        plug in as a drop-in replacement of the heuristic feasibility filter:
          * filter low-confidence detection anchors (``det_conf < thr``)
          * threshold sigmoid → per-(anchor, mode) binary collision flag
          * ``any(anchor)`` reduction → per-mode binary
          * all-collide fallback: if every cmd-conditional ego mode collides,
            no rescore is applied (matches rescore line 305-306)
          * apply ``-999`` mask to colliding modes (matches rescore line 307-308)

        Falls back to a no-op if ``planning_output['conflict_logits']`` is absent.
        """
        if planning_output is None:
            return plan_cls
        conflict_logits = planning_output.get("conflict_logits")
        if not conflict_logits:
            return plan_cls

        logits = conflict_logits[-1]  # (bs, num_anchor, M_total = 3 * ego_fut_mode)
        bs, num_anchor, M_total = logits.shape
        M_per_cmd = self.ego_fut_mode
        # Reshape to (bs, A, num_cmd, M_per_cmd) and select cmd-conditional slice.
        logits = logits.reshape(bs, num_anchor, -1, M_per_cmd)
        bs_indices = torch.arange(bs, device=logits.device)
        logits_cmd = logits[bs_indices, :, cmd, :]  # (bs, A, M_per_cmd)

        # Per-(anchor, mode) collision probability.
        prob = torch.sigmoid(logits_cmd)
        prob_thr = float(self.rescore_learned_hard_prob_thresh)
        det_thr = float(self.rescore_learned_hard_score_thresh)
        agg = self.rescore_learned_hard_aggregation

        if agg == 'detweighted':
            # Soft det-confidence weighting on the prob; no hard det-conf gate.
            # `det_confidence` is in [0, 1]; multiplying preserves [0, 1] range.
            prob = prob * det_confidence.unsqueeze(-1)
            col = (prob > prob_thr).any(dim=1)  # (bs, M_per_cmd)
        else:
            # Drop low-confidence anchors: their collision probability is forced
            # to 0 so they cannot trigger the per-mode reduction.
            det_mask = (det_confidence < det_thr)  # (bs, A)
            if det_mask.any():
                zero = prob.new_tensor(0.0)
                prob = torch.where(
                    det_mask.unsqueeze(-1).expand_as(prob),
                    zero.expand_as(prob),
                    prob,
                )
            if agg == 'any':
                # Per-mode binary collision: any anchor exceeds prob threshold.
                col = (prob > prob_thr).any(dim=1)  # (bs, M_per_cmd)
            else:  # 'topk'
                # Require ≥ k anchors above threshold per mode. Robust to
                # isolated single-anchor false positives.
                k = max(1, self.rescore_learned_hard_topk_k)
                count = (prob > prob_thr).sum(dim=1)  # (bs, M_per_cmd)
                col = count >= k

        # All-collide fallback: if every mode collides, no rescore is applied.
        all_col = col.all(dim=-1)  # (bs,)
        col[all_col] = False

        return plan_cls + col.float() * -999.0


def check_collision(boxes1, boxes2):
    '''
        A rough check for collision detection: 
            check if any corner point of boxes1 is inside boxes2 and vice versa.
        
        boxes1: tensor with shape [N, 7], [x, y, z, w, l, h, yaw]
        boxes2: tensor with shape [N, 7]
    '''
    col_1 = corners_in_box(boxes1.clone(), boxes2.clone())
    col_2 = corners_in_box(boxes2.clone(), boxes1.clone())
    collision = torch.logical_or(col_1, col_2)

    return collision

def corners_in_box(boxes1, boxes2):
    if  boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return False

    boxes1_yaw = boxes1[:, 6].clone()
    boxes1_loc = boxes1[:, :3].clone()
    cos_yaw = torch.cos(-boxes1_yaw)
    sin_yaw = torch.sin(-boxes1_yaw)
    rot_mat_T = torch.stack(
        [
            torch.stack([cos_yaw, sin_yaw]),
            torch.stack([-sin_yaw, cos_yaw]),
        ]
    )
    # translate and rotate boxes
    boxes1[:, :3] = boxes1[:, :3] - boxes1_loc
    boxes1[:, :2] = torch.einsum('ij,jki->ik', boxes1[:, :2], rot_mat_T)
    boxes1[:, 6] = boxes1[:, 6] - boxes1_yaw

    boxes2[:, :3] = boxes2[:, :3] - boxes1_loc
    boxes2[:, :2] = torch.einsum('ij,jki->ik', boxes2[:, :2], rot_mat_T)
    boxes2[:, 6] = boxes2[:, 6] - boxes1_yaw

    corners_box2 = box3d_to_corners(boxes2)[:, [0, 3, 7, 4], :2]
    corners_box2 = torch.from_numpy(corners_box2).to(boxes2.device)
    H = boxes1[:, [3]]
    W = boxes1[:, [4]]

    collision = torch.logical_and(
        torch.logical_and(corners_box2[..., 0] <= H / 2, corners_box2[..., 0] >= -H / 2),
        torch.logical_and(corners_box2[..., 1] <= W / 2, corners_box2[..., 1] >= -W / 2),
    )
    collision = collision.any(dim=-1)

    return collision