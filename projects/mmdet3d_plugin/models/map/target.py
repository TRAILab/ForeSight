import torch
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mmdet.core.bbox.builder import (BBOX_SAMPLERS, BBOX_ASSIGNERS)
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core import (build_assigner, build_sampler)
from mmdet.core.bbox.assigners import (AssignResult, BaseAssigner)

from ..base_target import BaseTargetWithDenoising


@BBOX_SAMPLERS.register_module()
class SparsePoint3DTarget(BaseTargetWithDenoising):
    def __init__(
        self,
        assigner=None,
        num_dn_groups=0,
        dn_noise_scale=0.5,
        max_dn_gt=32,
        add_neg_dn=True,
        num_temp_dn_groups=0,
        num_cls=3,
        num_sample=20,
        roi_size=(30, 60),
    ):
        super(SparsePoint3DTarget, self).__init__(
            num_dn_groups, num_temp_dn_groups
        )
        self.assigner = build_assigner(assigner)
        self.dn_noise_scale = dn_noise_scale
        self.max_dn_gt = max_dn_gt
        self.add_neg_dn = add_neg_dn

        self.num_cls = num_cls
        self.num_sample = num_sample
        self.roi_size = roi_size

    def sample(
        self,
        cls_preds,
        pts_preds,
        cls_targets,
        pts_targets,
    ):
        pts_targets  = [x.flatten(2, 3) if len(x.shape)==4 else x for x in pts_targets]
        indices = []
        for(cls_pred, pts_pred, cls_target, pts_target) in zip(
            cls_preds, pts_preds, cls_targets, pts_targets
        ):
            # normalize to (0, 1)
            pts_pred = self.normalize_line(pts_pred)
            pts_target = self.normalize_line(pts_target)
            preds=dict(lines=pts_pred, scores=cls_pred)
            gts=dict(lines=pts_target, labels=cls_target)
            indice = self.assigner.assign(preds, gts)
            indices.append(indice)
        
        bs, num_pred, num_cls = cls_preds.shape
        output_cls_target = cls_targets[0].new_ones([bs, num_pred], dtype=torch.long) * num_cls
        output_box_target = pts_preds.new_zeros(pts_preds.shape)
        output_reg_weights = pts_preds.new_zeros(pts_preds.shape)
        for i, (pred_idx, target_idx, gt_permute_index) in enumerate(indices):
            if len(cls_targets[i]) == 0:
                continue
            permute_idx = gt_permute_index[pred_idx, target_idx]
            output_cls_target[i, pred_idx] = cls_targets[i][target_idx]
            output_box_target[i, pred_idx] = pts_targets[i][target_idx, permute_idx]
            output_reg_weights[i, pred_idx] = 1

        return output_cls_target, output_box_target, output_reg_weights

    def get_dn_anchors(self, cls_target, pts_target, gt_instance_id=None):
        """Generate denoising anchors for map polylines.

        cls_target: list of (N_i,) int class tensors per batch item.
        pts_target: list of (N_i, ...) polyline tensors per batch item.
        Returns the same 6-tuple as SparseBox3DTarget.get_dn_anchors, with
        dn_id_target=None (maps have no cross-frame tracking).
        """
        if self.num_dn_groups <= 0:
            return None

        if self.max_dn_gt > 0:
            cls_target = [x[: self.max_dn_gt] for x in cls_target]
            pts_target = [x[: self.max_dn_gt] for x in pts_target]

        max_dn_gt = max([len(x) for x in cls_target])
        if max_dn_gt == 0:
            return None

        bs = len(cls_target)
        state_dims = self.num_sample * 2

        # Pad class targets: (num_dn_groups * bs, max_dn_gt)
        cls_padded = torch.stack([
            F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
            for x in cls_target
        ])  # (bs, max_dn_gt)

        # Flatten GT polylines to (N_i, state_dims); keep raw and normalized copies.
        # Anchors live in [0,1] normalized space; targets must stay in raw BEV space
        # so that SparseLineLoss.normalize_line() is applied exactly once (same as
        # the normal loss path via sample()).
        pts_norm = []
        pts_raw = []
        for i, pts in enumerate(pts_target):
            if len(pts) == 0:
                pts_norm.append(cls_padded.new_zeros(0, state_dims).float())
                pts_raw.append(cls_padded.new_zeros(0, state_dims).float())
                continue
            if pts.dim() == 4:
                pts = pts[:, 0].flatten(-2, -1)
            elif pts.dim() == 3:
                pts = pts.reshape(pts.shape[0], -1)
            pts = pts[:, :state_dims]
            pts_raw.append(pts)
            pts_norm.append(self.normalize_line(pts))

        pts_padded = torch.stack([
            F.pad(x, (0, 0, 0, max_dn_gt - x.shape[0]))
            for x in pts_norm
        ])  # (bs, max_dn_gt, state_dims) in [0,1]
        pts_padded = torch.where(
            cls_padded[..., None] == -1, pts_padded.new_tensor(0.0), pts_padded
        )

        pts_raw_padded = torch.stack([
            F.pad(x, (0, 0, 0, max_dn_gt - x.shape[0]))
            for x in pts_raw
        ])  # (bs, max_dn_gt, state_dims) in raw BEV coords
        pts_raw_padded = torch.where(
            cls_padded[..., None] == -1, pts_raw_padded.new_tensor(0.0), pts_raw_padded
        )

        # Tile for num_dn_groups
        if self.num_dn_groups > 1:
            cls_tiled = cls_padded.tile(self.num_dn_groups, 1)
            pts_tiled = pts_padded.tile(self.num_dn_groups, 1, 1)
            pts_raw_tiled = pts_raw_padded.tile(self.num_dn_groups, 1, 1)
        else:
            cls_tiled = cls_padded
            pts_tiled = pts_padded
            pts_raw_tiled = pts_raw_padded

        # Positive DN: anchors in [0,1]; targets in raw BEV (normalized once by SparseLineLoss)
        noise = (torch.rand_like(pts_tiled) * 2 - 1) * self.dn_noise_scale
        dn_anchor = (pts_tiled + noise).clamp(0.0, 1.0)
        dn_pts_target = pts_raw_tiled.clone()
        dn_cls_target = -torch.ones_like(cls_tiled) * 3

        if self.add_neg_dn:
            noise_neg = (torch.rand_like(pts_tiled) + 1.0) * self.dn_noise_scale
            flag = torch.where(
                torch.rand_like(pts_tiled) > 0.5,
                pts_tiled.new_tensor(1.0),
                pts_tiled.new_tensor(-1.0),
            )
            dn_anchor = torch.cat(
                [dn_anchor, (pts_tiled + noise_neg * flag).clamp(0.0, 1.0)], dim=1
            )
            dn_pts_target = torch.cat([dn_pts_target, pts_raw_tiled.clone()], dim=1)
            dn_cls_target = torch.cat([dn_cls_target, dn_cls_target], dim=1)
            num_gt = max_dn_gt * 2
        else:
            num_gt = max_dn_gt

        # Identity assignment: set positive DN cls from GT, leave neg as -3
        for i in range(dn_anchor.shape[0]):
            n_valid = int((cls_tiled[i] >= 0).sum().item())
            if n_valid == 0:
                continue
            dn_cls_target[i, :n_valid] = cls_tiled[i, :n_valid]

        # Reshape: (num_dn_groups*bs, num_gt, ...) → (bs, num_dn_groups*num_gt, ...)
        dn_anchor = (
            dn_anchor.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3).flatten(1, 2)
        )
        dn_pts_target = (
            dn_pts_target.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3).flatten(1, 2)
        )
        dn_cls_target = (
            dn_cls_target.reshape(self.num_dn_groups, bs, num_gt)
            .permute(1, 0, 2).flatten(1)
        )

        valid_mask = dn_cls_target >= 0
        if self.add_neg_dn:
            orig_cls = (
                torch.cat([cls_tiled, cls_tiled], dim=1)
                .reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2).flatten(1)
            )
            valid_mask = torch.logical_or(
                valid_mask, (orig_cls >= 0) & (dn_cls_target == -3)
            )

        # Attention mask: prevent cross-group attention
        attn_mask = dn_pts_target.new_ones(
            num_gt * self.num_dn_groups, num_gt * self.num_dn_groups,
            dtype=torch.bool,
        )
        for i in range(self.num_dn_groups):
            start = num_gt * i
            end = start + num_gt
            attn_mask[start:end, start:end] = False

        return (dn_anchor, dn_pts_target, dn_cls_target.long(), attn_mask, valid_mask, None)

    def normalize_line(self, line):
        if line.shape[0] == 0:
            return line
        
        line = line.view(line.shape[:-1] + (self.num_sample, -1))
        
        origin = -line.new_tensor([self.roi_size[0]/2, self.roi_size[1]/2])
        line = line - origin

        # transform from range [0, 1] to (0, 1)
        eps = 1e-5
        norm = line.new_tensor([self.roi_size[0], self.roi_size[1]]) + eps
        line = line / norm
        line = line.flatten(-2, -1)

        return line


@BBOX_ASSIGNERS.register_module()
class HungarianLinesAssigner(BaseAssigner):
    """
        Computes one-to-one matching between predictions and ground truth.
        This class computes an assignment between the targets and the predictions
        based on the costs. The costs are weighted sum of three components:
        classification cost and regression L1 cost. The
        targets don't include the no_object, so generally there are more
        predictions than targets. After the one-to-one matching, the un-matched
        are treated as backgrounds. Thus each query prediction will be assigned
        with `0` or a positive integer indicating the ground truth index:
        - 0: negative sample, no assigned gt
        - positive integer: positive sample, index (1-based) of assigned gt
        Args:
            cls_weight (int | float, optional): The scale factor for classification
                cost. Default 1.0.
            bbox_weight (int | float, optional): The scale factor for regression
                L1 cost. Default 1.0.
    """

    def __init__(self, cost=dict, **kwargs):
        self.cost = build_match_cost(cost)

    def assign(self,
               preds: dict,
               gts: dict,
               ignore_cls_cost=False,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """
            Computes one-to-one matching based on the weighted costs.
            This method assign each query prediction to a ground truth or
            background. The `assigned_gt_inds` with -1 means don't care,
            0 means negative sample, and positive number is the index (1-based)
            of assigned gt.
            The assignment is done in the following steps, the order matters.
            1. assign every prediction to -1
            2. compute the weighted costs
            3. do Hungarian matching on CPU based on the costs
            4. assign all to 0 (background) first, then for each matched pair
            between predictions and gts, treat this prediction as foreground
            and assign the corresponding gt index (plus 1) to it.
            Args:
                lines_pred (Tensor): predicted normalized lines:
                    [num_query, num_points, 2]
                cls_pred (Tensor): Predicted classification logits, shape
                    [num_query, num_class].

                lines_gt (Tensor): Ground truth lines
                    [num_gt, num_points, 2].
                labels_gt (Tensor): Label of `gt_bboxes`, shape (num_gt,).
                gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                    labelled as `ignored`. Default None.
                eps (int | float, optional): A value added to the denominator for
                    numerical stability. Default 1e-7.
            Returns:
                :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        
        num_gts, num_lines = gts['lines'].size(0), preds['lines'].size(0)
        if num_gts == 0 or num_lines == 0:
            return None, None, None

        # compute the weighted costs
        gt_permute_idx = None # (num_preds, num_gts)
        if self.cost.reg_cost.permute:
            cost, gt_permute_idx = self.cost(preds, gts, ignore_cls_cost)
        else:
            cost = self.cost(preds, gts, ignore_cls_cost)

        # do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu().numpy()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        return matched_row_inds, matched_col_inds, gt_permute_idx