import torch
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import AssignResult, BaseAssigner
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core import bbox_cxcywh_to_xyxy

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@BBOX_ASSIGNERS.register_module()
class HungarianAssigner2D(BaseAssigner):
    """Hungarian one-to-one matcher for 2D auxiliary supervision.

    Matches predictions to ground-truth boxes using a weighted sum of
    classification, regression L1, IoU, and 2D-centre costs.

    Unlike the StreamPETR variant this class accepts ``image_wh`` (a
    2-element ``[W, H]`` tensor) directly rather than an ``img_meta``
    dict, because ForeSight does not use ``img_metas`` in its training
    pipeline.

    Args:
        cls_cost (dict): Config for classification cost.
        reg_cost (dict): Config for L1 regression cost.
        iou_cost (dict): Config for IoU cost.
        centers2d_cost (dict): Config for 2D-centre L1 cost.
    """

    def __init__(
        self,
        cls_cost=dict(type="FocalLossCost", weight=2.0),
        reg_cost=dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
        iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
        centers2d_cost=dict(type="BBox3DL1Cost", weight=10.0),
    ):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.centers2d_cost = build_match_cost(centers2d_cost)

    def assign(
        self,
        bbox_pred,
        cls_pred,
        pred_centers2d,
        gt_bboxes,
        gt_labels,
        centers2d,
        image_wh,
        gt_bboxes_ignore=None,
        eps=1e-7,
    ):
        """Compute one-to-one assignment.

        Args:
            bbox_pred (Tensor): Predicted boxes in normalised cxcywh,
                shape ``[num_query, 4]``.
            cls_pred (Tensor): Classification logits,
                shape ``[num_query, num_classes]``.
            pred_centers2d (Tensor): Predicted 2D centres (normalised),
                shape ``[num_query, 2]``.
            gt_bboxes (Tensor): GT boxes in pixel xyxy,
                shape ``[num_gt, 4]``.
            gt_labels (Tensor): GT class indices, shape ``[num_gt]``.
            centers2d (Tensor): GT 2D centres in pixel coords,
                shape ``[num_gt, 2]``.
            image_wh (Tensor): ``[W, H]`` of the image.
            gt_bboxes_ignore: Ignored (must be None).

        Returns:
            :obj:`AssignResult`
        """
        assert gt_bboxes_ignore is None, "gt_bboxes_ignore must be None."

        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        assigned_gt_inds = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

        img_w, img_h = image_wh[0], image_wh[1]
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)

        cls_cost = self.cls_cost(cls_pred, gt_labels)
        normalize_gt_bboxes = gt_bboxes / factor
        reg_cost = self.reg_cost(bbox_pred, normalize_gt_bboxes)
        bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
        iou_cost = self.iou_cost(bboxes, gt_bboxes)
        normalize_centers2d = centers2d / factor[:, :2]
        centers2d_cost = self.centers2d_cost(pred_centers2d, normalize_centers2d)

        cost = cls_cost + reg_cost + iou_cost + centers2d_cost
        cost = torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=-100.0)

        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost.detach().cpu())
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bbox_pred.device)

        assigned_gt_inds[:] = 0
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
