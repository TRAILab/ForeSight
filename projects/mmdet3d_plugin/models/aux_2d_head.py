import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob
from mmcv.runner import force_fp32
from mmdet.core import (
    bbox_cxcywh_to_xyxy,
    bbox_xyxy_to_cxcywh,
    bbox_overlaps,
    multi_apply,
    reduce_mean,
)
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def apply_ltrb(locations, pred_ltrb):
    pred_boxes = torch.zeros_like(pred_ltrb)
    pred_boxes[..., 0] = locations[..., 0] - pred_ltrb[..., 0]
    pred_boxes[..., 1] = locations[..., 1] - pred_ltrb[..., 1]
    pred_boxes[..., 2] = locations[..., 0] + pred_ltrb[..., 2]
    pred_boxes[..., 3] = locations[..., 1] + pred_ltrb[..., 3]
    pred_boxes = pred_boxes.clamp(min=0.0, max=1.0)
    return bbox_xyxy_to_cxcywh(pred_boxes)


def apply_center_offset(locations, center_offset):
    centers_2d = torch.zeros_like(center_offset)
    locations = inverse_sigmoid(locations)
    centers_2d[..., 0] = locations[..., 0] + center_offset[..., 0]
    centers_2d[..., 1] = locations[..., 1] + center_offset[..., 1]
    return centers_2d.sigmoid()


def gaussian_2d(shape, sigma=1.0):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = torch.meshgrid(
        torch.arange(-m, m + 1),
        torch.arange(-n, n + 1),
    )
    h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6).to(
        heatmap.device
    )

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    if left < 0 or right <= 0 or top < 0 or bottom <= 0:
        return heatmap

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def clip_sigmoid(x, eps=1e-4):
    return x.sigmoid().clamp(min=eps, max=1 - eps)


@HEADS.register_module()
class SparseDriveAux2DHead(AnchorFreeHead):
    def __init__(
        self,
        num_classes,
        in_channels=256,
        embed_dims=256,
        feat_level=0,
        stride=4,
        sync_cls_avg_factor=False,
        loss_cls2d=dict(
            type="QualityFocalLoss",
            use_sigmoid=True,
            beta=2.0,
            loss_weight=2.0,
        ),
        loss_centerness=dict(
            type="GaussianFocalLoss", reduction="mean", loss_weight=1.0
        ),
        loss_bbox2d=dict(type="L1Loss", loss_weight=5.0),
        loss_iou2d=dict(type="GIoULoss", loss_weight=2.0),
        loss_centers2d=dict(type="L1Loss", loss_weight=10.0),
        train_cfg=dict(
            assigner2d=dict(
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
                centers2d_cost=dict(weight=10.0),
            )
        ),
        init_cfg=None,
        **kwargs,
    ):
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.feat_level = feat_level
        self.stride = stride
        self.train_cfg = train_cfg
        self.fp16_enabled = False
        super(SparseDriveAux2DHead, self).__init__(
            num_classes, in_channels, init_cfg=init_cfg
        )

        assigner_cfg = train_cfg["assigner2d"]
        self.cls_cost = build_match_cost(assigner_cfg["cls_cost"])
        self.reg_cost = build_match_cost(assigner_cfg["reg_cost"])
        self.iou_cost = build_match_cost(assigner_cfg["iou_cost"])
        self.centers2d_cost_weight = assigner_cfg["centers2d_cost"].get("weight", 1.0)

        self.loss_cls2d = build_loss(loss_cls2d)
        self.loss_bbox2d = build_loss(loss_bbox2d)
        self.loss_iou2d = build_loss(loss_iou2d)
        self.loss_centers2d = build_loss(loss_centers2d)
        self.loss_centerness = build_loss(loss_centerness)
        self.cls_out_channels = num_classes

        self._init_layers()

    def _init_layers(self):
        self.cls = nn.Conv2d(self.embed_dims, self.num_classes, kernel_size=1)
        self.shared_reg = nn.Sequential(
            nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=3, padding=1),
            nn.GroupNorm(32, num_channels=self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.shared_cls = nn.Sequential(
            nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=3, padding=1),
            nn.GroupNorm(32, num_channels=self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.centerness = nn.Conv2d(self.embed_dims, 1, kernel_size=1)
        self.ltrb = nn.Conv2d(self.embed_dims, 4, kernel_size=1)
        self.center2d = nn.Conv2d(self.embed_dims, 2, kernel_size=1)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls.bias, bias_init)
        nn.init.constant_(self.centerness.bias, bias_init)

    def _get_image_wh(self, image_wh, device):
        if isinstance(image_wh, torch.Tensor):
            image_wh = image_wh.reshape(-1, 2)[0].to(device=device, dtype=torch.float32)
        else:
            image_wh = torch.as_tensor(image_wh, device=device, dtype=torch.float32).reshape(-1, 2)[0]
        return image_wh

    def _locations(self, features, image_wh):
        h, w = features.size()[-2:]
        device = features.device
        img_w, img_h = image_wh[0], image_wh[1]
        shifts_x = (torch.arange(0, self.stride * w, step=self.stride, device=device, dtype=torch.float32) + self.stride / 2) / img_w
        shifts_y = (torch.arange(0, self.stride * h, step=self.stride, device=device, dtype=torch.float32) + self.stride / 2) / img_h
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        return torch.stack((shift_x, shift_y), dim=-1)

    def forward(self, feature_maps, image_wh):
        if isinstance(feature_maps, (list, tuple)):
            src = feature_maps[self.feat_level]
        else:
            src = feature_maps
        bs, num_cams, _, _, _ = src.shape
        x = src.flatten(0, 1)
        image_wh = self._get_image_wh(image_wh, x.device)
        locations = self._locations(x, image_wh)[None]

        cls_feat = self.shared_cls(x)
        cls_logits = self.cls(cls_feat).permute(0, 2, 3, 1).reshape(
            bs * num_cams, -1, self.num_classes
        )
        centerness = self.centerness(cls_feat).permute(0, 2, 3, 1).reshape(
            bs * num_cams, -1, 1
        )

        reg_feat = self.shared_reg(x)
        ltrb = self.ltrb(reg_feat).permute(0, 2, 3, 1).contiguous().sigmoid()
        centers2d_offset = self.center2d(reg_feat).permute(0, 2, 3, 1).contiguous()
        pred_centers2d = apply_center_offset(locations, centers2d_offset).view(
            bs * num_cams, -1, 2
        )
        pred_bboxes = apply_ltrb(locations, ltrb).view(bs * num_cams, -1, 4)

        return dict(
            cls_scores=cls_logits,
            bbox_preds=pred_bboxes,
            pred_centers2d=pred_centers2d,
            centerness=centerness,
            image_wh=image_wh,
        )

    @force_fp32(apply_to=("preds_dicts",))
    def loss(
        self,
        gt_bboxes_list,
        gt_labels_list,
        centers2d,
        depths,
        preds_dicts,
    ):
        cls_scores = preds_dicts["cls_scores"]
        bbox_preds = preds_dicts["bbox_preds"]
        pred_centers2d = preds_dicts["pred_centers2d"]
        centerness = preds_dicts["centerness"]
        image_wh = preds_dicts["image_wh"]

        gt_bboxes_list = self._flatten_view_targets(gt_bboxes_list, cls_scores.device)
        gt_labels_list = self._flatten_view_targets(gt_labels_list, cls_scores.device)
        centers2d = self._flatten_view_targets(centers2d, cls_scores.device)
        depths = self._flatten_view_targets(depths, cls_scores.device)

        loss_cls, loss_bbox, loss_iou, loss_centers2d, loss_centerness = self.loss_single(
            cls_scores,
            bbox_preds,
            pred_centers2d,
            centerness,
            gt_bboxes_list,
            gt_labels_list,
            centers2d,
            depths,
            image_wh,
        )
        return dict(
            loss_cls2d=loss_cls,
            loss_bbox2d=loss_bbox,
            loss_iou2d=loss_iou,
            loss_centers2d=loss_centers2d,
            loss_centerness2d=loss_centerness,
        )

    def loss_single(
        self,
        cls_scores,
        bbox_preds,
        pred_centers2d,
        centerness,
        gt_bboxes_list,
        gt_labels_list,
        centers2d_list,
        depths_list,
        image_wh,
    ):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        centers2d_preds_list = [pred_centers2d[i] for i in range(num_imgs)]

        targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            centers2d_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            centers2d_list,
            depths_list,
            image_wh,
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            centers2d_targets_list,
            num_total_pos,
            num_total_neg,
        ) = targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        centers2d_targets = torch.cat(centers2d_targets_list, 0)

        img_w, img_h = image_wh[0], image_wh[1]
        factor = bbox_preds.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        bbox_preds_flat = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds_flat) * factor
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factor

        iou_score = bbox_overlaps(bboxes_gt, bboxes, is_aligned=True).reshape(-1)

        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls2d(
            cls_scores,
            (labels, iou_score.detach()),
            label_weights,
            avg_factor=cls_avg_factor,
        )

        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        loss_iou = self.loss_iou2d(
            bboxes, bboxes_gt, bbox_weights, avg_factor=max(num_total_pos, 1)
        )

        heatmaps = [
            self._get_heatmap_single(cur_centers, cur_boxes, image_wh, centerness.device)
            for cur_centers, cur_boxes in zip(centers2d_list, gt_bboxes_list)
        ]
        heatmaps = torch.stack(heatmaps, dim=0).view(num_imgs, -1, 1)
        loss_centerness = self.loss_centerness(
            clip_sigmoid(centerness),
            heatmaps,
            avg_factor=max(num_total_pos, 1),
        )

        loss_bbox = self.loss_bbox2d(
            bbox_preds_flat,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_pos,
        )
        loss_centers2d = self.loss_centers2d(
            pred_centers2d.view(-1, 2),
            centers2d_targets,
            bbox_weights[:, :2],
            avg_factor=num_total_pos,
        )
        return loss_cls, loss_bbox, loss_iou, loss_centers2d, loss_centerness

    def _get_heatmap_single(self, obj_centers2d, obj_bboxes, image_wh, device):
        img_w, img_h = image_wh[0], image_wh[1]
        heatmap = torch.zeros(
            int(img_h.item() // self.stride),
            int(img_w.item() // self.stride),
            device=device,
        )
        if len(obj_centers2d) == 0:
            return heatmap
        l = obj_centers2d[..., 0:1] - obj_bboxes[..., 0:1]
        t = obj_centers2d[..., 1:2] - obj_bboxes[..., 1:2]
        r = obj_bboxes[..., 2:3] - obj_centers2d[..., 0:1]
        b = obj_bboxes[..., 3:4] - obj_centers2d[..., 1:2]
        bound = torch.cat([l, t, r, b], dim=-1)
        radius = torch.ceil(torch.min(bound, dim=-1)[0] / self.stride)
        radius = torch.clamp(radius, min=1.0).tolist()
        for center, cur_radius in zip(obj_centers2d, radius):
            heatmap = draw_heatmap_gaussian(
                heatmap,
                center / self.stride,
                radius=int(cur_radius),
                k=1,
            )
        return heatmap

    def get_targets(
        self,
        cls_scores_list,
        bbox_preds_list,
        centers2d_preds_list,
        gt_bboxes_list,
        gt_labels_list,
        centers2d_list,
        depths_list,
        image_wh,
    ):
        outputs = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            centers2d_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            centers2d_list,
            depths_list,
            image_wh=image_wh,
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            centers2d_targets_list,
            pos_inds_list,
            neg_inds_list,
        ) = outputs
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            centers2d_targets_list,
            num_total_pos,
            num_total_neg,
        )

    def _get_target_single(
        self,
        cls_score,
        bbox_pred,
        pred_centers2d,
        gt_bboxes,
        gt_labels,
        centers2d,
        depths,
        image_wh,
    ):
        del depths
        num_bboxes = bbox_pred.size(0)
        matched_row_inds, matched_col_inds = self.assign(
            bbox_pred,
            cls_score,
            pred_centers2d,
            gt_bboxes,
            gt_labels,
            centers2d,
            image_wh,
        )

        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        label_weights = gt_bboxes.new_ones(num_bboxes)
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        centers2d_targets = bbox_pred.new_zeros((num_bboxes, 2))

        assigned_gt_inds = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_gt_inds[:] = 0
        if matched_row_inds.numel() > 0:
            assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
            labels[matched_row_inds] = gt_labels[matched_col_inds].long()
            bbox_weights[matched_row_inds] = 1.0

            img_w, img_h = image_wh[0], image_wh[1]
            factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
            pos_gt_bboxes_normalized = gt_bboxes[matched_col_inds] / factor
            bbox_targets[matched_row_inds] = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
            centers2d_targets[matched_row_inds] = centers2d[matched_col_inds] / factor[:, :2]

        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1)
        neg_inds = torch.nonzero(assigned_gt_inds == 0, as_tuple=False).squeeze(-1)
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            centers2d_targets,
            pos_inds,
            neg_inds,
        )

    def assign(
        self,
        bbox_pred,
        cls_pred,
        pred_centers2d,
        gt_bboxes,
        gt_labels,
        centers2d,
        image_wh,
    ):
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)
        if num_gts == 0 or num_bboxes == 0:
            empty = bbox_pred.new_zeros((0,), dtype=torch.long)
            return empty, empty
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" to install scipy first.')

        img_w, img_h = image_wh[0], image_wh[1]
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        reg_cost = self.reg_cost(bbox_pred, gt_bboxes / factor)
        bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
        iou_cost = self.iou_cost(bboxes, gt_bboxes)
        centers2d_cost = torch.cdist(
            pred_centers2d, centers2d / factor[:, :2], p=1
        ) * self.centers2d_cost_weight
        cost = cls_cost + reg_cost + iou_cost + centers2d_cost
        cost = torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=-100.0)
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost.detach().cpu())
        return (
            torch.from_numpy(matched_row_inds).to(bbox_pred.device),
            torch.from_numpy(matched_col_inds).to(bbox_pred.device),
        )

    def _flatten_view_targets(self, targets, device):
        flat_targets = []
        for sample in targets:
            if isinstance(sample, (list, tuple)):
                views = sample
            else:
                views = [sample]
            for view in views:
                flat_targets.append(view.to(device=device))
        return flat_targets
