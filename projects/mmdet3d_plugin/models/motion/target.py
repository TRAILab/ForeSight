import torch

from mmdet.core.bbox.builder import BBOX_SAMPLERS

__all__ = ["MotionTarget", "PlanningTarget"]


def get_cls_target(
    reg_preds, 
    reg_target,
    reg_weight,
):
    bs, num_pred, mode, ts, d = reg_preds.shape
    reg_preds_cum = reg_preds.cumsum(dim=-2)
    reg_target_cum = reg_target.cumsum(dim=-2)
    dist = torch.linalg.norm(reg_target_cum.unsqueeze(2) - reg_preds_cum, dim=-1)
    dist = dist * reg_weight.unsqueeze(2)
    dist = dist.mean(dim=-1)
    mode_idx = torch.argmin(dist, dim=-1)
    return mode_idx

def get_best_reg(
    reg_preds, 
    reg_target,
    reg_weight,
):
    bs, num_pred, mode, ts, d = reg_preds.shape
    reg_preds_cum = reg_preds.cumsum(dim=-2)
    reg_target_cum = reg_target.cumsum(dim=-2)
    dist = torch.linalg.norm(reg_target_cum.unsqueeze(2) - reg_preds_cum, dim=-1)
    dist = dist * reg_weight.unsqueeze(2)
    dist = dist.mean(dim=-1)
    mode_idx = torch.argmin(dist, dim=-1)
    mode_idx = mode_idx[..., None, None, None].repeat(1, 1, 1, ts, d)
    best_reg = torch.gather(reg_preds, 2, mode_idx).squeeze(2)
    return best_reg


def _displacement_weight_scale(reg_target, cfg):
    """Per-sample regression-loss scaling by GT endpoint displacement.

    Stationary / short-trajectory samples get a smaller weight (`min_w`),
    long-trajectory samples (turns, fast motion) get a larger weight (`max_w`).
    Mitigates the standard nuScenes data-skew issue where the L1 trajectory
    loss is dominated by trivial stationary/straight cases.

    `cfg` keys:
        min_w (default 0.5) — floor on the weight scale.
        max_w (default 2.0) — ceiling on the weight scale.
        scale_m (default 10.0) — endpoint-displacement value (meters) that
            maps to weight scale 1.0. Lower → more aggressive upscaling of
            small motions; higher → only large motions get full weight.

    Returns a tensor of shape (bs, num_anchor) (or (bs,) for ego) to broadcast
    over the time axis when multiplied into `reg_weight`.
    """
    min_w = float(cfg.get('min_w', 0.5))
    max_w = float(cfg.get('max_w', 2.0))
    scale_m = float(cfg.get('scale_m', 10.0))
    # GT is per-step displacement deltas; cumsum + endpoint norm = total path length proxy.
    endpoint_disp = reg_target.cumsum(dim=-2)[..., -1, :].norm(dim=-1)
    return (endpoint_disp / scale_m).clamp(min=min_w, max=max_w)


@BBOX_SAMPLERS.register_module()
class MotionTarget():
    def __init__(
        self,
        displacement_weight=None,
    ):
        super(MotionTarget, self).__init__()
        # See `_displacement_weight_scale` for cfg keys. None disables.
        self.displacement_weight = displacement_weight

    def sample(
        self,
        reg_pred,
        gt_reg_target,
        gt_reg_mask,
        motion_loss_cache,
    ):
        bs, num_anchor, mode, ts, d = reg_pred.shape
        reg_target = reg_pred.new_zeros((bs, num_anchor, ts, d))
        reg_weight = reg_pred.new_zeros((bs, num_anchor, ts))
        indices = motion_loss_cache['indices']
        num_pos = reg_pred.new_tensor([0])
        for i, (pred_idx, target_idx) in enumerate(indices):
            if len(gt_reg_target[i]) == 0:
                continue
            reg_target[i, pred_idx] = gt_reg_target[i][target_idx]
            reg_weight[i, pred_idx] = gt_reg_mask[i][target_idx]
            num_pos += len(pred_idx)

        cls_target = get_cls_target(reg_pred, reg_target, reg_weight)
        cls_weight = reg_weight.any(dim=-1)
        best_reg = get_best_reg(reg_pred, reg_target, reg_weight)

        # Optional displacement-based reg_weight scaling. Applied after cls
        # target/best_reg computation so mode selection is unchanged — only
        # the final regression loss magnitude is rebalanced toward harder
        # (longer) trajectories.
        if self.displacement_weight is not None:
            w_scale = _displacement_weight_scale(reg_target, self.displacement_weight)
            reg_weight = reg_weight * w_scale.unsqueeze(-1)

        return cls_target, cls_weight, best_reg, reg_target, reg_weight, num_pos


@BBOX_SAMPLERS.register_module()
class PlanningTarget():
    def __init__(
        self,
        ego_fut_ts,
        ego_fut_mode,
        num_driving_cmds=3,
        displacement_weight=None,
    ):
        super(PlanningTarget, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.num_driving_cmds = num_driving_cmds
        # See `_displacement_weight_scale` for cfg keys. None disables.
        self.displacement_weight = displacement_weight

    def sample(
        self,
        cls_pred,
        reg_pred,
        gt_reg_target,
        gt_reg_mask,
        data,
    ):
        gt_reg_target = gt_reg_target.unsqueeze(1)
        gt_reg_mask = gt_reg_mask.unsqueeze(1)

        bs = reg_pred.shape[0]
        bs_indices = torch.arange(bs, device=reg_pred.device)
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)

        cls_pred = cls_pred.reshape(bs, self.num_driving_cmds, 1, self.ego_fut_mode)
        reg_pred = reg_pred.reshape(bs, self.num_driving_cmds, 1, self.ego_fut_mode, self.ego_fut_ts, 2)
        cls_pred = cls_pred[bs_indices, cmd]
        reg_pred = reg_pred[bs_indices, cmd]
        cls_target = get_cls_target(reg_pred, gt_reg_target, gt_reg_mask)
        cls_weight = gt_reg_mask.any(dim=-1)
        best_reg = get_best_reg(reg_pred, gt_reg_target, gt_reg_mask)

        # Optional displacement-based reg_weight scaling. After cls/best_reg.
        if self.displacement_weight is not None:
            w_scale = _displacement_weight_scale(gt_reg_target, self.displacement_weight)
            gt_reg_mask = gt_reg_mask * w_scale.unsqueeze(-1)

        return cls_pred, cls_target, cls_weight, best_reg, gt_reg_target, gt_reg_mask


