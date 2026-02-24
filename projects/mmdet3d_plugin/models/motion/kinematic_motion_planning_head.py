import torch

from mmcv.runner import BaseModule, force_fp32
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.core import reduce_mean
from mmdet.core.bbox.builder import BBOX_SAMPLERS, BBOX_CODERS
from mmdet.models import HEADS, build_loss

from projects.mmdet3d_plugin.core.box3d import VX, VY


@HEADS.register_module()
class KinematicMotionPlanningHead(BaseModule):
    """Heuristic kinematic baseline for motion and ego-planning prediction.

    Implements the CTRA (Constant Turn Rate and Acceleration) kinematic model.
    Special cases are selected via config flags:

        use_acceleration  use_turn_rate   Model
        ──────────────────────────────────────
        False             False           CV   (Constant Velocity)
        True              False           CA   (Constant Acceleration)
        False             True            CVTR (Constant Velocity + Turn Rate)
        True              True            CTRA (Constant Turn Rate + Acceleration)

    No prediction parameters are trained.  Trajectories are derived entirely
    from the GT box state stored in ``det_output["prediction"][-1]`` and, when
    acceleration/turn-rate estimation is enabled, the previous-frame anchor
    history in ``instance_queue.anchor_queue``.

    CTRA integration
    ----------------
    Given scalar speed ``s₀``, heading ``θ₀``, turn rate ``ω`` and longitudinal
    acceleration ``a`` (all estimated at the current frame), each future
    timestep delta is computed via midpoint integration::

        s_mid(t) = s₀ + a · (t + ½) · dt        (clipped to ≥ 0)
        θ_mid(t) = θ₀ + ω · (t + ½) · dt
        Δx_t     = s_mid(t) · cos(θ_mid(t)) · dt
        Δy_t     = s_mid(t) · sin(θ_mid(t)) · dt

    where ``t`` is zero-indexed over ``fut_ts`` steps.

    When ``anchor_queue`` is ``None`` (first frame of a sequence), acceleration
    and turn-rate estimates are unavailable and the model falls back to CV.

    Interface
    ---------
    Signature-compatible with ``MotionPlanningHead``; swap via config
    ``motion_plan_head.type``.

    Args:
        fut_ts:            Agent future timesteps.
        fut_mode:          Number of trajectory hypothesis modes.
        ego_fut_ts:        Ego future timesteps.
        ego_fut_mode:      Number of ego hypothesis modes.
        dt:                Seconds per timestep (0.5 s for nuScenes).
        use_acceleration:  Estimate longitudinal acceleration from the
                           velocity delta between current and previous frame.
        use_turn_rate:     Estimate yaw rate from the heading delta between
                           current and previous frame.
        instance_queue:    Config for InstanceQueue (anchor history tracking).
        motion_sampler / motion_loss_{cls,reg}: Motion loss modules.
        planning_sampler / plan_loss_{cls,reg,status}: Planning loss modules.
        motion_decoder / planning_decoder: Decoder configs.
        num_det / num_map: API compatibility (unused).
        **kwargs:          Absorbs unused ``MotionPlanningHead`` params so that
                           the same config dict works with just a type change.
    """

    def __init__(
        self,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=6,
        dt=0.5,
        use_acceleration=False,
        use_turn_rate=False,
        instance_queue=None,
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
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.dt = dt
        self.use_acceleration = use_acceleration
        self.use_turn_rate = use_turn_rate
        self.num_det = num_det
        self.num_map = num_map

        def _build(cfg, registry):
            return build_from_cfg(cfg, registry) if cfg is not None else None

        self.instance_queue = _build(instance_queue, PLUGIN_LAYERS)
        if self.instance_queue is not None:
            # ego_feature_encoder is unused in kinematic mode; freeze to avoid
            # DDP "unused parameter" errors.
            for p in self.instance_queue.ego_feature_encoder.parameters():
                p.requires_grad_(False)

        self.motion_sampler = _build(motion_sampler, BBOX_SAMPLERS)
        self.planning_sampler = _build(planning_sampler, BBOX_SAMPLERS)
        self.motion_decoder = _build(motion_decoder, BBOX_CODERS)
        self.planning_decoder = _build(planning_decoder, BBOX_CODERS)

        self.motion_loss_cls = build_loss(motion_loss_cls)
        self.motion_loss_reg = build_loss(motion_loss_reg)
        self.plan_loss_cls = build_loss(plan_loss_cls)
        self.plan_loss_reg = build_loss(plan_loss_reg)
        self.plan_loss_status = build_loss(plan_loss_status)

    def init_weights(self):
        if self.instance_queue is not None:
            for m in self.instance_queue.modules():
                if hasattr(m, "init_weight"):
                    m.init_weight()

    # ------------------------------------------------------------------ #
    #  Kinematic helpers
    # ------------------------------------------------------------------ #

    def _ctra_integrate(self, speed, heading, accel, omega, fut_ts, dt, device):
        """Midpoint-rule CTRA integration.

        Args:
            speed:   (bs, N) or (bs,) — current scalar speed.
            heading: (bs, N) or (bs,) — current heading in radians.
            accel:   same shape — longitudinal acceleration (m/s²).
            omega:   same shape — yaw rate (rad/s).
            fut_ts:  number of future timesteps.
            dt:      seconds per timestep.

        Returns:
            pred: (*shape, fut_ts, 2) per-step XY deltas.
        """
        shape = speed.shape
        t = torch.arange(fut_ts, device=device, dtype=speed.dtype)   # (fut_ts,)
        t_mid = t + 0.5                                                # midpoint

        # Broadcast: (*shape, 1) × (fut_ts,) → (*shape, fut_ts)
        s_mid = (speed.unsqueeze(-1) + accel.unsqueeze(-1) * t_mid * dt).clamp(min=0)
        h_mid = heading.unsqueeze(-1) + omega.unsqueeze(-1) * t_mid * dt

        dx = s_mid * torch.cos(h_mid) * dt   # (*shape, fut_ts)
        dy = s_mid * torch.sin(h_mid) * dt
        return torch.stack([dx, dy], dim=-1)  # (*shape, fut_ts, 2)

    def _agent_kinematics(self, gt_anchors):
        """Return (speed, heading, accel, omega) for each agent anchor.

        Falls back to zero accel/omega if anchor_queue is unavailable (first
        frame) or if the corresponding flag is disabled.

        Args:
            gt_anchors: (bs, N, 11) encoded GT anchors.

        Returns:
            speed, heading, accel, omega — each (bs, N).
        """
        vx = gt_anchors[..., VX]
        vy = gt_anchors[..., VY]
        speed   = torch.sqrt(vx ** 2 + vy ** 2).clamp(min=1e-6)
        heading = torch.atan2(vy, vx)

        queue = self.instance_queue.anchor_queue   # (bs, queue_len, N, 11) or None
        have_history = queue is not None

        if self.use_acceleration and have_history:
            prev = queue[:, -1]                     # (bs, N, 11)
            prev_speed = torch.sqrt(
                prev[..., VX] ** 2 + prev[..., VY] ** 2
            ).clamp(min=1e-6)
            accel = (speed - prev_speed) / self.dt
        else:
            accel = torch.zeros_like(speed)

        if self.use_turn_rate and have_history:
            prev = queue[:, -1]
            prev_heading = torch.atan2(prev[..., VY], prev[..., VX])
            omega = (heading - prev_heading) / self.dt
        else:
            omega = torch.zeros_like(heading)

        return speed, heading, accel, omega

    def _ego_kinematics(self, ego_status, device):
        """Return (speed, heading, accel, omega) for the ego vehicle.

        Args:
            ego_status: (bs, 9) — index 6/7 = vx/vy in current frame.

        Returns:
            speed, heading, accel, omega — each (bs,).
        """
        vx = ego_status[:, 6]
        vy = ego_status[:, 7]
        speed   = torch.sqrt(vx ** 2 + vy ** 2).clamp(min=1e-6)
        heading = torch.atan2(vy, vx)

        ego_queue = self.instance_queue.ego_anchor_queue  # (bs, queue_len, 4) or None
        have_history = ego_queue is not None

        if self.use_acceleration and have_history:
            # ego_anchor_queue stores [x, y, cos_h, sin_h] per step; use
            # the heading difference to derive speed change as a proxy.
            prev_ego = ego_queue[:, -1]                   # (bs, 4)
            prev_vx  = prev_ego[:, 0]                     # stored as vx in slot 0
            prev_vy  = prev_ego[:, 1]                     # stored as vy in slot 1
            prev_speed = torch.sqrt(prev_vx ** 2 + prev_vy ** 2).clamp(min=1e-6)
            accel = (speed - prev_speed) / self.dt
        else:
            accel = torch.zeros_like(speed)

        if self.use_turn_rate and have_history:
            prev_ego = ego_queue[:, -1]
            prev_heading = torch.atan2(prev_ego[:, 1], prev_ego[:, 0])
            omega = (heading - prev_heading) / self.dt
        else:
            omega = torch.zeros_like(heading)

        return speed, heading, accel, omega

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #

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
        bs = len(metas["img_metas"])
        gt_anchors = det_output["prediction"][-1]   # (bs, num_anchor, 11)
        num_anchor  = gt_anchors.shape[1]
        device      = gt_anchors.device

        # Update instance_queue for anchor history tracking.
        ego_feature, ego_anchor, _, _, _ = self.instance_queue.get(
            det_output, feature_maps, metas, bs, mask, anchor_handler
        )

        # -------- agent motion -------------------------------------------- #
        speed, heading, accel, omega = self._agent_kinematics(gt_anchors)
        # pred: (bs, N, fut_ts, 2) → expand modes → (bs, N, fut_mode, fut_ts, 2)
        pred = self._ctra_integrate(speed, heading, accel, omega,
                                    self.fut_ts, self.dt, device)
        motion_pred = (
            pred[:, :, None, :, :]
            .expand(bs, num_anchor, self.fut_mode, self.fut_ts, 2)
            .contiguous()
        )
        motion_cls = gt_anchors.new_zeros(bs, num_anchor, self.fut_mode)

        # -------- ego planning -------------------------------------------- #
        ego_status = metas["ego_status"]   # (bs, 9)
        e_speed, e_heading, e_accel, e_omega = self._ego_kinematics(
            ego_status, device
        )
        # pred: (bs, ego_fut_ts, 2) → expand → (bs, 3*ego_fut_mode, ego_fut_ts, 2)
        ego_pred = self._ctra_integrate(e_speed, e_heading, e_accel, e_omega,
                                        self.ego_fut_ts, self.dt, device)
        plan_pred = (
            ego_pred[:, None, :, :]
            .expand(bs, 3 * self.ego_fut_mode, self.ego_fut_ts, 2)
            .contiguous()
        )
        plan_cls    = gt_anchors.new_zeros(bs, 3 * self.ego_fut_mode)
        plan_status = ego_status.unsqueeze(1)   # (bs, 1, 9)

        # -------- update queue state -------------------------------------- #
        zero_feats = gt_anchors.new_zeros(
            bs, num_anchor, det_output["instance_feature"].shape[-1]
        )
        self.instance_queue.cache_motion(zero_feats, det_output, metas)
        self.instance_queue.cache_planning(ego_feature, plan_status)

        motion_output = {
            "classification": [motion_cls],
            "prediction":     [motion_pred],
            "period":         self.instance_queue.period,
            "anchor_queue":   self.instance_queue.anchor_queue,
        }
        planning_output = {
            "classification": [plan_cls],
            "prediction":     [plan_pred],
            "status":         [plan_status],
            "period":         self.instance_queue.ego_period,
            "anchor_queue":   self.instance_queue.ego_anchor_queue,
        }
        return motion_output, planning_output

    # ------------------------------------------------------------------ #
    #  Loss  (computed for monitoring; no params are optimised)
    # ------------------------------------------------------------------ #

    @force_fp32(apply_to=("model_outs",))
    def loss(self, motion_model_outs, planning_model_outs, data, motion_loss_cache):
        loss = {}
        loss.update(self.loss_motion(motion_model_outs, data, motion_loss_cache))
        loss.update(self.loss_planning(planning_model_outs, data))
        return loss

    @force_fp32(apply_to=("model_outs",))
    def loss_motion(self, model_outs, data, motion_loss_cache):
        output = {}
        for i, (cls, reg) in enumerate(zip(
            model_outs["classification"], model_outs["prediction"]
        )):
            cls_target, cls_weight, reg_pred, reg_target, reg_weight, num_pos = (
                self.motion_sampler.sample(
                    reg,
                    data["gt_agent_fut_trajs"],
                    data["gt_agent_fut_masks"],
                    motion_loss_cache,
                )
            )
            num_pos = max(reduce_mean(num_pos), 1.0)

            cls_loss = self.motion_loss_cls(
                cls.flatten(end_dim=1),
                cls_target.flatten(end_dim=1),
                weight=cls_weight.flatten(end_dim=1),
                avg_factor=num_pos,
            )
            reg_pred   = reg_pred.flatten(end_dim=1).cumsum(dim=-2)
            reg_target = reg_target.flatten(end_dim=1).cumsum(dim=-2)
            reg_loss = self.motion_loss_reg(
                reg_pred, reg_target,
                weight=reg_weight.flatten(end_dim=1).unsqueeze(-1),
                avg_factor=num_pos,
            )
            output[f"motion_loss_cls_{i}"] = cls_loss
            output[f"motion_loss_reg_{i}"] = reg_loss
        return output

    @force_fp32(apply_to=("model_outs",))
    def loss_planning(self, model_outs, data):
        output = {}
        for i, (cls, reg, status) in enumerate(zip(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs["status"],
        )):
            cls, cls_target, cls_weight, reg_pred, reg_target, reg_weight = (
                self.planning_sampler.sample(
                    cls, reg,
                    data["gt_ego_fut_trajs"],
                    data["gt_ego_fut_masks"],
                    data,
                )
            )
            cls_loss = self.plan_loss_cls(
                cls.flatten(end_dim=1),
                cls_target.flatten(end_dim=1),
                weight=cls_weight.flatten(end_dim=1),
            )
            reg_loss = self.plan_loss_reg(
                reg_pred.flatten(end_dim=1),
                reg_target.flatten(end_dim=1),
                weight=reg_weight.flatten(end_dim=1).unsqueeze(-1),
            )
            status_loss = self.plan_loss_status(
                status.squeeze(1), data["ego_status"]
            )
            output[f"planning_loss_cls_{i}"]    = cls_loss
            output[f"planning_loss_reg_{i}"]    = reg_loss
            output[f"planning_loss_status_{i}"] = status_loss
        return output

    # ------------------------------------------------------------------ #
    #  Post-process
    # ------------------------------------------------------------------ #

    @force_fp32(apply_to=("model_outs",))
    def post_process(self, det_output, motion_output, planning_output, data):
        motion_result = self.motion_decoder.decode(
            det_output["classification"],
            det_output["prediction"],
            det_output.get("instance_id"),
            det_output.get("quality"),
            motion_output,
        )
        planning_result = self.planning_decoder.decode(
            det_output, motion_output, planning_output, data
        )
        return motion_result, planning_result
