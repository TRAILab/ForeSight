import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

__all__ = ["InstanceBank"]


def topk(confidence, k, *inputs):
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    indices = (
        indices + torch.arange(bs, device=indices.device)[:, None] * N
    ).reshape(-1)
    outputs = []
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs


@PLUGIN_LAYERS.register_module()
class InstanceBank(nn.Module):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor,
        anchor_handler=None,
        num_temp_instances=0,
        default_time_interval=0.5,
        confidence_decay=0.6,
        anchor_grad=True,
        feat_grad=True,
        max_time_interval=2,
        use_motion_for_anchor_propagation=True,
        motion_step_sec=0.5,
        motion_confidence_threshold=0.0,
    ):
        super(InstanceBank, self).__init__()
        self.embed_dims = embed_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval
        self.use_motion_for_anchor_propagation = use_motion_for_anchor_propagation
        self.motion_step_sec = float(motion_step_sec)
        self.motion_confidence_threshold = float(motion_confidence_threshold)

        if anchor_handler is not None:
            anchor_handler = build_from_cfg(anchor_handler, PLUGIN_LAYERS)
            assert hasattr(anchor_handler, "anchor_projection")
        self.anchor_handler = anchor_handler
        if isinstance(anchor, str):
            anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)
        if len(anchor.shape) == 3: # for map
            anchor = anchor.reshape(anchor.shape[0], -1)
        self.num_anchor = min(len(anchor), num_anchor)
        anchor = anchor[:num_anchor]
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )
        self.reset()

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def reset(self):
        self.cached_feature = None
        self.cached_anchor = None
        self.cached_motion_displacement = None
        self.cached_motion_valid_mask = None
        self.metas = None
        self.mask = None
        self.confidence = None
        self.temp_confidence = None
        self.instance_id = None
        self.prev_id = 0

    def get(self, batch_size, metas=None, dn_metas=None):
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1)
        )
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))

        if (
            self.cached_anchor is not None
            and batch_size == self.cached_anchor.shape[0]
        ):
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(dtype=instance_feature.dtype)
            self.mask = torch.abs(time_interval) <= self.max_time_interval

            if self.anchor_handler is not None:
                T_temp2cur = self.cached_anchor.new_tensor(
                    np.stack(
                        [
                            x["T_global_inv"]
                            @ self.metas["img_metas"][i]["T_global"]
                            for i, x in enumerate(metas["img_metas"])
                        ]
                    )
                )
                proj_kwargs = dict(
                    anchor=self.cached_anchor,
                    T_src2dst_list=[T_temp2cur],
                    time_intervals=[-time_interval],
                )
                if (
                    self.use_motion_for_anchor_propagation
                    and self.cached_motion_displacement is not None
                    and self.cached_motion_valid_mask is not None
                ):
                    motion_disp = self.cached_motion_displacement
                    if self.motion_step_sec > 0:
                        scale = (
                            time_interval.to(motion_disp.dtype) / self.motion_step_sec
                        ).view(-1, 1, 1)
                        scale = torch.clamp(scale, min=0.0)
                        motion_disp = motion_disp * scale
                    proj_kwargs["motion_displacement"] = motion_disp
                    proj_kwargs["motion_valid_mask"] = self.cached_motion_valid_mask
                self.cached_anchor = self.anchor_handler.anchor_projection(
                    **proj_kwargs
                )[0]

            if (
                self.anchor_handler is not None
                and dn_metas is not None
                and batch_size == dn_metas["dn_anchor"].shape[0]
            ):
                num_dn_group, num_dn = dn_metas["dn_anchor"].shape[1:3]
                dn_anchor = self.anchor_handler.anchor_projection(
                    dn_metas["dn_anchor"].flatten(1, 2),
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]
                dn_metas["dn_anchor"] = dn_anchor.reshape(
                    batch_size, num_dn_group, num_dn, -1
                )
            time_interval = torch.where(
                torch.logical_and(time_interval != 0, self.mask),
                time_interval,
                time_interval.new_tensor(self.default_time_interval),
            )
        else:
            self.reset()
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )

        return (
            instance_feature,
            anchor,
            self.cached_feature,
            self.cached_anchor,
            time_interval,
        )

    def update(self, instance_feature, anchor, confidence):
        if self.cached_feature is None:
            return instance_feature, anchor

        num_dn = 0
        if instance_feature.shape[1] > self.num_anchor:
            num_dn = instance_feature.shape[1] - self.num_anchor
            dn_instance_feature = instance_feature[:, -num_dn:]
            dn_anchor = anchor[:, -num_dn:]
            instance_feature = instance_feature[:, : self.num_anchor]
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]

        N = self.num_anchor - self.num_temp_instances
        confidence = confidence.max(dim=-1).values
        _, (selected_feature, selected_anchor) = topk(
            confidence, N, instance_feature, anchor
        )
        selected_feature = torch.cat(
            [self.cached_feature, selected_feature], dim=1
        )
        selected_anchor = torch.cat(
            [self.cached_anchor, selected_anchor], dim=1
        )
        instance_feature = torch.where(
            self.mask[:, None, None], selected_feature, instance_feature
        )
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)
        self.confidence = torch.where(
            self.mask[:, None],
            self.confidence,
            self.confidence.new_tensor(0)
        )
        if self.instance_id is not None:
            self.instance_id = torch.where(
                self.mask[:, None],
                self.instance_id,
                self.instance_id.new_tensor(-1),
            )

        if num_dn > 0:
            instance_feature = torch.cat(
                [instance_feature, dn_instance_feature], dim=1
            )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
        return instance_feature, anchor

    def cache(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
        feature_maps=None,
    ):
        if self.num_temp_instances <= 0:
            return
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach()

        self.metas = metas
        confidence = confidence.max(dim=-1).values.sigmoid()
        if self.confidence is not None:
            confidence[:, : self.num_temp_instances] = torch.maximum(
                self.confidence * self.confidence_decay,
                confidence[:, : self.num_temp_instances],
            )
        self.temp_confidence = confidence

        (
            self.confidence,
            (self.cached_feature, self.cached_anchor),
        ) = topk(confidence, self.num_temp_instances, instance_feature, anchor)

    def set_cached_motion_displacement(self, displacement, mode_confidence=None):
        """
        Set motion-predicted displacement for the next frame's anchor projection.
        displacement: (batch_size, num_agents_with_motion, 2) in lidar/source frame.
        mode_confidence: optional (batch_size, num_agents_with_motion), best-mode score.
        When provided with motion_confidence_threshold > 0, only slots with
        mode_confidence >= threshold use motion; others fall back to velocity.
        """
        if not self.use_motion_for_anchor_propagation:
            return
        if self.cached_anchor is None or displacement is None:
            return
        displacement = displacement.detach()
        if mode_confidence is not None:
            mode_confidence = mode_confidence.detach()
        bs = displacement.shape[0]
        num_motion = displacement.shape[1]
        device = displacement.device
        n = min(num_motion, self.num_temp_instances)
        self.cached_motion_displacement = displacement.new_zeros(
            bs, self.num_temp_instances, 2
        )
        self.cached_motion_displacement[:, :n] = displacement[:, :n]
        self.cached_motion_valid_mask = torch.zeros(
            bs, self.num_temp_instances, dtype=torch.bool, device=device
        )
        self.cached_motion_valid_mask[:, :n] = True
        if (
            mode_confidence is not None
            and self.motion_confidence_threshold > 0
            and mode_confidence.shape[0] == bs
            and mode_confidence.shape[1] >= n
        ):
            low_conf = mode_confidence[:, :n] < self.motion_confidence_threshold
            self.cached_motion_valid_mask[:, :n] = torch.logical_and(
                self.cached_motion_valid_mask[:, :n], ~low_conf
            )

    def get_instance_id(self, confidence, anchor=None, threshold=None):
        confidence = confidence.max(dim=-1).values.sigmoid()
        instance_id = confidence.new_full(confidence.shape, -1).long()

        if (
            self.instance_id is not None
            and self.instance_id.shape[0] == instance_id.shape[0]
        ):
            instance_id[:, : self.instance_id.shape[1]] = self.instance_id

        mask = instance_id < 0
        if threshold is not None:
            mask = mask & (confidence >= threshold)
        num_new_instance = mask.sum()
        new_ids = torch.arange(num_new_instance).to(instance_id) + self.prev_id
        instance_id[torch.where(mask)] = new_ids
        self.prev_id += num_new_instance
        self.update_instance_id(instance_id, confidence)
        return instance_id

    def update_instance_id(self, instance_id=None, confidence=None):
        if self.temp_confidence is None:
            if confidence.dim() == 3:  # bs, num_anchor, num_cls
                temp_conf = confidence.max(dim=-1).values
            else:  # bs, num_anchor
                temp_conf = confidence
        else:
            temp_conf = self.temp_confidence
        instance_id = topk(temp_conf, self.num_temp_instances, instance_id)[1][
            0
        ]
        instance_id = instance_id.squeeze(dim=-1)
        self.instance_id = F.pad(
            instance_id,
            (0, self.num_anchor - self.num_temp_instances),
            value=-1,
        )