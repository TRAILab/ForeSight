from typing import List, Union

import torch

from mmcv.runner import BaseModule
from mmdet.models import HEADS
from mmdet.models import build_head


def _detach_output(output):
    """Detach all tensors in a head output dict (stop-gradient)."""
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


@HEADS.register_module()
class SparseDriveHead(BaseModule):
    def __init__(
        self,
        task_config: dict,
        det_head=dict,
        map_head=dict,
        motion_plan_head=dict,
        init_cfg=None,
        **kwargs,
    ):
        super(SparseDriveHead, self).__init__(init_cfg)
        self.task_config = task_config
        use_gt_det = self.task_config.get('use_gt_det', False)

        if self.task_config['with_det'] or use_gt_det:
            self.det_head = build_head(det_head)
            if use_gt_det and not self.task_config['with_det']:
                for p in self.det_head.parameters():
                    p.requires_grad_(False)

        if self.task_config['with_map'] or (use_gt_det and map_head is not None and map_head != dict):
            self.map_head = build_head(map_head)
            if use_gt_det and not self.task_config['with_map']:
                for p in self.map_head.parameters():
                    p.requires_grad_(False)

        if self.task_config['with_motion_plan']:
            self.motion_plan_head = build_head(motion_plan_head)

        if self.task_config.get('gt_det_warmup_iters', 0) > 0:
            self.register_buffer('_iters_trained', torch.tensor(0, dtype=torch.long))
        # cached per-forward flag so loss() knows which path was taken
        self._used_gt = bool(self.task_config.get('use_gt_det', False))

    def _should_use_gt_det(self):
        warmup_iters = self.task_config.get('gt_det_warmup_iters', 0)
        if warmup_iters > 0:
            return self.training and (self._iters_trained < warmup_iters)
        return self.training and bool(self.task_config.get('use_gt_det', False))

    @property
    def _gt_det_active(self):
        """Whether GT det is active at inference (curriculum always reverts to predicted)."""
        if self.task_config.get('gt_det_warmup_iters', 0) > 0:
            return False
        return bool(self.task_config.get('use_gt_det', False))

    def init_weights(self):
        if hasattr(self, 'det_head'):
            self.det_head.init_weights()
        if hasattr(self, 'map_head'):
            self.map_head.init_weights()
        if self.task_config['with_motion_plan']:
            self.motion_plan_head.init_weights()

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
    ):
        if self.task_config['with_det']:
            det_output = self.det_head(feature_maps, metas)
        else:
            det_output = None

        if self.task_config['with_map']:
            map_output = self.map_head(feature_maps, metas)
        else:
            map_output = None

        if self.task_config['with_motion_plan']:
            use_gt = self._should_use_gt_det()
            self._used_gt = use_gt
            if use_gt:
                batch_size = len(metas['img_metas'])
                device = (feature_maps[0].device
                          if isinstance(feature_maps[0], torch.Tensor)
                          else feature_maps[0][0].device)
                motion_det_input = self.det_head.build_gt_det_output(
                    metas, batch_size, device)
                motion_map_input = (
                    self.map_head.build_gt_map_output(
                        metas, batch_size, device)
                    if hasattr(self, 'map_head') else None
                )
            else:
                detach = self.task_config.get('detach_det', False)
                motion_det_input = _detach_output(det_output) if detach else det_output
                motion_map_input = _detach_output(map_output) if detach else map_output
            if self.training and self.task_config.get('gt_det_warmup_iters', 0) > 0:
                self._iters_trained += 1

            motion_output, planning_output = self.motion_plan_head(
                motion_det_input,
                motion_map_input,
                feature_maps,
                metas,
                self.det_head.anchor_encoder,
                self.det_head.instance_bank.mask,
                self.det_head.instance_bank.anchor_handler,
            )
        else:
            motion_output, planning_output = None, None

        return det_output, map_output, motion_output, planning_output

    def loss(self, model_outs, data):
        det_output, map_output, motion_output, planning_output = model_outs
        losses = dict()
        if self.task_config['with_det']:
            losses.update(self.det_head.loss(det_output, data))
        
        if self.task_config['with_map']:
            losses.update(self.map_head.loss(map_output, data))

        if self.task_config['with_motion_plan']:
            if self._used_gt:
                device = next(self.parameters()).device
                motion_loss_cache = dict(
                    indices=self.motion_plan_head.build_gt_identity_indices(
                        data['gt_labels_3d'], device)
                )
            else:
                motion_loss_cache = dict(indices=self.det_head.sampler.indices)
            losses.update(self.motion_plan_head.loss(
                motion_output, planning_output, data, motion_loss_cache))

        return losses

    def post_process(self, model_outs, data):
        det_output, map_output, motion_output, planning_output = model_outs
        if self.task_config['with_det']:
            det_result = self.det_head.post_process(det_output)
            batch_size = len(det_result)

        if self.task_config['with_map']:
            map_result = self.map_head.post_process(map_output)
            batch_size = len(map_result)

        if self.task_config['with_motion_plan']:
            if self._gt_det_active and not self.task_config['with_det']:
                batch_size = len(data['img_metas']) if 'img_metas' in data else 1
                device = motion_output[0].device if motion_output is not None else 'cpu'
                pp_det_output = self.det_head.build_gt_det_output(
                    data, batch_size, device)
            else:
                pp_det_output = det_output
            motion_result, planning_result = self.motion_plan_head.post_process(
                pp_det_output, motion_output, planning_output, data)

        results = [dict()] * batch_size
        for i in range(batch_size):
            if self.task_config['with_det']:
                results[i].update(det_result[i])
            if self.task_config['with_map']:
                results[i].update(map_result[i])
            if self.task_config['with_motion_plan']:
                results[i].update(motion_result[i])
                results[i].update(planning_result[i])

        return results
