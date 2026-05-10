from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn

from mmcv.runner import BaseModule
from mmdet.models import HEADS
from mmdet.models import build_head


@HEADS.register_module()
class SparseDriveHead(BaseModule):
    def __init__(
        self,
        task_config: dict,
        det_head = dict,
        map_head = dict,
        motion_plan_head = dict,
        init_cfg=None,
        eval_skip_map: bool = False,
        **kwargs,
    ):
        super(SparseDriveHead, self).__init__(init_cfg)
        self.task_config = task_config
        # Planning-only eval optimization: when motion_plan_head doesn't
        # consume map outputs (num_map=0 or skip_perception_kv=True), the
        # map_head's forward + post_process are pure waste at eval time.
        # Setting this True skips both. Default False to keep training and
        # any planning-with-map eval untouched.
        self.eval_skip_map = bool(eval_skip_map)
        if self.task_config['with_det']:
            self.det_head = build_head(det_head)
        if self.task_config['with_map']:
            self.map_head = build_head(map_head)
        if self.task_config['with_motion_plan']:
            self.motion_plan_head = build_head(motion_plan_head)

    def init_weights(self):
        if self.task_config['with_det']:
            self.det_head.init_weights()
        if self.task_config['with_map']:
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

        skip_map_now = self.eval_skip_map and not self.training
        if self.task_config['with_map'] and not skip_map_now:
            map_output = self.map_head(feature_maps, metas)
        else:
            map_output = None
        
        if self.task_config['with_motion_plan']:
            motion_output, planning_output = self.motion_plan_head(
                det_output,
                map_output,
                feature_maps,
                metas,
                self.det_head.anchor_encoder,
                self.det_head.instance_bank.mask,
                self.det_head.instance_bank.anchor_handler,
            )
            # Cache mode-aggregated motion feature + predicted endpoint into
            # the det InstanceBank so the next frame's TPD can cross-attend
            # to forward-looking context (not just last-frame det state).
            if motion_output is not None and self.task_config.get('with_det', True):
                self.det_head.instance_bank.cache_motion(
                    motion_output.get('motion_feature'),
                    motion_output.get('motion_endpoint'),
                )
        else:
            motion_output, planning_output = None, None

        return det_output, map_output, motion_output, planning_output

    def loss(self, model_outs, data):
        det_output, map_output, motion_output, planning_output = model_outs
        losses = dict()
        if self.task_config['with_det']:
            loss_det = self.det_head.loss(det_output, data)
            losses.update(loss_det)
        
        if self.task_config['with_map']:
            loss_map = self.map_head.loss(map_output, data)
            losses.update(loss_map)

        if self.task_config['with_motion_plan']:
            motion_loss_cache = dict(
                indices=self.det_head.sampler.indices, 
            )
            loss_motion = self.motion_plan_head.loss(
                motion_output,
                planning_output,
                data,
                motion_loss_cache,
                det_output=det_output,
            )
            losses.update(loss_motion)
        
        return losses

    def post_process(self, model_outs, data):
        det_output, map_output, motion_output, planning_output = model_outs
        skip_map_now = self.eval_skip_map and not self.training
        if self.task_config['with_det']:
            det_result = self.det_head.post_process(det_output)
            batch_size = len(det_result)

        if self.task_config['with_map'] and not skip_map_now:
            map_result = self.map_head.post_process(map_output)
            batch_size = len(map_result)

        if self.task_config['with_motion_plan']:
            motion_result, planning_result = self.motion_plan_head.post_process(
                det_output,
                motion_output,
                planning_output,
                data,
            )

        results = [dict()] * batch_size
        for i in range(batch_size):
            if self.task_config['with_det']:
                results[i].update(det_result[i])
            if self.task_config['with_map'] and not skip_map_now:
                results[i].update(map_result[i])
            if self.task_config['with_motion_plan']:
                results[i].update(motion_result[i])
                results[i].update(planning_result[i])

        return results
