from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn

from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmcv.runner import BaseModule, force_fp32
from mmcv.utils import build_from_cfg
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.models import HEADS, LOSSES
from mmdet.core import reduce_mean

from ..blocks import DeformableFeatureAggregation as DFG

__all__ = ["Sparse4DHead"]


@HEADS.register_module()
class Sparse4DHead(BaseModule):
    def __init__(
        self,
        instance_bank: dict,
        anchor_encoder: dict,
        graph_model: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        temp_graph_model: dict = None,
        loss_cls: dict = None,
        loss_reg: dict = None,
        loss_visibility: dict = None,
        decoder: dict = None,
        sampler: dict = None,
        gt_cls_key: str = "gt_labels_3d",
        gt_reg_key: str = "gt_bboxes_3d",
        gt_id_key: str = "instance_id",
        gt_visibility_key: str = "gt_visibility",
        with_instance_id: bool = True,
        task_prefix: str = 'det',
        reg_weights: List = None,
        operation_order: Optional[List[str]] = None,
        cls_threshold_to_reg: float = -1,
        dn_loss_weight: float = 5.0,
        decouple_attn: bool = True,
        temporal_warmup_order: Optional[List[str]] = None,
        warmup_refine_layer: dict = None,
        warmup_ffn: dict = None,
        init_cfg: dict = None,
        **kwargs,
    ):
        super(Sparse4DHead, self).__init__(init_cfg)
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.gt_id_key = gt_id_key
        self.with_instance_id = with_instance_id
        self.task_prefix = task_prefix
        self.cls_threshold_to_reg = cls_threshold_to_reg
        self.dn_loss_weight = dn_loss_weight
        self.decouple_attn = decouple_attn

        if reg_weights is None:
            self.reg_weights = [1.0] * 10
        else:
            self.reg_weights = reg_weights

        if operation_order is None:
            operation_order = [
                "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
            # delete the 'gnn' and 'norm' layers in the first transformer blocks
            operation_order = operation_order[3:]
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.gt_visibility_key = gt_visibility_key
        self.instance_bank = build(instance_bank, PLUGIN_LAYERS)
        self.anchor_encoder = build(anchor_encoder, POSITIONAL_ENCODING)
        self.sampler = build(sampler, BBOX_SAMPLERS)
        self.decoder = build(decoder, BBOX_CODERS)
        self.loss_cls = build(loss_cls, LOSSES)
        self.loss_reg = build(loss_reg, LOSSES)
        self.loss_visibility = build(loss_visibility, LOSSES) if loss_visibility else None
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "deformable": [deformable_model, ATTENTION],
            "refine": [refine_layer, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        self.temporal_warmup_order = list(temporal_warmup_order) if temporal_warmup_order else []
        # For "refine" in warmup, use warmup_refine_layer if provided (should have
        # with_cls_branch=False, with_quality_estimation=False to avoid unused params).
        # Falls back to refine_layer if warmup_refine_layer is not specified.
        warmup_op_config_map = dict(self.op_config_map)
        if warmup_refine_layer is not None:
            warmup_op_config_map["refine"] = [warmup_refine_layer, PLUGIN_LAYERS]
        if warmup_ffn is not None:
            warmup_op_config_map["ffn"] = [warmup_ffn, FEEDFORWARD_NETWORK]
        self.warmup_layers = nn.ModuleList(
            [
                build(*warmup_op_config_map.get(op, [None, None]))
                for op in self.temporal_warmup_order
            ]
        )
        self.embed_dims = self.instance_bank.embed_dims
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
        # Dedicated fc projections for warmup GNN — must NOT share with fc_before/fc_after
        # because gradient checkpointing would fire DDP hooks twice for shared params.
        has_warmup_gnn = any(op == "gnn" for op in self.temporal_warmup_order)
        if has_warmup_gnn and self.decouple_attn:
            self.warmup_fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.warmup_fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else:
            self.warmup_fc_before = nn.Identity()
            self.warmup_fc_after = nn.Identity()

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for i, op in enumerate(self.temporal_warmup_order):
            if self.warmup_layers[i] is None:
                continue
            elif op != "refine":
                for p in self.warmup_layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        if isinstance(self.warmup_fc_before, nn.Linear):
            nn.init.xavier_uniform_(self.warmup_fc_before.weight)
        if isinstance(self.warmup_fc_after, nn.Linear):
            nn.init.xavier_uniform_(self.warmup_fc_after.weight)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def _gnn_with_layer(self, layer, feat, anchor_embed):
        """Self-attention (GNN) using an explicit layer rather than self.layers[i].
        Used by the temporal warmup block where queries attend only to each other.
        Uses warmup_fc_before/after (not shared with main decoder fc projections)."""
        if self.decouple_attn:
            q = torch.cat([feat, anchor_embed], dim=-1)
            v = self.warmup_fc_before(feat)
            return self.warmup_fc_after(layer(q, q, v))
        else:
            v = self.warmup_fc_before(feat)
            return self.warmup_fc_after(layer(feat, feat, v, query_pos=anchor_embed))

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
    ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        # ========= get instance info ============
        if (
            self.sampler.dn_metas is not None
            and self.sampler.dn_metas["dn_anchor"].shape[0] != batch_size
        ):
            self.sampler.dn_metas = None
        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
        ) = self.instance_bank.get(
            batch_size, metas, dn_metas=self.sampler.dn_metas
        )

        # ========= prepare for denosing training ============
        # 1. get dn metas: noisy-anchors and corresponding GT
        # 2. concat learnable instances and noisy instances
        # 3. get attention mask
        attn_mask = None
        dn_metas = None
        temp_dn_reg_target = None
        if self.training and hasattr(self.sampler, "get_dn_anchors"):
            if self.gt_id_key in metas["img_metas"][0]:
                gt_instance_id = [
                    torch.from_numpy(x[self.gt_id_key]).cuda()
                    for x in metas["img_metas"]
                ]
            else:
                gt_instance_id = None
            dn_metas = self.sampler.get_dn_anchors(
                metas[self.gt_cls_key],
                metas[self.gt_reg_key],
                gt_instance_id,
            )
        if dn_metas is not None:
            (
                dn_anchor,
                dn_reg_target,
                dn_cls_target,
                dn_attn_mask,
                valid_mask,
                dn_id_target,
            ) = dn_metas
            num_dn_anchor = dn_anchor.shape[1]
            if dn_anchor.shape[-1] != anchor.shape[-1]:
                remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]
                dn_anchor = torch.cat(
                    [
                        dn_anchor,
                        dn_anchor.new_zeros(
                            batch_size, num_dn_anchor, remain_state_dims
                        ),
                    ],
                    dim=-1,
                )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
            instance_feature = torch.cat(
                [
                    instance_feature,
                    instance_feature.new_zeros(
                        batch_size, num_dn_anchor, instance_feature.shape[-1]
                    ),
                ],
                dim=1,
            )
            num_instance = instance_feature.shape[1]
            num_free_instance = num_instance - num_dn_anchor
            attn_mask = anchor.new_ones(
                (num_instance, num_instance), dtype=torch.bool
            )
            attn_mask[:num_free_instance, :num_free_instance] = False
            attn_mask[num_free_instance:, num_free_instance:] = dn_attn_mask

        anchor_embed = self.anchor_encoder(anchor)
        if temp_anchor is not None:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        # =========== temporal warmup (Block 0) ====================
        # Social self-attention among temporal queries before they are merged
        # with current-frame detections. No image features used here.
        # Always runs (even on first frame) so warmup params always receive
        # gradients — avoids the need for find_unused_parameters=True.
        if self.temporal_warmup_order:
            if temp_instance_feature is not None:
                # Temporal case: warm up the cached temporal features
                w_feat = temp_instance_feature
                w_anchor = temp_anchor
                w_anchor_embed = temp_anchor_embed
                is_temporal = True
            else:
                # First frame: warm up the first num_temp_instances current slots
                num_ti = self.instance_bank.num_temp_instances
                w_feat = instance_feature[:, :num_ti]
                w_anchor = anchor[:, :num_ti]
                w_anchor_embed = anchor_embed[:, :num_ti]
                is_temporal = False
            w_cls, w_qt, w_vis = None, None, None
            for i, op in enumerate(self.temporal_warmup_order):
                if self.warmup_layers[i] is None:
                    continue
                if op == "gnn":
                    w_feat = self._gnn_with_layer(
                        self.warmup_layers[i], w_feat, w_anchor_embed
                    )
                elif op in ("norm", "ffn"):
                    w_feat = self.warmup_layers[i](w_feat)
                elif op == "refine":
                    w_anchor, w_cls, w_qt, w_vis = self.warmup_layers[i](
                        w_feat,
                        w_anchor,
                        w_anchor_embed,
                        time_interval=time_interval,
                        return_cls=True,
                    )
                    w_anchor_embed = self.anchor_encoder(w_anchor)
            if is_temporal:
                temp_instance_feature = w_feat
                temp_anchor = w_anchor
                temp_anchor_embed = w_anchor_embed
            else:
                # Inject warmed first-frame features back so warmup params
                # connect to the loss via the main decoder.
                num_ti = self.instance_bank.num_temp_instances
                instance_feature = torch.cat(
                    [w_feat, instance_feature[:, num_ti:]], dim=1
                )
                anchor = torch.cat([w_anchor, anchor[:, num_ti:]], dim=1)
                anchor_embed = torch.cat(
                    [w_anchor_embed, anchor_embed[:, num_ti:]], dim=1
                )

        # =================== forward the layers ====================
        prediction = []
        classification = []
        quality = []
        visibility = []
        # If warmup produced a refine prediction on temporal instances, prepend it
        # so it gets supervised like any other intermediate decoder stage.
        # Pads non-temporal slots (num_ti:num_anchor) with initial anchor positions
        # and near-zero cls logits so the sampler treats them as background.
        # NOTE: do NOT gate this on is_temporal — the cls branch must always
        # participate in the loss so DDP doesn't see unused parameters on
        # first-frame batches (which have is_temporal=False).
        if (
            self.temporal_warmup_order
            and w_cls is not None
            and dn_metas is None
        ):
            num_ti = self.instance_bank.num_temp_instances
            num_anchor = self.instance_bank.num_anchor
            warmup_pred = torch.cat(
                [w_anchor, anchor[:, num_ti:num_anchor]], dim=1
            )
            warmup_cls = torch.cat(
                [
                    w_cls,
                    w_cls.new_full(
                        [batch_size, num_anchor - num_ti, w_cls.shape[-1]], -10.0
                    ),
                ],
                dim=1,
            )
            warmup_qt = (
                torch.cat(
                    [
                        w_qt,
                        w_qt.new_zeros(batch_size, num_anchor - num_ti, w_qt.shape[-1]),
                    ],
                    dim=1,
                )
                if w_qt is not None
                else None
            )
            warmup_vis = (
                torch.cat(
                    [
                        w_vis,
                        w_vis.new_zeros(batch_size, num_anchor - num_ti, w_vis.shape[-1]),
                    ],
                    dim=1,
                )
                if w_vis is not None
                else None
            )
            prediction.append(warmup_pred)
            classification.append(warmup_cls)
            quality.append(warmup_qt)
            visibility.append(warmup_vis)
        num_main_decoder_refines = 0
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=attn_mask
                    if temp_instance_feature is None
                    else None,
                )
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                )
            elif op == "refine":
                anchor, cls, qt, vis = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=True,
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)
                visibility.append(vis)
                num_main_decoder_refines += 1
                if num_main_decoder_refines == self.num_single_frame_decoder:
                    instance_feature, anchor = self.instance_bank.update(
                        instance_feature, anchor, cls,
                        cached_feature_override=temp_instance_feature,
                        cached_anchor_override=temp_anchor,
                    )
                    if (
                        dn_metas is not None
                        and self.sampler.num_temp_dn_groups > 0
                        and dn_id_target is not None
                    ):
                        (
                            instance_feature,
                            anchor,
                            temp_dn_reg_target,
                            temp_dn_cls_target,
                            temp_valid_mask,
                            dn_id_target,
                        ) = self.sampler.update_dn(
                            instance_feature,
                            anchor,
                            dn_reg_target,
                            dn_cls_target,
                            valid_mask,
                            dn_id_target,
                            self.instance_bank.num_anchor,
                            self.instance_bank.mask,
                        )
                anchor_embed = self.anchor_encoder(anchor)
                if (
                    len(prediction) > self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[
                        :, : self.instance_bank.num_temp_instances
                    ]
            else:
                raise NotImplementedError(f"{op} is not supported.")

        output = {}

        # split predictions of learnable instances and noisy instances
        if dn_metas is not None:
            dn_classification = [
                x[:, num_free_instance:] for x in classification
            ]
            classification = [x[:, :num_free_instance] for x in classification]
            dn_prediction = [x[:, num_free_instance:] for x in prediction]
            prediction = [x[:, :num_free_instance] for x in prediction]
            quality = [
                x[:, :num_free_instance] if x is not None else None
                for x in quality
            ]
            visibility = [
                x[:, :num_free_instance] if x is not None else None
                for x in visibility
            ]
            output.update(
                {
                    "dn_prediction": dn_prediction,
                    "dn_classification": dn_classification,
                    "dn_reg_target": dn_reg_target,
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": valid_mask,
                }
            )
            if temp_dn_reg_target is not None:
                output.update(
                    {
                        "temp_dn_reg_target": temp_dn_reg_target,
                        "temp_dn_cls_target": temp_dn_cls_target,
                        "temp_dn_valid_mask": temp_valid_mask,
                        "dn_id_target": dn_id_target,
                    }
                )
                dn_cls_target = temp_dn_cls_target
                valid_mask = temp_valid_mask
            dn_instance_feature = instance_feature[:, num_free_instance:]
            dn_anchor = anchor[:, num_free_instance:]
            instance_feature = instance_feature[:, :num_free_instance]
            anchor_embed = anchor_embed[:, :num_free_instance]
            anchor = anchor[:, :num_free_instance]
            cls = cls[:, :num_free_instance]

            # cache dn_metas for temporal denoising
            self.sampler.cache_dn(
                dn_instance_feature,
                dn_anchor,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )
        output.update(
            {
                "classification": classification,
                "prediction": prediction,
                "quality": quality,
                "visibility": visibility,
                "instance_feature": instance_feature,
                "anchor_embed": anchor_embed,
            }
        )

        # cache current instances for temporal modeling
        self.instance_bank.cache(
            instance_feature, anchor, cls, metas, feature_maps
        )
        if self.with_instance_id:
            instance_id = self.instance_bank.get_instance_id(
                cls, anchor, self.decoder.score_threshold
            )
            output["instance_id"] = instance_id
        return output

    @force_fp32(apply_to=("model_outs"))
    def loss(self, model_outs, data, feature_maps=None):
        # ===================== prediction losses ======================
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]
        vis_scores = model_outs.get("visibility", [None] * len(cls_scores))
        output = {}
        for decoder_idx, (cls, reg, qt, vis) in enumerate(
            zip(cls_scores, reg_preds, quality, vis_scores)
        ):
            reg = reg[..., : len(self.reg_weights)]
            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls,
                reg,
                data[self.gt_cls_key],
                data[self.gt_reg_key],
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            reg_target_full = reg_target.clone()
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            mask_valid = mask.clone()

            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
            )
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > threshold
                )

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)

            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            cls_target = cls_target[mask]
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]

            reg_loss = self.loss_reg(
                reg,
                reg_target,
                weight=reg_weights,
                avg_factor=num_pos,
                prefix=f"{self.task_prefix}_",
                suffix=f"_{decoder_idx}",
                quality=qt,
                cls_target=cls_target,
            )

            output[f"{self.task_prefix}_loss_cls_{decoder_idx}"] = cls_loss
            output.update(reg_loss)

            # ---- visibility loss (only on matched / positive anchors) ----
            if (
                vis is not None
                and self.loss_visibility is not None
                and self.gt_visibility_key in data
            ):
                gt_vis_list = data[self.gt_visibility_key]
                bs_v, num_pred_v = vis.shape[:2]
                vis_target = vis.new_zeros(bs_v, num_pred_v)
                for b_i, (pred_idx, target_idx) in enumerate(
                    self.sampler.indices
                ):
                    if (
                        pred_idx is not None
                        and len(pred_idx) > 0
                        and len(gt_vis_list[b_i]) > 0
                    ):
                        vis_target[b_i, pred_idx] = (
                            gt_vis_list[b_i]
                            .to(vis.device)
                            .float()[target_idx]
                        )
                matched = mask_valid.reshape(-1)
                vis_loss = self.loss_visibility(
                    vis.squeeze(-1).flatten(end_dim=1)[matched],
                    vis_target.flatten(end_dim=1)[matched],
                    avg_factor=num_pos,
                )
                output[
                    f"{self.task_prefix}_loss_visibility_{decoder_idx}"
                ] = vis_loss

        if "dn_prediction" not in model_outs:
            return output

        # ===================== denoising losses ======================
        dn_cls_scores = model_outs["dn_classification"]
        dn_reg_preds = model_outs["dn_prediction"]

        (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        ) = self.prepare_for_dn_loss(model_outs)
        for decoder_idx, (cls, reg) in enumerate(
            zip(dn_cls_scores, dn_reg_preds)
        ):
            if (
                "temp_dn_valid_mask" in model_outs
                and decoder_idx == self.num_single_frame_decoder
            ):
                (
                    dn_valid_mask,
                    dn_cls_target,
                    dn_reg_target,
                    dn_pos_mask,
                    reg_weights,
                    num_dn_pos,
                ) = self.prepare_for_dn_loss(model_outs, prefix="temp_")

            cls_loss = self.loss_cls(
                cls.flatten(end_dim=1)[dn_valid_mask],
                dn_cls_target,
                avg_factor=num_dn_pos,
            )
            reg_loss = self.loss_reg(
                reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                    ..., : len(self.reg_weights)
                ],
                dn_reg_target,
                avg_factor=num_dn_pos,
                weight=reg_weights,
                prefix=f"{self.task_prefix}_",
                suffix=f"_dn_{decoder_idx}",
            )
            output[f"{self.task_prefix}_loss_cls_dn_{decoder_idx}"] = cls_loss
            output.update(reg_loss)
        return output

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(
            end_dim=1
        )[dn_valid_mask]
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(
            end_dim=1
        )[dn_valid_mask][..., : len(self.reg_weights)]
        dn_pos_mask = dn_cls_target >= 0
        dn_reg_target = dn_reg_target[dn_pos_mask]
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(
            dn_reg_target.shape[0], 1
        )
        num_dn_pos = max(
            reduce_mean(torch.sum(dn_valid_mask).to(dtype=reg_weights.dtype)),
            1.0,
        )
        return (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        )

    @force_fp32(apply_to=("model_outs"))
    def post_process(self, model_outs, output_idx=-1):
        vis_list = model_outs.get("visibility")
        # Only pass visibility to decode() when the head is active (not all-None).
        # This keeps backward-compatibility with decoders that lack the param.
        vis_kwarg = {}
        if vis_list is not None and any(v is not None for v in vis_list):
            vis_kwarg = {"visibility": vis_list}
        return self.decoder.decode(
            model_outs["classification"],
            model_outs["prediction"],
            instance_id=model_outs.get("instance_id"),
            quality=model_outs.get("quality"),
            output_idx=output_idx,
            **vis_kwarg,
        )
