from typing import List

import torch
from torch import nn

from mmcv.runner import BaseModule
from mmdet.models import HEADS, build_head

from projects.mmdet3d_plugin.core.box3d import SIN_YAW, COS_YAW, YAW


@HEADS.register_module()
class GTSparseDriveHead(BaseModule):
    """SparseDrive head that uses GT boxes and optionally GT map as oracle inputs.

    Bypasses the detection transformer and feeds GT bounding boxes (and
    optionally GT map polylines) directly into the motion/planning head
    for oracle prediction evaluation.

    The det_head config is still required to provide:
      - anchor_encoder (SparseBox3DEncoder)
      - instance_bank (InstanceBank for temporal mask and anchor_handler)
      - sampler (for the indices interface expected by MotionTarget)

    If map_head config is provided, its anchor_encoder and instance_bank
    are used to encode GT map polylines into the map_output structure
    expected by MotionPlanningHead (enables cross_gnn map attention).

    Args:
        task_config: Task flags (with_det, with_map, with_motion_plan).
        det_head:    Config for Sparse4DHead (used for sub-modules only).
        map_head:    Config for map Sparse4DHead (used for sub-modules only).
                     If provided, GT map is injected into motion/planning head.
        motion_plan_head: Config for MotionPlanningHead.
        num_classes: Number of detection classes.
    """

    def __init__(
        self,
        task_config: dict,
        det_head: dict = None,
        map_head: dict = None,
        motion_plan_head: dict = None,
        num_classes: int = 10,
        num_map_classes: int = 3,
        instance_feature_init: str = "zero",
        init_cfg=None,
        **kwargs,
    ):
        super(GTSparseDriveHead, self).__init__(init_cfg)
        self.task_config = task_config
        self.num_classes = num_classes
        # `instance_feature_init`:
        #   - "zero": classic zero-init (legacy GT oracle behavior)
        #   - "class_embed": learnable per-class embedding looked up by GT class
        assert instance_feature_init in ("zero", "class_embed"), (
            f"instance_feature_init={instance_feature_init!r} not recognised"
        )
        self.instance_feature_init = instance_feature_init

        assert det_head is not None, (
            "det_head config is required to provide anchor_encoder "
            "and instance_bank sub-modules."
        )
        self.det_head = build_head(det_head)
        # Freeze det_head's bypassed sub-modules (transformer, refine,
        # fc_before/after, warmup_layers) so DDP doesn't see unused params.
        # `anchor_encoder` IS consumed every forward to encode GT boxes;
        # unfreeze it so it can adapt the box embedding for motion
        # downstream (the goal is best offline prediction, not a pure
        # GT-perception oracle).
        for p in self.det_head.parameters():
            p.requires_grad_(False)
        for p in self.det_head.anchor_encoder.parameters():
            p.requires_grad_(True)

        self.num_map_classes = num_map_classes
        if map_head is not None:
            self.map_head = build_head(map_head)
            # Same pattern as det_head: freeze bypassed transformer/refine
            # modules, leave anchor_encoder trainable so it can adapt the
            # polyline embedding for motion's cross_gnn.
            for p in self.map_head.parameters():
                p.requires_grad_(False)
            for p in self.map_head.anchor_encoder.parameters():
                p.requires_grad_(True)

        assert motion_plan_head is not None
        self.motion_plan_head = build_head(motion_plan_head)

        if self.instance_feature_init == "class_embed":
            embed_dims = self.det_head.instance_bank.embed_dims
            # Per-class learnable embedding; trainable (the rest of det_head
            # is frozen, but this module is owned by GTSparseDriveHead).
            self.gt_class_embed = nn.Embedding(num_classes, embed_dims)
            nn.init.normal_(self.gt_class_embed.weight, std=0.02)

    def init_weights(self):
        self.det_head.init_weights()
        if hasattr(self, "map_head"):
            self.map_head.init_weights()
        self.motion_plan_head.init_weights()

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #

    def forward(self, feature_maps, metas: dict):
        batch_size = len(metas["img_metas"])
        device = (
            feature_maps[0].device
            if isinstance(feature_maps[0], torch.Tensor)
            else feature_maps[0][0].device
        )

        # 1. Update instance_bank temporal mask.
        #    We discard the returned bank features; we only need self.mask.
        self.det_head.instance_bank.get(batch_size, metas)

        # 2. Build det_output from GT boxes.
        det_output = self._build_gt_det_output(metas, batch_size, feature_maps)

        # 3. Cache GT features/anchors in the bank for temporal tracking.
        self.det_head.instance_bank.cache(
            det_output["instance_feature"],
            det_output["prediction"][-1],
            det_output["classification"][-1],
            metas,
            feature_maps,
        )

        # 4. Build GT map_output if map_head sub-modules are available *and*
        #    the batch carries map GT. The test pipeline drops gt_map_* keys
        #    (use_map=False), so eval falls through to map_output=None which
        #    MotionPlanningHead's cross_gnn step already handles.
        if hasattr(self, "map_head") and "gt_map_labels" in metas:
            map_output = self._build_gt_map_output(metas, batch_size, device)
        else:
            map_output = None

        # 5. Forward motion/planning head.
        motion_output, planning_output = self.motion_plan_head(
            det_output,
            map_output,
            feature_maps,
            metas,
            self.det_head.anchor_encoder,
            self.det_head.instance_bank.mask,
            self.det_head.instance_bank.anchor_handler,
        )

        return det_output, map_output, motion_output, planning_output

    # ------------------------------------------------------------------ #
    #  GT det_output construction helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _encode_gt_boxes(boxes: torch.Tensor) -> torch.Tensor:
        """Convert decoded 9-dim GT boxes to encoded 11-dim anchor format.

        Input:  (N, >=7) [..., x, y, z, w, l, h, yaw, vx, vy]
        Output: (N, 11)  [x, y, z, log_w, log_l, log_h,
                          sin_yaw, cos_yaw, vx, vy, 0]
        """
        xyz = boxes[:, :3]
        wlh = boxes[:, 3:6].clamp(min=1e-3).log()
        sin_yaw = torch.sin(boxes[:, YAW : YAW + 1])
        cos_yaw = torch.cos(boxes[:, YAW : YAW + 1])
        vel = boxes[:, 7:9] if boxes.shape[-1] >= 9 else boxes.new_zeros(len(boxes), 2)
        vz = boxes.new_zeros(len(boxes), 1)
        return torch.cat([xyz, wlh, sin_yaw, cos_yaw, vel, vz], dim=-1)

    def _build_gt_det_output(
        self, metas: dict, batch_size: int, feature_maps
    ) -> dict:
        """Build the det_output dict populated with GT boxes."""
        if isinstance(feature_maps[0], torch.Tensor):
            device = feature_maps[0].device
        else:
            device = feature_maps[0][0].device

        num_anchor = self.det_head.instance_bank.num_anchor
        embed_dims = self.det_head.instance_bank.embed_dims

        gt_bboxes = metas["gt_bboxes_3d"]   # list[Tensor(N_i, 9)]
        gt_labels = metas["gt_labels_3d"]   # list[Tensor(N_i,)]

        anchors = torch.zeros(batch_size, num_anchor, 11, device=device)
        # Very-negative logits → near-zero confidence after sigmoid (padding).
        cls_logits = anchors.new_full(
            (batch_size, num_anchor, self.num_classes), -100.0
        )

        for i in range(batch_size):
            bboxes_i = gt_bboxes[i]
            labels_i = gt_labels[i]

            if not isinstance(bboxes_i, torch.Tensor):
                bboxes_i = torch.tensor(
                    bboxes_i, device=device, dtype=torch.float32
                )
            else:
                bboxes_i = bboxes_i.to(device=device, dtype=torch.float32)

            if not isinstance(labels_i, torch.Tensor):
                labels_i = torch.tensor(
                    labels_i, device=device, dtype=torch.long
                )
            else:
                labels_i = labels_i.to(device=device)

            N_i = len(bboxes_i)
            if N_i == 0:
                continue
            N_i = min(N_i, num_anchor)
            bboxes_i = bboxes_i[:N_i]
            labels_i = labels_i[:N_i]

            anchors[i, :N_i] = self._encode_gt_boxes(bboxes_i)

            # High-positive logit at GT class, very-negative elsewhere.
            cls_logits[i, :N_i] = -100.0
            cls_logits[i, torch.arange(N_i, device=device), labels_i] = 100.0

        # Anchor embeddings from the det_head's encoder.
        anchor_embed = self.det_head.anchor_encoder(anchors)

        # Instance feature init.
        instance_feature = torch.zeros(
            batch_size, num_anchor, embed_dims, device=device
        )
        if self.instance_feature_init == "class_embed":
            # Look up per-class embedding for valid GT slots; padding stays zero.
            for i in range(batch_size):
                labels_i = gt_labels[i]
                if not isinstance(labels_i, torch.Tensor):
                    labels_i = torch.tensor(labels_i, device=device, dtype=torch.long)
                else:
                    labels_i = labels_i.to(device=device, dtype=torch.long)
                N_i = min(len(labels_i), num_anchor)
                if N_i == 0:
                    continue
                instance_feature[i, :N_i] = self.gt_class_embed(labels_i[:N_i])

        # GT instance IDs for temporal tracking in InstanceQueue.
        instance_id = self._get_gt_instance_ids(
            metas, batch_size, num_anchor, device
        )

        return {
            "instance_feature": instance_feature,
            "anchor_embed": anchor_embed,
            "classification": [cls_logits],
            "prediction": [anchors],
            "quality": [None],
            "instance_id": instance_id,
        }

    def _build_gt_map_output(
        self, metas: dict, batch_size: int, device
    ) -> dict:
        """Build the map_output dict populated with GT map polylines.

        GT map pts are encoded via the map_head's anchor_encoder into the
        same anchor_embed space that MotionPlanningHead's cross_gnn expects.
        Classification logits are set to +100 at the GT class so that the
        confidence-based top-k selection in MotionPlanningHead picks the
        true map elements.

        Args:
            metas: Batch metas containing 'gt_map_labels' and 'gt_map_pts'.
                   gt_map_labels: list[Tensor(M_i,)]
                   gt_map_pts:    list[Tensor(M_i, num_sample, 2)] (test)
                                  or list[Tensor(M_i, num_perms, num_sample, 2)]
                                  (train, when VectorizeMap has permute=True).
        """
        num_anchor = self.map_head.instance_bank.num_anchor  # 100
        embed_dims = self.map_head.instance_bank.embed_dims  # 256
        num_sample_x2 = self.map_head.anchor_encoder.input_dims  # num_sample * 2

        gt_map_labels = metas["gt_map_labels"]  # list[Tensor(M_i,)]
        gt_map_pts = metas["gt_map_pts"]        # list[Tensor(...)]

        predictions = torch.zeros(
            batch_size, num_anchor, num_sample_x2, device=device
        )
        cls_logits = torch.full(
            (batch_size, num_anchor, self.num_map_classes), -100.0, device=device
        )

        for i in range(batch_size):
            map_pts_i = gt_map_pts[i]
            map_labels_i = gt_map_labels[i]

            if not isinstance(map_pts_i, torch.Tensor):
                map_pts_i = torch.tensor(
                    map_pts_i, device=device, dtype=torch.float32
                )
            else:
                map_pts_i = map_pts_i.to(device=device, dtype=torch.float32)

            if not isinstance(map_labels_i, torch.Tensor):
                map_labels_i = torch.tensor(
                    map_labels_i, device=device, dtype=torch.long
                )
            else:
                map_labels_i = map_labels_i.to(device=device)

            M_i = len(map_pts_i)
            if M_i == 0:
                continue
            M_i = min(M_i, num_anchor)
            map_pts_i = map_pts_i[:M_i]
            map_labels_i = map_labels_i[:M_i]

            # Train pipeline uses permute=True: (M, num_perms, num_sample, 2).
            # Take the first permutation (canonical polyline direction).
            if map_pts_i.dim() == 4:
                map_pts_i = map_pts_i[:, 0]  # (M, num_sample, 2)

            predictions[i, :M_i] = map_pts_i.reshape(M_i, -1)
            cls_logits[i, :M_i] = -100.0
            cls_logits[i, torch.arange(M_i, device=device), map_labels_i] = 100.0

        # Encode GT map point coordinates into positional embeddings.
        anchor_embed = self.map_head.anchor_encoder(predictions)

        # Instance features initialised from anchor_embed so motion's
        # cross_gnn receives non-zero V content. With instance_feature=0
        # the attention output is `softmax(Q·Kᵀ) @ 0 = const_bias` —
        # the per-polyline information is lost. Copying anchor_embed into
        # the content channel preserves the polyline shape encoding in V.
        instance_feature = anchor_embed.clone()

        return {
            "instance_feature": instance_feature,
            "anchor_embed": anchor_embed,
            "classification": [cls_logits],
            "prediction": [predictions],
        }

    @staticmethod
    def _get_gt_instance_ids(
        metas: dict,
        batch_size: int,
        num_anchor: int,
        device,
    ) -> torch.Tensor:
        """Read GT instance IDs from metas, padded to (bs, num_anchor)."""
        instance_id = torch.full(
            (batch_size, num_anchor), -1, dtype=torch.long, device=device
        )
        for i, img_meta in enumerate(metas["img_metas"]):
            gt_id = img_meta.get("instance_id", None)
            if gt_id is None:
                continue
            if not isinstance(gt_id, torch.Tensor):
                gt_id = torch.tensor(gt_id, dtype=torch.long, device=device)
            else:
                gt_id = gt_id.to(device=device)
            N_i = min(len(gt_id), num_anchor)
            instance_id[i, :N_i] = gt_id[:N_i]
        return instance_id

    # ------------------------------------------------------------------ #
    #  Loss
    # ------------------------------------------------------------------ #

    def loss(self, model_outs, data):
        _, _, motion_output, planning_output = model_outs

        motion_loss_cache = dict(
            indices=self._build_identity_indices(data),
        )
        return self.motion_plan_head.loss(
            motion_output, planning_output, data, motion_loss_cache
        )

    def _build_identity_indices(self, data) -> List:
        """Identity matching: pred anchor i → GT agent i for each batch item.

        The motion sampler uses these indices to assign GT future trajectories
        to predicted agent slots. With GT-as-detection, agent i directly
        corresponds to GT agent i, so both pred_idx and target_idx are [0..N_i).
        """
        gt_labels = data["gt_labels_3d"]   # list of (N_i,) tensors
        device = next(self.parameters()).device
        indices = []
        for i in range(len(gt_labels)):
            N_i = len(gt_labels[i])
            if N_i == 0:
                indices.append([None, None])
            else:
                idx = torch.arange(N_i, device=device, dtype=torch.long)
                indices.append([idx, idx])
        return indices

    # ------------------------------------------------------------------ #
    #  Post-process
    # ------------------------------------------------------------------ #

    def post_process(self, model_outs, data):
        det_output, _, motion_output, planning_output = model_outs

        det_result = self.det_head.post_process(det_output)
        motion_result, planning_result = self.motion_plan_head.post_process(
            det_output, motion_output, planning_output, data
        )

        batch_size = len(motion_result)
        results = [dict() for _ in range(batch_size)]
        for i in range(batch_size):
            results[i].update(det_result[i])
            results[i].update(motion_result[i])
            results[i].update(planning_result[i])

        return results
