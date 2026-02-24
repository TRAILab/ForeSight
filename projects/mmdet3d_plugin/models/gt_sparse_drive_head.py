from typing import List

import torch

from mmcv.runner import BaseModule
from mmdet.models import HEADS, build_head

from projects.mmdet3d_plugin.core.box3d import SIN_YAW, COS_YAW, YAW


@HEADS.register_module()
class GTSparseDriveHead(BaseModule):
    """SparseDrive head that uses GT boxes as oracle detection inputs.

    Bypasses the detection transformer and feeds GT bounding boxes
    directly into the motion/planning head for oracle prediction evaluation.

    The det_head config is still required to provide:
      - anchor_encoder (SparseBox3DEncoder)
      - instance_bank (InstanceBank for temporal mask and anchor_handler)
      - sampler (for the indices interface expected by MotionTarget)

    Args:
        task_config: Task flags (with_det, with_map, with_motion_plan).
        det_head:    Config for Sparse4DHead (used for sub-modules only).
        map_head:    Unused; kept for API compatibility.
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
        init_cfg=None,
        **kwargs,
    ):
        super(GTSparseDriveHead, self).__init__(init_cfg)
        self.task_config = task_config
        self.num_classes = num_classes

        assert det_head is not None, (
            "det_head config is required to provide anchor_encoder "
            "and instance_bank sub-modules."
        )
        self.det_head = build_head(det_head)
        # Freeze all det_head parameters: we only use its sub-modules
        # (anchor_encoder, instance_bank) as non-trainable components.
        # This prevents DDP from complaining about unused parameters.
        for p in self.det_head.parameters():
            p.requires_grad_(False)

        assert motion_plan_head is not None
        self.motion_plan_head = build_head(motion_plan_head)

    def init_weights(self):
        self.det_head.init_weights()
        self.motion_plan_head.init_weights()

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #

    def forward(self, feature_maps, metas: dict):
        batch_size = len(metas["img_metas"])

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

        # 4. Forward motion/planning head.
        motion_output, planning_output = self.motion_plan_head(
            det_output,
            None,  # no map output
            feature_maps,
            metas,
            self.det_head.anchor_encoder,
            self.det_head.instance_bank.mask,
            self.det_head.instance_bank.anchor_handler,
        )

        return det_output, None, motion_output, planning_output

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

        # Instance features initialised to zero; the motion GNN refines them.
        instance_feature = torch.zeros(
            batch_size, num_anchor, embed_dims, device=device
        )

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
