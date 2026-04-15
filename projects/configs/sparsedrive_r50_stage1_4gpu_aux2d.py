from sparsedrive_r50_stage1_4gpu import *


log_config["hooks"][1]["init_kwargs"]["name"] = "sparsedrive_r50_stage1_4gpu_aux2d"

model["aux_2d_head"] = dict(
    type="SparseDriveAux2DHead",
    num_classes=num_classes,
    in_channels=embed_dims,
    embed_dims=embed_dims,
    feat_level=0,
    stride=strides[0],
    loss_cls2d=dict(
        type="QualityFocalLoss",
        use_sigmoid=True,
        beta=2.0,
        loss_weight=2.0,
    ),
    loss_centerness=dict(
        type="GaussianFocalLoss",
        reduction="mean",
        loss_weight=1.0,
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
)

adaptor_idx = next(
    idx
    for idx, step in enumerate(train_pipeline)
    if step["type"] == "NuScenesSparse4DAdaptor"
)
train_pipeline.insert(adaptor_idx, dict(type="GenerateProjected2DTargets"))

collect_step = next(step for step in train_pipeline if step["type"] == "Collect")
collect_step["keys"].extend(
    [
        "gt_bboxes",
        "gt_labels",
        "centers2d",
        "depths",
    ]
)
