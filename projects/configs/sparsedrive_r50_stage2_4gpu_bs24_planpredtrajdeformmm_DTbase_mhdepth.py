# ================ base config ===================
version = 'mini'
version = 'trainval'
length = {'trainval': 28130, 'mini': 323}

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = None

total_batch_size = 24
num_gpus = 4
batch_size = total_batch_size // num_gpus
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
num_epochs = 10
checkpoint_epoch_interval = 10

checkpoint_config = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval
)
log_config = dict(
    interval=51,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type='WandbLoggerHook',
            init_kwargs=dict(
                entity='trailab',
                project='ForeSight',
                name='sparsedrive_r50_stage2_4gpu_bs24_planpredtrajdeformmm_DTbase_mhdepth',),
            interval=50)
    ],
)
load_from = None
resume_from = None
workflow = [("train", 1)]
fp16 = dict(loss_scale=32.0)
input_shape = (704, 256)


# ================== model ========================
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
map_class_names = [
    'ped_crossing',
    'divider',
    'boundary',
]
num_classes = len(class_names)
num_map_classes = len(map_class_names)
roi_size = (30, 60)

num_sample = 20
fut_ts = 12
fut_mode = 6
ego_fut_ts = 6
ego_fut_mode = 6
queue_length = 4 # history + current

embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
num_single_frame_decoder_map = 1
motion_num_heads = 16
motion_decoder_repeats = 6
use_deformable_func = True  # mmdet3d_plugin/ops/setup.py needs to be executed
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
drop_out = 0.1
temporal = True
temporal_map = True
decouple_attn = True
decouple_attn_map = False
decouple_attn_motion = True
with_quality_estimation = True

task_config = dict(
    with_det=True,
    with_map=True,
    with_motion_plan=True,
)

model = dict(
    type="SparseDrive",
    use_grid_mask=True,
    use_deformable_func=use_deformable_func,
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        pretrained="ckpt/resnet50-19c8e357.pth",
    ),
    img_neck=dict(
        type="FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    depth_branch=dict(  # for auxiliary supervision only
        type="DenseDepthNet",
        embed_dims=embed_dims,
        num_depth_layers=num_depth_layers,
        loss_weight=0.2,
    ),
    head=dict(
        type="SparseDriveHead",
        task_config=task_config,
        det_head=dict(
            type="Sparse4DHead",
            cls_threshold_to_reg=0.05,
            decouple_attn=decouple_attn,
            instance_bank=dict(
                type="InstanceBank",
                num_anchor=900,
                embed_dims=embed_dims,
                anchor="data/kmeans/kmeans_det_900.npy",
                anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
                num_temp_instances=600 if temporal else -1,
                confidence_decay=0.6,
                feat_grad=False,
            ),
            anchor_encoder=dict(
                type="SparseBox3DEncoder",
                vel_dims=3,
                embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
                mode="cat" if decouple_attn else "add",
                output_fc=not decouple_attn,
                in_loops=1,
                out_loops=4 if decouple_attn else 2,
            ),
            num_single_frame_decoder=num_single_frame_decoder,
            operation_order=(
                [
                    "gnn",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                * num_single_frame_decoder
                + [
                    "temp_gnn",
                    "gnn",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                * (num_decoder - num_single_frame_decoder)
            )[2:],
            temp_graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            )
            if temporal
            else None,
            graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims * 2,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 4,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            deformable_model=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=use_deformable_func,
                use_camera_embed=True,
                residual_mode="cat",
                kps_generator=dict(
                    type="SparseBox3DKeyPointsGenerator",
                    num_learnable_pts=6,
                    fix_scale=[
                        [0, 0, 0],
                        [0.45, 0, 0],
                        [-0.45, 0, 0],
                        [0, 0.45, 0],
                        [0, -0.45, 0],
                        [0, 0, 0.45],
                        [0, 0, -0.45],
                    ],
                ),
            ),
            refine_layer=dict(
                type="SparseBox3DRefinementModule",
                embed_dims=embed_dims,
                num_cls=num_classes,
                refine_yaw=True,
                with_quality_estimation=with_quality_estimation,
            ),
            sampler=dict(
                type="SparseBox3DTarget",
                num_dn_groups=0,
                num_temp_dn_groups=0,
                dn_noise_scale=[2.0] * 3 + [0.5] * 7,
                max_dn_gt=32,
                add_neg_dn=True,
                cls_weight=2.0,
                box_weight=0.25,
                reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
                cls_wise_reg_weights={
                    class_names.index("traffic_cone"): [
                        2.0,
                        2.0,
                        2.0,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                    ],
                },
            ),
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            ),
            loss_reg=dict(
                type="SparseBox3DLoss",
                loss_box=dict(type="L1Loss", loss_weight=0.25),
                loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
                loss_yawness=dict(type="GaussianFocalLoss"),
                cls_allow_reverse=[class_names.index("barrier")],
            ),
            decoder=dict(type="SparseBox3DDecoder"),
            reg_weights=[2.0] * 3 + [1.0] * 7,
        ),
        map_head=dict(
            type='Sparse4DHead',
            cls_threshold_to_reg=0.05,
            decouple_attn=decouple_attn_map,
            instance_bank=dict(
                type="InstanceBank",
                num_anchor=100,
                embed_dims=embed_dims,
                anchor="data/kmeans/kmeans_map_100.npy",
                anchor_handler=dict(type="SparsePoint3DKeyPointsGenerator"),
                num_temp_instances=33 if temporal_map else -1,
                confidence_decay=0.6,
                feat_grad=True,
            ),
            anchor_encoder=dict(
                type="SparsePoint3DEncoder",
                embed_dims=embed_dims,
                num_sample=num_sample,
            ),
            num_single_frame_decoder=num_single_frame_decoder_map,
            operation_order=(
                [
                    "gnn",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                * num_single_frame_decoder_map
                + [
                    "temp_gnn",
                    "gnn",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                * (num_decoder - num_single_frame_decoder_map)
            )[2:],
            temp_graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn_map else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            )
            if temporal_map
            else None,
            graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn_map else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims * 2,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 4,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            deformable_model=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=use_deformable_func,
                use_camera_embed=True,
                residual_mode="cat",
                kps_generator=dict(
                    type="SparsePoint3DKeyPointsGenerator",
                    num_sample=num_sample,
                    num_learnable_pts=3,
                    fix_height=(0, 0.5, -0.5),
                ),
            ),
            refine_layer=dict(
                type="SparsePoint3DRefinementModule",
                embed_dims=embed_dims,
                num_sample=num_sample,
                num_cls=num_map_classes,
            ),
            sampler=dict(
                type="SparsePoint3DTarget",
                num_dn_groups=0,
                num_temp_dn_groups=0,
                dn_noise_scale=0.5,
                max_dn_gt=32,
                add_neg_dn=True,
                num_cls=num_map_classes,
                num_sample=num_sample,
                roi_size=roi_size,
            ),
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            ),
            loss_reg=dict(
                type="SparseLineLoss",
                loss_line=dict(type="LinesL1Loss", loss_weight=0.1),
                num_sample=num_sample,
                roi_size=roi_size,
            ),
            decoder=dict(
                type="SparsePoint3DDecoder",
                score_threshold=0.5,
                num_output=20,
            ),
            reg_weights=[1.0] * 40,
        ),
        motion_plan_head=dict(
            type='MotionPlanningHead',
            fut_ts=fut_ts,
            fut_mode=fut_mode,
            ego_fut_ts=ego_fut_ts,
            ego_fut_mode=ego_fut_mode,
            motion_anchor=f'data/kmeans/kmeans_motion_{fut_mode}.npy',
            plan_anchor=f'data/kmeans/kmeans_plan_{ego_fut_mode}.npy',
            embed_dims=embed_dims,
            decouple_attn=decouple_attn_motion,
            instance_queue=dict(
                type="InstanceQueue",
                embed_dims=embed_dims,
                queue_length=queue_length,
                tracking_threshold=0.2,
                feature_map_scale=(input_shape[1]/strides[-1], input_shape[0]/strides[-1]),
            ),
            operation_order=(
                [
                    "temp_gnn",
                    "gnn",
                    "norm",
                    "cross_gnn",
                    "norm",
                    "deformable",
                    "norm",
                    "ffn",
                    "norm",
                    "refine",
                ] * motion_decoder_repeats
            ),
            temp_graph_model=dict(
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn_motion else embed_dims * 2,
                num_heads=motion_num_heads,
                batch_first=True,
                dropout=drop_out,
            ),
            graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn_motion else embed_dims * 2,
                num_heads=motion_num_heads,
                batch_first=True,
                dropout=drop_out,
            ),
            cross_graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims,
                num_heads=motion_num_heads,
                batch_first=True,
                dropout=drop_out,
            ),
            deformable_model=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=use_deformable_func,
                use_camera_embed=True,
                residual_mode="add",
                kps_generator=dict(
                    type="SparseBox3DKeyPointsGenerator",
                    num_learnable_pts=6,
                    fix_scale=[
                        [0, 0, 0],
                        [0.45, 0, 0],
                        [-0.45, 0, 0],
                        [0, 0.45, 0],
                        [0, -0.45, 0],
                        [0, 0, 0.45],
                        [0, 0, -0.45],
                    ],
                ),
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 2,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            refine_layer=dict(
                type="MotionPlanningRefinementModule",
                embed_dims=embed_dims,
                fut_ts=fut_ts,
                fut_mode=fut_mode,
                ego_fut_ts=ego_fut_ts,
                ego_fut_mode=ego_fut_mode,
            ),
            motion_sampler=dict(
                type="MotionTarget",
            ),
            motion_loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.2
            ),
            motion_loss_reg=dict(type='L1Loss', loss_weight=0.2),
            planning_sampler=dict(
                type="PlanningTarget",
                ego_fut_ts=ego_fut_ts,
                ego_fut_mode=ego_fut_mode,
            ),
            plan_loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.5,
            ),
            plan_loss_reg=dict(type='L1Loss', loss_weight=1.0),
            plan_loss_status=dict(type='L1Loss', loss_weight=1.0),
            motion_decoder=dict(type="SparseBox3DMotionDecoder"),
            planning_decoder=dict(type="HierarchicalPlanningDecoder"),
        ),
    ),
)

dataset_type = "NuScenes3DDataset"
data_root = "data/nuscenes/"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
ida_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
    "rot3d_range": [-0.3925, 0.3925],
}

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5, file_client_args=dict(backend="disk")),
    dict(type="ResizeCropFlipImage"),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * 10),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img", "timestamp", "projection_mat", "image_wh", "gt_depth",
            "focal", "gt_bboxes_3d", "gt_labels_3d", "gt_map_labels",
            "gt_map_pts", "gt_agent_fut_trajs", "gt_agent_fut_masks",
            "gt_ego_fut_trajs", "gt_ego_fut_masks", "gt_ego_fut_cmd",
            "ego_status", "tp_near", "tp_far", "fut_boxes",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id"],
    ),
]
test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=["img", "timestamp", "projection_mat", "image_wh", "ego_status", "gt_ego_fut_cmd"],
        meta_keys=["T_global", "T_global_inv", "timestamp"],
    ),
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "infos/nuscenes_infos_train.pkl",
        pipeline=train_pipeline,
        classes=class_names,
        map_classes=map_class_names,
        modality=dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False),
        test_mode=False,
        use_valid_flag=True,
        box_type_3d="LiDAR",
        filter_empty_gt=False,
        img_info_prototype="bevdet",
        speed_mode="abs_dis",
        pre_eval=False,
        sequential=True,
        n_times=queue_length,
        train_adj_ids=[1],
        test_adj_ids=[1],
        max_interval=3,
        min_interval=0,
        fix_direction=False,
        prev_only=False,
        next_only=False,
        stop_prev_grad=0,
        interval_test=False,
        map_ann_file=data_root + "nuscenes_map_infos_train.pkl",
        map_fixed_ptsnum_per_line=num_sample,
        map_eval_use_same_gt_sample_num_flag=True,
        padding_value=-10000,
        classes_attr=[],
        with_box2d=False,
        with_attr=False,
        queue_length=queue_length,
        id_format='index',
        data_aug_conf=ida_aug_conf,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "infos/nuscenes_infos_val.pkl",
        pipeline=test_pipeline,
        classes=class_names,
        map_classes=map_class_names,
        modality=dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False),
        test_mode=True,
        box_type_3d="LiDAR",
        img_info_prototype="bevdet",
        speed_mode="abs_dis",
        pre_eval=False,
        sequential=True,
        n_times=queue_length,
        train_adj_ids=[1],
        test_adj_ids=[1],
        max_interval=3,
        min_interval=0,
        fix_direction=False,
        prev_only=False,
        next_only=False,
        stop_prev_grad=0,
        interval_test=False,
        map_ann_file=data_root + "nuscenes_map_infos_val.pkl",
        map_fixed_ptsnum_per_line=num_sample,
        map_eval_use_same_gt_sample_num_flag=True,
        padding_value=-10000,
        classes_attr=[],
        with_box2d=False,
        with_attr=False,
        queue_length=queue_length,
        id_format='index',
        data_aug_conf=ida_aug_conf,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "infos/nuscenes_infos_val.pkl",
        pipeline=test_pipeline,
        classes=class_names,
        map_classes=map_class_names,
        modality=dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False),
        test_mode=True,
        box_type_3d="LiDAR",
        img_info_prototype="bevdet",
        speed_mode="abs_dis",
        pre_eval=False,
        sequential=True,
        n_times=queue_length,
        train_adj_ids=[1],
        test_adj_ids=[1],
        max_interval=3,
        min_interval=0,
        fix_direction=False,
        prev_only=False,
        next_only=False,
        stop_prev_grad=0,
        interval_test=False,
        map_ann_file=data_root + "nuscenes_map_infos_val.pkl",
        map_fixed_ptsnum_per_line=num_sample,
        map_eval_use_same_gt_sample_num_flag=True,
        padding_value=-10000,
        classes_attr=[],
        with_box2d=False,
        with_attr=False,
        queue_length=queue_length,
        id_format='index',
        data_aug_conf=ida_aug_conf,
    ),
)

optimizer = dict(
    type="AdamW",
    lr=3e-4,
    weight_decay=0.001,
    paramwise_cfg=dict(custom_keys={"img_backbone": dict(lr_mult=0.1)}),
)
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
runner = dict(type="IterBasedRunner", max_iters=num_epochs * num_iters_per_epoch)

evaluation = dict(interval=num_iters_per_epoch * checkpoint_epoch_interval, pipeline=test_pipeline)

checkpoint_config = dict(interval=num_iters_per_epoch * checkpoint_epoch_interval)

custom_hooks = [dict(type="SetEpochInfoHook")]

load_from = "ckpt/sparsedrive_stage1.pth"
