import random
import math
import os
from os import path as osp
import cv2
import tempfile
import copy
import prettytable

import numpy as np
import torch
from torch.utils.data import Dataset
import pyquaternion
from shapely.geometry import LineString
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.eval.detection.config import config_factory as det_configs
from nuscenes.eval.common.config import config_factory as track_configs

import mmcv
from mmcv.utils import print_log
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose
from .utils import (
    draw_lidar_bbox3d_on_img,
    draw_lidar_bbox3d_on_bev,
)


@DATASETS.register_module()
class NuScenes3DDataset(Dataset):
    DefaultAttribute = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.moving",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }
    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }
    CLASSES = (
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    )
    MAP_CLASSES = (
        'ped_crossing',
        'divider',
        'boundary',
    )
    ID_COLOR_MAP = [
        (59, 59, 238),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 255),
        (0, 127, 255),
        (71, 130, 255),
        (127, 127, 0),
    ]

    def __init__(
        self,
        ann_file,
        pipeline=None,
        data_root=None,
        classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        with_visibility=True,
        modality=None,
        test_mode=False,
        det3d_eval_version="detection_cvpr_2019",
        track3d_eval_version="tracking_nips_2019",
        version="v1.0-trainval",
        use_valid_flag=False,
        use_gt_mask=True,
        vis_score_threshold=0.25,
        data_aug_conf=None,
        sequences_split_num=1,
        with_seq_flag=False,
        keep_consistent_seq_aug=True,
        work_dir=None,
        eval_config=None,
        polygon_geom_layers=None,
        polygon_geom_roi_size=(60, 30),
    ):
        self.version = version
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        self.use_gt_mask = use_gt_mask
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.box_mode_3d = 0

        if classes is not None:
            self.CLASSES = classes
        if map_classes is not None:
            self.MAP_CLASSES = map_classes
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.data_infos = self.load_annotations(self.ann_file)

        # Optional live polygon-layer extraction for dseg v2-style supervision.
        # `polygon_geom_layers` is a tuple of nuScenes map polygon layer names
        # (e.g. ('drivable_area', 'walkway', 'stop_line')) to inject into
        # `input_dict['map_geoms']` at load time. Saved `map_annos` only
        # contains LineStrings (geom2anno strips polygons), so polygons are
        # re-extracted live via NuscMapExtractor. None disables (default).
        self.polygon_geom_layers = (
            tuple(polygon_geom_layers) if polygon_geom_layers else None
        )
        self._polygon_extractor = None
        if self.polygon_geom_layers and data_root is not None:
            from .map_utils.nuscmap_extractor import NuscMapExtractor
            self._polygon_extractor = NuscMapExtractor(
                data_root=data_root, roi_size=polygon_geom_roi_size,
            )

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        self.with_velocity = with_velocity
        self.with_visibility = with_visibility
        self.det3d_eval_version = det3d_eval_version
        self.det3d_eval_configs = det_configs(self.det3d_eval_version)
        self.det3d_eval_configs.class_names = list(self.det3d_eval_configs.class_range.keys())
        self.track3d_eval_version = track3d_eval_version
        self.track3d_eval_configs = track_configs(self.track3d_eval_version)
        self.track3d_eval_configs.class_names = list(self.track3d_eval_configs.class_range.keys())
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        self.vis_score_threshold = vis_score_threshold

        self.data_aug_conf = data_aug_conf
        self.sequences_split_num = sequences_split_num
        self.keep_consistent_seq_aug = keep_consistent_seq_aug
        if with_seq_flag:
            self._set_sequence_group_flag()
        
        self.work_dir = work_dir
        self.eval_config = eval_config

    def __len__(self):
        return len(self.data_infos)

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        if self.sequences_split_num == -1:
            self.flag = np.arange(len(self.data_infos))
            return
        
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and len(self.data_infos[idx]["sweeps"]) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.sequences_split_num != 1:
            if self.sequences_split_num == "all":
                self.flag = np.array(
                    range(len(self.data_infos)), dtype=np.int64
                )
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(
                            range(
                                0,
                                bin_counts[curr_flag],
                                math.ceil(
                                    bin_counts[curr_flag]
                                    / self.sequences_split_num
                                ),
                            )
                        )
                        + [bin_counts[curr_flag]]
                    )

                    for sub_seq_idx in (
                        curr_sequence_length[1:] - curr_sequence_length[:-1]
                    ):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert (
                    len(np.bincount(new_flags))
                    == len(np.bincount(self.flag)) * self.sequences_split_num
                )
                self.flag = np.array(new_flags, dtype=np.int64)

    def get_augmentation(self):
        if self.data_aug_conf is None:
            return None
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
            rotate_3d = np.random.uniform(*self.data_aug_conf["rot3d_range"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
            rotate_3d = 0
        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }
        return aug_config

    def __getitem__(self, idx):
        if isinstance(idx, dict):
            aug_config = idx["aug_config"]
            idx = idx["idx"]
        else:
            aug_config = self.get_augmentation()
        data = self.get_data_info(idx)
        data["aug_config"] = aug_config
        data = self.pipeline(data)
        return data

    def get_cat_ids(self, idx):
        info = self.data_infos[idx]
        if self.use_valid_flag and self.use_gt_mask:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file, file_format="pkl")
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        print(self.metadata)
        return data_infos

    def anno2geom(self, annos):
        map_geoms = {}
        for label, anno_list in annos.items():
            map_geoms[label] = []
            for anno in anno_list:
                geom = LineString(anno)
                map_geoms[label].append(geom)
        return map_geoms
    
    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            token=info["token"],
            map_location=info["map_location"],
            pts_filename=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"] / 1e6,
            lidar2ego_translation=info["lidar2ego_translation"],
            lidar2ego_rotation=info["lidar2ego_rotation"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
            ego_status=info['ego_status'].astype(np.float32),
            map_infos=info["map_annos"],
        )
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = pyquaternion.Quaternion(
            info["lidar2ego_rotation"]
        ).rotation_matrix
        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])
        ego2global = np.eye(4)
        ego2global[:3, :3] = pyquaternion.Quaternion(
            info["ego2global_rotation"]
        ).rotation_matrix
        ego2global[:3, 3] = np.array(info["ego2global_translation"])
        input_dict["lidar2global"] = ego2global @ lidar2ego

        map_geoms = self.anno2geom(info["map_annos"])
        if self._polygon_extractor is not None:
            # Re-extract polygon layers live (saved `map_annos` only has LineStrings).
            # Pose is the lidar-to-global transform (matches data prep convention).
            l2g = input_dict["lidar2global"]
            translation = l2g[:3, 3].tolist()
            rotation_q = pyquaternion.Quaternion(matrix=l2g).q.tolist()
            try:
                live = self._polygon_extractor.get_map_geom(
                    info["map_location"], translation, rotation_q,
                )
                for layer in self.polygon_geom_layers:
                    map_geoms[layer] = live.get(layer, [])
            except Exception:
                # Defensive: if extraction fails for one sample, fall back to
                # empty polygon list rather than crashing the dataloader.
                for layer in self.polygon_geom_layers:
                    map_geoms[layer] = []
        input_dict["map_geoms"] = map_geoms

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsic = []
            for cam_type, cam_info in info["cams"].items():
                image_paths.append(cam_info["data_path"])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
                lidar2cam_t = (
                    cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
                )
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = copy.deepcopy(cam_info["cam_intrinsic"])
                cam_intrinsic.append(intrinsic)
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt.T
                lidar2img_rts.append(lidar2img_rt)
                lidar2cam_rts.append(lidar2cam_rt)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    lidar2cam=lidar2cam_rts,
                    cam_intrinsic=cam_intrinsic,
                )
            )

        annos = self.get_ann_info(index)
        input_dict.update(annos)
        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        if self.use_gt_mask:
            if self.use_valid_flag:
                mask = info["valid_flag"]
            else:
                mask = info["num_lidar_pts"] > 0
        else:
            mask = np.ones(len(info["gt_boxes"]), dtype=bool)
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        if self.with_visibility:
            gt_visibility = (info["num_lidar_pts"][mask] > 0).astype(np.float32)

        gt_occluded = (info["num_lidar_pts"][mask] == 0).astype(np.float32)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            gt_visibility=gt_visibility,
            gt_occluded=gt_occluded,
        )
        if "instance_inds" in info:
            instance_inds = np.array(info["instance_inds"], dtype=np.int)[mask]
            anns_results["instance_inds"] = instance_inds
            
        if 'gt_agent_fut_trajs' in info:
            anns_results['gt_agent_fut_trajs'] = info['gt_agent_fut_trajs'][mask]
            anns_results['gt_agent_fut_masks'] = info['gt_agent_fut_masks'][mask]

        if 'gt_ego_fut_trajs' in info:
            anns_results['gt_ego_fut_trajs'] = info['gt_ego_fut_trajs']
            anns_results['gt_ego_fut_masks'] = info['gt_ego_fut_masks']
            anns_results['gt_ego_fut_cmd'] = info['gt_ego_fut_cmd']
        
            ## get future box for planning eval
            fut_ts = int(info['gt_ego_fut_masks'].sum())
            fut_boxes = []
            fut_boxes_occluded = []
            cur_scene_token = info["scene_token"]
            cur_T_global = get_T_global(info)
            for i in range(1, fut_ts + 1):
                fut_info = self.data_infos[index + i]
                fut_scene_token = fut_info["scene_token"]
                if cur_scene_token != fut_scene_token:
                    break
                if self.use_gt_mask:
                    if self.use_valid_flag:
                        mask = fut_info["valid_flag"]
                    else:
                        mask = fut_info["num_lidar_pts"] > 0
                else:
                    mask = np.ones(len(fut_info["gt_boxes"]), dtype=bool)
                if self.use_valid_flag:
                    occluded_mask = ~fut_info["valid_flag"]
                else:
                    occluded_mask = ~(fut_info["num_lidar_pts"] > 0)

                fut_gt_bboxes_3d = fut_info["gt_boxes"][mask]
                fut_gt_bboxes_occluded = fut_info["gt_boxes"][occluded_mask]

                fut_T_global = get_T_global(fut_info)
                T_fut2cur = np.linalg.inv(cur_T_global) @ fut_T_global

                for bboxes in (fut_gt_bboxes_3d, fut_gt_bboxes_occluded):
                    if len(bboxes):
                        center = bboxes[:, :3] @ T_fut2cur[:3, :3].T + T_fut2cur[:3, 3]
                        yaw = np.stack([np.cos(bboxes[:, 6]), np.sin(bboxes[:, 6])], axis=-1)
                        yaw = yaw @ T_fut2cur[:2, :2].T
                        bboxes[:, :3] = center
                        bboxes[:, 6] = np.arctan2(yaw[..., 1], yaw[..., 0])

                fut_boxes.append(fut_gt_bboxes_3d)
                fut_boxes_occluded.append(fut_gt_bboxes_occluded)

            anns_results['fut_boxes'] = fut_boxes
            anns_results['fut_boxes_occluded'] = fut_boxes_occluded
        
        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None, tracking=False):
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(
                det, threshold=self.tracking_threshold if tracking else None
            )
            sample_token = self.data_infos[sample_id]["token"]
            # Attach visibility scores before the class-range filter so
            # index correspondence is preserved after lidar_nusc_box_to_global.
            if not tracking and 'visibility_scores' in det:
                vis_scores = det['visibility_scores'].numpy()
                for j, box in enumerate(boxes):
                    box.vis_score = float(vis_scores[j])
            boxes = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                mapped_class_names,
                self.det3d_eval_configs,
                self.det3d_eval_version,
            )
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if tracking and name in [
                    "barrier",
                    "traffic_cone",
                    "construction_vehicle",
                ]:
                    continue
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = NuScenes3DDataset.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = NuScenes3DDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                )
                if not tracking:
                    nusc_anno.update(
                        dict(
                            detection_name=name,
                            detection_score=box.score,
                            attribute_name=attr,
                        )
                    )
                    if hasattr(box, 'vis_score'):
                        nusc_anno['visibility_score'] = box.vis_score
                else:
                    nusc_anno.update(
                        dict(
                            tracking_name=name,
                            tracking_score=box.score,
                            tracking_id=str(box.token),
                        )
                    )

                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        filename = "results_nusc_tracking.json" if tracking else "results_nusc.json"
        res_path = osp.join(jsonfile_prefix, filename)
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(
        self, result_path, logger=None, result_name="img_bbox", tracking=False
    ):
        from nuscenes import NuScenes

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False
        )
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
        }
        if not tracking:
            from nuscenes.eval.detection.evaluate import NuScenesEval

            nusc_eval = NuScenesEval(
                nusc,
                config=self.det3d_eval_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
            )
            nusc_eval.main(render_curves=False)

            # record metrics
            metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
            detail = dict()
            metric_prefix = f"{result_name}_NuScenes"
            for name in self.CLASSES:
                for k, v in metrics["label_aps"][name].items():
                    val = float("{:.4f}".format(v))
                    detail[
                        "{}/{}_AP_dist_{}".format(metric_prefix, name, k)
                    ] = val
                for k, v in metrics["label_tp_errors"][name].items():
                    val = float("{:.4f}".format(v))
                    detail["{}/{}_{}".format(metric_prefix, name, k)] = val
                for k, v in metrics["tp_errors"].items():
                    val = float("{:.4f}".format(v))
                    detail[
                        "{}/{}".format(metric_prefix, self.ErrNameMapping[k])
                    ] = val

            detail["{}/NDS".format(metric_prefix)] = metrics["nd_score"]
            detail["{}/mAP".format(metric_prefix)] = metrics["mean_ap"]

            from .evaluation.det.occluded_det_eval import compute_tpr_fdr
            tpr_fdr = compute_tpr_fdr(
                osp.join(output_dir, 'metrics_details.json'),
                self.CLASSES,
                self.det3d_eval_configs.dist_ths,
            )
            detail[f'{metric_prefix}/mTPR'] = round(tpr_fdr['mean_tpr'], 4)
            detail[f'{metric_prefix}/mFDR'] = round(tpr_fdr['mean_fdr'], 4)
            dist_th_tp_str = str(self.det3d_eval_configs.dist_th_tp)
            for cls, by_th in tpr_fdr['per_class'].items():
                if dist_th_tp_str in by_th:
                    detail[f'{metric_prefix}/{cls}_tpr'] = round(by_th[dist_th_tp_str]['tpr'], 4)
                    detail[f'{metric_prefix}/{cls}_fdr'] = round(by_th[dist_th_tp_str]['fdr'], 4)
        else:
            from nuscenes.eval.tracking.evaluate import TrackingEval

            nusc_eval = TrackingEval(
                config=self.track3d_eval_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
                nusc_version=self.version,
                nusc_dataroot=self.data_root,
            )
            metrics = nusc_eval.main()

            # record metrics
            metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
            print(metrics)
            detail = dict()
            metric_prefix = f"{result_name}_NuScenes"
            keys = [
                "amota",
                "amotp",
                "recall",
                "motar",
                "gt",
                "mota",
                "motp",
                "mt",
                "ml",
                "faf",
                "tp",
                "fp",
                "fn",
                "ids",
                "frag",
                "tid",
                "lgd",
            ]
            for key in keys:
                detail["{}/{}".format(metric_prefix, key)] = metrics[key]

        return detail

    def format_results(self, results, jsonfile_prefix=None, tracking=False):
        assert isinstance(results, list), "results must be a list"

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        if not ("pts_bbox" in results[0] or "img_bbox" in results[0]):
            result_files = self._format_bbox(
                results, jsonfile_prefix, tracking=tracking
            )
        else:
            result_files = dict()
            for name in results[0]:
                print(f"\nFormating bboxes of {name}")
                results_ = [out[name] for out in results]
                tmp_file_ = jsonfile_prefix
                result_files.update(
                    {
                        name: self._format_bbox(
                            results_, tmp_file_, tracking=tracking
                        )
                    }
                )
        return result_files, tmp_dir

    def format_map_results(self, results, prefix=None):
        submissions = {'results': {},}
        
        for j, pred in enumerate(results):
            '''
            For each case, the result should be formatted as Dict{'vectors': [], 'scores': [], 'labels': []}
            'vectors': List of vector, each vector is a array([[x1, y1], [x2, y2] ...]),
                contain all vectors predicted in this sample.
            'scores: List of score(float), 
                contain scores of all instances in this sample.
            'labels': List of label(int), 
                contain labels of all instances in this sample.
            '''
            if pred is None: # empty prediction
                continue
            pred = pred['img_bbox']

            single_case = {'vectors': [], 'scores': [], 'labels': []}
            token = self.data_infos[j]['token']
            for i in range(len(pred['scores'])):
                score = pred['scores'][i]
                label = pred['labels'][i]
                vector = pred['vectors'][i]

                # A line should have >=2 points
                if len(vector) < 2:
                    continue
                
                single_case['vectors'].append(vector)
                single_case['scores'].append(score)
                single_case['labels'].append(label)
            
            submissions['results'][token] = single_case
        
        out_path = osp.join(prefix, 'submission_vector.json')
        print(f'saving submissions results to {out_path}')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mmcv.dump(submissions, out_path)
        return out_path

    def format_motion_results(self, results, jsonfile_prefix=None, tracking=False, thresh=None):
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(
                det['img_bbox'], threshold=None
            )
            sample_token = self.data_infos[sample_id]["token"]
            boxes = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                mapped_class_names,
                self.det3d_eval_configs,
                self.det3d_eval_version,
                filter_with_cls_range=False,
            )
            for i, box in enumerate(boxes):
                if thresh is not None and box.score < thresh:
                    continue
                name = mapped_class_names[box.label]
                if tracking and name in [
                    "barrier",
                    "traffic_cone",
                    "construction_vehicle",
                ]:
                    continue
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = NuScenes3DDataset.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = NuScenes3DDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                )
                if not tracking:
                    nusc_anno.update(
                        dict(
                            detection_name=name,
                            detection_score=box.score,
                            attribute_name=attr,
                        )
                    )
                else:
                    nusc_anno.update(
                        dict(
                            tracking_name=name,
                            tracking_score=box.score,
                            tracking_id=str(box.token),
                        )
                    )
                if 'trajs_3d' in det['img_bbox']:
                    nusc_anno['trajs'] = det['img_bbox']['trajs_3d'][i].numpy()
                if 'trajs_score' in det['img_bbox']:
                    nusc_anno['trajs_score'] = det['img_bbox']['trajs_score'][i].numpy()
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        return nusc_submissions 

    def _evaluate_single_motion(self,
                         results,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        from .evaluation.motion.motion_eval_uniad import NuScenesEval as NuScenesEvalMotion

        output_dir = result_path
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEvalMotion(
            nusc,
            config=copy.deepcopy(self.det3d_eval_configs),
            result_path=results,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False,
            seconds=6)
        metrics = nusc_eval.main(render_curves=False)
        
        MOTION_METRICS = ['EPA', 'min_ade_err', 'min_fde_err', 'miss_rate_err']
        class_names = ['car', 'pedestrian']

        table = prettytable.PrettyTable()
        table.field_names = ["class names"] + MOTION_METRICS
        for class_name in class_names:
            row_data = [class_name]
            for m in MOTION_METRICS:
                row_data.append('%.4f' % metrics[f'{class_name}_{m}'])
            table.add_row(row_data)
        print_log('\n'+str(table), logger=logger)
        return metrics

    def _evaluate_single_det_occluded(self, result_path, logger=None, result_name='img_bbox',
                                       vis_threshold=None):
        """Evaluate detection on occluded objects only (num_lidar_pts == 0).

        If vis_threshold is set and result_path contains visibility_score fields,
        only predictions with visibility_score < vis_threshold (i.e. classified as
        occluded by the model) are passed to the evaluator.  The filtered JSON is
        written alongside the original as results_nusc_occ_filtered.json.
        """
        import json
        from nuscenes import NuScenes
        from .evaluation.det.occluded_det_eval import OccludedDetectionEval

        # Apply classifier filter if threshold provided and scores are present.
        if vis_threshold is not None:
            with open(result_path) as f:
                raw = json.load(f)
            sample_anns = next(iter(raw['results'].values()), [])
            if sample_anns and 'visibility_score' in sample_anns[0]:
                filtered = {'meta': raw['meta'], 'results': {}}
                for token, anns in raw['results'].items():
                    filtered['results'][token] = [
                        a for a in anns if a['visibility_score'] < vis_threshold
                    ]
                filtered_path = osp.join(
                    osp.dirname(result_path), 'results_nusc_occ_filtered.json'
                )
                with open(filtered_path, 'w') as f:
                    json.dump(filtered, f)
                result_path = filtered_path

        output_dir = osp.join(osp.dirname(result_path), 'occluded_det')
        nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = OccludedDetectionEval(
            nusc,
            config=self.det3d_eval_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False,
        )
        nusc_eval.main(render_curves=False)

        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = {}
        for name in self.CLASSES:
            for k, v in metrics['label_aps'].get(name, {}).items():
                detail[f'occluded/{name}_AP_dist_{k}'] = float('{:.4f}'.format(v))
            for k, v in metrics['label_tp_errors'].get(name, {}).items():
                detail[f'occluded/{name}_{k}'] = float('{:.4f}'.format(v))
        for k, v in metrics['tp_errors'].items():
            detail[f'occluded/{self.ErrNameMapping[k]}'] = float('{:.4f}'.format(v))
        detail['occluded/NDS'] = metrics['nd_score']
        detail['occluded/mAP'] = metrics['mean_ap']

        from .evaluation.det.occluded_det_eval import compute_tpr_fdr
        tpr_fdr = compute_tpr_fdr(
            osp.join(output_dir, 'metrics_details.json'),
            self.CLASSES,
            self.det3d_eval_configs.dist_ths,
        )
        detail['occluded/mTPR'] = round(tpr_fdr['mean_tpr'], 4)
        detail['occluded/mFDR'] = round(tpr_fdr['mean_fdr'], 4)
        dist_th_tp_str = str(self.det3d_eval_configs.dist_th_tp)
        for cls, by_th in tpr_fdr['per_class'].items():
            if dist_th_tp_str in by_th:
                detail[f'occluded/{cls}_tpr'] = round(by_th[dist_th_tp_str]['tpr'], 4)
                detail[f'occluded/{cls}_fdr'] = round(by_th[dist_th_tp_str]['fdr'], 4)
        return detail

    def _evaluate_single_motion_occluded(self,
                                         results,
                                         result_path,
                                         logger=None):
        """Evaluate motion prediction restricted to occluded objects (num_lidar_pts == 0)."""
        from nuscenes import NuScenes
        from .evaluation.motion.motion_eval_uniad import OccludedMotionEval

        output_dir = osp.join(result_path, 'occluded_motion')
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = OccludedMotionEval(
            nusc,
            config=copy.deepcopy(self.det3d_eval_configs),
            result_path=results,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False,
            seconds=6)
        metrics = nusc_eval.main(render_curves=False)

        MOTION_METRICS = ['EPA', 'min_ade_err', 'min_fde_err', 'miss_rate_err']
        class_names = ['car', 'pedestrian']

        table = prettytable.PrettyTable()
        table.field_names = ["class names (occluded)"] + MOTION_METRICS
        for class_name in class_names:
            row_data = [class_name]
            for m in MOTION_METRICS:
                row_data.append('%.4f' % metrics[f'{class_name}_{m}'])
            table.add_row(row_data)
        print_log('\n[Occluded Objects]\n' + str(table), logger=logger)

        return {f'occluded/{k}': v for k, v in metrics.items()}

    def _evaluate_single_det_visible(self, result_path, logger=None, result_name='img_bbox',
                                      vis_threshold=None):
        """Evaluate detection on visible objects, ignoring predictions that match occluded GT.

        This gives a fair vis/mAP comparison between a visible-only baseline and
        a model trained to also predict occluded objects: detections of occluded
        objects are not penalised as false positives.

        If vis_threshold is set and result_path contains visibility_score fields,
        only predictions with visibility_score >= vis_threshold (i.e. classified as
        visible by the model) are passed to the evaluator.  The filtered JSON is
        written alongside the original as results_nusc_vis_filtered.json.
        """
        import json
        from nuscenes import NuScenes
        from .evaluation.det.occluded_det_eval import VisibleDetectionEval

        # Apply classifier filter if threshold provided and scores are present.
        if vis_threshold is not None:
            with open(result_path) as f:
                raw = json.load(f)
            sample_anns = next(iter(raw['results'].values()), [])
            if sample_anns and 'visibility_score' in sample_anns[0]:
                filtered = {'meta': raw['meta'], 'results': {}}
                for token, anns in raw['results'].items():
                    filtered['results'][token] = [
                        a for a in anns if a['visibility_score'] >= vis_threshold
                    ]
                filtered_path = osp.join(
                    osp.dirname(result_path), 'results_nusc_vis_filtered.json'
                )
                with open(filtered_path, 'w') as f:
                    json.dump(filtered, f)
                result_path = filtered_path

        output_dir = osp.join(osp.dirname(result_path), 'visible_det')
        nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = VisibleDetectionEval(
            nusc,
            config=self.det3d_eval_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False,
        )
        nusc_eval.main(render_curves=False)

        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = {}
        for name in self.CLASSES:
            for k, v in metrics['label_aps'].get(name, {}).items():
                detail[f'vis/{name}_AP_dist_{k}'] = float('{:.4f}'.format(v))
            for k, v in metrics['label_tp_errors'].get(name, {}).items():
                detail[f'vis/{name}_{k}'] = float('{:.4f}'.format(v))
        for k, v in metrics['tp_errors'].items():
            detail[f'vis/{self.ErrNameMapping[k]}'] = float('{:.4f}'.format(v))
        detail['vis/NDS'] = metrics['nd_score']
        detail['vis/mAP'] = metrics['mean_ap']

        from .evaluation.det.occluded_det_eval import compute_tpr_fdr
        tpr_fdr = compute_tpr_fdr(
            osp.join(output_dir, 'metrics_details.json'),
            self.CLASSES,
            self.det3d_eval_configs.dist_ths,
        )
        detail['vis/mTPR'] = round(tpr_fdr['mean_tpr'], 4)
        detail['vis/mFDR'] = round(tpr_fdr['mean_fdr'], 4)
        dist_th_tp_str = str(self.det3d_eval_configs.dist_th_tp)
        for cls, by_th in tpr_fdr['per_class'].items():
            if dist_th_tp_str in by_th:
                detail[f'vis/{cls}_tpr'] = round(by_th[dist_th_tp_str]['tpr'], 4)
                detail[f'vis/{cls}_fdr'] = round(by_th[dist_th_tp_str]['fdr'], 4)
        return detail

    def _evaluate_single_det_all(self, result_path, logger=None, result_name='img_bbox'):
        """Evaluate detection on all objects (visible + occluded)."""
        from nuscenes import NuScenes
        from .evaluation.det.occluded_det_eval import AllDetectionEval

        output_dir = osp.join(osp.dirname(result_path), 'all_det')
        nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = AllDetectionEval(
            nusc,
            config=self.det3d_eval_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False,
        )
        nusc_eval.main(render_curves=False)

        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = {}
        for name in self.CLASSES:
            for k, v in metrics['label_aps'].get(name, {}).items():
                detail[f'all/{name}_AP_dist_{k}'] = float('{:.4f}'.format(v))
            for k, v in metrics['label_tp_errors'].get(name, {}).items():
                detail[f'all/{name}_{k}'] = float('{:.4f}'.format(v))
        for k, v in metrics['tp_errors'].items():
            detail[f'all/{self.ErrNameMapping[k]}'] = float('{:.4f}'.format(v))
        detail['all/NDS'] = metrics['nd_score']
        detail['all/mAP'] = metrics['mean_ap']

        from .evaluation.det.occluded_det_eval import compute_tpr_fdr
        tpr_fdr = compute_tpr_fdr(
            osp.join(output_dir, 'metrics_details.json'),
            self.CLASSES,
            self.det3d_eval_configs.dist_ths,
        )
        detail['all/mTPR'] = round(tpr_fdr['mean_tpr'], 4)
        detail['all/mFDR'] = round(tpr_fdr['mean_fdr'], 4)
        dist_th_tp_str = str(self.det3d_eval_configs.dist_th_tp)
        for cls, by_th in tpr_fdr['per_class'].items():
            if dist_th_tp_str in by_th:
                detail[f'all/{cls}_tpr'] = round(by_th[dist_th_tp_str]['tpr'], 4)
                detail[f'all/{cls}_fdr'] = round(by_th[dist_th_tp_str]['fdr'], 4)
        return detail

    def _evaluate_single_motion_all(self, results, result_path, logger=None):
        """Evaluate motion prediction on all objects (visible + occluded)."""
        from nuscenes import NuScenes
        from .evaluation.motion.motion_eval_uniad import AllMotionEval

        output_dir = osp.join(result_path, 'all_motion')
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = AllMotionEval(
            nusc,
            config=copy.deepcopy(self.det3d_eval_configs),
            result_path=results,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False,
            seconds=6)
        metrics = nusc_eval.main(render_curves=False)

        MOTION_METRICS = ['EPA', 'min_ade_err', 'min_fde_err', 'miss_rate_err']
        class_names = ['car', 'pedestrian']

        table = prettytable.PrettyTable()
        table.field_names = ["class names (all)"] + MOTION_METRICS
        for class_name in class_names:
            row_data = [class_name]
            for m in MOTION_METRICS:
                row_data.append('%.4f' % metrics[f'{class_name}_{m}'])
            table.add_row(row_data)
        print_log('\n[All Objects]\n' + str(table), logger=logger)

        return {f'all/{k}': v for k, v in metrics.items()}

    def _evaluate_visibility_accuracy(self, results, invert_visibility=True):
        """Evaluate the visibility head calibration on matched GT boxes.

        For each GT box (visible or occluded) we find the closest same-class
        prediction within MATCH_DIST metres (BEV).  We then compare the
        prediction's visibility score to the GT sensor-visibility flag
        (num_lidar_pts > 0 = 1 for visible).

        invert_visibility must match the decoder config:
          True  (default) — decoder output is P(visible); metrics computed as-is.
          False           — decoder output is P(occluded); scores are flipped to
                           P(visible) before computing any metric.

        Returned metrics
        ----------------
        visibility/accuracy          : fraction correct at 0.5 threshold
        visibility/auroc             : area under the ROC curve (positive = visible)
        visibility/accuracy_visible  : accuracy on visible-GT-matched pairs
        visibility/accuracy_occluded : accuracy on occluded-GT-matched pairs
        visibility/n_matched         : total matched pairs across the val set
        visibility/n_visible         : matched pairs where GT is visible
        visibility/n_occluded        : matched pairs where GT is occluded
        """
        MATCH_DIST = 4.0  # BEV centre-distance threshold in metres

        all_scores = []   # predicted visibility scores, in P(visible) convention
        all_targets = []  # GT sensor-visibility (0.0 / 1.0)

        for i, result in enumerate(results):
            det = result.get('img_bbox', result)
            if 'visibility_scores' not in det:
                return {}   # head not enabled — skip entirely

            vis_scores  = det['visibility_scores'].numpy()   # (N,)
            boxes       = det['boxes_3d'].numpy()            # (N, ≥2)
            pred_labels = det['labels_3d'].numpy()           # (N,)

            info     = self.data_infos[i]
            gt_boxes = info['gt_boxes']    # (M, 7) lidar frame
            gt_names = info['gt_names']    # (M,)

            if 'num_lidar_pts' in info:
                gt_vis = (info['num_lidar_pts'] > 0).astype(np.float32)
            elif 'valid_flag' in info:
                gt_vis = info['valid_flag'].astype(np.float32)
            else:
                continue

            if len(gt_boxes) == 0 or len(boxes) == 0:
                continue

            pred_centers = boxes[:, :2]    # (N, 2) BEV
            gt_centers   = gt_boxes[:, :2] # (M, 2) BEV

            for gi in range(len(gt_names)):
                gt_cls = gt_names[gi]
                if gt_cls not in self.CLASSES:
                    continue
                gt_label = self.CLASSES.index(gt_cls)

                cls_idx = np.where(pred_labels == gt_label)[0]
                if len(cls_idx) == 0:
                    continue

                dists   = np.linalg.norm(pred_centers[cls_idx] - gt_centers[gi], axis=1)
                nearest = dists.argmin()
                if dists[nearest] <= MATCH_DIST:
                    all_scores.append(float(vis_scores[cls_idx[nearest]]))
                    all_targets.append(float(gt_vis[gi]))

        if len(all_targets) < 2:
            return {}

        scores  = np.array(all_scores,  dtype=np.float64)
        targets = np.array(all_targets, dtype=np.float64)
        if not invert_visibility:
            # Decoder output is P(occluded); flip to P(visible) so that higher
            # score means more likely visible, consistent with gt_vis = 1 for visible.
            scores = 1.0 - scores
        preds   = (scores > 0.5).astype(np.float64)

        accuracy = float((preds == targets).mean())

        # AUROC via trapezoidal rule (no external dependency)
        pos = targets.sum()
        neg = len(targets) - pos
        if pos > 0 and neg > 0:
            order    = np.argsort(scores)[::-1]
            t_sorted = targets[order]
            tprs = np.concatenate([[0.0], np.cumsum(t_sorted == 1) / pos, [1.0]])
            fprs = np.concatenate([[0.0], np.cumsum(t_sorted == 0) / neg, [1.0]])
            auroc = float(np.trapz(tprs, fprs))
            j = tprs[1:-1] - fprs[1:-1]
            opt_threshold = float(scores[order][int(np.argmax(j))])
        else:
            auroc = float('nan')
            opt_threshold = float('nan')

        vis_mask = targets == 1.0
        occ_mask = targets == 0.0
        acc_vis = float((preds[vis_mask] == targets[vis_mask]).mean()) if vis_mask.any() else float('nan')
        acc_occ = float((preds[occ_mask] == targets[occ_mask]).mean()) if occ_mask.any() else float('nan')

        return {
            'visibility/accuracy':          accuracy,
            'visibility/auroc':             auroc,
            'visibility/opt_threshold':     opt_threshold,
            'visibility/accuracy_visible':  acc_vis,
            'visibility/accuracy_occluded': acc_occ,
            'visibility/n_matched':         int(len(targets)),
            'visibility/n_visible':         int(vis_mask.sum()),
            'visibility/n_occluded':        int(occ_mask.sum()),
        }

    def evaluate(
        self,
        results,
        eval_mode,
        metric=None,
        logger=None,
        jsonfile_prefix=None,
        result_names=["img_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        res_path = "results.pkl" if "trainval" in self.version else "results_mini.pkl"
        res_path = osp.join(self.work_dir, res_path)
        print('All Results write to', res_path)
        mmcv.dump(results, res_path)

        results_dict = dict()
        detection_result_files = None
        if eval_mode['with_det']:
            self.tracking = eval_mode["with_tracking"]
            self.tracking_threshold = eval_mode["tracking_threshold"]
            for metric in ["detection", "tracking"]:
                tracking = metric == "tracking"
                if tracking and not self.tracking:
                    continue
                result_files, tmp_dir = self.format_results(
                    results, jsonfile_prefix=self.work_dir, tracking=tracking
                )
                if not tracking:
                    detection_result_files = result_files

                if isinstance(result_files, dict):
                    for name in result_names:
                        ret_dict = self._evaluate_single(
                            result_files[name], tracking=tracking
                        )
                    results_dict.update(ret_dict)
                elif isinstance(result_files, str):
                    ret_dict = self._evaluate_single(
                        result_files, tracking=tracking
                    )
                    results_dict.update(ret_dict)

                if tmp_dir is not None:
                    tmp_dir.cleanup()

            vis_metrics = self._evaluate_visibility_accuracy(
                results,
                invert_visibility=eval_mode.get('invert_visibility', True),
            )
            results_dict.update(vis_metrics)

        if eval_mode['with_map']:
            from .evaluation.map.vector_eval import VectorEvaluate
            self.map_evaluator = VectorEvaluate(self.eval_config)
            result_path = self.format_map_results(results, prefix=self.work_dir)
            map_results_dict = self.map_evaluator.evaluate(result_path, logger=logger)
            results_dict.update(map_results_dict)

        motion_result_files = None
        if eval_mode['with_motion']:
            thresh = eval_mode["motion_threshhold"]
            motion_result_files = self.format_motion_results(results, jsonfile_prefix=self.work_dir, thresh=thresh)
            motion_results_dict = self._evaluate_single_motion(motion_result_files, self.work_dir, logger=logger)
            results_dict.update(motion_results_dict)

        if eval_mode.get('with_occlusion', False):
            thresh = eval_mode["motion_threshhold"]
            if eval_mode.get('with_motion', False):
                if motion_result_files is None:
                    motion_result_files = self.format_motion_results(results, jsonfile_prefix=self.work_dir, thresh=thresh)
                occluded_results_dict = self._evaluate_single_motion_occluded(
                    motion_result_files, self.work_dir, logger=logger)
                results_dict.update(occluded_results_dict)

            if detection_result_files is not None:
                if isinstance(detection_result_files, dict):
                    for name in result_names:
                        vis_det_dict = self._evaluate_single_det_visible(
                            detection_result_files[name], logger=logger, result_name=name,
                            vis_threshold=eval_mode.get('occ_vis_threshold'))
                        results_dict.update(vis_det_dict)
                        occ_det_dict = self._evaluate_single_det_occluded(
                            detection_result_files[name], logger=logger, result_name=name,
                            vis_threshold=eval_mode.get('occ_vis_threshold'))
                        results_dict.update(occ_det_dict)
                        all_det_dict = self._evaluate_single_det_all(
                            detection_result_files[name], logger=logger, result_name=name)
                        results_dict.update(all_det_dict)
                elif isinstance(detection_result_files, str):
                    vis_det_dict = self._evaluate_single_det_visible(
                        detection_result_files, logger=logger,
                        vis_threshold=eval_mode.get('occ_vis_threshold'))
                    results_dict.update(vis_det_dict)
                    occ_det_dict = self._evaluate_single_det_occluded(
                        detection_result_files, logger=logger,
                        vis_threshold=eval_mode.get('occ_vis_threshold'))
                    results_dict.update(occ_det_dict)
                    all_det_dict = self._evaluate_single_det_all(
                        detection_result_files, logger=logger)
                    results_dict.update(all_det_dict)

            if eval_mode.get('with_motion', False) and motion_result_files is not None:
                all_results_dict = self._evaluate_single_motion_all(
                    motion_result_files, self.work_dir, logger=logger)
                results_dict.update(all_results_dict)

        if eval_mode['with_planning']:
            from .evaluation.planning.planning_eval import planning_eval
            planning_results_dict = planning_eval(
                results, self.eval_config, logger=logger,
                with_occlusion=eval_mode.get('with_occlusion', False))
            results_dict.update(planning_results_dict)

        if show or out_dir:
            self.show(results, save_dir=out_dir, show=show, pipeline=pipeline)
        
        # print main metrics for recording
        metric_str = '\n'
        if "img_bbox_NuScenes/NDS" in results_dict:
            metric_str += f'mAP: {results_dict.get("img_bbox_NuScenes/mAP"):.4f}\n'
            metric_str += f'mATE: {results_dict.get("img_bbox_NuScenes/mATE"):.4f}\n'
            metric_str += f'mASE: {results_dict.get("img_bbox_NuScenes/mASE"):.4f}\n'
            metric_str += f'mAOE: {results_dict.get("img_bbox_NuScenes/mAOE"):.4f}\n' 
            metric_str += f'mAVE: {results_dict.get("img_bbox_NuScenes/mAVE"):.4f}\n' 
            metric_str += f'mAAE: {results_dict.get("img_bbox_NuScenes/mAAE"):.4f}\n' 
            metric_str += f'NDS: {results_dict.get("img_bbox_NuScenes/NDS"):.4f}\n'
            if 'img_bbox_NuScenes/mTPR' in results_dict:
                metric_str += f'mTPR: {results_dict["img_bbox_NuScenes/mTPR"]:.4f}  mFDR: {results_dict["img_bbox_NuScenes/mFDR"]:.4f}\n'
            metric_str += '\n'
        
        if "img_bbox_NuScenes/amota" in results_dict:
            metric_str += f'AMOTA: {results_dict["img_bbox_NuScenes/amota"]:.4f}\n' 
            metric_str += f'AMOTP: {results_dict["img_bbox_NuScenes/amotp"]:.4f}\n' 
            metric_str += f'RECALL: {results_dict["img_bbox_NuScenes/recall"]:.4f}\n' 
            metric_str += f'MOTAR: {results_dict["img_bbox_NuScenes/motar"]:.4f}\n' 
            metric_str += f'MOTA: {results_dict["img_bbox_NuScenes/mota"]:.4f}\n' 
            metric_str += f'MOTP: {results_dict["img_bbox_NuScenes/motp"]:.4f}\n' 
            metric_str += f'IDS: {results_dict["img_bbox_NuScenes/ids"]}\n\n' 
        
        if "mAP_normal" in results_dict:
            metric_str += f'ped_crossing= {results_dict["ped_crossing"]:.4f}\n' 
            metric_str += f'divider= {results_dict["divider"]:.4f}\n' 
            metric_str += f'boundary= {results_dict["boundary"]:.4f}\n' 
            metric_str += f'mAP_normal= {results_dict["mAP_normal"]:.4f}\n\n' 

        if "car_EPA" in results_dict:
            metric_str += f'Car / Ped\n'
            metric_str += f'epa= {results_dict["car_EPA"]:.4f} / {results_dict["pedestrian_EPA"]:.4f}\n'
            metric_str += f'ade= {results_dict["car_min_ade_err"]:.4f} / {results_dict["pedestrian_min_ade_err"]:.4f}\n'
            metric_str += f'fde= {results_dict["car_min_fde_err"]:.4f} / {results_dict["pedestrian_min_fde_err"]:.4f}\n'
            metric_str += f'mr= {results_dict["car_miss_rate_err"]:.4f} / {results_dict["pedestrian_miss_rate_err"]:.4f}\n\n'

        if "occluded/NDS" in results_dict:
            metric_str += f'[Occluded Det]\n'
            metric_str += f'mAP: {results_dict["occluded/mAP"]:.4f}\n'
            metric_str += f'mATE: {results_dict["occluded/mATE"]:.4f}\n'
            metric_str += f'mASE: {results_dict["occluded/mASE"]:.4f}\n'
            metric_str += f'mAOE: {results_dict["occluded/mAOE"]:.4f}\n'
            metric_str += f'mAVE: {results_dict["occluded/mAVE"]:.4f}\n'
            metric_str += f'mAAE: {results_dict["occluded/mAAE"]:.4f}\n'
            metric_str += f'NDS: {results_dict["occluded/NDS"]:.4f}\n'
            if 'occluded/mTPR' in results_dict:
                metric_str += f'mTPR: {results_dict["occluded/mTPR"]:.4f}  mFDR: {results_dict["occluded/mFDR"]:.4f}\n'
            metric_str += '\n'

        if "occluded/car_EPA" in results_dict:
            metric_str += f'[Occluded Motion] Car / Ped\n'
            metric_str += f'epa= {results_dict["occluded/car_EPA"]:.4f} / {results_dict["occluded/pedestrian_EPA"]:.4f}\n'
            metric_str += f'ade= {results_dict["occluded/car_min_ade_err"]:.4f} / {results_dict["occluded/pedestrian_min_ade_err"]:.4f}\n'
            metric_str += f'fde= {results_dict["occluded/car_min_fde_err"]:.4f} / {results_dict["occluded/pedestrian_min_fde_err"]:.4f}\n'
            metric_str += f'mr= {results_dict["occluded/car_miss_rate_err"]:.4f} / {results_dict["occluded/pedestrian_miss_rate_err"]:.4f}\n\n'

        if "all/NDS" in results_dict:
            metric_str += f'[All Det]\n'
            metric_str += f'mAP: {results_dict["all/mAP"]:.4f}\n'
            metric_str += f'mATE: {results_dict["all/mATE"]:.4f}\n'
            metric_str += f'mASE: {results_dict["all/mASE"]:.4f}\n'
            metric_str += f'mAOE: {results_dict["all/mAOE"]:.4f}\n'
            metric_str += f'mAVE: {results_dict["all/mAVE"]:.4f}\n'
            metric_str += f'mAAE: {results_dict["all/mAAE"]:.4f}\n'
            metric_str += f'NDS: {results_dict["all/NDS"]:.4f}\n'
            if 'all/mTPR' in results_dict:
                metric_str += f'mTPR: {results_dict["all/mTPR"]:.4f}  mFDR: {results_dict["all/mFDR"]:.4f}\n'
            metric_str += '\n'

        if "all/car_EPA" in results_dict:
            metric_str += f'[All Motion] Car / Ped\n'
            metric_str += f'epa= {results_dict["all/car_EPA"]:.4f} / {results_dict["all/pedestrian_EPA"]:.4f}\n'
            metric_str += f'ade= {results_dict["all/car_min_ade_err"]:.4f} / {results_dict["all/pedestrian_min_ade_err"]:.4f}\n'
            metric_str += f'fde= {results_dict["all/car_min_fde_err"]:.4f} / {results_dict["all/pedestrian_min_fde_err"]:.4f}\n'
            metric_str += f'mr= {results_dict["all/car_miss_rate_err"]:.4f} / {results_dict["all/pedestrian_miss_rate_err"]:.4f}\n\n'

        if 'visibility/accuracy' in results_dict:
            rd = results_dict
            metric_str += f'[Visibility Head]\n'
            metric_str += (
                f'accuracy= {rd["visibility/accuracy"]:.4f}  '
                f'(visible={rd["visibility/accuracy_visible"]:.4f}, '
                f'occluded={rd["visibility/accuracy_occluded"]:.4f})\n'
            )
            metric_str += f'auroc= {rd["visibility/auroc"]:.4f}\n'
            metric_str += (
                f'matched: {rd["visibility/n_matched"]} '
                f'(visible={rd["visibility/n_visible"]}, '
                f'occluded={rd["visibility/n_occluded"]})\n\n'
            )

        if "L2" in results_dict:
            metric_str += f'obj_box_col: {(results_dict["obj_box_col"]*100):.3f}%\n'
            metric_str += f'L2: {results_dict["L2"]:.4f}\n'
            if "occluded/obj_box_col" in results_dict:
                metric_str += f'obj_box_col_occluded: {(results_dict["occluded/obj_box_col"]*100):.3f}%\n'
            metric_str += '\n'
        
        print_log(metric_str, logger=logger)
        return results_dict

    def show(self, results, save_dir=None, show=False, pipeline=None):
        save_dir = "./" if save_dir is None else save_dir
        save_dir = os.path.join(save_dir, "visual")
        print_log(os.path.abspath(save_dir))
        pipeline = Compose(pipeline)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        videoWriter = None

        for i, result in enumerate(results):
            if "img_bbox" in result.keys():
                result = result["img_bbox"]
            data_info = pipeline(self.get_data_info(i))
            imgs = []

            raw_imgs = data_info["img"]
            lidar2img = data_info["img_metas"].data["lidar2img"]
            pred_bboxes_3d = result["boxes_3d"][
                result["scores_3d"] > self.vis_score_threshold
            ]
            if "instance_ids" in result and self.tracking:
                color = []
                for id in result["instance_ids"].cpu().numpy().tolist():
                    color.append(
                        self.ID_COLOR_MAP[int(id % len(self.ID_COLOR_MAP))]
                    )
            elif "labels_3d" in result:
                color = []
                for id in result["labels_3d"].cpu().numpy().tolist():
                    color.append(self.ID_COLOR_MAP[id])
            else:
                color = (255, 0, 0)

            # ===== draw boxes_3d to images =====
            for j, img_origin in enumerate(raw_imgs):
                img = img_origin.copy()
                if len(pred_bboxes_3d) != 0:
                    img = draw_lidar_bbox3d_on_img(
                        pred_bboxes_3d,
                        img,
                        lidar2img[j],
                        img_metas=None,
                        color=color,
                        thickness=3,
                    )
                imgs.append(img)

            # ===== draw boxes_3d to BEV =====
            bev = draw_lidar_bbox3d_on_bev(
                pred_bboxes_3d,
                bev_size=img.shape[0] * 2,
                color=color,
            )

            # ===== put text and concat =====
            for j, name in enumerate(
                [
                    "front",
                    "front right",
                    "front left",
                    "rear",
                    "rear left",
                    "rear right",
                ]
            ):
                imgs[j] = cv2.rectangle(
                    imgs[j],
                    (0, 0),
                    (440, 80),
                    color=(255, 255, 255),
                    thickness=-1,
                )
                w, h = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
                text_x = int(220 - w / 2)
                text_y = int(40 + h / 2)

                imgs[j] = cv2.putText(
                    imgs[j],
                    name,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
            image = np.concatenate(
                [
                    np.concatenate([imgs[2], imgs[0], imgs[1]], axis=1),
                    np.concatenate([imgs[5], imgs[3], imgs[4]], axis=1),
                ],
                axis=0,
            )
            image = np.concatenate([image, bev], axis=1)

            # ===== save video =====
            if videoWriter is None:
                videoWriter = cv2.VideoWriter(
                    os.path.join(save_dir, "video.avi"),
                    fourcc,
                    7,
                    image.shape[:2][::-1],
                )
            cv2.imwrite(os.path.join(save_dir, f"{i}.jpg"), image)
            videoWriter.write(image)
        videoWriter.release()


def output_to_nusc_box(detection, threshold=None):
    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()
    if "instance_ids" in detection:
        ids = detection["instance_ids"]  # .numpy()
    if threshold is not None:
        if "cls_scores" in detection:
            mask = detection["cls_scores"].numpy() >= threshold
        else:
            mask = scores >= threshold
        box3d = box3d[mask]
        scores = scores[mask]
        labels = labels[mask]
        ids = ids[mask]

    if hasattr(box3d, "gravity_center"):
        box_gravity_center = box3d.gravity_center.numpy()
        box_dims = box3d.dims.numpy()
        nus_box_dims = box_dims[:, [1, 0, 2]]
        box_yaw = box3d.yaw.numpy()
    else:
        box3d = box3d.numpy()
        box_gravity_center = box3d[..., :3].copy()
        box_dims = box3d[..., 3:6].copy()
        nus_box_dims = box_dims[..., [1, 0, 2]]
        box_yaw = box3d[..., 6].copy()

    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    # box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        # Skip NaN/inf rows: configs that zero detection losses (e.g. s2nopercep)
        # produce an unsupervised det head whose outputs drift to NaN. NuScenesBox
        # asserts no NaN; without this filter the entire eval crashes before
        # planning metrics are computed.
        if (not np.all(np.isfinite(box_gravity_center[i]))
                or not np.all(np.isfinite(nus_box_dims[i]))
                or not np.isfinite(box_yaw[i])):
            continue
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if hasattr(box3d, "gravity_center"):
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (*box3d[i, 7:9], 0.0)
        velocity = tuple(v if np.isfinite(v) else 0.0 for v in velocity)
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i] if np.isfinite(scores[i]) else 0.0,
            velocity=velocity,
        )
        if "instance_ids" in detection:
            box.token = ids[i]
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(
    info,
    boxes,
    classes,
    eval_configs,
    eval_version="detection_cvpr_2019",
    filter_with_cls_range=True,
):
    box_list = []
    for i, box in enumerate(boxes):
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info["lidar2ego_rotation"]))
        box.translate(np.array(info["lidar2ego_translation"]))
        # filter det in ego.
        if filter_with_cls_range:
            cls_range_map = eval_configs.class_range
            radius = np.linalg.norm(box.center[:2], 2)
            det_range = cls_range_map[classes[box.label]]
            if radius > det_range:
                continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
    return box_list


def get_T_global(info):
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = pyquaternion.Quaternion(
        info["lidar2ego_rotation"]
    ).rotation_matrix
    lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])
    ego2global = np.eye(4)
    ego2global[:3, :3] = pyquaternion.Quaternion(
        info["ego2global_rotation"]
    ).rotation_matrix
    ego2global[:3, 3] = np.array(info["ego2global_translation"])
    return ego2global @ lidar2ego