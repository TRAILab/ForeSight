import numpy as np
import mmcv
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from shapely.geometry import MultiPoint, box
from ..utils import box3d_to_corners


@PIPELINES.register_module()
class MultiScaleDepthMapGenerator(object):
    def __init__(self, downsample=1, max_depth=60):
        if not isinstance(downsample, (list, tuple)):
            downsample = [downsample]
        self.downsample = downsample
        self.max_depth = max_depth

    def __call__(self, input_dict):
        points = input_dict["points"][..., :3, None]
        gt_depth = []
        for i, lidar2img in enumerate(input_dict["lidar2img"]):
            H, W = input_dict["img_shape"][i][:2]

            pts_2d = (
                np.squeeze(lidar2img[:3, :3] @ points, axis=-1)
                + lidar2img[:3, 3]
            )
            pts_2d[:, :2] /= pts_2d[:, 2:3]
            U = np.round(pts_2d[:, 0]).astype(np.int32)
            V = np.round(pts_2d[:, 1]).astype(np.int32)
            depths = pts_2d[:, 2]
            mask = np.logical_and.reduce(
                [
                    V >= 0,
                    V < H,
                    U >= 0,
                    U < W,
                    depths >= 0.1,
                    # depths <= self.max_depth,
                ]
            )
            V, U, depths = V[mask], U[mask], depths[mask]
            sort_idx = np.argsort(depths)[::-1]
            V, U, depths = V[sort_idx], U[sort_idx], depths[sort_idx]
            depths = np.clip(depths, 0.1, self.max_depth)
            for j, downsample in enumerate(self.downsample):
                if len(gt_depth) < j + 1:
                    gt_depth.append([])
                h, w = (int(H / downsample), int(W / downsample))
                u = np.floor(U / downsample).astype(np.int32)
                v = np.floor(V / downsample).astype(np.int32)
                depth_map = np.ones([h, w], dtype=np.float32) * -1
                depth_map[v, u] = depths
                gt_depth[j].append(depth_map)

        input_dict["gt_depth"] = [np.stack(x) for x in gt_depth]
        return input_dict


@PIPELINES.register_module()
class GenerateProjected2DTargets(object):
    def __init__(self, min_depth=1e-3, min_size=2.0):
        self.min_depth = min_depth
        self.min_size = min_size

    def _post_process_coords(self, corner_coords, img_w, img_h):
        polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
        img_canvas = box(0, 0, img_w, img_h)
        if not polygon_from_2d_box.intersects(img_canvas):
            return None

        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        if img_intersection.is_empty:
            return None

        if hasattr(img_intersection, "geoms"):
            intersection_coords = np.concatenate(
                [np.array(geom.exterior.coords) for geom in img_intersection.geoms],
                axis=0,
            )
        else:
            intersection_coords = np.array(img_intersection.exterior.coords)

        min_x = np.min(intersection_coords[:, 0])
        min_y = np.min(intersection_coords[:, 1])
        max_x = np.max(intersection_coords[:, 0])
        max_y = np.max(intersection_coords[:, 1])
        return min_x, min_y, max_x, max_y

    def __call__(self, input_dict):
        num_cams = len(input_dict["lidar2img"])
        empty_boxes = np.zeros((0, 4), dtype=np.float32)
        empty_labels = np.zeros((0,), dtype=np.int64)
        empty_centers = np.zeros((0, 2), dtype=np.float32)
        empty_depths = np.zeros((0,), dtype=np.float32)

        gt_bboxes_3d = input_dict.get("gt_bboxes_3d")
        gt_labels_3d = input_dict.get("gt_labels_3d")
        if gt_bboxes_3d is None or gt_labels_3d is None or len(gt_bboxes_3d) == 0:
            input_dict["gt_bboxes"] = [empty_boxes.copy() for _ in range(num_cams)]
            input_dict["gt_labels"] = [empty_labels.copy() for _ in range(num_cams)]
            input_dict["centers2d"] = [empty_centers.copy() for _ in range(num_cams)]
            input_dict["depths"] = [empty_depths.copy() for _ in range(num_cams)]
            return input_dict

        centers_3d = gt_bboxes_3d[:, :3]
        corners_3d = box3d_to_corners(gt_bboxes_3d)
        centers_4d = np.concatenate(
            [centers_3d, np.ones((centers_3d.shape[0], 1), dtype=np.float32)],
            axis=-1,
        )
        corners_4d = np.concatenate(
            [corners_3d, np.ones((*corners_3d.shape[:2], 1), dtype=np.float32)],
            axis=-1,
        )

        gt_bboxes = []
        gt_labels = []
        centers2d = []
        depths = []
        for cam_idx, lidar2img in enumerate(input_dict["lidar2img"]):
            img_shape = input_dict["img_shape"][cam_idx]
            img_h, img_w = img_shape[:2]

            proj_centers = centers_4d @ lidar2img.T
            center_depth = proj_centers[:, 2]
            center_xy = proj_centers[:, :2] / np.clip(
                center_depth[:, None], a_min=self.min_depth, a_max=None
            )

            proj_corners = corners_4d @ lidar2img.T
            corner_depth = proj_corners[..., 2]
            valid_corners = corner_depth > self.min_depth
            proj_xy = proj_corners[..., :2] / np.clip(
                corner_depth[..., None], a_min=self.min_depth, a_max=None
            )

            center_visible = np.logical_and.reduce(
                [
                    center_depth > self.min_depth,
                    center_xy[:, 0] >= 0,
                    center_xy[:, 0] < img_w,
                    center_xy[:, 1] >= 0,
                    center_xy[:, 1] < img_h,
                ]
            )
            cam_boxes = []
            cam_labels = []
            cam_centers2d = []
            cam_depths = []
            for obj_idx in range(len(gt_labels_3d)):
                if not center_visible[obj_idx]:
                    continue
                if not np.any(valid_corners[obj_idx]):
                    continue

                corner_coords = proj_xy[obj_idx][valid_corners[obj_idx]].tolist()
                final_coords = self._post_process_coords(corner_coords, img_w, img_h)
                if final_coords is None:
                    continue

                x1, y1, x2, y2 = final_coords
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                if bbox_w < self.min_size or bbox_h < self.min_size:
                    continue

                cam_boxes.append([x1, y1, x2, y2])
                cam_labels.append(gt_labels_3d[obj_idx])
                cam_centers2d.append(center_xy[obj_idx])
                cam_depths.append(center_depth[obj_idx])

            gt_bboxes.append(
                np.array(cam_boxes, dtype=np.float32) if cam_boxes else empty_boxes.copy()
            )
            gt_labels.append(
                np.array(cam_labels, dtype=np.int64) if cam_labels else empty_labels.copy()
            )
            centers2d.append(
                np.array(cam_centers2d, dtype=np.float32)
                if cam_centers2d
                else empty_centers.copy()
            )
            depths.append(
                np.array(cam_depths, dtype=np.float32) if cam_depths else empty_depths.copy()
            )

        input_dict["gt_bboxes"] = gt_bboxes
        input_dict["gt_labels"] = gt_labels
        input_dict["centers2d"] = centers2d
        input_dict["depths"] = depths
        return input_dict


@PIPELINES.register_module()
class NuScenesSparse4DAdaptor(object):
    def __init(self):
        pass

    def __call__(self, input_dict):
        input_dict["projection_mat"] = np.float32(
            np.stack(input_dict["lidar2img"])
        )
        input_dict["image_wh"] = np.ascontiguousarray(
            np.array(input_dict["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
        )
        input_dict["T_global_inv"] = np.linalg.inv(input_dict["lidar2global"])
        input_dict["T_global"] = input_dict["lidar2global"]
        if "cam_intrinsic" in input_dict:
            input_dict["cam_intrinsic"] = np.float32(
                np.stack(input_dict["cam_intrinsic"])
            )
            input_dict["focal"] = input_dict["cam_intrinsic"][..., 0, 0]
        if "instance_inds" in input_dict:
            input_dict["instance_id"] = input_dict["instance_inds"]

        if "gt_bboxes_3d" in input_dict:
            input_dict["gt_bboxes_3d"][:, 6] = self.limit_period(
                input_dict["gt_bboxes_3d"][:, 6], offset=0.5, period=2 * np.pi
            )
            input_dict["gt_bboxes_3d"] = DC(
                to_tensor(input_dict["gt_bboxes_3d"]).float()
            )
        if "gt_labels_3d" in input_dict:
            input_dict["gt_labels_3d"] = DC(
                to_tensor(input_dict["gt_labels_3d"]).long()
            )
        if "gt_visibility" in input_dict:
            input_dict["gt_visibility"] = DC(
                to_tensor(input_dict["gt_visibility"]).float()
            )
        if "gt_occluded" in input_dict:
            input_dict["gt_occluded"] = DC(
                to_tensor(input_dict["gt_occluded"]).float()
            )
        for key in ["gt_bboxes", "gt_labels", "centers2d", "depths"]:
            if key not in input_dict:
                continue
            input_dict[key] = DC(
                [to_tensor(x) for x in input_dict[key]],
                stack=False,
                cpu_only=False,
            )

        imgs = [img.transpose(2, 0, 1) for img in input_dict["img"]]
        imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        input_dict["img"] = DC(to_tensor(imgs), stack=True)

        for key in [
            'gt_map_labels', 
            'gt_map_pts',
            'gt_agent_fut_trajs',
            'gt_agent_fut_masks',
        ]:
            if key not in input_dict:
                continue
            input_dict[key] = DC(to_tensor(input_dict[key]), stack=False, cpu_only=False) 

        for key in [
            'gt_ego_fut_trajs',
            'gt_ego_fut_masks',
            'gt_ego_fut_cmd',
            'ego_status',
        ]:
            if key not in input_dict:
                continue
            input_dict[key] = DC(to_tensor(input_dict[key]), stack=True, cpu_only=False, pad_dims=None)
        
        return input_dict

    def limit_period(
        self, val: np.ndarray, offset: float = 0.5, period: float = np.pi
    ) -> np.ndarray:
        limited_val = val - np.floor(val / period + offset) * period
        return limited_val


@PIPELINES.register_module()
class InstanceNameFilter(object):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict["gt_labels_3d"]
        gt_bboxes_mask = np.array(
            [n in self.labels for n in gt_labels_3d], dtype=np.bool_
        )
        input_dict["gt_bboxes_3d"] = input_dict["gt_bboxes_3d"][gt_bboxes_mask]
        input_dict["gt_labels_3d"] = input_dict["gt_labels_3d"][gt_bboxes_mask]
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][gt_bboxes_mask]
        if "gt_agent_fut_trajs" in input_dict:
            input_dict["gt_agent_fut_trajs"] = input_dict["gt_agent_fut_trajs"][gt_bboxes_mask]
            input_dict["gt_agent_fut_masks"] = input_dict["gt_agent_fut_masks"][gt_bboxes_mask]
        if "gt_visibility" in input_dict:
            input_dict["gt_visibility"] = input_dict["gt_visibility"][gt_bboxes_mask]
        if "gt_occluded" in input_dict:
            input_dict["gt_occluded"] = input_dict["gt_occluded"][gt_bboxes_mask]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(classes={self.classes})"
        return repr_str


@PIPELINES.register_module()
class CircleObjectRangeFilter(object):
    def __init__(
        self, class_dist_thred=[52.5] * 5 + [31.5] + [42] * 3 + [31.5]
    ):
        self.class_dist_thred = class_dist_thred

    def __call__(self, input_dict):
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]
        dist = np.sqrt(
            np.sum(gt_bboxes_3d[:, :2] ** 2, axis=-1)
        )
        mask = np.array([False] * len(dist))
        for label_idx, dist_thred in enumerate(self.class_dist_thred):
            mask = np.logical_or(
                mask,
                np.logical_and(gt_labels_3d == label_idx, dist <= dist_thred),
            )

        gt_bboxes_3d = gt_bboxes_3d[mask]
        gt_labels_3d = gt_labels_3d[mask]

        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][mask]
        if "gt_agent_fut_trajs" in input_dict:
            input_dict["gt_agent_fut_trajs"] = input_dict["gt_agent_fut_trajs"][mask]
            input_dict["gt_agent_fut_masks"] = input_dict["gt_agent_fut_masks"][mask]
        if "gt_visibility" in input_dict:
            input_dict["gt_visibility"] = input_dict["gt_visibility"][mask]
        if "gt_occluded" in input_dict:
            input_dict["gt_occluded"] = input_dict["gt_occluded"][mask]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(class_dist_thred={self.class_dist_thred})"
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str
