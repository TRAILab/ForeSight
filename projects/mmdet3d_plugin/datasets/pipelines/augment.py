import torch

import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from PIL import Image
from shapely.affinity import affine_transform as shapely_affine_transform


@PIPELINES.register_module()
class ResizeCropFlipImage(object):
    def __call__(self, results):
        aug_config = results.get("aug_config")
        if aug_config is None:
            return results
        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        for i in range(N):
            img, mat = self._img_transform(
                np.uint8(imgs[i]), aug_config,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            results["lidar2img"][i] = mat @ results["lidar2img"][i]
            if "cam_intrinsic" in results:
                results["cam_intrinsic"][i][:3, :3] *= aug_config["resize"]
                # results["cam_intrinsic"][i][:3, :3] = (
                #     mat[:3, :3] @ results["cam_intrinsic"][i][:3, :3]
                # )

        results["img"] = new_imgs
        results["img_shape"] = [x.shape[:2] for x in new_imgs]
        return results

    def _img_transform(self, img, aug_configs):
        H, W = img.shape[:2]
        resize = aug_configs.get("resize", 1)
        resize_dims = (int(W * resize), int(H * resize))
        crop = aug_configs.get("crop", [0, 0, *resize_dims])
        flip = aug_configs.get("flip", False)
        rotate = aug_configs.get("rotate", 0)

        origin_dtype = img.dtype
        if origin_dtype != np.uint8:
            min_value = img.min()
            max_vaule = img.max()
            scale = 255 / (max_vaule - min_value)
            img = (img - min_value) * scale
            img = np.uint8(img)
        img = Image.fromarray(img)
        img = img.resize(resize_dims).crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        img = np.array(img).astype(np.float32)
        if origin_dtype != np.uint8:
            img = img.astype(np.float32)
            img = img / scale + min_value

        transform_matrix = np.eye(3)
        transform_matrix[:2, :2] *= resize
        transform_matrix[:2, 2] -= np.array(crop[:2])
        if flip:
            flip_matrix = np.array(
                [[-1, 0, crop[2] - crop[0]], [0, 1, 0], [0, 0, 1]]
            )
            transform_matrix = flip_matrix @ transform_matrix
        rotate = rotate / 180 * np.pi
        rot_matrix = np.array(
            [
                [np.cos(rotate), np.sin(rotate), 0],
                [-np.sin(rotate), np.cos(rotate), 0],
                [0, 0, 1],
            ]
        )
        rot_center = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        rot_matrix[:2, 2] = -rot_matrix[:2, :2] @ rot_center + rot_center
        transform_matrix = rot_matrix @ transform_matrix
        extend_matrix = np.eye(4)
        extend_matrix[:3, :3] = transform_matrix
        return img, extend_matrix


@PIPELINES.register_module()
class BBoxRotation(object):
    def __call__(self, results):
        angle = results["aug_config"]["rotate_3d"]
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)

        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (
                results["lidar2img"][view] @ rot_mat_inv
            )
        if "lidar2global" in results:
            results["lidar2global"] = results["lidar2global"] @ rot_mat_inv
        if "gt_bboxes_3d" in results:
            results["gt_bboxes_3d"] = self.box_rotate(
                results["gt_bboxes_3d"], angle
            )

        # 2D rotation matrix (row-vector convention: v' = v @ rot_mat_T_2d)
        rot_mat_T_2d = np.array(
            [[rot_cos, rot_sin], [-rot_sin, rot_cos]]
        )

        if "map_geoms" in results:
            results["map_geoms"] = self.geoms_rotate(results["map_geoms"], rot_cos, rot_sin)

        if "gt_agent_fut_trajs" in results:
            results["gt_agent_fut_trajs"] = self.traj_rotate(
                results["gt_agent_fut_trajs"], rot_mat_T_2d
            )

        if "gt_ego_fut_trajs" in results:
            results["gt_ego_fut_trajs"] = self.traj_rotate(
                results["gt_ego_fut_trajs"], rot_mat_T_2d
            )
            # Recompute ego command from cumulative rotated trajectory.
            # The command is determined by the X offset of the final waypoint
            # (>= 2m = right, <= -2m = left, else straight). After rotation a
            # straight forward path acquires a large X component, so the stored
            # command is no longer valid.
            if "gt_ego_fut_cmd" in results:
                final_x = float(np.cumsum(results["gt_ego_fut_trajs"], axis=0)[-1, 0])
                if final_x >= 2.0:
                    results["gt_ego_fut_cmd"] = np.array([1, 0, 0], dtype=np.float32)
                elif final_x <= -2.0:
                    results["gt_ego_fut_cmd"] = np.array([0, 1, 0], dtype=np.float32)
                else:
                    results["gt_ego_fut_cmd"] = np.array([0, 0, 1], dtype=np.float32)

        if "ego_status" in results:
            results["ego_status"] = self.ego_status_rotate(
                results["ego_status"], rot_mat_T_2d
            )

        return results

    @staticmethod
    def box_rotate(bbox_3d, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat_T = np.array(
            [[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]]
        )
        bbox_3d[:, :3] = bbox_3d[:, :3] @ rot_mat_T
        bbox_3d[:, 6] += angle
        if bbox_3d.shape[-1] > 7:
            vel_dims = bbox_3d[:, 7:].shape[-1]
            bbox_3d[:, 7:] = bbox_3d[:, 7:] @ rot_mat_T[:vel_dims, :vel_dims]
        return bbox_3d

    @staticmethod
    def geoms_rotate(map_geoms, rot_cos, rot_sin):
        """Rotate Shapely map geometries (LineStrings) about the Z-axis."""
        # shapely affine_transform 2D: [a, b, d, e, xoff, yoff]
        # x' = a*x + b*y + xoff,  y' = d*x + e*y + yoff
        matrix = [rot_cos, -rot_sin, rot_sin, rot_cos, 0, 0]
        rotated = {}
        for label, geom_list in map_geoms.items():
            rotated[label] = [shapely_affine_transform(g, matrix) for g in geom_list]
        return rotated

    @staticmethod
    def traj_rotate(trajs, rot_mat_T_2d):
        """Rotate 2D delta trajectory vectors (dtype-preserving).

        Args:
            trajs: ndarray (..., 2), row vectors in LiDAR XY frame.
            rot_mat_T_2d: (2, 2) transposed rotation matrix (float64).
        Returns:
            Rotated ndarray with the same dtype as `trajs`.
        """
        dtype = trajs.dtype
        rot = rot_mat_T_2d.astype(dtype)
        result = trajs.copy()
        result[..., :2] = trajs[..., :2] @ rot
        return result

    @staticmethod
    def ego_status_rotate(ego_status, rot_mat_T_2d):
        """Rotate XY acceleration (indices 0:2) and XY velocity (indices 6:8)
        in ego_status, which are expressed in the LiDAR-aligned ego frame."""
        dtype = ego_status.dtype
        rot = rot_mat_T_2d.astype(dtype)
        result = ego_status.copy()
        result[0:2] = ego_status[0:2] @ rot
        result[6:8] = ego_status[6:8] @ rot
        return result


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(
                    -self.brightness_delta, self.brightness_delta
                )
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results["img"] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str
