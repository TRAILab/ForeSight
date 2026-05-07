import os
import numpy as np
import cv2
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix

from tools.visualization.bev_render import (
    color_mapping, 
    SCORE_THRESH, 
    MAP_SCORE_THRESH,
    CMD_LIST
)


CAM_NAMES_NUSC = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
]
CAM_NAMES_NUSC_converter = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]

class CamRender:
    def __init__(
        self, 
        plot_choices,
        out_dir,
    ):
        self.plot_choices = plot_choices
        self.pred_dir = os.path.join(out_dir, "cam_pred")
        os.makedirs(self.pred_dir, exist_ok=True)

    def reset_canvas(self):
        plt.close()
        plt.gca().set_axis_off()
        plt.axis('off')
        self.fig, self.axes = plt.subplots(2, 3, figsize=(160 /3 , 20))
        plt.tight_layout()

    def render(
        self,
        data,
        result,
        index,
    ):
        self.reset_canvas()
        self.render_image_data(data, index)
        self.draw_detection_pred(data, result)
        self.draw_motion_pred(data, result)
        self.draw_planning_pred(data, result)
        save_path = os.path.join(self.pred_dir, str(index).zfill(4) + '.jpg')
        self.save_fig(save_path)
        return save_path

    def render_plan_only(self, data, result, index, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.reset_canvas()
        self.render_image_data(data, index)
        self.draw_detection_gt(data)
        self.draw_motion_gt(data)
        self.draw_planning_pred(data, result)
        self.draw_planning_gt(data)
        self._render_cam_legend()
        save_path = os.path.join(out_dir, str(index).zfill(4) + '.jpg')
        self.save_fig(save_path)
        return save_path

    def _render_cam_legend(self):
        ax = self.axes[0, 1]
        items = [
            ('Pred plan', (1.0, 0.5, 0.0)),
            ('GT plan',   (0.1, 0.1, 0.1)),
            ('GT motion', (0.0, 0.4, 0.8)),
        ]
        for k, (name, color) in enumerate(items):
            ax.scatter([60], [60 + k * 60], color=color, s=180,
                       edgecolors='white', linewidths=2)
            ax.text(95, 60 + k * 60, name, fontsize=18, color='white',
                    va='center',
                    bbox=dict(facecolor='black', alpha=0.5, pad=2))

    def draw_detection_gt(self, data):
        bboxes = data['gt_bboxes_3d']
        if hasattr(bboxes, 'numpy'):
            bboxes = bboxes.numpy()
        bboxes = np.asarray(bboxes)
        labels = np.asarray(data['gt_labels_3d'])
        for j, cam in enumerate(CAM_NAMES_NUSC):
            idx = CAM_NAMES_NUSC_converter.index(cam)
            cam_intrinsic = data['cam_intrinsic'][idx]
            extrinsic = data['lidar2cam'][idx]
            trans = extrinsic[3, :3]
            rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
            imsize = (1600, 900)
            for i in range(bboxes.shape[0]):
                if labels[i] == -1:
                    continue
                color = color_mapping[i % len(color_mapping)]
                center = bboxes[i, 0:3]
                nusc_dims = bboxes[i, 3:6][..., [1, 0, 2]]
                quat = Quaternion(axis=[0, 0, 1], radians=bboxes[i, 6])
                box = NuScenesBox(center, nusc_dims, quat)
                box.rotate(rot)
                box.translate(trans)
                if box_in_image(box, cam_intrinsic, imsize):
                    box.render(self.axes[j // 3, j % 3], view=cam_intrinsic,
                               normalize=True, colors=(color, color, color), linewidth=4)
            self.axes[j // 3, j % 3].set_xlim(0, imsize[0])
            self.axes[j // 3, j % 3].set_ylim(imsize[1], 0)

    def draw_motion_gt(self, data):
        bboxes = np.asarray(data['gt_bboxes_3d'])
        labels = np.asarray(data['gt_labels_3d'])
        fut_trajs = np.asarray(data['gt_agent_fut_trajs'])
        fut_masks = np.asarray(data['gt_agent_fut_masks'])
        for j, cam in enumerate(CAM_NAMES_NUSC):
            idx = CAM_NAMES_NUSC_converter.index(cam)
            cam_intrinsic = data['cam_intrinsic'][idx]
            extrinsic = data['lidar2cam'][idx]
            trans = extrinsic[3, :3]
            rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
            imsize = (1600, 900)
            for i in range(bboxes.shape[0]):
                if labels[i] == -1:
                    continue
                masks = fut_masks[i].astype(bool)
                if not masks[0]:
                    continue
                color = color_mapping[i % len(color_mapping)]
                center_xy = bboxes[i, :2]
                trajs = fut_trajs[i][masks].cumsum(axis=0) + center_xy
                trajs = np.concatenate([center_xy.reshape(1, 2), trajs], axis=0)
                z = np.full((trajs.shape[0], 1), bboxes[i, 2] - bboxes[i, 5] / 2)
                traj3d = np.concatenate([trajs, z], axis=1)
                quat = Quaternion(axis=[0, 0, 1], radians=bboxes[i, 6])
                box = NuScenesBox(bboxes[i, 0:3], bboxes[i, 3:6][..., [1, 0, 2]], quat)
                box.rotate(rot); box.translate(trans)
                if not box_in_image(box, cam_intrinsic, imsize):
                    continue
                traj_points = traj3d @ extrinsic[:3, :3] + trans
                self._render_traj(traj_points, cam_intrinsic, j, color=tuple(color), s=15)

    def draw_planning_gt(self, data):
        masks = np.asarray(data['gt_ego_fut_masks']).astype(bool)
        if not masks[0]:
            return
        plan = np.asarray(data['gt_ego_fut_trajs'])[masks]
        plan[np.abs(plan) < 0.01] = 0.0
        plan = plan.cumsum(axis=0)
        plan = np.concatenate([np.zeros((1, 2)), plan], axis=0)
        z = np.full((plan.shape[0], 1), -1.8)
        plan = np.concatenate([plan, z], axis=1)
        idx = 0
        cam_intrinsic = data['cam_intrinsic'][idx]
        extrinsic = data['lidar2cam'][idx]
        trans = extrinsic[3, :3]
        traj_points = plan @ extrinsic[:3, :3] + trans
        self._render_traj_gradient(traj_points, cam_intrinsic, j=1,
                                   colormap='Greys', cmap_range=(0.4, 1.0),
                                   s=140)

    def _render_traj_gradient(self, traj_points, cam_intrinsic, j,
                              colormap='Greys', cmap_range=(0.0, 1.0),
                              s=120, points_per_step=10):
        total_steps = (len(traj_points) - 1) * points_per_step + 1
        total_xy = np.zeros((total_steps, 3))
        for k in range(total_steps - 1):
            unit_vec = traj_points[k // points_per_step + 1] - \
                       traj_points[k // points_per_step]
            total_xy[k] = (k / points_per_step - k // points_per_step) * \
                          unit_vec + traj_points[k // points_per_step]
        total_xy[-1] = traj_points[-1]
        in_range_mask = total_xy[:, 2] > 0.1
        proj = view_points(total_xy.T, cam_intrinsic, normalize=True)[:2, :]
        proj = proj[:, in_range_mask]
        colors = matplotlib.colormaps[colormap](
            np.linspace(cmap_range[0], cmap_range[1], total_steps))[:, :3]
        colors = colors[in_range_mask]
        self.axes[j // 3, j % 3].scatter(proj[0], proj[1], c=colors, s=s)

    def load_image(self, data_path, cam):
        """Update the axis of the plot with the provided image."""
        image = np.array(Image.open(data_path))
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 60)
        fontScale = 2
        color = (0, 0, 0)
        thickness = 4
        return cv2.putText(image, cam, org, font, fontScale, color, thickness, cv2.LINE_AA)

    def update_image(self, image, index, cam):
        """Render image data for each camera."""
        ax = self.get_axis(index)
        ax.imshow(image)
        plt.axis('off')
        ax.axis('off')
        ax.grid(False)

    def get_axis(self, index):
        """Retrieve the corresponding axis based on the index."""
        return self.axes[index//3, index % 3]

    def save_fig(self, filename):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(filename)

    def render_image_data(self, data, index):
        """Load and annotate image based on the provided path."""
        for i, cam in enumerate(CAM_NAMES_NUSC):
            idx = CAM_NAMES_NUSC_converter.index(cam)
            img_path = data['img_filename'][idx]
            image = self.load_image(img_path, cam)
            self.update_image(image, i, cam)
    
    def draw_detection_pred(self, data, result):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['det'] and "boxes_3d" in result):
            return

        bboxes = result['boxes_3d'].numpy()
        for j, cam in enumerate(CAM_NAMES_NUSC):
            idx = CAM_NAMES_NUSC_converter.index(cam)
            cam_intrinsic = data['cam_intrinsic'][idx]
            lidar2cam = data['lidar2cam']
            extrinsic = lidar2cam[idx]
            trans = extrinsic[3, :3]
            rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
            imsize = (1600, 900)

            for i in range(result['labels_3d'].shape[0]):
                score = result['scores_3d'][i]
                if score < SCORE_THRESH: 
                    continue
                color = color_mapping[result['instance_ids'][i] % len(color_mapping)]
                
                center = bboxes[i, 0 : 3]
                box_dims = bboxes[i, 3 : 6]
                nusc_dims = box_dims[..., [1, 0, 2]]
                quat = Quaternion(axis=[0, 0, 1], radians=bboxes[i, 6])
                box = NuScenesBox(
                    center,
                    nusc_dims,
                    quat
                )
                box.rotate(rot)
                box.translate(trans)
                if box_in_image(box, cam_intrinsic, imsize):
                    box.render(
                        self.axes[j // 3, j % 3], 
                        view=cam_intrinsic, 
                        normalize=True, 
                        colors=(color, color, color),
                        linewidth=4,
                    )
            
            self.axes[j//3, j % 3].set_xlim(0, imsize[0])
            self.axes[j//3, j % 3].set_ylim(imsize[1], 0)

    def draw_motion_pred(self, data, result, points_per_step=10):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['motion'] and "trajs_3d" in result):
            return

        bboxes = result['boxes_3d'].numpy()
        for j, cam in enumerate(CAM_NAMES_NUSC):
            idx = CAM_NAMES_NUSC_converter.index(cam)
            cam_intrinsic = data['cam_intrinsic'][idx]
            lidar2cam = data['lidar2cam']
            extrinsic = lidar2cam[idx]
            trans = extrinsic[3, :3]
            rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
            imsize = (1600, 900)

            for i in range(result['labels_3d'].shape[0]):
                score = result['scores_3d'][i]
                if score < SCORE_THRESH: 
                    continue
                color = color_mapping[result['instance_ids'][i] % len(color_mapping)]
                
                traj_score = result['trajs_score'][i].numpy()
                traj = result['trajs_3d'][i].numpy()
                
                mode_idx = traj_score.argmax()
                traj = traj[mode_idx]
                origin = bboxes[i, :2][None]
                traj = np.concatenate([origin, traj], axis=0)
                traj_expand = np.ones((traj.shape[0], 1)) 
                traj_expand[:] = bboxes[i, 2] - bboxes[i, 5] / 2
                traj = np.concatenate([traj, traj_expand], axis=1)

                center = bboxes[i, 0 : 3]
                box_dims = bboxes[i, 3 : 6]
                nusc_dims = box_dims[..., [1, 0, 2]]
                quat = Quaternion(axis=[0, 0, 1], radians=bboxes[i, 6])
                box = NuScenesBox(
                    center,
                    nusc_dims,
                    quat
                )
                box.rotate(rot)
                box.translate(trans)
                if not box_in_image(box, cam_intrinsic, imsize):
                    continue
                traj_points = traj @ extrinsic[:3, :3] + trans
                self._render_traj(traj_points, cam_intrinsic, j, color=color, s=15)

        
    def draw_planning_pred(self, data, result):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['planning'] and "planning" in result):
            return
        # for j, cam in enumerate(CAM_NAMES_NUSC[1]):
        #     idx = CAM_NAMES_NUSC_converter.index(cam)
        #     cam_intrinsic = data['cam_intrinsic'][idx]
        #     lidar2cam = data['lidar2cam']
        #     extrinsic = lidar2cam[idx]
        #     trans = extrinsic[3, :3]
        #     rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
        #     imsize = (1600, 900)

        #     plan_trajs = result['planning'][0].cpu().numpy()
        #     plan_trajs = plan_trajs.reshape(3, -1, 6, 2)
        #     num_cmd = len(CMD_LIST)
        #     num_mode = plan_trajs.shape[1]
        #     plan_trajs = np.concatenate((np.zeros((num_cmd, num_mode, 1, 2)), plan_trajs), axis=2)
        #     plan_trajs = plan_trajs.cumsum(axis=-2)
        #     plan_score = result['planning_score'][0].cpu().numpy()
        #     plan_score = plan_score.reshape(3, -1)

        #     cmd = data['gt_ego_fut_cmd'].argmax()
        #     plan_trajs = plan_trajs[cmd]
        #     plan_score = plan_score[cmd]

        #     mode_idx = plan_score.argmax()
        #     plan_traj = plan_trajs[mode_idx]
        #     traj_expand = np.ones((plan_traj.shape[0], 1)) * -2
        #     # traj_expand[:] = bboxes[i, 2] - bboxes[i, 5] / 2
        #     plan_traj = np.concatenate([plan_traj, traj_expand], axis=1)

        #     traj_points = plan_traj @ extrinsic[:3, :3] + trans
        #     self._render_traj(traj_points, cam_intrinsic, j)

        idx = 0 ## front camera
        cam_intrinsic = data['cam_intrinsic'][idx]
        lidar2cam = data['lidar2cam']
        extrinsic = lidar2cam[idx]
        trans = extrinsic[3, :3]
        rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
        # plan_trajs = result['planning'][0].cpu().numpy()
        # plan_trajs = plan_trajs.reshape(3, -1, 6, 2)
        # num_cmd = len(CMD_LIST)
        # num_mode = plan_trajs.shape[1]
        # plan_trajs = np.concatenate((np.zeros((num_cmd, num_mode, 1, 2)), plan_trajs), axis=2)
        # plan_trajs = plan_trajs.cumsum(axis=-2)
        # plan_score = result['planning_score'][0].cpu().numpy()
        # plan_score = plan_score.reshape(3, -1)

        # cmd = data['gt_ego_fut_cmd'].argmax()
        # plan_trajs = plan_trajs[cmd]
        # plan_score = plan_score[cmd]

        # mode_idx = plan_score.argmax()
        # plan_traj = plan_trajs[mode_idx]
        plan_traj = result["final_planning"]
        plan_traj = np.concatenate((np.zeros((1, 2)), plan_traj), axis=0)
        traj_expand = np.ones((plan_traj.shape[0], 1)) * -1.8
        plan_traj = np.concatenate([plan_traj, traj_expand], axis=1)

        traj_points = plan_traj @ extrinsic[:3, :3] + trans
        self._render_traj(traj_points, cam_intrinsic, j=1)

    def _render_traj(self, traj_points, cam_intrinsic, j, color=(1, 0.5, 0), s=150, points_per_step=10):
        total_steps = (len(traj_points)-1) * points_per_step + 1
        total_xy = np.zeros((total_steps, 3))
        for k in range(total_steps-1):
            unit_vec = traj_points[k//points_per_step +
                                    1] - traj_points[k//points_per_step]
            total_xy[k] = (k/points_per_step - k//points_per_step) * \
                unit_vec + traj_points[k//points_per_step]
        in_range_mask = total_xy[:, 2] > 0.1
        traj_points = view_points(
            total_xy.T, cam_intrinsic, normalize=True)[:2, :]
        traj_points = traj_points[:2, in_range_mask]
        self.axes[j // 3, j % 3].scatter(traj_points[0], traj_points[1], color=color, s=s)