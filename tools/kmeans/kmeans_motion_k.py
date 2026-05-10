"""Parameterized version of kmeans_motion.py.

Usage: python tools/kmeans/kmeans_motion_k.py <K>

Produces data/kmeans/kmeans_motion_<K>.npy with shape
(num_classes, K, 12, 2). Skips matplotlib viz so it runs headless.
"""
import sys

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

import mmcv

CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]


def lidar2agent(trajs_offset, boxes):
    origin = np.zeros((trajs_offset.shape[0], 1, 2), dtype=np.float32)
    trajs_offset = np.concatenate([origin, trajs_offset], axis=1)
    trajs = trajs_offset.cumsum(axis=1)
    yaws = -boxes[:, 6]
    rot_sin = np.sin(yaws)
    rot_cos = np.cos(yaws)
    rot_mat_T = np.stack(
        [np.stack([rot_cos, rot_sin]), np.stack([-rot_sin, rot_cos])]
    )
    trajs_new = np.einsum('aij,jka->aik', trajs, rot_mat_T)
    trajs_new = trajs_new[:, 1:]
    return trajs_new


def main():
    K = int(sys.argv[1])
    DIS_THRESH = 55

    fp = 'data/infos/nuscenes_infos_train.pkl'
    data = mmcv.load(fp)
    data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))

    intention = {i: [] for i in range(len(CLASSES))}
    for info in tqdm(data_infos):
        boxes = info['gt_boxes']
        names = info['gt_names']
        fut_masks = info['gt_agent_fut_masks']
        trajs = info['gt_agent_fut_trajs']
        labels = np.array([
            CLASSES.index(c) if c in CLASSES else -1 for c in names
        ])
        if len(boxes) == 0:
            continue
        for i in range(len(CLASSES)):
            cls_mask = (labels == i)
            box_cls = boxes[cls_mask]
            fut_masks_cls = fut_masks[cls_mask]
            trajs_cls = trajs[cls_mask]
            distance = np.linalg.norm(box_cls[:, :2], axis=1)
            mask = np.logical_and(
                fut_masks_cls.sum(axis=1) == 12,
                distance < DIS_THRESH,
            )
            trajs_cls = trajs_cls[mask]
            box_cls = box_cls[mask]
            trajs_agent = lidar2agent(trajs_cls, box_cls)
            if trajs_agent.shape[0] == 0:
                continue
            intention[i].append(trajs_agent)

    clusters = []
    for i in range(len(CLASSES)):
        if not intention[i]:
            raise RuntimeError(f'No trajectories for class {CLASSES[i]}')
        intention_cls = np.concatenate(intention[i], axis=0).reshape(-1, 24)
        if intention_cls.shape[0] < K:
            raise RuntimeError(
                f'Class {CLASSES[i]} has {intention_cls.shape[0]} trajs < K={K}'
            )
        # KMeans with K=1 returns the mean as the single centroid.
        cluster = KMeans(n_clusters=K, n_init=10, random_state=0).fit(
            intention_cls
        ).cluster_centers_
        cluster = cluster.reshape(-1, 12, 2)
        clusters.append(cluster)
        print(
            f'{CLASSES[i]}: N={intention_cls.shape[0]}, '
            f'centroids shape={cluster.shape}'
        )

    clusters = np.stack(clusters, axis=0)
    out = f'data/kmeans/kmeans_motion_{K}.npy'
    np.save(out, clusters)
    print(f'Wrote {out} shape={clusters.shape}')


if __name__ == '__main__':
    main()
