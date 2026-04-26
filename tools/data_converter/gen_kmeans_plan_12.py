"""One-shot generator for kmeans_plan_12.npy.

Mirrors tools/kmeans/kmeans_plan.py with K=12 on the full GT distribution
(no diversity reg, no speed stratification). Produces an array shaped
(3, 12, 6, 2) saved to data/kmeans/kmeans_plan_12.npy.
"""
import os

import mmcv
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

K = 12
FP = "data/infos/nuscenes_infos_train.pkl"
OUT = f"data/kmeans/kmeans_plan_{K}.npy"

data = mmcv.load(FP)
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
navi_trajs = [[], [], []]
for info in tqdm(data_infos):
    plan_traj = info["gt_ego_fut_trajs"].cumsum(axis=-2)
    plan_mask = info["gt_ego_fut_masks"]
    cmd = int(info["gt_ego_fut_cmd"].astype(np.int32).argmax(axis=-1))
    if plan_mask.sum() != 6:
        continue
    navi_trajs[cmd].append(plan_traj)

clusters = []
for cmd_idx, trajs in enumerate(navi_trajs):
    flat = np.concatenate(trajs, axis=0).reshape(-1, 12)
    cluster = KMeans(n_clusters=K, n_init=10, random_state=0).fit(flat).cluster_centers_
    cluster = cluster.reshape(-1, 6, 2)
    print(f"cmd {cmd_idx}: {flat.shape[0]} trajs -> {K} clusters")
    clusters.append(cluster)

clusters = np.stack(clusters, axis=0)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.save(OUT, clusters)
print("saved", OUT, "shape", clusters.shape)
