"""One-shot generator for speed-stratified plan k-means anchors.

Splits the train GT trajectories into low-speed (<3 m/s) and high-speed
(>=3 m/s) buckets using ||ego_status[6:8]|| (XY velocity) at the current
frame. Runs K=6 k-means per bucket per cmd. Produces two files of shape
(3, 6, 6, 2):
  data/kmeans/kmeans_plan_6_lowspeed.npy
  data/kmeans/kmeans_plan_6_highspeed.npy

Concatenated at MotionPlanningHead init via plan_anchor=[low, high].
"""
import os

import mmcv
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

K = 6
SPEED_THRESH = 3.0  # m/s
FP = "data/infos/nuscenes_infos_train.pkl"
OUT_LOW = f"data/kmeans/kmeans_plan_{K}_lowspeed.npy"
OUT_HIGH = f"data/kmeans/kmeans_plan_{K}_highspeed.npy"

data = mmcv.load(FP)
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))

# navi_trajs[bucket][cmd] = list of (6, 2) arrays
navi_trajs = [[[], [], []], [[], [], []]]
for info in tqdm(data_infos):
    plan_traj = info["gt_ego_fut_trajs"].cumsum(axis=-2)
    plan_mask = info["gt_ego_fut_masks"]
    cmd = int(info["gt_ego_fut_cmd"].astype(np.int32).argmax(axis=-1))
    if plan_mask.sum() != 6:
        continue
    speed = float(np.linalg.norm(np.asarray(info["ego_status"])[6:8]))
    bucket = 0 if speed < SPEED_THRESH else 1
    navi_trajs[bucket][cmd].append(plan_traj)

for bucket_idx, (bucket_name, out_path) in enumerate(
    [("lowspeed", OUT_LOW), ("highspeed", OUT_HIGH)]
):
    clusters = []
    for cmd_idx, trajs in enumerate(navi_trajs[bucket_idx]):
        if len(trajs) < K:
            raise RuntimeError(
                f"bucket {bucket_name} cmd {cmd_idx} has only {len(trajs)} "
                f"trajectories; need at least K={K}"
            )
        flat = np.concatenate(trajs, axis=0).reshape(-1, 12)
        cluster = (
            KMeans(n_clusters=K, n_init=10, random_state=0)
            .fit(flat)
            .cluster_centers_
        )
        cluster = cluster.reshape(-1, 6, 2)
        print(f"{bucket_name} cmd {cmd_idx}: {flat.shape[0]} trajs -> {K} clusters")
        clusters.append(cluster)

    clusters = np.stack(clusters, axis=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, clusters)
    print("saved", out_path, "shape", clusters.shape)
