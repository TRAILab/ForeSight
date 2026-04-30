"""Endpoint-normalized plan k-means anchors (variant 1a).

Clusters `traj_cumsum / max(||traj_end||, EPS)` per cmd with K=6, producing
shape anchors that decouple maneuver template from trajectory magnitude.

Saves three files:
  data/kmeans/kmeans_plan_shape_endptnorm.npy           (3, 6, 6, 2) shape anchors
  data/kmeans/kmeans_plan_shape_endptnorm_refmag.npy    (3, 6) per-cluster median ||end||
  reports/kmeans_plan_shape_endptnorm.png               cluster-center plot per cmd

The reference magnitude file is consumed at MotionPlanningHead init when
plan_anchor_norm_mode='endptnorm' so the runtime can reconstruct metric
sampling positions as shape_anchor * reference_magnitude_per_mode.
EPS is 1.0 m, picked from the histogram elbow in
reports/anchor_norm_histograms.png.
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

K = 6
EPS = 1.0  # meters; floor on ||traj_end|| for normalization
FP = "data/infos/nuscenes_infos_train.pkl"
OUT_ANCHOR = "data/kmeans/kmeans_plan_shape_endptnorm.npy"
OUT_REFMAG = "data/kmeans/kmeans_plan_shape_endptnorm_refmag.npy"
OUT_PLOT = "reports/kmeans_plan_shape_endptnorm.png"

with open(FP, "rb") as f:
    data = pickle.load(f)
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))

# navi_trajs[cmd] = list of (6, 2) cumulative XY arrays
# end_norms[cmd] = list of float ||end|| for the same trajectories
navi_trajs = [[], [], []]
end_norms_per_cmd = [[], [], []]
for info in tqdm(data_infos):
    plan_mask = info["gt_ego_fut_masks"]
    if plan_mask.sum() != 6:
        continue
    cum = info["gt_ego_fut_trajs"].cumsum(axis=-2)  # (6, 2)
    cmd = int(info["gt_ego_fut_cmd"].astype(np.int32).argmax(axis=-1))
    end_norm = float(np.linalg.norm(cum[-1]))
    navi_trajs[cmd].append(cum)
    end_norms_per_cmd[cmd].append(end_norm)

cmd_names = {0: "Right", 1: "Left", 2: "Straight"}
clusters = []
ref_mags = []
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for cmd_idx, (trajs, ends) in enumerate(zip(navi_trajs, end_norms_per_cmd)):
    if len(trajs) < K:
        raise RuntimeError(
            f"cmd {cmd_idx} has only {len(trajs)} trajectories; need K={K}"
        )
    cum_arr = np.stack(trajs, axis=0)             # (n, 6, 2)
    end_arr = np.asarray(ends)                    # (n,)
    scale = np.maximum(end_arr, EPS)              # (n,)
    shape = cum_arr / scale[:, None, None]        # (n, 6, 2)

    flat = shape.reshape(-1, 12)
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(flat)
    centers = km.cluster_centers_.reshape(-1, 6, 2)  # (K, 6, 2)
    labels = km.labels_                           # (n,)

    # Cluster-median raw endpoint magnitude, in cluster-id order matching `centers`.
    median_mag = np.zeros(K, dtype=np.float32)
    for k in range(K):
        median_mag[k] = float(np.median(end_arr[labels == k])) if (labels == k).any() else 0.0
    print(f"cmd {cmd_idx} ({cmd_names[cmd_idx]}): {flat.shape[0]} trajs -> {K} clusters")
    for k in range(K):
        endpt = centers[k, -1]
        n_k = int((labels == k).sum())
        print(f"  k={k}: shape_endpoint=({endpt[0]:+.3f}, {endpt[1]:+.3f})"
              f"  med_mag={median_mag[k]:6.2f}  n={n_k}")

    clusters.append(centers)
    ref_mags.append(median_mag)

    ax = axes[cmd_idx]
    for k in range(K):
        x = np.concatenate([[0], centers[k, :, 0]])
        y = np.concatenate([[0], centers[k, :, 1]])
        ax.plot(x, y, marker="o", markersize=3,
                label=f"k={k}  m̃={median_mag[k]:.1f}m")
    ax.set_title(f"cmd={cmd_names[cmd_idx]}  (n={flat.shape[0]})")
    ax.set_xlabel("x_lidar (right, normalized)")
    if cmd_idx == 0:
        ax.set_ylabel("y_lidar (forward, normalized)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")

clusters = np.stack(clusters, axis=0).astype(np.float32)  # (3, 6, 6, 2)
ref_mags = np.stack(ref_mags, axis=0).astype(np.float32)   # (3, 6)

os.makedirs(os.path.dirname(OUT_ANCHOR), exist_ok=True)
os.makedirs(os.path.dirname(OUT_PLOT), exist_ok=True)
np.save(OUT_ANCHOR, clusters)
np.save(OUT_REFMAG, ref_mags)
fig.suptitle(f"Endpoint-normalized plan anchors (K={K} per cmd, ε={EPS} m)")
fig.tight_layout()
fig.savefig(OUT_PLOT, dpi=130)
print("saved", OUT_ANCHOR, "shape", clusters.shape)
print("saved", OUT_REFMAG, "shape", ref_mags.shape)
print("saved", OUT_PLOT)
