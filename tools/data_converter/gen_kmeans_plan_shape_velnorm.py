"""Velocity-normalized plan k-means anchors (variant 1b).

Clusters `traj_cumsum / max(||v_0||, EPS_V)` per cmd with K=6, where v_0 =
ego_status[6:8] (CAN-frame planar velocity). Anchor shapes have units of
seconds; multiplied by per-batch ||v_0|| at runtime to recover meters.

Saves two files:
  data/kmeans/kmeans_plan_shape_velnorm.npy    (3, 6, 6, 2) shape anchors (units: seconds)
  reports/kmeans_plan_shape_velnorm.png        cluster-center plot per cmd

No reference magnitudes are saved — runtime always uses v_0 from ego_status.
EPS_V is 1.0 m/s, picked from the histogram elbow in
reports/anchor_norm_histograms.png.
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

K = 6
EPS_V = 1.0  # m/s; floor on ||v_0|| for normalization
FP = "data/infos/nuscenes_infos_train.pkl"
OUT_ANCHOR = "data/kmeans/kmeans_plan_shape_velnorm.npy"
OUT_PLOT = "reports/kmeans_plan_shape_velnorm.png"

with open(FP, "rb") as f:
    data = pickle.load(f)
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))

navi_trajs = [[], [], []]
v0_norms_per_cmd = [[], [], []]
for info in tqdm(data_infos):
    plan_mask = info["gt_ego_fut_masks"]
    if plan_mask.sum() != 6:
        continue
    cum = info["gt_ego_fut_trajs"].cumsum(axis=-2)
    cmd = int(info["gt_ego_fut_cmd"].astype(np.int32).argmax(axis=-1))
    v0_norm = float(np.linalg.norm(np.asarray(info["ego_status"])[6:8]))
    navi_trajs[cmd].append(cum)
    v0_norms_per_cmd[cmd].append(v0_norm)

cmd_names = {0: "Right", 1: "Left", 2: "Straight"}
clusters = []
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for cmd_idx, (trajs, v0s) in enumerate(zip(navi_trajs, v0_norms_per_cmd)):
    if len(trajs) < K:
        raise RuntimeError(
            f"cmd {cmd_idx} has only {len(trajs)} trajectories; need K={K}"
        )
    cum_arr = np.stack(trajs, axis=0)              # (n, 6, 2) meters
    v0_arr = np.asarray(v0s)                       # (n,) m/s
    scale = np.maximum(v0_arr, EPS_V)              # (n,) m/s
    shape = cum_arr / scale[:, None, None]         # (n, 6, 2) seconds

    flat = shape.reshape(-1, 12)
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(flat)
    centers = km.cluster_centers_.reshape(-1, 6, 2)  # (K, 6, 2) seconds
    labels = km.labels_

    median_v0 = np.zeros(K, dtype=np.float32)
    for k in range(K):
        median_v0[k] = float(np.median(v0_arr[labels == k])) if (labels == k).any() else 0.0
    print(f"cmd {cmd_idx} ({cmd_names[cmd_idx]}): {flat.shape[0]} trajs -> {K} clusters")
    for k in range(K):
        endpt = centers[k, -1]
        n_k = int((labels == k).sum())
        print(f"  k={k}: shape_endpoint=({endpt[0]:+.3f}, {endpt[1]:+.3f}) sec"
              f"  med_v0={median_v0[k]:5.2f}m/s  n={n_k}")

    clusters.append(centers)

    ax = axes[cmd_idx]
    for k in range(K):
        x = np.concatenate([[0], centers[k, :, 0]])
        y = np.concatenate([[0], centers[k, :, 1]])
        ax.plot(x, y, marker="o", markersize=3,
                label=f"k={k}  v̄₀={median_v0[k]:.1f} m/s")
    ax.set_title(f"cmd={cmd_names[cmd_idx]}  (n={flat.shape[0]})")
    ax.set_xlabel("x_lidar / v_0  (right, sec)")
    if cmd_idx == 0:
        ax.set_ylabel("y_lidar / v_0  (forward, sec)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")

clusters = np.stack(clusters, axis=0).astype(np.float32)  # (3, 6, 6, 2)

os.makedirs(os.path.dirname(OUT_ANCHOR), exist_ok=True)
os.makedirs(os.path.dirname(OUT_PLOT), exist_ok=True)
np.save(OUT_ANCHOR, clusters)
fig.suptitle(f"Velocity-normalized plan anchors (K={K} per cmd, ε_v={EPS_V} m/s; units = seconds)")
fig.tight_layout()
fig.savefig(OUT_PLOT, dpi=130)
print("saved", OUT_ANCHOR, "shape", clusters.shape)
print("saved", OUT_PLOT)
