"""K-means on ego-interacting agents.

Filter: agents that come within `dist_thresh` meters of ego at any timestep
within `time_horizon` seconds. K-means on their (x, y, z, w, l, h, sin_yaw,
cos_yaw) features at t=0 (ego frame ≈ lidar frame at t=0).

Output: (K, 11) numpy array in SparseDrive anchor format
[X, Y, Z, log_W, log_L, log_H, SIN_YAW, COS_YAW, VX, VY, VZ].
"""
import argparse
import pickle

import numpy as np
from sklearn.cluster import KMeans


def collect_ego_interacting_features(
    infos_path,
    dist_thresh=20.0,
    time_horizon_steps=6,  # 6 × 0.5s = 3s
    verbose=True,
):
    with open(infos_path, "rb") as f:
        d = pickle.load(f)
    infos = d["infos"] if isinstance(d, dict) else d
    feats = []
    n_total = 0
    n_keep = 0
    for i, s in enumerate(infos):
        boxes = s["gt_boxes"]  # (N, 7) [x, y, z, w, l, h, yaw]
        if boxes is None or len(boxes) == 0:
            continue
        agent_trajs = s["gt_agent_fut_trajs"]  # (N, 12, 2) deltas
        agent_masks = s["gt_agent_fut_masks"]  # (N, 12)
        ego_trajs = s["gt_ego_fut_trajs"]  # (T_ego, 2) deltas; T_ego usually 6
        T = min(time_horizon_steps, agent_trajs.shape[1], ego_trajs.shape[0])

        # ego absolute position per t in [0..T]: ego_pos[0] = origin, ego_pos[t] = sum_{k<t} delta_k
        ego_cum = np.concatenate([np.zeros((1, 2)), np.cumsum(ego_trajs[:T], axis=0)], axis=0)  # (T+1, 2)
        # We compare agents at timesteps t=0..T (positions at t=0 from box, t=1..T from cumsum)
        # Agent absolute position at t: box_xy + cumsum(agent_trajs[:t])
        agent_xy0 = boxes[:, :2]  # (N, 2)
        # agent_pos_full[t]: t=0 is agent_xy0; t=1..T add cumsum of first t deltas
        agent_cum = np.concatenate(
            [np.zeros((agent_trajs.shape[0], 1, 2)), np.cumsum(agent_trajs[:, :T], axis=1)],
            axis=1,
        )  # (N, T+1, 2)
        agent_pos = agent_xy0[:, None, :] + agent_cum  # (N, T+1, 2)

        # Distance ego ↔ agent at each timestep
        diff = agent_pos - ego_cum[None, :, :]  # (N, T+1, 2)
        d = np.linalg.norm(diff, axis=-1)  # (N, T+1)

        # Mask validity: t=0 always valid for agents present in sample; t>=1 by agent_masks
        # agent_masks has shape (N, 12); we use first T entries for t=1..T validity
        valid_t = np.concatenate(
            [np.ones((agent_trajs.shape[0], 1), dtype=bool), agent_masks[:, :T].astype(bool)],
            axis=1,
        )  # (N, T+1)
        d_masked = np.where(valid_t, d, np.inf)
        within_thresh_any_t = (d_masked < dist_thresh).any(axis=1)  # (N,)

        n_total += boxes.shape[0]
        n_keep += int(within_thresh_any_t.sum())

        if not within_thresh_any_t.any():
            continue
        kept = boxes[within_thresh_any_t]  # (k, 7)
        # Build 8-dim feature [x, y, z, w, l, h, sin_yaw, cos_yaw]
        x, y, z, w, l, h, yaw = kept[:, 0], kept[:, 1], kept[:, 2], kept[:, 3], kept[:, 4], kept[:, 5], kept[:, 6]
        feat = np.stack([x, y, z, w, l, h, np.sin(yaw), np.cos(yaw)], axis=-1)  # (k, 8)
        feats.append(feat)

        if verbose and (i + 1) % 5000 == 0:
            print(f"  processed {i + 1}/{len(infos)}: keep {n_keep} of {n_total} agents so far")

    feats = np.concatenate(feats, axis=0) if feats else np.zeros((0, 8))
    if verbose:
        print(f"Total: kept {n_keep} of {n_total} agents ({n_keep / max(n_total, 1) * 100:.1f}%)")
    return feats


def kmeans_to_anchor11(features, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(features)
    centers = km.cluster_centers_  # (k, 8): [x, y, z, w, l, h, sin_yaw, cos_yaw]
    # Convert to 11-dim anchor: [X, Y, Z, log_W, log_L, log_H, SIN_YAW, COS_YAW, VX=0, VY=0, VZ=0]
    eps = 1e-4
    log_w = np.log(np.clip(centers[:, 3], eps, None))
    log_l = np.log(np.clip(centers[:, 4], eps, None))
    log_h = np.log(np.clip(centers[:, 5], eps, None))
    anchors = np.stack(
        [
            centers[:, 0], centers[:, 1], centers[:, 2],
            log_w, log_l, log_h,
            centers[:, 6], centers[:, 7],
            np.zeros(k), np.zeros(k), np.zeros(k),
        ],
        axis=-1,
    ).astype(np.float64)
    return anchors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--infos", default="data/infos/nuscenes_infos_train.pkl")
    p.add_argument("--out", default="data/kmeans/kmeans_ego_interact_K50_d20m_T3s.npy")
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--dist_thresh", type=float, default=20.0)
    p.add_argument("--time_horizon_steps", type=int, default=6)
    args = p.parse_args()

    print(f"Loading {args.infos} ...")
    feats = collect_ego_interacting_features(
        args.infos,
        dist_thresh=args.dist_thresh,
        time_horizon_steps=args.time_horizon_steps,
    )
    print(f"Feature pool: {feats.shape}")
    if feats.shape[0] < args.K:
        raise RuntimeError(f"Pool too small for K={args.K}: only {feats.shape[0]} agents")
    print(f"K-means K={args.K} ...")
    anchors = kmeans_to_anchor11(feats, args.K)
    print(f"Anchor shape: {anchors.shape}")
    print("Sample centers (x, y, z):")
    print(anchors[:5, :3])
    np.save(args.out, anchors)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
