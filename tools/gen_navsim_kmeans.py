"""Generate real navtrain k-means anchors for SparseDrive's planning + motion
heads. Replaces the random placeholders from
`tools/make_navsim_kmeans_placeholder.py`.

Anchor shapes match the navsim variant config
(projects/configs/sparsedrive_r50_stage2_navsim_planonly.py):
  - kmeans_plan_navsim_<ego_fut_mode>_cmd<num_driving_cmds>.npy
      shape: (num_driving_cmds, ego_fut_mode, ego_fut_ts, 2)
  - kmeans_motion_navsim_<fut_mode>.npy
      shape: (num_classes=10, fut_mode, fut_ts, 2)

Both stored as deltas (per-step displacement); cumsum reconstructs absolute
trajectories.

Usage (after navtrain is downloaded and on the right machine):
  python tools/gen_navsim_kmeans.py \
      --logs-dir /scratch/spapais/data/openscene/navsim_logs/trainval \
      --maps-dir /scratch/spapais/data/nuplan-maps-v1.0 \
      --out-dir /scratch/spapais/ForeSight/data/kmeans \
      --max-scenes 20000   # optional: subsample for faster runs
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", required=True, help="navsim_logs/<split> dir")
    parser.add_argument("--maps-dir", required=True, help="nuplan-maps-v1.0 dir")
    parser.add_argument("--out-dir", required=True, help="dir to save anchor .npy files")
    parser.add_argument("--num-driving-cmds", type=int, default=4)
    parser.add_argument("--ego-fut-mode", type=int, default=6)
    parser.add_argument("--ego-fut-ts", type=int, default=8)
    parser.add_argument("--fut-mode", type=int, default=6)
    parser.add_argument("--fut-ts", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-history-frames", type=int, default=4)
    args = parser.parse_args()

    os.environ.setdefault("NUPLAN_MAPS_ROOT", args.maps_dir)
    sys.path.insert(0, "/workspace/ForeSight")

    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading scenes from {args.logs_dir} ...")
    scene_filter = SceneFilter(
        num_history_frames=args.num_history_frames,
        num_future_frames=max(args.ego_fut_ts, args.fut_ts),
        max_scenes=args.max_scenes,
    )
    loader = SceneLoader(
        sensor_blobs_path=Path(args.logs_dir),  # not loading sensors so dummy is fine
        data_path=Path(args.logs_dir),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    print(f"  found {len(loader)} scenes")

    # Per-cmd ego future deltas: list[(num_driving_cmds, list of (ego_fut_ts, 2))]
    ego_deltas_per_cmd = [[] for _ in range(args.num_driving_cmds)]
    # Per-class motion future deltas
    motion_deltas_per_class = [[] for _ in range(args.num_classes)]

    print("Extracting trajectories...")
    for token in tqdm(loader.tokens):
        scene = loader.get_scene_from_token(token)
        # Ego future
        traj = scene.get_future_trajectory(num_trajectory_frames=args.ego_fut_ts)
        poses = traj.poses[:, :2]  # (T, 2)
        deltas = np.zeros_like(poses)
        deltas[0] = poses[0]
        deltas[1:] = poses[1:] - poses[:-1]
        # Driving command from current frame
        current_idx = scene.scene_metadata.num_history_frames - 1
        cmd_onehot = scene.frames[current_idx].ego_status.driving_command
        cmd_idx = int(np.argmax(cmd_onehot))
        if 0 <= cmd_idx < args.num_driving_cmds:
            ego_deltas_per_cmd[cmd_idx].append(deltas)
        # Agent motion futures: TODO — needs per-track future aggregation across
        # frames, which navsim's Scene.frames already exposes via track_tokens.
        # Left as a follow-up; placeholder anchors stay random for motion.

    # Plan k-means: cluster per cmd
    plan_anchors = np.zeros(
        (args.num_driving_cmds, args.ego_fut_mode, args.ego_fut_ts, 2),
        dtype=np.float32,
    )
    for cmd_idx, deltas_list in enumerate(ego_deltas_per_cmd):
        if not deltas_list:
            print(f"  cmd {cmd_idx}: no scenes — keeping zero anchors")
            continue
        arr = np.stack(deltas_list, axis=0)  # (N, T, 2)
        flat = arr.reshape(arr.shape[0], -1)
        n_clusters = min(args.ego_fut_mode, len(arr))
        km = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10).fit(flat)
        centers = km.cluster_centers_.reshape(n_clusters, args.ego_fut_ts, 2)
        plan_anchors[cmd_idx, :n_clusters] = centers
        print(f"  cmd {cmd_idx}: clustered {len(arr)} trajs into {n_clusters} modes")

    plan_path = out_dir / f"kmeans_plan_navsim_{args.ego_fut_mode}_cmd{args.num_driving_cmds}.npy"
    np.save(plan_path, plan_anchors)
    print(f"  wrote {plan_path}  shape={plan_anchors.shape}")

    # Motion anchors: random for now (per-class clustering needs real per-track
    # future trajectories from scene.frames track_tokens, which is a follow-up).
    motion_anchors = (
        np.random.RandomState(args.seed)
        .randn(args.num_classes, args.fut_mode, args.fut_ts, 2)
        .astype(np.float32)
        * 0.5
    )
    motion_path = out_dir / f"kmeans_motion_navsim_{args.fut_mode}.npy"
    np.save(motion_path, motion_anchors)
    print(f"  wrote {motion_path}  shape={motion_anchors.shape}  (random — TODO: real per-class)")


if __name__ == "__main__":
    main()
