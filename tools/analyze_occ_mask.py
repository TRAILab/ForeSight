#!/usr/bin/env python3
"""
Compute statistics for NuScenes objects that don't meet the num_lidar_pts > 0
mask criteria (i.e. num_lidar_pts <= 0) used in NuScenes3DDataset.get_ann_info().

Usage:
    python tools/analyze_num_lidar_pts_mask.py [ann_file]
    python tools/analyze_num_lidar_pts_mask.py data/infos/nuscenes_infos_train.pkl
    python tools/analyze_num_lidar_pts_mask.py data/infos/nuscenes_infos_val.pkl
"""

import argparse
from collections import defaultdict

import mmcv
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Stats for objects with num_lidar_pts <= 0 (filtered out by dataset mask)"
    )
    parser.add_argument(
        "ann_file",
        nargs="?",
        default="data/infos/nuscenes_infos_train.pkl",
        help="Path to NuScenes info pkl (default: data/infos/nuscenes_infos_train.pkl)",
    )
    parser.add_argument(
        "--load-interval",
        type=int,
        default=1,
        help="Same as dataset load_interval (default: 1)",
    )
    parser.add_argument(
        "--by-class",
        action="store_true",
        help="Print per-class breakdown of filtered objects",
    )
    args = parser.parse_args()

    print(f"Loading {args.ann_file} ...")
    data = mmcv.load(args.ann_file, file_format="pkl")
    infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
    infos = infos[:: args.load_interval]
    print(f"Loaded {len(infos)} samples (load_interval={args.load_interval})\n")

    total_objs = 0
    filtered_out = 0  # num_lidar_pts <= 0
    samples_with_any_filtered = 0
    samples_with_all_filtered = 0
    # num_lidar_pts value distribution for filtered objects (0, -1, etc.)
    filtered_pts_dist = defaultdict(int)
    # per-class: (total, filtered_out)
    class_total = defaultdict(int)
    class_filtered = defaultdict(int)

    for info in infos:
        num_pts = np.asarray(info["num_lidar_pts"])
        gt_names = info["gt_names"]
        n = len(num_pts)
        if n == 0:
            continue

        total_objs += n
        mask_kept = num_pts > 0
        mask_filtered = ~mask_kept
        n_filtered = mask_filtered.sum()
        filtered_out += n_filtered

        if n_filtered > 0:
            samples_with_any_filtered += 1
        if n_filtered == n:
            samples_with_all_filtered += 1

        for v in num_pts[mask_filtered]:
            filtered_pts_dist[int(v)] += 1

        for i in range(n):
            name = gt_names[i] if isinstance(gt_names[i], str) else gt_names[i].item()
            class_total[name] += 1
            if num_pts[i] <= 0:
                class_filtered[name] += 1

    # Summary
    print("=" * 60)
    print("Summary (mask: num_lidar_pts > 0)")
    print("=" * 60)
    print(f"  Total objects:              {total_objs}")
    print(f"  Filtered (num_lidar_pts <= 0): {filtered_out}")
    pct = 100.0 * filtered_out / total_objs
    print(f"  Filtered %:                  {pct:.2f}%")

    if args.by_class and class_total:
        print("\n" + "=" * 60)
        print("Per-class (filtered = num_lidar_pts <= 0)")
        print("=" * 60)
        for name in sorted(class_total.keys()):
            tot = class_total[name]
            flt = class_filtered[name]
            pct = 100.0 * flt / tot if tot else 0
            print(f"  {name:25s}  total: {tot:6d}  filtered: {flt:6d}  ({pct:5.2f}%)")

    # -------------------------------------------------------------------------
    # Track-level stats: per track, how many annotations are not present at all
    # in the infos (instance not labelled in that sample) after the first
    # sample where the track appears.
    # -------------------------------------------------------------------------
    if "instance_inds" in infos[0]:
        # scene_token -> sorted list of sample_idx belonging to that scene
        scene_to_sample_indices = defaultdict(list)
        # (scene_token, instance_ind) -> list of sample_idx where instance is annotated
        tracks = defaultdict(lambda: defaultdict(list))

        for sample_idx, info in enumerate(infos):
            scene_token = info["scene_token"]
            scene_to_sample_indices[scene_token].append(sample_idx)
            instance_inds = info.get("instance_inds", [])
            if len(instance_inds) == 0:
                continue
            instance_inds = np.asarray(instance_inds)
            for i in range(len(instance_inds)):
                instance_ind = int(instance_inds[i])
                tracks[scene_token][instance_ind].append(sample_idx)

        for scene_token in scene_to_sample_indices:
            scene_to_sample_indices[scene_token].sort()

        # For each track: span = [first_sample, last_sample]. Missing = number of
        # samples in that span (in this scene) where the instance is not in the infos.
        total_tracks = 0
        total_missing_not_labelled = 0
        total_missing_after_track_end = 0
        tracks_with_any_missing = 0
        missing_per_track_dist = defaultdict(int)
        after_end_per_track_dist = defaultdict(int)

        for scene_token, instances in tracks.items():
            scene_samples = scene_to_sample_indices[scene_token]
            for instance_ind, appearances in instances.items():
                if len(appearances) == 0:
                    continue
                appearances = sorted(appearances)
                first_sample = appearances[0]
                last_sample = appearances[-1]
                # Samples in this scene that fall in the track span
                span_samples = [s for s in scene_samples if first_sample <= s <= last_sample]
                expected_frames = len(span_samples)
                present_frames = len(appearances)
                missing = expected_frames - present_frames

                # Samples in this scene after the last labelled sample for this track
                after_end = sum(1 for s in scene_samples if s > last_sample)
                total_missing_after_track_end += after_end
                after_end_per_track_dist[after_end] += 1

                total_tracks += 1
                total_missing_not_labelled += missing
                if missing > 0:
                    tracks_with_any_missing += 1
                missing_per_track_dist[missing] += 1

        print("\n" + "=" * 60)
        print("Track stats (missing = not present/labelled in infos at all)")
        print("=" * 60)
        print(f"  Total tracks:                          {total_tracks}")
        print(f"  Total annotations missing mid track span (not in infos): {total_missing_not_labelled}")
        print(f"  Total annotations missing after track end (not in infos): {total_missing_after_track_end}")
        print("\n  Distribution of # missing per track mid span (not in infos):")
        for k in sorted(missing_per_track_dist.keys()):
            n_tracks = missing_per_track_dist[k]
            print(f"    {k:3d} missing: {n_tracks:6d} tracks")
        print("\n  Distribution of # missing after track end per track:")
        for k in sorted(after_end_per_track_dist.keys()):
            n_tracks = after_end_per_track_dist[k]
            print(f"    {k:3d} after end: {n_tracks:6d} tracks")
    else:
        print("\n  (Skipping track stats: no 'instance_inds' in infos)")

    print()


if __name__ == "__main__":
    main()
