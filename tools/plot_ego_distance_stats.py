#!/usr/bin/env python3
"""Plot ego-distance histograms and statistics per object class.

Loads the combined trainval split from:
  data/infos/nuscenes_infos_train.pkl
  data/infos/nuscenes_infos_val.pkl

For every annotated box, ego_distance = sqrt(x^2 + y^2) where (x, y) are the
first two columns of gt_boxes (already in the ego-vehicle frame).

Produces two sets of analysis:

1. Raw annotation classes (all classes present in the pkl files)
   Outputs: per_class_histograms.png, combined_kde.png, stats_table.png,
            stats_summary.txt

2. NuScenes detection classes (10 canonical classes used for eval)
   Merges all pedestrian sub-types into "pedestrian"; drops animal,
   movable_object.*, static_object.*, vehicle.emergency.*.
   Outputs: det_per_class_histograms.png, det_combined_kde.png,
            det_stats_table.png, det_stats_summary.txt
"""

import os
import pickle
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_PKL = "data/infos/nuscenes_infos_train.pkl"
VAL_PKL   = "data/infos/nuscenes_infos_val.pkl"
OUTPUT_DIR = "vis/ego_distance_stats"

# ---------------------------------------------------------------------------
# NuScenes detection class mapping
#   Keys  : raw annotation names to remap / keep
#   Values: target detection class name  (None = drop)
# ---------------------------------------------------------------------------
DETECTION_CLASS_MAP = {
    # --- vehicles ---
    "car":                                  "car",
    "truck":                                "truck",
    "bus":                                  "bus",
    "trailer":                              "trailer",
    "construction_vehicle":                 "construction_vehicle",
    "motorcycle":                           "motorcycle",
    "bicycle":                              "bicycle",
    # --- pedestrians (all sub-types merged) ---
    "pedestrian":                           "pedestrian",
    "human.pedestrian.personal_mobility":   "pedestrian",
    "human.pedestrian.stroller":            "pedestrian",
    "human.pedestrian.wheelchair":          "pedestrian",
    # --- static / infrastructure ---
    "traffic_cone":                         "traffic_cone",
    "barrier":                              "barrier",
    # --- dropped ---
    "animal":                               None,
    "movable_object.debris":                None,
    "movable_object.pushable_pullable":     None,
    "static_object.bicycle_rack":           None,
    "vehicle.emergency.ambulance":          None,
    "vehicle.emergency.police":             None,
}

# Fixed colour per detection class for consistent cross-plot appearance
DET_CLASS_COLORS = {
    "car":                  "#377eb8",
    "truck":                "#e41a1c",
    "bus":                  "#4daf4a",
    "trailer":              "#984ea3",
    "construction_vehicle": "#ff7f00",
    "motorcycle":           "#a65628",
    "bicycle":              "#f781bf",
    "pedestrian":           "#66c2a5",
    "traffic_cone":         "#fc8d62",
    "barrier":              "#8da0cb",
}

# ---------------------------------------------------------------------------
# Color palette (one per class, qualitative)
# ---------------------------------------------------------------------------
PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
    "#a6d854",
]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def collect_distances(infos, class_map=None):
    """Return dict: class_name -> np.ndarray of ego distances.

    Args:
        class_map: optional dict mapping raw annotation names to target names.
                   Entries with value None are dropped. If None, raw names are
                   used as-is.
    """
    distances = defaultdict(list)
    for info in infos:
        boxes = np.asarray(info["gt_boxes"])   # (N, 7)  x y z w l h yaw
        names = np.asarray(info["gt_names"])   # (N,)
        if boxes.ndim != 2 or boxes.shape[0] == 0:
            continue
        dist = np.sqrt(boxes[:, 0] ** 2 + boxes[:, 1] ** 2)
        for d, cls in zip(dist, names):
            if class_map is not None:
                target = class_map.get(cls)
                if target is None:
                    continue
                cls = target
            distances[cls].append(float(d))
    return {cls: np.array(vals) for cls, vals in distances.items()}


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------
def compute_stats(arr):
    return {
        "count":  len(arr),
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std":    float(np.std(arr)),
        "p75":    float(np.percentile(arr, 75)),
        "p90":    float(np.percentile(arr, 90)),
        "max":    float(np.max(arr)),
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def kde_curve(arr, x_min=0, x_max=None, n=400):
    if x_max is None:
        x_max = np.percentile(arr, 99)
    xs = np.linspace(x_min, x_max, n)
    kde = gaussian_kde(arr, bw_method="scott")
    ys = kde(xs)
    return xs, ys


# ---------------------------------------------------------------------------
# Figure 1 – per-class histogram grid
# ---------------------------------------------------------------------------
def plot_per_class(distances, out_path, title_tag="", color_map=None):
    classes = sorted(distances.keys())
    n_cls = len(classes)
    ncols = 3 if n_cls > 4 else min(n_cls, 3)
    nrows = (n_cls + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for idx, cls in enumerate(classes):
        ax = axes[idx]
        arr = distances[cls]
        color = (color_map or {}).get(cls, PALETTE[idx % len(PALETTE)])
        clip = np.percentile(arr, 99.5)
        arr_clip = arr[arr <= clip]

        ax.hist(arr_clip, bins=60, color=color, alpha=0.6, density=True,
                edgecolor="none", label="histogram")

        try:
            xs, ys = kde_curve(arr_clip)
            ax.plot(xs, ys, color=color, linewidth=2, label="KDE")
        except Exception:
            pass

        stats = compute_stats(arr)
        ax.axvline(stats["mean"],   color="black",  linestyle="--", linewidth=1.2,
                   label=f"mean={stats['mean']:.1f}m")
        ax.axvline(stats["median"], color="dimgray", linestyle=":",  linewidth=1.2,
                   label=f"median={stats['median']:.1f}m")
        ax.axvline(stats["p75"],    color="saddlebrown", linestyle="-.", linewidth=1.0,
                   label=f"P75={stats['p75']:.1f}m")

        ax.set_title(f"{cls}  (n={stats['count']:,})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Ego distance (m)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(left=0)
        ax.grid(axis="y", alpha=0.3)

    for ax in axes[n_cls:]:
        ax.set_visible(False)

    suptitle = f"Ego-distance distribution per object class\n(NuScenes trainval{title_tag})"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 – combined KDE overlay
# ---------------------------------------------------------------------------
def plot_combined(distances, out_path, title_tag="", color_map=None):
    classes = sorted(distances.keys())
    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, cls in enumerate(classes):
        arr = distances[cls]
        color = (color_map or {}).get(cls, PALETTE[idx % len(PALETTE)])
        clip = np.percentile(arr, 99.5)
        arr_clip = arr[arr <= clip]
        try:
            xs, ys = kde_curve(arr_clip)
            ax.plot(xs, ys, color=color, linewidth=2, label=cls)
        except Exception:
            pass

    ax.set_xlabel("Ego distance (m)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Ego-distance KDE — all classes (NuScenes trainval{title_tag})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 – stats table
# ---------------------------------------------------------------------------
def plot_stats_table(stats_dict, out_path, title_tag=""):
    classes = sorted(stats_dict.keys())
    col_labels = ["Class", "Count", "Mean (m)", "Median (m)", "Std (m)", "P75 (m)", "P90 (m)", "Max (m)"]
    rows = []
    for cls in classes:
        s = stats_dict[cls]
        rows.append([
            cls,
            f"{s['count']:,}",
            f"{s['mean']:.2f}",
            f"{s['median']:.2f}",
            f"{s['std']:.2f}",
            f"{s['p75']:.2f}",
            f"{s['p90']:.2f}",
            f"{s['max']:.2f}",
        ])

    fig, ax = plt.subplots(figsize=(14, 0.5 + 0.55 * len(classes)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(list(range(len(col_labels))))

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#ecf0f1")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#bdc3c7")

    fig.suptitle(f"Ego-distance statistics per class (NuScenes trainval{title_tag})",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------
def write_text_summary(stats_dict, out_path, title_tag=""):
    classes = sorted(stats_dict.keys())
    lines = [
        f"Ego-distance statistics per class (NuScenes trainval{title_tag})",
        "=" * 60,
        "",
    ]
    for cls in classes:
        s = stats_dict[cls]
        lines += [
            f"Class: {cls}",
            f"  Count  : {s['count']:,}",
            f"  Mean   : {s['mean']:.2f} m",
            f"  Median : {s['median']:.2f} m",
            f"  Std    : {s['std']:.2f} m",
            f"  P75    : {s['p75']:.2f} m",
            f"  P90    : {s['p90']:.2f} m",
            f"  Max    : {s['max']:.2f} m",
            "",
        ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading train pkl …")
    train_data = load_pkl(TRAIN_PKL)
    print("Loading val pkl …")
    val_data = load_pkl(VAL_PKL)

    all_infos = train_data["infos"] + val_data["infos"]
    print(f"Total frames: {len(all_infos):,}")

    # -----------------------------------------------------------------------
    # Pass 1 – raw annotation classes
    # -----------------------------------------------------------------------
    print("\n--- Pass 1: raw annotation classes ---")
    distances = collect_distances(all_infos)
    for cls, arr in sorted(distances.items()):
        print(f"  {cls:40s}  {len(arr):>8,} instances")

    stats_dict = {cls: compute_stats(arr) for cls, arr in distances.items()}

    print("\nGenerating raw-class plots …")
    plot_per_class(distances,
                   os.path.join(OUTPUT_DIR, "per_class_histograms.png"))
    plot_combined(distances,
                  os.path.join(OUTPUT_DIR, "combined_kde.png"))
    plot_stats_table(stats_dict,
                     os.path.join(OUTPUT_DIR, "stats_table.png"))
    write_text_summary(stats_dict,
                       os.path.join(OUTPUT_DIR, "stats_summary.txt"))

    # -----------------------------------------------------------------------
    # Pass 2 – NuScenes detection classes
    # -----------------------------------------------------------------------
    print("\n--- Pass 2: NuScenes detection classes ---")
    det_distances = collect_distances(all_infos, class_map=DETECTION_CLASS_MAP)
    for cls, arr in sorted(det_distances.items()):
        print(f"  {cls:40s}  {len(arr):>8,} instances")

    det_stats = {cls: compute_stats(arr) for cls, arr in det_distances.items()}

    print("\nGenerating detection-class plots …")
    plot_per_class(det_distances,
                   os.path.join(OUTPUT_DIR, "det_per_class_histograms.png"),
                   title_tag=" — detection classes",
                   color_map=DET_CLASS_COLORS)
    plot_combined(det_distances,
                  os.path.join(OUTPUT_DIR, "det_combined_kde.png"),
                  title_tag=" — detection classes",
                  color_map=DET_CLASS_COLORS)
    plot_stats_table(det_stats,
                     os.path.join(OUTPUT_DIR, "det_stats_table.png"),
                     title_tag=" — detection classes")
    write_text_summary(det_stats,
                       os.path.join(OUTPUT_DIR, "det_stats_summary.txt"),
                       title_tag=" — detection classes")

    print("\nDone.")


if __name__ == "__main__":
    main()
