"""Histograms of ||gt_endpoint|| and ||v_0|| across the train split.

Produces two histograms used to pick the floor ε for the two anchor
normalization variants in reports/2026_04_30_anchor_capacity.md:
  - endpoint-norm: ε on ||gt_ego_fut_trajs.cumsum[-1]||
  - velocity-norm: ε_v on ||ego_status[6:8]||

Saves to reports/ as PNGs and prints summary stats including elbow
candidates so we can pick ε without re-running.
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

FP = "data/infos/nuscenes_infos_train.pkl"
OUT_DIR = "reports"
OUT_PATH = os.path.join(OUT_DIR, "anchor_norm_histograms.png")

with open(FP, "rb") as f:
    data = pickle.load(f)
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))

end_norms = []
v0_norms = []
cmds = []
for info in tqdm(data_infos):
    plan_mask = info["gt_ego_fut_masks"]
    if plan_mask.sum() != 6:
        continue
    cum = info["gt_ego_fut_trajs"].cumsum(axis=-2)
    end_norm = float(np.linalg.norm(cum[-1]))
    v0_norm = float(np.linalg.norm(np.asarray(info["ego_status"])[6:8]))
    cmd = int(info["gt_ego_fut_cmd"].astype(np.int32).argmax(axis=-1))
    end_norms.append(end_norm)
    v0_norms.append(v0_norm)
    cmds.append(cmd)

end_norms = np.array(end_norms)
v0_norms = np.array(v0_norms)
cmds = np.array(cmds)
N = end_norms.shape[0]
print(f"\nLoaded {N} valid trajectories (mask sum == 6)")


def quantile_table(name, vals):
    qs = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    print(f"\n{name} stats: min={vals.min():.3f}  max={vals.max():.3f}  "
          f"mean={vals.mean():.3f}  median={np.median(vals):.3f}")
    print(f"{name} quantiles:")
    for q in qs:
        print(f"  q{q*100:>5.1f}% = {np.quantile(vals, q):.3f}")
    # Bucket counts at common epsilon candidates
    print(f"{name} fraction below thresholds:")
    for t in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        frac = float((vals < t).mean())
        print(f"  < {t:>4.2f}: {frac*100:6.3f}%  ({int((vals < t).sum())} / {N})")


quantile_table("||gt_endpoint||", end_norms)
quantile_table("||v_0||", v0_norms)

cmd_names = {0: "Right", 1: "Left", 2: "Straight"}
print("\nPer-cmd counts:")
for c in [0, 1, 2]:
    n = int((cmds == c).sum())
    print(f"  cmd {c} ({cmd_names[c]}): {n}  ({100*n/N:.2f}%)")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

ax = axes[0, 0]
ax.hist(end_norms, bins=120, range=(0, np.quantile(end_norms, 0.999)),
        color="#3a7ca5", edgecolor="white")
ax.set_title(r"$\|\!|\mathrm{gt\_endpoint}\|\!|$ — full")
ax.set_xlabel("meters")
ax.set_ylabel("count")
ax.axvline(1.0, color="red", linestyle="--", label="ε = 1.0 m default")
ax.legend()

ax = axes[0, 1]
ax.hist(end_norms, bins=200, range=(0, 5.0), color="#3a7ca5", edgecolor="white")
ax.set_title(r"$\|\!|\mathrm{gt\_endpoint}\|\!|$ — zoom 0–5 m")
ax.set_xlabel("meters")
ax.set_ylabel("count")
ax.axvline(1.0, color="red", linestyle="--", label="ε = 1.0 m default")
ax.legend()

ax = axes[1, 0]
ax.hist(v0_norms, bins=120, range=(0, np.quantile(v0_norms, 0.999)),
        color="#9c4a3a", edgecolor="white")
ax.set_title(r"$\|\!|v_0\|\!|$ — full")
ax.set_xlabel("m/s")
ax.set_ylabel("count")
ax.axvline(1.0, color="red", linestyle="--", label="ε_v = 1.0 m/s default")
ax.legend()

ax = axes[1, 1]
ax.hist(v0_norms, bins=200, range=(0, 5.0), color="#9c4a3a", edgecolor="white")
ax.set_title(r"$\|\!|v_0\|\!|$ — zoom 0–5 m/s")
ax.set_xlabel("m/s")
ax.set_ylabel("count")
ax.axvline(1.0, color="red", linestyle="--", label="ε_v = 1.0 m/s default")
ax.legend()

fig.suptitle(f"Anchor-normalization factor distributions  (train, N={N})")
fig.tight_layout()
os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(OUT_PATH, dpi=130)
print(f"\nsaved {OUT_PATH}")

# Extra deep zoom: 0-1 m and 0-1 m/s, with extra-fine bins to inspect the
# shape of the spike at zero vs. the rest of the distribution.
fig_zoom, axes_zoom = plt.subplots(1, 2, figsize=(13, 5))
ax = axes_zoom[0]
ax.hist(end_norms[end_norms <= 1.0], bins=100, range=(0, 1.0),
        color="#3a7ca5", edgecolor="white")
ax.set_title(r"$\|\!|\mathrm{gt\_endpoint}\|\!|$ — 0–1 m (deep zoom)")
ax.set_xlabel("meters")
ax.set_ylabel("count")
ax.axvline(1.0, color="red", linestyle="--", label="ε = 1.0 m default")
ax.legend()

ax = axes_zoom[1]
ax.hist(v0_norms[v0_norms <= 1.0], bins=100, range=(0, 1.0),
        color="#9c4a3a", edgecolor="white")
ax.set_title(r"$\|\!|v_0\|\!|$ — 0–1 m/s (deep zoom)")
ax.set_xlabel("m/s")
ax.set_ylabel("count")
ax.axvline(1.0, color="red", linestyle="--", label="ε_v = 1.0 m/s default")
ax.legend()

# Print per-bin counts so we can read the elbow without staring at the plot
print("\n||gt_endpoint|| counts in 0.05-m bins, 0–1 m:")
for lo in np.arange(0.0, 1.0, 0.05):
    hi = lo + 0.05
    n = int(((end_norms >= lo) & (end_norms < hi)).sum())
    print(f"  [{lo:.2f}, {hi:.2f}): {n}")

print("\n||v_0|| counts in 0.05-m/s bins, 0–1 m/s:")
for lo in np.arange(0.0, 1.0, 0.05):
    hi = lo + 0.05
    n = int(((v0_norms >= lo) & (v0_norms < hi)).sum())
    print(f"  [{lo:.2f}, {hi:.2f}): {n}")

fig_zoom.suptitle(f"Anchor-normalization factor — deep zoom 0–1  (train, N={N})")
fig_zoom.tight_layout()
out_zoom = os.path.join(OUT_DIR, "anchor_norm_histograms_deep_zoom.png")
fig_zoom.savefig(out_zoom, dpi=130)
print(f"saved {out_zoom}")

# Ultra-deep zoom: 0–0.1 m and 0–0.1 m/s, 0.005-bin width.
# Resolves the structure of the "stopped" delta peak.
fig_uz, axes_uz = plt.subplots(1, 2, figsize=(13, 5))
ax = axes_uz[0]
ax.hist(end_norms[end_norms <= 0.1], bins=100, range=(0, 0.1),
        color="#3a7ca5", edgecolor="white")
ax.set_title(r"$\|\!|\mathrm{gt\_endpoint}\|\!|$ — 0–0.1 m (ultra zoom)")
ax.set_xlabel("meters")
ax.set_ylabel("count")

ax = axes_uz[1]
ax.hist(v0_norms[v0_norms <= 0.1], bins=100, range=(0, 0.1),
        color="#9c4a3a", edgecolor="white")
ax.set_title(r"$\|\!|v_0\|\!|$ — 0–0.1 m/s (ultra zoom)")
ax.set_xlabel("m/s")
ax.set_ylabel("count")

print("\n||gt_endpoint|| counts in 0.005-m bins, 0–0.1 m:")
for lo in np.arange(0.0, 0.1, 0.005):
    hi = lo + 0.005
    n = int(((end_norms >= lo) & (end_norms < hi)).sum())
    print(f"  [{lo:.3f}, {hi:.3f}): {n}")

print("\n||v_0|| counts in 0.005-m/s bins, 0–0.1 m/s:")
for lo in np.arange(0.0, 0.1, 0.005):
    hi = lo + 0.005
    n = int(((v0_norms >= lo) & (v0_norms < hi)).sum())
    print(f"  [{lo:.3f}, {hi:.3f}): {n}")

print(f"\nExact zero counts:  ||end||=0: {int((end_norms == 0).sum())}  "
      f"||v_0||=0: {int((v0_norms == 0).sum())}")

fig_uz.suptitle(f"Anchor-normalization factor — ultra zoom 0–0.1  (train, N={N})")
fig_uz.tight_layout()
out_uz = os.path.join(OUT_DIR, "anchor_norm_histograms_ultra_zoom.png")
fig_uz.savefig(out_uz, dpi=130)
print(f"saved {out_uz}")

# Per-cmd zoomed histograms (sanity)
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
for idx, c in enumerate([2, 0, 1]):  # Straight, Right, Left
    sel = cmds == c
    ax = axes2[0, idx]
    ax.hist(end_norms[sel], bins=120, range=(0, 60), color="#3a7ca5",
            edgecolor="white")
    ax.set_title(f"||end||  cmd={cmd_names[c]} (n={int(sel.sum())})")
    ax.set_xlabel("meters")
    ax.axvline(1.0, color="red", linestyle="--")

    ax = axes2[1, idx]
    ax.hist(v0_norms[sel], bins=120, range=(0, 25), color="#9c4a3a",
            edgecolor="white")
    ax.set_title(f"||v_0||  cmd={cmd_names[c]} (n={int(sel.sum())})")
    ax.set_xlabel("m/s")
    ax.axvline(1.0, color="red", linestyle="--")

fig2.suptitle("Per-cmd normalization factor distributions")
fig2.tight_layout()
out2 = os.path.join(OUT_DIR, "anchor_norm_histograms_per_cmd.png")
fig2.savefig(out2, dpi=130)
print(f"saved {out2}")
