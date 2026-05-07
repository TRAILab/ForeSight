#!/usr/bin/env python3
"""Constant-velocity (CV) trajectory error analysis against GT future trajectory labels.

For every annotated agent box in the dataset the script:
  1. Predicts future positions using Constant Velocity (CV):
       (dx_k, dy_k) = (vx * t_k,  vy * t_k)   where t_k = (k+1) * dt
  2. Accumulates the GT future delta trajectory (gt_agent_fut_trajs) to get
     absolute displacement from the current position at each horizon.
  3. Records the L2 (Euclidean) error at each future timestep where
     gt_agent_fut_masks == 1.

The future trajectory deltas are stored in the current sample's lidar frame
and the velocity (gt_velocity) is also in the lidar frame, so no coordinate
transform is needed before comparison.

Outputs
-------
  • Per-timestep statistics table  (N, mean, median, std, RMSE, P25/P75/P90/P95)
  • ADE / FDE summary
  • Per-class ADE / FDE table
  • Histogram grid        — one panel per future horizon
  • Box-plot              — error distribution across horizons
  • ADE curve             — mean / median / P90 vs horizon with IQR band
  • CDF curves            — per horizon
  • Per-class ADE bar chart

Usage
-----
    # Observed annotations, validation split
    python tools/eval_cv_trajectory_error.py \\
        --pkl data/infos/nuscenes_infos_val.pkl \\
        --output-dir vis/cv_traj_error

    # Include interpolated / extrapolated synthetic boxes (_occ pkl)
    python tools/eval_cv_trajectory_error.py \\
        --pkl data/infos/nuscenes_infos_val_occ.pkl \\
        --include-synthetic \\
        --output-dir vis/cv_traj_error_occ

    # Vehicles only, skip boxes with 0 sensor points
    python tools/eval_cv_trajectory_error.py \\
        --pkl data/infos/nuscenes_infos_val.pkl \\
        --classes car truck bus trailer \\
        --valid-only \\
        --output-dir vis/cv_traj_error_vehicles

    # Train split with per-split loop (omit --pkl to process train + val)
    python tools/eval_cv_trajectory_error.py \\
        --data-dir data/infos \\
        --output-dir vis/cv_traj_error
"""

import argparse
import os
import pickle
import numpy as np
from collections import defaultdict


# ===========================================================================
# Acceleration pre-computation
# ===========================================================================

def _compute_instance_accelerations(infos):
    """Compute per-box acceleration magnitudes via central differences in global frame.

    Velocities are first rotated into the global frame (so ego-motion does not
    corrupt the difference) then differenced between consecutive appearances of
    the same instance within a scene.

    Scheme
    ------
    Interior frames  : central difference  a = (v_{k+1} − v_{k−1}) / (t_{k+1} − t_{k−1})
    Boundary frames  : one-sided difference using the single available neighbour
    Single-appearance: acceleration set to 0

    Returns
    -------
    dict mapping (global_info_index, box_index) → acceleration magnitude (m/s²).
    Value is NaN when any required velocity is NaN.
    """
    from pyquaternion import Quaternion

    scene_map = defaultdict(list)
    for gi, info in enumerate(infos):
        scene_map[info['scene_token']].append(gi)
    for sc in scene_map:
        scene_map[sc].sort(key=lambda i: infos[i]['timestamp'])

    accel_map = {}

    for _, gidxs in scene_map.items():
        # Build per-instance track in global-frame velocity coordinates
        # entry: (frame_pos, gi, bi, vx_g, vy_g, t_s)
        inst_tracks = defaultdict(list)
        for fp, gi in enumerate(gidxs):
            info = infos[gi]
            t_s  = info['timestamp'] * 1e-6
            R_l2e = Quaternion(info['lidar2ego_rotation']).rotation_matrix
            R_e2g = Quaternion(info['ego2global_rotation']).rotation_matrix
            R     = R_e2g @ R_l2e          # lidar → global rotation
            for bi, inst_ind in enumerate(info['instance_inds']):
                vx_l = float(info['gt_velocity'][bi, 0])
                vy_l = float(info['gt_velocity'][bi, 1])
                if np.isnan(vx_l) or np.isnan(vy_l):
                    vx_g, vy_g = float('nan'), float('nan')
                else:
                    v_g = R @ np.array([vx_l, vy_l, 0.0])
                    vx_g, vy_g = float(v_g[0]), float(v_g[1])
                inst_tracks[inst_ind].append((fp, gi, bi, vx_g, vy_g, t_s))

        for _, track in inst_tracks.items():
            track.sort(key=lambda x: x[0])
            n = len(track)
            for k, (fp, gi, bi, vx, vy, t) in enumerate(track):
                key = (gi, bi)
                if np.isnan(vx) or np.isnan(vy):
                    accel_map[key] = float('nan')
                    continue
                if n == 1:
                    accel_map[key] = 0.0
                    continue
                if k == 0:
                    _, _, _, vx1, vy1, t1 = track[1]
                    dt = t1 - t
                elif k == n - 1:
                    _, _, _, vx1, vy1, t1 = track[k - 1]
                    vx, vy, t, vx1, vy1, t1 = vx1, vy1, t1, vx, vy, t  # swap so dt = t1-t
                    dt = t1 - t
                else:
                    _, _, _, vx0, vy0, t0 = track[k - 1]
                    _, _, _, vx1, vy1, t1 = track[k + 1]
                    vx, vy, t, vx1, vy1, t1 = vx0, vy0, t0, vx1, vy1, t1
                    dt = t1 - t

                if (np.isnan(vx1) or np.isnan(vy1) or dt <= 0):
                    accel_map[(gi, bi)] = float('nan')
                else:
                    accel_map[(gi, bi)] = float(np.hypot((vx1 - vx) / dt,
                                                          (vy1 - vy) / dt))
    return accel_map


# ===========================================================================
# Error collection
# ===========================================================================

def collect_cv_errors(infos, dt=0.5,
                      include_synthetic=False,
                      valid_only=False,
                      classes=None):
    """Collect per-timestep CV L2 errors and per-class breakdowns in one pass.

    Parameters
    ----------
    infos             : list[dict]  — sample info dicts from a pkl file
    dt                : float       — seconds per future step (0.5 s for nuScenes 2 Hz)
    include_synthetic : bool        — include is_interpolated / is_extrapolated boxes
    valid_only        : bool        — skip valid_flag=False boxes (0 sensor pts)
    classes           : list[str]   — restrict to these class names (None = all)

    Returns
    -------
    errors_per_step : list[list[float]]   — length T; each sublist contains all
                                            L2 errors (m) at that horizon index
    class_errors    : dict[str, list[list[float]]]
                                          — same structure but keyed by class name
    T               : int                 — number of future timesteps
    """
    has_synthetic = 'is_interpolated' in infos[0]

    # Determine T from the first info that has agents
    T = None
    for info in infos:
        fut = info.get('gt_agent_fut_trajs')
        if fut is not None and len(fut):
            T = fut.shape[1]
            break
    if T is None:
        raise RuntimeError(
            'No gt_agent_fut_trajs found in the dataset. '
            'Make sure the pkl was produced by NuScenesSparse4DAdaptor.')

    print('  Pre-computing instance accelerations …')
    accel_map = _compute_instance_accelerations(infos)

    errors_per_step = [[] for _ in range(T)]
    class_errors    = defaultdict(lambda: [[] for _ in range(T)])
    # Each entry: (speed_m_s, accel_m_s2, t_k_s, error_m)
    vel_errors      = []

    n_skipped_nan_vel   = 0
    n_skipped_no_mask   = 0
    n_skipped_synthetic = 0
    n_skipped_invalid   = 0
    n_skipped_class     = 0
    n_total             = 0

    for gi, info in enumerate(infos):
        n = len(info.get('gt_boxes', []))
        if n == 0:
            continue

        boxes     = info['gt_boxes']              # (N, 7) lidar frame
        vels      = info['gt_velocity']           # (N, 2) lidar frame [vx, vy]
        names     = info['gt_names']              # (N,)
        valid     = info['valid_flag']            # (N,) bool
        fut_trajs = info['gt_agent_fut_trajs']    # (N, T, 2) delta displacements, lidar frame
        fut_masks = info['gt_agent_fut_masks']    # (N, T)

        is_interp = (info['is_interpolated']
                     if has_synthetic else np.zeros(n, dtype=bool))
        is_extrap = (info['is_extrapolated']
                     if has_synthetic else np.zeros(n, dtype=bool))

        for bi in range(n):
            n_total += 1

            # ── filters ──────────────────────────────────────────────────────
            if not include_synthetic and (bool(is_interp[bi]) or bool(is_extrap[bi])):
                n_skipped_synthetic += 1
                continue
            if valid_only and not bool(valid[bi]):
                n_skipped_invalid += 1
                continue
            cls = str(names[bi])
            if classes is not None and cls not in classes:
                n_skipped_class += 1
                continue

            vx, vy = float(vels[bi, 0]), float(vels[bi, 1])
            if np.isnan(vx) or np.isnan(vy):
                n_skipped_nan_vel += 1
                continue

            mask = fut_masks[bi]   # (T,)
            if mask.sum() < 0.5:
                n_skipped_no_mask += 1
                continue

            # ── GT: accumulate delta displacements → abs. displacement from t=0 ──
            # fut_trajs[bi][k] = pos_{k+1} - pos_{k}  (lidar frame of this sample)
            # cumsum[k]        = pos_{k+1} - pos_0     = displacement at horizon k+1
            gt_disp = np.cumsum(fut_trajs[bi].astype(np.float64), axis=0)  # (T, 2)

            # ── CV prediction at each horizon ─────────────────────────────────
            speed = float(np.hypot(vx, vy))
            accel = accel_map.get((gi, bi), float('nan'))
            for k in range(T):
                if float(mask[k]) < 0.5:
                    continue
                t_k    = (k + 1) * dt
                cv_dx  = vx * t_k
                cv_dy  = vy * t_k
                err    = float(np.hypot(cv_dx - gt_disp[k, 0],
                                        cv_dy - gt_disp[k, 1]))
                errors_per_step[k].append(err)
                class_errors[cls][k].append(err)
                vel_errors.append((speed, accel, t_k, err))

    print(f'  Boxes processed  : {n_total:,}')
    if n_skipped_synthetic:
        print(f'  Skipped synthetic: {n_skipped_synthetic:,}')
    if n_skipped_invalid:
        print(f'  Skipped invalid  : {n_skipped_invalid:,}')
    if n_skipped_class:
        print(f'  Skipped by class : {n_skipped_class:,}')
    if n_skipped_nan_vel:
        print(f'  Skipped NaN vel  : {n_skipped_nan_vel:,}')
    if n_skipped_no_mask:
        print(f'  Skipped no mask  : {n_skipped_no_mask:,}')
    n_used = sum(len(e) for e in errors_per_step)
    print(f'  Error samples    : {n_used:,}  '
          f'(step 0: {len(errors_per_step[0]):,}, '
          f'step {T-1}: {len(errors_per_step[T-1]):,})')

    return errors_per_step, dict(class_errors), vel_errors, T


# ===========================================================================
# Statistics
# ===========================================================================

def compute_stats(errors_per_step, dt):
    """Return a list of stat dicts, one per future timestep."""
    stats = []
    for k, errs in enumerate(errors_per_step):
        t_k = (k + 1) * dt
        if not errs:
            stats.append(dict(horizon=t_k, n=0))
            continue
        arr = np.array(errs)
        stats.append(dict(
            horizon = t_k,
            n       = len(arr),
            mean    = float(np.mean(arr)),
            median  = float(np.median(arr)),
            std     = float(np.std(arr)),
            rmse    = float(np.sqrt(np.mean(arr ** 2))),
            p25     = float(np.percentile(arr, 25)),
            p75     = float(np.percentile(arr, 75)),
            p90     = float(np.percentile(arr, 90)),
            p95     = float(np.percentile(arr, 95)),
        ))
    return stats


def print_stats(stats):
    header = (f'{"Horizon":>8}  {"N":>8}  {"Mean":>6}  {"Median":>6}  '
              f'{"Std":>6}  {"RMSE":>6}  {"P25":>6}  {"P75":>6}  '
              f'{"P90":>6}  {"P95":>6}')
    sep = '─' * len(header)
    print()
    print('CV Trajectory Error (L2, metres)')
    print(sep)
    print(header)
    print(sep)
    for s in stats:
        if s['n'] == 0:
            print(f'{s["horizon"]:>7.1f}s  {"—":>8}')
            continue
        print(f'{s["horizon"]:>7.1f}s  {s["n"]:>8,}  '
              f'{s["mean"]:>6.3f}  {s["median"]:>6.3f}  '
              f'{s["std"]:>6.3f}  {s["rmse"]:>6.3f}  '
              f'{s["p25"]:>6.3f}  {s["p75"]:>6.3f}  '
              f'{s["p90"]:>6.3f}  {s["p95"]:>6.3f}')
    print(sep)

    # ADE = mean of per-horizon mean L2 errors
    means = [s['mean'] for s in stats if s['n'] > 0]
    if means:
        print(f'  ADE (mean L2 over all horizons)  : {np.mean(means):.4f} m')
    last_valid = next((s for s in reversed(stats) if s['n'] > 0), None)
    if last_valid:
        print(f'  FDE (mean L2 at t={last_valid["horizon"]:.1f}s)        : '
              f'{last_valid["mean"]:.4f} m')
    print()


def print_class_stats(class_errors, dt):
    print('Per-class ADE / FDE (mean L2, metres):')
    print(f'  {"Class":<22}  {"ADE":>7}  {"FDE":>7}  {"N(FDE)":>8}')
    print('  ' + '─' * 52)
    rows = []
    for cls, per_step in sorted(class_errors.items()):
        means_cls = [np.mean(e) for e in per_step if e]
        if not means_cls:
            continue
        ade = float(np.mean(means_cls))
        last_errs = next((e for e in reversed(per_step) if e), None)
        fde   = float(np.mean(last_errs)) if last_errs else float('nan')
        n_fde = len(last_errs) if last_errs else 0
        rows.append((cls, ade, fde, n_fde))
    for cls, ade, fde, n_fde in sorted(rows, key=lambda x: -x[1]):
        print(f'  {cls:<22}  {ade:>7.3f}  {fde:>7.3f}  {n_fde:>8,}')
    print()


# ===========================================================================
# Plotting helpers
# ===========================================================================

_BG  = '#0d1117'
_FG  = '#e0e0e0'
_C1  = '#4fc3f7'   # primary accent (blue)
_C2  = '#ffb74d'   # secondary accent (amber) — mean lines
_C3  = '#ef5350'   # tertiary (red) — median
_C4  = '#ce93d8'   # quaternary (purple) — P90


def _style_ax(ax):
    ax.set_facecolor(_BG)
    for sp in ax.spines.values():
        sp.set_color('#333333')
    ax.tick_params(colors=_FG, labelsize=8)
    ax.grid(True, color='white', alpha=0.07, linewidth=0.5)


def plot_histograms(errors_per_step, stats, output_dir, dt, max_err=20.0):
    """Grid of histograms — one panel per future horizon."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    T     = len(errors_per_step)
    ncols = min(4, T)
    nrows = (T + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             facecolor=_BG)
    axes = np.array(axes).flatten()

    bins = np.linspace(0, max_err, 60)

    for k, (errs, s) in enumerate(zip(errors_per_step, stats)):
        ax = axes[k]
        _style_ax(ax)

        if not errs or s['n'] == 0:
            ax.set_visible(False)
            continue

        arr       = np.array(errs)
        clip      = arr[arr <= max_err]
        n_clipped = len(arr) - len(clip)

        ax.hist(clip, bins=bins, color=_C1, alpha=0.78, edgecolor='none')
        ax.axvline(s['mean'],   color=_C2, linewidth=1.6,
                   linestyle='--', label=f'mean={s["mean"]:.2f}')
        ax.axvline(s['median'], color=_C3, linewidth=1.4,
                   linestyle=':',  label=f'median={s["median"]:.2f}')

        t_k      = (k + 1) * dt
        subtitle = f'N={s["n"]:,}  σ={s["std"]:.2f}  RMSE={s["rmse"]:.2f}  P90={s["p90"]:.2f}'
        if n_clipped:
            subtitle += f'\n(+{n_clipped:,} samples > {max_err:.0f} m not shown)'

        ax.set_title(f't = {t_k:.1f} s\n{subtitle}',
                     color=_FG, fontsize=7.5, pad=4)
        ax.set_xlabel('L2 error (m)', color=_FG, fontsize=8)
        ax.set_ylabel('Count',        color=_FG, fontsize=8)
        ax.legend(fontsize=7, labelcolor=_FG,
                  facecolor='#1e2530', edgecolor='#333333', framealpha=0.7)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2.0))

    for j in range(T, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('CV Trajectory L2 Error Distribution per Future Horizon',
                 color=_FG, fontsize=12, y=1.01)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'cv_error_histograms.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


def plot_boxplots(errors_per_step, stats, output_dir, dt, max_err=20.0):
    """Box-plot of error distribution across horizons, with mean overlay."""
    import matplotlib.pyplot as plt

    T        = len(errors_per_step)
    horizons = [(k + 1) * dt for k in range(T)]
    # Clip extreme outliers for visual clarity; cap at 1.5 * max_err so
    # the whiskers / box still show the bulk of the distribution.
    clip_at  = max_err * 1.5

    pairs = [(h, np.clip(np.array(e), 0, clip_at))
             for h, e in zip(horizons, errors_per_step) if e]
    if not pairs:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(pairs) * 1.3), 5),
                           facecolor=_BG)
    _style_ax(ax)

    labels = [f'{h:.1f}s' for h, _ in pairs]
    bplot  = ax.boxplot(
        [d for _, d in pairs],
        labels=labels,
        patch_artist=True,
        medianprops=dict(color=_C3,  linewidth=2.0),
        whiskerprops=dict(color=_FG, linewidth=0.9),
        capprops=dict(color=_FG,     linewidth=0.9),
        flierprops=dict(marker='.', markerfacecolor=_C1,
                        markersize=2, alpha=0.25, linestyle='none'),
        boxprops=dict(facecolor='#1a3a4a', edgecolor=_C1, linewidth=1.2),
    )

    # Mean overlay
    means_v = [s['mean'] for s in stats if s['n'] > 0]
    ax.plot(range(1, len(means_v) + 1), means_v,
            color=_C2, marker='o', markersize=5,
            linewidth=1.8, linestyle='--', label='Mean', zorder=5)
    # P90 overlay
    p90s = [s['p90'] for s in stats if s['n'] > 0]
    ax.plot(range(1, len(p90s) + 1), p90s,
            color=_C4, marker='^', markersize=4,
            linewidth=1.2, linestyle=':', label='P90', zorder=5)

    ax.set_xlabel('Future horizon', color=_FG, fontsize=10)
    ax.set_ylabel('L2 error (m)',   color=_FG, fontsize=10)
    ax.set_title('CV Error Distribution vs. Future Horizon',
                 color=_FG, fontsize=11)
    ax.legend(fontsize=9, labelcolor=_FG,
              facecolor='#1e2530', edgecolor='#333333', framealpha=0.7)
    ax.set_ylim(0, clip_at * 0.6)
    ax.text(0.99, 0.98,
            f'(y-axis clipped to {clip_at * 0.6:.0f} m for readability)',
            transform=ax.transAxes, ha='right', va='top',
            color=_FG, fontsize=7, alpha=0.55)

    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'cv_error_boxplot.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


def plot_ade_curve(stats, output_dir, dt):
    """Mean / median / P90 vs horizon with IQR and mean±σ bands."""
    import matplotlib.pyplot as plt

    valid = [(s['horizon'], s['mean'], s['std'],
              s['p25'], s['p75'], s['p90'], s['median'])
             for s in stats if s['n'] > 0]
    if not valid:
        return

    horizons = np.array([v[0] for v in valid])
    means    = np.array([v[1] for v in valid])
    stds     = np.array([v[2] for v in valid])
    p25s     = np.array([v[3] for v in valid])
    p75s     = np.array([v[4] for v in valid])
    p90s     = np.array([v[5] for v in valid])
    medians  = np.array([v[6] for v in valid])

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=_BG)
    _style_ax(ax)

    # Shaded bands
    ax.fill_between(horizons, p25s, p75s,
                    color=_C1, alpha=0.18, label='IQR (25–75th pct)')
    ax.fill_between(horizons, np.maximum(0, means - stds), means + stds,
                    color=_C2, alpha=0.16, label='Mean ± σ')

    # Lines
    ax.plot(horizons, means,   color=_C2, linewidth=2.0,
            marker='o', markersize=5, label='Mean')
    ax.plot(horizons, medians, color=_C3, linewidth=1.6,
            linestyle='--', marker='s', markersize=4, label='Median')
    ax.plot(horizons, p90s,    color=_C4, linewidth=1.2,
            linestyle=':', marker='^', markersize=4, label='P90')

    ax.set_xlabel('Future horizon (s)', color=_FG, fontsize=10)
    ax.set_ylabel('L2 error (m)',       color=_FG, fontsize=10)
    ax.set_title('CV Trajectory Error vs. Horizon  (ADE curve)',
                 color=_FG, fontsize=11)
    ax.legend(fontsize=9, labelcolor=_FG,
              facecolor='#1e2530', edgecolor='#333333', framealpha=0.7)
    ax.set_xlim(0, horizons[-1] + dt * 0.6)
    ax.set_ylim(0)
    ax.xaxis.set_major_locator(
        __import__('matplotlib').ticker.MultipleLocator(dt))

    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'cv_error_ade_curve.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


def plot_cdf(errors_per_step, stats, output_dir, dt, x_max=15.0):
    """Cumulative distribution function for each horizon on one axes."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    valid_steps = [(k, errs) for k, errs in enumerate(errors_per_step) if errs]
    if not valid_steps:
        return

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=_BG)
    _style_ax(ax)

    cmap = cm.get_cmap('plasma', len(valid_steps))
    for i, (k, errs) in enumerate(valid_steps):
        arr   = np.sort(np.array(errs))
        cdf   = np.arange(1, len(arr) + 1) / len(arr)
        t_k   = (k + 1) * dt
        color = cmap(i / max(len(valid_steps) - 1, 1))
        ax.plot(arr, cdf, color=color, linewidth=1.5, label=f't={t_k:.1f}s')

    for y_ref, label in [(0.5, 'P50'), (0.9, 'P90'), (0.95, 'P95')]:
        ax.axhline(y_ref, color='white', linewidth=0.7, linestyle=':', alpha=0.45)
        ax.text(0.01, y_ref + 0.01, label, color='white',
                fontsize=7, alpha=0.55, transform=ax.get_yaxis_transform())

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('L2 error (m)',       color=_FG, fontsize=10)
    ax.set_ylabel('Cumulative fraction', color=_FG, fontsize=10)
    ax.set_title('CDF of CV Trajectory Error per Horizon', color=_FG, fontsize=11)
    T = len(valid_steps)
    ax.legend(fontsize=8, labelcolor=_FG,
              facecolor='#1e2530', edgecolor='#333333', framealpha=0.7,
              ncol=max(1, T // 4))

    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'cv_error_cdf.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


def plot_class_ade(class_errors, output_dir, dt):
    """Grouped bar chart — ADE and FDE side-by-side for each class."""
    import matplotlib.pyplot as plt

    rows = []
    for cls, per_step in sorted(class_errors.items()):
        means_cls = [np.mean(e) for e in per_step if e]
        if not means_cls:
            continue
        ade       = float(np.mean(means_cls))
        last_errs = next((e for e in reversed(per_step) if e), None)
        fde       = float(np.mean(last_errs)) if last_errs else float('nan')
        rows.append((cls, ade, fde))
    if not rows:
        return

    rows.sort(key=lambda x: -x[1])
    classes = [r[0] for r in rows]
    ades    = [r[1] for r in rows]
    fdes    = [r[2] for r in rows]

    x      = np.arange(len(classes))
    width  = 0.38
    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 1.3), 5),
                           facecolor=_BG)
    _style_ax(ax)

    bars_ade = ax.bar(x - width / 2, ades, width, color=_C1,
                      alpha=0.85, label='ADE', edgecolor='none')
    bars_fde = ax.bar(x + width / 2, fdes, width, color=_C2,
                      alpha=0.85, label='FDE', edgecolor='none')
    ax.bar_label(bars_ade, fmt='%.2f', color=_FG, fontsize=7.5, padding=3)
    ax.bar_label(bars_fde, fmt='%.2f', color=_FG, fontsize=7.5, padding=3)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=25, ha='right', fontsize=9, color=_FG)
    ax.set_xlabel('Agent class', color=_FG, fontsize=10)
    ax.set_ylabel('Error (m)',   color=_FG, fontsize=10)
    ax.set_title('CV ADE / FDE by Agent Class', color=_FG, fontsize=11)
    ax.legend(fontsize=9, labelcolor=_FG,
              facecolor='#1e2530', edgecolor='#333333', framealpha=0.7)
    ax.set_ylim(0)

    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'cv_error_by_class.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


# ===========================================================================
# Velocity-based ADE curves
# ===========================================================================

def _binned_stats(xs, ys, bin_edges):
    """Return (bin_centres, mean, median, p25, p75, p90, counts) for binned data."""
    centres, means, medians, p25s, p75s, p90s, counts = [], [], [], [], [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (xs >= lo) & (xs < hi)
        vals = ys[mask]
        if len(vals) < 5:          # skip sparse bins
            continue
        centres.append((lo + hi) / 2)
        means.append(float(np.mean(vals)))
        medians.append(float(np.median(vals)))
        p25s.append(float(np.percentile(vals, 25)))
        p75s.append(float(np.percentile(vals, 75)))
        p90s.append(float(np.percentile(vals, 90)))
        counts.append(len(vals))
    return (np.array(centres), np.array(means), np.array(medians),
            np.array(p25s), np.array(p75s), np.array(p90s), np.array(counts))


# Speed buckets used by the two new ADE-by-speed plots.
# Each entry: (lo m/s, hi m/s, label, hex colour)
_SPEED_BUCKETS = [
    (0,   1,   '0–1 m/s',  '#90caf9'),   # light blue  — near-stationary
    (1,   3,   '1–3 m/s',  '#4fc3f7'),   # cyan        — walking
    (3,   6,   '3–6 m/s',  '#69f0ae'),   # green       — cycling / slow vehicle
    (6,  10,   '6–10 m/s', '#ffb74d'),   # amber       — urban vehicle
    (10, 999,  '≥10 m/s',  '#ef5350'),   # red         — fast vehicle
]


def _bucket_series(vel_errors):
    """Return per-speed-bucket, per-horizon mean statistics.

    Returns
    -------
    series : list of (label, color, [(t_k, mean, median, p25, p75, p90, n), ...])
             Sorted by t_k within each bucket; buckets with too few samples skipped.
    horizons : sorted list of unique t_k values present in vel_errors
    """
    arr     = np.array(vel_errors)   # (M, 4): speed, accel, t_k, error
    speeds  = arr[:, 0]
    tks     = arr[:, 2]
    errs    = arr[:, 3]
    horizons = sorted(set(tks.tolist()))

    series = []
    for lo, hi, label, color in _SPEED_BUCKETS:
        mask_b = (speeds >= lo) & (speeds < hi)
        pts = []
        for t_k in horizons:
            vals = errs[mask_b & (tks == t_k)]
            if len(vals) < 5:
                continue
            pts.append((t_k,
                        float(np.mean(vals)),
                        float(np.median(vals)),
                        float(np.percentile(vals, 25)),
                        float(np.percentile(vals, 75)),
                        float(np.percentile(vals, 90)),
                        len(vals)))
        if pts:
            series.append((label, color, pts))
    return series, horizons


def plot_ade_curve_by_speed(vel_errors, stats, output_dir, dt):
    """ADE curve broken out by speed bucket.

    X-axis : future horizon t (s)
    Y-axis : mean L2 error (m)
    Lines  : one per speed bucket + overall mean (dashed white)
    Shading: IQR (P25–P75) per bucket (light fill)
    """
    import matplotlib.pyplot as plt

    if not vel_errors:
        return

    series, horizons = _bucket_series(vel_errors)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=_BG)
    _style_ax(ax)

    for label, color, pts in series:
        ts    = np.array([p[0] for p in pts])
        means = np.array([p[1] for p in pts])
        p25s  = np.array([p[3] for p in pts])
        p75s  = np.array([p[4] for p in pts])
        ax.fill_between(ts, p25s, p75s, color=color, alpha=0.10)
        ax.plot(ts, means, color=color, linewidth=1.8,
                marker='o', markersize=4, label=label)

    # Overall mean from pre-computed stats
    ov_t = np.array([s['horizon'] for s in stats if s['n'] > 0])
    ov_m = np.array([s['mean']    for s in stats if s['n'] > 0])
    ax.plot(ov_t, ov_m, color='white', linewidth=1.6, linestyle='--',
            marker='s', markersize=3.5, alpha=0.70, label='Overall mean')

    ax.set_xlabel('Future horizon t (s)', color=_FG, fontsize=10)
    ax.set_ylabel('Mean L2 error (m)',    color=_FG, fontsize=10)
    ax.set_title('CV Trajectory Error vs. Horizon  (by speed bucket)',
                 color=_FG, fontsize=11)
    ax.set_xlim(0, max(horizons) + dt * 0.5)
    ax.set_ylim(0)
    ax.xaxis.set_major_locator(
        __import__('matplotlib').ticker.MultipleLocator(dt))
    ax.legend(fontsize=9, labelcolor=_FG,
              facecolor='#1e2530', edgecolor='#333333', framealpha=0.7)

    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'cv_error_ade_curve_by_speed.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


def plot_error_vs_t2(vel_errors, stats, output_dir, dt):
    """CV error vs t²  — two panels side by side.

    Left  : overall mean / median / P90 vs t² with a linear fit overlay.
            A straight line here confirms the quadratic-in-t hypothesis.
    Right : per-speed-bucket mean vs t² — shows which regime drives curvature.
    """
    import matplotlib.pyplot as plt

    if not vel_errors:
        return

    series, horizons = _bucket_series(vel_errors)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=_BG)

    # ── Left: overall ────────────────────────────────────────────────────────
    ax = axes[0]
    _style_ax(ax)

    ov_t2  = np.array([s['horizon'] ** 2 for s in stats if s['n'] > 0])
    ov_m   = np.array([s['mean']         for s in stats if s['n'] > 0])
    ov_med = np.array([s['median']       for s in stats if s['n'] > 0])
    ov_p90 = np.array([s['p90']          for s in stats if s['n'] > 0])
    ov_p25 = np.array([s['p25']          for s in stats if s['n'] > 0])
    ov_p75 = np.array([s['p75']          for s in stats if s['n'] > 0])

    ax.fill_between(ov_t2, ov_p25, ov_p75, color=_C1, alpha=0.18,
                    label='IQR (25–75th pct)')
    ax.plot(ov_t2, ov_m,   color=_C2, linewidth=2.0,
            marker='o', markersize=5, label='Mean')
    ax.plot(ov_t2, ov_med, color=_C3, linewidth=1.6, linestyle='--',
            marker='s', markersize=4, label='Median')
    ax.plot(ov_t2, ov_p90, color=_C4, linewidth=1.2, linestyle=':',
            marker='^', markersize=4, label='P90')

    # Linear fit: error = a + b*t²  (through the stats points)
    if len(ov_t2) >= 2:
        coeffs = np.polyfit(ov_t2, ov_m, 1)
        t2_line = np.array([0, ov_t2[-1]])
        ax.plot(t2_line, np.polyval(coeffs, t2_line),
                color='white', linewidth=1.3, linestyle='--', alpha=0.50,
                label=f'Linear fit: {coeffs[0]:.3f}·t² + {coeffs[1]:.3f}')

    # x-tick labels show both t² and t values
    ax.set_xticks(ov_t2)
    ax.set_xticklabels([f'{t2:.2f}\n(t={np.sqrt(t2):.1f}s)' for t2 in ov_t2],
                       fontsize=7, color=_FG)
    ax.set_xlabel('t²  (s²)', color=_FG, fontsize=10)
    ax.set_ylabel('L2 error (m)', color=_FG, fontsize=10)
    ax.set_title('CV Error vs. t²  (overall)\nstraight line ↔ quadratic growth in t',
                 color=_FG, fontsize=10)
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.legend(fontsize=8, labelcolor=_FG,
              facecolor='#1e2530', edgecolor='#333333', framealpha=0.7)

    # ── Right: by speed bucket ────────────────────────────────────────────────
    ax = axes[1]
    _style_ax(ax)

    for label, color, pts in series:
        ts2   = np.array([p[0] ** 2 for p in pts])
        means = np.array([p[1]      for p in pts])
        ax.plot(ts2, means, color=color, linewidth=1.8,
                marker='o', markersize=4, label=label)

    ax.set_xticks(ov_t2)
    ax.set_xticklabels([f'{t2:.2f}\n(t={np.sqrt(t2):.1f}s)' for t2 in ov_t2],
                       fontsize=7, color=_FG)
    ax.set_xlabel('t²  (s²)', color=_FG, fontsize=10)
    ax.set_ylabel('Mean L2 error (m)', color=_FG, fontsize=10)
    ax.set_title('CV Error vs. t²  (by speed bucket)',
                 color=_FG, fontsize=10)
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.legend(fontsize=9, labelcolor=_FG,
              facecolor='#1e2530', edgecolor='#333333', framealpha=0.7)

    fig.suptitle('CV Trajectory Error vs. t²',
                 color=_FG, fontsize=12, y=1.01)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'cv_error_vs_t2.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


def plot_error_vs_speed(vel_errors, output_dir, dt,
                        speed_max=15.0, speed_bin=1.0):
    """CV error vs agent speed, with one line per future horizon.

    X-axis : agent speed |v| (m/s)
    Y-axis : mean L2 error (m)
    Lines  : one per future horizon, coloured from cool → warm as t increases
    Shading: IQR (P25–P75) for the first and last horizon only (to keep it readable)
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if not vel_errors:
        return

    arr = np.array(vel_errors)   # (M, 4): speed, accel, t_k, error
    speeds = arr[:, 0]
    tks    = arr[:, 2]
    errs   = arr[:, 3]

    horizons = sorted(set(tks.tolist()))
    bin_edges = np.arange(0, speed_max + speed_bin, speed_bin)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=_BG)
    _style_ax(ax)

    cmap = cm.get_cmap('plasma', len(horizons))

    for i, t_k in enumerate(horizons):
        mask_h = tks == t_k
        sx, sy = speeds[mask_h], errs[mask_h]
        centres, means, medians, p25s, p75s, p90s, counts = \
            _binned_stats(sx, sy, bin_edges)
        if len(centres) == 0:
            continue
        color = cmap(i / max(len(horizons) - 1, 1))
        ax.plot(centres, means, color=color, linewidth=1.8,
                marker='o', markersize=3.5, label=f't={t_k:.1f}s')
        # IQR shading only for first and last horizon
        if i == 0 or i == len(horizons) - 1:
            ax.fill_between(centres, p25s, p75s, color=color, alpha=0.12)

    ax.set_xlabel('Agent speed |v| (m/s)', color=_FG, fontsize=10)
    ax.set_ylabel('Mean L2 error (m)',      color=_FG, fontsize=10)
    ax.set_title('CV Trajectory Error vs. Agent Speed per Horizon',
                 color=_FG, fontsize=11)
    ax.set_xlim(0, speed_max)
    ax.set_ylim(0)
    ax.legend(fontsize=8, labelcolor=_FG,
              facecolor='#1e2530', edgecolor='#333333', framealpha=0.7,
              ncol=max(1, len(horizons) // 3))

    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'cv_error_vs_speed.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


def plot_error_vs_vt(vel_errors, output_dir, dt,
                     vt_max=30.0, vt_bin=2.0):
    """CV error vs predicted displacement (speed × horizon), all horizons collapsed.

    X-axis : |v| · t  (m) — the CV-predicted displacement magnitude
    Y-axis : L2 error (m)
    All (speed, horizon) combinations collapse onto this single axis.
    If CV error is proportional to the manoeuvre magnitude, all horizons
    should fall on the same curve — deviations reveal non-CV behaviour.

    A least-squares linear fit through the origin is overlaid:
        error ≈ α · (|v| · t)
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if not vel_errors:
        return

    arr = np.array(vel_errors)   # (M, 4): speed, accel, t_k, error
    speeds = arr[:, 0]
    tks    = arr[:, 2]
    errs   = arr[:, 3]
    vt     = speeds * tks        # predicted displacement magnitude

    horizons  = sorted(set(tks.tolist()))
    bin_edges = np.arange(0, vt_max + vt_bin, vt_bin)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=_BG)

    # ── Left panel: all horizons overlaid ────────────────────────────────────
    ax = axes[0]
    _style_ax(ax)
    cmap = cm.get_cmap('plasma', len(horizons))

    for i, t_k in enumerate(horizons):
        mask_h = tks == t_k
        sx, sy = vt[mask_h], errs[mask_h]
        centres, means, medians, p25s, p75s, p90s, counts = \
            _binned_stats(sx, sy, bin_edges)
        if len(centres) == 0:
            continue
        color = cmap(i / max(len(horizons) - 1, 1))
        ax.plot(centres, means, color=color, linewidth=1.6,
                marker='o', markersize=3.5, label=f't={t_k:.1f}s', alpha=0.9)

    ax.set_xlabel('|v| · t  (m)', color=_FG, fontsize=10)
    ax.set_ylabel('Mean L2 error (m)', color=_FG, fontsize=10)
    ax.set_title('CV Error vs. Predicted Displacement\n(per horizon)',
                 color=_FG, fontsize=10)
    ax.set_xlim(0, vt_max)
    ax.set_ylim(0)
    ax.legend(fontsize=8, labelcolor=_FG,
              facecolor='#1e2530', edgecolor='#333333', framealpha=0.7,
              ncol=max(1, len(horizons) // 3))

    # ── Right panel: all data collapsed + linear fit ──────────────────────────
    ax = axes[1]
    _style_ax(ax)

    # Binned stats over all horizons combined
    centres, means, medians, p25s, p75s, p90s, counts = \
        _binned_stats(vt, errs, bin_edges)

    if len(centres) > 0:
        ax.fill_between(centres, p25s, p75s, color=_C1, alpha=0.18,
                        label='IQR (25–75th pct)')
        ax.plot(centres, means,   color=_C2, linewidth=2.0,
                marker='o', markersize=4, label='Mean')
        ax.plot(centres, medians, color=_C3, linewidth=1.6,
                linestyle='--', marker='s', markersize=3.5, label='Median')
        ax.plot(centres, p90s,   color=_C4, linewidth=1.2,
                linestyle=':', marker='^', markersize=3.5, label='P90')

        # Linear fit through origin: error ≈ α · vt
        in_range = vt <= vt_max
        vt_fit, err_fit = vt[in_range], errs[in_range]
        if len(vt_fit) > 10:
            alpha = float(np.dot(vt_fit, err_fit) / np.dot(vt_fit, vt_fit))
            x_line = np.array([0, vt_max])
            ax.plot(x_line, alpha * x_line,
                    color='white', linewidth=1.4, linestyle='--', alpha=0.55,
                    label=f'Linear fit: α={alpha:.3f}  (error ≈ {alpha:.3f}·|v|·t)')

    ax.set_xlabel('|v| · t  (m)', color=_FG, fontsize=10)
    ax.set_ylabel('L2 error (m)', color=_FG, fontsize=10)
    ax.set_title('CV Error vs. Predicted Displacement\n(all horizons collapsed)',
                 color=_FG, fontsize=10)
    ax.set_xlim(0, vt_max)
    ax.set_ylim(0)
    ax.legend(fontsize=8, labelcolor=_FG,
              facecolor='#1e2530', edgecolor='#333333', framealpha=0.7)

    fig.suptitle('CV Trajectory Error vs. Predicted Displacement  (|v| · t)',
                 color=_FG, fontsize=12, y=1.01)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'cv_error_vs_vt.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


def plot_error_vs_vt2(vel_errors, output_dir, dt,
                      vt2_max=45.0, vt2_bin=3.0):
    """CV error vs |v| · t²  — two panels side by side.

    X-axis : |v| · t²  (m · s)
    Y-axis : L2 error (m)

    From the curvature model  error ≈ v² · t² / (2R),  dividing both sides
    by v gives  error/v ≈ v · t² / (2R).  Equivalently, if R is roughly
    constant across agents, error should be linear in  v · t²  with slope
    ≈ v / (2R) — but v itself varies, so the collapsed plot tests whether
    v · t²  is a better single predictor than  v · t  or  t²  alone.

    Left  : per-horizon lines (plasma colormap cool→warm)
    Right : all horizons collapsed with binned mean/median/P90 + linear fit
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if not vel_errors:
        return

    arr    = np.array(vel_errors)   # (M, 4): speed, accel, t_k, error
    speeds = arr[:, 0]
    tks    = arr[:, 2]
    errs   = arr[:, 3]
    vt2    = speeds * tks ** 2      # |v| · t²

    horizons  = sorted(set(tks.tolist()))
    bin_edges = np.arange(0, vt2_max + vt2_bin, vt2_bin)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=_BG)

    # ── Left: per-horizon ────────────────────────────────────────────────────
    ax = axes[0]
    _style_ax(ax)
    cmap = cm.get_cmap('plasma', len(horizons))

    for i, t_k in enumerate(horizons):
        mask_h = tks == t_k
        sx, sy = vt2[mask_h], errs[mask_h]
        centres, means, _, _, _, _, _ = _binned_stats(sx, sy, bin_edges)
        if len(centres) == 0:
            continue
        color = cmap(i / max(len(horizons) - 1, 1))
        ax.plot(centres, means, color=color, linewidth=1.6,
                marker='o', markersize=3.5, label=f't={t_k:.1f}s', alpha=0.9)

    ax.set_xlabel('|v| · t²  (m · s)', color=_FG, fontsize=10)
    ax.set_ylabel('Mean L2 error (m)',  color=_FG, fontsize=10)
    ax.set_title('CV Error vs. |v| · t²\n(per horizon)',
                 color=_FG, fontsize=10)
    ax.set_xlim(0, vt2_max)
    ax.set_ylim(0)
    ax.legend(fontsize=8, labelcolor=_FG,
              facecolor='#1e2530', edgecolor='#333333', framealpha=0.7,
              ncol=max(1, len(horizons) // 3))

    # ── Right: collapsed + linear fit ────────────────────────────────────────
    ax = axes[1]
    _style_ax(ax)

    centres, means, medians, p25s, p75s, p90s, _ = \
        _binned_stats(vt2, errs, bin_edges)

    if len(centres) > 0:
        ax.fill_between(centres, p25s, p75s, color=_C1, alpha=0.18,
                        label='IQR (25–75th pct)')
        ax.plot(centres, means,   color=_C2, linewidth=2.0,
                marker='o', markersize=4, label='Mean')
        ax.plot(centres, medians, color=_C3, linewidth=1.6,
                linestyle='--', marker='s', markersize=3.5, label='Median')
        ax.plot(centres, p90s,    color=_C4, linewidth=1.2,
                linestyle=':', marker='^', markersize=3.5, label='P90')

        # Linear fit through origin: error ≈ α · vt²
        in_range = vt2 <= vt2_max
        x_fit, y_fit = vt2[in_range], errs[in_range]
        if len(x_fit) > 10:
            alpha = float(np.dot(x_fit, y_fit) / np.dot(x_fit, x_fit))
            x_line = np.array([0, vt2_max])
            ax.plot(x_line, alpha * x_line,
                    color='white', linewidth=1.4, linestyle='--', alpha=0.55,
                    label=f'Linear fit: α={alpha:.4f}  (error ≈ {alpha:.4f}·|v|·t²)')

    ax.set_xlabel('|v| · t²  (m · s)', color=_FG, fontsize=10)
    ax.set_ylabel('L2 error (m)',       color=_FG, fontsize=10)
    ax.set_title('CV Error vs. |v| · t²\n(all horizons collapsed)',
                 color=_FG, fontsize=10)
    ax.set_xlim(0, vt2_max)
    ax.set_ylim(0)
    ax.legend(fontsize=8, labelcolor=_FG,
              facecolor='#1e2530', edgecolor='#333333', framealpha=0.7)

    fig.suptitle('CV Trajectory Error vs.  |v| · t²',
                 color=_FG, fontsize=12, y=1.01)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'cv_error_vs_vt2.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


# ===========================================================================
# Acceleration-based plots
# ===========================================================================

def _accel_unpack(vel_errors):
    """Extract (accels, tks, errs) arrays, dropping NaN-acceleration entries."""
    arr    = np.array(vel_errors)   # (M, 4): speed, accel, t_k, error
    valid  = ~np.isnan(arr[:, 1])
    accels = arr[valid, 1]
    tks    = arr[valid, 2]
    errs   = arr[valid, 3]
    return accels, tks, errs


def plot_error_vs_accel(vel_errors, output_dir, dt,
                        accel_max=5.0, accel_bin=0.25):
    """CV error vs agent acceleration magnitude, one line per future horizon.

    X-axis : acceleration |a|  (m/s²) — central difference in global frame
    Y-axis : mean L2 error (m)
    Lines  : one per future horizon (plasma colormap)
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    accels, tks, errs = _accel_unpack(vel_errors)
    if len(accels) == 0:
        return

    horizons  = sorted(set(tks.tolist()))
    bin_edges = np.arange(0, accel_max + accel_bin, accel_bin)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=_BG)
    _style_ax(ax)

    cmap = cm.get_cmap('plasma', len(horizons))
    for i, t_k in enumerate(horizons):
        mask_h = tks == t_k
        centres, means, _, p25s, p75s, _, _ = \
            _binned_stats(accels[mask_h], errs[mask_h], bin_edges)
        if len(centres) == 0:
            continue
        color = cmap(i / max(len(horizons) - 1, 1))
        ax.fill_between(centres, p25s, p75s, color=color, alpha=0.08)
        ax.plot(centres, means, color=color, linewidth=1.8,
                marker='o', markersize=3.5, label=f't={t_k:.1f}s')

    ax.set_xlabel('Acceleration |a|  (m/s²)', color=_FG, fontsize=10)
    ax.set_ylabel('Mean L2 error (m)',         color=_FG, fontsize=10)
    ax.set_title('CV Trajectory Error vs. Agent Acceleration per Horizon',
                 color=_FG, fontsize=11)
    ax.set_xlim(0, accel_max)
    ax.set_ylim(0)
    ax.legend(fontsize=8, labelcolor=_FG,
              facecolor='#1e2530', edgecolor='#333333', framealpha=0.7,
              ncol=max(1, len(horizons) // 3))

    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'cv_error_vs_accel.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


def _two_panel_accel(vel_errors, output_dir, dt,
                     x_arr, x_label, x_max, x_bin, fname):
    """Shared two-panel layout for acceleration-derived x-axes (|a|·t and |a|·t²).

    Left  : per-horizon lines (plasma colormap)
    Right : all horizons collapsed with binned stats + linear fit through origin
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    accels, tks, errs = _accel_unpack(vel_errors)
    if len(accels) == 0:
        return

    horizons  = sorted(set(tks.tolist()))
    bin_edges = np.arange(0, x_max + x_bin, x_bin)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=_BG)

    # ── Left: per-horizon ────────────────────────────────────────────────────
    ax = axes[0]
    _style_ax(ax)
    cmap = cm.get_cmap('plasma', len(horizons))
    for i, t_k in enumerate(horizons):
        mask_h = tks == t_k
        xs = x_arr[mask_h]
        centres, means, _, _, _, _, _ = _binned_stats(xs, errs[mask_h], bin_edges)
        if len(centres) == 0:
            continue
        color = cmap(i / max(len(horizons) - 1, 1))
        ax.plot(centres, means, color=color, linewidth=1.6,
                marker='o', markersize=3.5, label=f't={t_k:.1f}s', alpha=0.9)
    ax.set_xlabel(x_label,               color=_FG, fontsize=10)
    ax.set_ylabel('Mean L2 error (m)',   color=_FG, fontsize=10)
    ax.set_title(f'CV Error vs. {x_label}\n(per horizon)', color=_FG, fontsize=10)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0)
    ax.legend(fontsize=8, labelcolor=_FG, facecolor='#1e2530',
              edgecolor='#333333', framealpha=0.7,
              ncol=max(1, len(horizons) // 3))

    # ── Right: collapsed + linear fit ────────────────────────────────────────
    ax = axes[1]
    _style_ax(ax)
    centres, means, medians, p25s, p75s, p90s, _ = \
        _binned_stats(x_arr, errs, bin_edges)
    if len(centres) > 0:
        ax.fill_between(centres, p25s, p75s, color=_C1, alpha=0.18,
                        label='IQR (25–75th pct)')
        ax.plot(centres, means,   color=_C2, linewidth=2.0,
                marker='o', markersize=4, label='Mean')
        ax.plot(centres, medians, color=_C3, linewidth=1.6,
                linestyle='--', marker='s', markersize=3.5, label='Median')
        ax.plot(centres, p90s,    color=_C4, linewidth=1.2,
                linestyle=':', marker='^', markersize=3.5, label='P90')
        in_range = x_arr <= x_max
        xf, yf = x_arr[in_range], errs[in_range]
        if len(xf) > 10:
            alpha_fit = float(np.dot(xf, yf) / np.dot(xf, xf))
            x_line = np.array([0, x_max])
            ax.plot(x_line, alpha_fit * x_line,
                    color='white', linewidth=1.4, linestyle='--', alpha=0.55,
                    label=f'Linear fit α={alpha_fit:.4f}')
    ax.set_xlabel(x_label,             color=_FG, fontsize=10)
    ax.set_ylabel('L2 error (m)',      color=_FG, fontsize=10)
    ax.set_title(f'CV Error vs. {x_label}\n(all horizons collapsed)',
                 color=_FG, fontsize=10)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0)
    ax.legend(fontsize=8, labelcolor=_FG, facecolor='#1e2530',
              edgecolor='#333333', framealpha=0.7)

    fig.suptitle(f'CV Trajectory Error vs.  {x_label}',
                 color=_FG, fontsize=12, y=1.01)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, fname)
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


def plot_error_vs_at(vel_errors, output_dir, dt,
                     at_max=10.0, at_bin=0.5):
    """CV error vs |a|·t  (units: m/s = Δv magnitude over horizon t)."""
    accels, tks, errs = _accel_unpack(vel_errors)
    if len(accels) == 0:
        return
    _two_panel_accel(vel_errors, output_dir, dt,
                     x_arr   = accels * tks,
                     x_label = '|a| · t  (m/s)',
                     x_max   = at_max,
                     x_bin   = at_bin,
                     fname   = 'cv_error_vs_at.png')


def plot_error_vs_at2(vel_errors, output_dir, dt,
                      at2_max=15.0, at2_bin=0.75):
    """CV error vs |a|·t²  (units: m — the kinematic 0.5·a·t² correction term)."""
    accels, tks, errs = _accel_unpack(vel_errors)
    if len(accels) == 0:
        return
    _two_panel_accel(vel_errors, output_dir, dt,
                     x_arr   = accels * tks ** 2,
                     x_label = '|a| · t²  (m)',
                     x_max   = at2_max,
                     x_bin   = at2_bin,
                     fname   = 'cv_error_vs_at2.png')


# ===========================================================================
# Feature correlation analysis
# ===========================================================================

# Each entry: (display_name, fn(v, a, t) → feature_value)
# All inputs are 1-D numpy arrays of the same length (NaN-free).
_FEATURE_DEFS = [
    ('t',          lambda v, a, t: t),
    ('t²',         lambda v, a, t: t ** 2),
    ('v',          lambda v, a, t: v),
    ('v²',         lambda v, a, t: v ** 2),
    ('a',          lambda v, a, t: a),
    ('a²',         lambda v, a, t: a ** 2),
    ('v·t',        lambda v, a, t: v * t),
    ('v·t²',       lambda v, a, t: v * t ** 2),
    ('a·t',        lambda v, a, t: a * t),
    ('a·t²',       lambda v, a, t: a * t ** 2),
    ('v·t + a·t²', lambda v, a, t: v * t + a * t ** 2),
]


def compute_feature_correlations(vel_errors):
    """Compute Pearson r and Spearman ρ between each feature and L2 error.

    Returns
    -------
    overall    : dict[feature_name] → dict with pearson_r, pearson_r2,
                                                  spearman_r, spearman_r2
    per_horizon: dict[t_k] → dict[feature_name] → dict with pearson_r, spearman_r
    """
    from scipy.stats import pearsonr, spearmanr

    arr    = np.array(vel_errors)
    valid  = ~np.isnan(arr[:, 1])   # drop NaN-accel rows
    speeds = arr[valid, 0]
    accels = arr[valid, 1]
    tks    = arr[valid, 2]
    errs   = arr[valid, 3]

    # ── Overall ──────────────────────────────────────────────────────────────
    overall = {}
    for name, fn in _FEATURE_DEFS:
        x = fn(speeds, accels, tks)
        pr, _ = pearsonr(x, errs)
        sr, _ = spearmanr(x, errs)
        overall[name] = dict(pearson_r=float(pr),  pearson_r2=float(pr ** 2),
                             spearman_r=float(sr), spearman_r2=float(sr ** 2))

    # ── Per-horizon ───────────────────────────────────────────────────────────
    per_horizon = {}
    for t_k in sorted(set(tks.tolist())):
        mask = tks == t_k
        v_h, a_h, t_h, e_h = speeds[mask], accels[mask], tks[mask], errs[mask]
        if len(e_h) < 10:
            continue
        horizon_corr = {}
        for name, fn in _FEATURE_DEFS:
            x = fn(v_h, a_h, t_h)
            # t and t² are constant within a horizon — skip (correlation undefined)
            if np.std(x) < 1e-12:
                horizon_corr[name] = dict(pearson_r=float('nan'),
                                          spearman_r=float('nan'))
                continue
            pr, _ = pearsonr(x, e_h)
            sr, _ = spearmanr(x, e_h)
            horizon_corr[name] = dict(pearson_r=float(pr), spearman_r=float(sr))
        per_horizon[t_k] = horizon_corr

    return overall, per_horizon


def print_correlation_table(overall):
    """Print a sorted table of overall feature correlations."""
    print()
    print('Feature correlation with CV L2 error  (NaN-accel rows excluded)')
    header = f'  {"Feature":<18}  {"Pearson r":>10}  {"Pearson R²":>10}  {"Spearman ρ":>10}  {"Spearman R²":>11}'
    sep    = '  ' + '─' * (len(header) - 2)
    print(sep)
    print(header)
    print(sep)
    rows = sorted(overall.items(), key=lambda x: -abs(x[1]['pearson_r']))
    for name, c in rows:
        print(f'  {name:<18}  {c["pearson_r"]:>10.4f}  {c["pearson_r2"]:>10.4f}'
              f'  {c["spearman_r"]:>10.4f}  {c["spearman_r2"]:>11.4f}')
    print(sep)
    best = rows[0]
    print(f'  Best predictor (|Pearson r|): {best[0]}  '
          f'r={best[1]["pearson_r"]:.4f}  R²={best[1]["pearson_r2"]:.4f}')
    print()


def plot_correlation_bars(overall, output_dir):
    """Grouped bar chart of Pearson r and Spearman ρ per feature."""
    import matplotlib.pyplot as plt

    names  = [n for n, _ in _FEATURE_DEFS]
    pr     = [overall[n]['pearson_r']  for n in names]
    sr     = [overall[n]['spearman_r'] for n in names]

    # Sort by descending |Pearson r|
    order  = sorted(range(len(names)), key=lambda i: -abs(pr[i]))
    names  = [names[i]  for i in order]
    pr     = [pr[i]     for i in order]
    sr     = [sr[i]     for i in order]

    x     = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(9, len(names) * 1.2), 5), facecolor=_BG)
    _style_ax(ax)

    bars_p = ax.bar(x - width / 2, pr, width, color=_C1,
                    alpha=0.85, label='Pearson r',  edgecolor='none')
    bars_s = ax.bar(x + width / 2, sr, width, color=_C2,
                    alpha=0.85, label='Spearman ρ', edgecolor='none')
    ax.bar_label(bars_p, fmt='%.3f', color=_FG, fontsize=7, padding=3)
    ax.bar_label(bars_s, fmt='%.3f', color=_FG, fontsize=7, padding=3)

    ax.axhline(0, color='white', linewidth=0.6, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9, color=_FG)
    ax.set_ylabel('Correlation with L2 error', color=_FG, fontsize=10)
    ax.set_title('Feature Correlations with CV Trajectory Error\n'
                 '(sorted by |Pearson r|, NaN-accel rows excluded)',
                 color=_FG, fontsize=11)
    ax.legend(fontsize=9, labelcolor=_FG,
              facecolor='#1e2530', edgecolor='#333333', framealpha=0.7)
    ax.set_ylim(-0.05, 1.05)

    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'cv_error_feature_correlations.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


def plot_correlation_per_horizon(per_horizon, output_dir, dt):
    """Line chart: how each feature's Pearson r changes with horizon t."""
    import matplotlib.pyplot as plt

    if not per_horizon:
        return

    horizons = sorted(per_horizon.keys())
    names    = [n for n, _ in _FEATURE_DEFS]

    # Fixed colour per feature — use a qualitative palette
    palette  = ['#4fc3f7', '#ffb74d', '#69f0ae', '#ef5350', '#ce93d8',
                '#fff176', '#80cbc4', '#ffcc02', '#ff8a65', '#a5d6a7', '#f48fb1']
    color_of = {n: palette[i % len(palette)] for i, n in enumerate(names)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=_BG)

    for ax_idx, metric in enumerate(['pearson_r', 'spearman_r']):
        ax = axes[ax_idx]
        _style_ax(ax)
        label_str = 'Pearson r' if metric == 'pearson_r' else 'Spearman ρ'

        for name in names:
            ys = []
            xs = []
            for t_k in horizons:
                val = per_horizon[t_k].get(name, {}).get(metric, float('nan'))
                if not np.isnan(val):
                    xs.append(t_k)
                    ys.append(val)
            if xs:
                ax.plot(xs, ys, color=color_of[name], linewidth=1.6,
                        marker='o', markersize=4, label=name)

        ax.axhline(0, color='white', linewidth=0.6, alpha=0.4)
        ax.set_xlabel('Future horizon t (s)', color=_FG, fontsize=10)
        ax.set_ylabel(label_str,              color=_FG, fontsize=10)
        ax.set_title(f'{label_str} with L2 error vs. horizon',
                     color=_FG, fontsize=10)
        ax.set_xlim(0, max(horizons) + dt * 0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_locator(
            __import__('matplotlib').ticker.MultipleLocator(dt))
        ax.legend(fontsize=7.5, labelcolor=_FG, facecolor='#1e2530',
                  edgecolor='#333333', framealpha=0.7, ncol=2)

    fig.suptitle('Feature Correlations with CV Error per Horizon\n'
                 '(t and t² omitted — constant within each horizon)',
                 color=_FG, fontsize=11, y=1.01)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'cv_error_correlations_per_horizon.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


# ===========================================================================
# Matching-distance model fitting
# ===========================================================================

# Candidate models: (name, fn(v, a, t, params) → prediction, n_params, init)
# All are constrained to pass through origin (no intercept) for physical sense.
# params are fitted via quantile regression (pinball loss minimisation).

_MODEL_DEFS = [
    # ── Single-feature baselines ──────────────────────────────────────────
    ('α·t',            lambda v, a, t: np.column_stack([t])),
    ('α·t²',           lambda v, a, t: np.column_stack([t ** 2])),
    ('α·v·t',          lambda v, a, t: np.column_stack([v * t])),
    ('α·v·t²',         lambda v, a, t: np.column_stack([v * t ** 2])),
    ('α·a·t',          lambda v, a, t: np.column_stack([a * t])),
    ('α·a·t²',         lambda v, a, t: np.column_stack([a * t ** 2])),
    ('α·v²·t²',        lambda v, a, t: np.column_stack([v ** 2 * t ** 2])),
    ('α·a²·t²',        lambda v, a, t: np.column_stack([a ** 2 * t ** 2])),
    # ── Two-feature combinations ──────────────────────────────────────────
    ('α·t + β·v·t',    lambda v, a, t: np.column_stack([t, v * t])),
    ('α·v·t + β·t²',   lambda v, a, t: np.column_stack([v * t, t ** 2])),
    ('α·v·t + β·a·t²', lambda v, a, t: np.column_stack([v * t, a * t ** 2])),
    ('α·v·t + β·v·t²', lambda v, a, t: np.column_stack([v * t, v * t ** 2])),
    ('α·t + β·v²·t²',  lambda v, a, t: np.column_stack([t, v ** 2 * t ** 2])),
    ('α·t + β·a·t²',   lambda v, a, t: np.column_stack([t, a * t ** 2])),
    ('α·a·t + β·v·t',  lambda v, a, t: np.column_stack([a * t, v * t])),
]


def _fit_quantile(X, y, quantile=0.90, max_iter=2000):
    """Fit a non-negative least-squares quantile regression (pinball loss).

    Uses scipy.optimize.minimize with L-BFGS-B so all coefficients >= 0
    (physically: distance thresholds should increase with each feature).

    Returns
    -------
    coeffs : ndarray (n_params,)
    coverage : float  — fraction of samples with y <= X @ coeffs
    mae_pos  : float  — mean pinball loss on positive residuals
    """
    from scipy.optimize import minimize

    n, p = X.shape

    def pinball(c):
        resid = y - X @ c
        loss  = np.where(resid >= 0,
                         quantile * resid,
                         (quantile - 1.0) * resid)
        return float(loss.mean())

    def pinball_grad(c):
        resid = y - X @ c
        w     = np.where(resid >= 0, -quantile, (1.0 - quantile))
        return (X.T @ w) / n

    x0  = np.ones(p) * 0.1
    res = minimize(pinball, x0, jac=pinball_grad, method='L-BFGS-B',
                   bounds=[(0, None)] * p,
                   options={'maxiter': max_iter, 'ftol': 1e-10})
    c        = res.x
    pred     = X @ c
    coverage = float(np.mean(y <= pred))
    # Quantile loss on positive residuals only (as a scale)
    pos      = y[y > pred]
    mae_pos  = float(pos.mean()) if len(pos) else 0.0
    return c, coverage, mae_pos


def fit_matching_distance_models(vel_errors, output_dir, dt,
                                 quantile=0.90,
                                 v_max=20.0, t_max=6.5):
    """Fit and compare candidate matching-distance models.

    For each model f(v, a, t; θ) the coefficients θ are found by minimising
    the pinball (quantile) loss at the requested quantile — so the fitted
    curve bounds `quantile` fraction of CV errors from below.  This directly
    gives a variable matching distance that adapts to speed, acceleration, and
    horizon.

    Outputs
    -------
    • Console table: model name, coefficients, coverage, pinball loss
    • Plot: fitted curves vs empirical quantile at each horizon (by speed bucket)
    • Saved npy: best-model coefficients for downstream use
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    arr    = np.array(vel_errors)
    valid  = ~np.isnan(arr[:, 1])      # drop NaN-accel rows
    speeds = arr[valid, 0]
    accels = arr[valid, 1]
    tks    = arr[valid, 2]
    errs   = arr[valid, 3]

    # Clip extreme outliers (>99th pct) so they don't dominate the fit
    p99    = np.percentile(errs, 99)
    keep   = (errs <= p99) & (speeds <= v_max) & (tks <= t_max)
    v, a, t, e = speeds[keep], accels[keep], tks[keep], errs[keep]

    print(f'\n  Fitting matching-distance models  '
          f'(quantile={quantile:.0%}, N={keep.sum():,}) …')

    results = []
    for name, feat_fn in _MODEL_DEFS:
        X = feat_fn(v, a, t)
        if X.ndim == 1:
            X = X[:, None]
        coeffs, coverage, mae_pos = _fit_quantile(X, e, quantile=quantile)
        # Pseudo-R²: 1 - loss_model / loss_null
        # Null model: predict the empirical quantile everywhere
        null_pred = np.full(len(e), np.quantile(e, quantile))
        null_resid = e - null_pred
        null_loss  = float(np.where(null_resid >= 0,
                                    quantile * null_resid,
                                    (quantile - 1) * null_resid).mean())
        model_resid = e - X @ coeffs
        model_loss  = float(np.where(model_resid >= 0,
                                     quantile * model_resid,
                                     (quantile - 1) * model_resid).mean())
        pseudo_r2 = 1.0 - model_loss / null_loss if null_loss > 0 else float('nan')
        results.append(dict(name=name, coeffs=coeffs, coverage=coverage,
                            mae_pos=mae_pos, pseudo_r2=pseudo_r2,
                            feat_fn=feat_fn))

    # ── Print table ──────────────────────────────────────────────────────────
    print()
    print(f'  Quantile ({quantile:.0%}) regression — matching distance models')
    print(f'  {"Model":<26}  {"Coverage":>8}  {"PseudoR²":>9}  Coefficients')
    print('  ' + '─' * 70)
    for r in sorted(results, key=lambda x: -x['pseudo_r2']):
        coeff_str = '  '.join(f'{c:.4f}' for c in r['coeffs'])
        print(f'  {r["name"]:<26}  {r["coverage"]:>7.1%}  {r["pseudo_r2"]:>9.4f}  '
              f'[{coeff_str}]')
    print()

    best = max(results, key=lambda x: x['pseudo_r2'])
    print(f'  Best model: {best["name"]}  (pseudo-R²={best["pseudo_r2"]:.4f})')

    # Save best-model coefficients
    np.save(os.path.join(output_dir, 'matching_distance_model.npy'),
            {'name': best['name'], 'coeffs': best['coeffs'],
             'quantile': quantile})

    # ── Plot: empirical quantile per horizon vs fitted models ─────────────────
    horizons = sorted(set(tks.tolist()))
    emp_q    = []
    for t_k in horizons:
        mask = tks == t_k
        if mask.sum() > 0:
            emp_q.append(float(np.quantile(errs[mask], quantile)))
        else:
            emp_q.append(float('nan'))
    emp_q = np.array(emp_q)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=_BG)

    # Left: all models vs empirical quantile (median-speed agent)
    ax = axes[0]
    _style_ax(ax)
    v_med = float(np.median(speeds))
    a_med = float(np.median(accels))
    t_arr = np.array(horizons)

    ax.plot(t_arr, emp_q, color='white', linewidth=2.2, linestyle='--',
            marker='o', markersize=5, label=f'Empirical P{quantile*100:.0f}', zorder=5)

    palette = ['#4fc3f7', '#ffb74d', '#69f0ae', '#ef5350', '#ce93d8',
               '#fff176', '#ff8a65']
    for idx, r in enumerate(results):
        X_line = r['feat_fn'](
            np.full_like(t_arr, v_med),
            np.full_like(t_arr, a_med),
            t_arr)
        if X_line.ndim == 1:
            X_line = X_line[:, None]
        pred = X_line @ r['coeffs']
        lw   = 2.2 if r['name'] == best['name'] else 1.2
        ls   = '-'  if r['name'] == best['name'] else ':'
        ax.plot(t_arr, pred, color=palette[idx % len(palette)],
                linewidth=lw, linestyle=ls,
                label=f'{r["name"]}  (R²={r["pseudo_r2"]:.3f})')

    ax.set_xlabel('Future horizon t (s)', color=_FG, fontsize=10)
    ax.set_ylabel(f'Matching distance (m)  [P{quantile*100:.0f}]',
                  color=_FG, fontsize=10)
    ax.set_title(f'Model fits vs empirical P{quantile*100:.0f}\n'
                 f'(v={v_med:.1f} m/s, a={a_med:.2f} m/s² agent)',
                 color=_FG, fontsize=10)
    ax.set_xlim(0, max(horizons) + dt * 0.5)
    ax.set_ylim(0)
    ax.legend(fontsize=7.5, labelcolor=_FG, facecolor='#1e2530',
              edgecolor='#333333', framealpha=0.8, ncol=1)

    # Right: best model vs empirical quantile per speed bucket
    ax = axes[1]
    _style_ax(ax)
    ax.plot(t_arr, emp_q, color='white', linewidth=2.0, linestyle='--',
            marker='o', markersize=5, label=f'Empirical P{quantile*100:.0f} (all)', zorder=5)

    for (lo, hi, label, color) in _SPEED_BUCKETS:
        bucket_q = []
        bucket_pred = []
        v_rep = (lo + hi) / 2
        for t_k in horizons:
            mask = (tks == t_k) & (speeds >= lo) & (speeds < hi)
            if mask.sum() >= 10:
                bucket_q.append(float(np.quantile(errs[mask], quantile)))
            else:
                bucket_q.append(float('nan'))
            X1 = best['feat_fn'](
                np.array([v_rep]), np.array([a_med]), np.array([t_k]))
            if X1.ndim == 1:
                X1 = X1[:, None]
            bucket_pred.append(float(X1 @ best['coeffs']))

        bq = np.array(bucket_q)
        bp = np.array(bucket_pred)
        valid_b = ~np.isnan(bq)
        if valid_b.sum() < 2:
            continue
        ax.plot(t_arr[valid_b], bq[valid_b], color=color, linewidth=1.4,
                linestyle='--', marker='s', markersize=3.5, alpha=0.8)
        ax.plot(t_arr, bp, color=color, linewidth=1.8,
                linestyle='-', label=label)

    ax.set_xlabel('Future horizon t (s)', color=_FG, fontsize=10)
    ax.set_ylabel(f'Matching distance (m)  [P{quantile*100:.0f}]',
                  color=_FG, fontsize=10)
    ax.set_title(f'Best model ({best["name"]}) vs empirical P{quantile*100:.0f}\n'
                 'per speed bucket  (solid=model, dashed=empirical)',
                 color=_FG, fontsize=10)
    ax.set_xlim(0, max(horizons) + dt * 0.5)
    ax.set_ylim(0)
    ax.legend(fontsize=8, labelcolor=_FG, facecolor='#1e2530',
              edgecolor='#333333', framealpha=0.7)

    fig.suptitle(f'Matching Distance Model Comparison  '
                 f'(quantile={quantile:.0%})',
                 color=_FG, fontsize=12, y=1.01)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'matching_distance_model_fit.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'  Saved → {out}')
    plt.close(fig)

    return best


# ===========================================================================
# Per-split runner
# ===========================================================================

def run_split(pkl_path, args, split_tag=''):
    tag  = f'[{split_tag}] ' if split_tag else ''
    out  = os.path.join(args.output_dir, split_tag) if split_tag else args.output_dir
    os.makedirs(out, exist_ok=True)

    print(f'\n{tag}Loading {pkl_path} …')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    infos    = data['infos']
    n_scenes = len({i['scene_token'] for i in infos})
    print(f'{tag}{len(infos):,} samples across {n_scenes} scenes')

    print(f'{tag}Collecting CV errors …')
    errors_per_step, class_errors, vel_errors, T = collect_cv_errors(
        infos,
        dt                = args.dt,
        include_synthetic = args.include_synthetic,
        valid_only        = args.valid_only,
        classes           = args.classes,
    )

    stats = compute_stats(errors_per_step, args.dt)
    print_stats(stats)
    print_class_stats(class_errors, args.dt)

    print(f'{tag}Generating plots …')
    plot_histograms(errors_per_step, stats, out, args.dt,
                    max_err=args.max_err_plot)
    plot_boxplots(errors_per_step, stats, out, args.dt,
                  max_err=args.max_err_plot)
    plot_ade_curve(stats, out, args.dt)
    plot_cdf(errors_per_step, stats, out, args.dt,
             x_max=args.max_err_plot)
    plot_class_ade(class_errors, out, args.dt)
    plot_ade_curve_by_speed(vel_errors, stats, out, args.dt)
    plot_error_vs_t2(vel_errors, stats, out, args.dt)
    plot_error_vs_speed(vel_errors, out, args.dt)
    plot_error_vs_vt(vel_errors, out, args.dt)
    plot_error_vs_vt2(vel_errors, out, args.dt)
    plot_error_vs_accel(vel_errors, out, args.dt)
    plot_error_vs_at(vel_errors, out, args.dt)
    plot_error_vs_at2(vel_errors, out, args.dt)

    print(f'{tag}Computing feature correlations …')
    overall_corr, per_horizon_corr = compute_feature_correlations(vel_errors)
    print_correlation_table(overall_corr)
    plot_correlation_bars(overall_corr, out)
    plot_correlation_per_horizon(per_horizon_corr, out, args.dt)

    print(f'{tag}Fitting matching-distance models …')
    fit_matching_distance_models(vel_errors, out, args.dt,
                                 quantile=args.match_quantile)

    print(f'{tag}Done.  Outputs in {out}/')


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Input — either a single pkl or a directory to loop over train+val
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument('--pkl', default='data/infos/nuscenes_infos_train.pkl',
                     help='Single input pkl file to evaluate')
    grp.add_argument('--data-dir', default=None, metavar='DIR',
                     help='Directory containing nuscenes_infos_*.pkl files. '
                          'Processes train and val splits.')

    ap.add_argument('--dt', type=float, default=0.5,
                    help='Seconds per future trajectory step (default: 0.5 — '
                         'nuScenes 2 Hz keyframe rate)')
    ap.add_argument('--include-synthetic', action='store_true',
                    help='Include is_interpolated / is_extrapolated boxes '
                         '(requires an _occ pkl from nuscenes_occlusion_converter.py)')
    ap.add_argument('--valid-only', action='store_true',
                    help='Only process boxes with valid_flag=True '
                         '(≥1 lidar or radar point)')
    ap.add_argument('--classes', nargs='+', default=None, metavar='CLS',
                    help='Agent class names to include (default: all). '
                         'E.g. --classes car truck bus trailer')
    ap.add_argument('--max-err-plot', type=float, default=20.0,
                    help='X-axis clip for histograms and box-plots (default: 20 m). '
                         'Outliers beyond this are excluded from the plot '
                         'but still counted in statistics.')
    ap.add_argument('--match-quantile', type=float, default=0.90, metavar='Q',
                    help='Quantile to fit for matching-distance models '
                         '(default: 0.90 → 90%% of CV errors bounded)')
    ap.add_argument('--output-dir', default='vis/cv_traj_error',
                    help='Root directory for output plots '
                         '(default: vis/cv_traj_error)')
    args = ap.parse_args()

    # Default to val pkl if neither --pkl nor --data-dir given
    if args.pkl is None and args.data_dir is None:
        args.pkl = 'data/infos/nuscenes_infos_val.pkl'

    if args.pkl is not None:
        run_split(args.pkl, args)
    else:
        # Loop over train + val (skip missing)
        suffix = '_occ' if args.include_synthetic else ''
        for split in ('train', 'val'):
            pkl = os.path.join(args.data_dir,
                               f'nuscenes_infos_{split}{suffix}.pkl')
            if not os.path.exists(pkl):
                print(f'Skipping {pkl} (not found)')
                continue
            run_split(pkl, args, split_tag=split)

    print('\nAll done.')


if __name__ == '__main__':
    main()
