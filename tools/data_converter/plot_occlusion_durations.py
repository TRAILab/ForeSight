#!/usr/bin/env python3
"""Plot histograms of occluded-annotation statistics from nuScenes occ pkl files.

Uses the converter output pkls (nuscenes_infos_{split}_occ.pkl) which already
have is_interpolated / is_extrapolated flags on every box, so all three
occluded types have known positions and can be included in every plot.

Occluded box types
------------------
  interp  : is_interpolated == True   (mid-track gap, agent reappears)
  extrap  : is_extrapolated == True   (tail after last observation)
  zero    : num_lidar_pts < 1, original box (annotated but no lidar returns)

Outputs  (all in vis/occ_stats/)
-------
  • occlusion_duration_splits.png   — occluded intervals vs duration, train vs val
  • occlusion_duration_types.png    — val intervals by type (interp/zero/extrap)
  • occlusion_ego_dist_hist.png     — occluded boxes vs ego distance, train vs val
  • occlusion_duration_cdf.png      — val CDF of occluded-interval duration (to 20 s)
  • occlusion_ego_dist_cdf.png      — val CDF of occluded-box ego distance (to 60 m)
  • occlusion_ego_dist_types.png    — val ego distance by type (interp vs extrap)

Run
---
    python tools/data_converter/plot_occlusion_durations.py
"""

import os
import pickle
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Hard-coded configuration
# ---------------------------------------------------------------------------
TRAIN_PKL = 'data/infos/nuscenes_infos_train_occ.pkl'
VAL_PKL   = 'data/infos/nuscenes_infos_val_occ.pkl'
OUT_DIR   = 'vis/occ_stats'

# Duration histogram settings.
MAX_DURATION = 20.0   # seconds — clip long-tail for display
BIN_WIDTH    = 0.5    # seconds — matches nuScenes 2 Hz keyframe interval

# Ego-distance histogram settings.
MAX_DIST      = 100.0  # metres — used for CDF x-axis
HIST_DIST_MAX =  60.0  # metres — cutoff for ego-distance histograms and CDF
DIST_BIN      =   2.0  # metres per bin


# ---------------------------------------------------------------------------
# Plot style  (white background for publication)
# ---------------------------------------------------------------------------
_COLORS = {
    'train':  '#2171b5',   # blue
    'val':    '#238b45',   # green
    'interp': '#238b45',   # green — interpolation gaps (temporary)
    'zero':   '#d94801',   # orange — zero lidar pts
    'extrap': '#2171b5',   # blue  — extrapolation tail (permanent)
}
_GRID = '#cccccc'

_RC = {
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'axes.edgecolor':    'black',
    'axes.labelcolor':   'black',
    'axes.grid':         True,
    'grid.color':        _GRID,
    'grid.linewidth':    0.6,
    'grid.linestyle':    '--',
    'xtick.color':       'black',
    'ytick.color':       'black',
    'legend.framealpha': 0.9,
    'legend.edgecolor':  '#aaaaaa',
    'font.size':         14,
    'axes.labelsize':    15,
    'xtick.labelsize':   13,
    'ytick.labelsize':   13,
    'legend.fontsize':   12,
}


# ---------------------------------------------------------------------------
# Core collection
# ---------------------------------------------------------------------------

def collect_stats(infos):
    """Collect occlusion durations, ego distances, and counts from an occ pkl.

    Every box has is_interpolated / is_extrapolated flags and a known position,
    so all three occluded types are included in every metric.

    Returns
    -------
    all_durs         : list[float]  — duration (s) of every contiguous occluded interval
    interp_durs      : list[float]  — interpolation-gap intervals + mid-track zero-lidar runs
    zero_durs        : list[float]  — zero-lidar-pt intervals (raw, for terminal stats only)
    extrap_durs      : list[float]  — extrapolation-tail intervals + trailing zero-lidar runs
    ego_dists        : list[float]  — ego distance (m) for every occluded box
    ego_dists_interp : list[float]  — ego distance for interp + mid-track zero-lidar boxes
    ego_dists_extrap : list[float]  — ego distance for extrap + trailing zero-lidar boxes
    n_tracks_total    : int
    n_tracks_occluded : int
    """
    scene_to_indices = defaultdict(list)
    for gi, info in enumerate(infos):
        scene_to_indices[info['scene_token']].append(gi)
    for sc in scene_to_indices:
        scene_to_indices[sc].sort(key=lambda i: infos[i]['timestamp'])

    all_durs    = []
    interp_durs = []
    zero_durs   = []
    extrap_durs = []
    ego_dists        = []
    ego_dists_interp = []
    ego_dists_extrap = []

    n_tracks_total    = 0
    n_tracks_occluded = 0

    for scene_token, global_indices in scene_to_indices.items():
        timestamps = [infos[gi]['timestamp'] * 1e-6 for gi in global_indices]

        tracks = defaultdict(list)
        for frame_pos, gi in enumerate(global_indices):
            info = infos[gi]
            for bi, inst_ind in enumerate(info['instance_inds']):
                tracks[inst_ind].append((frame_pos, gi, bi))

        for inst_ind, appearances in tracks.items():
            n_tracks_total += 1

            # Build per-frame list: (frame_pos, t, is_occluded, occ_type, ego_dist).
            frames = []
            for fp, gi, bi in appearances:
                info = infos[gi]
                is_interp  = bool(info['is_interpolated'][bi])
                is_extrap  = bool(info['is_extrapolated'][bi])
                zero_lidar = (int(info['num_lidar_pts'][bi])
                              + int(info['num_radar_pts'][bi])) < 1
                is_occ     = is_interp or is_extrap or zero_lidar
                occ_type   = ('interp' if is_interp
                               else 'extrap' if is_extrap
                               else 'zero' if zero_lidar
                               else None)
                bx = float(info['gt_boxes'][bi, 0])
                by = float(info['gt_boxes'][bi, 1])
                frames.append((fp, timestamps[fp], is_occ, occ_type,
                               float(np.hypot(bx, by))))

            # Last frame position with an actual observation (for zero-lidar
            # classification into interp vs extrap ego-distance lists).
            last_obs_fp = -1
            for fp, _, is_occ, _, _ in frames:
                if not is_occ:
                    last_obs_fp = fp

            # Ego distances for all occluded boxes, split by type.
            for fp, _, is_occ, occ_type, dist in frames:
                if is_occ:
                    ego_dists.append(dist)
                    if occ_type == 'interp':
                        ego_dists_interp.append(dist)
                    elif occ_type == 'extrap':
                        ego_dists_extrap.append(dist)
                    else:  # zero-lidar: mid-track → interp, trailing → extrap
                        if fp < last_obs_fp:
                            ego_dists_interp.append(dist)
                        else:
                            ego_dists_extrap.append(dist)

            # Find contiguous occluded intervals.
            occ_start_t    = None
            occ_type_start = None
            track_has_occ  = False

            for _, t, is_occ, occ_type, _ in frames:
                if is_occ:
                    if occ_start_t is None:
                        occ_start_t    = t
                        occ_type_start = occ_type
                        track_has_occ  = True
                else:
                    if occ_start_t is not None:
                        dur = t - occ_start_t
                        if dur > 0:
                            all_durs.append(dur)
                            # zero-lidar mid-track → merge into interp
                            resolved = (occ_type_start if occ_type_start != 'zero'
                                        else 'interp')
                            {'interp': interp_durs,
                             'extrap': extrap_durs}[resolved].append(dur)
                            if occ_type_start == 'zero':
                                zero_durs.append(dur)
                        occ_start_t = None

            # Close any trailing interval.
            if occ_start_t is not None:
                dur = frames[-1][1] - occ_start_t
                if dur > 0:
                    all_durs.append(dur)
                    # zero-lidar trailing → merge into extrap
                    resolved = (occ_type_start if occ_type_start != 'zero'
                                else 'extrap')
                    {'interp': interp_durs,
                     'extrap': extrap_durs}[resolved].append(dur)
                    if occ_type_start == 'zero':
                        zero_durs.append(dur)

            if track_has_occ:
                n_tracks_occluded += 1

    return (all_durs, interp_durs, zero_durs, extrap_durs, ego_dists,
            ego_dists_interp, ego_dists_extrap,
            n_tracks_total, n_tracks_occluded)


def _print_stats(label, durs):
    if not durs:
        print(f'  {label}: no intervals found.')
        return
    a = np.array(durs)
    print(f'  {label}: {len(a):,} intervals | '
          f'median={np.median(a):.2f}s  mean={np.mean(a):.2f}s  '
          f'p90={np.percentile(a, 90):.2f}s  max={np.max(a):.2f}s')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    split_data = {}
    for split, path in [('train', TRAIN_PKL), ('val', VAL_PKL)]:
        print(f'\nLoading {split}: {path} …')
        with open(path, 'rb') as f:
            data = pickle.load(f)
        infos    = data['infos']
        n_scenes = len({i['scene_token'] for i in infos})
        print(f'  {len(infos):,} samples across {n_scenes} scenes.')

        print('  Collecting statistics …')
        all_d, interp_d, zero_d, extrap_d, ego_d, \
            ego_d_interp, ego_d_extrap, n_total, n_occ = \
            collect_stats(infos)

        # Box counts directly from flags.
        n_boxes_total  = sum(len(info['gt_boxes']) for info in infos)
        n_interp       = sum(int(np.sum( info['is_interpolated'])) for info in infos)
        n_extrap       = sum(int(np.sum( info['is_extrapolated'])) for info in infos)
        n_zero         = sum(int(np.sum(
            ~info['is_interpolated'] & ~info['is_extrapolated']
            & ((info['num_lidar_pts'] + info['num_radar_pts']) < 1))) for info in infos)
        n_occ_boxes    = n_interp + n_extrap + n_zero

        def pct(n): return 100.0 * n / n_boxes_total

        print(f'  Tracks total        : {n_total:,}')
        print(f'  Tracks occluded     : {n_occ:,}  '
              f'({100.0 * n_occ / n_total:.1f}% of all tracks)')
        print(f'  Boxes total         : {n_boxes_total:,}')
        print(f'  Boxes 0 pts         : {n_zero:,}  ({pct(n_zero):.1f}%)'
              f'  ← annotated, no lidar returns')
        print(f'  Boxes interp        : {n_interp:,}  ({pct(n_interp):.1f}%)'
              f'  ← mid-track gap, agent reappears')
        print(f'  Boxes extrap        : {n_extrap:,}  ({pct(n_extrap):.1f}%)'
              f'  ← tail after last observation')
        print(f'  Boxes occluded total: {n_occ_boxes:,}  ({pct(n_occ_boxes):.1f}%)')
        _print_stats('All intervals',       all_d)
        _print_stats('  Interp (gap)',       interp_d)
        _print_stats('  Zero-lidar',         zero_d)
        _print_stats('  Extrap (tail)',       extrap_d)
        print(f'  Ego dist (occluded) : {len(ego_d):,} boxes | '
              f'median={np.median(ego_d):.1f}m  '
              f'p90={np.percentile(ego_d, 90):.1f}m  '
              f'max={np.max(ego_d):.1f}m')

        split_data[split] = {
            'all':         np.array(all_d,        dtype=np.float32),
            'interp':      np.array(interp_d,      dtype=np.float32),
            'zero':        np.array(zero_d,        dtype=np.float32),
            'extrap':      np.array(extrap_d,      dtype=np.float32),
            'dist':        np.array(ego_d,         dtype=np.float32),
            'dist_interp': np.array(ego_d_interp,  dtype=np.float32),
            'dist_extrap': np.array(ego_d_extrap,  dtype=np.float32),
        }

    dur_bins       = np.arange(0, MAX_DURATION + BIN_WIDTH, BIN_WIDTH)
    dist_bins      = np.arange(0, MAX_DIST      + DIST_BIN,  DIST_BIN)
    hist_dist_bins = np.arange(0, HIST_DIST_MAX + DIST_BIN,  DIST_BIN)

    def _save(fig, name):
        path = os.path.join(OUT_DIR, name)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved → {path}')
        plt.close(fig)

    with plt.rc_context(_RC):

        # ── Plot 1: Duration — train vs val ───────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for split, arrays in split_data.items():
            durs = arrays['all']
            ax.hist(np.clip(durs, 0, MAX_DURATION), bins=dur_bins,
                    alpha=0.70, color=_COLORS[split], edgecolor='none',
                    label=f'{split.capitalize()} ({len(durs):,})')
        ax.set_yscale('log')
        ax.set_xlabel('Occlusion duration (s)')
        ax.set_ylabel('Occluded intervals')
        ax.set_xlim(0, MAX_DURATION)
        ax.legend()
        plt.tight_layout()
        _save(fig, 'occlusion_duration_splits.png')

        # ── Plot 2: Duration — by type (val) ──────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for key, label in [('extrap', 'Permanent'),
                            ('interp', 'Temporary')]:
            durs = split_data['val'][key]
            if not len(durs):
                continue
            ax.hist(np.clip(durs, 0, MAX_DURATION), bins=dur_bins,
                    alpha=0.75, color=_COLORS[key], edgecolor='none',
                    label=f'{label} ({len(durs):,} intervals)')
        ax.set_yscale('log')
        ax.set_xlabel('Occlusion duration (s)')
        ax.set_ylabel('Occluded intervals')
        ax.set_xlim(0, MAX_DURATION)
        ax.legend()
        plt.tight_layout()
        _save(fig, 'occlusion_duration_types.png')

        # ── Plot 3: Duration CDF (val, normalised to 20 s) ───────────────────
        fig, ax = plt.subplots(figsize=(7, 4.5))
        val_durs_clip = np.sort(
            split_data['val']['all'][split_data['val']['all'] <= MAX_DURATION])
        if len(val_durs_clip):
            cdf = np.arange(1, len(val_durs_clip) + 1) / len(val_durs_clip)
            ax.plot(val_durs_clip, cdf, color=_COLORS['val'], linewidth=2.0)
            for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
                v = float(np.quantile(val_durs_clip, q))
                ax.axvline(v, color='black', linewidth=0.9, linestyle='--', alpha=0.45)
                ax.text(v + 0.2, q - 0.05, f'P{int(q*100)}={v:.1f} s',
                        color='black', fontsize=9)
        ax.set_xlabel('Occlusion duration (s)')
        ax.set_ylabel('Cumulative fraction')
        ax.set_xlim(0, MAX_DURATION)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        _save(fig, 'occlusion_duration_cdf.png')

        # ── Plot 4: Ego distance histogram — train vs val ─────────────────────
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for split, arrays in split_data.items():
            dists = arrays['dist'][arrays['dist'] <= HIST_DIST_MAX]
            ax.hist(dists, bins=hist_dist_bins,
                    alpha=0.70, color=_COLORS[split], edgecolor='none',
                    label=f'{split.capitalize()} ({len(arrays["dist"]):,} boxes)')
        ax.set_yscale('log')
        ax.set_xlabel('Ego distance (m)')
        ax.set_ylabel('Occluded boxes')
        ax.set_xlim(0, HIST_DIST_MAX)
        ax.legend()
        plt.tight_layout()
        _save(fig, 'occlusion_ego_dist_hist.png')

        # ── Plot 4: Ego distance CDF (val, normalised to 60 m) ───────────────
        fig, ax = plt.subplots(figsize=(7, 4.5))
        val_dists_60 = np.sort(
            split_data['val']['dist'][split_data['val']['dist'] <= HIST_DIST_MAX])
        if len(val_dists_60):
            cdf = np.arange(1, len(val_dists_60) + 1) / len(val_dists_60)
            ax.plot(val_dists_60, cdf, color=_COLORS['val'], linewidth=2.0)
            for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
                v = float(np.quantile(val_dists_60, q))
                ax.axvline(v, color='black', linewidth=0.9, linestyle='--', alpha=0.45)
                ax.text(v + 0.8, q - 0.05, f'P{int(q*100)}={v:.0f} m',
                        color='black', fontsize=9)
        ax.set_xlabel('Ego distance (m)')
        ax.set_ylabel('Cumulative fraction')
        ax.set_xlim(0, HIST_DIST_MAX)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        _save(fig, 'occlusion_ego_dist_cdf.png')

        # ── Plot 5: Ego distance histogram — by type (val) ───────────────────
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for key, label in [('dist_extrap', 'Permanent'),
                            ('dist_interp', 'Temporary')]:
            arr = split_data['val'][key]
            dists = arr[arr <= HIST_DIST_MAX]
            if not len(dists):
                continue
            color_key = key.split('_')[1]   # 'interp' or 'extrap'
            ax.hist(dists, bins=hist_dist_bins,
                    alpha=0.75, color=_COLORS[color_key], edgecolor='none',
                    label=f'{label} ({len(arr):,} boxes)')
        ax.set_yscale('log')
        ax.set_xlabel('Ego distance (m)')
        ax.set_ylabel('Occluded boxes')
        ax.set_xlim(0, HIST_DIST_MAX)
        ax.legend()
        plt.tight_layout()
        _save(fig, 'occlusion_ego_dist_types.png')


if __name__ == '__main__':
    main()
