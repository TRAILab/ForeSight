#!/usr/bin/env python3
"""Visualise raw UniTraj ML predictions from an inference NPZ file.

For each sampled prediction row the script draws a bird's-eye-view plot showing:
  • All 6 predicted modes in thin blue, with mode 0 (used by the converter) bold.
  • A vertical marker at the converter handover step (_ml_last = 54) so you can
    see how much of the horizon is actually used.
  • The discarded tail (steps _ml_last+1 … 59) in red so artefacts are obvious.
  • A constant-velocity baseline computed from the world-state velocity.
  • The agent's heading arrow at the prediction origin.

Usage
-----
# 12 random predictions from the file:
python viz_ml_predictions.py --npz path/to/inference.npz

# Fix the random seed for reproducibility:
python viz_ml_predictions.py --npz path/to/inference.npz --seed 0

# Only show predictions for a specific scene:
python viz_ml_predictions.py --npz path/to/inference.npz --scene scene-0003

# Only show predictions for a specific instance token:
python viz_ml_predictions.py --npz path/to/inference.npz \
    --instance c8ac146d6cef45bfadbec2f08737c79f

# Save to a custom path:
python viz_ml_predictions.py --npz path/to/inference.npz --output preds.png
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ── constants matching the converter ──────────────────────────────────────────
_UNITRAJ_DT = 0.1   # seconds per step
_ML_SPAN    = 5
_ML_LAST    = 58    # last step used by converter (step 59 snaps to ~zero)
_N_STEPS    = 60

_TYPE_NAMES = {1: 'VEHICLE', 2: 'PEDESTRIAN', 3: 'CYCLIST'}

# ── colour palette (white background) ─────────────────────────────────────────
_C_BG        = '#ffffff'
_C_MODE0     = '#1565c0'   # mode 0 (bold blue)
_C_MODES     = '#90caf9'   # modes 1–5 (light blue)
_C_TAIL      = '#b71c1c'   # used only in step-59 histogram
_C_CV        = '#2e7d32'   # CV baseline (dark green)
_C_ORIGIN    = '#555555'   # agent origin dot
_C_GRID      = '#bbbbbb'
_C_VX        = '#c62828'   # vx velocity line (red)
_C_VY        = '#6a1b9a'   # vy velocity line (purple)


# ── coordinate helpers ────────────────────────────────────────────────────────

def _pred_to_global(preds_row, world):
    """Convert all 60 steps of all 6 modes to global XY.

    Parameters
    ----------
    preds_row : (6, 60, 2)  agent-centric cumulative offsets
    world     : (10,)  [x_g, y_g, z_g, l, w, h, heading, vx, vy, ...]

    Returns
    -------
    global_xy : (6, 60, 2)  in global (metric) coordinates
    """
    theta = float(world[6])
    c, s  = np.cos(theta), np.sin(theta)
    R     = np.array([[c, -s], [s, c]])          # agent-centric → global
    origin = world[:2]
    # preds_row: (6, 60, 2) → (6, 60, 2)
    return (preds_row @ R.T) + origin            # broadcast over modes & steps


def _cv_baseline(world, n_steps=_N_STEPS, dt=_UNITRAJ_DT):
    """Constant-velocity prediction in global coordinates.

    Returns (n_steps, 2).
    """
    x0, y0 = world[0], world[1]
    vx, vy = world[7], world[8]
    t = (np.arange(1, n_steps + 1) * dt)[:, None]
    return np.column_stack([x0 + vx * t[:, 0], y0 + vy * t[:, 0]])


# ── single-prediction subplot ─────────────────────────────────────────────────

_MDOT = dict(marker='o', markersize=2.5, markeredgewidth=0)   # shared dot style


def _draw_prediction(ax, preds_row, world, row_idx, obj_type, scene_id):
    """Draw one prediction panel onto *ax*."""
    global_xy = _pred_to_global(preds_row, world)   # (6, 60, 2)
    cv_xy     = _cv_baseline(world)                  # (60, 2)

    ox, oy = float(world[0]), float(world[1])

    ax.set_facecolor(_C_BG)
    ax.grid(True, color=_C_GRID, alpha=0.4, linewidth=0.5, zorder=0)

    # ── CV baseline ──────────────────────────────────────────────────────────
    dx, dy = cv_xy[:, 0] - ox, cv_xy[:, 1] - oy
    ax.plot(dx, dy, color=_C_CV, linewidth=1.0, linestyle='--', alpha=0.8,
            zorder=2, **_MDOT)

    # ── modes 1–5 (thin) ─────────────────────────────────────────────────────
    for m in range(1, 6):
        xy = global_xy[m]
        dx_m, dy_m = xy[:, 0] - ox, xy[:, 1] - oy
        ax.plot(dx_m, dy_m, color=_C_MODES, linewidth=0.7, alpha=0.6,
                zorder=3, **_MDOT)

    # ── mode 0 (bold) ────────────────────────────────────────────────────────
    xy0 = global_xy[0]
    dx0, dy0 = xy0[:, 0] - ox, xy0[:, 1] - oy
    ax.plot(dx0, dy0, color=_C_MODE0, linewidth=1.8, alpha=0.9,
            zorder=4, **_MDOT)

    # ── origin ───────────────────────────────────────────────────────────────
    ax.scatter([0], [0], s=50, color=_C_ORIGIN, zorder=7)

    # ── labels ────────────────────────────────────────────────────────────────
    type_str = _TYPE_NAMES.get(int(obj_type), str(obj_type))
    speed    = float(np.hypot(world[7], world[8]))
    ax.set_title(f'row {row_idx} · {type_str} · {scene_id}\n'
                 f'speed={speed:.1f} m/s  heading={np.degrees(float(world[6])):.0f}°',
                 fontsize=7, color='#111111', pad=3)
    ax.set_xlabel('Δx (m)', fontsize=7, color='#444444')
    ax.set_ylabel('Δy (m)', fontsize=7, color='#444444')
    ax.tick_params(colors='#444444', labelsize=6)
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
    ax.set_aspect('equal')


# ── velocity subplot ──────────────────────────────────────────────────────────

def _axes_style(ax, xlabel, ylabel):
    """Apply common axis style."""
    ax.set_facecolor(_C_BG)
    ax.grid(True, color=_C_GRID, alpha=0.4, linewidth=0.5, zorder=0)
    ax.set_xlabel(xlabel, fontsize=7, color='#444444')
    ax.set_ylabel(ylabel, fontsize=7, color='#444444')
    ax.tick_params(colors='#444444', labelsize=6)
    for spine in ax.spines.values():
        spine.set_color('#cccccc')


def _draw_velocity(ax, preds_row, world, row_idx, obj_type, scene_id):
    """Draw vx/vy vs step (1–59) for mode 0 onto *ax*."""
    global_xy = _pred_to_global(preds_row, world)
    xy0 = global_xy[0]
    vel   = (xy0[1:] - xy0[:-1]) / _UNITRAJ_DT   # (59, 2)
    steps = np.arange(1, 60)

    _axes_style(ax, 'step', 'velocity (m/s)')
    ax.axhline(0, color=_C_GRID, linewidth=0.8, zorder=1)
    ax.plot(steps, vel[:, 0], color=_C_VX, linewidth=1.2, alpha=0.9, zorder=2, **_MDOT)
    ax.plot(steps, vel[:, 1], color=_C_VY, linewidth=1.2, alpha=0.9, zorder=2, **_MDOT)

    type_str = _TYPE_NAMES.get(int(obj_type), str(obj_type))
    ax.set_title(f'row {row_idx} · {type_str} · {scene_id}',
                 fontsize=7, color='#111111', pad=3)


def _draw_position_vs_step(ax, preds_row, world, row_idx, obj_type, scene_id):
    """Draw x/y vs step (0–59) for mode 0 onto *ax* (agent-relative coords)."""
    global_xy = _pred_to_global(preds_row, world)
    xy0 = global_xy[0]
    ox, oy = float(world[0]), float(world[1])
    steps = np.arange(60)

    _axes_style(ax, 'step', 'position (m)')
    ax.axhline(0, color=_C_GRID, linewidth=0.8, zorder=1)
    ax.plot(steps, xy0[:, 0] - ox, color=_C_VX, linewidth=1.2, alpha=0.9, zorder=2, **_MDOT)
    ax.plot(steps, xy0[:, 1] - oy, color=_C_VY, linewidth=1.2, alpha=0.9, zorder=2, **_MDOT)

    type_str = _TYPE_NAMES.get(int(obj_type), str(obj_type))
    ax.set_title(f'row {row_idx} · {type_str} · {scene_id}',
                 fontsize=7, color='#111111', pad=3)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--npz', default='data/occlusions/fmae_nuscenes_v1trainval_3class_sw_inference.npz',
                        help='Path to the UniTraj inference NPZ file.')
    parser.add_argument('--instance', default=None,
                        help='Instance token to visualise (32-char hex). '
                             'If omitted, samples are chosen randomly.')
    parser.add_argument('--scene', default=None,
                        help='Restrict to rows from this scene name '
                             '(e.g. scene-0001).')
    parser.add_argument('--object-type', type=int, default=None,
                        choices=[1, 2, 3],
                        help='Filter by object type: 1=VEHICLE, 2=PED, 3=CYC.')
    parser.add_argument('--n-samples', type=int, default=12,
                        help='Number of random predictions to plot (default 12).')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible sampling.')
    parser.add_argument('--output', default='vis/unitraj_predictions/predictions.png',
                        help='Path to save the output image.')
    parser.add_argument('--show-last-step-hist', action='store_true',
                        help='Also save a histogram of step-59 displacement from '
                             'step-58 vs mean step displacement, to quantify the '
                             'artefact.')
    args = parser.parse_args()

    print(f'Loading {args.npz} …')
    data  = np.load(args.npz, allow_pickle=True)
    preds = data['predictions']               # (N, 6, 60, 2)
    meta  = data['metadata'].item()
    worlds     = meta['center_objects_world'] # (N, 10)
    scene_ids  = meta['scenario_ids']         # (N,)
    inst_ids   = meta['center_objects_id']    # (N,)
    obj_types  = meta['center_objects_type']  # (N,)
    N = len(preds)
    print(f'  {N:,} prediction rows.')

    # ── build candidate index ─────────────────────────────────────────────────
    mask = np.ones(N, dtype=bool)
    if args.instance is not None:
        mask &= (inst_ids == args.instance)
        if not mask.any():
            sys.exit(f'ERROR: instance token "{args.instance}" not found in NPZ.')
    if args.scene is not None:
        mask &= (scene_ids == args.scene)
        if not mask.any():
            sys.exit(f'ERROR: scene "{args.scene}" not found in NPZ.')
    if args.object_type is not None:
        mask &= (obj_types == args.object_type)
    candidates = np.where(mask)[0]
    print(f'  {len(candidates):,} candidates after filtering.')

    if len(candidates) == 0:
        sys.exit('No candidates to visualise.')

    rng = np.random.default_rng(args.seed)
    n   = min(args.n_samples, len(candidates))
    chosen = rng.choice(candidates, size=n, replace=False)
    chosen = np.sort(chosen)

    # ── optional: histogram of step-59 artefact magnitude ────────────────────
    if args.show_last_step_hist:
        # Compute per-row, mode-0: displacement from step 58 → 59 vs mean step disp
        d_last  = np.linalg.norm(preds[:, 0, 59] - preds[:, 0, 58], axis=-1)
        d_mean  = np.mean(np.linalg.norm(np.diff(preds[:, 0], axis=1), axis=-1), axis=-1)
        ratio   = d_last / (d_mean + 1e-6)
        fig_h, ax_h = plt.subplots(1, 2, figsize=(12, 4), facecolor=_C_BG)
        ax_h[0].set_facecolor(_C_BG)
        ax_h[1].set_facecolor(_C_BG)
        ax_h[0].hist(d_last, bins=100, color=_C_MODE0, alpha=0.8)
        ax_h[0].set_xlabel('Step-59 displacement (m)', fontsize=9)
        ax_h[0].set_ylabel('Count', fontsize=9)
        ax_h[0].set_title('Step 59 absolute displacement (mode 0)', fontsize=10)
        ax_h[1].hist(np.clip(ratio, 0, 20), bins=100, color=_C_TAIL, alpha=0.8)
        ax_h[1].set_xlabel('Step-59 disp / mean step disp', fontsize=9)
        ax_h[1].set_ylabel('Count', fontsize=9)
        ax_h[1].set_title('Step-59 artefact ratio (>1 = worse than average)', fontsize=10)
        for ax_h_i in ax_h:
            ax_h_i.grid(True, color=_C_GRID, alpha=0.4, linewidth=0.5)
            for spine in ax_h_i.spines.values():
                spine.set_color('#cccccc')
            ax_h_i.tick_params(colors='#444444')
        plt.tight_layout()
        hist_path = args.output.replace('.png', '_step59_hist.png')
        os.makedirs(os.path.dirname(hist_path) or '.', exist_ok=True)
        plt.savefig(hist_path, dpi=150, bbox_inches='tight',
                    facecolor=fig_h.get_facecolor())
        print(f'Saved step-59 histogram → {hist_path}')
        plt.close(fig_h)

    # ── prediction grid ───────────────────────────────────────────────────────
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4.5 * nrows),
                             facecolor=_C_BG)
    axes = np.array(axes).flatten()

    for k, row_idx in enumerate(chosen):
        ax = axes[k]
        _draw_prediction(ax, preds[row_idx], worlds[row_idx],
                         row_idx, obj_types[row_idx], scene_ids[row_idx])

    # Hide unused panels
    for k in range(n, len(axes)):
        axes[k].set_visible(False)

    # ── shared legend ─────────────────────────────────────────────────────────
    _dot = dict(marker='o', markersize=4, linestyle='-')
    legend_handles = [
        mlines.Line2D([], [], color=_C_MODE0, linewidth=1.8, label='Mode 0', **_dot),
        mlines.Line2D([], [], color=_C_MODES, linewidth=0.7, label='Modes 1–5', **_dot),
        mlines.Line2D([], [], color=_C_CV,    linewidth=1.0, linestyle='--',
                      marker='o', markersize=4, label='CV baseline'),
        mlines.Line2D([], [], color='none', marker='o', markersize=5,
                      markerfacecolor=_C_ORIGIN, label='Origin'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles),
               fontsize=8, labelcolor='#111111', facecolor='white',
               edgecolor='#cccccc', framealpha=0.9,
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        f'UniTraj ML predictions  —  {n} samples  (dot per step)\n'
        f'NPZ: {args.npz}',
        fontsize=9, color='#111111', y=1.01)

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'Saved → {args.output}')
    plt.close(fig)

    # ── velocity figure ───────────────────────────────────────────────────────
    fig_v, axes_v = plt.subplots(nrows, ncols,
                                 figsize=(4.5 * ncols, 3.0 * nrows),
                                 facecolor=_C_BG)
    axes_v = np.array(axes_v).flatten()

    for k, row_idx in enumerate(chosen):
        _draw_velocity(axes_v[k], preds[row_idx], worlds[row_idx],
                       row_idx, obj_types[row_idx], scene_ids[row_idx])

    for k in range(n, len(axes_v)):
        axes_v[k].set_visible(False)

    _vdot = dict(marker='o', markersize=4)
    vel_legend = [
        mlines.Line2D([], [], color=_C_VX, linewidth=1.2, label='vx (mode 0)', **_vdot),
        mlines.Line2D([], [], color=_C_VY, linewidth=1.2, label='vy (mode 0)', **_vdot),
    ]
    fig_v.legend(handles=vel_legend, loc='lower center', ncol=2,
                 fontsize=8, labelcolor='#111111', facecolor='white',
                 edgecolor='#cccccc', framealpha=0.9,
                 bbox_to_anchor=(0.5, 0.0))

    fig_v.suptitle(
        f'UniTraj predicted velocity (mode 0)  —  {n} samples  (dot per step)\n'
        f'NPZ: {args.npz}',
        fontsize=9, color='#111111', y=1.01)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    vel_path = args.output.replace('.png', '_velocity.png')
    plt.savefig(vel_path, dpi=200, bbox_inches='tight',
                facecolor=fig_v.get_facecolor())
    print(f'Saved → {vel_path}')
    plt.close(fig_v)

    # ── position-vs-step figure ───────────────────────────────────────────────
    fig_p, axes_p = plt.subplots(nrows, ncols,
                                 figsize=(4.5 * ncols, 3.0 * nrows),
                                 facecolor=_C_BG)
    axes_p = np.array(axes_p).flatten()

    for k, row_idx in enumerate(chosen):
        _draw_position_vs_step(axes_p[k], preds[row_idx], worlds[row_idx],
                               row_idx, obj_types[row_idx], scene_ids[row_idx])

    for k in range(n, len(axes_p)):
        axes_p[k].set_visible(False)

    _pdot = dict(marker='o', markersize=4)
    pos_legend = [
        mlines.Line2D([], [], color=_C_VX, linewidth=1.2, label='x (mode 0)', **_pdot),
        mlines.Line2D([], [], color=_C_VY, linewidth=1.2, label='y (mode 0)', **_pdot),
    ]
    fig_p.legend(handles=pos_legend, loc='lower center', ncol=2,
                 fontsize=8, labelcolor='#111111', facecolor='white',
                 edgecolor='#cccccc', framealpha=0.9,
                 bbox_to_anchor=(0.5, 0.0))

    fig_p.suptitle(
        f'UniTraj predicted position (mode 0, agent-relative)  —  {n} samples  (dot per step)\n'
        f'NPZ: {args.npz}',
        fontsize=9, color='#111111', y=1.01)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    pos_path = args.output.replace('.png', '_position_vs_step.png')
    plt.savefig(pos_path, dpi=200, bbox_inches='tight',
                facecolor=fig_p.get_facecolor())
    print(f'Saved → {pos_path}')
    plt.close(fig_p)


if __name__ == '__main__':
    main()
