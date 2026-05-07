#!/usr/bin/env python3
"""Check for 2-D OBB collisions between the ego vehicle and GT agent boxes.

For every frame in the dataset the script:
  1. Places the ego bounding box at its recorded global pose.
  2. Transforms each agent gt_box into the global frame.
  3. Tests OBB overlap using the Separating Axis Theorem (SAT).

Outputs a per-scene collision report and a dataset-level summary.

Usage
-----
    # Observed annotations only
    python tools/check_ego_collisions.py \\
        --pkl data/infos/nuscenes_infos_val.pkl

    # Include interpolated / extrapolated boxes (requires _occ pkl)
    python tools/check_ego_collisions.py \\
        --pkl data/infos/nuscenes_infos_val_occ.pkl --include-synthetic

    # Restrict to vehicle classes only
    python tools/check_ego_collisions.py \\
        --pkl data/infos/nuscenes_infos_val.pkl \\
        --classes car truck bus trailer

    # Save per-collision details to JSON
    python tools/check_ego_collisions.py \\
        --pkl data/infos/nuscenes_infos_val.pkl --save-json collisions.json
"""

import argparse
import json
import os
import pickle
import numpy as np
from collections import defaultdict

# nuScenes sensor rig (Renault ZOE) — override via --ego-length / --ego-width
EGO_LENGTH = 4.08
EGO_WIDTH  = 1.73


# ===========================================================================
# Geometry
# ===========================================================================

def _obb_corners(cx, cy, length, width, yaw):
    """Return (4, 2) corner array for an oriented bounding box."""
    c, s = np.cos(yaw), np.sin(yaw)
    R    = np.array([[c, -s], [s, c]])
    half = np.array([[ length / 2,  width / 2],
                     [-length / 2,  width / 2],
                     [-length / 2, -width / 2],
                     [ length / 2, -width / 2]])
    return half @ R.T + np.array([cx, cy])


def _sat_overlaps(a, b):
    """Separating Axis Theorem for two convex polygons ((N,2) corner arrays).

    Returns True when the polygons overlap (collision).
    """
    def _normals(poly):
        # Edge normals for each side of the polygon
        edges = np.roll(poly, -1, axis=0) - poly
        return np.stack([-edges[:, 1], edges[:, 0]], axis=1)

    for ax in np.vstack([_normals(a), _normals(b)]):
        pa = a @ ax
        pb = b @ ax
        if pa.max() < pb.min() or pb.max() < pa.min():
            return False   # separating axis found
    return True


# ===========================================================================
# nuScenes pose helpers
# ===========================================================================

def _ego_global(info):
    """Return (pos_xy (2,), yaw) of the ego vehicle in global frame."""
    from pyquaternion import Quaternion
    q   = Quaternion(info['ego2global_rotation'])
    pos = np.array(info['ego2global_translation'])[:2]
    yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y),
                     1 - 2 * (q.y ** 2 + q.z ** 2))
    return pos, yaw


def _l2g_RT(info):
    """Return (R 3×3, t 3) for the lidar → global transform."""
    from pyquaternion import Quaternion
    R_l2e = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    t_l2e = np.array(info['lidar2ego_translation'])
    R_e2g = Quaternion(info['ego2global_rotation']).rotation_matrix
    t_e2g = np.array(info['ego2global_translation'])
    R = R_e2g @ R_l2e
    t = R_e2g @ t_l2e + t_e2g
    return R, t


# ===========================================================================
# Collision check
# ===========================================================================

def check_collisions(infos, include_synthetic=False, classes=None,
                     ego_length=EGO_LENGTH, ego_width=EGO_WIDTH):
    """Iterate over all frames and return a list of collision event dicts.

    Each dict contains:
      scene_token, timestamp, frame_idx, inst_ind, class_name,
      is_interp, is_extrap, prev_occluded, dist_centers, ego_xy, agent_xy

    prev_occluded (bool) — True for observed (non-synthetic) agents that
    had at least one appearance with valid_flag=False *or* is_interpolated=True
    at any frame strictly before the collision frame in the same scene.
    For synthetic agents the field is always False.
    """
    has_synthetic = 'is_interpolated' in infos[0]

    # Group and sort samples by scene
    scene_map = defaultdict(list)
    for gi, info in enumerate(infos):
        scene_map[info['scene_token']].append(gi)
    for sc in scene_map:
        scene_map[sc].sort(key=lambda i: infos[i]['timestamp'])

    # Maximum possible centre-to-centre distance for any collision
    ego_half_diag = np.hypot(ego_length, ego_width) / 2

    collisions = []
    for sc_tok, gidxs in scene_map.items():
        # Pre-compute ego speed at each frame from consecutive global poses.
        # speed[i] = distance(pos[i], pos[i-1]) / dt.  Frame 0 uses frame 1's speed.
        ego_speeds = np.zeros(len(gidxs), dtype=np.float64)
        for fi in range(1, len(gidxs)):
            pos_cur, _ = _ego_global(infos[gidxs[fi]])
            pos_prv, _ = _ego_global(infos[gidxs[fi - 1]])
            dt = max((infos[gidxs[fi]]['timestamp'] -
                      infos[gidxs[fi - 1]]['timestamp']) * 1e-6, 1e-6)
            ego_speeds[fi] = np.hypot(*(pos_cur - pos_prv)) / dt
        ego_speeds[0] = ego_speeds[1] if len(gidxs) > 1 else 0.0

        # Per-instance: first frame_idx when the instance was occluded (invalid
        # or interpolated), updated AFTER collision detection so "previous"
        # means strictly before the collision frame.
        inst_first_occ_frame = {}  # inst_ind (int) -> frame_idx (int)

        for frame_idx, gi in enumerate(gidxs):
            info    = infos[gi]
            boxes   = info['gt_boxes']      # (N, 7) in lidar frame
            names   = info['gt_names']
            n_boxes = len(boxes)

            if n_boxes == 0:
                continue

            is_interp = (info['is_interpolated']
                         if has_synthetic else np.zeros(n_boxes, bool))
            is_extrap = (info['is_extrapolated']
                         if has_synthetic else np.zeros(n_boxes, bool))
            is_synth  = is_interp | is_extrap
            valid     = info['valid_flag']

            # Ego pose and corners in global frame
            ego_pos, ego_yaw = _ego_global(info)
            ego_corners = _obb_corners(
                ego_pos[0], ego_pos[1], ego_length, ego_width, ego_yaw)

            # Transform all agent boxes to global frame in one pass
            R, t    = _l2g_RT(info)
            xy_g    = (boxes[:, :3] @ R.T + t)[:, :2]
            yaw_g   = boxes[:, 6] + np.arctan2(R[1, 0], R[0, 0])

            # --- collision detection (uses occlusion history up to prev frame) ---
            for bi in range(n_boxes):
                if not include_synthetic and is_synth[bi]:
                    continue
                if classes is not None and names[bi] not in classes:
                    continue

                # Quick distance pre-filter before full SAT
                dist = float(np.hypot(xy_g[bi, 0] - ego_pos[0],
                                      xy_g[bi, 1] - ego_pos[1]))
                agent_half_diag = np.hypot(boxes[bi, 3], boxes[bi, 4]) / 2
                if dist > ego_half_diag + agent_half_diag + 0.5:
                    continue

                agent_corners = _obb_corners(
                    float(xy_g[bi, 0]), float(xy_g[bi, 1]),
                    float(boxes[bi, 3]), float(boxes[bi, 4]),
                    float(yaw_g[bi]))

                if _sat_overlaps(ego_corners, agent_corners):
                    inst_ind = int(info['instance_inds'][bi])
                    vel = info['gt_velocity'][bi]
                    speed = float(np.hypot(vel[0], vel[1])) \
                            if not (np.isnan(vel[0]) or np.isnan(vel[1])) else 0.0
                    occ_start = inst_first_occ_frame.get(inst_ind)
                    collisions.append(dict(
                        scene_token         = sc_tok,
                        timestamp           = int(info['timestamp']),
                        frame_idx           = frame_idx,
                        inst_ind            = inst_ind,
                        class_name          = str(names[bi]),
                        is_interp           = bool(is_interp[bi]),
                        is_extrap           = bool(is_extrap[bi]),
                        prev_occluded       = occ_start is not None,
                        occ_start_frame_idx = occ_start,  # None if not prev occluded
                        agent_speed         = speed,
                        ego_speed           = float(ego_speeds[frame_idx]),
                        dist_centers        = dist,
                        ego_xy              = ego_pos.tolist(),
                        agent_xy            = xy_g[bi].tolist(),
                    ))

            # --- update occlusion history for future frames ---
            for bi in range(n_boxes):
                inst_ind = int(info['instance_inds'][bi])
                # Record FIRST frame when occluded (invalid or interpolated gap)
                if not valid[bi] or (has_synthetic and is_interp[bi]):
                    if inst_ind not in inst_first_occ_frame:
                        inst_first_occ_frame[inst_ind] = frame_idx

    return collisions


# ===========================================================================
# Reporting
# ===========================================================================

def print_report(collisions):
    if not collisions:
        print('\nNo collisions found.')
        return

    by_scene = defaultdict(list)
    for c in collisions:
        by_scene[c['scene_token']].append(c)

    # Per-scene table
    print(f'\n{"Scene token":>12}  {"Frames":>6}  {"Colls":>5}  '
          f'{"Obs(new)":>8}  {"Obs(occ)":>8}  {"Synth":>5}  Classes')
    print('─' * 85)
    for sc_tok, evts in sorted(by_scene.items(), key=lambda x: -len(x[1])):
        obs_evts   = [e for e in evts if not e['is_interp'] and not e['is_extrap']]
        n_frames   = len({e['frame_idx'] for e in evts})
        n_obs_new  = sum(1 for e in obs_evts if not e['prev_occluded'])
        n_obs_occ  = sum(1 for e in obs_evts if     e['prev_occluded'])
        n_synth    = sum(1 for e in evts if e['is_interp'] or e['is_extrap'])
        n_interp   = sum(1 for e in evts if e['is_interp'])
        n_extrap   = sum(1 for e in evts if e['is_extrap'])
        cls_str    = ', '.join(sorted({e['class_name'] for e in evts}))
        synth_str  = ''
        if n_synth:
            parts = []
            if n_interp: parts.append(f'{n_interp}i')
            if n_extrap: parts.append(f'{n_extrap}e')
            synth_str = f'{n_synth}({",".join(parts)})'
        print(f'  {sc_tok[:10]}  {n_frames:>6}  {len(evts):>5}  '
              f'{n_obs_new:>8}  {n_obs_occ:>8}  {synth_str:>5}  {cls_str}')

    # Summary
    obs      = [c for c in collisions if not c['is_interp'] and not c['is_extrap']]
    n_interp = sum(1 for c in collisions if c['is_interp'])
    n_extrap = sum(1 for c in collisions if c['is_extrap'])
    n_obs_prev_occ  = sum(1 for c in obs if     c['prev_occluded'])
    n_obs_clean     = sum(1 for c in obs if not c['prev_occluded'])
    print()
    print(f'Total collision events          : {len(collisions)}')
    print(f'  Observed agents               : {len(obs)}')
    print(f'    previously occluded          : {n_obs_prev_occ}'
          f'  (had valid_flag=False or gap-interpolated earlier in track)')
    print(f'    not previously occluded      : {n_obs_clean}')
    print(f'  Interpolated agents (synth)   : {n_interp}')
    print(f'  Extrapolated agents (synth)   : {n_extrap}')
    print(f'  Unique scenes                 : {len(by_scene)}')
    print(f'  Unique instances              : {len({c["inst_ind"] for c in collisions})}')

    # Class breakdown
    by_class = defaultdict(int)
    for c in collisions:
        by_class[c['class_name']] += 1
    print('\nBy class:')
    for cls, cnt in sorted(by_class.items(), key=lambda x: -x[1]):
        print(f'  {cls:<20} {cnt}')


def print_occluded_stats(occ_collisions, threshold):
    """Print a summary of the occluded + non-stationary filtered collisions."""
    if not occ_collisions:
        print(f'\nNo occluded + non-stationary collisions '
              f'(speed ≥ {threshold} m/s) found.')
        return

    by_scene = defaultdict(list)
    for c in occ_collisions:
        by_scene[c['scene_token']].append(c)

    ego_speeds   = [c['ego_speed']   for c in occ_collisions]
    agent_speeds = [c['agent_speed'] for c in occ_collisions]

    n_prev  = sum(1 for c in occ_collisions if c['prev_occluded']
                  and not c['is_interp'] and not c['is_extrap'])
    n_interp = sum(1 for c in occ_collisions if c['is_interp'])
    n_extrap = sum(1 for c in occ_collisions if c['is_extrap'])

    print(f'\n── Occluded + non-stationary collisions (speed ≥ {threshold} m/s) ──')
    print(f'  Total events          : {len(occ_collisions)}')
    print(f'  Unique scenes         : {len(by_scene)}')
    print(f'  Unique instances      : {len({c["inst_ind"] for c in occ_collisions})}')
    print(f'  Prev-occluded obs     : {n_prev}')
    print(f'  Interpolated (synth)  : {n_interp}')
    print(f'  Extrapolated (synth)  : {n_extrap}')
    print(f'  Ego speed   — '
          f'mean {np.mean(ego_speeds):.2f}  '
          f'min {np.min(ego_speeds):.2f}  '
          f'max {np.max(ego_speeds):.2f} m/s')
    print(f'  Agent speed — '
          f'mean {np.mean(agent_speeds):.2f}  '
          f'min {np.min(agent_speeds):.2f}  '
          f'max {np.max(agent_speeds):.2f} m/s')

    by_class = defaultdict(int)
    for c in occ_collisions:
        by_class[c['class_name']] += 1
    print('  By class:')
    for cls, cnt in sorted(by_class.items(), key=lambda x: -x[1]):
        print(f'    {cls:<20} {cnt}')


# ===========================================================================
# Visualisation
# ===========================================================================

_CV_OBS_CLEAN = '#0288d1'   # observed, visible
_CV_OCCLUDED  = '#7b1fa2'   # any occluded agent (invalid flag or synthetic)
_CV_EGO       = '#00695c'   # ego vehicle
_CV_COLL      = '#ff1744'   # collision highlight
_CV_DRIVABLE  = '#b0bec5'   # drivable area (legend swatch only)
_CV_BG        = 'white'


def _draw_box_viz(ax, cx, cy, length, width, yaw, color,
                  alpha_face=0.20, alpha_edge=0.85, lw=0.8, zorder=3):
    import matplotlib.pyplot as plt
    c, s    = np.cos(yaw), np.sin(yaw)
    R2      = np.array([[c, -s], [s, c]])
    half    = np.array([[ length/2,  width/2],
                        [-length/2,  width/2],
                        [-length/2, -width/2],
                        [ length/2, -width/2]])
    corners = half @ R2.T + np.array([cx, cy])
    ax.add_patch(plt.Polygon(corners, closed=True, facecolor=color,
                             edgecolor=color, alpha=alpha_face,
                             linewidth=lw, zorder=zorder))
    ax.add_patch(plt.Polygon(corners, closed=True, facecolor='none',
                             edgecolor=color, alpha=alpha_edge,
                             linewidth=lw, zorder=zorder + 1))


def _visualize_collision_scene(infos, gidxs, coll_info, ax,
                                title='', ego_length=EGO_LENGTH,
                                ego_width=EGO_WIDTH, nusc=None,
                                only_collided=False,
                                show_coll_circles=True,
                                ego_noise_towards_agent=0.0):
    """Draw a BEV scene with agents coloured by type and collisions highlighted.

    coll_info             : dict mapping (frame_idx, inst_ind) ->
                              {'agent_speed': float, 'ego_speed': float}
    nusc                  : NuScenes instance for drivable-area map overlay (optional)
    only_collided         : if True, only draw the agents that appear in coll_info
    show_coll_circles     : if False, suppress the red collision circle annotation
    ego_noise_towards_agent: if > 0, nudge the rendered ego box towards the colliding
                              agent at each collision frame by a random displacement
                              with magnitude scale * ego_trajectory_length (metres).
    Everything is rendered in the first-ego-frame coordinate system.
    """
    from pyquaternion import Quaternion
    import matplotlib.patches as _patches

    # Reference frame: first ego pose
    q0  = Quaternion(infos[gidxs[0]]['ego2global_rotation'])
    p0  = np.array(infos[gidxs[0]]['ego2global_translation'])[:2]
    yaw0 = np.arctan2(2*(q0.w*q0.z + q0.x*q0.y),
                      1 - 2*(q0.y**2 + q0.z**2))
    c0, s0  = np.cos(yaw0), np.sin(yaw0)
    R_inv   = np.array([[c0, s0], [-s0, c0]])

    has_synthetic = 'is_interpolated' in infos[gidxs[0]]
    collided_inst_ids = {inst_ind for (_, inst_ind) in coll_info}

    def to_ego(xy):
        return (np.asarray(xy) - p0) @ R_inv.T

    # --- Drivable-area map (rendered first, behind everything) ---------------
    if nusc is not None:
        try:
            first_token  = infos[gidxs[0]]['token']
            lidar_token  = nusc.get('sample', first_token)['data']['LIDAR_TOP']
            nusc.explorer.render_ego_centric_map(
                sample_data_token=lidar_token, axes_limit=200, ax=ax)
        except Exception:
            pass  # map unavailable — continue without it

    ego_pos, ego_yaws = [], []
    # (frame_idx, x_e, y_e, l, w, yaw_e, color, speeds)
    agent_entries = []

    for frame_idx, gi in enumerate(gidxs):
        info  = infos[gi]
        boxes = info['gt_boxes']
        n     = len(boxes)
        if n > 0:
            R, t  = _l2g_RT(info)
            xy_g  = (boxes[:, :3] @ R.T + t)[:, :2]
            yaw_g = boxes[:, 6] + np.arctan2(R[1, 0], R[0, 0])
            xy_e  = to_ego(xy_g)
            yaw_e = yaw_g - yaw0

            is_i  = (info['is_interpolated']
                     if has_synthetic else np.zeros(n, bool))
            is_e  = (info['is_extrapolated']
                     if has_synthetic else np.zeros(n, bool))
            valid = info['valid_flag']

            for bi in range(n):
                inst_ind = info['instance_inds'][bi]
                if only_collided and inst_ind not in collided_inst_ids:
                    continue
                coll_key = (frame_idx, inst_ind)
                speeds   = coll_info.get(coll_key)
                if is_i[bi] or is_e[bi] or not valid[bi]:
                    color = _CV_OCCLUDED
                else:
                    color = _CV_OBS_CLEAN
                agent_entries.append((
                    frame_idx,
                    float(xy_e[bi, 0]), float(xy_e[bi, 1]),
                    float(boxes[bi, 3]), float(boxes[bi, 4]),
                    float(yaw_e[bi]), color, speeds,
                ))

        q   = Quaternion(info['ego2global_rotation'])
        pos = np.array(info['ego2global_translation'])[:2]
        yaw = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))
        ego_pos.append(to_ego(pos))
        ego_yaws.append(yaw - yaw0)

    ego_pos = np.array(ego_pos)

    # Total ego path length — used to scale optional position noise
    traj_length = (float(np.sum(np.hypot(np.diff(ego_pos[:, 0]),
                                         np.diff(ego_pos[:, 1]))))
                   if len(ego_pos) > 1 else 1.0)

    rng = np.random.default_rng(42)

    # Build frame_idx -> mean position of colliding agents (for ego noise direction)
    coll_agent_pos_by_frame: dict = {}
    for e in agent_entries:
        if e[7] is not None:  # collision frame
            fi = e[0]
            coll_agent_pos_by_frame.setdefault(fi, []).append((e[1], e[2]))
    coll_agent_mean_by_frame = {
        fi: np.mean(positions, axis=0)
        for fi, positions in coll_agent_pos_by_frame.items()
    }

    # Draw agents (non-collision first, collision on top)
    for target_is_coll in [False, True]:
        for e in agent_entries:
            speeds = e[7]
            if (speeds is not None) != target_is_coll:
                continue
            _draw_box_viz(ax, e[1], e[2], e[3], e[4], e[5], color=e[6],
                          alpha_face=0.18, alpha_edge=0.75, lw=0.7, zorder=3)
            if speeds is not None and show_coll_circles:
                r = np.hypot(e[3], e[4]) / 2 + 0.6
                ax.add_patch(_patches.Circle(
                    (e[1], e[2]), r, fill=False,
                    edgecolor=_CV_COLL, linewidth=1.5, alpha=0.9, zorder=6))
                ax.text(e[1], e[2] + r + 0.4,
                        f"e:{speeds['ego_speed']:.1f}  a:{speeds['agent_speed']:.1f} m/s",
                        color=_CV_COLL, fontsize=5.5, ha='center', va='bottom',
                        zorder=7, alpha=0.95)

    # Ego trajectory and boxes
    # Build the (optionally noisy) ego positions used for both trajectory and boxes
    ego_pos_draw = ego_pos.copy().astype(float)
    if ego_noise_towards_agent > 0.0:
        for fi, agent_mean in coll_agent_mean_by_frame.items():
            ex, ey = ego_pos_draw[fi]
            dx, dy = float(agent_mean[0]) - ex, float(agent_mean[1]) - ey
            dist = max(float(np.hypot(dx, dy)), 1e-6)
            noise_mag = ego_noise_towards_agent * traj_length * rng.uniform(0.0, 1.0)
            ego_pos_draw[fi, 0] += (dx / dist) * noise_mag
            ego_pos_draw[fi, 1] += (dy / dist) * noise_mag

    ax.plot(ego_pos_draw[:, 0], ego_pos_draw[:, 1],
            color='#333333', linewidth=1.5, linestyle='--', zorder=7, alpha=0.8)
    ax.scatter(*ego_pos_draw[0],  color='#333333', s=30, zorder=9, marker='o')
    ax.scatter(*ego_pos_draw[-1], color='#333333', s=30, zorder=9, marker='x')
    for pos, yaw in zip(ego_pos_draw, ego_yaws):
        _draw_box_viz(ax, pos[0], pos[1], ego_length, ego_width, yaw,
                      color=_CV_EGO, alpha_face=0.30, alpha_edge=0.90,
                      lw=1.0, zorder=8)

    coll_entries = [e for e in agent_entries if e[7] is not None]
    n_coll = len(coll_entries)
    if coll_entries:
        avg_ego   = np.mean([e[7]['ego_speed']   for e in coll_entries])
        avg_agent = np.mean([e[7]['agent_speed']  for e in coll_entries])
        speed_str = f' | avg e:{avg_ego:.1f}  a:{avg_agent:.1f} m/s'
    else:
        speed_str = ''

    # Zoom to ego trajectory + padding
    pad = 30.0
    xlim = (ego_pos[:, 0].min() - pad, ego_pos[:, 0].max() + pad)
    ylim = (ego_pos[:, 1].min() - pad, ego_pos[:, 1].max() + pad)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_aspect('equal')
    if nusc is None:
        ax.set_facecolor(_CV_BG)
    ax.grid(True, color='#cccccc', alpha=0.4, linewidth=0.5)
    ax.tick_params(colors='#444444', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
    ax.set_xlabel('X (m)', color='#333333', fontsize=8)
    ax.set_ylabel('Y (m)', color='#333333', fontsize=8)
    ax.set_title(f'{title}\n{len(gidxs)} frames | {n_coll} collisions{speed_str}',
                 color='#111111', fontsize=8, pad=4)

    # Per-subplot legend
    import matplotlib.lines as _mlines
    legend_handles = [
        _patches.Patch(color=_CV_OBS_CLEAN, label='Future'),
        _patches.Patch(color=_CV_OCCLUDED,  label='Occluded'),
        _patches.Patch(color=_CV_EGO,       label='Ego Plan'),
        _patches.Patch(color=_CV_DRIVABLE,  label='Drivable area'),
        _mlines.Line2D([], [], color=_CV_COLL, linewidth=1.5, label='Collision'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=6,
              framealpha=0.85, facecolor='white', edgecolor='#cccccc',
              labelcolor='#111111')


def render_collision_cameras(infos, occ_collisions, output_dir,
                              dataroot='data/nuscenes',
                              nusc_version='v1.0-trainval'):
    """Render nuScenes camera views for each unique occluded-collision frame.

    For every unique (scene_token, frame_idx) pair in *occ_collisions* the
    function calls ``nusc.render_sample`` and saves the composite camera image
    to ``output_dir/cameras/collision_<scene8>_<frame03d>.png``.

    Requires the ``nuscenes-devkit`` package and access to the raw nuScenes
    data at *dataroot*.
    """
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError:
        print('WARNING: nuscenes-devkit not available — skipping camera renders.')
        return

    if not occ_collisions:
        return

    cam_dir = os.path.join(output_dir, 'cameras')
    os.makedirs(cam_dir, exist_ok=True)

    print(f'\nInitialising NuScenes ({nusc_version}) from {dataroot} …')
    nusc = NuScenes(version=nusc_version, dataroot=dataroot, verbose=False)

    # Build (scene_token, frame_idx) → sample_token lookup.
    # frame_idx is the 0-based index within the scene sorted by timestamp,
    # matching the convention used in check_collisions().
    scene_map = defaultdict(list)
    for gi, info in enumerate(infos):
        scene_map[info['scene_token']].append(gi)
    for sc in scene_map:
        scene_map[sc].sort(key=lambda i: infos[i]['timestamp'])

    frame_to_token = {}   # (scene_token, frame_idx) -> sample_token
    for sc_tok, gidxs in scene_map.items():
        for fi, gi in enumerate(gidxs):
            token = infos[gi].get('token')
            if token is not None:
                frame_to_token[(sc_tok, fi)] = token

    # Deduplicate: one render per (scene_token, frame_idx)
    unique_frames = {(c['scene_token'], c['frame_idx']) for c in occ_collisions}

    rendered, skipped = 0, 0
    for sc_tok, frame_idx in sorted(unique_frames):
        sample_token = frame_to_token.get((sc_tok, frame_idx))
        if sample_token is None:
            skipped += 1
            continue
        out_path = os.path.join(
            cam_dir,
            f'collision_{sc_tok[:8]}_{frame_idx:03d}.png',
        )
        try:
            nusc.render_sample(sample_token, out_path=out_path, verbose=False)
            rendered += 1
        except Exception as exc:
            print(f'  WARNING: render failed for {sample_token}: {exc}')
            skipped += 1

    print(f'Camera renders saved → {cam_dir}  '
          f'({rendered} rendered, {skipped} skipped)')


def plot_collision_scenes(infos, collisions, num_scenes=6,
                          output_dir='vis/collisions',
                          ego_length=EGO_LENGTH, ego_width=EGO_WIDTH,
                          use_occ_window=False,
                          nusc=None, only_collided=False,
                          filename='collision_scenes.png',
                          show_coll_circles=True,
                          ego_noise_towards_agent=0.0):
    """Plot BEV collision scenes.

    use_occ_window          : clip each scene to [last occ_start, first collision] frames.
    nusc                    : NuScenes instance for drivable-area overlay (optional).
    only_collided           : only draw the actors involved in collisions (not all agents).
    filename                : output file name inside output_dir.
    show_coll_circles       : draw red collision-highlight circles (default True).
    ego_noise_towards_agent : if > 0, nudge the ego towards the colliding agent at each
                               collision frame by scale * traj_length metres (default 0).
    """
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Build scene map
    scene_map = defaultdict(list)
    for gi, info in enumerate(infos):
        scene_map[info['scene_token']].append(gi)
    for sc in scene_map:
        scene_map[sc].sort(key=lambda i: infos[i]['timestamp'])

    # Rank scenes by collision count
    by_scene = defaultdict(list)
    for c in collisions:
        by_scene[c['scene_token']].append(c)
    top_scenes = sorted(by_scene.items(), key=lambda x: -len(x[1]))[:num_scenes]

    if not top_scenes:
        print('No scenes to plot.')
        return

    ncols = min(3, len(top_scenes))
    nrows = (len(top_scenes) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 7 * nrows),
                             facecolor=_CV_BG)
    axes = np.array(axes).flatten()

    for i, (sc_tok, evts) in enumerate(top_scenes):
        all_gidxs = scene_map[sc_tok]

        if use_occ_window:
            # Window: last occlusion start → first collision frame
            occ_starts = [e['occ_start_frame_idx'] for e in evts
                          if e['occ_start_frame_idx'] is not None]
            coll_frames = [e['frame_idx'] for e in evts]
            f_start = max(occ_starts) if occ_starts else max(0, min(coll_frames) - 1)
            f_end   = min(coll_frames)
            f_start = max(0, min(f_start, f_end))  # guard against degenerate cases
            gidxs = all_gidxs[f_start: f_end + 1]
            # Remap frame_idx keys to local indices within the sliced window
            coll_info = {
                (e['frame_idx'] - f_start, e['inst_ind']): {
                    'agent_speed': e['agent_speed'],
                    'ego_speed':   e['ego_speed'],
                }
                for e in evts
            }
            occ_duration = f_end - f_start
            title = (f'Scene {sc_tok[:8]}… | occ→coll: {occ_duration} frames')
        else:
            gidxs = all_gidxs
            coll_info = {
                (e['frame_idx'], e['inst_ind']): {
                    'agent_speed': e['agent_speed'],
                    'ego_speed':   e['ego_speed'],
                }
                for e in evts
            }
            title = f'Scene {sc_tok[:8]}… ({len(evts)} collisions)'

        _visualize_collision_scene(
            infos, gidxs, coll_info, axes[i],
            title=title,
            ego_length=ego_length, ego_width=ego_width,
            nusc=nusc, only_collided=only_collided,
            show_coll_circles=show_coll_circles,
            ego_noise_towards_agent=ego_noise_towards_agent)

    for j in range(len(top_scenes), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Top {len(top_scenes)} scenes by collision count',
                 color='#111111', fontsize=11, y=1.01)
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    out = os.path.join(output_dir, filename)
    plt.savefig(out, dpi=130, bbox_inches='tight', facecolor='white')
    print(f'\nPlot saved → {out}')
    plt.close(fig)


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pkl', default='data/infos/nuscenes_infos_train.pkl',
                    help='Path to nuScenes info pkl file')
    ap.add_argument('--include-synthetic', action='store_true',
                    help='Also check interpolated / extrapolated boxes '
                         '(requires an _occ pkl produced by nuscenes_occlusion_converter.py)')
    ap.add_argument('--classes', nargs='+', default=None,
                    metavar='CLS',
                    help='Agent class names to check (default: all). '
                         'E.g. --classes car truck bus trailer')
    ap.add_argument('--ego-length', type=float, default=EGO_LENGTH,
                    help=f'Ego vehicle length in metres (default: {EGO_LENGTH})')
    ap.add_argument('--ego-width', type=float, default=EGO_WIDTH,
                    help=f'Ego vehicle width in metres (default: {EGO_WIDTH})')
    ap.add_argument('--save-json', default=None, metavar='PATH',
                    help='Save per-collision details to a JSON file')
    ap.add_argument('--plot-dir', default='vis/collisions', metavar='DIR',
                    help='Save BEV collision plots to this directory '
                         '(omit to skip plotting)')
    ap.add_argument('--plot-occluded-dir', default='vis/collisions/occ', metavar='DIR',
                    help='Save a second BEV plot grid filtered to collisions '
                         'with previously/currently occluded non-stationary agents')
    ap.add_argument('--num-plot-scenes', type=int, default=6,
                    help='Number of top-collision scenes to plot (default: 6)')
    ap.add_argument('--nonstationary-threshold', type=float, default=1.0,
                    metavar='M/S',
                    help='Minimum agent speed (m/s) to be considered '
                         'non-stationary (default: 0.5)')
    ap.add_argument('--dataroot', default='data/nuscenes', metavar='DIR',
                    help='nuScenes dataset root directory '
                         '(default: data/nuscenes).  Used only when '
                         '--plot-occluded-dir is set.')
    ap.add_argument('--nusc-version', default='v1.0-trainval', metavar='VER',
                    help='nuScenes version string (default: v1.0-trainval).')
    args = ap.parse_args()

    print(f'Loading {args.pkl} ...')
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos']
    n_scenes = len({i['scene_token'] for i in infos})
    print(f'  {len(infos)} samples across {n_scenes} scenes')
    print(f'  Ego box: {args.ego_length:.2f} m × {args.ego_width:.2f} m')
    if args.classes:
        print(f'  Filtering classes: {args.classes}')
    if args.include_synthetic:
        has_flags = 'is_interpolated' in infos[0]
        if not has_flags:
            print('  WARNING: --include-synthetic set but pkl has no '
                  'is_interpolated/is_extrapolated flags — ignoring flag.')

    collisions = check_collisions(
        infos,
        include_synthetic=args.include_synthetic,
        classes=args.classes,
        ego_length=args.ego_length,
        ego_width=args.ego_width,
    )

    print_report(collisions)

    if args.save_json and collisions:
        with open(args.save_json, 'w') as f:
            json.dump(collisions, f, indent=2)
        print(f'\nCollision details saved → {args.save_json}')

    # Load NuScenes once (for map overlay) if dataroot looks valid
    nusc = None
    if args.dataroot and os.path.isdir(args.dataroot):
        try:
            from nuscenes.nuscenes import NuScenes
            print(f'\nLoading NuScenes map ({args.nusc_version}) from {args.dataroot} …')
            nusc = NuScenes(version=args.nusc_version,
                            dataroot=args.dataroot, verbose=False)
        except Exception as e:
            print(f'WARNING: Could not load NuScenes — map overlay disabled ({e})')

    if args.plot_dir:
        plot_collision_scenes(
            infos, collisions,
            num_scenes=args.num_plot_scenes,
            output_dir=args.plot_dir,
            ego_length=args.ego_length,
            ego_width=args.ego_width,
            nusc=nusc,
        )

    if args.plot_occluded_dir:
        thr = args.nonstationary_threshold
        occ_collisions = [
            c for c in collisions
            if (c['prev_occluded'] or c['is_interp'] or c['is_extrap'])
            and c['agent_speed'] >= thr
            and c['ego_speed']   >= thr
            # and c['occ_start_frame_idx'] is not None
            # and (c['frame_idx'] - c['occ_start_frame_idx']) <= 12
        ]
        print_occluded_stats(occ_collisions, thr)
        plot_collision_scenes(
            infos, occ_collisions,
            num_scenes=args.num_plot_scenes,
            output_dir=args.plot_occluded_dir,
            ego_length=args.ego_length,
            ego_width=args.ego_width,
            use_occ_window=True,
            nusc=nusc,
            only_collided=True,
        )

        # Variant 1: same layout but collision circles suppressed
        plot_collision_scenes(
            infos, occ_collisions,
            num_scenes=args.num_plot_scenes,
            output_dir=args.plot_occluded_dir,
            ego_length=args.ego_length,
            ego_width=args.ego_width,
            use_occ_window=True,
            nusc=nusc,
            only_collided=True,
            filename='nocol.png',
            show_coll_circles=False,
        )

        # Variant 2: collision circles kept, but colliding agent positions shifted
        # slightly towards the ego with magnitude ∝ ego trajectory length
        plot_collision_scenes(
            infos, occ_collisions,
            num_scenes=args.num_plot_scenes,
            output_dir=args.plot_occluded_dir,
            ego_length=args.ego_length,
            ego_width=args.ego_width,
            use_occ_window=True,
            nusc=nusc,
            only_collided=True,
            filename='noisecol.png',
            show_coll_circles=True,
            ego_noise_towards_agent=0.004,
        )

        render_collision_cameras(
            infos, occ_collisions,
            output_dir=args.plot_occluded_dir,
            dataroot=args.dataroot,
            nusc_version=args.nusc_version,
        )


if __name__ == '__main__':
    main()
