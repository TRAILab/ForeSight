#!/usr/bin/env python3
"""Fill annotation gaps in nuScenes info pkls and visualise the results.

Two processing passes:

1. **Gap interpolation** (CATR): fills frames between two consecutive
   observations of the same instance using Constant Acceleration and Turn
   Rate kinematics with a linear endpoint position correction.

2. **Forward extrapolation** (CV): projects each track's last observed
   state forward by up to ``--max-extrap-frames`` frames using Constant
   Velocity.

New flags in every sample info dict:
  ``is_interpolated`` (bool array) — CATR gap-fill boxes
  ``is_extrapolated``  (bool array) — CV tail-extrapolation boxes
Directly observed boxes have both flags False.

Sub-commands
------------
convert   Run the annotation pipeline and save a new pkl.
visualize Draw BEV track plots for example scenes from an existing pkl.

Examples
--------
    python tools/data_converter/nuscenes_occlusion_converter.py convert \\
        --input  data/infos/nuscenes_infos_val.pkl \\
        --output data/infos/nuscenes_infos_val_occ.pkl

    python tools/data_converter/nuscenes_occlusion_converter.py visualize \\
        --pkl data/infos/nuscenes_infos_val_occ.pkl \\
        --num-scenes 6 --output-dir viz/
"""

import pickle
import argparse
import os
import numpy as np
from collections import defaultdict


# ===========================================================================
# CATR kinematics helpers
# ===========================================================================

def _normalize_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _integrate_catr(x0, y0, yaw0, v0, omega, a, t):
    """Vectorised Riemann-sum integration of CATR kinematics from 0 to t."""
    n = max(200, int(abs(t) * 400))
    dt = t / n
    s = np.arange(n) * dt
    theta = yaw0 + omega * s
    v = v0 + a * s
    x = x0 + float(np.sum(v * np.cos(theta))) * dt
    y = y0 + float(np.sum(v * np.sin(theta))) * dt
    return x, y


def catr_interpolate(state0, state1, t, T):
    """Interpolate between two kinematic states using the CATR model.

    Derives constant turn rate (omega = dtheta/T) and linear acceleration
    (a = dv/T) from the endpoint states, integrates to get intermediate
    positions, then applies a linear endpoint correction so the trajectory
    passes through both endpoints exactly.
    """
    x0, y0, z0, l, w, h, yaw0, vx0, vy0 = state0
    x1, y1, z1, _, _, _, yaw1, vx1, vy1 = state1
    alpha = t / T
    v0 = 0.0 if np.isnan(vx0) or np.isnan(vy0) else float(np.hypot(vx0, vy0))
    v1 = 0.0 if np.isnan(vx1) or np.isnan(vy1) else float(np.hypot(vx1, vy1))
    omega = _normalize_angle(yaw1 - yaw0) / T
    a     = (v1 - v0) / T
    x_t, y_t = _integrate_catr(x0, y0, yaw0, v0, omega, a, t)
    x_T, y_T = _integrate_catr(x0, y0, yaw0, v0, omega, a, T)
    x_interp = x_t + alpha * (x1 - x_T)
    y_interp = y_t + alpha * (y1 - y_T)
    z_interp = z0 + alpha * (z1 - z0)
    yaw_t = _normalize_angle(yaw0 + omega * t)
    v_t   = max(0.0, v0 + a * t)
    return (float(x_interp), float(y_interp), float(z_interp),
            float(l), float(w), float(h), float(yaw_t),
            float(v_t * np.cos(yaw_t)), float(v_t * np.sin(yaw_t)))


# ===========================================================================
# Coordinate transform helpers
# ===========================================================================

def _lidar2global_RT(info):
    """Return (R 3×3, t 3) for the lidar → global transform of this sample."""
    from pyquaternion import Quaternion
    R_l2e = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    t_l2e = np.array(info['lidar2ego_translation'])
    R_e2g = Quaternion(info['ego2global_rotation']).rotation_matrix
    t_e2g = np.array(info['ego2global_translation'])
    R = R_e2g @ R_l2e
    t = R_e2g @ t_l2e + t_e2g
    return R, t


def _box_to_global(box7, vel2, info):
    """Convert a lidar-frame box + lidar-frame velocity to a global-frame state.

    Parameters
    ----------
    box7 : array-like (7,)  [x, y, z, l, w, h, yaw] in lidar frame
    vel2 : array-like (2,)  [vx, vy] in lidar frame
                            (the nuScenes converter rotates box_velocity into lidar)
    info : sample info dict

    Returns
    -------
    9-tuple (x_g, y_g, z_g, l, w, h, yaw_g, vx_g, vy_g) all in global frame
    """
    R, t = _lidar2global_RT(info)
    p_g = R @ np.array([box7[0], box7[1], box7[2]]) + t
    yaw_offset = np.arctan2(R[1, 0], R[0, 0])
    yaw_g = _normalize_angle(float(box7[6]) + yaw_offset)
    # Rotate velocity from lidar frame to global frame (translation has no
    # effect on vectors, only the rotation part of R matters).
    if np.isnan(vel2[0]) or np.isnan(vel2[1]):
        vx_g, vy_g = 0.0, 0.0
    else:
        v_g = R @ np.array([float(vel2[0]), float(vel2[1]), 0.0])
        vx_g, vy_g = float(v_g[0]), float(v_g[1])
    return (float(p_g[0]), float(p_g[1]), float(p_g[2]),
            float(box7[3]), float(box7[4]), float(box7[5]),
            yaw_g, vx_g, vy_g)


def _state_global_to_lidar(state_g, info):
    """Convert a global-frame state tuple to the lidar frame of *info*.

    Both position/yaw and velocity are rotated into the target lidar frame,
    matching the gt_velocity convention used by the nuScenes converter.

    Parameters
    ----------
    state_g : 9-tuple (x_g, y_g, z_g, l, w, h, yaw_g, vx_g, vy_g)
    info    : target sample info dict

    Returns
    -------
    9-tuple (x, y, z, l, w, h, yaw, vx, vy) — all in target lidar frame.
    """
    x_g, y_g, z_g, l, w, h, yaw_g, vx_g, vy_g = state_g
    R, t = _lidar2global_RT(info)
    p_l = R.T @ (np.array([x_g, y_g, z_g]) - t)
    yaw_offset = np.arctan2(R[1, 0], R[0, 0])
    yaw_l = _normalize_angle(yaw_g - yaw_offset)
    # Rotate velocity from global frame back into the target lidar frame.
    v_l = R.T @ np.array([vx_g, vy_g, 0.0])
    return (float(p_l[0]), float(p_l[1]), float(p_l[2]),
            float(l), float(w), float(h), float(yaw_l), float(v_l[0]), float(v_l[1]))


def _global_xy_to_lidar_xy(positions_g, info):
    """Transform (N,2) global XY positions into the lidar frame of *info*.

    Uses the full 3-D lidar→global rotation (R.T) applied to the XY plane;
    the Z component is ignored since we only need XY deltas.
    """
    R, t = _lidar2global_RT(info)
    pos3 = np.zeros((len(positions_g), 3))
    pos3[:, :2] = positions_g
    return ((pos3 - t) @ R)[:, :2]   # R.T @ (p - t) written as row-vec form


# ===========================================================================
# Shared annotation-append helper
# ===========================================================================

def _append_annotation(info, x, y, z, l, w, h, yaw, vx, vy,
                        inst_ind, class_name, is_interp, is_extrap,
                        fut_trajs=None, fut_masks=None):
    new_box = np.array([[x, y, z, l, w, h, yaw]], dtype=np.float32)
    new_vel = np.array([[vx, vy]], dtype=np.float32)
    fut_ts  = info['gt_agent_fut_trajs'].shape[1]
    if fut_trajs is None:
        fut_trajs = np.zeros((1, fut_ts, 2), dtype=np.float32)
    else:
        fut_trajs = np.asarray(fut_trajs, dtype=np.float32).reshape(1, fut_ts, 2)
    if fut_masks is None:
        fut_masks = np.zeros((1, fut_ts), dtype=np.float32)
    else:
        fut_masks = np.asarray(fut_masks, dtype=np.float32).reshape(1, fut_ts)
    info['gt_boxes']           = np.concatenate([info['gt_boxes'],           new_box],     axis=0)
    info['gt_velocity']        = np.concatenate([info['gt_velocity'],        new_vel],     axis=0)
    info['gt_agent_fut_trajs'] = np.concatenate([info['gt_agent_fut_trajs'], fut_trajs],   axis=0)
    info['gt_agent_fut_masks'] = np.concatenate([info['gt_agent_fut_masks'], fut_masks],   axis=0)
    info['gt_names']        = np.append(info['gt_names'],        class_name)
    info['valid_flag']      = np.append(info['valid_flag'],       False)
    info['num_lidar_pts']   = np.append(info['num_lidar_pts'],    0)
    info['num_radar_pts']   = np.append(info['num_radar_pts'],    0)
    info['is_interpolated'] = np.append(info['is_interpolated'],  is_interp)
    info['is_extrapolated'] = np.append(info['is_extrapolated'],  is_extrap)
    info['instance_inds'].append(inst_ind)


# ===========================================================================
# Convert sub-command
# ===========================================================================

def cmd_convert(args):
    print(f'Loading {args.input} ...')
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos']
    print(f'  {len(infos)} samples loaded.')

    for info in infos:
        n = len(info['gt_boxes'])
        info['is_interpolated'] = np.zeros(n, dtype=bool)
        info['is_extrapolated'] = np.zeros(n, dtype=bool)

    scene_to_indices = defaultdict(list)
    for gi, info in enumerate(infos):
        scene_to_indices[info['scene_token']].append(gi)
    for sc in scene_to_indices:
        scene_to_indices[sc].sort(key=lambda i: infos[i]['timestamp'])

    # ------------------------------------------------------------------
    # Pass 1: gap interpolation (CATR)
    # ------------------------------------------------------------------
    total_interpolated = 0
    for _, global_indices in scene_to_indices.items():
        tracks = defaultdict(list)
        for frame_pos, gi in enumerate(global_indices):
            for bi, inst_ind in enumerate(infos[gi]['instance_inds']):
                tracks[inst_ind].append((frame_pos, gi, bi))

        for inst_ind, appearances in tracks.items():
            for k in range(len(appearances) - 1):
                fp0, gi0, bi0 = appearances[k]
                fp1, gi1, bi1 = appearances[k + 1]
                if fp1 - fp0 <= 1:
                    continue
                info0, info1 = infos[gi0], infos[gi1]
                # Convert both endpoints to global frame so CATR operates
                # in a single consistent coordinate system.
                state0_g = _box_to_global(
                    info0['gt_boxes'][bi0], info0['gt_velocity'][bi0], info0)
                state1_g = _box_to_global(
                    info1['gt_boxes'][bi1], info1['gt_velocity'][bi1], info1)
                class_name = info0['gt_names'][bi0]
                t0_s = info0['timestamp'] * 1e-6
                T    = info1['timestamp'] * 1e-6 - t0_s
                for gap_fp in range(fp0 + 1, fp1):
                    gap_gi   = global_indices[gap_fp]
                    gap_info = infos[gap_gi]
                    t = gap_info['timestamp'] * 1e-6 - t0_s
                    result_g = catr_interpolate(state0_g, state1_g, t, T)
                    # Convert the global-frame result back to the target frame's
                    # lidar coordinates before storing.
                    x, y, z, l, w, h, yaw, vx, vy = _state_global_to_lidar(
                        result_g, gap_info)
                    # Build future trajectory: CATR positions for future scene
                    # frames that still fall within the known gap [t0_s, t0_s+T].
                    fut_ts_n = gap_info['gt_agent_fut_trajs'].shape[1]
                    fut_positions_g = [np.array([result_g[0], result_g[1]])]
                    for k in range(1, fut_ts_n + 1):
                        fut_fp = gap_fp + k
                        if fut_fp >= len(global_indices):
                            break
                        t_fut_rel = infos[global_indices[fut_fp]]['timestamp'] * 1e-6 - t0_s
                        if t_fut_rel > T:
                            break
                        fut_g = catr_interpolate(state0_g, state1_g, t_fut_rel, T)
                        fut_positions_g.append(np.array([fut_g[0], fut_g[1]]))
                    all_l = _global_xy_to_lidar_xy(np.array(fut_positions_g), gap_info)
                    fut_trajs = np.zeros((fut_ts_n, 2), dtype=np.float32)
                    fut_masks = np.zeros(fut_ts_n, dtype=np.float32)
                    for k in range(1, len(fut_positions_g)):
                        fut_trajs[k - 1] = all_l[k] - all_l[k - 1]
                        fut_masks[k - 1] = 1.0
                    _append_annotation(gap_info, x, y, z, l, w, h, yaw, vx, vy,
                                       inst_ind, class_name,
                                       is_interp=True, is_extrap=False,
                                       fut_trajs=fut_trajs, fut_masks=fut_masks)
                    total_interpolated += 1

    # ------------------------------------------------------------------
    # Pass 2: forward extrapolation (CV)
    # ------------------------------------------------------------------
    total_extrap = 0
    if not args.no_extrapolate:
        for _, global_indices in scene_to_indices.items():
            n_frames = len(global_indices)
            present = [set(infos[gi]['instance_inds']) for gi in global_indices]

            orig_tracks = defaultdict(list)
            for frame_pos, gi in enumerate(global_indices):
                info = infos[gi]
                for bi, inst_ind in enumerate(info['instance_inds']):
                    if not info['is_interpolated'][bi] and not info['is_extrapolated'][bi]:
                        orig_tracks[inst_ind].append((frame_pos, gi, bi))

            for inst_ind, appearances in orig_tracks.items():
                last_fp, last_gi, last_bi = appearances[-1]
                if last_fp >= n_frames - 1:
                    continue
                info_ref = infos[last_gi]
                class_name = info_ref['gt_names'][last_bi]
                # Convert last observation to global frame so CV extrapolation
                # uses a consistent coordinate system.
                x_g, y_g, z_g, l, w, h, yaw_g, vx_g, vy_g = _box_to_global(
                    info_ref['gt_boxes'][last_bi],
                    info_ref['gt_velocity'][last_bi],
                    info_ref)
                t_ref = info_ref['timestamp'] * 1e-6
                frames_added = 0
                for fp in range(last_fp + 1, n_frames):
                    if inst_ind in present[fp] or frames_added >= args.max_extrap_frames:
                        break
                    gi  = global_indices[fp]
                    dt  = infos[gi]['timestamp'] * 1e-6 - t_ref
                    # Extrapolate position in global frame, then convert to
                    # the target frame's lidar coordinates.
                    state_g = (x_g + vx_g*dt, y_g + vy_g*dt, z_g,
                               l, w, h, yaw_g, vx_g, vy_g)
                    x, y, z, l_, w_, h_, yaw, vx, vy = _state_global_to_lidar(
                        state_g, infos[gi])
                    # Build future trajectory using CV from the current extrap
                    # position, for scene frames within the remaining scene.
                    fut_ts_n = infos[gi]['gt_agent_fut_trajs'].shape[1]
                    fut_positions_g = [np.array([state_g[0], state_g[1]])]
                    for k in range(1, fut_ts_n + 1):
                        fut_fp = fp + k
                        if fut_fp >= n_frames:
                            break
                        dt_fut = infos[global_indices[fut_fp]]['timestamp'] * 1e-6 - t_ref
                        fut_positions_g.append(
                            np.array([x_g + vx_g*dt_fut, y_g + vy_g*dt_fut]))
                    all_l = _global_xy_to_lidar_xy(np.array(fut_positions_g), infos[gi])
                    fut_trajs = np.zeros((fut_ts_n, 2), dtype=np.float32)
                    fut_masks = np.zeros(fut_ts_n, dtype=np.float32)
                    for k in range(1, len(fut_positions_g)):
                        fut_trajs[k - 1] = all_l[k] - all_l[k - 1]
                        fut_masks[k - 1] = 1.0
                    _append_annotation(infos[gi], x, y, z, l_, w_, h_, yaw,
                                       vx, vy, inst_ind, class_name,
                                       is_interp=False, is_extrap=True,
                                       fut_trajs=fut_trajs, fut_masks=fut_masks)
                    present[fp].add(inst_ind)
                    frames_added += 1
                    total_extrap += 1

    n_orig   = sum(int(np.sum(~i['is_interpolated'] & ~i['is_extrapolated'])) for i in infos)
    n_interp = sum(int(np.sum( i['is_interpolated'])) for i in infos)
    n_extrap = sum(int(np.sum( i['is_extrapolated'])) for i in infos)
    print(f'Original annotations       : {n_orig}')
    print(f'Interpolated (CATR)        : {n_interp}')
    print(f'Extrapolated forward (CV)  : {total_extrap}')
    print(f'Total annotations          : {n_orig + n_interp + n_extrap}')

    print(f'Saving to {args.output} ...')
    data['infos'] = infos
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
    print('Done.')


# ===========================================================================
# Visualize sub-command
# ===========================================================================

_C_OBS     = '#4fc3f7'   # observed, valid_flag=True
_C_INVALID = '#9575cd'   # observed, valid_flag=False (0 lidar/radar pts)
_C_INTERP  = '#ffb74d'
_C_EXTRAP  = '#ef5350'
_C_EGO     = '#69f0ae'
_C_BG      = '#0d1117'


def _ego_pose_global(info):
    from pyquaternion import Quaternion
    q   = Quaternion(info['ego2global_rotation'])
    pos = np.array(info['ego2global_translation'])[:2]
    yaw = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))
    return pos, yaw


def _draw_box(ax, cx, cy, length, width, yaw, color,
              alpha_face=0.22, alpha_edge=0.85, lw=0.8, zorder=3):
    import matplotlib.pyplot as plt
    c, s   = np.cos(yaw), np.sin(yaw)
    R2     = np.array([[c, -s], [s, c]])
    half   = np.array([[ length/2,  width/2],
                       [-length/2,  width/2],
                       [-length/2, -width/2],
                       [ length/2, -width/2]])
    corners = half @ R2.T + np.array([cx, cy])
    ax.add_patch(plt.Polygon(corners, closed=True, facecolor=color, edgecolor=color,
                             alpha=alpha_face, linewidth=lw, zorder=zorder))
    ax.add_patch(plt.Polygon(corners, closed=True, facecolor='none', edgecolor=color,
                             alpha=alpha_edge, linewidth=lw, zorder=zorder+1))


def _visualize_scene(infos, gidxs, ax, title='', show_forecast=False):
    inst_data = defaultdict(list)
    ego_pos, ego_yaws = [], []

    # Reference frame: first ego pose in the scene.
    # All global-frame coordinates are transformed so that the first ego
    # position is the origin and the first ego heading points along +X.
    p0, yaw0 = _ego_pose_global(infos[gidxs[0]])
    c0, s0 = np.cos(yaw0), np.sin(yaw0)
    R_inv = np.array([[c0, s0], [-s0, c0]])  # global → first-ego-frame rotation

    def to_ego(xy_global):
        """Transform (N,2) or (2,) from global frame to first-ego frame."""
        return (np.asarray(xy_global) - p0) @ R_inv.T

    for frame_idx, gi in enumerate(gidxs):
        info = infos[gi]
        R, t = _lidar2global_RT(info)
        boxes = info['gt_boxes']
        is_i  = info['is_interpolated']
        is_e  = info['is_extrapolated']

        if len(boxes):
            xy_g   = (boxes[:, :3] @ R.T + t)[:, :2]
            yaw_g  = boxes[:, 6] + np.arctan2(R[1, 0], R[0, 0])
            xy_e   = to_ego(xy_g)
            yaw_e  = yaw_g - yaw0
            valid  = info['valid_flag']
            for bi in range(len(boxes)):
                if is_i[bi]:
                    color = _C_INTERP
                elif is_e[bi]:
                    color = _C_EXTRAP
                elif valid[bi]:
                    color = _C_OBS
                else:
                    color = _C_INVALID

                # Reconstruct future waypoints in first-ego frame.
                # Deltas are in the current sample's lidar frame; accumulate
                # them to get absolute lidar positions, then transform to global
                # and finally to first-ego frame.
                fut_traj_raw = info['gt_agent_fut_trajs'][bi]  # (T, 2) lidar deltas
                fut_mask_raw = info['gt_agent_fut_masks'][bi]   # (T,)
                pos_l = np.array([boxes[bi, 0], boxes[bi, 1]], dtype=np.float64)
                fut_pts_l = []
                for k in range(len(fut_traj_raw)):
                    if fut_mask_raw[k] < 0.5:
                        break
                    pos_l = pos_l + fut_traj_raw[k]
                    fut_pts_l.append(pos_l.copy())
                if fut_pts_l:
                    fpl = np.zeros((len(fut_pts_l), 3))
                    fpl[:, :2] = np.array(fut_pts_l)
                    fut_pts_ego = to_ego((fpl @ R.T + t)[:, :2])
                else:
                    fut_pts_ego = np.zeros((0, 2))

                inst_data[info['instance_inds'][bi]].append((
                    frame_idx,
                    float(xy_e[bi, 0]), float(xy_e[bi, 1]),
                    float(boxes[bi, 3]), float(boxes[bi, 4]),
                    float(yaw_e[bi]), color,
                    fut_pts_ego,   # index 7: (K,2) future waypoints in ego frame
                ))

        pos, yaw = _ego_pose_global(info)
        ego_pos.append(to_ego(pos))
        ego_yaws.append(yaw - yaw0)

    ego_pos = np.array(ego_pos)

    # Track lines (adjacent frames only)
    for frames in inst_data.values():
        frames = sorted(frames, key=lambda f: f[0])
        for k in range(len(frames) - 1):
            f0, f1 = frames[k], frames[k + 1]
            if f1[0] - f0[0] > 1:
                continue
            ax.plot([f0[1], f1[1]], [f0[2], f1[2]],
                    color=f0[6], linewidth=0.7, alpha=0.55, zorder=2,
                    solid_capstyle='round')

    # Boxes — draw back-to-front so valid observed renders on top
    _alpha_face = {_C_EXTRAP: 0.13, _C_INTERP: 0.22, _C_INVALID: 0.18, _C_OBS: 0.22}
    _alpha_edge = {_C_EXTRAP: 0.55, _C_INTERP: 0.85, _C_INVALID: 0.70, _C_OBS: 0.85}
    _lw         = {_C_EXTRAP: 0.5,  _C_INTERP: 0.8,  _C_INVALID: 0.7,  _C_OBS: 0.8}
    for target_color in [_C_EXTRAP, _C_INTERP, _C_INVALID, _C_OBS]:
        for frames in inst_data.values():
            for f in frames:
                if f[6] != target_color:
                    continue
                _draw_box(ax, f[1], f[2], f[3], f[4], f[5], color=f[6],
                          alpha_face=_alpha_face[f[6]],
                          alpha_edge=_alpha_edge[f[6]],
                          lw=_lw[f[6]])

    # Forecast trajectories (one dotted line per box appearance with valid future steps)
    if show_forecast:
        for frames in inst_data.values():
            for f in frames:
                fut_pts = f[7]
                if len(fut_pts) == 0:
                    continue
                all_pts = np.vstack([[[f[1], f[2]]], fut_pts])
                ax.plot(all_pts[:, 0], all_pts[:, 1],
                        color=f[6], alpha=0.55, linewidth=1.0,
                        linestyle=':', zorder=4, solid_capstyle='round')
                ax.scatter(fut_pts[:, 0], fut_pts[:, 1],
                           c=f[6], s=5, alpha=0.65, zorder=5, linewidths=0)

    # Ego trajectory and boxes
    ax.plot(ego_pos[:, 0], ego_pos[:, 1],
            color='white', linewidth=1.5, linestyle='--', zorder=6, alpha=0.8)
    ax.scatter(*ego_pos[0],  color='white', s=30, zorder=8, marker='o')
    ax.scatter(*ego_pos[-1], color='white', s=30, zorder=8, marker='x')
    for pos, yaw in zip(ego_pos, ego_yaws):
        _draw_box(ax, pos[0], pos[1], 4.08, 1.73, yaw,
                  color=_C_EGO, alpha_face=0.30, alpha_edge=0.9, lw=1.0, zorder=7)

    ax.set_aspect('equal')
    ax.set_facecolor(_C_BG)
    ax.grid(True, color='white', alpha=0.07, linewidth=0.5)
    ax.tick_params(colors='#aaaaaa', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#333333')
    ax.set_xlabel('X (m)', color='#aaaaaa', fontsize=8)
    ax.set_ylabel('Y (m)', color='#aaaaaa', fontsize=8)
    n_valid   = sum(sum(1 for f in v if f[6] == _C_OBS)     for v in inst_data.values())
    n_invalid = sum(sum(1 for f in v if f[6] == _C_INVALID) for v in inst_data.values())
    n_interp  = sum(sum(1 for f in v if f[6] == _C_INTERP)  for v in inst_data.values())
    n_extrap  = sum(sum(1 for f in v if f[6] == _C_EXTRAP)  for v in inst_data.values())
    ax.set_title(f'{title}\n{len(gidxs)} frames | '
                 f'valid {n_valid}  invalid {n_invalid}  interp {n_interp}  extrap {n_extrap}',
                 color='white', fontsize=8, pad=4)


def cmd_visualize(args):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    print(f'Loading {args.pkl} ...')
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos']

    scene_map = defaultdict(list)
    for gi, info in enumerate(infos):
        scene_map[info['scene_token']].append(gi)
    for sc in scene_map:
        scene_map[sc].sort(key=lambda i: infos[i]['timestamp'])
    all_scenes = list(scene_map.values())

    os.makedirs(args.output_dir, exist_ok=True)

    legend_handles = [
        mpatches.Patch(color=_C_OBS,     label='Observed (valid)'),
        mpatches.Patch(color=_C_INVALID, label='Observed (invalid, 0 pts)'),
        mpatches.Patch(color=_C_INTERP,  label='Interpolated (CATR)'),
        mpatches.Patch(color=_C_EXTRAP,  label='Extrapolated (CV, ≤12 frames)'),
        mpatches.Patch(color=_C_EGO,     label='Ego vehicle'),
    ]

    if args.scene_idx is not None:
        scene_list = [all_scenes[args.scene_idx]]
        fig, ax = plt.subplots(figsize=(10, 10), facecolor=_C_BG)
        gidxs  = scene_list[0]
        sc_tok = infos[gidxs[0]]['scene_token']
        _visualize_scene(infos, gidxs, ax, title=f'Scene {sc_tok[:8]}…')
        ax.legend(handles=legend_handles, loc='upper right',
                  framealpha=0.5, fontsize=8,
                  labelcolor='white', facecolor='#222222', edgecolor='#444444')
        out = args.output or os.path.join(args.output_dir, 'occ_viz.png')
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f'Saved → {out}')
        plt.close(fig)
        return

    import matplotlib.lines as mlines
    forecast_handle = mlines.Line2D(
        [], [], color='white', linestyle=':', linewidth=1.2, alpha=0.7,
        label='Forecast trajectory (gt_agent_fut_trajs)')

    def _score_interp(gidxs):
        return sum(int(infos[gi]['is_interpolated'].sum()) for gi in gidxs)

    def _score_extrap(gidxs):
        return sum(int(infos[gi]['is_extrapolated'].sum()) for gi in gidxs)

    def _score_invalid(gidxs):
        total = 0
        for gi in gidxs:
            info = infos[gi]
            obs_mask = ~info['is_interpolated'] & ~info['is_extrapolated']
            total += int((obs_mask & ~info['valid_flag']).sum())
        return total

    def _score_forecast(gidxs):
        # Total valid future steps across all agents — favours scenes with many
        # agents that have rich forecast data (interp/extrap agents included).
        return sum(int(infos[gi]['gt_agent_fut_masks'].sum()) for gi in gidxs)

    top_interp   = sorted(all_scenes, key=_score_interp,   reverse=True)[:args.num_scenes]
    top_extrap   = sorted(all_scenes, key=_score_extrap,   reverse=True)[:args.num_scenes]
    top_invalid  = sorted(all_scenes, key=_score_invalid,  reverse=True)[:args.num_scenes]
    top_forecast = sorted(all_scenes, key=_score_forecast, reverse=True)[:args.num_scenes]

    def _save_grid(scene_list, out_path, suptitle, show_forecast=False):
        ncols = min(3, len(scene_list))
        nrows = (len(scene_list) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 7*nrows), facecolor=_C_BG)
        axes = np.array(axes).flatten()
        for i, gidxs in enumerate(scene_list):
            sc_tok = infos[gidxs[0]]['scene_token']
            _visualize_scene(infos, gidxs, axes[i],
                             title=f'Scene {sc_tok[:8]}…',
                             show_forecast=show_forecast)
        for j in range(len(scene_list), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(suptitle, color='white', fontsize=11, y=1.01)
        handles = legend_handles + ([forecast_handle] if show_forecast else [])
        fig.legend(handles=handles, loc='lower center', ncol=len(handles),
                   framealpha=0.5, fontsize=9,
                   labelcolor='white', facecolor='#222222', edgecolor='#444444',
                   bbox_to_anchor=(0.5, 0.01))
        fig.subplots_adjust(hspace=0.35, wspace=0.25, bottom=0.07)
        plt.savefig(out_path, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f'Saved → {out_path}')
        plt.close(fig)

    _save_grid(top_interp,
               os.path.join(args.output_dir, 'occ_viz_interp.png'),
               f'Top {args.num_scenes} scenes by interpolated annotation count')
    _save_grid(top_extrap,
               os.path.join(args.output_dir, 'occ_viz_extrap.png'),
               f'Top {args.num_scenes} scenes by extrapolated annotation count')
    _save_grid(top_invalid,
               os.path.join(args.output_dir, 'occ_viz_invalid.png'),
               f'Top {args.num_scenes} scenes by invalid (0-pt) observed annotation count')
    _save_grid(top_forecast,
               os.path.join(args.output_dir, 'occ_viz_forecast.png'),
               f'Top {args.num_scenes} scenes by forecast coverage (gt_agent_fut_trajs)',
               show_forecast=True)


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_conv = sub.add_parser('convert', help='Run annotation pipeline and save pkl.')
    p_conv.add_argument('--data-dir', default='data/infos',
                        help='Directory containing the nuScenes info pkls. '
                             'All three splits (train/val/test) are processed '
                             'unless --input is given.')
    p_conv.add_argument('--input',  default=None,
                        help='Single input pkl (overrides --data-dir loop)')
    p_conv.add_argument('--output', default=None,
                        help='Single output pkl (required when --input is set)')
    p_conv.add_argument('--no-extrapolate', action='store_true',
                        help='Disable forward CV extrapolation')
    p_conv.add_argument('--max-extrap-frames', type=int, default=12,
                        help='Max frames to extrapolate forward (default: 12)')

    p_viz = sub.add_parser('visualize', help='BEV plots of occluded annotations.')
    p_viz.add_argument('--pkl', default='data/infos/nuscenes_infos_val_occ.pkl')
    p_viz.add_argument('--scene-idx',  type=int, default=None,
                       help='Scene index (0-based)')
    p_viz.add_argument('--num-scenes', type=int, default=6,
                       help='Number of example scenes to plot (default 6)')
    p_viz.add_argument('--output-dir', default='vis/gt_occ',
                       help='Directory for output images')
    p_viz.add_argument('--output', default=None,
                       help='Single output filename (overrides --output-dir)')

    args = parser.parse_args()
    if args.cmd == 'convert':
        if args.input is not None:
            if args.output is None:
                parser.error('--output is required when --input is specified')
            cmd_convert(args)
        else:
            _SPLITS = ('train', 'val')
            for split in _SPLITS:
                inp = os.path.join(args.data_dir, f'nuscenes_infos_{split}.pkl')
                out = os.path.join(args.data_dir, f'nuscenes_infos_{split}_occ.pkl')
                if not os.path.exists(inp):
                    print(f'Skipping {inp} (not found)')
                    continue
                args.input  = inp
                args.output = out
                print(f'\n=== {split} ===')
                cmd_convert(args)
    else:
        cmd_visualize(args)


if __name__ == '__main__':
    main()
