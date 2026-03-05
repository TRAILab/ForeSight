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


def _integrate_catr(x0, y0, yaw0, v0, omega, a, t, clamp_v=False):
    """Vectorised Riemann-sum integration of CATR kinematics from 0 to t.

    clamp_v : if True, speed is clamped to [0, inf) so a decelerating object
              stops rather than reversing (use for forward extrapolation).
    """
    n = max(200, int(abs(t) * 400))
    dt = t / n
    s = np.arange(n) * dt
    theta = yaw0 + omega * s
    v = v0 + a * s
    if clamp_v:
        v = np.maximum(v, 0.0)
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
# ML-prediction helpers
# ===========================================================================

def _load_ml_predictions(pred_path):
    """Load a UniTraj inference NPZ and build an instance-token index.

    Returns
    -------
    preds  : ndarray (N, 6, 60, 2)  agent-centric absolute XY offsets from
                                    the agent's last-observed position, in a
                                    frame whose +X axis aligns with the agent's
                                    heading.
    worlds : ndarray (N, 10)        center_objects_world — [:2] global XY
                                    origin, [6] global heading (yaw).
    lookup : dict  str → list[int]  instance token → row indices in preds.
    """
    npz    = np.load(pred_path, allow_pickle=True)
    preds  = npz['predictions']                            # (N, 6, 60, 2)
    meta   = npz['metadata'].item()                        # flat dict
    worlds = meta['center_objects_world'].astype(np.float64)  # (N, 10)
    ids    = meta['center_objects_id']                     # (N,) str
    lookup = defaultdict(list)
    for i, inst in enumerate(ids):
        lookup[str(inst)].append(i)
    return preds, worlds, lookup


def _find_pred_index(inst_id, x_g, y_g, worlds, lookup, pos_tol=2.0):
    """Return the row index in *worlds*/*preds* for this instance near (x_g, y_g).

    Matches by instance token first, then picks the candidate whose stored
    global XY is within *pos_tol* metres.  Returns None if no match.
    """
    cands = lookup.get(str(inst_id), [])
    if not cands:
        return None
    dists = [np.hypot(worlds[ci, 0] - x_g, worlds[ci, 1] - y_g) for ci in cands]
    best  = int(np.argmin(dists))
    return cands[best] if dists[best] < pos_tol else None


def _pred_global_xy(ml_preds_i, world, step):
    """Convert agent-centric prediction at *step* to global (x, y).

    ml_preds_i : (6, 60, 2) — mode 0 is used (probabilities not saved yet).
    world      : (10,) — world[:2] = global origin, world[6] = global heading.
    step       : 0-based UniTraj step (0 → 0.1 s ahead of prediction moment).
    Returns (x_g, y_g) or None when step is out of [0, 59].
    """
    if not (0 <= step < ml_preds_i.shape[1]):
        return None
    pred_ac = ml_preds_i[0, step]          # mode 0, shape (2,)
    theta   = float(world[6])
    c, s    = np.cos(theta), np.sin(theta)
    return (float(c * pred_ac[0] - s * pred_ac[1] + world[0]),
            float(s * pred_ac[0] + c * pred_ac[1] + world[1]))


def _dt_to_step(dt_seconds, unitraj_dt):
    """Seconds offset from prediction moment → 0-based UniTraj step index.

    Step 0 corresponds to *unitraj_dt* seconds ahead.
    Use unitraj_dt=0.1 for 10 Hz models, 0.5 for 2 Hz models.
    """
    return int(round(dt_seconds / unitraj_dt)) - 1


def _build_instance_token_map(nuscenes_dataroot, version='v1.0-trainval'):
    """Return a dict mapping integer nuScenes instance indices → token strings.

    ``nusc.getind('instance', token)`` returns the 0-based position of a record
    in the instance table, which is the same as its index in instance.json.
    This map lets us convert the integer ``inst_ind`` stored by ForeSight back
    to the 32-char token string stored in the UniTraj NPZ.
    """
    import json
    path = os.path.join(nuscenes_dataroot, version, 'instance.json')
    with open(path) as fh:
        instances = json.load(fh)
    return {i: rec['token'] for i, rec in enumerate(instances)}


# ===========================================================================
# Motion-model fitting helper
# ===========================================================================

def _fit_motion_model(appearances, infos, min_history,
                      ca_noise_thr, ca_max_thr, ca_consistency_thr,
                      omega_noise_thr, omega_max_thr, omega_consistency_thr):
    """Fit scalar acceleration and turn-rate from the trailing consecutive
    observation run of a track.

    Only frames that are immediately consecutive in the scene (frame_pos
    difference == 1) are used, starting from the last observation and
    working backwards.  Heading is estimated from velocity direction when
    speed > 0.5 m/s (more reliable than box yaw for moving objects).

    Returns
    -------
    (a_fit, omega_fit) :
        a_fit     — scalar acceleration (m/s²); 0.0 when not reliably fitted.
        omega_fit — turn rate       (rad/s);   0.0 when not reliably fitted.
    """
    # Build the longest trailing run of consecutive scene frames.
    run = [appearances[-1]]
    for i in range(len(appearances) - 2, -1, -1):
        if appearances[i + 1][0] - appearances[i][0] == 1:
            run.insert(0, appearances[i])
        else:
            break
    if len(run) < min_history:
        return 0.0, 0.0

    # Collect (timestamp, speed, box yaw) per frame.
    # Box yaw is directly annotated; velocity is inferred from position differences
    # and is therefore noisier — use box yaw for turn-rate estimation.
    states = []
    for _, gi, bi in run:
        info = infos[gi]
        sg  = _box_to_global(info['gt_boxes'][bi], info['gt_velocity'][bi], info)
        spd = float(np.hypot(sg[7], sg[8]))
        states.append((info['timestamp'] * 1e-6, spd, float(sg[6])))

    # Per-interval acceleration and turn-rate.
    accels, omegas = [], []
    for i in range(len(states) - 1):
        t0, v0, h0 = states[i]
        t1, v1, h1 = states[i + 1]
        dt = t1 - t0
        if dt <= 0:
            continue
        accels.append((v1 - v0) / dt)
        omegas.append(_normalize_angle(h1 - h0) / dt)

    if not accels:
        return 0.0, 0.0

    a_mean  = float(np.mean(accels))
    a_std   = float(np.std(accels))  if len(accels) > 1 else 0.0
    om_mean = float(np.mean(omegas))
    om_std  = float(np.std(omegas))  if len(omegas) > 1 else 0.0

    a_fit = (a_mean
             if a_std < ca_consistency_thr
             and ca_noise_thr <= abs(a_mean) <= ca_max_thr
             else 0.0)
    omega_fit = (om_mean
                 if om_std < omega_consistency_thr
                 and omega_noise_thr <= abs(om_mean) <= omega_max_thr
                 else 0.0)
    return a_fit, omega_fit


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

    # Load ML predictions if provided
    ml_preds = ml_worlds = ml_lookup = None
    inst_token_map = {}   # int index → 32-char token string
    if getattr(args, 'predictions', None):
        print(f'Loading ML predictions from {args.predictions} ...')
        ml_preds, ml_worlds, ml_lookup = _load_ml_predictions(args.predictions)
        print(f'  {len(ml_preds)} prediction entries loaded.')
        dataroot = getattr(args, 'nuscenes_dataroot', None)
        if dataroot:
            print(f'Building instance-token map from {dataroot} ...')
            inst_token_map = _build_instance_token_map(dataroot)
            print(f'  {len(inst_token_map)} instance records indexed.')
        else:
            print('WARNING: --nuscenes-dataroot not set; ML predictions will '
                  'not be matched (all extrapolations fall back to CV).')

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
    # Pass 2: forward extrapolation (ML prediction or CV fallback)
    # ------------------------------------------------------------------
    # NuScenes gt_names that map to MetaDrive VEHICLE / PEDESTRIAN / CYCLIST and
    # are therefore included in UniTraj inference predictions.
    # Source: scenarionet/converter/nuscenes/type.py
    #   VEHICLE_TYPE  → MetaDriveType.VEHICLE   → object_type 1
    #   HUMAN_TYPE    → MetaDriveType.PEDESTRIAN → object_type 2
    #   BICYCLE_TYPE  → MetaDriveType.CYCLIST    → object_type 3
    # Other categories (barrier, traffic_cone, movable_object.*, animal …)
    # are not predicted — CV fallback is correct for those.
    _ML_CLASSES = {
        # VEHICLE_TYPE
        'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
        'vehicle.emergency.ambulance', 'vehicle.emergency.police',
        # HUMAN_TYPE
        'pedestrian',
        'human.pedestrian.stroller', 'human.pedestrian.personal_mobility',
        'human.pedestrian.construction_worker', 'human.pedestrian.police_officer',
        # BICYCLE_TYPE
        'motorcycle', 'bicycle',
    }
    # Motorized road users eligible for CA/CTR motion-model fitting.
    _MOTORIZED_CLASSES = {
        'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
        'vehicle.emergency.ambulance', 'vehicle.emergency.police',
        'motorcycle',
    }
    # UniTraj VEHICLE type — stationary instances (total displacement < 2 m)
    # are filtered out during training, so ML predictions for these are
    # unreliable when the agent is near-stationary.  PEDESTRIAN and CYCLIST
    # have no displacement filter and keep their ML predictions when slow.
    _VEHICLE_CLASSES = {
        'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
        'vehicle.emergency.ambulance', 'vehicle.emergency.police',
    }
    total_extrap = 0
    total_extrap_ml = 0
    total_extrap_ml_shifted = 0   # ML matches via earlier-frame fallback
    cv_by_class: dict    = defaultdict(int)   # CV fallback counts broken down by class
    cv_model_counter     = defaultdict(int)   # fallback model breakdown (CP/CV/CA/CTR/CATR)
    cv_no_entry: int     = 0   # CV because instance has zero NPZ rows (unrecoverable)
    cv_dist_fail: int    = 0   # CV because closest NPZ row exceeds pos_tol (fixable?)
    cv_sanity_fail: int  = 0   # CV because ML 0.5 s point diverges too far from CV
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
                # Convert last observation to global frame.
                x_g, y_g, z_g, l, w, h, yaw_g, vx_g, vy_g = _box_to_global(
                    info_ref['gt_boxes'][last_bi],
                    info_ref['gt_velocity'][last_bi],
                    info_ref)
                t_ref = info_ref['timestamp'] * 1e-6

                # ML-prediction lookup: find the NPZ row whose stored global
                # position is closest to (x_g, y_g) for this instance.
                # ForeSight stores integer indices; the NPZ stores token strings
                # — convert via inst_token_map before querying ml_lookup.
                #
                # Primary match: prediction moment == last observed frame.
                # Fallback match: use an earlier observed frame whose position
                # aligns with a sliding-window prediction moment.  This lets
                # late-scene instances (last observed at frames 140-195 in 10 Hz
                # terms, beyond the sf=115 coverage) still receive ML-based
                # extrapolation.  pred_time_offset (seconds) is then added to
                # every dt so that step indices remain relative to the NPZ
                # prediction moment rather than the last observation.
                pred_row = None
                pred_time_offset = 0.0   # seconds from NPZ pred-moment to last_obs
                if ml_lookup is not None and class_name in _ML_CLASSES:
                    inst_token = inst_token_map.get(inst_ind)
                    if inst_token is not None:
                        pred_row = _find_pred_index(
                            inst_token, x_g, y_g, ml_worlds, ml_lookup)

                        if pred_row is None and len(appearances) > 1:
                            # Try earlier observations, most-recent first.
                            for earlier_fp, earlier_gi, earlier_bi in reversed(appearances[:-1]):
                                info_e = infos[earlier_gi]
                                state_e = _box_to_global(
                                    info_e['gt_boxes'][earlier_bi],
                                    info_e['gt_velocity'][earlier_bi],
                                    info_e)
                                row = _find_pred_index(
                                    inst_token, state_e[0], state_e[1],
                                    ml_worlds, ml_lookup)
                                if row is not None:
                                    dt_off = t_ref - info_e['timestamp'] * 1e-6
                                    # dt_off must be positive (earlier frame) and
                                    # within the 6 s prediction horizon.
                                    if 0 < dt_off < 6.0:
                                        pred_row = row
                                        pred_time_offset = dt_off
                                        break

                # Sanity-check: reject the ML prediction if its 0.5 s point
                # diverges too far from the CV prediction at the same time.
                # Threshold = 0.5 + alpha * speed (m) — tighter for slow movers.
                if pred_row is not None:
                    step_check = _dt_to_step(pred_time_offset + 0.5, args.unitraj_dt)
                    if 0 <= step_check < 60:
                        xy_ml_check = _pred_global_xy(
                            ml_preds[pred_row], ml_worlds[pred_row], step_check)
                        if xy_ml_check is not None:
                            speed = float(np.hypot(vx_g, vy_g))
                            x_cv_check = x_g + vx_g * 0.5
                            y_cv_check = y_g + vy_g * 0.5
                            tol = 0.5 + 0.3 * speed
                            if np.hypot(xy_ml_check[0] - x_cv_check,
                                        xy_ml_check[1] - y_cv_check) > tol:
                                pred_row = None
                                cv_sanity_fail += 1

                # Diagnostic: classify why ML lookup failed for ML classes.
                if pred_row is None and class_name in _ML_CLASSES and ml_lookup is not None:
                    inst_token_d = inst_token_map.get(inst_ind)
                    if inst_token_d is not None:
                        all_cands = ml_lookup.get(str(inst_token_d), [])
                        if all_cands:
                            cv_dist_fail += 1   # has NPZ entries but all exceeded pos_tol
                        else:
                            cv_no_entry += 1    # never appeared as center agent in inference

                # Per-instance motion state (used by all fallback models below).
                # Use box yaw directly — it is annotated by humans and more
                # reliable than velocity-direction heading, which is inferred
                # from position differences.
                v0          = float(np.hypot(vx_g, vy_g))
                drive_yaw_g = yaw_g

                # Override ML for stationary vehicles: UniTraj filters out
                # VEHICLE instances with total displacement < 2 m, so ML
                # predictions for slow vehicles are unreliable.  Pedestrians
                # and cyclists have no displacement filter and keep their ML
                # predictions even when near-stationary.
                if (pred_row is not None
                        and v0 < args.cv_stationary_thr
                        and class_name in _VEHICLE_CLASSES):
                    pred_row = None

                # Fit CA/CTR model from history for motorized classes without ML.
                ca_accel  = 0.0
                ca_omega  = 0.0
                if pred_row is None and class_name in _MOTORIZED_CLASSES:
                    ca_accel, ca_omega = _fit_motion_model(
                        appearances, infos, args.min_ca_history,
                        args.ca_noise_thr, args.ca_max_thr, args.ca_consistency_thr,
                        args.omega_noise_thr, args.omega_max_thr,
                        args.omega_consistency_thr)

                # Pre-compute ML handover state.  Step 59 (the final UniTraj
                # step) snaps close to zero for most predictions and is
                # discarded.  Steps 0–58 are reliable and used as-is.
                #   _ml_last : last ML step we use  (= 58, skipping only 59)
                _span    = 5
                _ml_last = 58
                ml_end_x,  ml_end_y   = x_g,        y_g
                ml_end_vx, ml_end_vy  = vx_g,       vy_g
                ml_end_yaw            = drive_yaw_g
                ml_end_dt_offset      = 0.0
                if pred_row is not None:
                    xy_end = _pred_global_xy(
                        ml_preds[pred_row], ml_worlds[pred_row], _ml_last)
                    if xy_end is not None:
                        ml_end_x, ml_end_y = xy_end
                        # Backward-difference velocity at _ml_last.
                        xy_bend = _pred_global_xy(
                            ml_preds[pred_row], ml_worlds[pred_row], _ml_last - _span)
                        if xy_bend is not None:
                            ml_end_vx = (ml_end_x - xy_bend[0]) / (_span * args.unitraj_dt)
                            ml_end_vy = (ml_end_y - xy_bend[1]) / (_span * args.unitraj_dt)
                        # Yaw: central difference centred on _ml_last.
                        xy_pend = _pred_global_xy(
                            ml_preds[pred_row], ml_worlds[pred_row], _ml_last - 2 * _span)
                        if xy_pend is not None:
                            dx_end = ml_end_x - xy_pend[0]
                            dy_end = ml_end_y - xy_pend[1]
                            if np.hypot(dx_end, dy_end) > 0.05:
                                ml_end_yaw = float(np.arctan2(dy_end, dx_end))
                    ml_end_dt_offset = (_ml_last + 1) * args.unitraj_dt - pred_time_offset

                frames_added = 0
                for fp in range(last_fp + 1, n_frames):
                    if inst_ind in present[fp] or frames_added >= args.max_extrap_frames:
                        break
                    gi  = global_indices[fp]
                    dt  = infos[gi]['timestamp'] * 1e-6 - t_ref

                    # --- Position, heading, velocity ---
                    # pred_time_offset shifts dt so steps are relative to the
                    # NPZ prediction moment (which may precede last_obs).
                    step = (_dt_to_step(pred_time_offset + dt, args.unitraj_dt)
                            if pred_row is not None else -1)
                    xy   = (_pred_global_xy(ml_preds[pred_row], ml_worlds[pred_row], step)
                            if (pred_row is not None and 0 <= step <= _ml_last) else None)

                    if xy is not None:
                        x_ep, y_ep = xy
                        # Heading: clamp to nearest step where central difference
                        # is valid (both neighbours in [0, 59]) so boundary steps
                        # borrow the adjacent central-diff yaw instead of using
                        # noisy one-sided differences.
                        _span   = 5
                        step_cd = max(_span, min(step, 59 - _span))
                        xy_n_cd = _pred_global_xy(ml_preds[pred_row], ml_worlds[pred_row], step_cd + _span)
                        xy_p_cd = _pred_global_xy(ml_preds[pred_row], ml_worlds[pred_row], step_cd - _span)
                        if xy_n_cd is not None and xy_p_cd is not None:
                            dx, dy = xy_n_cd[0] - xy_p_cd[0], xy_n_cd[1] - xy_p_cd[1]
                        else:
                            dx, dy = 0.0, 0.0
                        yaw_ep = (float(np.arctan2(dy, dx))
                                  if np.hypot(dx, dy) > 0.05 else yaw_g)
                        # Velocity: same clamped central difference as yaw —
                        # reuse xy_n_cd / xy_p_cd already fetched above.
                        if xy_n_cd is not None and xy_p_cd is not None:
                            vx_ep = (xy_n_cd[0] - xy_p_cd[0]) / (2 * _span * args.unitraj_dt)
                            vy_ep = (xy_n_cd[1] - xy_p_cd[1]) / (2 * _span * args.unitraj_dt)
                        else:
                            vx_ep, vy_ep = vx_g, vy_g
                        state_g = (x_ep, y_ep, z_g, l, w, h, yaw_ep, vx_ep, vy_ep)
                        total_extrap_ml += 1
                        if pred_time_offset > 0:
                            total_extrap_ml_shifted += 1
                    else:
                        # Fallback hierarchy: CP → CATR/CA/CTR → CV.
                        # When ML was available but its horizon is exhausted,
                        # anchor from the ML step-59 endpoint to avoid a
                        # positional jump back to the original observation.
                        cv_by_class[class_name] += 1
                        if pred_row is not None:
                            ref_x,  ref_y   = ml_end_x,  ml_end_y
                            ref_vx, ref_vy  = ml_end_vx, ml_end_vy
                            ref_yaw         = ml_end_yaw
                            ref_v0          = float(np.hypot(ref_vx, ref_vy))
                            ref_dt          = dt - ml_end_dt_offset
                        else:
                            ref_x,  ref_y   = x_g,        y_g
                            ref_vx, ref_vy  = vx_g,        vy_g
                            ref_yaw         = drive_yaw_g
                            ref_v0          = v0
                            ref_dt          = dt
                        if ref_v0 < args.cv_stationary_thr:
                            model_key = 'CP'
                            state_g   = (ref_x, ref_y, z_g, l, w, h, ref_yaw, 0.0, 0.0)
                        elif ca_accel != 0.0 or ca_omega != 0.0:
                            if   ca_accel != 0.0 and ca_omega != 0.0:
                                model_key = 'CATR'
                            elif ca_accel != 0.0:
                                model_key = 'CA'
                            else:
                                model_key = 'CTR'
                            x_ca, y_ca = _integrate_catr(
                                ref_x, ref_y, ref_yaw, ref_v0, ca_omega, ca_accel, ref_dt,
                                clamp_v=True)
                            yaw_ca = _normalize_angle(ref_yaw + ca_omega * ref_dt)
                            v_ca   = max(0.0, ref_v0 + ca_accel * ref_dt)
                            state_g = (x_ca, y_ca, z_g, l, w, h, yaw_ca,
                                       v_ca * np.cos(yaw_ca), v_ca * np.sin(yaw_ca))
                        else:
                            model_key = 'CV'
                            state_g   = (ref_x + ref_vx*ref_dt, ref_y + ref_vy*ref_dt, z_g,
                                         l, w, h, ref_yaw, ref_vx, ref_vy)
                        cv_model_counter[model_key] += 1

                    x, y, z, l_, w_, h_, yaw, vx, vy = _state_global_to_lidar(
                        state_g, infos[gi])

                    # --- Future trajectory ---
                    fut_ts_n = infos[gi]['gt_agent_fut_trajs'].shape[1]
                    fut_positions_g = [np.array([state_g[0], state_g[1]])]
                    for k in range(1, fut_ts_n + 1):
                        fut_fp = fp + k
                        if fut_fp >= n_frames:
                            break
                        dt_fut   = infos[global_indices[fut_fp]]['timestamp'] * 1e-6 - t_ref
                        step_fut = (_dt_to_step(pred_time_offset + dt_fut, args.unitraj_dt)
                                    if pred_row is not None else -1)
                        xy_fut   = (_pred_global_xy(ml_preds[pred_row], ml_worlds[pred_row], step_fut)
                                    if (pred_row is not None and 0 <= step_fut <= _ml_last) else None)
                        if xy_fut is not None:
                            fut_positions_g.append(np.array([xy_fut[0], xy_fut[1]]))
                        else:
                            # Same ML-end anchoring as the main fallback above.
                            if pred_row is not None:
                                f_x,  f_y   = ml_end_x,  ml_end_y
                                f_vx, f_vy  = ml_end_vx, ml_end_vy
                                f_yaw       = ml_end_yaw
                                f_v0        = float(np.hypot(f_vx, f_vy))
                                f_dt        = dt_fut - ml_end_dt_offset
                            else:
                                f_x,  f_y   = x_g,        y_g
                                f_vx, f_vy  = vx_g,        vy_g
                                f_yaw       = drive_yaw_g
                                f_v0        = v0
                                f_dt        = dt_fut
                            if f_v0 < args.cv_stationary_thr:
                                fut_positions_g.append(np.array([f_x, f_y]))
                            elif ca_accel != 0.0 or ca_omega != 0.0:
                                x_ca, y_ca = _integrate_catr(
                                    f_x, f_y, f_yaw, f_v0, ca_omega, ca_accel, f_dt,
                                    clamp_v=True)
                                fut_positions_g.append(np.array([x_ca, y_ca]))
                            else:
                                fut_positions_g.append(
                                    np.array([f_x + f_vx*f_dt, f_y + f_vy*f_dt]))
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
    n_cv_non_ml = sum(v for k, v in cv_by_class.items() if k not in _ML_CLASSES)
    n_cv_ml_cls = sum(v for k, v in cv_by_class.items() if k in _ML_CLASSES)
    print(f'Original annotations       : {n_orig}')
    print(f'Interpolated (CATR)        : {n_interp}')
    print(f'Extrapolated (ML exact)    : {total_extrap_ml - total_extrap_ml_shifted}')
    print(f'Extrapolated (ML shifted)  : {total_extrap_ml_shifted}')
    print(f'Extrapolated (CV fallback) : {total_extrap - total_extrap_ml}')
    print(f'  of which non-ML classes  : {n_cv_non_ml}  '
          f'(traffic_cone/barrier/… — CV is correct here)')
    print(f'  of which ML classes      : {n_cv_ml_cls}  '
          f'(temporal gap or no NPZ entry)')
    print(f'    no NPZ entry (unrecoverable) : {cv_no_entry}')
    print(f'    pos_tol exceeded (fixable?)  : {cv_dist_fail}')
    print(f'    sanity filter (0.5s/speed)   : {cv_sanity_fail}')
    if cv_model_counter:
        mdl = sorted(cv_model_counter.items())
        print(f'  fallback model breakdown : ' +
              '  '.join(f'{k}:{v}' for k, v in mdl))
    if cv_by_class:
        by_cls = sorted(cv_by_class.items(), key=lambda x: -x[1])
        print(f'  CV breakdown by class    : ' +
              '  '.join(f'{k}:{v}' for k, v in by_cls))
    print(f'Total annotations          : {n_orig + n_interp + n_extrap}')

    # ------------------------------------------------------------------
    # Ego-distance filtering (two-level):
    # Level 2 — drop instances with no original obs within --max-dist.
    # Level 1 — tail-trim: drop all frames of an instance AFTER the last
    #           frame where it appears within --max-dist.  Using a tail-trim
    #           rather than an independent per-frame check ensures trajectories
    #           stay continuous; a per-frame check would punch holes in curved
    #           tracks that briefly exit and re-enter the range boundary,
    #           producing disconnected fragments 50+ m away.
    # ------------------------------------------------------------------
    if args.max_dist > 0:
        # Level-2: instances that have at least one original obs within range.
        valid_insts = set()
        for info in infos:
            boxes = info['gt_boxes']
            is_i  = info['is_interpolated']
            is_e  = info['is_extrapolated']
            for bi, inst_ind in enumerate(info['instance_inds']):
                if not is_i[bi] and not is_e[bi]:
                    if np.hypot(boxes[bi, 0], boxes[bi, 1]) <= args.max_dist:
                        valid_insts.add(inst_ind)
        print(f'  Ego-distance filter: {len(valid_insts):,} instances have '
              f'≥1 original obs within {args.max_dist} m.')

        # Level-1 pre-pass: for each valid instance record the timestamp of
        # its last frame that is within max_dist.  Timestamps are
        # monotonically increasing within a scene, so all frames up to and
        # including this timestamp are kept; frames after it are trimmed.
        last_ts_within: dict = {}
        for info in infos:
            ts    = info['timestamp']
            boxes = info['gt_boxes']
            for bi, inst_ind in enumerate(info['instance_inds']):
                if inst_ind not in valid_insts:
                    continue
                if np.hypot(boxes[bi, 0], boxes[bi, 1]) <= args.max_dist:
                    if ts > last_ts_within.get(inst_ind, -1):
                        last_ts_within[inst_ind] = ts

        # Level-1 apply: keep box if instance is valid AND the frame
        # timestamp is at or before the last within-range timestamp.
        n_before = sum(len(i['instance_inds']) for i in infos)
        for info in infos:
            ts    = info['timestamp']
            boxes = info['gt_boxes']
            keep = np.array([
                (inst_ind in valid_insts and
                 ts <= last_ts_within.get(inst_ind, -1))
                for bi, inst_ind in enumerate(info['instance_inds'])
            ], dtype=bool)
            info['gt_boxes']           = info['gt_boxes'][keep]
            info['gt_velocity']        = info['gt_velocity'][keep]
            info['gt_names']           = info['gt_names'][keep]
            info['valid_flag']         = info['valid_flag'][keep]
            info['num_lidar_pts']      = info['num_lidar_pts'][keep]
            info['num_radar_pts']      = info['num_radar_pts'][keep]
            info['is_interpolated']    = info['is_interpolated'][keep]
            info['is_extrapolated']    = info['is_extrapolated'][keep]
            info['gt_agent_fut_trajs'] = info['gt_agent_fut_trajs'][keep]
            info['gt_agent_fut_masks'] = info['gt_agent_fut_masks'][keep]
            info['instance_inds']      = [
                inst for inst, k in zip(info['instance_inds'], keep) if k]
        n_after = sum(len(i['instance_inds']) for i in infos)
        print(f'  Removed {n_before - n_after:,} box entries '
              f'({n_before:,} → {n_after:,}).')

    print(f'Saving to {args.output} ...')
    data['infos'] = infos
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
    print('Done.')


# ===========================================================================
# Visualize sub-command
# ===========================================================================

_C_OBS     = '#1565c0'   # observed, valid_flag=True        (blue)
_C_INVALID = '#6a1b9a'   # observed, valid_flag=False       (purple)
_C_INTERP  = '#e65100'   # interpolated (CATR)              (deep orange)
_C_EXTRAP  = '#b71c1c'   # extrapolated                     (dark red)
_C_EGO     = '#2e7d32'   # ego vehicle                      (dark green)
_C_BG      = '#ffffff'   # plot background                  (white)


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


def _visualize_scene(infos, gidxs, ax, title='', show_forecast=False, nusc=None):
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

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

    # Map background — drawn first so trajectories render on top.
    # nusc.explorer.render_ego_centric_map uses flat vehicle coordinates for
    # the reference sample, which matches our first-ego-frame (+X forward,
    # +Y left) exactly.  We pass a generous axes_limit so the full scene
    # extent is covered; explicit axis limits are set later from data.
    if nusc is not None:
        try:
            first_token = infos[gidxs[0]]['token']
            lidar_token = nusc.get('sample', first_token)['data']['LIDAR_TOP']
            nusc.explorer.render_ego_centric_map(
                sample_data_token=lidar_token, axes_limit=200, ax=ax)
        except Exception:
            pass  # map unavailable — continue without it

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

    # Track lines — drawn for every consecutive pair in chronological order.
    # Segments bridging a frame gap (diff > 1) are dashed to indicate the
    # discontinuity; all other segments are solid.
    for frames in inst_data.values():
        frames = sorted(frames, key=lambda f: f[0])
        for k in range(len(frames) - 1):
            f0, f1 = frames[k], frames[k + 1]
            gap    = f1[0] - f0[0]
            ax.plot([f0[1], f1[1]], [f0[2], f1[2]],
                    color=f0[6], linewidth=0.7,
                    alpha=0.35 if gap > 1 else 0.55,
                    linestyle='--' if gap > 1 else '-',
                    zorder=2, solid_capstyle='round')

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
            color='#333333', linewidth=1.5, linestyle='--', zorder=6, alpha=0.8)
    ax.scatter(*ego_pos[0],  color='#333333', s=30, zorder=8, marker='o')
    ax.scatter(*ego_pos[-1], color='#333333', s=30, zorder=8, marker='x')
    for pos, yaw in zip(ego_pos, ego_yaws):
        _draw_box(ax, pos[0], pos[1], 4.08, 1.73, yaw,
                  color=_C_EGO, alpha_face=0.30, alpha_edge=0.9, lw=1.0, zorder=7)

    # Explicit axis limits from data so the map imshow (which calls set_xlim/
    # set_ylim internally) does not dictate the final view.
    all_x = ([f[1] for v in inst_data.values() for f in v]
             + list(ego_pos[:, 0]))
    all_y = ([f[2] for v in inst_data.values() for f in v]
             + list(ego_pos[:, 1]))
    if all_x:
        xspan = max(all_x) - min(all_x)
        yspan = max(all_y) - min(all_y)
        span  = max(xspan, yspan, 20.0)
        pad   = span * 0.08
        cx    = (max(all_x) + min(all_x)) / 2
        cy    = (max(all_y) + min(all_y)) / 2
        ax.set_xlim(cx - span / 2 - pad, cx + span / 2 + pad)
        ax.set_ylim(cy - span / 2 - pad, cy + span / 2 + pad)
    ax.set_aspect('equal')
    ax.set_facecolor(_C_BG)
    ax.grid(True, color='#bbbbbb', alpha=0.4, linewidth=0.5)
    ax.tick_params(colors='#444444', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
    ax.set_xlabel('X (m)', color='#444444', fontsize=8)
    ax.set_ylabel('Y (m)', color='#444444', fontsize=8)
    n_valid   = sum(sum(1 for f in v if f[6] == _C_OBS)     for v in inst_data.values())
    n_invalid = sum(sum(1 for f in v if f[6] == _C_INVALID) for v in inst_data.values())
    n_interp  = sum(sum(1 for f in v if f[6] == _C_INTERP)  for v in inst_data.values())
    n_extrap  = sum(sum(1 for f in v if f[6] == _C_EXTRAP)  for v in inst_data.values())
    ax.set_title(
        f'{title} | Visible {n_valid} Occluded {n_invalid} '
        f'Interpolated {n_interp} Extrapolated {n_extrap}',
        color='#111111', fontsize=8, pad=4)

    leg_handles = [
        mpatches.Patch(color='#aaaaaa',  label='Driveable Area'),
        mpatches.Patch(color=_C_OBS,     label='Visible'),
        mpatches.Patch(color=_C_INVALID, label='Occluded'),
        mpatches.Patch(color=_C_INTERP,  label='Interpolated'),
        mpatches.Patch(color=_C_EXTRAP,  label='Extrapolated'),
        mpatches.Patch(color=_C_EGO,     label='Ego'),
    ]
    if show_forecast:
        leg_handles.append(mlines.Line2D(
            [], [], color='#333333', linestyle=':', linewidth=1.2, alpha=0.7,
            label='Forecast'))
    ax.legend(handles=leg_handles, loc='upper right', framealpha=0.85,
              fontsize=6, labelcolor='#111111', facecolor='white',
              edgecolor='#cccccc', ncol=1)


# ===========================================================================
# Single-track helpers  (individual track plots)
# ===========================================================================

def _collect_all_tracks(infos, all_scenes):
    """Return a list of per-instance track dicts across all scenes.

    Uses the same first-ego-frame coordinate system as ``_visualize_scene``:
    each scene is centred on its first ego pose so coordinates are directly
    comparable to the scene-level BEV plots.

    Each frame tuple is ``(frame_idx, x_e, y_e, l, w, yaw_e, color)`` —
    identical to the entries stored in ``inst_data`` inside ``_visualize_scene``.
    ``cv_frames`` is a parallel list of ``(x_e, y_e)`` CV predictions for each
    extrapolated frame, used by the mismatch plot.
    """
    tracks = []
    for gidxs in all_scenes:
        sc_tok = infos[gidxs[0]]['scene_token']

        # Same reference frame as _visualize_scene: first ego pose of the scene.
        p0, yaw0 = _ego_pose_global(infos[gidxs[0]])
        c0, s0   = np.cos(yaw0), np.sin(yaw0)
        R_ego    = np.array([[c0, s0], [-s0, c0]])   # global → first-ego rotation

        def to_ego(xy):
            return (np.asarray(xy) - p0) @ R_ego.T

        # inst_frames: same format as inst_data in _visualize_scene.
        # inst_meta:   raw (gi, bi) lists for CV anchor computation.
        inst_frames = defaultdict(list)
        inst_meta   = defaultdict(lambda: {'obs_raw': [], 'extrap_raw': []})

        for frame_idx, gi in enumerate(gidxs):
            info  = infos[gi]
            R, t  = _lidar2global_RT(info)
            boxes = info['gt_boxes']
            if not len(boxes):
                continue
            xy_g  = (boxes[:, :3] @ R.T + t)[:, :2]   # lidar → global
            yaw_g = boxes[:, 6] + np.arctan2(R[1, 0], R[0, 0])
            xy_e  = to_ego(xy_g)                        # global → first-ego
            yaw_e = yaw_g - yaw0
            is_i  = info['is_interpolated']
            is_e  = info['is_extrapolated']
            valid = info['valid_flag']

            for bi, inst_ind in enumerate(info['instance_inds']):
                if is_i[bi]:
                    color = _C_INTERP
                elif is_e[bi]:
                    color = _C_EXTRAP
                    inst_meta[inst_ind]['extrap_raw'].append((gi, bi))
                elif valid[bi]:
                    color = _C_OBS
                    inst_meta[inst_ind]['obs_raw'].append((gi, bi))
                else:
                    color = _C_INVALID
                    inst_meta[inst_ind]['obs_raw'].append((gi, bi))

                inst_frames[inst_ind].append((
                    frame_idx,
                    float(xy_e[bi, 0]), float(xy_e[bi, 1]),
                    float(boxes[bi, 3]), float(boxes[bi, 4]),
                    float(yaw_e[bi]), color,
                ))

        for inst_ind, raw_frames in inst_frames.items():
            frames_sorted = sorted(raw_frames, key=lambda f: f[0])
            n_interp = sum(1 for f in frames_sorted if f[6] == _C_INTERP)
            n_extrap = sum(1 for f in frames_sorted if f[6] == _C_EXTRAP)

            def _path_len(color):
                pts = [(f[1], f[2]) for f in frames_sorted if f[6] == color]
                return sum(
                    np.hypot(pts[i+1][0] - pts[i][0], pts[i+1][1] - pts[i][1])
                    for i in range(len(pts) - 1)
                )
            extrap_dist = _path_len(_C_EXTRAP)
            interp_dist = _path_len(_C_INTERP)

            # CV mismatch: project last-observed velocity forward in ego frame.
            # Rigid transform preserves distances, so scores are identical to
            # computing in global frame.
            max_cv_ml_diff = 0.0
            cv_frames      = []   # (x_e, y_e) per extrapolated frame
            meta = inst_meta[inst_ind]
            if meta['obs_raw'] and meta['extrap_raw']:
                # Use the last observed frame BEFORE the extrapolation starts.
                # obs_raw may contain re-appearance frames that come after the
                # extrap gap; using those as the anchor gives negative dt and
                # a completely wrong CV projection.
                first_extrap_gi = meta['extrap_raw'][0][0]
                obs_before = [(g, b) for g, b in meta['obs_raw']
                              if g < first_extrap_gi]
                if not obs_before:
                    obs_before = meta['obs_raw']
                anc_gi, anc_bi = obs_before[-1]
                anc_info = infos[anc_gi]
                state_g  = _box_to_global(
                    anc_info['gt_boxes'][anc_bi],
                    anc_info['gt_velocity'][anc_bi],
                    anc_info)
                xy0_e        = to_ego(np.array([state_g[0], state_g[1]]))
                x0_e, y0_e   = float(xy0_e[0]), float(xy0_e[1])
                v_e          = R_ego @ np.array([state_g[7], state_g[8]])
                vx_e, vy_e   = float(v_e[0]), float(v_e[1])
                t0           = anc_info['timestamp'] * 1e-6

                for gi_e, bi_e in meta['extrap_raw']:
                    dt    = infos[gi_e]['timestamp'] * 1e-6 - t0
                    cv_frames.append((x0_e + vx_e * dt, y0_e + vy_e * dt))

                ext_e  = [(f[1], f[2]) for f in frames_sorted if f[6] == _C_EXTRAP]
                if cv_frames and len(cv_frames) == len(ext_e):
                    diffs = [np.hypot(e[0] - c[0], e[1] - c[1])
                             for e, c in zip(ext_e, cv_frames)]
                    max_cv_ml_diff = float(max(diffs))

            # Class name from last observed frame (or first extrap/interp).
            class_name = ''
            for src, idx in [(meta['obs_raw'], -1), (meta['extrap_raw'], 0)]:
                if src:
                    gi_c, bi_c = src[idx]
                    class_name = infos[gi_c]['gt_names'][bi_c]
                    break

            tracks.append({
                'inst_ind':          inst_ind,
                'scene_token':       sc_tok,
                'first_sample_token': infos[gidxs[0]]['token'],
                'class_name':        class_name,
                'frames':            frames_sorted,   # (frame_idx, x_e, y_e, l, w, yaw_e, color)
                'cv_frames':         cv_frames,        # (x_e, y_e) per extrap frame
                'n_interp':          n_interp,
                'n_extrap':          n_extrap,
                'extrap_dist':       extrap_dist,
                'interp_dist':       interp_dist,
                'max_cv_ml_diff':    max_cv_ml_diff,
            })
    return tracks


def _visualize_single_track_ax(track, ax, show_cv=False, nusc=None):
    """Render one instance track using the same style as ``_visualize_scene``.

    Draws track lines between adjacent frames and oriented boxes at every
    frame position.  The view is auto-zoomed to the track's spatial extent.
    Coordinates are already in first-ego frame (set by ``_collect_all_tracks``).

    If *nusc* is provided, a map background is rendered first using the scene's
    first-frame LIDAR_TOP token.  The auto-zoom axis limits are always applied
    last so the view is unchanged.
    """
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    frames    = track['frames']   # (frame_idx, x_e, y_e, l, w, yaw_e, color)
    cv_frames = track['cv_frames']

    if not frames:
        return

    # --- Compute auto-zoom bounds first (needed for map axes_limit) ----------
    all_xy = [(f[1], f[2]) for f in frames]
    if show_cv:
        all_xy.extend(cv_frames)
    xs   = [p[0] for p in all_xy]
    ys   = [p[1] for p in all_xy]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 8.0)
    pad  = span * 0.20
    cx   = (max(xs) + min(xs)) / 2
    cy   = (max(ys) + min(ys)) / 2
    xlim = (cx - span / 2 - pad, cx + span / 2 + pad)
    ylim = (cy - span / 2 - pad, cy + span / 2 + pad)

    # --- Map background (rendered before track elements) ---------------------
    ax.set_facecolor(_C_BG)
    if nusc is not None:
        try:
            first_token = track.get('first_sample_token')
            if first_token:
                lidar_token = nusc.get('sample', first_token)['data']['LIDAR_TOP']
                # axes_limit must reach the farthest corner of the track from
                # the scene origin (0, 0) so the full map tile is loaded.
                map_limit = max(
                    abs(cx) + span / 2 + pad,
                    abs(cy) + span / 2 + pad,
                ) + 20.0
                nusc.explorer.render_ego_centric_map(
                    sample_data_token=lidar_token,
                    axes_limit=map_limit,
                    ax=ax)
        except Exception:
            pass   # map unavailable — continue without it

    # --- Track lines (all consecutive pairs; dashed across frame gaps) -------
    for k in range(len(frames) - 1):
        f0, f1 = frames[k], frames[k + 1]
        gap    = f1[0] - f0[0]
        ax.plot([f0[1], f1[1]], [f0[2], f1[2]],
                color=f0[6], lw=0.7,
                alpha=0.35 if gap > 1 else 0.55,
                linestyle='--' if gap > 1 else '-',
                zorder=2, solid_capstyle='round')

    # --- Boxes back-to-front (matching _visualize_scene render order) --------
    _alpha_face = {_C_EXTRAP: 0.13, _C_INTERP: 0.22, _C_INVALID: 0.18, _C_OBS: 0.22}
    _alpha_edge = {_C_EXTRAP: 0.55, _C_INTERP: 0.85, _C_INVALID: 0.70, _C_OBS: 0.85}
    _lw_d       = {_C_EXTRAP: 0.5,  _C_INTERP: 0.8,  _C_INVALID: 0.7,  _C_OBS: 0.8}
    for target_color in [_C_EXTRAP, _C_INTERP, _C_INVALID, _C_OBS]:
        for f in frames:
            if f[6] != target_color:
                continue
            _draw_box(ax, f[1], f[2], f[3], f[4], f[5], color=f[6],
                      alpha_face=_alpha_face[f[6]],
                      alpha_edge=_alpha_edge[f[6]],
                      lw=_lw_d[f[6]])

    # --- CV baseline ---------------------------------------------------------
    if show_cv and len(cv_frames) >= 2:
        cv_xs = [p[0] for p in cv_frames]
        cv_ys = [p[1] for p in cv_frames]
        ax.plot(cv_xs, cv_ys, color='#ffd54f', lw=1.0, ls='--',
                marker='.', ms=2, alpha=0.75, zorder=3)

    # Star at the obs→extrap/interp handover point.
    obs_frames = [f for f in frames if f[6] in (_C_OBS, _C_INVALID)]
    if obs_frames:
        lf = obs_frames[-1]
        ax.scatter(lf[1], lf[2], color='#333333', s=22, zorder=5,
                   marker='*', linewidths=0)

    # --- Apply auto-zoom limits (identical whether map was rendered or not) --
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.grid(True, color='#bbbbbb', alpha=0.4, lw=0.5)
    ax.tick_params(colors='#444444', labelsize=6)
    for spine in ax.spines.values():
        spine.set_color('#cccccc')

    leg_handles = [
        mpatches.Patch(color=_C_OBS,     label='Visible'),
        mpatches.Patch(color=_C_INVALID, label='Occluded'),
        mpatches.Patch(color=_C_INTERP,  label='Interpolated'),
        mpatches.Patch(color=_C_EXTRAP,  label='Extrapolated'),
    ]
    if show_cv:
        leg_handles.append(mlines.Line2D(
            [], [], color='#ffd54f', ls='--', lw=1.2, label='CV baseline'))
    ax.legend(handles=leg_handles, loc='upper right', framealpha=0.85,
              fontsize=5.5, labelcolor='#111111', facecolor='white',
              edgecolor='#cccccc', ncol=1)

    cls  = track['class_name']
    tok  = track['scene_token'][:6]
    n_e  = track['n_extrap']
    n_i  = track['n_interp']
    d_e  = track.get('extrap_dist', 0.0)
    d_i  = track.get('interp_dist', 0.0)
    diff = track['max_cv_ml_diff']
    parts = [f'{cls}', f'{tok}…']
    if n_i > 0:
        parts.append(f'{n_i} interp ({d_i:.1f} m)')
    if n_e > 0:
        parts.append(f'{n_e} extrap ({d_e:.1f} m)')
    if show_cv:
        parts.append(f'max CV Δ={diff:.1f} m')
    ax.set_title(' | '.join(parts), color='#111111', fontsize=7, pad=3)


def _save_single_track_grid(tracks, out_path, suptitle, score_fn,
                             top_n=12, show_cv=False, nusc=None):
    """Save a 4-column grid of single-track BEV plots ranked by *score_fn*."""
    import matplotlib.pyplot as plt

    ranked = sorted(tracks, key=score_fn, reverse=True)[:top_n]
    if not ranked:
        print(f'No tracks for {out_path}, skipping.')
        return

    ncols = 4
    nrows = (len(ranked) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 5 * nrows),
                             facecolor=_C_BG)
    axes = np.array(axes).flatten()
    for i, track in enumerate(ranked):
        _visualize_single_track_ax(track, axes[i], show_cv=show_cv, nusc=nusc)
    for j in range(len(ranked), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(suptitle, color='#111111', fontsize=11, y=1.01)
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'Saved → {out_path}')
    plt.close(fig)


# ===========================================================================
# Ego-distance histogram helper
# ===========================================================================

def _count_missing_under_dist(infos, all_scenes, max_dist):
    """Count instances whose last annotation ends before the scene and within max_dist.

    For each instance in each scene, finds the last frame where it appears
    (observed or extrapolated).  If that frame is not the final scene frame AND
    the ego distance at that position is below *max_dist*, the instance is
    counted as still-missing — our augmentation did not reach the scene end.

    Returns
    -------
    (n_instances, n_frames) :
        n_instances — number of such instance-scene pairs.
        n_frames    — total unannotated frame-slots those instances leave behind.
    """
    n_instances = 0
    n_gaps      = 0
    for gidxs in all_scenes:
        if len(gidxs) <= 1:
            continue
        inst_last = {}   # inst_ind → (frame_pos, gi, bi) of last annotation
        for frame_pos, gi in enumerate(gidxs):
            info = infos[gi]
            for bi, inst_ind in enumerate(info['instance_inds']):
                inst_last[inst_ind] = (frame_pos, gi, bi)
        for inst_ind, (fp, gi, bi) in inst_last.items():
            if fp >= len(gidxs) - 1:
                continue   # reaches the scene end — not missing
            box = infos[gi]['gt_boxes'][bi]
            if float(np.hypot(box[0], box[1])) < max_dist:
                n_instances += 1
                n_gaps      += len(gidxs) - 1 - fp
    return n_instances, n_gaps


def _save_ego_dist_hist(infos, out_path, max_dist=60, n_missing=None):
    """Overlay histograms of ego-distance for original vs augmented annotations.

    Boxes are stored in the lidar frame whose origin is the ego vehicle, so
    ego distance = hypot(box_x, box_y) directly.

    * 'Original'  — boxes where both is_interpolated and is_extrapolated are False.
    * 'Augmented' — all boxes (original + interpolated + extrapolated).

    Parameters
    ----------
    max_dist  : upper x-axis limit in metres (bins span [0, max_dist]).
    n_missing : optional (n_instances, n_frames) from ``_count_missing_under_dist``
                — when provided, annotated as text on the plot.
    """
    import matplotlib.pyplot as plt

    orig_dists, aug_dists = [], []
    for info in infos:
        boxes = info['gt_boxes']
        is_i  = info['is_interpolated']
        is_e  = info['is_extrapolated']
        for bi in range(len(info['instance_inds'])):
            d = float(np.hypot(boxes[bi, 0], boxes[bi, 1]))
            aug_dists.append(d)
            if not is_i[bi] and not is_e[bi]:
                orig_dists.append(d)

    orig_dists = np.array(orig_dists)
    aug_dists  = np.array(aug_dists)

    bins = np.linspace(0, max_dist, 51)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=_C_BG)
    ax.set_facecolor(_C_BG)
    orig_filt = orig_dists[orig_dists <= max_dist]
    aug_filt  = aug_dists[aug_dists   <= max_dist]
    ax.hist(orig_filt, bins=bins, color=_C_OBS,    alpha=0.75,
            label=f'Original  (n={len(orig_filt):,})')
    ax.hist(aug_filt,  bins=bins, color=_C_EXTRAP, alpha=0.55,
            label=f'Augmented (n={len(aug_filt):,})')

    ax.set_xlabel('Ego distance (m)', color='#444444', fontsize=10)
    ax.set_ylabel('Count',            color='#444444', fontsize=10)
    ax.set_title(
        f'Ego-distance distribution (0–{max_dist} m) — original vs augmented',
        color='#111111', fontsize=11, pad=6)
    ax.tick_params(colors='#444444')
    ax.legend(labelcolor='#111111', facecolor='white', edgecolor='#cccccc',
              framealpha=0.9, fontsize=10)
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
    ax.grid(True, color='#bbbbbb', alpha=0.4, linewidth=0.5)

    if n_missing is not None:
        n_inst, n_frm = n_missing
        print(f'Still missing (< {max_dist} m): '
              f'{n_inst:,} instances, {n_frm:,} frame-slots')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'Saved → {out_path}')
    plt.close(fig)


# ===========================================================================
# Visualize sub-command
# ===========================================================================

def cmd_visualize(args):
    import matplotlib.pyplot as plt

    print(f'Loading {args.pkl} ...')
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos']

    nusc = None
    if getattr(args, 'nuscenes_dataroot', None):
        try:
            from nuscenes import NuScenes
            print(f'Loading NuScenes from {args.nuscenes_dataroot} '
                  f'(version={args.nuscenes_version}) for map rendering …')
            nusc = NuScenes(version=args.nuscenes_version,
                            dataroot=args.nuscenes_dataroot, verbose=False)
            print('  NuScenes loaded.')
        except Exception as e:
            print(f'  WARNING: could not load NuScenes ({e}); map will be skipped.')

    scene_map = defaultdict(list)
    for gi, info in enumerate(infos):
        scene_map[info['scene_token']].append(gi)
    for sc in scene_map:
        scene_map[sc].sort(key=lambda i: infos[i]['timestamp'])
    all_scenes = list(scene_map.values())

    os.makedirs(args.output_dir, exist_ok=True)

    if args.scene_idx is not None:
        scene_list = [all_scenes[args.scene_idx]]
        fig, ax = plt.subplots(figsize=(10, 10), facecolor=_C_BG)
        gidxs  = scene_list[0]
        sc_tok = infos[gidxs[0]]['scene_token']
        _visualize_scene(infos, gidxs, ax, title=f'Scene {sc_tok[:8]}…', nusc=nusc)
        out = args.output or os.path.join(args.output_dir, 'occ_viz.png')
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f'Saved → {out}')
        plt.close(fig)
        return

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

    top_interp  = sorted(all_scenes, key=_score_interp,  reverse=True)[:args.num_scenes]
    top_extrap  = sorted(all_scenes, key=_score_extrap,  reverse=True)[:args.num_scenes]
    top_invalid = sorted(all_scenes, key=_score_invalid, reverse=True)[:args.num_scenes]

    def _save_grid(scene_list, out_path, suptitle, show_forecast=False):
        ncols = min(3, len(scene_list))
        nrows = (len(scene_list) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 7*nrows), facecolor=_C_BG)
        axes = np.array(axes).flatten()
        for i, gidxs in enumerate(scene_list):
            sc_tok = infos[gidxs[0]]['scene_token']
            _visualize_scene(infos, gidxs, axes[i],
                             title=f'Scene {sc_tok[:8]}…',
                             show_forecast=show_forecast, nusc=nusc)
        for j in range(len(scene_list), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(suptitle, color='#111111', fontsize=11, y=1.01)
        fig.subplots_adjust(hspace=0.35, wspace=0.25)
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
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
    # ------------------------------------------------------------------
    # Single-track grids
    # ------------------------------------------------------------------
    print('Collecting per-instance track data …')
    all_tracks    = _collect_all_tracks(infos, all_scenes)
    extrap_tracks = [t for t in all_tracks if t['n_extrap'] > 0]
    interp_tracks = [t for t in all_tracks if t['n_interp'] > 0]

    _save_single_track_grid(
        extrap_tracks,
        os.path.join(args.output_dir, 'occ_viz_extrap_single.png'),
        'Top 12 extrapolated tracks — most distance covered during extrapolation',
        score_fn=lambda t: t['extrap_dist'],
        nusc=nusc,
    )
    _save_single_track_grid(
        interp_tracks,
        os.path.join(args.output_dir, 'occ_viz_interp_single.png'),
        'Top 12 interpolated tracks — most distance covered during interpolation',
        score_fn=lambda t: t['interp_dist'],
        nusc=nusc,
    )
    _save_single_track_grid(
        extrap_tracks,
        os.path.join(args.output_dir, 'occ_viz_extrap_single_mismatch.png'),
        'Top 12 extrapolated tracks — largest extrapolation-vs-CV position mismatch',
        score_fn=lambda t: t['max_cv_ml_diff'],
        show_cv=True,
        nusc=nusc,
    )
    _save_ego_dist_hist(
        infos,
        os.path.join(args.output_dir, 'occ_viz_ego_dist.png'),
    )


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
    p_conv.add_argument('--input',  default='data/infos/nuscenes_infos_val.pkl',
                        help='Single input pkl (overrides --data-dir loop)')
    p_conv.add_argument('--output', default='data/infos/nuscenes_infos_val_occ.pkl',
                        help='Single output pkl (required when --input is set)')
    p_conv.add_argument('--predictions', default='data/occlusions/fmae_nuscenes_v1trainval_3class_sw_inference.npz',
                        help='Path to UniTraj inference NPZ for ML-based '
                             'extrapolation; falls back to CV when no match')
    p_conv.add_argument('--nuscenes-dataroot', default='data/nuscenes',
                        dest='nuscenes_dataroot',
                        help='nuScenes dataset root (contains v1.0-trainval/); '
                             'required when --predictions is used to map '
                             'integer instance indices to token strings')
    p_conv.add_argument('--no-extrapolate', action='store_true',
                        help='Disable forward extrapolation')
    p_conv.add_argument('--max-extrap-frames', type=int, default=9999,
                        help='Max frames to extrapolate forward per instance. '
                             'Default 9999 is effectively unlimited — the loop '
                             'always stops when the instance reappears or the '
                             'scene ends. Lower this to restrict to the ML '
                             'prediction horizon (e.g. 12 at 2 Hz).')
    p_conv.add_argument('--unitraj-dt', type=float, default=0.1,
                        dest='unitraj_dt',
                        help='Seconds per UniTraj prediction step. '
                             '0.1 for 10 Hz models (default), 0.5 for 2 Hz models.')
    p_conv.add_argument('--cv-stationary-thr', type=float, default=0.3,
                        dest='cv_stationary_thr',
                        help='Speed threshold (m/s) below which the CV fallback '
                             'uses constant position instead of constant velocity '
                             '(default: 0.3).')
    # CA / CTR motion-model fitting
    p_conv.add_argument('--min-ca-history', type=int, default=4,
                        dest='min_ca_history',
                        help='Minimum consecutive observed frames required to fit '
                             'a CA/CTR model (default: 3).')
    p_conv.add_argument('--ca-noise-thr', type=float, default=0.5,
                        dest='ca_noise_thr',
                        help='Minimum |acceleration| (m/s²) to apply CA model; '
                             'below this the track is treated as constant velocity '
                             '(default: 0.5).')
    p_conv.add_argument('--ca-max-thr', type=float, default=6.0,
                        dest='ca_max_thr',
                        help='Maximum |acceleration| (m/s²) accepted as realistic; '
                             'above this CA is rejected (default: 5.0).')
    p_conv.add_argument('--ca-consistency-thr', type=float, default=0.5,
                        dest='ca_consistency_thr',
                        help='Maximum standard deviation of per-interval acceleration '
                             '(m/s²) for the CA model to be accepted (default: 0.5).')
    p_conv.add_argument('--omega-noise-thr', type=float, default=0.07,
                        dest='omega_noise_thr',
                        help='Minimum |turn rate| (rad/s) to apply CTR model; '
                             'below this the track is treated as straight '
                             '(default: 0.07 ≈ 4 deg/s).')
    p_conv.add_argument('--omega-max-thr', type=float, default=0.6,
                        dest='omega_max_thr',
                        help='Maximum |turn rate| (rad/s) accepted as realistic; '
                             'above this CTR is rejected (default: 0.6 ≈ 34 deg/s).')
    p_conv.add_argument('--omega-consistency-thr', type=float, default=0.12,
                        dest='omega_consistency_thr',
                        help='Maximum standard deviation of per-interval turn rate '
                             '(rad/s) for the CTR model to be accepted (default: 0.12).')
    p_conv.add_argument('--max-dist', type=float, default=60.0,
                        dest='max_dist',
                        help='Two-level ego-distance filter applied after conversion: '
                             'drops instances with no original obs within this range '
                             'and removes per-frame boxes beyond it. '
                             'Set to 0 or a negative value to disable (default: 60.0).')

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
    p_viz.add_argument('--nuscenes-dataroot', default='data/nuscenes',
                       dest='nuscenes_dataroot',
                       help='nuScenes dataset root (e.g. /data/sets/nuscenes). '
                            'When provided, drivable-area map is rendered behind '
                            'each BEV plot using nusc.explorer.render_ego_centric_map.')
    p_viz.add_argument('--nuscenes-version', default='v1.0-trainval',
                       dest='nuscenes_version',
                       help='nuScenes version string (default: v1.0-trainval)')

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
