#!/usr/bin/env python3
"""ML + heuristics pipeline trajectory error analysis against GT future trajectory labels.

For every observed agent box the script:
  1. Looks up the UniTraj ML prediction for that instance at that moment (if any).
  2. For each GT future horizon k:
       - Uses ML prediction if the corresponding UniTraj step is in [0, 58]
       - Falls back to CP / CA / CTR / CV from the box's kinematic state otherwise
  3. Records the L2 (Euclidean) error at each future timestep where
     gt_agent_fut_masks == 1.

The full pipeline mirrors the forward-extrapolation logic in
nuscenes_occlusion_converter.py Pass 2, evaluating the same prediction
strategy on every original (non-synthetic) annotated box rather than only on
the track's final observation.

Outputs (same structure as eval_cv_trajectory_error.py)
-------
  • Per-timestep statistics table  (N, mean, median, std, RMSE, P25/P75/P90/P95)
  • ADE / FDE summary
  • Per-class ADE / FDE table
  • Histogram grid, box-plot, ADE curve, CDF curves, per-class ADE bar chart
  • Speed-bucket and acceleration-based plots
  • Feature correlation analysis
  • Matching-distance model fitting
  • Method breakdown bar chart  (ML / CP / CV / CA / CTR / CATR counts)

Usage
-----
    python tools/eval_pipeline_trajectory_error.py \\
        --pkl data/infos/nuscenes_infos_val.pkl \\
        --npz /path/to/unitraj_preds.npz \\
        --nuscenes-dataroot /data/sets/nuscenes \\
        --output-dir vis/pipeline_traj_error

    # Vehicles only, skip 0-sensor-point boxes
    python tools/eval_pipeline_trajectory_error.py \\
        --pkl data/infos/nuscenes_infos_val.pkl \\
        --npz /path/to/unitraj_preds.npz \\
        --nuscenes-dataroot /data/sets/nuscenes \\
        --classes car truck bus trailer \\
        --valid-only \\
        --output-dir vis/pipeline_traj_error_vehicles

    # Without ML predictions — pure heuristic (CP/CA/CTR/CV) evaluation
    python tools/eval_pipeline_trajectory_error.py \\
        --pkl data/infos/nuscenes_infos_val.pkl \\
        --output-dir vis/pipeline_traj_error_heuristic
"""

import argparse
import json
import os
import pickle
import numpy as np
from collections import defaultdict


# ===========================================================================
# Kinematic helpers  (copied from nuscenes_occlusion_converter.py)
# ===========================================================================

def _normalize_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _integrate_catr(x0, y0, yaw0, v0, omega, a, t, clamp_v=False):
    """Vectorised Riemann-sum integration of CATR kinematics from 0 to t."""
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


# ===========================================================================
# Coordinate helpers  (copied from nuscenes_occlusion_converter.py)
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
    """Convert a lidar-frame box + lidar-frame velocity to a global-frame state."""
    R, t = _lidar2global_RT(info)
    p_g = R @ np.array([box7[0], box7[1], box7[2]]) + t
    yaw_offset = np.arctan2(R[1, 0], R[0, 0])
    yaw_g = _normalize_angle(float(box7[6]) + yaw_offset)
    if np.isnan(vel2[0]) or np.isnan(vel2[1]):
        vx_g, vy_g = 0.0, 0.0
    else:
        v_g = R @ np.array([float(vel2[0]), float(vel2[1]), 0.0])
        vx_g, vy_g = float(v_g[0]), float(v_g[1])
    return (float(p_g[0]), float(p_g[1]), float(p_g[2]),
            float(box7[3]), float(box7[4]), float(box7[5]),
            yaw_g, vx_g, vy_g)


# ===========================================================================
# ML-prediction helpers  (copied from nuscenes_occlusion_converter.py)
# ===========================================================================

def _load_ml_predictions(pred_path):
    """Load a UniTraj inference NPZ and build an instance-token index.

    Returns
    -------
    preds  : ndarray (N, 6, 60, 2)
    worlds : ndarray (N, 10)  — [:2] global XY origin, [6] global heading
    lookup : dict  str → list[int]
    """
    npz    = np.load(pred_path, allow_pickle=True)
    preds  = npz['predictions']                            # (N, 6, 60, 2)
    meta   = npz['metadata'].item()
    worlds = meta['center_objects_world'].astype(np.float64)  # (N, 10)
    ids    = meta['center_objects_id']                     # (N,) str
    lookup = defaultdict(list)
    for i, inst in enumerate(ids):
        lookup[str(inst)].append(i)
    return preds, worlds, lookup


def _find_pred_index(inst_id, x_g, y_g, worlds, lookup, pos_tol=2.0):
    """Return the NPZ row index closest to (x_g, y_g) for this instance."""
    cands = lookup.get(str(inst_id), [])
    if not cands:
        return None
    dists = [np.hypot(worlds[ci, 0] - x_g, worlds[ci, 1] - y_g) for ci in cands]
    best  = int(np.argmin(dists))
    return cands[best] if dists[best] < pos_tol else None


def _pred_global_xy(ml_preds_i, world, step):
    """Convert agent-centric prediction at *step* to global (x, y).

    ml_preds_i : (6, 60, 2) — mode 0 used.
    world      : (10,) — world[:2] = global origin, world[6] = global heading.
    """
    if not (0 <= step < ml_preds_i.shape[1]):
        return None
    pred_ac = ml_preds_i[0, step]  # mode 0
    theta   = float(world[6])
    c, s    = np.cos(theta), np.sin(theta)
    return (float(c * pred_ac[0] - s * pred_ac[1] + world[0]),
            float(s * pred_ac[0] + c * pred_ac[1] + world[1]))


def _dt_to_step(dt_seconds, unitraj_dt):
    """Seconds offset from prediction moment → 0-based UniTraj step index."""
    return int(round(dt_seconds / unitraj_dt)) - 1


def _build_instance_token_map(nuscenes_dataroot, version='v1.0-trainval'):
    """Return a dict mapping integer nuScenes instance indices → token strings."""
    path = os.path.join(nuscenes_dataroot, version, 'instance.json')
    with open(path) as fh:
        instances = json.load(fh)
    return {i: rec['token'] for i, rec in enumerate(instances)}


# ===========================================================================
# Motion-model fitting  (copied from nuscenes_occlusion_converter.py)
# ===========================================================================

def _fit_motion_model(appearances, infos, min_history,
                      ca_noise_thr, ca_max_thr, ca_consistency_thr,
                      omega_noise_thr, omega_max_thr, omega_consistency_thr):
    """Fit scalar acceleration and turn-rate from trailing consecutive run."""
    run = [appearances[-1]]
    for i in range(len(appearances) - 2, -1, -1):
        if appearances[i + 1][0] - appearances[i][0] == 1:
            run.insert(0, appearances[i])
        else:
            break
    if len(run) < min_history:
        return 0.0, 0.0

    states = []
    for _, gi, bi in run:
        info = infos[gi]
        sg   = _box_to_global(info['gt_boxes'][bi], info['gt_velocity'][bi], info)
        spd  = float(np.hypot(sg[7], sg[8]))
        states.append((info['timestamp'] * 1e-6, spd, float(sg[6])))

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
# Acceleration pre-computation  (adapted from eval_cv_trajectory_error.py)
# ===========================================================================

def _compute_instance_accelerations(infos):
    """Compute per-box acceleration magnitudes via central differences in global frame."""
    from pyquaternion import Quaternion

    scene_map = defaultdict(list)
    for gi, info in enumerate(infos):
        scene_map[info['scene_token']].append(gi)
    for sc in scene_map:
        scene_map[sc].sort(key=lambda i: infos[i]['timestamp'])

    accel_map = {}

    for _, gidxs in scene_map.items():
        inst_tracks = defaultdict(list)
        for fp, gi in enumerate(gidxs):
            info = infos[gi]
            t_s  = info['timestamp'] * 1e-6
            R_l2e = Quaternion(info['lidar2ego_rotation']).rotation_matrix
            R_e2g = Quaternion(info['ego2global_rotation']).rotation_matrix
            R     = R_e2g @ R_l2e
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
                    vx, vy, t, vx1, vy1, t1 = vx1, vy1, t1, vx, vy, t
                    dt = t1 - t
                else:
                    _, _, _, vx0, vy0, t0 = track[k - 1]
                    _, _, _, vx1, vy1, t1 = track[k + 1]
                    vx, vy, t, vx1, vy1, t1 = vx0, vy0, t0, vx1, vy1, t1
                    dt = t1 - t
                if np.isnan(vx1) or np.isnan(vy1) or dt <= 0:
                    accel_map[(gi, bi)] = float('nan')
                else:
                    accel_map[(gi, bi)] = float(np.hypot((vx1 - vx) / dt,
                                                          (vy1 - vy) / dt))
    return accel_map


# ===========================================================================
# Pipeline constants
# ===========================================================================

# Step 59 (the final UniTraj step) snaps close to zero — discarded.
_ML_LAST = 58

_ML_CLASSES = {
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'vehicle.emergency.ambulance', 'vehicle.emergency.police',
    'pedestrian',
    'human.pedestrian.stroller', 'human.pedestrian.personal_mobility',
    'human.pedestrian.construction_worker', 'human.pedestrian.police_officer',
    'motorcycle', 'bicycle',
}
_MOTORIZED_CLASSES = {
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'vehicle.emergency.ambulance', 'vehicle.emergency.police',
    'motorcycle',
}
_VEHICLE_CLASSES = {
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'vehicle.emergency.ambulance', 'vehicle.emergency.police',
}


# ===========================================================================
# Pipeline error collection
# ===========================================================================

def collect_pipeline_errors(
        infos,
        preds, worlds, ml_lookup,
        inst_token_map,
        dt=0.5,
        unitraj_dt=0.1,
        include_synthetic=False,
        valid_only=False,
        classes=None,
        cv_stationary_thr=0.3,
        min_ca_history=4,
        ca_noise_thr=0.5,
        ca_max_thr=6.0,
        ca_consistency_thr=0.5,
        omega_noise_thr=0.07,
        omega_max_thr=0.6,
        omega_consistency_thr=0.12):
    """Collect per-timestep pipeline L2 errors and per-class breakdowns.

    For each observed box, looks up the UniTraj ML prediction (if available
    and passing the sanity check) and uses ML steps 0–58, falling back to
    CP / CA / CTR / CV beyond that or when no ML match exists.

    Parameters
    ----------
    preds, worlds, ml_lookup : outputs of _load_ml_predictions, or all None
                               to evaluate pure-heuristic mode (no ML).
    inst_token_map           : int index → token string; empty dict disables ML.

    Returns
    -------
    errors_per_step : list[list[float]]
    class_errors    : dict[str, list[list[float]]]
    vel_errors      : list[tuple]   — (speed, accel, t_k, err)
    vel_classes     : list[str]     — class name per sample (parallel to vel_errors)
    T               : int
    method_counter  : dict[str, int]  — {ML, CP, CV, CA, CTR, CATR} counts
    """
    has_synthetic = 'is_interpolated' in infos[0]

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

    use_ml = (preds is not None and ml_lookup is not None
              and worlds is not None and bool(inst_token_map))

    # ------------------------------------------------------------------
    # Pre-compute per-scene original instance tracks for CA/CTR fitting.
    # For each appearance (frame_pos, gi, bi) we fit the CA/CTR model
    # from the trailing consecutive run of observations up to that frame.
    # ------------------------------------------------------------------
    print('  Pre-computing instance tracks and CA/CTR models …')
    scene_to_indices = defaultdict(list)
    for gi, info in enumerate(infos):
        scene_to_indices[info['scene_token']].append(gi)
    for sc in scene_to_indices:
        scene_to_indices[sc].sort(key=lambda i: infos[i]['timestamp'])

    # (gi, bi) → (ca_accel, ca_omega) fitted from history up to that box
    catr_fits = {}
    for _, global_indices in scene_to_indices.items():
        orig_tracks = defaultdict(list)
        for frame_pos, gi in enumerate(global_indices):
            info = infos[gi]
            for bi, inst_ind in enumerate(info['instance_inds']):
                is_i = bool(info['is_interpolated'][bi]) if has_synthetic else False
                is_e = bool(info['is_extrapolated'][bi])  if has_synthetic else False
                if not is_i and not is_e:
                    orig_tracks[inst_ind].append((frame_pos, gi, bi))

        for inst_ind, appearances in orig_tracks.items():
            cls_name = infos[appearances[0][1]]['gt_names'][appearances[0][2]]
            for k in range(len(appearances)):
                gi_k = appearances[k][1]
                bi_k = appearances[k][2]
                if cls_name not in _MOTORIZED_CLASSES:
                    catr_fits[(gi_k, bi_k)] = (0.0, 0.0)
                    continue
                ca_a, ca_o = _fit_motion_model(
                    appearances[:k + 1], infos,
                    min_ca_history, ca_noise_thr, ca_max_thr, ca_consistency_thr,
                    omega_noise_thr, omega_max_thr, omega_consistency_thr)
                catr_fits[(gi_k, bi_k)] = (ca_a, ca_o)

    print('  Pre-computing instance accelerations …')
    accel_map = _compute_instance_accelerations(infos)

    # ------------------------------------------------------------------
    # Main collection loop
    # ------------------------------------------------------------------
    errors_per_step = [[] for _ in range(T)]
    class_errors    = defaultdict(lambda: [[] for _ in range(T)])
    vel_errors      = []
    vel_classes     = []
    method_counter  = defaultdict(int)

    n_skipped_nan_vel        = 0
    n_skipped_no_mask        = 0
    n_skipped_synthetic      = 0
    n_skipped_invalid        = 0
    n_skipped_class          = 0
    n_ml_matched             = 0
    n_ml_sanity_fail         = 0
    n_ml_stationary_override = 0
    n_total                  = 0

    for gi, info in enumerate(infos):
        n = len(info.get('gt_boxes', []))
        if n == 0:
            continue

        boxes     = info['gt_boxes']              # (N, 7) lidar frame
        vels      = info['gt_velocity']           # (N, 2) lidar frame
        names     = info['gt_names']
        valid     = info['valid_flag']
        fut_trajs = info['gt_agent_fut_trajs']    # (N, T, 2)
        fut_masks = info['gt_agent_fut_masks']    # (N, T)

        is_interp = (info['is_interpolated']
                     if has_synthetic else np.zeros(n, dtype=bool))
        is_extrap = (info['is_extrapolated']
                     if has_synthetic else np.zeros(n, dtype=bool))

        for bi in range(n):
            n_total += 1

            # ── filters ──────────────────────────────────────────────────
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

            mask = fut_masks[bi]
            if mask.sum() < 0.5:
                n_skipped_no_mask += 1
                continue

            # ── Global state of this box ──────────────────────────────────
            state_g = _box_to_global(boxes[bi], vels[bi], info)
            x_g, y_g, z_g, l, w, h, yaw_g, vx_g, vy_g = state_g
            v0 = float(np.hypot(vx_g, vy_g))

            # Rotation matrix (lidar → global), used to convert
            # global displacements back to lidar frame.
            R, _ = _lidar2global_RT(info)

            # ── ML prediction lookup ──────────────────────────────────────
            inst_ind = info['instance_inds'][bi]
            pred_row = None
            if use_ml and cls in _ML_CLASSES:
                inst_token = inst_token_map.get(inst_ind)
                if inst_token is not None:
                    pred_row = _find_pred_index(
                        inst_token, x_g, y_g, worlds, ml_lookup)

            # Sanity-check: reject ML if its 0.5 s point diverges too far
            # from the CV prediction.
            if pred_row is not None:
                step_check = _dt_to_step(0.5, unitraj_dt)
                if 0 <= step_check < 60:
                    xy_check = _pred_global_xy(
                        preds[pred_row], worlds[pred_row], step_check)
                    if xy_check is not None:
                        tol = 0.5 + 0.3 * v0
                        if np.hypot(xy_check[0] - (x_g + vx_g * 0.5),
                                    xy_check[1] - (y_g + vy_g * 0.5)) > tol:
                            pred_row = None
                            n_ml_sanity_fail += 1

            # Stationary vehicle override: UniTraj filters near-stationary
            # VEHICLE instances (total displacement < 2 m), so ML is unreliable.
            if (pred_row is not None
                    and v0 < cv_stationary_thr
                    and cls in _VEHICLE_CLASSES):
                pred_row = None
                n_ml_stationary_override += 1

            if pred_row is not None:
                n_ml_matched += 1

            # ── CA/CTR fit for this box ───────────────────────────────────
            ca_accel, ca_omega = catr_fits.get((gi, bi), (0.0, 0.0))

            # ── GT cumulative displacement (lidar frame) ──────────────────
            # fut_trajs[bi][k] = pos_{k+1} - pos_k   (lidar frame)
            # cumsum[k]        = pos_{k+1} - pos_0  = displacement at t_{k+1}
            gt_disp = np.cumsum(fut_trajs[bi].astype(np.float64), axis=0)  # (T, 2)

            speed = float(np.hypot(vx, vy))
            accel = accel_map.get((gi, bi), float('nan'))

            # ── Per-horizon predictions ───────────────────────────────────
            for k in range(T):
                if float(mask[k]) < 0.5:
                    continue

                t_k  = (k + 1) * dt
                step = _dt_to_step(t_k, unitraj_dt)
                xy   = (_pred_global_xy(preds[pred_row], worlds[pred_row], step)
                        if (pred_row is not None and 0 <= step <= _ML_LAST) else None)

                if xy is not None:
                    # Convert global ML position to lidar-frame displacement.
                    d_l      = R.T @ np.array([xy[0] - x_g, xy[1] - y_g, 0.0])
                    pred_dx  = float(d_l[0])
                    pred_dy  = float(d_l[1])
                    model    = 'ML'
                else:
                    # Heuristic fallback hierarchy.
                    if v0 < cv_stationary_thr:
                        pred_dx, pred_dy = 0.0, 0.0
                        model = 'CP'
                    elif ca_accel != 0.0 or ca_omega != 0.0:
                        x_ca, y_ca = _integrate_catr(
                            x_g, y_g, yaw_g, v0, ca_omega, ca_accel, t_k,
                            clamp_v=True)
                        d_l     = R.T @ np.array([x_ca - x_g, y_ca - y_g, 0.0])
                        pred_dx = float(d_l[0])
                        pred_dy = float(d_l[1])
                        if   ca_accel != 0.0 and ca_omega != 0.0:
                            model = 'CATR'
                        elif ca_accel != 0.0:
                            model = 'CA'
                        else:
                            model = 'CTR'
                    else:
                        pred_dx = vx * t_k   # vx/vy already in lidar frame
                        pred_dy = vy * t_k
                        model   = 'CV'

                err = float(np.hypot(pred_dx - gt_disp[k, 0],
                                     pred_dy - gt_disp[k, 1]))
                errors_per_step[k].append(err)
                class_errors[cls][k].append(err)
                vel_errors.append((speed, accel, t_k, err))
                vel_classes.append(cls)
                method_counter[model] += 1

    print(f'  Boxes processed           : {n_total:,}')
    if n_skipped_synthetic:
        print(f'  Skipped synthetic         : {n_skipped_synthetic:,}')
    if n_skipped_invalid:
        print(f'  Skipped invalid           : {n_skipped_invalid:,}')
    if n_skipped_class:
        print(f'  Skipped by class          : {n_skipped_class:,}')
    if n_skipped_nan_vel:
        print(f'  Skipped NaN vel           : {n_skipped_nan_vel:,}')
    if n_skipped_no_mask:
        print(f'  Skipped no mask           : {n_skipped_no_mask:,}')
    if use_ml:
        print(f'  ML matched (boxes)        : {n_ml_matched:,}')
        if n_ml_sanity_fail:
            print(f'  ML sanity-check fail      : {n_ml_sanity_fail:,}')
        if n_ml_stationary_override:
            print(f'  ML stationary override    : {n_ml_stationary_override:,}')
    total_preds = sum(method_counter.values())
    print(f'  Prediction method breakdown (box × horizon, total {total_preds:,}):')
    for m, cnt in sorted(method_counter.items(), key=lambda x: -x[1]):
        pct = 100.0 * cnt / total_preds if total_preds else 0.0
        print(f'    {m:<8}: {cnt:>12,}  ({pct:.1f}%)')
    n_used = sum(len(e) for e in errors_per_step)
    print(f'  Error samples             : {n_used:,}  '
          f'(step 0: {len(errors_per_step[0]):,}, '
          f'step {T-1}: {len(errors_per_step[T-1]):,})')

    return errors_per_step, dict(class_errors), vel_errors, vel_classes, T, dict(method_counter)


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
    print('Pipeline Trajectory Error (L2, metres)')
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

_BG  = 'white'
_FG  = '#222222'
_C1  = '#1f77b4'
_C2  = '#d6770a'
_C3  = '#d62728'
_C4  = '#9467bd'


def _style_ax(ax):
    ax.set_facecolor(_BG)
    for sp in ax.spines.values():
        sp.set_color('#bbbbbb')
    ax.tick_params(colors=_FG, labelsize=8)
    ax.grid(True, color='#cccccc', alpha=0.5, linewidth=0.5)


def plot_method_breakdown(method_counter, output_dir):
    """Bar chart of prediction-method usage counts."""
    import matplotlib.pyplot as plt

    if not method_counter:
        return

    method_colors = {
        'ML':   '#4fc3f7',
        'CP':   '#69f0ae',
        'CV':   '#ffb74d',
        'CA':   '#ef5350',
        'CTR':  '#ce93d8',
        'CATR': '#ff8a65',
    }
    methods = sorted(method_counter.keys())
    counts  = [method_counter[m] for m in methods]
    total   = sum(counts)
    pcts    = [100.0 * c / total for c in counts] if total else [0.0] * len(counts)

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.6), 4.5),
                           facecolor=_BG)
    _style_ax(ax)

    bars = ax.bar(methods, counts,
                  color=[method_colors.get(m, _C1) for m in methods],
                  edgecolor='none', alpha=0.85)
    ax.bar_label(bars,
                 labels=[f'{c:,}\n({p:.1f}%)' for c, p in zip(counts, pcts)],
                 color=_FG, fontsize=8.5, padding=4)

    ax.set_xlabel('Prediction model', color=_FG, fontsize=10)
    ax.set_ylabel('Count  (box × horizon)', color=_FG, fontsize=10)
    ax.set_title(f'Pipeline Prediction Method Breakdown  (total: {total:,})',
                 color=_FG, fontsize=11)
    ax.set_ylim(0, max(counts) * 1.22)

    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'pipeline_method_breakdown.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


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
        ax.set_title(f't = {t_k:.1f} s\n{subtitle}', color=_FG, fontsize=7.5, pad=4)
        ax.set_xlabel('L2 error (m)', color=_FG, fontsize=8)
        ax.set_ylabel('Count',        color=_FG, fontsize=8)
        ax.legend(fontsize=7, labelcolor=_FG,
                  facecolor='white', edgecolor='#cccccc', framealpha=0.9)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2.0))

    for j in range(T, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Pipeline Trajectory L2 Error Distribution per Future Horizon',
                 color=_FG, fontsize=12, y=1.01)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'pipeline_error_histograms.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


def plot_boxplots(errors_per_step, stats, output_dir, dt, max_err=20.0):
    """Box-plot of error distribution across horizons."""
    import matplotlib.pyplot as plt

    T        = len(errors_per_step)
    horizons = [(k + 1) * dt for k in range(T)]
    clip_at  = max_err * 1.5
    pairs    = [(h, np.clip(np.array(e), 0, clip_at))
                for h, e in zip(horizons, errors_per_step) if e]
    if not pairs:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(pairs) * 1.3), 5), facecolor=_BG)
    _style_ax(ax)

    labels = [f'{h:.1f}s' for h, _ in pairs]
    ax.boxplot(
        [d for _, d in pairs],
        labels=labels,
        patch_artist=True,
        medianprops=dict(color=_C3,  linewidth=2.0),
        whiskerprops=dict(color=_FG, linewidth=0.9),
        capprops=dict(color=_FG,     linewidth=0.9),
        flierprops=dict(marker='.', markerfacecolor=_C1,
                        markersize=2, alpha=0.25, linestyle='none'),
        boxprops=dict(facecolor='#ddeef8', edgecolor=_C1, linewidth=1.2),
    )
    means_v = [s['mean'] for s in stats if s['n'] > 0]
    ax.plot(range(1, len(means_v) + 1), means_v,
            color=_C2, marker='o', markersize=5,
            linewidth=1.8, linestyle='--', label='Mean', zorder=5)
    p90s = [s['p90'] for s in stats if s['n'] > 0]
    ax.plot(range(1, len(p90s) + 1), p90s,
            color=_C4, marker='^', markersize=4,
            linewidth=1.2, linestyle=':', label='P90', zorder=5)

    ax.set_xlabel('Future horizon', color=_FG, fontsize=10)
    ax.set_ylabel('L2 error (m)',   color=_FG, fontsize=10)
    ax.set_title('Pipeline Error Distribution vs. Future Horizon',
                 color=_FG, fontsize=11)
    ax.legend(fontsize=9, labelcolor=_FG,
              facecolor='white', edgecolor='#cccccc', framealpha=0.9)
    ax.set_ylim(0, clip_at * 0.6)
    ax.text(0.99, 0.98,
            f'(y-axis clipped to {clip_at * 0.6:.0f} m for readability)',
            transform=ax.transAxes, ha='right', va='top',
            color=_FG, fontsize=7, alpha=0.55)

    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'pipeline_error_boxplot.png')
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
    ax.fill_between(horizons, p25s, p75s,
                    color=_C1, alpha=0.18, label='IQR (25–75th pct)')
    ax.fill_between(horizons, np.maximum(0, means - stds), means + stds,
                    color=_C2, alpha=0.16, label='Mean ± σ')
    ax.plot(horizons, means,   color=_C2, linewidth=2.0,
            marker='o', markersize=5, label='Mean')
    ax.plot(horizons, medians, color=_C3, linewidth=1.6,
            linestyle='--', marker='s', markersize=4, label='Median')
    ax.plot(horizons, p90s,    color=_C4, linewidth=1.2,
            linestyle=':', marker='^', markersize=4, label='P90')

    ax.set_xlabel('Future horizon (s)', color=_FG, fontsize=10)
    ax.set_ylabel('L2 error (m)',       color=_FG, fontsize=10)
    ax.set_title('Pipeline Trajectory Error vs. Horizon  (ADE curve)',
                 color=_FG, fontsize=11)
    ax.legend(fontsize=9, labelcolor=_FG,
              facecolor='white', edgecolor='#cccccc', framealpha=0.9)
    ax.set_xlim(0, horizons[-1] + dt * 0.6)
    ax.set_ylim(0)
    ax.xaxis.set_major_locator(
        __import__('matplotlib').ticker.MultipleLocator(dt))

    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'pipeline_error_ade_curve.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


def plot_cdf(errors_per_step, stats, output_dir, dt, x_max=15.0):
    """CDF per horizon on one axes."""
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
    ax.set_xlabel('L2 error (m)',        color=_FG, fontsize=10)
    ax.set_ylabel('Cumulative fraction', color=_FG, fontsize=10)
    ax.set_title('CDF of Pipeline Trajectory Error per Horizon', color=_FG, fontsize=11)
    T = len(valid_steps)
    ax.legend(fontsize=8, labelcolor=_FG,
              facecolor='white', edgecolor='#cccccc', framealpha=0.9,
              ncol=max(1, T // 4))

    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'pipeline_error_cdf.png')
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

    x     = np.arange(len(classes))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 1.3), 5), facecolor=_BG)
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
    ax.set_title('Pipeline ADE / FDE by Agent Class', color=_FG, fontsize=11)
    ax.legend(fontsize=9, labelcolor=_FG,
              facecolor='white', edgecolor='#cccccc', framealpha=0.9)
    ax.set_ylim(0)

    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'pipeline_error_by_class.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


# ===========================================================================
# Velocity / acceleration based plots
# ===========================================================================

_SPEED_BUCKETS = [
    (0,   1,   '0–1 m/s',  '#90caf9'),
    (1,   3,   '1–3 m/s',  '#4fc3f7'),
    (3,   6,   '3–6 m/s',  '#69f0ae'),
    (6,  10,   '6–10 m/s', '#ffb74d'),
    (10, 999,  '≥10 m/s',  '#ef5350'),
]


def _binned_stats(xs, ys, bin_edges):
    """Return (bin_centres, mean, median, p25, p75, p90, counts) for binned data."""
    centres, means, medians, p25s, p75s, p90s, counts = [], [], [], [], [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (xs >= lo) & (xs < hi)
        vals = ys[mask]
        if len(vals) < 5:
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


def _bucket_series(vel_errors):
    arr      = np.array(vel_errors)
    speeds   = arr[:, 0]
    tks      = arr[:, 2]
    errs     = arr[:, 3]
    horizons = sorted(set(tks.tolist()))
    series   = []
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
    """ADE curve broken out by speed bucket."""
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
    ov_t = np.array([s['horizon'] for s in stats if s['n'] > 0])
    ov_m = np.array([s['mean']    for s in stats if s['n'] > 0])
    ax.plot(ov_t, ov_m, color='white', linewidth=1.6, linestyle='--',
            marker='s', markersize=3.5, alpha=0.70, label='Overall mean')
    ax.set_xlabel('Future horizon t (s)', color=_FG, fontsize=10)
    ax.set_ylabel('Mean L2 error (m)',    color=_FG, fontsize=10)
    ax.set_title('Pipeline Error vs. Horizon  (by speed bucket)',
                 color=_FG, fontsize=11)
    ax.set_xlim(0, max(horizons) + dt * 0.5)
    ax.set_ylim(0)
    ax.xaxis.set_major_locator(
        __import__('matplotlib').ticker.MultipleLocator(dt))
    ax.legend(fontsize=9, labelcolor=_FG,
              facecolor='white', edgecolor='#cccccc', framealpha=0.9)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'pipeline_error_ade_curve_by_speed.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


def plot_error_vs_t2(vel_errors, stats, output_dir, dt):
    """Pipeline error vs t² — two panels."""
    import matplotlib.pyplot as plt

    if not vel_errors:
        return
    series, horizons = _bucket_series(vel_errors)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=_BG)

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
    ax.plot(ov_t2, ov_m,   color=_C2, linewidth=2.0, marker='o', markersize=5, label='Mean')
    ax.plot(ov_t2, ov_med, color=_C3, linewidth=1.6, linestyle='--',
            marker='s', markersize=4, label='Median')
    ax.plot(ov_t2, ov_p90, color=_C4, linewidth=1.2, linestyle=':',
            marker='^', markersize=4, label='P90')
    if len(ov_t2) >= 2:
        coeffs = np.polyfit(ov_t2, ov_m, 1)
        t2_line = np.array([0, ov_t2[-1]])
        ax.plot(t2_line, np.polyval(coeffs, t2_line),
                color='white', linewidth=1.3, linestyle='--', alpha=0.50,
                label=f'Linear fit: {coeffs[0]:.3f}·t² + {coeffs[1]:.3f}')
    ax.set_xticks(ov_t2)
    ax.set_xticklabels([f'{t2:.2f}\n(t={np.sqrt(t2):.1f}s)' for t2 in ov_t2],
                       fontsize=7, color=_FG)
    ax.set_xlabel('t²  (s²)', color=_FG, fontsize=10)
    ax.set_ylabel('L2 error (m)', color=_FG, fontsize=10)
    ax.set_title('Pipeline Error vs. t²  (overall)', color=_FG, fontsize=10)
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.legend(fontsize=8, labelcolor=_FG, facecolor='white',
              edgecolor='#cccccc', framealpha=0.9)

    ax = axes[1]
    _style_ax(ax)
    for label, color, pts in series:
        ts2   = np.array([p[0] ** 2 for p in pts])
        means = np.array([p[1]      for p in pts])
        ax.plot(ts2, means, color=color, linewidth=1.8, marker='o', markersize=4, label=label)
    ax.set_xticks(ov_t2)
    ax.set_xticklabels([f'{t2:.2f}\n(t={np.sqrt(t2):.1f}s)' for t2 in ov_t2],
                       fontsize=7, color=_FG)
    ax.set_xlabel('t²  (s²)', color=_FG, fontsize=10)
    ax.set_ylabel('Mean L2 error (m)', color=_FG, fontsize=10)
    ax.set_title('Pipeline Error vs. t²  (by speed bucket)', color=_FG, fontsize=10)
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.legend(fontsize=9, labelcolor=_FG, facecolor='white',
              edgecolor='#cccccc', framealpha=0.9)

    fig.suptitle('Pipeline Trajectory Error vs. t²', color=_FG, fontsize=12, y=1.01)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'pipeline_error_vs_t2.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


def plot_error_vs_speed(vel_errors, output_dir, dt, speed_max=15.0, speed_bin=1.0):
    """Pipeline error vs agent speed."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if not vel_errors:
        return
    arr      = np.array(vel_errors)
    speeds   = arr[:, 0]
    tks      = arr[:, 2]
    errs     = arr[:, 3]
    horizons = sorted(set(tks.tolist()))
    bin_edges = np.arange(0, speed_max + speed_bin, speed_bin)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=_BG)
    _style_ax(ax)
    cmap = cm.get_cmap('plasma', len(horizons))
    for i, t_k in enumerate(horizons):
        mask_h = tks == t_k
        centres, means, _, p25s, p75s, _, _ = \
            _binned_stats(speeds[mask_h], errs[mask_h], bin_edges)
        if len(centres) == 0:
            continue
        color = cmap(i / max(len(horizons) - 1, 1))
        ax.plot(centres, means, color=color, linewidth=1.8,
                marker='o', markersize=3.5, label=f't={t_k:.1f}s')
        if i == 0 or i == len(horizons) - 1:
            ax.fill_between(centres, p25s, p75s, color=color, alpha=0.12)

    ax.set_xlabel('Agent speed |v| (m/s)', color=_FG, fontsize=10)
    ax.set_ylabel('Mean L2 error (m)',     color=_FG, fontsize=10)
    ax.set_title('Pipeline Trajectory Error vs. Agent Speed per Horizon',
                 color=_FG, fontsize=11)
    ax.set_xlim(0, speed_max)
    ax.set_ylim(0)
    ax.legend(fontsize=8, labelcolor=_FG, facecolor='white',
              edgecolor='#cccccc', framealpha=0.9,
              ncol=max(1, len(horizons) // 3))
    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'pipeline_error_vs_speed.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


def _two_panel_scatter(vel_errors, output_dir, x_arr_all, x_label, x_max, x_bin,
                       fname, suptitle):
    """Shared two-panel layout (per-horizon lines + collapsed with linear fit)."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    arr    = np.array(vel_errors)
    tks    = arr[:, 2]
    errs   = arr[:, 3]

    horizons  = sorted(set(tks.tolist()))
    bin_edges = np.arange(0, x_max + x_bin, x_bin)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=_BG)

    ax = axes[0]
    _style_ax(ax)
    cmap = cm.get_cmap('plasma', len(horizons))
    for i, t_k in enumerate(horizons):
        mask_h = tks == t_k
        xs = x_arr_all[mask_h]
        centres, means, _, _, _, _, _ = _binned_stats(xs, errs[mask_h], bin_edges)
        if len(centres) == 0:
            continue
        color = cmap(i / max(len(horizons) - 1, 1))
        ax.plot(centres, means, color=color, linewidth=1.6,
                marker='o', markersize=3.5, label=f't={t_k:.1f}s', alpha=0.9)
    ax.set_xlabel(x_label,             color=_FG, fontsize=10)
    ax.set_ylabel('Mean L2 error (m)', color=_FG, fontsize=10)
    ax.set_title(f'Pipeline Error vs. {x_label}\n(per horizon)', color=_FG, fontsize=10)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0)
    ax.legend(fontsize=8, labelcolor=_FG, facecolor='white',
              edgecolor='#cccccc', framealpha=0.9, ncol=max(1, len(horizons) // 3))

    ax = axes[1]
    _style_ax(ax)
    centres, means, medians, p25s, p75s, p90s, _ = \
        _binned_stats(x_arr_all, errs, bin_edges)
    if len(centres) > 0:
        ax.fill_between(centres, p25s, p75s, color=_C1, alpha=0.18, label='IQR (25–75th pct)')
        ax.plot(centres, means,   color=_C2, linewidth=2.0, marker='o', markersize=4, label='Mean')
        ax.plot(centres, medians, color=_C3, linewidth=1.6, linestyle='--',
                marker='s', markersize=3.5, label='Median')
        ax.plot(centres, p90s,    color=_C4, linewidth=1.2, linestyle=':',
                marker='^', markersize=3.5, label='P90')
        in_range = x_arr_all <= x_max
        xf, yf = x_arr_all[in_range], errs[in_range]
        if len(xf) > 10:
            alpha_fit = float(np.dot(xf, yf) / np.dot(xf, xf))
            x_line = np.array([0, x_max])
            ax.plot(x_line, alpha_fit * x_line,
                    color='white', linewidth=1.4, linestyle='--', alpha=0.55,
                    label=f'Linear fit α={alpha_fit:.4f}')
    ax.set_xlabel(x_label,        color=_FG, fontsize=10)
    ax.set_ylabel('L2 error (m)', color=_FG, fontsize=10)
    ax.set_title(f'Pipeline Error vs. {x_label}\n(all horizons collapsed)',
                 color=_FG, fontsize=10)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0)
    ax.legend(fontsize=8, labelcolor=_FG, facecolor='white',
              edgecolor='#cccccc', framealpha=0.9)

    fig.suptitle(suptitle, color=_FG, fontsize=12, y=1.01)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, fname)
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


def plot_error_vs_vt(vel_errors, output_dir, dt, vt_max=30.0, vt_bin=2.0):
    if not vel_errors:
        return
    arr    = np.array(vel_errors)
    speeds = arr[:, 0]
    tks    = arr[:, 2]
    errs   = arr[:, 3]
    vt     = speeds * tks
    # Two-panel: per-horizon + collapsed (same structure, call the helper)
    _two_panel_scatter(vel_errors, output_dir, vt, '|v| · t  (m)',
                       vt_max, vt_bin, 'pipeline_error_vs_vt.png',
                       'Pipeline Error vs. Predicted Displacement  (|v| · t)')


def plot_error_vs_vt2(vel_errors, output_dir, dt, vt2_max=45.0, vt2_bin=3.0):
    if not vel_errors:
        return
    arr    = np.array(vel_errors)
    speeds = arr[:, 0]
    tks    = arr[:, 2]
    vt2    = speeds * tks ** 2
    _two_panel_scatter(vel_errors, output_dir, vt2, '|v| · t²  (m · s)',
                       vt2_max, vt2_bin, 'pipeline_error_vs_vt2.png',
                       'Pipeline Error vs.  |v| · t²')


def _accel_unpack(vel_errors):
    arr   = np.array(vel_errors)
    valid = ~np.isnan(arr[:, 1])
    return arr[valid, 1], arr[valid, 2], arr[valid, 3]


def plot_error_vs_accel(vel_errors, output_dir, dt,
                        accel_max=5.0, accel_bin=0.25):
    """Pipeline error vs acceleration magnitude, one line per horizon."""
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
    ax.set_title('Pipeline Error vs. Agent Acceleration per Horizon',
                 color=_FG, fontsize=11)
    ax.set_xlim(0, accel_max)
    ax.set_ylim(0)
    ax.legend(fontsize=8, labelcolor=_FG, facecolor='white',
              edgecolor='#cccccc', framealpha=0.9,
              ncol=max(1, len(horizons) // 3))
    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'pipeline_error_vs_accel.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


def plot_error_vs_at(vel_errors, output_dir, dt, at_max=10.0, at_bin=0.5):
    accels, tks, _ = _accel_unpack(vel_errors)
    if len(accels) == 0:
        return
    arr = np.array(vel_errors)
    valid = ~np.isnan(arr[:, 1])
    _two_panel_scatter(
        [v for v, ok in zip(vel_errors, valid) if ok],
        output_dir,
        accels * tks, '|a| · t  (m/s)',
        at_max, at_bin, 'pipeline_error_vs_at.png',
        'Pipeline Error vs.  |a| · t')


def plot_error_vs_at2(vel_errors, output_dir, dt, at2_max=15.0, at2_bin=0.75):
    accels, tks, _ = _accel_unpack(vel_errors)
    if len(accels) == 0:
        return
    arr = np.array(vel_errors)
    valid = ~np.isnan(arr[:, 1])
    _two_panel_scatter(
        [v for v, ok in zip(vel_errors, valid) if ok],
        output_dir,
        accels * tks ** 2, '|a| · t²  (m)',
        at2_max, at2_bin, 'pipeline_error_vs_at2.png',
        'Pipeline Error vs.  |a| · t²')


# ===========================================================================
# Feature correlation analysis  (same formulas as eval_cv)
# ===========================================================================

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
    from scipy.stats import pearsonr, spearmanr
    arr    = np.array(vel_errors)
    valid  = ~np.isnan(arr[:, 1])
    speeds = arr[valid, 0]
    accels = arr[valid, 1]
    tks    = arr[valid, 2]
    errs   = arr[valid, 3]

    overall = {}
    for name, fn in _FEATURE_DEFS:
        x = fn(speeds, accels, tks)
        pr, _ = pearsonr(x, errs)
        sr, _ = spearmanr(x, errs)
        overall[name] = dict(pearson_r=float(pr),  pearson_r2=float(pr ** 2),
                             spearman_r=float(sr), spearman_r2=float(sr ** 2))

    per_horizon = {}
    for t_k in sorted(set(tks.tolist())):
        mask = tks == t_k
        v_h, a_h, t_h, e_h = speeds[mask], accels[mask], tks[mask], errs[mask]
        if len(e_h) < 10:
            continue
        horizon_corr = {}
        for name, fn in _FEATURE_DEFS:
            x = fn(v_h, a_h, t_h)
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
    print()
    print('Feature correlation with Pipeline L2 error  (NaN-accel rows excluded)')
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
    import matplotlib.pyplot as plt
    names = [n for n, _ in _FEATURE_DEFS]
    pr    = [overall[n]['pearson_r']  for n in names]
    sr    = [overall[n]['spearman_r'] for n in names]
    order = sorted(range(len(names)), key=lambda i: -abs(pr[i]))
    names = [names[i] for i in order]
    pr    = [pr[i]    for i in order]
    sr    = [sr[i]    for i in order]
    x     = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(9, len(names) * 1.2), 5), facecolor=_BG)
    _style_ax(ax)
    bars_p = ax.bar(x - width / 2, pr, width, color=_C1, alpha=0.85,
                    label='Pearson r', edgecolor='none')
    bars_s = ax.bar(x + width / 2, sr, width, color=_C2, alpha=0.85,
                    label='Spearman ρ', edgecolor='none')
    ax.bar_label(bars_p, fmt='%.3f', color=_FG, fontsize=7, padding=3)
    ax.bar_label(bars_s, fmt='%.3f', color=_FG, fontsize=7, padding=3)
    ax.axhline(0, color='white', linewidth=0.6, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9, color=_FG)
    ax.set_ylabel('Correlation with L2 error', color=_FG, fontsize=10)
    ax.set_title('Feature Correlations with Pipeline Trajectory Error\n'
                 '(sorted by |Pearson r|, NaN-accel rows excluded)',
                 color=_FG, fontsize=11)
    ax.legend(fontsize=9, labelcolor=_FG, facecolor='white',
              edgecolor='#cccccc', framealpha=0.9)
    ax.set_ylim(-0.05, 1.05)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'pipeline_error_feature_correlations.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


def plot_correlation_per_horizon(per_horizon, output_dir, dt):
    import matplotlib.pyplot as plt
    if not per_horizon:
        return
    horizons = sorted(per_horizon.keys())
    names    = [n for n, _ in _FEATURE_DEFS]
    palette  = ['#4fc3f7', '#ffb74d', '#69f0ae', '#ef5350', '#ce93d8',
                '#fff176', '#80cbc4', '#ffcc02', '#ff8a65', '#a5d6a7', '#f48fb1']
    color_of = {n: palette[i % len(palette)] for i, n in enumerate(names)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=_BG)
    for ax_idx, metric in enumerate(['pearson_r', 'spearman_r']):
        ax = axes[ax_idx]
        _style_ax(ax)
        label_str = 'Pearson r' if metric == 'pearson_r' else 'Spearman ρ'
        for name in names:
            xs, ys = [], []
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
        ax.set_title(f'{label_str} with L2 error vs. horizon', color=_FG, fontsize=10)
        ax.set_xlim(0, max(horizons) + dt * 0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_locator(
            __import__('matplotlib').ticker.MultipleLocator(dt))
        ax.legend(fontsize=7.5, labelcolor=_FG, facecolor='white',
                  edgecolor='#cccccc', framealpha=0.9, ncol=2)

    fig.suptitle('Feature Correlations with Pipeline Error per Horizon\n'
                 '(t and t² omitted — constant within each horizon)',
                 color=_FG, fontsize=11, y=1.01)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out = os.path.join(output_dir, 'pipeline_error_correlations_per_horizon.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


# ===========================================================================
# Matching-distance model fitting
# ===========================================================================

# Candidate models: (name, feature_matrix_fn(v, a, t) → ndarray)
# All include a non-negative constant term c (minimum detection-noise floor).
# Coefficients are non-negative (L-BFGS-B with lower bound = 0).
_MODEL_DEFS = [
    # ── Single-feature baselines ──────────────────────────────────────────
    ('c + α·t',            lambda v, a, t: np.column_stack([np.ones_like(t), t])),
    ('c + α·t²',           lambda v, a, t: np.column_stack([np.ones_like(t), t ** 2])),
    ('c + α·v·t',          lambda v, a, t: np.column_stack([np.ones_like(t), v * t])),
    ('c + α·v·t²',         lambda v, a, t: np.column_stack([np.ones_like(t), v * t ** 2])),
    ('c + α·a·t',          lambda v, a, t: np.column_stack([np.ones_like(t), a * t])),
    ('c + α·a·t²',         lambda v, a, t: np.column_stack([np.ones_like(t), a * t ** 2])),
    ('c + α·v²·t²',        lambda v, a, t: np.column_stack([np.ones_like(t), v ** 2 * t ** 2])),
    ('c + α·a²·t²',        lambda v, a, t: np.column_stack([np.ones_like(t), a ** 2 * t ** 2])),
    # ── Two-feature combinations ──────────────────────────────────────────
    ('c + α·t + β·v·t',    lambda v, a, t: np.column_stack([np.ones_like(t), t, v * t])),
    ('c + α·v·t + β·t²',   lambda v, a, t: np.column_stack([np.ones_like(t), v * t, t ** 2])),
    ('c + α·v·t + β·a·t²', lambda v, a, t: np.column_stack([np.ones_like(t), v * t, a * t ** 2])),
    ('c + α·v·t + β·v·t²', lambda v, a, t: np.column_stack([np.ones_like(t), v * t, v * t ** 2])),
    ('c + α·t + β·v²·t²',  lambda v, a, t: np.column_stack([np.ones_like(t), t, v ** 2 * t ** 2])),
    ('c + α·t + β·a·t²',   lambda v, a, t: np.column_stack([np.ones_like(t), t, a * t ** 2])),
    ('c + α·a·t + β·v·t',  lambda v, a, t: np.column_stack([np.ones_like(t), a * t, v * t])),
    # ── Three-feature combinations ─────────────────────────────────────────
    ('c + α·t + β·a·t + γ·v·t',  lambda v, a, t: np.column_stack([np.ones_like(t), t, a * t, v * t])),
    ('c + α·t + β·a·t + γ·t²',   lambda v, a, t: np.column_stack([np.ones_like(t), t, a * t, t ** 2])),
    ('c + α·a·t + β·v·t + γ·t²', lambda v, a, t: np.column_stack([np.ones_like(t), a * t, v * t, t ** 2])),
    ('c + α·t + β·v·t + γ·a·t²', lambda v, a, t: np.column_stack([np.ones_like(t), t, v * t, a * t ** 2])),
]


def _format_model_equation(name, coeffs):
    """Substitute c/α/β/γ in a model-name string with numeric values."""
    syms = ['c', 'α', 'β', 'γ', 'δ']
    eq   = name
    for sym, val in zip(syms, coeffs):
        eq = eq.replace(sym, str(round(float(val), 4)))
    return eq


def _fit_quantile(X, y, quantile=0.90, max_iter=2000):
    """Non-negative quantile (pinball) regression via L-BFGS-B.

    Returns
    -------
    coeffs   : ndarray (n_params,)
    coverage : float  — fraction of samples with y <= X @ coeffs
    mae_pos  : float  — mean excess on samples above the fitted curve
    """
    from scipy.optimize import minimize

    n, p = X.shape

    def pinball(c):
        resid = y - X @ c
        loss  = np.where(resid >= 0, quantile * resid, (quantile - 1.0) * resid)
        return float(loss.mean())

    def pinball_grad(c):
        resid = y - X @ c
        w     = np.where(resid >= 0, -quantile, (1.0 - quantile))
        return (X.T @ w) / n

    res = minimize(pinball, np.ones(p) * 0.1, jac=pinball_grad, method='L-BFGS-B',
                   bounds=[(0, None)] * p,
                   options={'maxiter': max_iter, 'ftol': 1e-10})
    c        = res.x
    pred     = X @ c
    coverage = float(np.mean(y <= pred))
    pos      = y[y > pred]
    mae_pos  = float(pos.mean()) if len(pos) else 0.0
    return c, coverage, mae_pos


def fit_matching_distance_models(vel_errors, output_dir, dt,
                                 quantile=0.90, v_max=20.0, t_max=6.5,
                                 vel_classes=None):
    """Fit and compare candidate matching-distance models against pipeline errors.

    For each candidate model f(v, a, t; θ) the coefficients are found by
    minimising the pinball loss at *quantile* — the fitted curve bounds
    that fraction of pipeline errors from below.  The best-fit model and
    its coefficients are saved as ``matching_distance_model.npy`` and can
    be used directly as an adaptive matching threshold in downstream
    evaluation.

    If *vel_classes* is provided (parallel list of class strings, same length
    as *vel_errors*), additional per-class-group fits are performed for
    VEHICLE, CYCLIST, and PEDESTRIAN groups matching UniTraj categories.

    Outputs
    -------
    • Console table: model name, coverage, pseudo-R², coefficients
    • Plot: fitted curves vs empirical quantile per horizon (and by speed bucket)
    • ``matching_distance_model.npy``: best-model name + coefficients
    • Per-group npy + plots when vel_classes is supplied
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    arr    = np.array(vel_errors)
    valid  = ~np.isnan(arr[:, 1])      # drop NaN-accel rows
    speeds = arr[valid, 0]
    accels = arr[valid, 1]
    tks    = arr[valid, 2]
    errs   = arr[valid, 3]

    # Align vel_classes with the same valid / keep masks.
    cls_arr_filt = None
    if vel_classes is not None:
        cls_arr       = np.array(vel_classes)
        cls_arr_valid = cls_arr[valid]

    # Clip extreme outliers (>99th pct) so they don't dominate the fit.
    p99  = np.percentile(errs, 99)
    keep = (errs <= p99) & (speeds <= v_max) & (tks <= t_max)
    v, a, t, e = speeds[keep], accels[keep], tks[keep], errs[keep]

    if vel_classes is not None:
        cls_arr_filt = cls_arr_valid[keep]

    print(f'\n  Fitting matching-distance models  '
          f'(quantile={quantile:.0%}, N={keep.sum():,}) …')

    # Compute null-model pinball loss (predict the empirical quantile everywhere).
    null_pred  = np.full(len(e), np.quantile(e, quantile))
    null_resid = e - null_pred
    null_loss  = float(np.where(null_resid >= 0,
                                quantile * null_resid,
                                (quantile - 1) * null_resid).mean())

    results = []
    for name, feat_fn in _MODEL_DEFS:
        X = feat_fn(v, a, t)
        coeffs, coverage, mae_pos = _fit_quantile(X, e, quantile=quantile)
        model_resid = e - X @ coeffs
        model_loss  = float(np.where(model_resid >= 0,
                                     quantile * model_resid,
                                     (quantile - 1) * model_resid).mean())
        pseudo_r2 = 1.0 - model_loss / null_loss if null_loss > 0 else float('nan')
        results.append(dict(name=name, coeffs=coeffs, coverage=coverage,
                            mae_pos=mae_pos, pseudo_r2=pseudo_r2, feat_fn=feat_fn))

    # ── Console table ─────────────────────────────────────────────────────
    print()
    print(f'  Quantile ({quantile:.0%}) regression — matching distance models  '
          f'(pipeline errors)')
    print(f'  {"Model":<26}  {"Coverage":>8}  {"PseudoR²":>9}  Coefficients')
    print('  ' + '─' * 70)
    for r in sorted(results, key=lambda x: -x['pseudo_r2']):
        coeff_str = '  '.join(f'{c:.4f}' for c in r['coeffs'])
        print(f'  {r["name"]:<26}  {r["coverage"]:>7.1%}  {r["pseudo_r2"]:>9.4f}  '
              f'[{coeff_str}]')
    print()

    best = max(results, key=lambda x: x['pseudo_r2'])
    print(f'  Best model: {best["name"]}  (pseudo-R²={best["pseudo_r2"]:.4f})')
    coeff_str = '  '.join(f'{c:.4f}' for c in best['coeffs'])
    print(f'  Coefficients: [{coeff_str}]')
    print(f'  → d_match = max(d_min, {_format_model_equation(best["name"], best["coeffs"])})')

    # Save best-model coefficients.
    npy_path = os.path.join(output_dir, 'matching_distance_model.npy')
    np.save(npy_path, {'name': best['name'], 'coeffs': best['coeffs'],
                       'quantile': quantile})
    print(f'  Saved coefficients → {npy_path}')

    # ── Plots ──────────────────────────────────────────────────────────────
    horizons = sorted(set(tks.tolist()))
    emp_q    = np.array([
        float(np.quantile(errs[tks == t_k], quantile))
        if (tks == t_k).sum() > 0 else float('nan')
        for t_k in horizons
    ])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=_BG)

    # Left: all models vs empirical quantile at median-speed agent.
    ax = axes[0]
    _style_ax(ax)
    v_med = float(np.median(speeds))
    a_med = float(np.median(accels))
    t_arr = np.array(horizons)

    ax.plot(t_arr, emp_q, color='white', linewidth=2.2, linestyle='--',
            marker='o', markersize=5,
            label=f'Empirical P{quantile*100:.0f}', zorder=5)

    palette = ['#4fc3f7', '#ffb74d', '#69f0ae', '#ef5350', '#ce93d8',
               '#fff176', '#ff8a65']
    for idx, r in enumerate(results):
        X_line = r['feat_fn'](
            np.full_like(t_arr, v_med),
            np.full_like(t_arr, a_med),
            t_arr)
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
    ax.legend(fontsize=7.5, labelcolor=_FG, facecolor='white',
              edgecolor='#cccccc', framealpha=0.9, ncol=1)

    # Right: best model vs empirical quantile per speed bucket.
    ax = axes[1]
    _style_ax(ax)
    ax.plot(t_arr, emp_q, color='white', linewidth=2.0, linestyle='--',
            marker='o', markersize=5,
            label=f'Empirical P{quantile*100:.0f} (all)', zorder=5)

    for (lo, hi, label, color) in _SPEED_BUCKETS:
        v_rep = (lo + hi) / 2
        bucket_q    = []
        bucket_pred = []
        for t_k in horizons:
            mask = (tks == t_k) & (speeds >= lo) & (speeds < hi)
            bucket_q.append(
                float(np.quantile(errs[mask], quantile)) if mask.sum() >= 10
                else float('nan'))
            X1 = best['feat_fn'](
                np.array([v_rep]), np.array([a_med]), np.array([t_k]))
            bucket_pred.append(float(X1 @ best['coeffs']))

        bq = np.array(bucket_q)
        bp = np.array(bucket_pred)
        valid_b = ~np.isnan(bq)
        if valid_b.sum() < 2:
            continue
        ax.plot(t_arr[valid_b], bq[valid_b], color=color, linewidth=1.4,
                linestyle='--', marker='s', markersize=3.5, alpha=0.8)
        ax.plot(t_arr, bp, color=color, linewidth=1.8, linestyle='-', label=label)

    ax.set_xlabel('Future horizon t (s)', color=_FG, fontsize=10)
    ax.set_ylabel(f'Matching distance (m)  [P{quantile*100:.0f}]',
                  color=_FG, fontsize=10)
    ax.set_title(f'Best model ({best["name"]}) vs empirical P{quantile*100:.0f}\n'
                 'per speed bucket  (solid=model, dashed=empirical)',
                 color=_FG, fontsize=10)
    ax.set_xlim(0, max(horizons) + dt * 0.5)
    ax.set_ylim(0)
    ax.legend(fontsize=8, labelcolor=_FG, facecolor='white',
              edgecolor='#cccccc', framealpha=0.9)

    fig.suptitle(f'Pipeline Matching Distance Model Comparison  '
                 f'(quantile={quantile:.0%})',
                 color=_FG, fontsize=12, y=1.01)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'pipeline_matching_distance_model_fit.png')
    plt.savefig(out_path, dpi=140, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'  Saved → {out_path}')
    plt.close(fig)

    # ── Per-class-group fits ───────────────────────────────────────────────
    if cls_arr_filt is not None:
        _CLASS_GROUPS = [
            ('VEHICLE',    {'car', 'truck', 'bus', 'trailer', 'construction_vehicle'}),
            ('CYCLIST',    {'motorcycle', 'bicycle'}),
            ('PEDESTRIAN', {'pedestrian'}),
        ]

        for grp_name, grp_classes in _CLASS_GROUPS:
            mask_g = np.array([c in grp_classes for c in cls_arr_filt])
            n_g    = int(mask_g.sum())
            if n_g < 50:
                print(f'  Skipping {grp_name} per-group fit: only {n_g} samples.')
                continue

            v_g, a_g, t_g, e_g = v[mask_g], a[mask_g], t[mask_g], e[mask_g]

            # Null loss for this group.
            null_pred_g  = np.full(n_g, np.quantile(e_g, quantile))
            null_resid_g = e_g - null_pred_g
            null_loss_g  = float(np.where(null_resid_g >= 0,
                                          quantile * null_resid_g,
                                          (quantile - 1) * null_resid_g).mean())

            results_g = []
            for name, feat_fn in _MODEL_DEFS:
                X_g = feat_fn(v_g, a_g, t_g)
                coeffs_g, cov_g, mae_g = _fit_quantile(X_g, e_g, quantile=quantile)
                resid_g = e_g - X_g @ coeffs_g
                loss_g  = float(np.where(resid_g >= 0,
                                         quantile * resid_g,
                                         (quantile - 1) * resid_g).mean())
                pr2_g   = 1.0 - loss_g / null_loss_g if null_loss_g > 0 else float('nan')
                results_g.append(dict(name=name, coeffs=coeffs_g, coverage=cov_g,
                                      mae_pos=mae_g, pseudo_r2=pr2_g, feat_fn=feat_fn))

            best_g = max(results_g, key=lambda x: x['pseudo_r2'])

            print(f'\n  ── {grp_name} (N={n_g:,}) ──')
            print(f'  {"Model":<30}  {"Coverage":>8}  {"PseudoR²":>9}  Coefficients')
            print('  ' + '─' * 75)
            for r in sorted(results_g, key=lambda x: -x['pseudo_r2']):
                coeff_str = '  '.join(f'{c:.4f}' for c in r['coeffs'])
                print(f'  {r["name"]:<30}  {r["coverage"]:>7.1%}  '
                      f'{r["pseudo_r2"]:>9.4f}  [{coeff_str}]')
            print()
            coeff_str = '  '.join(f'{c:.4f}' for c in best_g['coeffs'])
            print(f'  Best ({grp_name}): {best_g["name"]}  '
                  f'(pseudo-R²={best_g["pseudo_r2"]:.4f})')
            print(f'  Coefficients: [{coeff_str}]')
            print(f'  → d_match = max(d_min, '
                  f'{_format_model_equation(best_g["name"], best_g["coeffs"])})')

            # Save per-group npy.
            npy_g = os.path.join(output_dir,
                                 f'matching_distance_model_{grp_name}.npy')
            np.save(npy_g, {'name': best_g['name'], 'coeffs': best_g['coeffs'],
                            'quantile': quantile, 'group': grp_name,
                            'group_classes': sorted(grp_classes)})
            print(f'  Saved → {npy_g}')

            # Per-group plot.
            horizons_g = sorted(set(t_g.tolist()))
            t_arr_g    = np.array(horizons_g)
            emp_q_g    = np.array([
                float(np.quantile(e_g[t_g == h], quantile))
                if (t_g == h).sum() > 0 else float('nan')
                for h in horizons_g
            ])
            v_med_g = float(np.median(v_g))
            a_med_g = float(np.median(a_g))

            fig_g, ax_g = plt.subplots(1, 1, figsize=(8, 5), facecolor=_BG)
            _style_ax(ax_g)
            ax_g.plot(t_arr_g, emp_q_g, color='white', linewidth=2.2,
                      linestyle='--', marker='o', markersize=5,
                      label=f'Empirical P{quantile*100:.0f}', zorder=5)

            for idx, r in enumerate(results_g):
                X_line_g = r['feat_fn'](
                    np.full_like(t_arr_g, v_med_g),
                    np.full_like(t_arr_g, a_med_g),
                    t_arr_g)
                pred_g = X_line_g @ r['coeffs']
                lw_g   = 2.2 if r['name'] == best_g['name'] else 1.2
                ls_g   = '-'  if r['name'] == best_g['name'] else ':'
                ax_g.plot(t_arr_g, pred_g,
                          color=palette[idx % len(palette)],
                          linewidth=lw_g, linestyle=ls_g,
                          label=f'{r["name"]}  (R²={r["pseudo_r2"]:.3f})')

            ax_g.set_xlabel('Future horizon t (s)', color=_FG, fontsize=10)
            ax_g.set_ylabel(f'Matching distance (m)  [P{quantile*100:.0f}]',
                            color=_FG, fontsize=10)
            ax_g.set_title(
                f'{grp_name} — Model fits vs empirical P{quantile*100:.0f}\n'
                f'(v={v_med_g:.1f} m/s, a={a_med_g:.2f} m/s², N={n_g:,})',
                color=_FG, fontsize=10)
            ax_g.set_xlim(0, max(horizons_g) + dt * 0.5)
            ax_g.set_ylim(0)
            ax_g.legend(fontsize=7.5, labelcolor=_FG, facecolor='white',
                        edgecolor='#cccccc', framealpha=0.9, ncol=1)
            fig_g.patch.set_facecolor(_BG)
            plt.tight_layout()
            out_g = os.path.join(
                output_dir,
                f'pipeline_matching_distance_{grp_name.lower()}.png')
            plt.savefig(out_g, dpi=140, bbox_inches='tight',
                        facecolor=fig_g.get_facecolor())
            print(f'  Saved → {out_g}')
            plt.close(fig_g)

    return best


# ===========================================================================
# Per-split runner
# ===========================================================================

def run_split(pkl_path, args, split_tag=''):
    tag = f'[{split_tag}] ' if split_tag else ''
    out = os.path.join(args.output_dir, split_tag) if split_tag else args.output_dir
    os.makedirs(out, exist_ok=True)

    print(f'\n{tag}Loading {pkl_path} …')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    infos    = data['infos']
    n_scenes = len({i['scene_token'] for i in infos})
    print(f'{tag}{len(infos):,} samples across {n_scenes} scenes')

    # Load ML predictions
    preds = worlds = ml_lookup = None
    inst_token_map = {}
    if args.npz:
        print(f'{tag}Loading ML predictions from {args.npz} …')
        preds, worlds, ml_lookup = _load_ml_predictions(args.npz)
        print(f'{tag}  {len(preds)} prediction entries loaded.')
        if args.nuscenes_dataroot:
            print(f'{tag}Building instance-token map from {args.nuscenes_dataroot} …')
            inst_token_map = _build_instance_token_map(
                args.nuscenes_dataroot, args.nuscenes_version)
            print(f'{tag}  {len(inst_token_map)} instance records indexed.')
        else:
            print(f'{tag}WARNING: --nuscenes-dataroot not set; '
                  'ML predictions will not be matched (all heuristic fallback).')
    else:
        print(f'{tag}No --npz provided — evaluating pure heuristic (CP/CA/CTR/CV).')

    print(f'{tag}Collecting pipeline errors …')
    errors_per_step, class_errors, vel_errors, vel_classes, T, method_counter = \
        collect_pipeline_errors(
            infos,
            preds, worlds, ml_lookup,
            inst_token_map,
            dt                   = args.dt,
            unitraj_dt           = args.unitraj_dt,
            include_synthetic    = args.include_synthetic,
            valid_only           = args.valid_only,
            classes              = args.classes,
            cv_stationary_thr    = args.cv_stationary_thr,
            min_ca_history       = args.min_ca_history,
            ca_noise_thr         = args.ca_noise_thr,
            ca_max_thr           = args.ca_max_thr,
            ca_consistency_thr   = args.ca_consistency_thr,
            omega_noise_thr      = args.omega_noise_thr,
            omega_max_thr        = args.omega_max_thr,
            omega_consistency_thr = args.omega_consistency_thr,
        )

    stats = compute_stats(errors_per_step, args.dt)
    print_stats(stats)
    print_class_stats(class_errors, args.dt)

    print(f'{tag}Generating plots …')
    plot_method_breakdown(method_counter, out)
    plot_histograms(errors_per_step, stats, out, args.dt, max_err=args.max_err_plot)
    plot_boxplots(errors_per_step, stats, out, args.dt, max_err=args.max_err_plot)
    plot_ade_curve(stats, out, args.dt)
    plot_cdf(errors_per_step, stats, out, args.dt, x_max=args.max_err_plot)
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
                                 quantile=args.match_quantile,
                                 vel_classes=vel_classes)

    print(f'{tag}Done.  Outputs in {out}/')


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    grp = ap.add_mutually_exclusive_group()
    grp.add_argument('--pkl', default='data/infos/nuscenes_infos_val.pkl',
                     help='Single input pkl file to evaluate')
    grp.add_argument('--data-dir', default='data/infos', metavar='DIR',
                     help='Directory containing nuscenes_infos_*.pkl files. '
                          'Processes train and val splits.')

    ap.add_argument('--npz', default='data/occlusions/fmae_nuscenes_v1trainval_3class_sw_inference.npz',
                    help='Path to UniTraj inference NPZ file with ML predictions. '
                         'Omit to evaluate pure heuristic (CP/CA/CTR/CV) mode.')
    ap.add_argument('--nuscenes-dataroot', default='data/nuscenes', dest='nuscenes_dataroot',
                    help='nuScenes dataset root (required to match ML predictions). '
                         'E.g. /data/sets/nuscenes')
    ap.add_argument('--nuscenes-version', default='v1.0-trainval',
                    dest='nuscenes_version',
                    help='nuScenes version string (default: v1.0-trainval)')
    ap.add_argument('--dt', type=float, default=0.5,
                    help='Seconds per future trajectory step (default: 0.5 — '
                         'nuScenes 2 Hz keyframe rate)')
    ap.add_argument('--unitraj-dt', type=float, default=0.1, dest='unitraj_dt',
                    help='Seconds per UniTraj prediction step '
                         '(default: 0.1 for 10 Hz models)')
    ap.add_argument('--include-synthetic', action='store_true',
                    help='Include is_interpolated / is_extrapolated boxes')
    ap.add_argument('--valid-only', action='store_true',
                    help='Only process boxes with valid_flag=True (≥1 sensor point)')
    ap.add_argument('--classes', nargs='+', default=None, metavar='CLS',
                    help='Agent class names to include (default: all)')
    ap.add_argument('--max-err-plot', type=float, default=20.0,
                    help='X-axis clip for histograms and box-plots (default: 20 m)')
    ap.add_argument('--match-quantile', type=float, default=0.90, metavar='Q',
                    dest='match_quantile',
                    help='Quantile to fit for matching-distance models '
                         '(default: 0.90 → 90%% of pipeline errors bounded)')
    ap.add_argument('--output-dir', default='vis/pipeline_traj_error',
                    help='Root directory for output plots '
                         '(default: vis/pipeline_traj_error)')

    # Heuristic thresholds (match nuscenes_occlusion_converter.py defaults)
    ap.add_argument('--cv-stationary-thr', type=float, default=0.3,
                    dest='cv_stationary_thr',
                    help='Speed (m/s) below which CP is used instead of CV/CA/CTR '
                         '(default: 0.3)')
    ap.add_argument('--min-ca-history', type=int, default=4,
                    dest='min_ca_history',
                    help='Minimum consecutive observed frames to fit CA/CTR '
                         '(default: 4)')
    ap.add_argument('--ca-noise-thr', type=float, default=0.5,
                    dest='ca_noise_thr',
                    help='Minimum |acceleration| (m/s²) to apply CA (default: 0.5)')
    ap.add_argument('--ca-max-thr', type=float, default=6.0,
                    dest='ca_max_thr',
                    help='Maximum |acceleration| (m/s²) for CA (default: 6.0)')
    ap.add_argument('--ca-consistency-thr', type=float, default=0.5,
                    dest='ca_consistency_thr',
                    help='Max std of per-interval acceleration for CA (default: 0.5)')
    ap.add_argument('--omega-noise-thr', type=float, default=0.07,
                    dest='omega_noise_thr',
                    help='Minimum |turn rate| (rad/s) to apply CTR (default: 0.07)')
    ap.add_argument('--omega-max-thr', type=float, default=0.6,
                    dest='omega_max_thr',
                    help='Maximum |turn rate| (rad/s) for CTR (default: 0.6)')
    ap.add_argument('--omega-consistency-thr', type=float, default=0.12,
                    dest='omega_consistency_thr',
                    help='Max std of per-interval turn rate for CTR (default: 0.12)')

    args = ap.parse_args()

    if args.pkl is None and args.data_dir is None:
        args.pkl = 'data/infos/nuscenes_infos_val.pkl'

    if args.pkl is not None:
        run_split(args.pkl, args)
    else:
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
