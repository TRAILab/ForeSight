"""CTRA interpolation baseline on the nuScenes val set.

For each interval Δt ∈ {0.5, 1.0, ..., 6.0}s, subsample every annotated agent
track at Δt and reconstruct the inner native-rate (0.5 s) positions/yaws using
a constant-acceleration + constant-turn-rate (CTRA) model. Per segment we
derive (a, ω) from the two endpoint GT speeds and yaws, so velocity and yaw
are continuous at every keyframe by construction (no jumps from either side).

Metrics, computed at the native-rate ticks that fall strictly between
keyframes:
  - L2 position error                  (mean, P90)
  - BEV IoU (oriented, GT W,L)         (mean, P10 = worst-decile IoU)
  - Hit rate (per-track ADE ≤ thr m)   for thr ∈ {2.0, 1.0}

A track = (instance, scene) sequence of consecutive annotated samples.
Δt = 0.5 s has no inner ticks (it's the native rate) — reported as N/A.

Usage:
    python tools/ctra_interp_baseline.py \
        --dataroot /data/sets/nuscenes \
        --version v1.0-trainval \
        --eval-set val \
        --output-md reports/ctra_interp_baseline.md
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from nuscenes import NuScenes
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
from shapely.geometry import Polygon


DT_NATIVE = 0.5  # nuScenes annotation rate (s)
HIT_THRESHOLDS_M = (2.0, 1.0)


# --- Geometry --------------------------------------------------------------


def wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def quat_to_yaw(rotation: List[float]) -> float:
    q = Quaternion(rotation)
    # heading = rotation of +x axis projected onto BEV
    v = q.rotation_matrix[:, 0]
    return float(np.arctan2(v[1], v[0]))


def box_corners(xy: np.ndarray, yaw: float, length: float, width: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    half = np.array([
        [ length / 2,  width / 2],
        [ length / 2, -width / 2],
        [-length / 2, -width / 2],
        [-length / 2,  width / 2],
    ])
    return xy[None, :] + half @ R.T


def bev_iou(pred_xy, pred_yaw, gt_xy, gt_yaw, w: float, l: float) -> float:
    if not (np.isfinite(pred_yaw) and np.isfinite(gt_yaw)):
        return float('nan')
    p = Polygon(box_corners(pred_xy, pred_yaw, l, w))
    g = Polygon(box_corners(gt_xy, gt_yaw, l, w))
    if not p.is_valid or not g.is_valid:
        return 0.0
    u = p.union(g).area
    return p.intersection(g).area / u if u > 1e-6 else 0.0


# --- Track building --------------------------------------------------------


def build_tracks(nusc: NuScenes, eval_set: str) -> List[dict]:
    """Build per-(instance, scene) tracks of consecutive annotated samples.

    Returns a list of dicts with arrays of length T (native 0.5s ticks):
        xy    (T, 2)   global frame
        yaw   (T,)     global frame, rad
        speed (T,)     m/s, from box_velocity with finite-diff fallback
        size  (3,)     W, L, H (constant per instance)
        category str
    """
    val_scenes = set(create_splits_scenes()[eval_set])

    # Map scene_token -> ordered sample tokens
    scene_samples: Dict[str, List[str]] = {}
    for scene in nusc.scene:
        if scene['name'] not in val_scenes:
            continue
        toks: List[str] = []
        st = scene['first_sample_token']
        while st:
            toks.append(st)
            st = nusc.get('sample', st)['next']
        scene_samples[scene['token']] = toks

    sample_to_pos: Dict[str, Tuple[str, int]] = {}
    for stok, toks in scene_samples.items():
        for i, t in enumerate(toks):
            sample_to_pos[t] = (stok, i)

    # Walk each instance's annotation chain, group annotations by scene.
    inst_scene_anns: Dict[Tuple[str, str], List[Tuple[int, str]]] = defaultdict(list)
    for inst in nusc.instance:
        ann_tok = inst['first_annotation_token']
        while ann_tok:
            ann = nusc.get('sample_annotation', ann_tok)
            spos = sample_to_pos.get(ann['sample_token'])
            if spos is not None:
                inst_scene_anns[(inst['token'], spos[0])].append((spos[1], ann_tok))
            ann_tok = ann['next']

    tracks: List[dict] = []
    for (inst_tok, scene_tok), items in inst_scene_anns.items():
        items.sort(key=lambda p: p[0])
        # Split into runs of consecutive scene-sample indices
        run: List[Tuple[int, str]] = []
        for spos, atok in items:
            if run and spos != run[-1][0] + 1:
                _emit_track(nusc, run, scene_tok, tracks)
                run = []
            run.append((spos, atok))
        if run:
            _emit_track(nusc, run, scene_tok, tracks)
    return tracks


def _emit_track(nusc: NuScenes, run, scene_tok, tracks):
    if len(run) < 2:
        return
    ann_toks = [a for _, a in run]
    first = nusc.get('sample_annotation', ann_toks[0])
    cat = category_to_detection_name(first['category_name'])
    if cat is None:
        return
    size = np.asarray(first['size'], dtype=np.float64)  # W, L, H

    T = len(ann_toks)
    xy = np.zeros((T, 2), dtype=np.float64)
    yaw = np.zeros(T, dtype=np.float64)
    speed = np.full(T, np.nan, dtype=np.float64)

    for i, atok in enumerate(ann_toks):
        ann = nusc.get('sample_annotation', atok)
        xy[i] = ann['translation'][:2]
        yaw[i] = quat_to_yaw(ann['rotation'])
        v = nusc.box_velocity(atok)  # (3,) global; NaN at endpoints
        if np.all(np.isfinite(v[:2])):
            speed[i] = float(np.linalg.norm(v[:2]))

    # Finite-difference fallback for NaN speeds.
    bad = ~np.isfinite(speed)
    if bad.any():
        dxy = np.diff(xy, axis=0) / DT_NATIVE
        sp_fd = np.linalg.norm(dxy, axis=1)  # length T-1, between sample i and i+1
        for i in np.where(bad)[0]:
            if i == 0:
                speed[i] = sp_fd[0]
            elif i == T - 1:
                speed[i] = sp_fd[-1]
            else:
                speed[i] = 0.5 * (sp_fd[i - 1] + sp_fd[i])

    tracks.append({
        'xy': xy,
        'yaw': yaw,
        'speed': speed,
        'size': size,
        'category': cat,
        'scene': scene_tok,
    })


# --- CTRA integration ------------------------------------------------------


def ctra_inner_states(
    x0: float, y0: float, yaw0: float, v0: float,
    a: float, omega: float, dt_total: float, n_inner: int,
    n_sub: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate CTRA from (x0,y0,yaw0,v0) with constant a, omega over [0, dt_total].

    Returns positions at the n_inner interior native-rate timestamps
    t_k = (k+1) * dt_total / (n_inner+1), k = 0..n_inner-1.

    Uses midpoint integration with n_sub substeps per inner interval.
    Yaw and speed evolve linearly in time; only the position needs integration.
    """
    if n_inner <= 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    n_steps = (n_inner + 1) * n_sub
    h = dt_total / n_steps
    x, y, yaw, v = x0, y0, yaw0, v0
    xs = np.empty(n_inner)
    ys = np.empty(n_inner)
    yaws = np.empty(n_inner)
    inner_idx = 0
    for step in range(1, n_steps + 1):
        v_mid = v + 0.5 * a * h
        yaw_mid = yaw + 0.5 * omega * h
        x += v_mid * np.cos(yaw_mid) * h
        y += v_mid * np.sin(yaw_mid) * h
        v += a * h
        yaw += omega * h
        if step % n_sub == 0 and inner_idx < n_inner:
            xs[inner_idx] = x
            ys[inner_idx] = y
            yaws[inner_idx] = yaw
            inner_idx += 1
    return xs, ys, yaws


# --- Per-Δt evaluation -----------------------------------------------------


def evaluate_interval(tracks: List[dict], k: int, want_iou: bool):
    """k = Δt / DT_NATIVE (positive int). Returns dict of aggregated metrics."""
    dt_seg = k * DT_NATIVE
    l2_all: List[float] = []
    iou_all: List[float] = []
    ade_per_track: List[float] = []

    if k == 1:
        return {
            'k': k, 'dt': dt_seg,
            'n_ticks': 0, 'n_tracks': 0,
            'l2_mean': float('nan'), 'l2_p90': float('nan'),
            'iou_mean': float('nan'), 'iou_p10': float('nan'),
            'hit_2m': float('nan'), 'hit_1m': float('nan'),
        }

    for tr in tracks:
        T = tr['xy'].shape[0]
        if T < k + 1:
            continue
        # Keyframe indices at 0, k, 2k, ..., last_kf <= T-1
        n_kf = (T - 1) // k + 1
        if n_kf < 2:
            continue
        kf_idx = np.arange(n_kf) * k  # length n_kf, last <= T-1

        track_l2: List[float] = []
        size_w, size_l = float(tr['size'][0]), float(tr['size'][1])
        xy = tr['xy']; yaw = tr['yaw']; sp = tr['speed']

        for seg in range(n_kf - 1):
            i0 = kf_idx[seg]
            i1 = kf_idx[seg + 1]
            n_inner = i1 - i0 - 1  # = k - 1
            if n_inner <= 0:
                continue
            yaw0 = yaw[i0]
            yaw1 = yaw[i1]
            v0 = sp[i0]
            v1 = sp[i1]
            omega = wrap_pi(np.array([yaw1 - yaw0]))[0] / dt_seg
            a = (v1 - v0) / dt_seg
            xs, ys, yaws_pred = ctra_inner_states(
                xy[i0, 0], xy[i0, 1], yaw0, v0, a, omega, dt_seg, n_inner,
            )
            gt_xy = xy[i0 + 1:i1]
            gt_yaw = yaw[i0 + 1:i1]
            d = np.sqrt((xs - gt_xy[:, 0]) ** 2 + (ys - gt_xy[:, 1]) ** 2)
            for j in range(n_inner):
                l2_all.append(float(d[j]))
                track_l2.append(float(d[j]))
                if want_iou:
                    iou_all.append(bev_iou(
                        np.array([xs[j], ys[j]]), float(yaws_pred[j]),
                        gt_xy[j], float(gt_yaw[j]),
                        size_w, size_l,
                    ))

        if track_l2:
            ade_per_track.append(float(np.mean(track_l2)))

    if not l2_all:
        return {
            'k': k, 'dt': dt_seg,
            'n_ticks': 0, 'n_tracks': 0,
            'l2_mean': float('nan'), 'l2_p90': float('nan'),
            'iou_mean': float('nan'), 'iou_p10': float('nan'),
            'hit_2m': float('nan'), 'hit_1m': float('nan'),
        }
    l2_arr = np.asarray(l2_all)
    ade_arr = np.asarray(ade_per_track)
    out = {
        'k': k, 'dt': dt_seg,
        'n_ticks': int(l2_arr.size), 'n_tracks': int(ade_arr.size),
        'l2_mean': float(l2_arr.mean()),
        'l2_p90': float(np.percentile(l2_arr, 90)),
        'hit_2m': float((ade_arr <= 2.0).mean()),
        'hit_1m': float((ade_arr <= 1.0).mean()),
    }
    if want_iou and iou_all:
        iou_arr = np.asarray(iou_all)
        iou_arr = iou_arr[np.isfinite(iou_arr)]
        out['iou_mean'] = float(iou_arr.mean())
        out['iou_p10'] = float(np.percentile(iou_arr, 10))
    else:
        out['iou_mean'] = float('nan')
        out['iou_p10'] = float('nan')
    return out


# --- Reporting -------------------------------------------------------------


def render_markdown(rows: List[dict]) -> str:
    hdr = (
        '| Δt (s) | #ticks | #tracks | L2 mean | L2 P90 | IoU mean | IoU P10 '
        '| Hit≤2m | Hit≤1m |\n'
        '|-------:|------:|--------:|--------:|------:|--------:|-------:|'
        '------:|------:|\n'
    )
    def fmt(v, prec=3):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return 'n/a'
        return f'{v:.{prec}f}'
    lines = [hdr.rstrip()]
    for r in rows:
        lines.append(
            f"| {r['dt']:.1f} | {r['n_ticks']} | {r['n_tracks']} "
            f"| {fmt(r['l2_mean'])} | {fmt(r['l2_p90'])} "
            f"| {fmt(r['iou_mean'])} | {fmt(r['iou_p10'])} "
            f"| {fmt(r['hit_2m'])} | {fmt(r['hit_1m'])} |"
        )
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', default='/data/sets/nuscenes')
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--eval-set', default='val')
    ap.add_argument('--output-md', default=None)
    ap.add_argument('--output-json', default=None)
    ap.add_argument('--max-scenes', type=int, default=0,
                    help='Cap on # val scenes (0 = all). Use for quick smoke tests.')
    ap.add_argument('--no-iou', action='store_true',
                    help='Skip BEV IoU computation (much faster).')
    ap.add_argument('--intervals', type=float, nargs='+',
                    default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
                             4.5, 5.0, 5.5, 6.0],
                    help='Δt values (s). Must be multiples of 0.5.')
    args = ap.parse_args()

    print(f'[ctra] Loading NuScenes {args.version} from {args.dataroot} ...')
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    print(f'[ctra] Building tracks for split {args.eval_set} ...')
    tracks = build_tracks(nusc, args.eval_set)
    print(f'[ctra]   {len(tracks)} tracks, '
          f'{sum(t["xy"].shape[0] for t in tracks)} total native ticks')

    if args.max_scenes > 0:
        keep = set(sorted({t['scene'] for t in tracks})[:args.max_scenes])
        tracks = [t for t in tracks if t['scene'] in keep]
        print(f'[ctra]   capped to {args.max_scenes} scenes → {len(tracks)} tracks')

    rows = []
    for dt in args.intervals:
        k = int(round(dt / DT_NATIVE))
        if abs(k * DT_NATIVE - dt) > 1e-6:
            print(f'[ctra] Skipping non-multiple-of-0.5 Δt={dt}')
            continue
        print(f'[ctra] Δt={dt:.1f}s (k={k}) ...', flush=True)
        rows.append(evaluate_interval(tracks, k, want_iou=not args.no_iou))

    md = render_markdown(rows)
    print()
    print(md)

    if args.output_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_md)) or '.', exist_ok=True)
        with open(args.output_md, 'w') as f:
            f.write(f'# CTRA interpolation baseline ({args.eval_set} split)\n\n')
            f.write('Velocity and yaw continuous at every keyframe '
                    '(both segments use GT endpoint values); positions '
                    'integrated from CTRA between keyframes. L2 is per-tick; '
                    'IoU uses the GT W,L for both boxes; hit-rate uses '
                    'per-track ADE.\n\n')
            f.write(md)
        print(f'[ctra] wrote {args.output_md}')
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(rows, f, indent=2)
        print(f'[ctra] wrote {args.output_json}')


if __name__ == '__main__':
    main()
