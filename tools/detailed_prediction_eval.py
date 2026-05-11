"""Detailed prediction evaluation against a motion_predictions.pkl dump.

Consumes the per-sample motion predictions dumped by
NuScenes3DDataset._dump_motion_predictions and produces a table broken down by
class group, forecast-horizon window, and GT trajectory motion behavior.

Metrics per row: (mu, P_90) for L2 and yaw err., (mu, P_10) for BEV IoU,
(<2m, <1m) hit-rate. All numbers are computed per (agent, valid tick) and
aggregated within the row's selection.

Usage:
    python tools/detailed_prediction_eval.py \
        --pred-pkl <path-to-motion_predictions.pkl> \
        --dataroot <nuscenes-root> \
        --version v1.0-trainval \
        --eval-set val \
        --output-tex <out.tex>
"""
import argparse
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from nuscenes import NuScenes
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
from shapely.geometry import Polygon


# --- Configuration ---------------------------------------------------------

# Class groups. Keep parity with motion_utils.motion_name_mapping lumps;
# two-wheelers stay in Vehicle.
CLASS_GROUPS: Dict[str, set] = {
    'Vehicle': {
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'motorcycle', 'bicycle',
    },
    'Pedestrian': {'pedestrian'},
    'Movable': {'barrier', 'traffic_cone'},
}

# Per-class match distance threshold (m) for greedy assignment in the
# global frame. Matches MotionEval (dist_th_tp = 2.0).
MATCH_DIST_TH = 2.0

# Future horizon: 6 s @ 0.5 s = 12 ticks.
FUT_TS = 12
DT = 0.5

# Forecast-horizon bins (tick index, inclusive start / exclusive end).
HORIZON_BINS: Dict[str, Tuple[int, int]] = {
    '0-2s': (0, 4),
    '2-4s': (4, 8),
    '4-6s': (8, 12),
}

# Motion-behavior bin thresholds, over GT trajectory only.
# Stationary: cumulative path length below STATIONARY_DISP_M.
# Turning: net heading change between first and last *motion-only* tangent
#   yaw exceeds TURNING_DHEAD_DEG. Standard motion-prediction benchmark
#   convention (Waymo Open Motion, Argoverse). Stationary ticks contribute
#   no yaw (box-yaw fallback disabled here), so an agent that drives straight
#   then rotates in place classifies Straight.
STATIONARY_DISP_M = 0.5
TURNING_DHEAD_DEG = 30.0

HIT_THRESHOLDS_M = (2.0, 1.0)

# Yaw computation: ticks whose available motion span yields < this many
# m/s of effective velocity are treated as stationary at that tick and
# fall back to the GT box yaw. 0.1 m/s ≈ slow shuffle; below it tangent
# direction is dominated by tracker noise.
YAW_V_THRESHOLD_MPS = 0.1


# --- Geometry helpers ------------------------------------------------------


def compute_traj_yaws(
    traj: np.ndarray,
    valid_mask: np.ndarray,
    box_center_xy: np.ndarray,
    box_yaw: float,
    v_threshold_mps: float = YAW_V_THRESHOLD_MPS,
    dt: float = DT,
) -> np.ndarray:
    """Per-tick yaw (rad) with cascading neighbor selection and stationary fallback.

    Rule:
      * Trajectories are augmented with ``box_center_xy`` at virtual tick -1
        (the current-frame agent position) so tick 0 has a natural "previous"
        reference equal to the GT box.
      * For each valid tick, prefer the immediate central difference (t-1, t+1).
        If only one side is available, use that side + current. If neither is
        available at distance 1, extend outward (k=2, 3, ...). When both sides
        are available at the same distance k, use the chord from t-k to t+k.
      * Each cascade step requires the displacement magnitude to exceed
        ``v_threshold_mps * span``, where ``span`` is the time interval covered
        by the chord. This keeps the stationary-fallback threshold expressed in
        velocity units (m/s), independent of k.
      * If no k yields a chord above threshold, the tick falls back to
        ``box_yaw`` — i.e., treat it as near-stationary and use the box pose.
      * Ticks where ``valid_mask`` is False get NaN.
    """
    T = traj.shape[0]
    if T == 0:
        return np.zeros(0)
    aug = np.vstack([np.asarray(box_center_xy, dtype=np.float64)[None, :2], traj])
    aug_valid = np.concatenate([[True], valid_mask])
    yaws = np.full(T, float(box_yaw), dtype=np.float64)

    for t in range(T):
        if not valid_mask[t]:
            yaws[t] = np.nan
            continue
        i = t + 1  # index in ``aug``
        k_max = max(i, len(aug) - i)
        for k in range(1, k_max + 1):
            left_idx = i - k
            right_idx = i + k
            left_ok = (left_idx >= 0) and aug_valid[left_idx]
            right_ok = (right_idx < len(aug)) and aug_valid[right_idx]
            if left_ok and right_ok:
                d = aug[right_idx] - aug[left_idx]
                span = 2 * k * dt
            elif left_ok:
                d = aug[i] - aug[left_idx]
                span = k * dt
            elif right_ok:
                d = aug[right_idx] - aug[i]
                span = k * dt
            else:
                continue
            if np.linalg.norm(d) > v_threshold_mps * span:
                yaws[t] = np.arctan2(d[1], d[0])
                break
        # If no chord exceeded the velocity threshold, ``box_yaw`` is retained
        # via the initial fill — i.e., near-stationary at every probed k.
    return yaws


def yaw_err_deg(pred_yaw: np.ndarray, gt_yaw: np.ndarray) -> np.ndarray:
    """Absolute angular difference in degrees, wrapped to [0, 180]."""
    err = (pred_yaw - gt_yaw + np.pi) % (2 * np.pi) - np.pi
    return np.degrees(np.abs(err))


def box_corners(xy: np.ndarray, yaw: float, length: float, width: float) -> np.ndarray:
    """Return the four BEV corners (4, 2) of a box at xy with given yaw.

    length is along the heading axis, width is lateral. nuScenes box "size"
    is (W, L, H), so caller should pass (size[1], size[0]) for (L, W).
    """
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    half = np.array([
        [ length / 2,  width / 2],
        [ length / 2, -width / 2],
        [-length / 2, -width / 2],
        [-length / 2,  width / 2],
    ])
    return xy[None, :] + half @ R.T


def bev_iou(pred_xy, pred_yaw, gt_xy, gt_yaw, size_wlh) -> float:
    """BEV (2D ground-plane) IoU between two boxes that share size_wlh.

    Both boxes use the GT box's W and L (we have no predicted size).
    """
    if not np.isfinite(pred_yaw) or not np.isfinite(gt_yaw):
        return np.nan
    w, l = float(size_wlh[0]), float(size_wlh[1])
    pred_poly = Polygon(box_corners(pred_xy, pred_yaw, l, w))
    gt_poly = Polygon(box_corners(gt_xy, gt_yaw, l, w))
    if not pred_poly.is_valid or not gt_poly.is_valid:
        return 0.0
    inter = pred_poly.intersection(gt_poly).area
    union = pred_poly.union(gt_poly).area
    return inter / union if union > 1e-6 else 0.0


# --- GT loading ------------------------------------------------------------


def split_sample_tokens(nusc: NuScenes, eval_set: str) -> List[str]:
    splits = create_splits_scenes()
    tokens = []
    for s in nusc.sample:
        scene = nusc.get('scene', s['scene_token'])
        if scene['name'] in splits[eval_set]:
            tokens.append(s['token'])
    return tokens


def load_gt(nusc: NuScenes, eval_set: str, seconds: int = 6) -> Dict[str, List[dict]]:
    """Returns sample_token -> list of GT records, each:
        {'detection_name': str (raw 10-class, no motion_name_mapping),
         'translation_global': (3,) np.ndarray,        # current-frame, global
         'size_wlh': (3,) np.ndarray,                  # W, L, H
         'box_lidar_center': (3,) np.ndarray,          # current-frame, lidar
         'box_lidar_yaw': float,                       # current-frame, lidar
         'gt_traj_lidar': (Tv, 2) np.ndarray,          # future XY in lidar frame
         'instance_token': str}
    """
    helper = PredictHelper(nusc)
    tokens = split_sample_tokens(nusc, eval_set)
    out: Dict[str, List[dict]] = {}
    for sample_token in tokens:
        sample = nusc.get('sample', sample_token)
        gts: List[dict] = []
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            det_name = category_to_detection_name(ann['category_name'])
            if det_name is None:
                continue
            fut_local = helper.get_future_for_agent(
                ann['instance_token'], sample_token, seconds=seconds,
                in_agent_frame=True,
            )
            if fut_local.shape[0] == 0:
                fut_lidar = np.zeros((0, 2))
                box_center = np.array(ann['translation'])  # fallback
                box_yaw = 0.0
            else:
                _, boxes, _ = nusc.get_sample_data(
                    sample['data']['LIDAR_TOP'],
                    selected_anntokens=[ann_token],
                )
                box = boxes[0]
                rot = Quaternion(matrix=box.rotation_matrix)
                fut_lidar = convert_local_coords_to_global(fut_local, box.center, rot)
                box_center = box.center
                box_yaw = float(np.arctan2(
                    box.rotation_matrix[1, 0], box.rotation_matrix[0, 0]
                ))
            gts.append({
                'detection_name': det_name,
                'translation_global': np.array(ann['translation'], dtype=np.float64),
                'size_wlh': np.array(ann['size'], dtype=np.float64),
                'box_lidar_center': np.array(box_center, dtype=np.float64),
                'box_lidar_yaw': box_yaw,
                'gt_traj_lidar': np.asarray(fut_lidar, dtype=np.float64),
                'instance_token': ann['instance_token'],
            })
        out[sample_token] = gts
    return out


# --- Prediction loading ---------------------------------------------------


def load_pred_pkl(path: str) -> Dict[str, List[dict]]:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['results']


def top1_traj(pred_anno: dict) -> np.ndarray:
    """Return the top-1 (highest score) predicted trajectory in lidar frame, (T, 2)."""
    trajs = np.asarray(pred_anno['trajs'], dtype=np.float64)  # (M, T, 2)
    scores = pred_anno.get('trajs_score', None)
    if scores is None or len(scores) != trajs.shape[0]:
        idx = 0
    else:
        idx = int(np.argmax(np.asarray(scores)))
    return trajs[idx]


# --- Matching --------------------------------------------------------------


def class_lump(detection_name: str) -> Optional[str]:
    for grp, members in CLASS_GROUPS.items():
        if detection_name in members:
            return grp
    return None


def match_sample(
    preds: List[dict], gts: List[dict], class_lumped: bool = True,
) -> List[Tuple[dict, dict]]:
    """Greedy match by detection_score (desc) in the global frame.

    A pred is matched to the closest unmatched GT of the same class lump if
    the distance is below MATCH_DIST_TH. Returns the matched (pred, gt) pairs.
    """
    # Sort preds by detection_score, descending. Predonly oracles use uniform
    # high scores; the resulting tie-break is just list order, which still
    # produces a valid matching for our purposes (unique-GT constraint holds).
    sortind = sorted(
        range(len(preds)),
        key=lambda i: -float(preds[i].get('detection_score', 0.0)),
    )
    taken = set()
    matches = []
    for i in sortind:
        pred = preds[i]
        pred_lump = class_lump(pred['detection_name'])
        if pred_lump is None:
            continue
        pred_xy = np.asarray(pred['translation'][:2], dtype=np.float64)
        best_gi, best_d = None, np.inf
        for gi, gt in enumerate(gts):
            if gi in taken:
                continue
            gt_lump = class_lump(gt['detection_name'])
            if class_lumped and gt_lump != pred_lump:
                continue
            d = float(np.linalg.norm(pred_xy - gt['translation_global'][:2]))
            if d < best_d:
                best_d = d
                best_gi = gi
        if best_gi is not None and best_d < MATCH_DIST_TH:
            taken.add(best_gi)
            matches.append((pred, gts[best_gi]))
    return matches


# --- Metric assembly ------------------------------------------------------


def classify_motion(gt_traj: np.ndarray) -> str:
    """Bin a GT trajectory into Stationary / Straight / Turning.

    Industry-standard convention (Waymo / Argoverse): heading delta between
    the first and last motion-derived tangent yaw of the trajectory.
    Stationary ticks are excluded from the yaw computation (no box-yaw
    fallback here) so an agent that drives forward then rotates in place
    classifies Straight — its motion-derived tangent never changes.
    """
    if gt_traj.shape[0] < 2:
        return 'Stationary'
    disp = float(np.linalg.norm(np.diff(gt_traj, axis=0), axis=1).sum())
    if disp < STATIONARY_DISP_M:
        return 'Stationary'
    # Motion-only tangent yaws: pass NaN as the "box yaw" so the cascade
    # marks stationary ticks NaN rather than filling with body yaw.
    yaws = compute_traj_yaws(
        gt_traj, np.ones(gt_traj.shape[0], dtype=bool),
        gt_traj[0], float('nan'),
    )
    valid = np.isfinite(yaws)
    if valid.sum() < 2:
        return 'Straight'
    idx = np.where(valid)[0]
    d = (yaws[idx[-1]] - yaws[idx[0]] + np.pi) % (2 * np.pi) - np.pi
    if abs(np.degrees(d)) > TURNING_DHEAD_DEG:
        return 'Turning'
    return 'Straight'


def compute_per_agent_arrays(matches_by_sample, fut_ts: int = FUT_TS):
    """Flatten matches into per-(agent, tick) numpy arrays plus per-agent metadata.

    Returns dict with:
        l2:        (N_agent, T)
        yaw_err:   (N_agent, T)  -- nan where tangent is undefined either side
        bev_iou:   (N_agent, T)
        valid:     (N_agent, T)  bool -- True where GT has a future point
        class_lump:    list of str length N_agent
        motion_bin:    list of str length N_agent
    """
    l2_rows, yaw_rows, iou_rows, valid_rows = [], [], [], []
    lump_list, motion_list = [], []
    for sample_token, matches in matches_by_sample.items():
        for pred, gt in matches:
            pred_traj = top1_traj(pred)  # (T, 2) lidar frame, absolute XY
            gt_traj = gt['gt_traj_lidar']  # (Tv, 2)
            Tv = gt_traj.shape[0]
            if Tv == 0:
                continue
            T = min(fut_ts, pred_traj.shape[0], Tv)
            if T == 0:
                continue

            l2_full = np.full(fut_ts, np.nan)
            yaw_full = np.full(fut_ts, np.nan)
            iou_full = np.full(fut_ts, np.nan)
            valid_full = np.zeros(fut_ts, dtype=bool)

            pred_xy = pred_traj[:T]
            gt_xy = gt_traj[:T]
            valid_T = np.ones(T, dtype=bool)
            box_xy = gt['box_lidar_center'][:2]
            box_yaw = gt['box_lidar_yaw']
            pred_yaw = compute_traj_yaws(pred_xy, valid_T, box_xy, box_yaw)
            gt_yaw_arr = compute_traj_yaws(gt_xy, valid_T, box_xy, box_yaw)
            l2 = np.linalg.norm(pred_xy - gt_xy, axis=1)
            yerr = yaw_err_deg(pred_yaw, gt_yaw_arr)
            ious = np.array([
                bev_iou(pred_xy[t], pred_yaw[t], gt_xy[t], gt_yaw_arr[t], gt['size_wlh'])
                for t in range(T)
            ])

            l2_full[:T] = l2
            yaw_full[:T] = yerr
            iou_full[:T] = ious
            valid_full[:T] = True

            l2_rows.append(l2_full)
            yaw_rows.append(yaw_full)
            iou_rows.append(iou_full)
            valid_rows.append(valid_full)
            lump_list.append(class_lump(gt['detection_name']))
            motion_list.append(classify_motion(gt_traj))

    return {
        'l2': np.stack(l2_rows) if l2_rows else np.zeros((0, fut_ts)),
        'yaw_err': np.stack(yaw_rows) if yaw_rows else np.zeros((0, fut_ts)),
        'bev_iou': np.stack(iou_rows) if iou_rows else np.zeros((0, fut_ts)),
        'valid': np.stack(valid_rows) if valid_rows else np.zeros((0, fut_ts), dtype=bool),
        'class_lump': lump_list,
        'motion_bin': motion_list,
    }


def aggregate_row(arrays, agent_mask: np.ndarray, tick_slice: slice) -> dict:
    """Compute (mu, P90) for L2 / yaw; (mu, P10) for IoU; hit rates for L2.

    Aggregation is over (agent, tick) pairs in the selection, masked by valid.
    """
    if agent_mask.sum() == 0:
        return None
    valid = arrays['valid'][agent_mask, tick_slice]
    l2 = arrays['l2'][agent_mask, tick_slice]
    yerr = arrays['yaw_err'][agent_mask, tick_slice]
    iou = arrays['bev_iou'][agent_mask, tick_slice]

    flat_l2 = l2[valid]
    flat_y = yerr[valid & ~np.isnan(yerr)]
    flat_i = iou[valid & ~np.isnan(iou)]
    n_pairs = int(flat_l2.size)
    if n_pairs == 0:
        return None
    return {
        'n_agents': int(agent_mask.sum()),
        'n_pairs': n_pairs,
        'l2_mu': float(np.mean(flat_l2)),
        'l2_p90': float(np.percentile(flat_l2, 90)),
        'yaw_mu': float(np.mean(flat_y)) if flat_y.size else float('nan'),
        'yaw_p90': float(np.percentile(flat_y, 90)) if flat_y.size else float('nan'),
        'iou_mu': float(np.mean(flat_i)) if flat_i.size else float('nan'),
        'iou_p10': float(np.percentile(flat_i, 10)) if flat_i.size else float('nan'),
        'hit_2m': float(np.mean(flat_l2 < HIT_THRESHOLDS_M[0])),
        'hit_1m': float(np.mean(flat_l2 < HIT_THRESHOLDS_M[1])),
    }


def build_table(arrays) -> List[dict]:
    """Returns ordered list of row dicts: {label, label_pct, stats}."""
    n_agents = arrays['l2'].shape[0]
    if n_agents == 0:
        raise RuntimeError('No matched (pred, gt) pairs to score.')
    lump_arr = np.array(arrays['class_lump'])
    motion_arr = np.array(arrays['motion_bin'])
    valid_total = int(arrays['valid'].sum())

    all_mask = np.ones(n_agents, dtype=bool)
    full_slice = slice(0, FUT_TS)

    rows = []

    def add(label, mask, tick_slice, label_pct):
        stats = aggregate_row(arrays, mask, tick_slice)
        rows.append({'label': label, 'label_pct': label_pct, 'stats': stats})

    add('All', all_mask, full_slice, 100.0)

    # Class-group rows. Label % is fraction of valid (agent, tick) pairs in
    # the group out of the All total.
    for grp in ('Vehicle', 'Pedestrian', 'Movable'):
        mask = lump_arr == grp
        if mask.sum() == 0:
            add(grp, mask, full_slice, 0.0)
            continue
        grp_pairs = int(arrays['valid'][mask, :].sum())
        pct = 100.0 * grp_pairs / valid_total if valid_total else 0.0
        add(grp, mask, full_slice, pct)

    # Horizon rows. Label % is fraction of ticks in the window of total
    # valid ticks (~33% each if all agents have full futures, less otherwise).
    for label, (lo, hi) in HORIZON_BINS.items():
        pairs = int(arrays['valid'][:, lo:hi].sum())
        pct = 100.0 * pairs / valid_total if valid_total else 0.0
        add(label, all_mask, slice(lo, hi), pct)

    # Motion-behavior rows.
    for behavior in ('Stationary', 'Straight', 'Turning'):
        mask = motion_arr == behavior
        if mask.sum() == 0:
            add(behavior, mask, full_slice, 0.0)
            continue
        b_pairs = int(arrays['valid'][mask, :].sum())
        pct = 100.0 * b_pairs / valid_total if valid_total else 0.0
        add(behavior, mask, full_slice, pct)

    return rows


# --- Rendering -------------------------------------------------------------


def _fmt(v, prec=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '-'
    return f'{v:.{prec}f}'


def render_markdown(rows: List[dict]) -> str:
    lines = []
    lines.append('| Label Subset | Labels % | L2 (m) (μ, P90) | Yaw (°) (μ, P90) | BEV IoU (μ, P10) | Hit (<2m, <1m) |')
    lines.append('|---|---:|---|---|---|---|')
    for row in rows:
        s = row['stats']
        if s is None:
            lines.append(f"| {row['label']} | {row['label_pct']:.1f} | - | - | - | - |")
            continue
        lines.append(
            f"| {row['label']} | {row['label_pct']:.1f} | "
            f"{_fmt(s['l2_mu'])} / {_fmt(s['l2_p90'])} | "
            f"{_fmt(s['yaw_mu'], 1)} / {_fmt(s['yaw_p90'], 1)} | "
            f"{_fmt(s['iou_mu'])} / {_fmt(s['iou_p10'])} | "
            f"{_fmt(s['hit_2m'])} / {_fmt(s['hit_1m'])} |"
        )
    return '\n'.join(lines)


def render_latex(rows: List[dict]) -> str:
    lines = []
    lines.append(r'\begin{tabular}{lcccccc}')
    lines.append(r'\toprule')
    lines.append(
        r'\textbf{Label Subset} & \textbf{Labels} & \textbf{L2 err. (m) $\downarrow$} '
        r'& \textbf{Yaw err. ($^\circ$) $\downarrow$} & \textbf{BEV IoU $\uparrow$} '
        r'& \textbf{Hit Rate $\uparrow$} \\'
    )
    lines.append(
        r' & \% & ($\mu$, $P_{90}$) & ($\mu$, $P_{90}$) & ($\mu$, $P_{10}$) '
        r'& ($<$2m, $<$1m) \\'
    )
    lines.append(r'\midrule \midrule')

    def row_str(row, bold=False):
        s = row['stats']
        label = (rf'\textbf{{{row["label"]}}}' if bold else row['label'])
        if s is None:
            return rf'{label} & {row["label_pct"]:.1f} & - & - & - & - \\'
        return (
            rf"{label} & {row['label_pct']:.1f} & "
            rf"{_fmt(s['l2_mu'])} / {_fmt(s['l2_p90'])} & "
            rf"{_fmt(s['yaw_mu'], 1)} / {_fmt(s['yaw_p90'], 1)} & "
            rf"{_fmt(s['iou_mu'])} / {_fmt(s['iou_p10'])} & "
            rf"{_fmt(s['hit_2m'])} / {_fmt(s['hit_1m'])} \\"
        )

    # The first 'All' row gets bold.
    lines.append(row_str(rows[0], bold=True))
    lines.append(r'\midrule')
    # Class rows (Vehicle, Pedestrian, Movable).
    for row in rows[1:4]:
        lines.append(row_str(row))
    lines.append(r'\midrule')
    # Horizon rows.
    for row in rows[4:7]:
        # Rewrite label with " (horizon)" suffix replaced by "0--2s" style.
        # Already labelled '0-2s'; convert dash to en-dash for LaTeX.
        row = dict(row)
        row['label'] = row['label'].replace('-', '--')
        lines.append(row_str(row))
    lines.append(r'\midrule')
    # Motion rows.
    for row in rows[7:10]:
        lines.append(row_str(row))
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    return '\n'.join(lines)


# --- CLI -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pred-pkl', required=True, help='Path to motion_predictions.pkl')
    parser.add_argument('--dataroot', required=True, help='nuScenes data root')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--eval-set', default='val')
    parser.add_argument('--seconds', type=int, default=6)
    parser.add_argument('--output-tex', default=None, help='Path to write LaTeX table; '
                        'defaults to <pred-pkl-dir>/detailed_prediction_eval.tex')
    args = parser.parse_args()

    if args.output_tex is None:
        args.output_tex = os.path.join(
            os.path.dirname(os.path.abspath(args.pred_pkl)),
            'detailed_prediction_eval.tex',
        )

    print(f'[detailed-eval] loading preds: {args.pred_pkl}')
    preds_by_sample = load_pred_pkl(args.pred_pkl)
    print(f'[detailed-eval] {len(preds_by_sample)} sample predictions')

    print(f'[detailed-eval] loading nuScenes {args.version} from {args.dataroot}')
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    print(f'[detailed-eval] loading GT for {args.eval_set}')
    gts_by_sample = load_gt(nusc, args.eval_set, seconds=args.seconds)

    # Restrict to overlap (in case the prediction dump only contains a subset).
    common = sorted(set(preds_by_sample) & set(gts_by_sample))
    print(f'[detailed-eval] matching on {len(common)} common samples')

    matches_by_sample = {}
    for tok in common:
        matches_by_sample[tok] = match_sample(
            preds_by_sample[tok], gts_by_sample[tok]
        )
    n_match = sum(len(v) for v in matches_by_sample.values())
    print(f'[detailed-eval] {n_match} (pred, gt) matches')

    arrays = compute_per_agent_arrays(matches_by_sample, fut_ts=args.seconds * 2)
    print(f'[detailed-eval] {arrays["l2"].shape[0]} agents with valid futures')

    rows = build_table(arrays)
    md = render_markdown(rows)
    tex = render_latex(rows)

    print('\n=== Detailed Prediction Eval ===\n')
    print(md)
    print()

    with open(args.output_tex, 'w') as f:
        f.write(tex + '\n')
    print(f'[detailed-eval] wrote LaTeX table -> {args.output_tex}')


if __name__ == '__main__':
    main()
