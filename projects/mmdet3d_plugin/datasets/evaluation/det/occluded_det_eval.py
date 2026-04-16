import numpy as np
from collections import Counter
from typing import Callable, Dict, Optional, Tuple

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_gt, add_center_dist
from nuscenes.eval.common.utils import (
    center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean,
)
from nuscenes.eval.detection.data_classes import DetectionBox, DetectionMetricData
from nuscenes.eval.detection.evaluate import NuScenesEval


# ---------------------------------------------------------------------------
# TPR / FDR helper
# ---------------------------------------------------------------------------

def compute_tpr_fdr(metrics_details_path, class_names, dist_ths):
    """Return TPR and FDR at the max-recall operating point.

    TPR = max achievable recall = TP_total / npos
    FDR = FP / (TP + FP) at that point = 1 - precision at max recall

    Reads from metrics_details.json written by nusc_eval.main().

    Returns
    -------
    dict:
      'mean_tpr'  : float  (mean over classes x dist_ths, like mAP)
      'mean_fdr'  : float
      'per_class' : {cls: {dist_th_str: {'tpr': float, 'fdr': float}}}
    """
    import json
    with open(metrics_details_path) as f:
        details = json.load(f)

    per_class = {}
    all_tprs, all_fdrs = [], []

    for cls in class_names:
        per_class[cls] = {}
        for dist_th in dist_ths:
            key = f'{cls}:{dist_th}'
            if key not in details:
                continue
            rec  = details[key]['recall']
            prec = details[key]['precision']
            conf = details[key]['confidence']

            nz = [i for i, c in enumerate(conf) if c > 0]
            if not nz:
                tpr, fdr = 0.0, 1.0
            else:
                idx = max(nz, key=lambda i: rec[i])
                tpr = rec[idx]
                fdr = 1.0 - prec[idx]

            per_class[cls][str(dist_th)] = {'tpr': tpr, 'fdr': fdr}
            all_tprs.append(tpr)
            all_fdrs.append(fdr)

    n = len(all_tprs)
    return {
        'mean_tpr':  sum(all_tprs) / n if n else 0.0,
        'mean_fdr':  sum(all_fdrs) / n if n else 0.0,
        'per_class': per_class,
    }


# ---------------------------------------------------------------------------
# Adaptive matching threshold coefficients (UniTraj class mapping)
# Formula: d = dist_th + alpha*t + beta*v*t + gamma*a*t^2
# ---------------------------------------------------------------------------

ADAPTIVE_COEFFS = {
    'vehicle':    (0.0568, 0.1962, 0.2133),
    'cyclist':    (0.1023, 0.1861, 0.2266),
    'pedestrian': (0.2641, 0.1457, 0.1774),
}

NUSCENES_TO_UNITRAJ = {
    'car':                  'vehicle',
    'truck':                'vehicle',
    'construction_vehicle': 'vehicle',
    'bus':                  'vehicle',
    'trailer':              'vehicle',
    'motorcycle':           'cyclist',
    'bicycle':              'cyclist',
    'pedestrian':           'pedestrian',
    # barrier, traffic_cone: no adaptive threshold
}


# ---------------------------------------------------------------------------
# Adaptive threshold helpers
# ---------------------------------------------------------------------------

def _ann_speed(nusc, ann: dict) -> float:
    """Speed (m/s) of an annotation estimated from position difference to prev."""
    if ann['prev'] == '':
        return 0.0
    prev_ann = nusc.get('sample_annotation', ann['prev'])
    curr_ts = nusc.get('sample', ann['sample_token'])['timestamp']
    prev_ts = nusc.get('sample', prev_ann['sample_token'])['timestamp']
    dt = (curr_ts - prev_ts) * 1e-6
    if dt <= 0:
        return 0.0
    dx = ann['translation'][0] - prev_ann['translation'][0]
    dy = ann['translation'][1] - prev_ann['translation'][1]
    return np.sqrt(dx ** 2 + dy ** 2) / dt


def _occ_metadata(nusc, ann_token: str) -> Tuple[float, float, float]:
    """Return (t, v, a) for an occluded annotation.

    t : occlusion duration in seconds (≥ 0.5 s since current frame is occluded)
    v : speed at the last visible annotation (m/s)
    a : acceleration magnitude at the last visible annotation (m/s²)
    """
    ann = nusc.get('sample_annotation', ann_token)

    # Walk prev-links counting consecutive occluded frames and finding the
    # last visible annotation and the one before it.
    t_frames = 1          # current frame counts as 1 occluded frame
    last_vis = None
    prev_of_last_vis = None

    prev_token = ann['prev']
    while prev_token != '':
        prev_ann = nusc.get('sample_annotation', prev_token)
        if prev_ann['num_lidar_pts'] > 0:
            last_vis = prev_ann
            if prev_ann['prev'] != '':
                prev_of_last_vis = nusc.get('sample_annotation', prev_ann['prev'])
            break
        t_frames += 1
        prev_token = prev_ann['prev']

    t = t_frames * 0.5  # NuScenes is 2 Hz → 0.5 s per frame

    v = _ann_speed(nusc, last_vis) if last_vis is not None else 0.0

    a = 0.0
    if last_vis is not None and prev_of_last_vis is not None:
        v_last = _ann_speed(nusc, last_vis)
        v_prev = _ann_speed(nusc, prev_of_last_vis)
        curr_ts = nusc.get('sample', last_vis['sample_token'])['timestamp']
        prev_ts = nusc.get('sample', prev_of_last_vis['sample_token'])['timestamp']
        dt = (curr_ts - prev_ts) * 1e-6
        if dt > 0:
            a = abs(v_last - v_prev) / dt

    return t, v, a


def _adaptive_dist_th(
    base_dist_th: float,
    class_name: str,
    t: float,
    v: float,
    a: float,
) -> float:
    """d = dist_th + alpha*t + beta*v*t + gamma*a*t^2"""
    unitraj_cls = NUSCENES_TO_UNITRAJ.get(class_name)
    if unitraj_cls is None:
        return base_dist_th  # static class (barrier, traffic_cone)
    alpha, beta, gamma = ADAPTIVE_COEFFS[unitraj_cls]
    return base_dist_th + alpha * t + beta * v * t + gamma * a * t ** 2


# ---------------------------------------------------------------------------
# Custom accumulate with three-outcome matching
# ---------------------------------------------------------------------------

def accumulate_with_ignore(
    gt_boxes: EvalBoxes,
    pred_boxes: EvalBoxes,
    ignore_boxes: EvalBoxes,
    class_name: str,
    dist_fcn: Callable,
    dist_th: float,
    per_gt_dist_ths: Optional[Dict[Tuple[str, int], float]] = None,
    verbose: bool = False,
) -> DetectionMetricData:
    """AP accumulation with a three-outcome matching rule.

    For each prediction (processed in descending confidence order):

      1. **TP** – prediction matches an unmatched visible GT box within its
         effective distance threshold (adaptive if per_gt_dist_ths provided,
         otherwise the fixed dist_th).
      2. **Ignored** – prediction does not match visible GT, but matches an
         ignore box within dist_th.  The prediction is excluded from both the
         numerator and denominator of the precision-recall curve.
      3. **FP** – prediction matches neither.

    Parameters
    ----------
    gt_boxes:          Visible GT boxes used for scoring (TP/FP/recall denominator).
    pred_boxes:        All model predictions.
    ignore_boxes:      GT boxes that neutralise unmatched preds.
    class_name:        Detection class to evaluate.
    dist_fcn:          BEV distance function.
    dist_th:           Base match / ignore distance threshold in metres.
    per_gt_dist_ths:   Optional dict mapping (sample_token, gt_idx) → adaptive
                       threshold for TP matching.  Ignore matching always uses
                       the fixed dist_th.
    """
    npos = len([1 for b in gt_boxes.all if b.detection_name == class_name])
    if verbose:
        print(f'Found {npos} GT of class {class_name} across '
              f'{len(gt_boxes.sample_tokens)} samples.')

    if npos == 0:
        return DetectionMetricData.no_predictions()

    ignore_by_token = {
        t: [b for b in ignore_boxes[t] if b.detection_name == class_name]
        for t in ignore_boxes.sample_tokens
    }

    pred_boxes_list = [b for b in pred_boxes.all if b.detection_name == class_name]
    pred_confs = [b.detection_score for b in pred_boxes_list]
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    tp = []
    fp = []
    conf = []
    match_data = {
        'trans_err': [], 'vel_err': [], 'scale_err': [],
        'orient_err': [], 'attr_err': [], 'conf': [],
    }

    taken = set()

    for ind in sortind:
        pred_box = pred_boxes_list[ind]

        # --- Step 1: find nearest unmatched GT ---
        min_dist = np.inf
        match_gt_idx = None
        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):
            if gt_box.detection_name != class_name:
                continue
            if (pred_box.sample_token, gt_idx) in taken:
                continue
            d = dist_fcn(gt_box, pred_box)
            if d < min_dist:
                min_dist = d
                match_gt_idx = gt_idx

        # Resolve effective threshold for the nearest GT box.
        if match_gt_idx is not None and per_gt_dist_ths is not None:
            eff_dist_th = per_gt_dist_ths.get(
                (pred_box.sample_token, match_gt_idx), dist_th
            )
        else:
            eff_dist_th = dist_th

        if min_dist < eff_dist_th:
            taken.add((pred_box.sample_token, match_gt_idx))
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            gt_match = gt_boxes[pred_box.sample_token][match_gt_idx]
            match_data['trans_err'].append(center_distance(gt_match, pred_box))
            match_data['vel_err'].append(velocity_l2(gt_match, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_match, pred_box))
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_match, pred_box, period=period))
            match_data['attr_err'].append(1 - attr_acc(gt_match, pred_box))
            match_data['conf'].append(pred_box.detection_score)

        else:
            # Step 2: check ignore boxes (always with fixed dist_th).
            ignores = ignore_by_token.get(pred_box.sample_token, [])
            is_ignored = any(dist_fcn(ign, pred_box) < dist_th for ign in ignores)

            if is_ignored:
                continue

            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)

    if len(match_data['trans_err']) == 0:
        return DetectionMetricData.no_predictions()

    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    for key in match_data:
        if key == 'conf':
            continue
        tmp = cummean(np.array(match_data[key]))
        match_data[key] = np.interp(
            conf[::-1], match_data['conf'][::-1], tmp[::-1]
        )[::-1]

    return DetectionMetricData(
        recall=rec, precision=prec, confidence=conf,
        trans_err=match_data['trans_err'], vel_err=match_data['vel_err'],
        scale_err=match_data['scale_err'], orient_err=match_data['orient_err'],
        attr_err=match_data['attr_err'],
    )


# ---------------------------------------------------------------------------
# Base evaluator
# ---------------------------------------------------------------------------

class _IgnoreAwareNuScenesEval(NuScenesEval):
    """Base class for ignore-aware detection evaluators."""

    def _filter_occluded_boxes(self, gt_boxes: EvalBoxes) -> EvalBoxes:
        filtered = EvalBoxes()
        for sample_token in gt_boxes.sample_tokens:
            boxes = [
                box for box in gt_boxes[sample_token]
                if box.num_pts == 0
                and box.detection_name in self.cfg.class_range
                and box.ego_dist < self.cfg.class_range[box.detection_name]
            ]
            filtered.add_boxes(sample_token, boxes)
        return filtered

    def _filter_visible_boxes(self, gt_boxes: EvalBoxes) -> EvalBoxes:
        filtered = EvalBoxes()
        for sample_token in gt_boxes.sample_tokens:
            boxes = [
                box for box in gt_boxes[sample_token]
                if box.num_pts > 0
                and box.detection_name in self.cfg.class_range
                and box.ego_dist < self.cfg.class_range[box.detection_name]
            ]
            filtered.add_boxes(sample_token, boxes)
        return filtered

    def evaluate(self):
        import time
        from nuscenes.eval.detection.algo import calc_ap, calc_tp
        from nuscenes.eval.detection.data_classes import (
            DetectionMetrics, DetectionMetricDataList,
        )
        from nuscenes.eval.detection.constants import TP_METRICS

        start_time = time.time()
        if self.verbose:
            print('Accumulating metric data...')

        metric_data_list = DetectionMetricDataList()
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate_with_ignore(
                    self.gt_boxes,
                    self.pred_boxes,
                    self._ignore_boxes,
                    class_name,
                    self.cfg.dist_fcn_callable,
                    dist_th,
                )
                metric_data_list.set(class_name, dist_th, md)

        if self.verbose:
            print('Calculating metrics...')

        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name == 'traffic_cone' and metric_name in ('attr_err', 'vel_err', 'orient_err'):
                    tp = np.nan
                elif class_name == 'barrier' and metric_name in ('attr_err', 'vel_err'):
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        metrics.add_runtime(time.time() - start_time)
        return metrics, metric_data_list


# ---------------------------------------------------------------------------
# Concrete evaluators
# ---------------------------------------------------------------------------

class OccludedDetectionEval(_IgnoreAwareNuScenesEval):
    """NuScenes detection evaluator restricted to occluded objects.

    Uses an adaptive matching threshold d = dist_th + alpha*t + beta*v*t +
    gamma*a*t^2 where t is occlusion duration, v is speed and a is
    acceleration at the last visible annotation.  Coefficients are
    class-specific (Vehicle / Cyclist / Pedestrian via UniTraj mapping).
    Static classes (barrier, traffic_cone) retain the fixed dist_th.
    """

    def __init__(self, nusc, config, result_path, eval_set, output_dir, verbose):
        super().__init__(nusc, config, result_path, eval_set, output_dir, verbose)

        all_gt = load_gt(nusc, eval_set, DetectionBox, verbose=verbose)
        all_gt = add_center_dist(nusc, all_gt)

        self.gt_boxes = self._filter_occluded_boxes(all_gt)
        self.sample_tokens = self.gt_boxes.sample_tokens
        self._ignore_boxes = self._filter_visible_boxes(all_gt)

        occ_total = sum(len(self.gt_boxes[t]) for t in self.gt_boxes.sample_tokens)
        class_counts = Counter(
            box.detection_name
            for t in self.gt_boxes.sample_tokens
            for box in self.gt_boxes[t]
        )
        vis_total = sum(len(self._ignore_boxes[t]) for t in self._ignore_boxes.sample_tokens)
        print(f'[Occluded Det] GT occluded boxes: {occ_total} | {dict(class_counts)}')
        print(f'[Occluded Det] Ignore (visible) boxes: {vis_total}')

        # Pre-compute adaptive thresholds for each (base_dist_th).
        self._adaptive_dist_ths = self._build_adaptive_dist_ths(nusc)

    def _build_adaptive_dist_ths(
        self, nusc
    ) -> Dict[float, Dict[Tuple[str, int], float]]:
        """Build per-GT-box adaptive thresholds for every base dist_th.

        Returns
        -------
        dict mapping base_dist_th → {(sample_token, gt_idx): adaptive_dist_th}
        """
        # Build sample_token → {rounded_translation: ann_token} for fast lookup.
        sample_ann_map: Dict[str, Dict[tuple, str]] = {}
        for sample_token in self.gt_boxes.sample_tokens:
            sample = nusc.get('sample', sample_token)
            pos_to_tok = {}
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                key = (round(ann['translation'][0], 2),
                       round(ann['translation'][1], 2))
                pos_to_tok[key] = ann_token
            sample_ann_map[sample_token] = pos_to_tok

        # Compute (t, v, a) and cache adaptive threshold per box per dist_th.
        result: Dict[float, Dict[Tuple[str, int], float]] = {
            d: {} for d in self.cfg.dist_ths
        }

        for sample_token in self.gt_boxes.sample_tokens:
            pos_to_tok = sample_ann_map[sample_token]
            for gt_idx, box in enumerate(self.gt_boxes[sample_token]):
                key_pos = (round(box.translation[0], 2),
                           round(box.translation[1], 2))
                ann_token = pos_to_tok.get(key_pos)
                if ann_token is None:
                    continue  # fallback: keep base dist_th (entry absent → default)

                t, v, a = _occ_metadata(nusc, ann_token)
                for base_dist_th in self.cfg.dist_ths:
                    adaptive = _adaptive_dist_th(base_dist_th, box.detection_name,
                                                 t, v, a)
                    result[base_dist_th][(sample_token, gt_idx)] = adaptive

        return result

    def evaluate(self):
        """Override to pass adaptive per-GT thresholds to accumulate_with_ignore."""
        import time
        from nuscenes.eval.detection.algo import calc_ap, calc_tp
        from nuscenes.eval.detection.data_classes import (
            DetectionMetrics, DetectionMetricDataList,
        )
        from nuscenes.eval.detection.constants import TP_METRICS

        start_time = time.time()
        if self.verbose:
            print('Accumulating metric data (adaptive thresholds)...')

        metric_data_list = DetectionMetricDataList()
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate_with_ignore(
                    self.gt_boxes,
                    self.pred_boxes,
                    self._ignore_boxes,
                    class_name,
                    self.cfg.dist_fcn_callable,
                    dist_th,
                    per_gt_dist_ths=self._adaptive_dist_ths.get(dist_th),
                )
                metric_data_list.set(class_name, dist_th, md)

        if self.verbose:
            print('Calculating metrics...')

        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name == 'traffic_cone' and metric_name in ('attr_err', 'vel_err', 'orient_err'):
                    tp = np.nan
                elif class_name == 'barrier' and metric_name in ('attr_err', 'vel_err'):
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        metrics.add_runtime(time.time() - start_time)
        return metrics, metric_data_list


class VisibleDetectionEval(_IgnoreAwareNuScenesEval):
    """NuScenes detection evaluator on visible objects (num_lidar_pts >= 1)."""

    def __init__(self, nusc, config, result_path, eval_set, output_dir, verbose):
        super().__init__(nusc, config, result_path, eval_set, output_dir, verbose)

        all_gt = load_gt(nusc, eval_set, DetectionBox, verbose=verbose)
        all_gt = add_center_dist(nusc, all_gt)
        self._ignore_boxes = self._filter_occluded_boxes(all_gt)

        vis_total = sum(len(self.gt_boxes[t]) for t in self.gt_boxes.sample_tokens)
        class_counts = Counter(
            box.detection_name
            for t in self.gt_boxes.sample_tokens
            for box in self.gt_boxes[t]
        )
        occ_total = sum(len(self._ignore_boxes[t]) for t in self._ignore_boxes.sample_tokens)
        print(f'[Visible Det] GT visible boxes: {vis_total} | {dict(class_counts)}')
        print(f'[Visible Det] Ignore (occluded) boxes: {occ_total}')


class AllDetectionEval(NuScenesEval):
    """NuScenes detection evaluator on all objects (visible + occluded)."""

    def __init__(self, nusc, config, result_path, eval_set, output_dir, verbose):
        super().__init__(nusc, config, result_path, eval_set, output_dir, verbose)

        self.gt_boxes = load_gt(nusc, eval_set, DetectionBox, verbose=verbose)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)
        self.gt_boxes = self._filter_all_gt(self.gt_boxes)
        self.sample_tokens = self.gt_boxes.sample_tokens

    def _filter_all_gt(self, gt_boxes: EvalBoxes) -> EvalBoxes:
        filtered = EvalBoxes()
        for sample_token in gt_boxes.sample_tokens:
            boxes = [
                box for box in gt_boxes[sample_token]
                if box.detection_name in self.cfg.class_range
                and box.ego_dist < self.cfg.class_range[box.detection_name]
            ]
            filtered.add_boxes(sample_token, boxes)
        total = sum(len(filtered[t]) for t in filtered.sample_tokens)
        class_counts = Counter(
            box.detection_name
            for t in filtered.sample_tokens
            for box in filtered[t]
        )
        print(f'[All Det] GT all boxes: {total} | {dict(class_counts)}')
        return filtered
