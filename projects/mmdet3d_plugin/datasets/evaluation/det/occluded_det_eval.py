import numpy as np
from collections import Counter
from typing import Callable

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_gt, add_center_dist
from nuscenes.eval.common.utils import (
    center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean,
)
from nuscenes.eval.detection.data_classes import DetectionBox, DetectionMetricData
from nuscenes.eval.detection.evaluate import NuScenesEval


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
    verbose: bool = False,
) -> DetectionMetricData:
    """AP accumulation with a three-outcome matching rule.

    For each prediction (processed in descending confidence order):

      1. **TP** – prediction matches an unmatched visible GT box within dist_th.
      2. **Ignored** – prediction does not match visible GT, but matches an
         ignore box within dist_th.  The prediction is excluded from both the
         numerator and denominator of the precision-recall curve.
      3. **FP** – prediction matches neither.

    This is strictly more correct than pre-filtering predictions before calling
    the standard accumulate(), because pre-filtering can remove legitimate TPs
    when a visible GT box and an ignore box are spatially close (within dist_th
    of each other).  Here, visible GT matching is resolved first under the
    greedy confidence-sorted algorithm, and the ignore check is only applied to
    predictions that failed to claim a visible GT box.

    Parameters
    ----------
    gt_boxes:     Visible GT boxes used for scoring (TP/FP/recall denominator).
    pred_boxes:   All model predictions.
    ignore_boxes: GT boxes that should neutralise unmatched predictions
                  (e.g. occluded GT for VisibleDetectionEval).
    class_name:   Detection class to evaluate.
    dist_fcn:     BEV distance function (same as used by nuScenes accumulate).
    dist_th:      Match / ignore distance threshold in metres.
    """
    # ------------------------------------------------------------------
    # Initialise
    # ------------------------------------------------------------------
    npos = len([1 for b in gt_boxes.all if b.detection_name == class_name])
    if verbose:
        print(f'Found {npos} GT of class {class_name} across '
              f'{len(gt_boxes.sample_tokens)} samples.')

    if npos == 0:
        return DetectionMetricData.no_predictions()

    # Pre-index ignore boxes by sample token for O(1) lookup.
    ignore_by_token = {
        t: [b for b in ignore_boxes[t] if b.detection_name == class_name]
        for t in ignore_boxes.sample_tokens
    }

    # Collect and sort predictions by confidence (descending).
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

    # ------------------------------------------------------------------
    # Greedy matching
    # ------------------------------------------------------------------
    taken = set()  # (sample_token, gt_idx) pairs already matched to a TP.

    for ind in sortind:
        pred_box = pred_boxes_list[ind]

        # --- Step 1: find nearest unmatched visible GT ---
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

        if min_dist < dist_th:
            # TP: prediction claimed a visible GT box.
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
            # --- Step 2: prediction did not match visible GT.
            #     Check if it is attributable to an ignore box. ---
            ignores = ignore_by_token.get(pred_box.sample_token, [])
            is_ignored = any(dist_fcn(ign, pred_box) < dist_th for ign in ignores)

            if is_ignored:
                # Excluded from the PR curve entirely — not TP, not FP.
                continue

            # FP: unmatched and not near any ignore box.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)

    # ------------------------------------------------------------------
    # Build precision-recall curve
    # ------------------------------------------------------------------
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
    """Base class for ignore-aware detection evaluators.

    Subclasses must set:
      self.gt_boxes     – EvalBoxes used as the scoring GT set.
      self._ignore_boxes – EvalBoxes whose proximity neutralises unmatched preds.

    The evaluate() override uses accumulate_with_ignore() which applies a
    three-outcome matching rule (TP / ignored / FP) so that visible GT matching
    is resolved before the ignore check.  This prevents the pre-filter approach
    from incorrectly removing TPs when a scoring GT box and an ignore box happen
    to be within dist_th of each other.
    """

    # ------------------------------------------------------------------
    # Shared GT filtering helpers
    # ------------------------------------------------------------------

    def _filter_occluded_boxes(self, gt_boxes: EvalBoxes) -> EvalBoxes:
        """Return GT boxes with zero sensor returns, within class distance range."""
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
        """Return GT boxes with at least one sensor return, within class distance range."""
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

    # ------------------------------------------------------------------
    # Evaluation override
    # ------------------------------------------------------------------

    def evaluate(self):
        """Override NuScenesEval.evaluate() using accumulate_with_ignore().

        For each (class, dist_th) pair, predictions are classified as TP,
        ignored, or FP according to the three-outcome rule in
        accumulate_with_ignore().  Ignored predictions are excluded from both
        precision and recall, removing the edge-case bias of pre-filtering.
        """
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
    """NuScenes detection evaluator restricted to occluded objects (num_lidar_pts == 0).

    Scores predictions against occluded GT only.  Predictions that match a
    visible GT box are ignored (not penalised as FPs) via the three-outcome
    matching rule in accumulate_with_ignore().
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


class VisibleDetectionEval(_IgnoreAwareNuScenesEval):
    """NuScenes detection evaluator on visible objects (num_lidar_pts >= 1).

    Scores predictions against visible GT only (same GT set as the standard
    nuScenes evaluator).  Predictions that fail to match visible GT but land
    within dist_th of an occluded GT box are ignored rather than penalised as
    FPs, via the three-outcome matching rule in accumulate_with_ignore().

    This makes vis/mAP a fair comparison between a visible-only baseline and a
    model trained on the full (visible + occluded) annotation set.
    """

    def __init__(self, nusc, config, result_path, eval_set, output_dir, verbose):
        super().__init__(nusc, config, result_path, eval_set, output_dir, verbose)

        # Parent filter_eval_boxes already set self.gt_boxes to visible-only.
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
    """NuScenes detection evaluator on all objects (visible + occluded, num_pts >= 0).

    The parent __init__ calls filter_eval_boxes which removes every box with
    num_pts < 1.  We reload GT afterwards and replace self.gt_boxes with all
    boxes (class + distance filter only, no num_pts gate).
    """

    def __init__(self, nusc, config, result_path, eval_set, output_dir, verbose):
        super().__init__(nusc, config, result_path, eval_set, output_dir, verbose)

        self.gt_boxes = load_gt(nusc, eval_set, DetectionBox, verbose=verbose)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)
        self.gt_boxes = self._filter_all_gt(self.gt_boxes)
        self.sample_tokens = self.gt_boxes.sample_tokens

    def _filter_all_gt(self, gt_boxes: EvalBoxes) -> EvalBoxes:
        """Keep all GT boxes (visible + occluded) within their class distance range."""
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
