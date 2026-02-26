import numpy as np
from collections import Counter

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_gt, add_center_dist
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.evaluate import NuScenesEval


class OccludedDetectionEval(NuScenesEval):
    """NuScenes detection evaluator restricted to occluded objects (num_lidar_pts == 0).

    The parent __init__ loads all GT and then applies filter_eval_boxes which
    removes every box with num_pts < 1.  We reload GT afterwards and replace
    self.gt_boxes with only the zero-point boxes so that evaluate() scores
    predictions against occluded ground truth only.

    Ignore mechanism
    ----------------
    At each evaluation distance threshold T, predictions whose nearest
    same-class visible GT is within T metres (BEV) are excluded from the
    prediction set before calling accumulate().  Such predictions are
    attributable to a visible object at that threshold and should not be
    penalised as false positives against the occluded-only GT set.

    One filtered prediction set is built per threshold (4 sets for the
    standard [0.5, 1, 2, 4] m config), making the ignore rule exactly
    correct at every threshold rather than using a fixed worst-case distance.
    """

    def __init__(self, nusc, config, result_path, eval_set, output_dir, verbose):
        super().__init__(nusc, config, result_path, eval_set, output_dir, verbose)

        # Reload GT once to recover all boxes the parent filtered out.
        all_gt = load_gt(nusc, eval_set, DetectionBox, verbose=verbose)
        all_gt = add_center_dist(nusc, all_gt)

        # GT for scoring: occluded objects only.
        self.gt_boxes = self._filter_occluded_gt(all_gt)
        self.sample_tokens = self.gt_boxes.sample_tokens

        # GT to ignore: visible objects — predictions close to these are not FPs.
        self._ignore_boxes = self._filter_visible_gt(all_gt)

    # ------------------------------------------------------------------
    # GT filtering helpers
    # ------------------------------------------------------------------

    def _filter_occluded_gt(self, gt_boxes):
        """Keep only GT boxes with zero sensor returns, within class distance range."""
        filtered = EvalBoxes()
        for sample_token in gt_boxes.sample_tokens:
            boxes = [
                box for box in gt_boxes[sample_token]
                if box.num_pts == 0
                and box.detection_name in self.cfg.class_range
                and box.ego_dist < self.cfg.class_range[box.detection_name]
            ]
            filtered.add_boxes(sample_token, boxes)
        total = sum(len(filtered[t]) for t in filtered.sample_tokens)
        class_counts = Counter(
            box.detection_name
            for t in filtered.sample_tokens
            for box in filtered[t]
        )
        print(f'[Occluded Det] GT occluded boxes: {total} | {dict(class_counts)}')
        return filtered

    def _filter_visible_gt(self, gt_boxes):
        """Keep GT boxes with at least one sensor return, within class distance range."""
        filtered = EvalBoxes()
        for sample_token in gt_boxes.sample_tokens:
            boxes = [
                box for box in gt_boxes[sample_token]
                if box.num_pts > 0
                and box.detection_name in self.cfg.class_range
                and box.ego_dist < self.cfg.class_range[box.detection_name]
            ]
            filtered.add_boxes(sample_token, boxes)
        total = sum(len(filtered[t]) for t in filtered.sample_tokens)
        print(f'[Occluded Det] Ignore (visible) boxes: {total}')
        return filtered

    # ------------------------------------------------------------------
    # Ignore-aware prediction filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_ignore_matches(pred_boxes, ignore_boxes, ignore_dist):
        """Return a copy of pred_boxes with predictions that match a visible GT removed.

        A prediction is removed when its BEV centre distance to the nearest
        same-class visible GT box is ≤ ignore_dist.
        """
        ignore_by_token = {t: ignore_boxes[t] for t in ignore_boxes.sample_tokens}

        filtered = EvalBoxes()
        for sample_token in pred_boxes.sample_tokens:
            ignores = ignore_by_token.get(sample_token, [])
            keep = []
            for pred in pred_boxes[sample_token]:
                matched = False
                for ign in ignores:
                    if ign.detection_name != pred.detection_name:
                        continue
                    dist = np.sqrt(
                        (pred.translation[0] - ign.translation[0]) ** 2 +
                        (pred.translation[1] - ign.translation[1]) ** 2
                    )
                    if dist <= ignore_dist:
                        matched = True
                        break
                if not matched:
                    keep.append(pred)
            filtered.add_boxes(sample_token, keep)
        return filtered

    # ------------------------------------------------------------------
    # Per-threshold-accurate evaluation override
    # ------------------------------------------------------------------

    def evaluate(self):
        """Override NuScenesEval.evaluate() to apply ignore filtering per dist_th.

        For each evaluation distance threshold T the nuScenes AP computation
        treats every unmatched prediction as a false positive.  When evaluating
        occluded objects only, predictions that sit within T metres of a
        *visible* GT box are genuinely detecting that visible object — they
        should be ignored at threshold T rather than penalised as FPs.

        Pre-computing one filtered pred set per threshold (4 sets for the
        standard [0.5, 1, 2, 4] m config) is cheaper than the fixed 4 m
        approximation while being exactly correct for every threshold.
        """
        import time
        from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
        from nuscenes.eval.detection.data_classes import (
            DetectionMetrics,
            DetectionMetricDataList,
        )
        from nuscenes.eval.detection.constants import TP_METRICS

        start_time = time.time()

        if self.verbose:
            print('Accumulating metric data...')

        # Build one filtered prediction set per distance threshold.
        # _filter_ignore_matches is called len(dist_ths) times (e.g. 4),
        # not len(class_names) × len(dist_ths) (e.g. 40).
        filtered_by_dist = {
            dist_th: self._filter_ignore_matches(
                self.pred_boxes, self._ignore_boxes, dist_th
            )
            for dist_th in self.cfg.dist_ths
        }

        metric_data_list = DetectionMetricDataList()
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(
                    self.gt_boxes,
                    filtered_by_dist[dist_th],
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


class AllDetectionEval(NuScenesEval):
    """NuScenes detection evaluator on all objects (visible + occluded, num_pts >= 0).

    The parent __init__ calls filter_eval_boxes which removes every box with
    num_pts < 1.  We reload GT afterwards and replace self.gt_boxes with all
    boxes (class + distance filter only, no num_pts gate).
    """

    def __init__(self, nusc, config, result_path, eval_set, output_dir, verbose):
        super().__init__(nusc, config, result_path, eval_set, output_dir, verbose)

        # Reload GT to recover the occluded boxes that the parent removed.
        self.gt_boxes = load_gt(nusc, eval_set, DetectionBox, verbose=verbose)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)
        self.gt_boxes = self._filter_all_gt(self.gt_boxes)
        self.sample_tokens = self.gt_boxes.sample_tokens

    def _filter_all_gt(self, gt_boxes):
        """Keep all GT boxes (visible + occluded) within their class distance range."""
        from collections import Counter
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
