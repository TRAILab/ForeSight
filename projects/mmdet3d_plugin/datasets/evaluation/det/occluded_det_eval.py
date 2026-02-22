from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_gt, add_center_dist
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.evaluate import NuScenesEval


class OccludedDetectionEval(NuScenesEval):
    """NuScenes detection evaluator restricted to occluded objects (num_lidar_pts == 0).

    The parent __init__ loads all GT and then applies filter_eval_boxes which
    removes every box with num_pts < 1.  We reload GT afterwards and replace
    self.gt_boxes with only the zero-point boxes so that evaluate() / main()
    score predictions against occluded ground truth only.
    """

    def __init__(self, nusc, config, result_path, eval_set, output_dir, verbose):
        super().__init__(nusc, config, result_path, eval_set, output_dir, verbose)

        # Reload GT to recover the occluded boxes that the parent removed.
        self.gt_boxes = load_gt(nusc, eval_set, DetectionBox, verbose=verbose)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)
        self.gt_boxes = self._filter_occluded_gt(self.gt_boxes)
        self.sample_tokens = self.gt_boxes.sample_tokens

    def _filter_occluded_gt(self, gt_boxes):
        """Keep only GT boxes that have zero sensor returns, within their class distance range."""
        from collections import Counter
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
