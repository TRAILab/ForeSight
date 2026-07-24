#!/usr/bin/env python3
"""Offline detection eval on occluded GT (num_lidar_pts == 0), bucketed by
time-since-last-visible (TSLV).

For each occluded GT annotation we walk the per-instance ``prev`` chain to
find the most recent annotation with ``num_lidar_pts > 0``. TSLV is the
elapsed time from that visible frame to the current frame, in 0.5 s steps
(nuScenes is 2 Hz). Buckets:

    0-2s : TSLV in {0.5, 1.0, 1.5, 2.0}
    2-4s : TSLV in {2.5, 3.0, 3.5, 4.0}
    4-6s : TSLV in {4.5, 5.0, 5.5, 6.0}

Occluded boxes with no prior visible annotation (never-visible) and boxes
with TSLV > 6 s are excluded from every bucket's scoring set, but kept in
each bucket's *ignore* set so predictions matching them are not penalised
as false positives.

Matching uses the standard nuScenes fixed dist_th (0.5/1/2/4 m). Within
each bucket the evaluator is the same ignore-aware accumulator used by
``OccludedDetectionEval`` / ``VisibleDetectionEval``: visible GTs and
out-of-bucket occluded GTs are treated as ignore boxes.

Usage
-----
    python tools/eval_det_occluded_tslv.py \
        --result-path work_dirs/<run>/Wed_.../results_nusc.json \
        --dataroot data/nuscenes \
        --version v1.0-trainval \
        --eval-set val \
        --output-dir work_dirs/<run>/Wed_.../tslv_det
"""

import argparse
import copy
import json
import os
import os.path as osp
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_gt, add_center_dist
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionBox

# Reuse existing ignore-aware machinery without triggering the heavy
# projects.mmdet3d_plugin package __init__ (which imports CUDA ops).
_EVAL_DIR = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))),
    "projects", "mmdet3d_plugin", "datasets", "evaluation", "det",
)
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from occluded_det_eval import (  # noqa: E402
    _IgnoreAwareNuScenesEval,
    accumulate_with_ignore,
    compute_tpr_fdr,
)


BUCKETS: List[Tuple[str, float, float]] = [
    ("0-2s", 0.0, 2.0),
    ("2-4s", 2.0, 4.0),
    ("4-6s", 4.0, 6.0),
]

EVAL_SET_MAP = {
    "v1.0-mini": "mini_val",
    "v1.0-trainval": "val",
    "v1.0-test": "test",
}


# ---------------------------------------------------------------------------
# TSLV computation
# ---------------------------------------------------------------------------


def _tslv_for_ann(nusc: NuScenes, ann_token: str) -> Optional[float]:
    """Time-since-last-visible (seconds) for an occluded annotation.

    Walks the per-instance ``prev`` chain. Returns None if no prior
    annotation has ``num_lidar_pts > 0`` (i.e. the track was never visible
    earlier in the scene).
    """
    ann = nusc.get("sample_annotation", ann_token)
    # The current annotation itself is occluded; the first prev step is 0.5 s.
    n_steps = 1
    prev_token = ann["prev"]
    while prev_token != "":
        prev_ann = nusc.get("sample_annotation", prev_token)
        if prev_ann["num_lidar_pts"] > 0:
            return n_steps * 0.5
        n_steps += 1
        prev_token = prev_ann["prev"]
    return None


def _build_pos_to_ann_index(nusc: NuScenes, sample_tokens) -> Dict[str, Dict[tuple, str]]:
    """For each sample, map rounded (x, y) translation -> ann_token."""
    out: Dict[str, Dict[tuple, str]] = {}
    for st in sample_tokens:
        sample = nusc.get("sample", st)
        pos_to_tok: Dict[tuple, str] = {}
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            key = (round(ann["translation"][0], 2),
                   round(ann["translation"][1], 2))
            pos_to_tok[key] = ann_token
        out[st] = pos_to_tok
    return out


# ---------------------------------------------------------------------------
# Bucketed evaluator
# ---------------------------------------------------------------------------


class TSLVBucketDetectionEval(_IgnoreAwareNuScenesEval):
    """Detection eval restricted to occluded GT in a TSLV bucket.

    Parameters
    ----------
    nusc, config, result_path, eval_set, output_dir, verbose:
        Forwarded to ``NuScenesEval.__init__``.
    bucket_name, bucket_lo, bucket_hi:
        TSLV range (in seconds). A box is in the bucket when
        ``bucket_lo < TSLV <= bucket_hi``.
    all_gt:
        Optional preloaded EvalBoxes (with ego_dist already added) to avoid
        re-reading annotations for each bucket. If None, the parent class's
        gt is overwritten by reloading.
    tslv_by_pos:
        Optional preloaded ``{sample_token: {(round_x, round_y): tslv_or_None}}``
        keyed by rounded translation. If None, computed on the fly.
    """

    def __init__(
        self,
        nusc: NuScenes,
        config,
        result_path: str,
        eval_set: str,
        output_dir: str,
        verbose: bool,
        bucket_name: str,
        bucket_lo: float,
        bucket_hi: float,
        all_gt: Optional[EvalBoxes] = None,
        tslv_by_pos: Optional[Dict[str, Dict[tuple, Optional[float]]]] = None,
    ) -> None:
        super().__init__(nusc, config, result_path, eval_set, output_dir, verbose)

        if all_gt is None:
            all_gt = load_gt(nusc, eval_set, DetectionBox, verbose=verbose)
            all_gt = add_center_dist(nusc, all_gt)

        self.bucket_name = bucket_name
        self.bucket_lo = bucket_lo
        self.bucket_hi = bucket_hi

        # Visible boxes are always in the ignore set.
        visible = self._filter_visible_boxes(all_gt)

        # Build / use TSLV map keyed by (sample_token, rounded translation).
        if tslv_by_pos is None:
            tslv_by_pos = self._compute_tslv_map(nusc, all_gt)
        self._tslv_by_pos = tslv_by_pos

        in_bucket, out_of_bucket = self._split_occluded_by_tslv(all_gt)

        self.gt_boxes = in_bucket
        self.sample_tokens = self.gt_boxes.sample_tokens

        # Ignore = visible + occluded NOT in this bucket (including never-vis
        # and TSLV > 6 s). Merge into one EvalBoxes.
        ignore = EvalBoxes()
        for st in set(visible.sample_tokens) | set(out_of_bucket.sample_tokens):
            boxes = list(visible[st]) + list(out_of_bucket[st])
            ignore.add_boxes(st, boxes)
        self._ignore_boxes = ignore

        # Summary.
        in_total = sum(len(in_bucket[t]) for t in in_bucket.sample_tokens)
        class_counts = Counter(
            b.detection_name for t in in_bucket.sample_tokens for b in in_bucket[t]
        )
        vis_total = sum(len(visible[t]) for t in visible.sample_tokens)
        out_total = sum(len(out_of_bucket[t]) for t in out_of_bucket.sample_tokens)
        print(
            f"[TSLV {bucket_name}] GT in-bucket: {in_total} | {dict(class_counts)}"
        )
        print(
            f"[TSLV {bucket_name}] Ignore (visible {vis_total} + "
            f"other-bucket/never-vis occluded {out_total}): "
            f"{vis_total + out_total}"
        )

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    def _compute_tslv_map(
        self, nusc: NuScenes, all_gt: EvalBoxes
    ) -> Dict[str, Dict[tuple, Optional[float]]]:
        """Compute TSLV for every occluded GT box in ``all_gt``."""
        # Only need a lookup for samples that hold at least one in-range occluded box.
        target_samples = [
            st for st in all_gt.sample_tokens
            if any(
                b.num_pts == 0
                and b.detection_name in self.cfg.class_range
                and b.ego_dist < self.cfg.class_range[b.detection_name]
                for b in all_gt[st]
            )
        ]
        pos_index = _build_pos_to_ann_index(nusc, target_samples)

        out: Dict[str, Dict[tuple, Optional[float]]] = {}
        for st in target_samples:
            pos_to_tok = pos_index[st]
            entry: Dict[tuple, Optional[float]] = {}
            for box in all_gt[st]:
                if box.num_pts != 0:
                    continue
                if box.detection_name not in self.cfg.class_range:
                    continue
                if box.ego_dist >= self.cfg.class_range[box.detection_name]:
                    continue
                key = (round(box.translation[0], 2),
                       round(box.translation[1], 2))
                ann_token = pos_to_tok.get(key)
                if ann_token is None:
                    entry[key] = None
                    continue
                entry[key] = _tslv_for_ann(nusc, ann_token)
            out[st] = entry
        return out

    def _split_occluded_by_tslv(
        self, all_gt: EvalBoxes
    ) -> Tuple[EvalBoxes, EvalBoxes]:
        """Partition occluded GTs into in-bucket and out-of-bucket EvalBoxes."""
        in_bucket = EvalBoxes()
        out_of_bucket = EvalBoxes()
        for st in all_gt.sample_tokens:
            in_list: List[DetectionBox] = []
            out_list: List[DetectionBox] = []
            tslv_entry = self._tslv_by_pos.get(st, {})
            for box in all_gt[st]:
                if box.num_pts != 0:
                    continue
                if box.detection_name not in self.cfg.class_range:
                    continue
                if box.ego_dist >= self.cfg.class_range[box.detection_name]:
                    continue
                key = (round(box.translation[0], 2),
                       round(box.translation[1], 2))
                tslv = tslv_entry.get(key)
                if (tslv is not None
                        and self.bucket_lo < tslv <= self.bucket_hi):
                    in_list.append(box)
                else:
                    out_list.append(box)
            in_bucket.add_boxes(st, in_list)
            out_of_bucket.add_boxes(st, out_list)
        return in_bucket, out_of_bucket

    # ------------------------------------------------------------------
    # Evaluation (fixed dist_th; no adaptive thresholds)
    # ------------------------------------------------------------------

    def evaluate(self):
        import time
        from nuscenes.eval.detection.algo import calc_ap, calc_tp
        from nuscenes.eval.detection.data_classes import (
            DetectionMetrics, DetectionMetricDataList,
        )
        from nuscenes.eval.detection.constants import TP_METRICS

        start = time.time()
        if self.verbose:
            print(f"[TSLV {self.bucket_name}] Accumulating metric data...")

        mdl = DetectionMetricDataList()
        for cls in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate_with_ignore(
                    self.gt_boxes,
                    self.pred_boxes,
                    self._ignore_boxes,
                    cls,
                    self.cfg.dist_fcn_callable,
                    dist_th,
                )
                mdl.set(cls, dist_th, md)

        metrics = DetectionMetrics(self.cfg)
        for cls in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                ap = calc_ap(mdl[(cls, dist_th)],
                             self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(cls, dist_th, ap)
            for metric_name in TP_METRICS:
                md = mdl[(cls, self.cfg.dist_th_tp)]
                if cls == "traffic_cone" and metric_name in ("attr_err", "vel_err", "orient_err"):
                    tp = np.nan
                elif cls == "barrier" and metric_name in ("attr_err", "vel_err"):
                    tp = np.nan
                else:
                    tp = calc_tp(md, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(cls, metric_name, tp)

        metrics.add_runtime(time.time() - start)
        return metrics, mdl


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline TSLV-bucketed occluded detection eval."
    )
    p.add_argument("--result-path", required=True,
                   help="Path to results_nusc.json (NuScenes-format predictions).")
    p.add_argument("--dataroot", required=True,
                   help="Path to nuScenes dataset root.")
    p.add_argument("--version", default="v1.0-trainval",
                   choices=list(EVAL_SET_MAP.keys()))
    p.add_argument("--eval-set", default=None,
                   help="nuScenes eval split (default: val for trainval, "
                        "mini_val for mini).")
    p.add_argument("--eval-version", default="detection_cvpr_2019",
                   help="nuScenes detection eval config name.")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: <result_dir>/tslv_det).")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _collect_metrics(detail: dict, bucket: str, classes, err_name_mapping, output_dir: str,
                     dist_ths) -> dict:
    """Read a bucket's metrics_summary.json + TPR/FDR and flatten to a dict."""
    metrics = json.load(open(osp.join(output_dir, "metrics_summary.json")))
    prefix = f"tslv_{bucket}"
    for name in classes:
        for k, v in metrics["label_aps"].get(name, {}).items():
            detail[f"{prefix}/{name}_AP_dist_{k}"] = round(float(v), 4)
        for k, v in metrics["label_tp_errors"].get(name, {}).items():
            detail[f"{prefix}/{name}_{k}"] = round(float(v), 4)
    for k, v in metrics["tp_errors"].items():
        detail[f"{prefix}/{err_name_mapping[k]}"] = round(float(v), 4)
    detail[f"{prefix}/NDS"] = round(float(metrics["nd_score"]), 4)
    detail[f"{prefix}/mAP"] = round(float(metrics["mean_ap"]), 4)

    tpr_fdr = compute_tpr_fdr(
        osp.join(output_dir, "metrics_details.json"), classes, dist_ths,
    )
    detail[f"{prefix}/mTPR"] = round(tpr_fdr["mean_tpr"], 4)
    detail[f"{prefix}/mFDR"] = round(tpr_fdr["mean_fdr"], 4)
    return detail


def _render_markdown_table(summary: dict, classes) -> str:
    headers = ["bucket", "mAP", "NDS", "mTPR", "mFDR", "mATE", "mASE",
               "mAOE", "mAVE", "mAAE"]
    err_keys = ["mATE", "mASE", "mAOE", "mAVE", "mAAE"]
    rows = []
    for name, _, _ in BUCKETS:
        prefix = f"tslv_{name}"
        row = [name,
               f"{summary.get(f'{prefix}/mAP', float('nan')):.4f}",
               f"{summary.get(f'{prefix}/NDS', float('nan')):.4f}",
               f"{summary.get(f'{prefix}/mTPR', float('nan')):.4f}",
               f"{summary.get(f'{prefix}/mFDR', float('nan')):.4f}"]
        for ek in err_keys:
            row.append(f"{summary.get(f'{prefix}/{ek}', float('nan')):.4f}")
        rows.append(row)

    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")

    # Per-class mAP block (averaged over dist_ths).
    lines.append("")
    lines.append("Per-class mAP (mean over dist thresholds):")
    cls_header = ["bucket"] + classes
    lines.append("| " + " | ".join(cls_header) + " |")
    lines.append("|" + "|".join(["---"] * len(cls_header)) + "|")
    for name, _, _ in BUCKETS:
        prefix = f"tslv_{name}"
        row = [name]
        for cls in classes:
            vals = [v for k, v in summary.items()
                    if k.startswith(f"{prefix}/{cls}_AP_dist_")]
            row.append(f"{(sum(vals) / len(vals)) if vals else float('nan'):.4f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


ERR_NAME_MAPPING = {
    "trans_err": "mATE", "scale_err": "mASE", "orient_err": "mAOE",
    "vel_err":   "mAVE", "attr_err":  "mAAE",
}


def main() -> None:
    args = _parse_args()

    result_path = osp.abspath(args.result_path)
    if not osp.isfile(result_path):
        raise FileNotFoundError(result_path)

    eval_set = args.eval_set or EVAL_SET_MAP[args.version]
    output_root = args.output_dir or osp.join(osp.dirname(result_path), "tslv_det")
    os.makedirs(output_root, exist_ok=True)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot,
                    verbose=args.verbose)
    cfg = config_factory(args.eval_version)
    cfg.class_names = list(cfg.class_range.keys())

    # Load + ego-dist annotate once, reuse across buckets.
    all_gt = load_gt(nusc, eval_set, DetectionBox, verbose=args.verbose)
    all_gt = add_center_dist(nusc, all_gt)

    summary: dict = {}
    tslv_by_pos: Optional[Dict[str, Dict[tuple, Optional[float]]]] = None

    for bucket_name, lo, hi in BUCKETS:
        out_dir = osp.join(output_root, bucket_name)
        os.makedirs(out_dir, exist_ok=True)

        evaluator = TSLVBucketDetectionEval(
            nusc=nusc,
            config=copy.deepcopy(cfg),
            result_path=result_path,
            eval_set=eval_set,
            output_dir=out_dir,
            verbose=args.verbose,
            bucket_name=bucket_name,
            bucket_lo=lo,
            bucket_hi=hi,
            all_gt=all_gt,
            tslv_by_pos=tslv_by_pos,
        )
        # Cache the TSLV map after the first bucket — buckets share the same map.
        if tslv_by_pos is None:
            tslv_by_pos = evaluator._tslv_by_pos

        evaluator.main(render_curves=False)

        _collect_metrics(
            summary, bucket_name, cfg.class_names, ERR_NAME_MAPPING,
            out_dir, cfg.dist_ths,
        )

    # Write combined summary.
    combined_json = osp.join(output_root, "tslv_metrics_summary.json")
    with open(combined_json, "w") as f:
        json.dump(summary, f, indent=2)

    table_md = _render_markdown_table(summary, cfg.class_names)
    md_path = osp.join(output_root, "tslv_metrics_summary.md")
    with open(md_path, "w") as f:
        f.write("# TSLV-bucketed occluded detection eval\n\n")
        f.write(f"Result file: `{result_path}`\n\n")
        f.write(f"Eval set: `{eval_set}` ({args.version})\n\n")
        f.write(table_md + "\n")

    print()
    print(table_md)
    print()
    print(f"Wrote summary JSON: {combined_json}")
    print(f"Wrote summary MD:   {md_path}")


if __name__ == "__main__":
    main()
