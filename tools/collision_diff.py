"""Per-sample collision diff between baseline and ours.

Mirrors PlanningMetric.evaluate_single_coll on the baseline 'final_planning'
and ours 'final_planning' using fut_boxes from the dataset. Skips occluded
boxes (no use_gt_mask flip). Writes a small JSON with sample indices where
baseline collided but ours didn't, plus the reverse and the per-sample flags.
"""
import argparse, json, pickle, io, os
import numpy as np
import torch
from tqdm import tqdm
from mmcv import Config
from mmdet.datasets import build_dataset, build_dataloader

from projects.mmdet3d_plugin.datasets.evaluation.planning.planning_eval import (
    PlanningMetric,
)


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def load_pkl(path):
    with open(path, "rb") as f:
        return CPUUnpickler(f).load()


def per_sample_flags(planning_traj, fut_boxes, gt_traj):
    """Return (collided_any_t, gt_collided_any_t) bools."""
    pm = PlanningMetric()
    # Mirror update(): apply X-flip, run evaluate_single_coll on both pred and gt
    traj = planning_traj.clone()[:6, :2]
    traj[..., 0] = -traj[..., 0]
    gt = gt_traj.clone()[:6, :2]
    gt[..., 0] = -gt[..., 0]
    # evaluate_coll multiplies by [-1, 1] internally too; redo logic plainly.
    # Apply the inner [-1, 1] flip:
    traj_inner = traj * torch.tensor([-1.0, 1.0])
    gt_inner = gt * torch.tensor([-1.0, 1.0])
    coll = pm.evaluate_single_coll(traj_inner, fut_boxes)
    gt_coll = pm.evaluate_single_coll(gt_inner, fut_boxes)
    box_coll = torch.logical_and(coll, torch.logical_not(gt_coll))
    return bool(box_coll.any()), bool(gt_coll.any()), box_coll.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.eval_config)
    dl = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2,
                          shuffle=False, dist=False)

    print("loading baseline", args.baseline)
    base = load_pkl(args.baseline)
    print("loading ours", args.ours)
    ours = load_pkl(args.ours)
    assert len(base) == len(ours) == len(dataset), \
        f"len mismatch base={len(base)} ours={len(ours)} ds={len(dataset)}"

    base_flags = []   # any-t collision (excluding gt-collide)
    ours_flags = []
    skipped = []
    base_per_t = []
    ours_per_t = []

    for i, data in enumerate(tqdm(dl)):
        mask = data["gt_ego_fut_masks"]
        if not mask.all():
            base_flags.append(False); ours_flags.append(False)
            base_per_t.append([False] * 6); ours_per_t.append([False] * 6)
            skipped.append(i)
            continue
        fut_boxes = data["fut_boxes"]
        gt_traj = data["gt_ego_fut_trajs"].cumsum(dim=-2)[0]  # (6,2)

        base_entry = base[i]["img_bbox"] if "img_bbox" in base[i] else base[i]
        ours_entry = ours[i]["img_bbox"] if "img_bbox" in ours[i] else ours[i]
        base_traj = base_entry["final_planning"]
        ours_traj = ours_entry["final_planning"]

        b, _, b_per = per_sample_flags(base_traj, fut_boxes, gt_traj)
        o, _, o_per = per_sample_flags(ours_traj, fut_boxes, gt_traj)
        base_flags.append(b)
        ours_flags.append(o)
        base_per_t.append(b_per)
        ours_per_t.append(o_per)

    base_arr = np.array(base_flags)
    ours_arr = np.array(ours_flags)
    base_only = np.where(base_arr & ~ours_arr)[0].tolist()
    ours_only = np.where(~base_arr & ours_arr)[0].tolist()
    both = np.where(base_arr & ours_arr)[0].tolist()

    summary = {
        "n": len(base_arr),
        "skipped": len(skipped),
        "baseline_collided": int(base_arr.sum()),
        "ours_collided": int(ours_arr.sum()),
        "baseline_only": base_only,
        "ours_only": ours_only,
        "both_collided": both,
        "base_per_t": base_per_t,
        "ours_per_t": ours_per_t,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f)
    print("baseline collided:", summary["baseline_collided"])
    print("ours collided:    ", summary["ours_collided"])
    print("baseline-only:    ", len(base_only), base_only[:30])
    print("ours-only:        ", len(ours_only), ours_only[:30])
    print("both:             ", len(both))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
