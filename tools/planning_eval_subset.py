"""Offline planning eval filtered to a sample-token subset.

Replays planning metrics from a cached mmcv ``results.pkl`` (produced by a
prior ``dist_test.sh --eval bbox`` run) without rerunning the GPU forward
pass. Only samples whose token appears in ``--subset-samples`` contribute
to the metrics.

Usage
-----
    python tools/planning_eval_subset.py <config> <results_pkl> \\
        --subset-samples data/occ_interact/top600_samples.txt \\
        [--with-occlusion] [--output-json <path>]

The config must match the run that produced ``results.pkl`` (same val
pipeline / load_interval); otherwise ``results[i]`` and
``data_infos[i]`` desync silently.
"""
import argparse
import importlib
import json
import os
import os.path as osp
import sys

sys.path.insert(0, os.getcwd())

import mmcv
import numpy as np
from mmcv import Config
from mmdet.datasets import build_dataset, build_dataloader
from tqdm import tqdm

from projects.mmdet3d_plugin.datasets.evaluation.planning.planning_eval import (
    PlanningMetric,
)


def _load_subset(path: str) -> set:
    with open(path) as f:
        tokens = {line.strip() for line in f if line.strip()}
    if not tokens:
        raise SystemExit(f'subset file {path} is empty')
    return tokens


def _load_results(results_pkl: str):
    import torch
    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **{**k, 'map_location': 'cpu'})
    try:
        return mmcv.load(results_pkl)
    finally:
        torch.load = _orig_load


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config', help='Path to the model config.')
    ap.add_argument('results_pkl', help='Path to the cached mmcv results.pkl.')
    ap.add_argument('--subset-samples', required=True,
                    help='Path to a text file with one sample_token per line.')
    ap.add_argument('--with-occlusion', action='store_true',
                    help='Also compute occluded/ and all/ obj_box_col variants.')
    ap.add_argument('--output-json', default=None,
                    help='Optional path to dump the per-horizon metrics as JSON.')
    args = ap.parse_args()

    subset = _load_subset(args.subset_samples)
    print(f'[plan-subset] subset: {len(subset)} sample tokens from {args.subset_samples}')

    cfg = Config.fromfile(args.config)
    if cfg.get('plugin', False):
        plg_lib = cfg.plugin_dir.replace('/', '.').rstrip('.')
        importlib.import_module(plg_lib)

    print(f'[plan-subset] building dataset from {args.config} (eval_config)')
    # planning_eval uses eval_config — its pipeline collects fut_boxes /
    # gt_ego_fut_* which the test pipeline does not.
    dataset = build_dataset(cfg.eval_config)

    print(f'[plan-subset] loading results.pkl ({osp.getsize(args.results_pkl) / 1e9:.2f} GB)')
    results = _load_results(args.results_pkl)
    if len(results) != len(dataset.data_infos):
        raise SystemExit(
            f'results length ({len(results)}) != dataset length '
            f'({len(dataset.data_infos)}); config/results.pkl are out of sync')
    print(f'[plan-subset]   {len(results)} samples in results')

    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=1, shuffle=False, dist=False)

    metric = PlanningMetric()
    matched = 0
    skipped_incomplete = 0
    occluded_samples = 0
    occluded_box_timesteps = 0

    for i, data in enumerate(tqdm(loader)):
        token = dataset.data_infos[i]['token']
        if token not in subset:
            continue

        mask = data['gt_ego_fut_masks'].unsqueeze(-1).repeat(1, 1, 2).unsqueeze(1)
        if not mask.all():
            skipped_incomplete += 1
            continue

        sdc_planning = data['gt_ego_fut_trajs'].cumsum(dim=-2).unsqueeze(1)
        fut_boxes = data['fut_boxes']
        fut_boxes_occ = (data.get('fut_boxes_occluded', None)
                         if args.with_occlusion else None)

        if fut_boxes_occ is not None:
            sample_occ_count = sum(boxes[0].shape[0] for boxes in fut_boxes_occ)
            if sample_occ_count > 0:
                occluded_samples += 1
            occluded_box_timesteps += sample_occ_count

        pred = results[i]['img_bbox']['final_planning'].unsqueeze(0)
        metric.update(pred[:, :6, :2],
                      sdc_planning[0, :, :6, :2],
                      mask[0, :, :6, :2],
                      fut_boxes, fut_boxes_occ)
        matched += 1

    missing = len(subset) - matched - skipped_incomplete
    print(f'[plan-subset] matched={matched} '
          f'skipped_incomplete_gt={skipped_incomplete} '
          f'not_in_dataset={missing}')
    if args.with_occlusion:
        print(f'[plan-subset] occluded_samples={occluded_samples} '
              f'occluded_box_timesteps={occluded_box_timesteps}')

    if matched == 0:
        raise SystemExit('no subset samples matched the dataset; check tokens / config split')

    raw = metric.compute()

    # Match planning_eval's aggregation: per-horizon running mean, then
    # the (1.0s, 2.0s, 3.0s) average reported as the headline number.
    summary = {}
    rows = []
    print()
    print(f'{"metric":<28} {"0.5s":>8} {"1.0s":>8} {"1.5s":>8} '
          f'{"2.0s":>8} {"2.5s":>8} {"3.0s":>8} {"avg":>8}')
    for key, vals in raw.items():
        vals = vals.tolist()
        running = [float(np.mean(vals[:k + 1])) for k in range(len(vals))]
        avg = float(np.mean([running[1], running[3], running[5]]))
        summary[key] = {
            'per_horizon': running,
            'avg_1_2_3s': avg,
        }
        fmt = '{:>7.3f}%' if 'col' in key else '{:>8.4f}'
        cells = [fmt.format(v * 100 if 'col' in key else v) for v in running]
        cells.append(fmt.format(avg * 100 if 'col' in key else avg))
        print(f'{key:<28} ' + ' '.join(cells))
        rows.append((key, running, avg))

    if args.output_json:
        out = {
            'subset_samples_path': args.subset_samples,
            'subset_size': len(subset),
            'matched': matched,
            'skipped_incomplete_gt': skipped_incomplete,
            'not_in_dataset': missing,
            'with_occlusion': args.with_occlusion,
            'occluded_samples': occluded_samples,
            'occluded_box_timesteps': occluded_box_timesteps,
            'metrics': summary,
        }
        os.makedirs(osp.dirname(osp.abspath(args.output_json)) or '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\n[plan-subset] wrote {args.output_json}')


if __name__ == '__main__':
    main()
