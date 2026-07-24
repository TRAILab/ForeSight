"""Re-run dataset.evaluate() from a cached results.pkl to populate motion_predictions.pkl.

Use when an older training run finished eval before _dump_motion_predictions
existed, but still saved its raw mmcv `results.pkl`. This skips the GPU
forward pass entirely and just replays the post-processing.

Usage:
    python tools/dump_motion_from_results.py <config> <results_pkl> --work-dir <out>
"""
import argparse
import importlib
import os
import os.path as osp
import sys

# Ensure the project root is on sys.path so the mmdet3d_plugin package imports.
sys.path.insert(0, os.getcwd())

import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to the model config.')
    parser.add_argument('results_pkl', help='Path to the cached mmcv results.pkl.')
    parser.add_argument('--work-dir', required=True,
                        help='Directory where motion_predictions.pkl will be written.')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    # Register custom dataset plugin (matches what tools/test.py does).
    if cfg.get('plugin', False):
        plg_lib = cfg.plugin_dir.replace('/', '.').rstrip('.')
        importlib.import_module(plg_lib)

    mmcv.mkdir_or_exist(args.work_dir)
    cfg.data.test.work_dir = args.work_dir

    print(f'[dump-motion] building dataset from {args.config}')
    dataset = build_dataset(cfg.data.test)

    print(f'[dump-motion] loading results.pkl ({osp.getsize(args.results_pkl) / 1e9:.2f} GB)')
    # mmcv-pickled tensors may reference CUDA devices that don't exist on this
    # host (e.g. a 4-GPU train run unpickled on a single-GPU machine). Force
    # CPU mapping during unpickling.
    import torch
    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **{**k, 'map_location': 'cpu'})
    try:
        results = mmcv.load(args.results_pkl)
    finally:
        torch.load = _orig_load
    print(f'[dump-motion]   {len(results)} samples')

    print('[dump-motion] formatting motion results + dumping (skipping det/track/map/etc.)')
    eval_mode = cfg.get('evaluation', {}).get('eval_mode', {})
    thresh = eval_mode.get('motion_threshhold', 0.2)
    motion_result_files = dataset.format_motion_results(
        results, jsonfile_prefix=args.work_dir, thresh=thresh,
    )
    dataset._dump_motion_predictions(motion_result_files)

    out = osp.join(args.work_dir, 'motion_predictions.pkl')
    print(f'[dump-motion] done. expected pickle at: {out}')


if __name__ == '__main__':
    main()
