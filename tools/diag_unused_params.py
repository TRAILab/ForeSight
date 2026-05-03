"""Diagnostic: list parameters that receive no gradient after one fwd+bwd.

Usage:
    python tools/diag_unused_params.py <config.py>
"""
import argparse
import sys
import os
import torch
import mmcv
from mmcv import Config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.datasets.builder import build_dataloader

# Trigger plugin imports (registers plugin modules)
import importlib
importlib.import_module("projects.mmdet3d_plugin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "plugin_dir"):
        importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))

    model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"),
                           test_cfg=cfg.get("test_cfg"))
    model.init_weights()
    model.train()

    from mmcv.parallel import MMDataParallel
    model = MMDataParallel(model.cuda(), device_ids=[0])

    dataset = build_dataset(cfg.data.train)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )

    batch = next(iter(loader))
    # MMDataParallel.train_step handles DataContainer scattering to CUDA.
    out = model.train_step(batch, optimizer=None)
    loss = out["loss"]
    loss.backward()

    unused = []
    used = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            unused.append((name, tuple(p.shape), p.numel()))
        else:
            used += 1

    total_unused_numel = sum(n for _, _, n in unused)
    print(f"Used parameters:   {used}")
    print(f"Unused parameters: {len(unused)}  ({total_unused_numel} weights)")
    print()
    if unused:
        # Group by top-level module for readable summary
        from collections import defaultdict
        by_top = defaultdict(list)
        for name, shape, n in unused:
            top = name.split(".")[0] + "." + name.split(".")[1] if "." in name else name
            by_top[top].append((name, shape, n))
        for top, entries in sorted(by_top.items(), key=lambda kv: -sum(e[2] for e in kv[1])):
            tot = sum(e[2] for e in entries)
            print(f"## {top}  ({len(entries)} params, {tot} weights)")
            for name, shape, n in entries[:20]:
                print(f"  {name:80s} {str(shape):20s} {n}")
            if len(entries) > 20:
                print(f"  ... ({len(entries) - 20} more)")
            print()


if __name__ == "__main__":
    main()
