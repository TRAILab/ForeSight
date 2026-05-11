"""Enumerate trainable named_parameters for a given config.

Run inside the container:
    apptainer exec ... python3 tools/debug/list_trainable_params.py <config>

Prints `idx name shape` for every Parameter with requires_grad=True, so
we can map DDP's "Parameter indices which did not receive grad: ..."
output back to module paths.
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import mmcv
from mmcv import Config
from mmdet.models import build_detector


def main():
    cfg_path = sys.argv[1]
    cfg = Config.fromfile(cfg_path)
    if hasattr(cfg, "plugin"):
        import importlib
        plugin_dir = cfg.plugin_dir.replace("/", ".").rstrip(".")
        importlib.import_module(plugin_dir)
    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    idx = 0
    print(f"=== trainable named_parameters for {cfg_path} ===")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"{idx:4d}  {name:80s}  {tuple(p.shape)}")
            idx += 1
    print(f"--- total trainable: {idx} ---")


if __name__ == "__main__":
    main()
