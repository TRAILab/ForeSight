"""Per-stage latency: backbone, backbone+neck, full forward.

Times each stage with cuda.synchronize between operations on the same
input batch, so end-to-end FPS = 1 / (backbone+neck+head).
"""
import argparse
import time
import sys
import torch
from mmcv import Config
from mmcv.parallel import scatter
from mmcv.runner import load_checkpoint, wrap_fp16_model

sys.path.append('.')
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.datasets import custom_build_dataset
from mmdet.models import build_detector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--samples', type=int, default=100)
    p.add_argument('--warmup', type=int, default=5)
    return p.parse_args()


def sync_time(fn, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    dataset = custom_build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False, shuffle=False)

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda(0)
    model.eval()

    t_backbone = []   # img_backbone(flat_img)
    t_neck = []       # img_neck(backbone_out)
    t_head = []       # remaining (extract_feat post-processing + head + post_process)
    t_total = []      # full forward

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = scatter(data, [0])[0]
            img = data["img"]
            flat_img = img.flatten(0, 1)  # (bs*num_cams, C, H, W)

            # Backbone alone
            _, dt_b = sync_time(model.img_backbone, flat_img)
            backbone_out = model.img_backbone(flat_img)
            # Neck alone
            _, dt_n = sync_time(model.img_neck, backbone_out)

            # Total full forward (matches benchmark.py loop)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            dt_total = time.perf_counter() - t0

            dt_h = dt_total - dt_b - dt_n  # head (incl extract_feat overhead, deformable, planner)

            if i >= args.warmup:
                t_backbone.append(dt_b)
                t_neck.append(dt_n)
                t_head.append(dt_h)
                t_total.append(dt_total)

            if (i + 1) == args.samples:
                break

    def stats(name, ts):
        ms = sum(ts) / len(ts) * 1000
        return f"{name:<20} {ms:7.2f} ms/img  ({1000/ms:5.2f} fps-equiv)"

    print(f"\n=== per-stage latency ({len(t_total)} samples after {args.warmup} warmup) ===")
    print(stats("img_backbone", t_backbone))
    print(stats("img_neck", t_neck))
    print(stats("head + rest", t_head))
    print(stats("TOTAL forward", t_total))
    bm = sum(t_backbone) / len(t_backbone)
    nm = sum(t_neck) / len(t_neck)
    tm = sum(t_total) / len(t_total)
    print(f"\nbreakdown: backbone={bm/tm*100:.1f}%  neck={nm/tm*100:.1f}%  head+rest={(tm-bm-nm)/tm*100:.1f}%")


if __name__ == '__main__':
    main()
