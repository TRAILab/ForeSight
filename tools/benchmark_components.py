"""Sub-component latency profiling via cuda-event forward hooks.

Reports per-module time over N samples. Hooks fire on every forward of
the target module — sums across multiple calls (e.g. per-stage deformable).

Optional ego-only mode (matches benchmark_egoonly.py): stub det_head /
map_head with zero-anchor returns, force motion_plan_head to ego-only.

Optional baseline ablations:
  --ablate motion_only : zero ego entry in planner inputs
  --ablate plan_only   : zero agent entries in planner inputs (num_det,num_map=0)
"""
import argparse
import time
import sys
import torch
import torch.nn as nn
from collections import defaultdict
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
    p.add_argument('--ego-only', action='store_true',
                   help='stub det/map heads, force ego-only motion_plan_head')
    p.add_argument('--ablate', choices=['none', 'motion_only', 'plan_only'],
                   default='none',
                   help='baseline coupled-head ablations')
    p.add_argument('--keep-conflict', action='store_true',
                   help='leave conflict head enabled (used with --ego-only)')
    return p.parse_args()


# --- stubs (mirrors benchmark_egoonly.py) ---
def make_stub_det_forward(det_head, num_classes):
    embed_dims = det_head.instance_bank.embed_dims
    def stub(feature_maps, metas):
        ref = feature_maps[0] if isinstance(feature_maps, (list, tuple)) else feature_maps
        bs = metas['image_wh'].shape[0]
        device = ref.device
        dtype = ref.dtype
        z_D = torch.zeros(bs, 0, embed_dims, device=device, dtype=dtype)
        z_C = torch.zeros(bs, 0, num_classes, device=device, dtype=dtype)
        z_11 = torch.zeros(bs, 0, 11, device=device, dtype=dtype)
        z_2 = torch.zeros(bs, 0, 2, device=device, dtype=dtype)
        z_l = torch.zeros(bs, 0, device=device, dtype=torch.long)
        det_head.instance_bank.mask = torch.zeros(bs, dtype=torch.bool, device=device)
        return {
            "classification": [z_C], "prediction": [z_11],
            "quality": [z_2], "visibility": [z_C], "relevance": [z_C],
            "instance_feature": z_D, "anchor_embed": z_D,
            "num_warmup_preds": 0, "instance_id": z_l,
        }
    return stub


def make_stub_map_forward(map_head, num_map_classes):
    embed_dims = map_head.instance_bank.embed_dims
    def stub(feature_maps, metas):
        ref = feature_maps[0] if isinstance(feature_maps, (list, tuple)) else feature_maps
        bs = metas['image_wh'].shape[0]
        device = ref.device; dtype = ref.dtype
        z_D = torch.zeros(bs, 0, embed_dims, device=device, dtype=dtype)
        z_C = torch.zeros(bs, 0, num_map_classes, device=device, dtype=dtype)
        z_pts = torch.zeros(bs, 0, 40, device=device, dtype=dtype)
        return {
            "classification": [z_C], "prediction": [z_pts],
            "quality": [z_C],
            "instance_feature": z_D, "anchor_embed": z_D,
            "num_warmup_preds": 0,
        }
    return stub


def empty_postprocess(num_classes):
    def fn(output):
        bs = output["instance_feature"].shape[0]
        return [{
            "boxes_3d": torch.zeros(0, 9),
            "scores_3d": torch.zeros(0),
            "labels_3d": torch.zeros(0, dtype=torch.long),
            "cls_scores": torch.zeros(0, num_classes),
            "instance_ids": torch.zeros(0, dtype=torch.long),
            "trajs_3d": torch.zeros(0, 6, 12, 2),
            "trajs_score": torch.zeros(0, 6),
            "anchor_queue": torch.zeros(0, 4, 11),
            "period": torch.zeros(0, dtype=torch.long),
        } for _ in range(bs)]
    return fn


class CudaTimer:
    """Records per-module cumulative time across all forward calls."""
    def __init__(self):
        self.cumulative_ms = defaultdict(float)
        self.call_count = defaultdict(int)
        self._pending = []  # list of (label, start_event, end_event)
        # Phase markers: ordered list of (label, event) per iter.
        self._pending_markers = []
        self._marker_intervals_ms = defaultdict(float)  # cumulative interval between consecutive markers
        self._marker_counts = defaultdict(int)

    def hook_marker(self, label, module):
        """Record a single GPU timestamp at module's pre_hook fire."""
        def pre_hook(mod, inp):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._pending_markers.append((label, ev))
        module.register_forward_pre_hook(pre_hook)

    def record_marker(self, label):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._pending_markers.append((label, ev))

    def hook_module(self, name, module):
        def pre_hook(mod, inp):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            mod.__bench_start_event = s
            mod.__bench_end_event = e
        def post_hook(mod, inp, out):
            e = mod.__bench_end_event
            e.record()
            self._pending.append((name, mod.__bench_start_event, e))
        module.register_forward_pre_hook(pre_hook)
        module.register_forward_hook(post_hook)

    def hook_callable(self, name, container, attr):
        """Wrap a non-Module callable (e.g., decoder.decode) for timing."""
        original = getattr(container, attr)
        timer_self = self
        def wrapped(*args, **kwargs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = original(*args, **kwargs)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
            timer_self._pending_callable.append((name, dt))
            return out
        setattr(container, attr, wrapped)

    _pending_callable = None
    def begin_iter(self):
        self._pending_callable = []

    def flush_iter(self, count_iter=True):
        torch.cuda.synchronize()
        for name, s, e in self._pending:
            if count_iter:
                self.cumulative_ms[name] += s.elapsed_time(e)
                self.call_count[name] += 1
        self._pending = []
        if self._pending_callable is not None and count_iter:
            for name, dt in self._pending_callable:
                self.cumulative_ms[name] += dt
                self.call_count[name] += 1
        self._pending_callable = []
        # Process phase markers: sequential intervals between consecutive
        # markers in this iter.
        if count_iter and len(self._pending_markers) >= 2:
            for j in range(1, len(self._pending_markers)):
                a_label, a_ev = self._pending_markers[j - 1]
                b_label, b_ev = self._pending_markers[j]
                key = f'PHASE  {a_label} → {b_label}'
                self._marker_intervals_ms[key] += a_ev.elapsed_time(b_ev)
                self._marker_counts[key] += 1
        self._pending_markers = []

    def report(self, n_iters, total_ms_per_iter):
        print("\n=== sub-component latency ===")
        print(f"{'module':<40} {'ms/iter':>10} {'% total':>10} {'calls/iter':>12}")
        for name in sorted(self.cumulative_ms.keys()):
            ms = self.cumulative_ms[name] / n_iters
            calls = self.call_count[name] / n_iters
            pct = ms / total_ms_per_iter * 100 if total_ms_per_iter else 0
            print(f"{name:<40} {ms:>10.3f} {pct:>9.1f}% {calls:>12.2f}")
        if self._marker_intervals_ms:
            print("\n=== phase intervals ===")
            for key in sorted(self._marker_intervals_ms.keys()):
                ms = self._marker_intervals_ms[key] / n_iters
                pct = ms / total_ms_per_iter * 100 if total_ms_per_iter else 0
                print(f"{key:<60} {ms:>10.3f} {pct:>9.1f}%")


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

    head = model.head
    num_classes = len(cfg.class_names)
    num_map_classes = len(cfg.map_class_names)

    if args.ego_only:
        head.det_head.forward = make_stub_det_forward(head.det_head, num_classes)
        head.map_head.forward = make_stub_map_forward(head.map_head, num_map_classes)
        head.det_head.post_process = empty_postprocess(num_classes)
        head.map_head.post_process = empty_postprocess(num_map_classes)
        head.motion_plan_head.num_det = 0
        head.motion_plan_head.num_map = 0
        head.motion_plan_head.with_conflict_head = False
        head.motion_plan_head.skip_perception_kv = True
        head.motion_plan_head.ego_only_planning = True
        head.eval_skip_map = True
        # skip_perception_kv only nulls gnn/cross_gnn layers at __init__.
        # Baseline doesn't set it at config time, so do it here at runtime.
        for i, op in enumerate(head.motion_plan_head.operation_order):
            if op in ('gnn', 'cross_gnn'):
                head.motion_plan_head.layers[i] = None
        # Rescorer needs agent trajectories — disable for ego-only.
        if hasattr(head.motion_plan_head.planning_decoder, 'use_rescore'):
            head.motion_plan_head.planning_decoder.use_rescore = False
        print('[ego-only mode] perception heads stubbed; gnn/cross_gnn layers nulled; rescorer off')

    if args.ablate in ('plan_only', 'motion_only'):
        # Baseline's coupled head hard-codes `num_anchor + 1` (assuming ego
        # exists) throughout its 700-line forward — stripping ego or
        # zeroing num_det/num_map cascades into shape mismatches and empty
        # flash-attn K/V crashes that need invasive rewrites to fix.
        # Reported as: (6) motion-only ≈ full motion_plan_head (1 ego is
        # marginal in N=61 attention); (7) plan-only requires the minS2
        # forward path (skip_perception_kv + ego_only_planning), not
        # baseline's. Ablation stub kept for documentation only.
        print(f'[ablate {args.ablate}] not separable in baseline; see report')
        return

    if args.keep_conflict:
        # Used only with --ego-only to measure conflict head cost (item 2).
        head.motion_plan_head.with_conflict_head = True
        print('[keep_conflict] conflict head enabled despite ego-only')

    model = model.cuda(0).eval()

    # ---- attach timing hooks ----
    timer = CudaTimer()
    timer.hook_module('img_backbone', model.img_backbone)
    timer.hook_module('img_neck', model.img_neck)
    timer.hook_module('det_head.forward', head.det_head)
    timer.hook_module('map_head.forward', head.map_head)
    timer.hook_module('motion_plan_head.forward', head.motion_plan_head)
    # planner's per-stage ops live inside motion_plan_head.layers (a
    # ModuleList aligned with operation_order). Hook each op type — the
    # timer accumulates across all stages with that op.
    mph = head.motion_plan_head
    stage_starts = []  # indices into operation_order where new stage begins (temp_gnn)
    if hasattr(mph, 'layers') and hasattr(mph, 'operation_order'):
        for i, op in enumerate(mph.operation_order):
            if mph.layers[i] is None or op == 'norm':
                continue
            timer.hook_module(f'  └ planner.{op}[stage]', mph.layers[i])
        # Phase markers: pre-hook temp_gnn at start of each stage + the
        # final refine layer's POST. Plus motion_plan_head pre/post.
        # Intervals between consecutive markers tell us where time goes
        # between hooked layer ops.
        timer.hook_marker('mph_start', mph)
        seen_temp = 0
        for i, op in enumerate(mph.operation_order):
            if op == 'temp_gnn' and mph.layers[i] is not None:
                timer.hook_marker(f'stage{seen_temp}_start', mph.layers[i])
                seen_temp += 1
        # Last refine layer — mark when its forward begins (end of last
        # full stage). We can't post-hook with the marker scheme cleanly,
        # but a pre-hook on last_refine + the existing motion_plan_head
        # post (via .forward time = total) gives us "post stage N work".
        last_refine_idx = None
        for i, op in enumerate(mph.operation_order):
            if op == 'refine' and mph.layers[i] is not None:
                last_refine_idx = i
        if last_refine_idx is not None:
            timer.hook_marker('last_refine_start', mph.layers[last_refine_idx])
    if getattr(mph, 'conflict_image_sampler', None) is not None:
        timer.hook_module('  └ conflict_image_sampler', mph.conflict_image_sampler)
    # Non-layer modules in motion_plan_head — anchor encoders, ego status,
    # instance_queue init, projections.
    for mod_name in ('motion_anchor_encoder', 'plan_anchor_encoder',
                     'plan_ego_status_encoder', 'fc_before', 'fc_after',
                     'adapter', 'instance_queue'):
        sub = getattr(mph, mod_name, None)
        if sub is not None and isinstance(sub, nn.Module):
            timer.hook_module(f'  └ mph.{mod_name}', sub)
    # ego_feature_encoder lives inside instance_queue.
    iq = getattr(mph, 'instance_queue', None)
    if iq is not None and getattr(iq, 'ego_feature_encoder', None) is not None:
        timer.hook_module('  └ iq.ego_feature_encoder', iq.ego_feature_encoder)
    # det_head.anchor_encoder is called from motion_plan_head.forward to
    # encode ego/temp anchors — separate from det_head.forward time.
    if hasattr(head, 'det_head') and hasattr(head.det_head, 'anchor_encoder'):
        timer.hook_module('  └ det.anchor_encoder', head.det_head.anchor_encoder)
    # Refine-layer MLP branches (plan_cls / plan_reg / plan_conflict) —
    # all small per-mode (or per-mode-per-anchor) MLPs.
    if hasattr(mph, 'layers') and hasattr(mph, 'operation_order'):
        for i, op in enumerate(mph.operation_order):
            if op == 'refine' and mph.layers[i] is not None:
                refine = mph.layers[i]
                for branch_name in ('plan_cls_branch', 'plan_reg_branch',
                                    'plan_conflict_branch', 'motion_cls_branch',
                                    'motion_reg_branch'):
                    sub = getattr(refine, branch_name, None)
                    if sub is not None:
                        timer.hook_module(f'  └ refine.{branch_name}', sub)
    # rescorer / planning_decoder.decode  — non-Module, wrap method
    pd = head.motion_plan_head.planning_decoder
    if hasattr(pd, 'decode'):
        timer.hook_callable('planning_decoder.decode (rescorer)', pd, 'decode')
    md = head.motion_plan_head.motion_decoder
    if hasattr(md, 'decode'):
        timer.hook_callable('motion_decoder.decode', md, 'decode')

    # ---- run ----
    n_timed = 0
    total_ms_acc = 0.0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data_dev = scatter(data, [0])[0]
            timer.begin_iter()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.forward_test(data_dev['img'], **{k: v for k, v in data_dev.items() if k != 'img'})
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000

            count = (i >= args.warmup)
            timer.flush_iter(count_iter=count)
            if count:
                n_timed += 1
                total_ms_acc += elapsed_ms
            if (i + 1) == args.samples:
                break

    total_ms_per_iter = total_ms_acc / max(n_timed, 1)
    print(f"\n=== overall ===  {total_ms_per_iter:.2f} ms/iter  ({1000/total_ms_per_iter:.2f} fps)")
    timer.report(n_timed, total_ms_per_iter)


if __name__ == '__main__':
    main()
