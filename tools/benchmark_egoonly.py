"""Latency bench for an ego-only inference path.

Stubs det_head and map_head with zero-anchor no-op forwards and forces
motion_plan_head to skip all agent/map token paths. Only the ego token
flows into the planner.
"""
import argparse
import time
import sys
import torch
import torch.nn as nn
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

sys.path.append('.')
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.datasets import custom_build_dataset
from mmdet.models import build_detector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--samples', type=int, default=200)
    p.add_argument('--log-interval', type=int, default=50)
    return p.parse_args()


def make_stub_det_forward(det_head, num_classes):
    """Return a forward that emits zero-anchor det_output. No GPU compute
    beyond constant-tensor allocation. Preserves all keys consumed by
    motion_planning_head and instance_queue.
    """
    embed_dims = det_head.instance_bank.embed_dims
    decoder_score_threshold = det_head.decoder.score_threshold

    def stub_forward(feature_maps, metas):
        # bs from any feature map in the list
        if isinstance(feature_maps, (list, tuple)):
            ref = feature_maps[0]
        else:
            ref = feature_maps
        bs = ref.shape[0] if ref.dim() >= 2 else metas['img'].shape[0] if 'img' in metas else 1
        # Note: when use_deformable_func, feature_maps is a list of stacked
        # tensors; first tensor's leading dim is bs * num_cams. Fall back
        # to metas['image_wh'] which has shape (bs, num_cams, 2).
        if 'image_wh' in metas:
            bs = metas['image_wh'].shape[0]
        device = ref.device if isinstance(ref, torch.Tensor) else torch.device('cuda')
        dtype = ref.dtype if isinstance(ref, torch.Tensor) else torch.float32
        zeros_iN_D = torch.zeros(bs, 0, embed_dims, device=device, dtype=dtype)
        zeros_iN_C = torch.zeros(bs, 0, num_classes, device=device, dtype=dtype)
        zeros_iN_11 = torch.zeros(bs, 0, 11, device=device, dtype=dtype)
        zeros_iN_2 = torch.zeros(bs, 0, 2, device=device, dtype=dtype)
        zeros_iN_long = torch.zeros(bs, 0, device=device, dtype=torch.long)
        return {
            "classification": [zeros_iN_C],
            "prediction": [zeros_iN_11],
            "quality": [zeros_iN_2],
            "visibility": [zeros_iN_C],
            "relevance": [zeros_iN_C],
            "instance_feature": zeros_iN_D,
            "anchor_embed": zeros_iN_D,
            "num_warmup_preds": 0,
            "instance_id": zeros_iN_long,
        }
    return stub_forward


def make_stub_map_forward(map_head, num_map_classes):
    embed_dims = map_head.instance_bank.embed_dims

    def stub_forward(feature_maps, metas):
        if isinstance(feature_maps, (list, tuple)):
            ref = feature_maps[0]
        else:
            ref = feature_maps
        bs = metas['image_wh'].shape[0] if 'image_wh' in metas else 1
        device = ref.device if isinstance(ref, torch.Tensor) else torch.device('cuda')
        dtype = ref.dtype if isinstance(ref, torch.Tensor) else torch.float32
        zeros_iN_D = torch.zeros(bs, 0, embed_dims, device=device, dtype=dtype)
        zeros_iN_C = torch.zeros(bs, 0, num_map_classes, device=device, dtype=dtype)
        zeros_iN_pts = torch.zeros(bs, 0, 40, device=device, dtype=dtype)
        return {
            "classification": [zeros_iN_C],
            "prediction": [zeros_iN_pts],
            "quality": [zeros_iN_C],
            "instance_feature": zeros_iN_D,
            "anchor_embed": zeros_iN_D,
            "num_warmup_preds": 0,
        }
    return stub_forward


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    print('=== ego-only latency bench ===')

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

    # ---- ego-only patches ----
    head = model.head
    num_classes = len(cfg.class_names)
    num_map_classes = len(cfg.map_class_names)
    # 1. Stub det_head and map_head forwards (zero anchors, no compute).
    head.det_head.forward = make_stub_det_forward(head.det_head, num_classes)
    head.map_head.forward = make_stub_map_forward(head.map_head, num_map_classes)
    # Also stub post_process to bypass topk on empty cls_scores.
    def _empty_postprocess(output):
        # bs comes from any tensor in the dict.
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
    head.det_head.post_process = _empty_postprocess
    head.map_head.post_process = _empty_postprocess
    # 1b. instance_bank.mask is normally set inside det_head.cache(). Wrap
    # det_head.forward to also seed a per-sample False mask (== "sequence
    # reset"), so motion_plan_head sees a valid tensor.
    _stub_det_fwd = head.det_head.forward
    def _det_with_mask_seed(feature_maps, metas):
        out = _stub_det_fwd(feature_maps, metas)
        bs = out["instance_feature"].shape[0]
        device = out["instance_feature"].device
        head.det_head.instance_bank.mask = torch.zeros(bs, dtype=torch.bool, device=device)
        return out
    head.det_head.forward = _det_with_mask_seed
    # 2. Force ego-only motion_plan_head settings.
    head.motion_plan_head.num_det = 0
    head.motion_plan_head.num_map = 0
    head.motion_plan_head.with_conflict_head = False
    head.motion_plan_head.skip_perception_kv = True
    head.motion_plan_head.ego_only_planning = True
    # 3. Skip map at eval (also belt-and-suspenders).
    head.eval_skip_map = True
    print('patched: det_head/map_head -> zero-anchor stubs; motion_plan_head ego-only')

    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    num_warmup = 5
    pure_inf_time = 0.0
    max_memory = 0
    t_back, t_neck, t_total = [], [], []
    inner_model = model.module if hasattr(model, 'module') else model
    from mmcv.parallel import scatter as _scatter
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            data_dev = _scatter(data, [0])[0]
            img = data_dev["img"]
            flat_img = img.flatten(0, 1)

            torch.cuda.synchronize(); tb0 = time.perf_counter()
            _ = inner_model.img_backbone(flat_img)
            torch.cuda.synchronize(); dt_b = time.perf_counter() - tb0

            backbone_out = inner_model.img_backbone(flat_img)
            torch.cuda.synchronize(); tn0 = time.perf_counter()
            _ = inner_model.img_neck(backbone_out)
            torch.cuda.synchronize(); dt_n = time.perf_counter() - tn0

            torch.cuda.synchronize(); tt0 = time.perf_counter()
            model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize(); elapsed = time.perf_counter() - tt0
            mem = torch.cuda.max_memory_allocated() // (1024 * 1024)
            max_memory = max(max_memory, mem)

        if i >= num_warmup:
            pure_inf_time += elapsed
            t_back.append(dt_b); t_neck.append(dt_n); t_total.append(elapsed)
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Done image [{i + 1:<3}/ {args.samples}], '
                      f'fps: {fps:.2f} img / s, gpu mem: {max_memory} M')

        if (i + 1) == args.samples:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.2f} img / s, peak gpu mem: {max_memory} M')
            bm = sum(t_back)/len(t_back)*1000; nm = sum(t_neck)/len(t_neck)*1000
            tm = sum(t_total)/len(t_total)*1000
            print(f"per-stage  backbone={bm:.2f}ms  neck={nm:.2f}ms  head+rest={tm-bm-nm:.2f}ms  total={tm:.2f}ms")
            print(f"breakdown  backbone={bm/tm*100:.1f}%  neck={nm/tm*100:.1f}%  head+rest={(tm-bm-nm)/tm*100:.1f}%")
            break


if __name__ == '__main__':
    main()
