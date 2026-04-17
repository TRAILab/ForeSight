# Auxiliary 2D Supervision for SparseDrive
2026-04-15

## TODO

- Parse metrics for Apollo job `8gpu_noflash_aux2d` once complete and fill in results table
- Move 2D target generation offline; compare against online reprojection
- Run aux-2D on Stage 2 to see whether the feature regularisation benefit carries through to motion/planning metrics.
- Add depth prediction branch to `SparseDriveAux2DHead` (log-depth `Conv2d` on reg branch, supervised by existing `depths` targets)
- Use top-k aux head predictions to unproject `(center2d, depth)` → 3D lidar anchors as data-driven query initialisation alongside k-means prior

## Intro

Tests whether auxiliary image-space supervision during Stage 1 training improves camera feature quality for 3D detection in `SparseDrive`. Motivated by `StreamPETR`, which trains an auxiliary 2D head alongside its main 3D decoder. The aux branch is train-only and leaves the core detection pathway unchanged.

## Method

A `SparseDriveAux2DHead` is attached to FPN level 0 (stride=4) features, before deformable aggregation. It predicts per-location class scores, 2D boxes, 2D centres, and centerness, supervised by five losses matching `StreamPETR`'s design:

| Loss | Type | Weight |
|------|------|--------|
| cls | QualityFocalLoss (sigmoid) | 2.0 |
| bbox | L1 | 5.0 |
| iou | GIoULoss | 2.0 |
| centers2d | L1 | 10.0 |
| centerness | GaussianFocalLoss | 1.0 |

Matching uses `HungarianAssigner2D` (ported from `StreamPETR`) with the same cost weights. 2D targets are generated online per camera: 3D box corners are projected via the augmented `lidar2img`, the convex hull of visible corners is intersected with the image canvas, and the projected centre provides the `centers2d` target.

Key differences from `StreamPETR`:
- **Supervision-only** — `StreamPETR` also uses the aux head to select top-k spatial tokens as decoder queries, but this is not needed in SparseDrive: deformable attention already solves the same problem by sampling sparse reference points per query rather than attending over all tokens. The aux head here purely provides a supervision signal.
- **Finer stride** — operates at stride=4 (FPN level 0) vs. `StreamPETR`'s stride=16. `StreamPETR` is constrained to coarser features because token selection must produce a tractable query set; without that constraint, the finest FPN level is preferable for supervision.
- **Online 2D targets** — projected from 3D GT at train time with augmentation-aware `lidar2img`; `StreamPETR` uses precomputed offline annotations.

## Results

| Server | Config | Job ID | Status |
| --- | --- | --- | --- |
| `dgx` | `sparsedrive_r50_stage1_4gpu_aux2d` | `3592` | `cancelled` — map failure (wrong base config) |
| `dgx` | `sparsedrive_r50_stage1_4gpu_aux2d_lwloss` | `3594` | `cancelled` — map failure (wrong base config) |
| `dgx` | `sparsedrive_r50_stage1_4gpu_bs24_aux2d` | `3595` | `cancelled` — superseded by Apollo run |
| `apollo` | `sparsedrive_r50_stage1_8gpu_noflash_aux2d` | — | `running` |

| Config | Server | det mAP | det NDS | map mAP | Notes |
|--------|--------|---------|---------|---------|-------|
| stage1 baseline (`8gpu_noflash`) | — | 0.413 | 0.523 | 0.488 | from `sparsedrive_r50_stage2_4gpu_bs24` baseline |
| stage1_4gpu_aux2d | DGX | — | — | ~0.018 | cancelled — same map failure as BS=16/GPU baseline |
| stage1_4gpu_aux2d_lwloss | DGX | — | — | ~0.017 | cancelled — same map failure as BS=16/GPU baseline |
| stage1_8gpu_noflash_aux2d | Apollo | — | — | — | running — original loss weights |

## Discussion

**Jobs 3592 and 3594 — wrong base config.** Both runs used `stage1_4gpu` (total_bs=64, **16/GPU**) as the base, which has a known map head failure at BS>6/GPU (see `2026_04_01_map_head_batchsize_failure.md`). The baseline `stage1_4gpu` itself only achieves `mAP_normal=0.019` at end of training for the same reason — making the ~0.018 val results in jobs 3592/3594 indistinguishable from the baseline and not indicative of aux2d-specific map collapse. The diagnosis of a gradient conflict was premature.

**Current run (`8gpu_noflash_aux2d`, Apollo).** Based on `stage1_8gpu_noflash` (total_bs=64, **8/GPU**), which is the correct baseline that achieves `mAP_normal≈0.41` at iter 8780. Uses original aux2d loss weights (cls=2, bbox=5, iou=2, centers2d=10, centerness=1). If map collapse is observed, reduce weights or investigate gradient conflict.

## Future Work

- Ablate aux-2D loss weights — in particular whether the centerness and centers2d losses are complementary or redundant.
