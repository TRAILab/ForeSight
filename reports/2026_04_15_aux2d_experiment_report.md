# Auxiliary 2D Supervision for SparseDrive
2026-04-15

## TODO

- Move 2D target generation offline; compare against online reprojection
- Parse metrics for Narval job `59608201` (`stage2_ptaux2d`) once complete and confirm DGX result.
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
| `apollo` | `sparsedrive_r50_stage1_8gpu_noflash_aux2d` | — | `complete` |
| `dgx` | `sparsedrive_r50_stage2_4gpu_bs24_ptaux2d_planpredtrajdeformmm` | `3605` | `complete` |
| `narval` | `sparsedrive_r50_stage2_4gpu_bs24_ptaux2d_planpredtrajdeformmm` | `59608201` | `running` |

### Stage 1

| Config | Server | det mAP | det NDS | map mAP | Notes |
|--------|--------|---------|---------|---------|-------|
| stage1 baseline (`8gpu_noflash`) | — | 0.413 | 0.523 | 0.488 | from `sparsedrive_r50_stage2_4gpu_bs24` baseline |
| stage1_8gpu_noflash_aux2d | Apollo | **0.437** | **0.546** | **0.570** | complete — original loss weights; +0.024/+0.023/+0.082 vs baseline |

### Stage 2

| Config | Server | L2 | obj_box_col | det mAP | det NDS | map mAP | car_epa | Notes |
|--------|--------|-----|-------------|---------|---------|---------|---------|-------|
| stage2 baseline (`planpredtrajdeformmm`) | DGX | 0.636 | 0.133% | 0.413 | 0.523 | 0.553 | 0.492 | baseline |
| stage2_ptaux2d_planpredtrajdeformmm | DGX | **0.560** | **0.062%** | **0.437** | **0.544** | 0.560 | **0.508** | complete — job 3605; −0.076 L2 (−12%), −53% collision |
| stage2_ptaux2d_planpredtrajdeformmm | Narval | — | — | — | — | — | — | running — job 59608201 |

## Discussion

**`8gpu_noflash_aux2d` (Apollo) — complete.** Based on `stage1_8gpu_noflash` (total_bs=64, **8/GPU**). Final val metrics: det mAP=0.437, NDS=0.546, map mAP=0.570. Versus the baseline (0.413/0.523/0.488), aux-2D improves all three metrics: +0.024 det mAP (+5.8%), +0.023 NDS (+4.4%), and a large +0.082 map mAP (+16.8%). No map collapse was observed with the original loss weights (cls=2, bbox=5, iou=2, centers2d=10, centerness=1).

**Stage 2 (`ptaux2d`, DGX job 3605) — complete.** L2=0.560 vs. baseline 0.636 (−0.076, −12%); obj_box_col=0.062% vs. 0.133% (−53%); det mAP=0.437, NDS=0.544, car_epa=0.508. All metrics improve over the baseline, with planning and collision showing the largest gains. This confirms that the improved Stage 1 backbone features carry through to motion/planning — the aux-2D supervision benefit is not limited to detection and map. A Narval replication run (job 59608201) is in progress to confirm cross-seed stability.

## Future Work

- Ablate aux-2D loss weights — in particular whether the centerness and centers2d losses are complementary or redundant.
