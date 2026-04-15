# Auxiliary 2D Supervision for SparseDrive
2026-04-15

## TODO

- Parse metrics for DGX job 3592 once complete and fill in results table and discussion once metrics are available
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
| `dgx` | `sparsedrive_r50_stage1_4gpu_aux2d` | `3592` | `running` |

| Config | Server | det mAP | det NDS | Notes |
|--------|--------|---------|---------|-------|
| stage1 baseline | — | 0.413 | 0.523 | from `sparsedrive_r50_stage2_4gpu_bs24` baseline |
| stage1_4gpu_aux2d | DGX | — | — | job 3592 running |

## Discussion

*(pending results)*

## Future Work

- Ablate aux-2D loss weights — in particular whether the centerness and centers2d losses are complementary or redundant.
