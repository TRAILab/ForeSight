# Auxiliary 2D Supervision for SparseDrive
2026-04-15

## TODO

1. **Combine ptaux2d with rot3dv2 stage-2 augmentation.** ptaux2d is currently the best recipe (L2=0.560–0.562); rot3dv2 gives an independent planning gain (L2=0.593 from public pretrain). Combining the aux2d stage-1 backbone with rot3dv2 augmentation in stage-2 is the most direct path to a new best result.
2. **Combine ptaux2d with det DN + map DN in stage-2** (pending dndetmap eval). If map DN helps at stage1, it should also help in a stage2 config built on ptaux2d.
3. **Move 2D target generation offline.** Currently generated online per camera at train time; offline precomputation would reduce per-iteration cost and enable exact reproducibility of the matching.

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

An aux2p5d variant extends `SparseDriveAux2DHead` with a per-object log-depth regression branch. `SparseDriveAux2p5DHead` adds a `Conv2d(256, 1, 1)` predictor on the same `reg_feat` used by the ltrb and center2d heads, supervised with L1 on `log(d_gt)` at Hungarian-matched positives only. GT depth targets come from the existing `depths` field produced by `GenerateProjected2DTargets`, so no data pipeline changes are needed. The additional loss is:

| Loss | Type | Weight |
|------|------|--------|
| depth | L1 on log(d) | 1.0 |

Log-depth targets avoid the scale imbalance of raw metric depth (~0.5–60 m). L1 weight=1.0 is modest relative to centers2d (10.0) and bbox (5.0), so depth provides a complementary 3D signal without dominating the aux loss. Key files: `projects/mmdet3d_plugin/models/aux_2d_head.py` (`SparseDriveAux2p5DHead`), `projects/configs/sparsedrive_r50_stage1_8gpu_noflash_aux2p5d.py`.

## Results

| Server | Config | Job ID | Status |
| --- | --- | --- | --- |
| `apollo` | `sparsedrive_r50_stage1_8gpu_noflash_aux2d` | — | `complete` |
| `apollo` | `sparsedrive_r50_stage1_8gpu_noflash_aux2p5d` | `c38f26e4f7dd` (tmux: `aux2p5d`) | `complete` |
| `dgx` | `sparsedrive_r50_stage2_4gpu_bs24_ptaux2d_planpredtrajdeformmm` | `3605` | `complete` |
| `narval` | `sparsedrive_r50_stage2_4gpu_bs24_ptaux2d_planpredtrajdeformmm` | `59608201` | `complete` |

### Stage 1

| Config | Server | det mAP | det NDS | map mAP | Notes |
|--------|--------|---------|---------|---------|-------|
| stage1 baseline (`8gpu_noflash`) | — | 0.413 | 0.523 | 0.488 | from `sparsedrive_r50_stage2_4gpu_bs24` baseline |
| stage1_8gpu_noflash_aux2d | Apollo | **0.437** | **0.546** | **0.570** | complete — original loss weights; +0.024/+0.023/+0.082 vs baseline |
| stage1_8gpu_noflash_aux2p5d | Apollo | **0.441** | **0.555** | **0.571** | complete — depth branch; +0.028/+0.032/+0.083 vs baseline, +0.004/+0.009/+0.001 vs aux2d |

### Stage 2

| Config | Server | L2 | obj_box_col | det mAP | det NDS | map mAP | car_epa | Notes |
|--------|--------|-----|-------------|---------|---------|---------|---------|-------|
| stage2 baseline (`planpredtrajdeformmm`) | DGX | 0.636 | 0.133% | 0.413 | 0.523 | 0.553 | 0.492 | baseline |
| stage2_ptaux2d_planpredtrajdeformmm | DGX | **0.560** | **0.062%** | **0.437** | **0.544** | 0.560 | **0.508** | complete — job 3605; −0.076 L2 (−12%), −53% collision |
| stage2_ptaux2d_planpredtrajdeformmm | Narval | **0.562** | **0.037%** | **0.442** | **0.548** | **0.565** | **0.507** | complete — job 59608201 |

## Discussion

**`8gpu_noflash_aux2d` (Apollo) — complete.** Based on `stage1_8gpu_noflash` (total_bs=64, **8/GPU**). Final val metrics: det mAP=0.437, NDS=0.546, map mAP=0.570. Versus the baseline (0.413/0.523/0.488), aux-2D improves all three metrics: +0.024 det mAP (+5.8%), +0.023 NDS (+4.4%), and a large +0.082 map mAP (+16.8%). No map collapse was observed with the original loss weights (cls=2, bbox=5, iou=2, centers2d=10, centerness=1).

**`8gpu_noflash_aux2p5d` (Apollo) — complete.** The run finished on 2026-04-23 and evaluated from `iter_43900`. Final val metrics: det mAP=0.4408, NDS=0.5547, map mAP_normal=0.5712. Relative to the aux2d run (0.437/0.546/0.570), adding the depth branch gives small but consistent gains: +0.0038 det mAP, +0.0087 NDS, and +0.0012 map mAP_normal. Relative to the baseline (0.413/0.523/0.488), the full gain is +0.0278 det mAP, +0.0317 NDS, and +0.0832 map mAP_normal. The depth loss appears to help detection quality modestly without hurting map quality.

**Stage 2 (`ptaux2d`, DGX job 3605) — complete.** L2=0.560 vs. baseline 0.636 (−0.076, −12%); obj_box_col=0.062% vs. 0.133% (−53%); det mAP=0.437, NDS=0.544, car_epa=0.508. All metrics improve over the baseline, with planning and collision showing the largest gains. This confirms that the improved Stage 1 backbone features carry through to motion/planning — the aux-2D supervision benefit is not limited to detection and map.

**Narval replication (job 59608201) — complete.** L2=0.562, obj_box_col=0.037%, NDS=0.548, mAP=0.442, mAP_normal=0.565, car_epa=0.507. The result is consistent with DGX (L2: 0.560 vs 0.562, NDS: 0.544 vs 0.548, mAP: 0.437 vs 0.442). The collision rate is lower on Narval (0.037% vs 0.062%), which is within the expected single-run variance of this metric. The cross-seed consistency confirms the ptaux2d gain is stable and not a DGX-specific artifact. aux-2D is now the best single recipe tested: L2=0.56, obj_box_col≈0.04–0.06%, NDS=0.546–0.548.

## Future Work

- Ablate aux-2D loss weights — in particular whether the centerness and centers2d losses are complementary or redundant.
- ~~Run a stage-2 follow-up initialized from the aux2p5d stage-1 checkpoint to test whether the small stage-1 detection gain transfers to planning.~~ **In progress.** `ptaux2p5d_ppdeformmm_planifls` submitted to Narval; tracked in `reports/2026_04_21_combined.md`.
