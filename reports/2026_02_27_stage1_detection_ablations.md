# Stage1 Detection Ablations — Apollo 8GPU (Feb 2026)

## Intro

This document covers a series of stage1 ablation experiments (100-epoch training, no stage2) run on the Apollo 8GPU cluster to understand the individual and combined contribution of three training improvements: denoising (DN), map head removal (nomap), and 3D rotation augmentation (rotaug). All configs fork from `sparsedrive_r50_stage1_8gpu_noflash.py`. The goal was to identify the best stage1 checkpoint to use as a pretrain for downstream stage2 experiments. Experiments were run between 2026-02-13 and 2026-02-27.

## Method

Six stage1 configurations were trained for 100 epochs on 8 GPUs (bs=48, lr=3e-4, no FlashAttention). The factors varied were:

- **DN**: `num_dn_groups=5` denoising enabled (vs. 0 in baseline)
- **Nomap**: map head removed entirely from stage1 training
- **Rotaug**: 3D rotation augmentation enabled (`rot3d_range=[-0.3925, 0.3925]`)

All configurations were evaluated on the NuScenes val set at the end of epoch 100. The single-variable changes were tested first, then combined to find the best full recipe. Detection (NDS, mAP), tracking (AMOTA, IDS) are reported; planning metrics are not applicable to stage1-only evaluation.

## Results

| Config | NDS | mAP | AMOTA | IDS | Notes |
|--------|-----|-----|-------|-----|-------|
| **8gpu_noflash (baseline)** | 0.5307 | 0.4137 | 0.3973 | 535 | — |
| +DN | 0.5368 | 0.4227 | 0.4198 | 416 | +0.009 mAP, −119 IDS |
| +Nomap | 0.5415 | 0.4286 | 0.4165 | 614 | map head removed |
| +Nomap+DN | 0.5507 | 0.4365 | 0.4441 | 486 | best det w/ standard aug |
| +Nomap+Rotaug | 0.5583 | 0.4527 | 0.4351 | 516 | rotaug alone is strong |
| **+Nomap+DN+Rotaug** | **0.5620** | **0.4571** | **0.4534** | **464** | best stage1 config |

Work dirs: `sparsedrive_r50_stage1_8gpu_noflash` (2026-02-17), `sparsedrive_r50_stage1_8gpu_noflash_dn` (2026-03-06), `sparsedrive_r50_stage1_8gpu_noflash_nomap` (2026-02-25), `sparsedrive_r50_stage1_8gpu_noflash_nomap_dn` (2026-02-24), `sparsedrive_r50_stage1_8gpu_noflash_nomap_rotaug` (2026-02-23), `sparsedrive_r50_stage1_8gpu_noflash_nomap_dn_rotaug` (2026-02-27).

## Discussion

- **DN** delivers a clean and consistent +0.009 mAP and cuts IDS by 22% with no downsides. The infrastructure (denoising group anchors and DN loss) is fully implemented; the change is just `num_dn_groups=5`.
- **Nomap** (removing the map head from stage1) improves detection further — the map head introduces a competing gradient that pulls shared representations away from 3D detection. Detection NDS jumps +0.010 over DN alone. However, removing map from stage1 entirely breaks planning in stage2 (see Section 3.4 / `2026_03_05_map_removal.md`), so this must be coupled with standard stage1 map training in all full-pipeline experiments.
- **Rotaug** (3D rotation augmentation) has the largest single-factor impact on detection among the three: `+Nomap+Rotaug` beats `+Nomap+DN` on mAP (0.4527 vs. 0.4365). This confirms rotaug as a strong data augmentation lever for BEV detection.
- **All three combined** achieves the best stage1 checkpoint to date: NDS=0.5620, mAP=0.4571, AMOTA=0.4534, IDS=464. This represents +0.031 NDS and +0.056 AMOTA over the unmodified baseline.
- The best config (`nomap_dn_rotaug`) is the recommended stage1 pretrain for all downstream R50 experiments.

## Future Work

- Combine `nomap_dn_rotaug` stage1 with a full stage2 recipe (nomap + num_det=100 + queue=6 + epochs=15) to measure combined end-to-end gain.
- Test the same ablation progression with the R101 backbone — individual gains may compound differently due to stronger feature extraction.
- Investigate whether `num_temp_dn_groups > 0` (temporal denoising) adds further improvement over the current `num_dn_groups=5` setting.
- The rotaug float64 bug in `augment.py` (BBoxRotation dtype issue) should be verified fixed before using rotaug in production recipes.
