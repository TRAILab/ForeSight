# SparseDrive R50 Baselines — Paper vs. Reproduced (Feb 2026)

## Intro

This document establishes the baseline reference numbers for SparseDrive-Small (R50) Stage 2 across four sources: the original paper, the project README checkpoint, and two families of reproduced runs — inference-only using the official released checkpoint, and independently trained runs on our own hardware. Three sets of reproduced evaluations were performed in February 2026: inference-only on the paper checkpoint evaluated with a flash-attention config (4GPU), without flash attention (8GPU noflash), and trained-from-scratch baselines on DGX 4GPU and Apollo 8GPU. Establishing these baselines was necessary to confirm our code environment replicates the reported numbers before investing in improvements.

## Method

**Paper / README**: Numbers taken directly from the SparseDrive paper (Table 1) and from the repository README. The paper reports an NDS-optimized training recipe on 8 GPUs.

**Inference-only reproduction**: The official `ckpt/sparsedrive_stage2.pth` checkpoint was downloaded and evaluated on the NuScenes val set using two configs:
- `sparsedrive_small_stage2` — standard config (4GPU evaluation, flash attention)
- `sparsedrive_small_stage2_noflash` — 8GPU evaluation without FlashAttention

These runs do not involve any retraining; they measure whether our evaluation pipeline matches the reported numbers given the same weights.

**Trained baselines**: Full stage1 (100 epochs) + stage2 (10 epochs) training runs from scratch:
- **DGX 4GPU** (`stage2_4gpu_bs24`): 4 GPU, bs=24, lr=1.5e-4, backbone lr_mult=0.1, loaded from `ckpt/sparsedrive_stage1.pth`. Run 2026-02-16.
- **Apollo 8GPU** (`stage2_8gpu_noflash`): 8 GPU, bs=48, lr=3e-4, all MultiheadAttention (no flash), num_dn_groups=0. Run 2026-02-13/15.

Both trained baselines use `num_dn_groups=0` (denoising disabled) and `rot3d_range=[0,0]` (rotation augmentation disabled in stage2).

## Results

### High-Level Comparison

| Source | det mAP | det NDS | track AMOTA | track IDS | map mAP | car minADE | car EPA | plan L2 (avg) | plan CR (avg) |
|--------|---------|---------|-------------|-----------|---------|------------|---------|--------------|--------------|
| **Paper** | 0.418 | 0.525 | 0.386 | 886 | 0.551 | 0.62 | 0.482 | 0.61 | **0.080%** |
| **README** | 0.4151 | 0.5257 | 0.372 | 810 | 0.5656 | 0.61 | 0.492 | 0.61 | 0.100% |
| **Reproduced (4gpu, ckpt)** | 0.4141 | 0.5250 | 0.3711 | 815 | 0.5656 | 0.6129 | 0.492 | 0.6063 | 0.096% |
| **Reproduced (8gpu noflash, ckpt)** | 0.4152 | 0.5255 | 0.3711 | 859 | 0.5657 | 0.6109 | 0.4911 | 0.6066 | 0.096% |
| **Trained DGX 4GPU (bs24)** | 0.4132 | 0.5232 | 0.3776 | 1045 | 0.5528 | 0.636 | 0.492 | 0.636 | 0.133% |
| **Trained Apollo 8GPU (noflash)** | 0.4076 | 0.5187 | 0.3714 | 1088 | 0.5471 | 0.6148 | 0.491 | 0.600 | 0.104% |

Work dirs: `sparsedrive_small_stage2` (2026-02-12), `sparsedrive_small_stage2_noflash` (2026-02-13), `sparsedrive_r50_stage2_4gpu_bs24` (2026-02-16), `sparsedrive_r50_stage2_8gpu_noflash` (2026-02-15).

### Detailed: Trained DGX 4GPU bs24 Baseline

**Detection (NuScenes)**

| Metric | Value |
|--------|-------|
| NDS | **0.5232** |
| mAP | **0.4132** |
| mATE | 0.5537 |
| mASE | 0.2749 |
| mAOE | 0.5437 |
| mAVE | 0.2702 |
| mAAE | 0.1917 |

Per-class AP highlights: Car AP@0.5=0.353, AP@4.0=0.832. Trailer AP@0.5=0.000 (non-functional at tight thresholds). Traffic cone AP@0.5=0.520 (strongest class).

**Tracking**

| Class | AMOTA | AMOTP | Recall | FAF |
|-------|-------|-------|--------|-----|
| Car | 0.617 | 0.875 | 0.670 | 125 |
| Pedestrian | 0.454 | 1.206 | 0.574 | 69 |
| Motorcycle | 0.408 | 1.245 | 0.506 | 15 |
| Bicycle | 0.406 | 1.199 | 0.540 | 24 |
| Truck | 0.366 | 1.262 | 0.514 | 46 |
| Bus | 0.391 | 1.307 | 0.517 | 20 |
| **Trailer** | **0.001** | 1.662 | **0.220** | **47** |
| **Overall** | **0.3776** | **1.2509** | **0.5058** | 49.4 |

Other: IDS=1045, FRAG=626, MT=2488, ML=2261, MOTA=0.346, MOTP=0.629

**HD Map**

| Category | AP@0.5 | AP@1.0 | AP@1.5 | AP (avg) |
|----------|--------|--------|--------|----------|
| ped_crossing | 0.185 | 0.552 | 0.726 | 0.488 |
| divider | 0.355 | 0.632 | 0.752 | 0.580 |
| boundary | 0.290 | 0.674 | 0.810 | 0.591 |
| **mAP** | | | | **0.553** |

**Motion Prediction**

| Class | EPA | min_ADE | min_FDE | Miss Rate |
|-------|-----|---------|---------|-----------|
| Car | 0.492 | 0.636 | 1.000 | 0.133 |
| Pedestrian | 0.411 | 0.728 | 1.066 | 0.147 |

**Planning**

| Metric | 0.5s | 1.0s | 1.5s | 2.0s | 2.5s | 3.0s | Avg |
|--------|------|------|------|------|------|------|-----|
| obj_col | 0.742% | 0.713% | 0.684% | 0.659% | 0.649% | 0.638% | 0.670% |
| obj_box_col | 0.000% | 0.020% | 0.052% | 0.098% | 0.168% | 0.283% | **0.133%** |
| L2 (m) | 0.197 | 0.310 | 0.444 | 0.603 | 0.787 | 0.995 | **0.636** |

## Discussion

**Inference-only reproduction closely matches the README**: NDS=0.5255 (vs. 0.5257), mAP=0.4152 (vs. 0.4151), plan L2=0.6066 (vs. 0.61). The small residual gap is within expected run-to-run variance from sequence ordering. The FlashAttention variant (4gpu) gives nearly identical results to the noflash variant (8gpu).

**The paper's collision rate (0.080%) is better than README/reproduced (0.096–0.100%)**: This ~0.016–0.020pp gap likely reflects checkpoint differences (a slightly better final epoch checkpoint used for the paper) or minor protocol differences in evaluating occluded object collisions. The README checkpoint is the public release, which may not be the exact paper checkpoint.

**Our trained baselines are slightly weaker than the paper checkpoint across the board**:
- DGX 4GPU: NDS=0.5232 (−0.0025), AMOTA=0.3776 (+0.006 vs. paper, −0.004 vs. README ckpt), obj_box_col=0.133% (much higher than paper)
- Apollo 8GPU: NDS=0.5187 (−0.0045 vs. DGX, −0.0068 vs. paper), obj_box_col=0.104%

The DGX trained baseline's higher collision rate (0.133% vs. 0.096% for inference on the paper checkpoint) is the most important gap. Likely causes: the paper uses a more optimized training recipe, possible checkpoint selection on a held-out val subset, or differences in the stage1 checkpoint we loaded.

**Trailer is catastrophic across all baselines**: AMOTA=0.001 in the DGX baseline, and AP@0.5=0.000 in the official checkpoint too. This is a fundamental architecture/dataset issue, not specific to our training.

**car FDE = 1.000 in the DGX trained baseline** is suspicious — the exact integer value suggests a mode-collapse artifact where the motion head converges to a fixed displacement. This does not appear in the checkpoint inference runs (FDE=0.964–0.966), suggesting our stage2 training is slightly suboptimal for the motion head.

**obj_col = 0.670% across all runs**: This measures GT ego trajectory collisions with GT objects — a lower bound reflecting unavoidable dataset collisions, not model error. It is stable across all variants as expected.

## Future Work

- Identify what differs between the paper's training recipe and our DGX bs24 run to explain the 0.133% vs. 0.096% collision gap. Candidates: more stage2 epochs, different loss weights, checkpoint selection.
- Use the inference-only reproduced numbers (0.096% collision, NDS=0.5255) as the reference ceiling for R50 performance, rather than our trained baselines, when benchmarking improvements.
- Investigate the car FDE = 1.000 artifact in the trained baseline; this exact value does not appear when using the paper's released checkpoint.
- Run the Apollo and DGX baselines with `num_dn_groups=5` (DN enabled) to match the best-practice training recipe documented in `2026_02_27_stage1_detection_ablations.md`.
