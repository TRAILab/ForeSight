# Stage2 with DN Pretrain — Apollo 8GPU (Mar 2026)

## Intro

This document covers experiments evaluating the end-to-end impact of using a denoising-pretrained stage1 checkpoint for stage2 training, compared to the standard Apollo 8GPU baseline. Two stage2 variants were evaluated: an Apollo 8GPU run (`pretrainv2_noflash`, `with_map=False`) and a DGX 4GPU run (`bs24_pt2`, with map head active). The goal was to determine whether the detection improvements from DN pretraining (documented in `2026_02_27_stage1_detection_ablations.md`) carry through to motion and planning metrics. Experiment completed 2026-03-08.

## Method

**Primary experiment** (`stage2_8gpu_pretrainv2_noflash`): Loads the `stage1_8gpu_noflash_dn` checkpoint, then runs stage2 with `with_map=False` on 8 GPUs (bs=48, lr=3e-4). No map head active during stage2.

**Secondary experiment** (`stage2_4gpu_bs24_pt2`): Loads the same DN stage1 checkpoint, then runs stage2 on 4 GPUs (bs=24, lr=1.5e-4) with the map head active. This tests whether the map head in stage2 interacts negatively with DN pretraining.

Both were compared against their respective baselines: the Apollo 8GPU baseline (`stage2_8gpu_noflash`) and the DGX 4GPU bs24 baseline (`stage2_4gpu_bs24`).

## Results

**Primary: Apollo 8GPU DN stage2 (`pretrainv2_noflash`)**

| Metric | Apollo baseline | DN stage2 | Δ |
|--------|----------------|-----------|---|
| NDS | 0.5187 | **0.5415** | +0.023 |
| mAP | 0.4076 | **0.4257** | +0.018 |
| AMOTA | 0.3714 | **0.4179** | +0.046 |
| IDS | 1088 | **425** | −663 |
| car ADE | 0.6148 | **0.6166** | +0.002 |
| Planning L2 | 0.600 | **0.602** | +0.002 |
| obj_box_col | 0.104% | **0.092%** | −0.012pp |

**Secondary: DGX 4GPU DN stage2 with map (`bs24_pt2`)**

| Metric | DGX bs24 baseline | DN stage2 + map | Δ |
|--------|------------------|-----------------|----|
| NDS | 0.5232 | 0.5332 | +0.010 |
| AMOTA | 0.3776 | 0.4180 | +0.040 |
| IDS | 1045 | 520 | −525 |
| Planning L2 | 0.636 | 0.700 | **+0.064 (worse)** |
| obj_box_col | 0.133% | 0.133% | no change |

Work dirs: `sparsedrive_r50_stage2_8gpu_pretrainv2_noflash` (2026-03-08), `sparsedrive_r50_stage2_4gpu_bs24_pt2` (2026-03-10).

## Discussion

- **DN pretraining is a net positive when `with_map=False` in stage2**: detection and tracking improve substantially (+0.023 NDS, +0.046 AMOTA, −663 IDS) and collision rate drops slightly (0.092% vs. 0.104%). Planning L2 is essentially unchanged (0.602 vs. 0.600), meaning the large tracking improvement does not hurt planning.
- **The map head in stage2 is the critical variable**: the secondary experiment (`bs24_pt2`, with map) yields significantly worse planning L2 (0.700 vs. 0.602 for the nomap variant). The map head's gradient competition in stage2 cancels out the benefits of DN pretraining for motion and planning.
- **IDS improvement is remarkable**: DN pretraining cuts ID switches from 1088 → 425 (−61%). The denoising training likely teaches the instance bank to maintain more temporally stable instance identities.
- **Recommendation**: DN pretraining (`num_dn_groups=5`) should be standard in the stage1 recipe for all future experiments. Stage2 should use `with_map=False` (built from a nomap-capable base config, not a runtime override) to preserve the planning benefits.

## Future Work

- Combine DN pretraining with the best stage1 recipe (nomap + rotaug) and best stage2 recipe (nomap + queue=6 + num_det=100 + 15 epochs) for a full-stack experiment.
- Investigate the root cause of the map-head planning degradation in `bs24_pt2` — is it gradient competition, learning rate mismatch, or the map head consuming capacity that should go to motion/planning?
- Test temporal DN (`num_temp_dn_groups=3`) on top of `num_dn_groups=5` to measure additional tracking consistency gain.
