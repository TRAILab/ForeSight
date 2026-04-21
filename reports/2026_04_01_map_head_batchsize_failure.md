# Map Head Batch-Size Failure — Stage-Specific Investigation (Apr 2026)

## Abstract

This report documents a reproducible HD map head convergence failure that is **confirmed for stage 2**, not uniformly for all training stages. In stage 2, per-GPU batch size **6/GPU works** while **12/GPU fails**: `sparsedrive_r50_stage2_4gpu_bs24` (24 total batch, 4 GPUs, 6/GPU) reaches map mAP=0.553, while `sparsedrive_r50_stage2_4gpu` (48 total batch, 4 GPUs, 12/GPU) reaches only map mAP=0.078. The 8-GPU control `sparsedrive_r50_stage2_8gpu_noflash` (48 total batch, 8 GPUs, 6/GPU, map mAP=0.547) rules out total batch size and nominal LR as the primary cause.

The earlier wording of this report overstated the result as a universal threshold of "per-GPU batch size > 6 fails." That is too broad. Current stage-1 evidence shows **6/GPU works**, **8/GPU works**, and a quick check at **16/GPU failed**, but stage 1 was not systematically ablated here. The strongest supported conclusion is therefore:

- Stage 2: safe at `6/GPU`, broken at `12/GPU`
- Stage 1: safe at `6/GPU` and `8/GPU`, observed broken at `16/GPU`
- Unknown: exact stage-1 failure boundary between `8/GPU` and `16/GPU`

## Intro

This document covers a systematic investigation into a stage-2 failure mode where the HD map head catastrophically fails to converge at high per-GPU batch size. The motivating observation was that `sparsedrive_r50_stage2_4gpu.py` (12/GPU) fails badly while `sparsedrive_r50_stage2_4gpu_bs24.py` (6/GPU) converges normally. Because both runs train the same map head, the goal was to determine whether the root cause was per-GPU batch size, total batch size, learning rate, BN behavior, or some map-specific implementation issue.

The original April 1 ablation was a stage-2 study. Stage-1 evidence is included only to clarify the final conclusion and avoid overgeneralizing the stage-2 threshold.

## Method

The core stage-2 comparison used these configs:

| Config | per-GPU BS | LR | map mAP | Status |
|--------|-----------|----|---------|--------|
| `4gpu` (bs48) | 12 | 3e-4 | 0.078 | failed |
| `4gpu_maplrdiv4` | 12 | 3e-4 | 0.074 | failed |
| `4gpu_bs24` | 6 | 1.5e-4 | 0.553 | converged |
| `8gpu_noflash` | 6 | 3e-4 | 0.547 | converged |

This setup isolates the main variables:

- `4gpu` vs `8gpu_noflash` keeps total batch size fixed at 48 while changing per-GPU batch size from 12 to 6.
- `4gpu` vs `4gpu_bs24` changes both total batch and per-GPU batch, but agrees with the 8-GPU control.
- `4gpu_maplrdiv4` tests whether reducing map-head optimization scale fixes the issue.

Three follow-up ablations were submitted on 2026-04-01:

| Config | Change | Purpose |
|--------|--------|---------|
| `4gpu_gradacc` | `cumulative_iters=2` | reduce optimizer-step variance without changing 12/GPU forward passes |
| `4gpu_normeval` | `norm_eval=True` | test whether BN running stats are the cause |
| `4gpu_mapfeatnograd` | `feat_grad=False` | test whether map anchor-init gradients destabilize training |

The map-head code path was also audited, including DAF, dropout, loss normalization, Hungarian assignment, FFN layers, and `GroupInBatchSampler`, to check for explicit batch-size-dependent behavior.

## Results

### Stage 2

The stage-2 outcome is unambiguous:

- `6/GPU` works: `4gpu_bs24` reaches map mAP=0.553 and `8gpu_noflash` reaches 0.547.
- `12/GPU` fails: `4gpu`, `4gpu_maplrdiv4`, `4gpu_normeval`, `4gpu_mapfeatnograd`, and `4gpu_gradacc` all fail to converge.

Key ablation results:

| Config | per-GPU BS | map_mAP | map_loss_line_5 (iter 51) | map_loss_line_5 (iter 1734) | Status |
|--------|-----------|---------|--------------------------|----------------------------|--------|
| `4gpu_bs24` | 6 | 0.553 | 0.21 | ~0.10 | converged |
| `4gpu_normeval` | 12 | 0.079 | ~0.88 | — | failed |
| `4gpu_mapfeatnograd` | 12 | 0.074 | ~0.93 | — | failed |
| `4gpu_gradacc` | 12 | TBD | 0.93 | 0.78 | failed |

`gradacc` is especially informative: even after reducing optimizer-step variance, the `12/GPU` run still plateaued near map loss 0.75 rather than approaching the `6/GPU` baseline of ~0.10. At iter 51, the `12/GPU` gradacc run already had `map_line_5=0.93` while the `6/GPU` baseline was at 0.21, indicating that the failure appears almost immediately.

The strongest current hypothesis is initialization-time scene composition. With fixed seed=0, `GroupInBatchSampler` assigns different deterministic scene groups to the first batch. At `6/GPU`, rank 0 sees `perm[0..5]`; at `12/GPU`, it sees `perm[0..11]`. The extra groups appear to be harder scenes for the map head, producing a much worse starting point that the model does not recover from.

### Stage 1

This report does **not** present a controlled stage-1 ablation. The current evidence relevant to stage 1 is:

- `6/GPU` works: `sparsedrive_r50_stage1_4gpu_bs24_aux2d`
- `8/GPU` works: standard `sparsedrive_r50_stage1_8gpu_noflash` baseline and `stage1_8gpu_noflash_aux2d`
- `16/GPU` failed in a quick check: `sparsedrive_r50_stage1_4gpu`

So stage 1 is more tolerant than stage 2, but its exact failure boundary is still unknown.

## Discussion

The main conclusion is stage-specific. For **stage 2**, the map head is reliable at `6/GPU` and broken at `12/GPU`; that conclusion is directly supported by controlled comparisons and multiple failed ablations. For **stage 1**, the data only support that `6/GPU` and `8/GPU` are safe while `16/GPU` is not. The earlier wording of this report overgeneralized the stage-2 result into a universal rule, which is not supported by the current evidence.

The failed `normeval`, `mapfeatnograd`, and `gradacc` ablations rule out several simple explanations: BN running statistics, anchor-init gradients, and optimizer-step variance are not the primary cause of the stage-2 collapse. The `8gpu_noflash` control also rules out total batch size and nominal LR. No explicit code-level batch-size bug was found in the audited map-head path. The best remaining explanation is that the map head is highly sensitive to scene composition early in training and that larger per-GPU batches in stage 2 expose it to a worse initial loss landscape.

Operationally, the safe rule is:

- keep **stage-2 with-map** training at `6/GPU`
- allow **stage-1 with-map** training at `8/GPU`
- avoid claiming a universal cross-stage threshold without a dedicated stage-1 sweep

## Future Work

1. Profile first-iteration scene assignments for stage-2 BS=6 versus BS=12 and measure per-scene map loss from the loaded stage-1 checkpoint.
2. If the exact stage-1 limit matters, run a small sweep at `10/GPU`, `12/GPU`, and `14/GPU`.
3. Update any other reports or summary documents that still state "per-GPU batch size > 6 fails" as a universal rule.
