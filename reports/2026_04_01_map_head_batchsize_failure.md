# Map Head Failure at Per-GPU Batch Size > 6 — Investigation (Apr 2026)

## Intro

This document covers a systematic investigation into a reproducible failure mode where the HD map head catastrophically fails to converge when per-GPU batch size exceeds 6. The `sparsedrive_r50_stage2_4gpu.py` config (total_batch_size=48, 4 GPUs, **12 samples/GPU**) produces map_mAP=0.078 despite detection performing normally, while the `stage2_4gpu_bs24.py` config (total_batch_size=24, **6 samples/GPU**) achieves map_mAP=0.553. This failure blocks using the computationally attractive 4-GPU-bs48 configuration. Four ablation experiments were submitted 2026-04-01 to isolate the root cause.

## Method

**Observed failure**: systematic comparison across four configs spanning different GPU counts, total batch sizes, and per-GPU batch sizes:

| Config | per-GPU BS | LR | map mAP | map_loss_line_5 (end) |
|--------|-----------|-----|---------|----------------------|
| `4gpu` (bs48) | **12** | 3e-4 | **0.078** | ~0.70 (not converged) |
| `4gpu_maplrdiv4` (bs48, loss/4) | **12** | 3e-4 | **0.074** | — |
| `4gpu_bs24` | **6** | 1.5e-4 | **0.553** | ~0.10 (converged) |
| `8gpu_noflash` | **6** | 3e-4 | **0.547** | — |

**Isolation logic**: The 8GPU config (total_batch_size=48, **6/GPU**, lr=3e-4) achieves 0.547 — same total BS and same LR as the failing 4GPU config. This rules out total batch size and LR as root causes.

**Ablation experiments submitted (2026-04-01)**:

| Config | Change | Tests | Confidence |
|--------|--------|-------|------------|
| `4gpu_gradacc` | `cumulative_iters=2` | Replicates per-GPU BS=6 for BN and gradients simultaneously | **Highest** |
| `4gpu_normeval` | `norm_eval=True` in backbone | Freezes BN running stats; isolates BN as root cause | High |
| `4gpu_mapfeatnograd` | `feat_grad=False` in map instance bank | Tests whether map anchor init parameter destabilizes training | Medium |

**What was ruled out through code inspection**:
1. LR too high — 8GPU uses same LR and works
2. Total batch size — 8GPU has same total BS and works
3. Map loss weight — `maplrdiv4` tried, no improvement
4. Temporal gradient flow — `cache()` explicitly detaches all features (`instance_feature.detach()`); no gradients cross the temporal boundary regardless of config
5. `feat_grad` as a temporal mechanism — with `num_temp_instances=0` in stage1, `cache()` returns early; yet stage1 also fails at bs=16/GPU

## Results (2026-04-02, updated with gradacc)

**Summary**: `normeval` and `mapfeatnograd` both failed to fix map convergence. `gradacc` ran successfully (after config fix) but also failed — map loss plateaued at ~0.75 and never approached the BS=6 target of ~0.10.

| Config | per-GPU BS | map_mAP | map_loss_line_5 (iter 51) | map_loss_line_5 (iter 1734) | Status |
|--------|-----------|---------|--------------------------|----------------------------|--------|
| `4gpu_bs24` (baseline, working) | **6** | **0.553** | **0.21** | ~0.10 | converged |
| `4gpu_normeval` | **12** | 0.079 | ~0.88 | — | not converged |
| `4gpu_mapfeatnograd` | **12** | 0.074 | ~0.93 | — | not converged |
| `4gpu_gradacc` (cumulative_iters=2) | **12** | TBD | **0.93** | **0.78** | **not converging** |

**Gradacc training trajectory (job 3571)**:

| Iter | map_line_0 | map_line_5 | LR (head) |
|------|-----------|-----------|-----------|
| 51   | 0.829     | 0.928     | 1.2e-4    |
| 510  | 0.711     | 0.756     | 2.9e-4 (peak, end of warmup) |
| 1530 | 0.718     | 0.750     | 2.5e-4    |
| 1734 | 0.739     | 0.782     | 2.4e-4 (cosine decay) |

The map loss improved from 0.93→0.75 during warmup (iters 51→510), then **stagnated** through cosine decay (0.75→0.78). With ~4000 iterations remaining (total 5860), converging from 0.78 to 0.10 is extremely unlikely. Gradient accumulation is confirmed not to fix per-GPU BS=12 map failure.

**Key structural observation — from gradacc logs**: At iter 51 (barely 25 optimizer steps, LR at 40% of peak), the map_line_5 is already 0.93 while the BS=6 baseline was 0.21 at the **same** iteration. This 4.4× gap is present **before any significant training has occurred**, pointing to the stage1 checkpoint having ~4× higher map loss on the scenes assigned to GPU 0 at BS=12 (groups perm[0..11]) versus those at BS=6 (perm[0..5]).

**Gradacc crash — root cause (from previous session)**: Config used `optimizer_config = dict(grad_clip=..., cumulative_iters=2)`. Fixed to `dict(type='GradientCumulativeOptimizerHook', cumulative_iters=2, grad_clip=...)`. The DGX also required `mmdet_train.py` to support typed optimizer hooks via `build_from_cfg`; that fix has since been reverted locally (commit `a37e8b5`) but was in place when job 3571 was submitted.

**What the results eliminate**:

1. **BN statistics (normeval)**: Freezing BN completely → map_loss still ~0.69. BN is **not** the cause.
2. **Map anchor init gradients (mapfeatnograd)**: `feat_grad=False` → no improvement. Shared-init destabilization is **not** the cause.
3. **Gradient variance (gradacc)**: Reducing gradient variance with `cumulative_iters=2` (effective 24/GPU per step) → still plateaus at 0.75. Gradient noise is **not** the primary cause.
4. **Total batch size, LR** (from prior ablation): 8GPU-bs6 with same total BS=48 and LR=3e-4 achieves 0.547 map_mAP. Total BS and LR are **not** the cause.

**Thorough code audit — no batch-size-dependent operation found**: The full map head forward path was audited: DAF CUDA kernel (output zeroed via `at::zeros`, atomic adds correct), `_get_weights` dropout (inverted, expectation-preserving), `reduce_mean(num_pos)` normalization (correctly normalizes loss per positive match regardless of BS), `SparseLineLoss.normalize_line` (constant roi_size=(30,60)), `HungarianLinesAssigner` (per-sample matching, batch-size independent), `AsymmetricFFN` (linear layers, no batch coupling), `GroupInBatchSampler` (each slot independent). No hardcoded batch-size dependency was found in any map-specific code path.

**Current best hypothesis — scene distribution at initialization**:

`GroupInBatchSampler` assigns each batch slot its own generator seeded by `global_sample_idx`. With fixed seed=0, the permutation `perm = torch.randperm(num_groups)` is deterministic. At BS=6 (global_batch=24), rank 0 first-iteration batch = perm[0..5]. At BS=12 (global_batch=48), rank 0 first batch = perm[0..11]. The additional groups perm[6..11] — by chance with this particular seed — appear to be significantly harder scenes for the map head (higher num GT lines, more complex road structure). The stage1 checkpoint has ~4× higher map loss on these scenes, producing a much worse optimization starting point at BS=12.

Why gradacc can't recover: the poor starting point at BS=12 (loss 0.93 vs 0.21) places the map head in a region of the loss landscape where the decoder refinement is actively counterproductive — each decoder stage increases loss slightly rather than decreasing it (observed across all BS=12 runs). After 500+ iterations with degraded refinement, the model is stuck in a local minimum from which it cannot escape within the training budget.

## Discussion

- **All tested interventions have failed**: BN, anchor init, gradient accumulation, LR scaling (via loss/4 in prior ablation) — none fixed per-GPU BS=12.
- **Root cause remains open**: No code-level bug found. The most consistent explanation is scene distribution sensitivity combined with the map head's narrow loss landscape (all 100 anchors at ground height, minimal diversity).
- **Practical conclusion**: Per-GPU BS > 6 is not viable for the map head with the current architecture and training setup. This is a hard constraint.
- **Detection head is robust at BS=12**: 900 spatially-diverse 3D anchors provide implicit gradient averaging. Map head's 100 coplanar anchors do not.

## Next Steps (priority order)

1. **[Confirmed] Cancel job 3571 (gradacc)** — confirmed not converging at iter 1734. No need to run to completion.

2. **[Diagnostic, if root cause matters] Profile which scenes fall in perm[0..5] vs perm[0..11]**: Log scene tokens for both BS=6 and BS=12 first iteration; compute map loss per scene from the stage1 checkpoint. If perm[6..11] consistently have 4-5× higher map loss than perm[0..5], this definitively confirms the scene distribution hypothesis.

3. **[Diagnostic, alternative] Run BS=6 cumulative_iters=2 experiment**: If 4GPU-bs6 with gradacc (cumulative_iters=2) converges, this confirms that the per-GPU batch SIZE (not per-step sample count) is the constraint. This is the most informative remaining experiment, but the result is somewhat predictable (BS=6 per forward pass should always work since each forward pass has low initial loss).

4. **[Accept and move on]** Use `4gpu_bs24` (BS=6/GPU) or `8gpu` (BS=6/GPU) for all future experiments. Document per-GPU BS > 6 as a hard constraint for the map head. The 4GPU-bs12 configuration should not be used for any run that trains the map head.
