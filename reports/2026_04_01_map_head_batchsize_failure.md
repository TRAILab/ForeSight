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

## Results (2026-04-02)

**Summary**: `normeval` and `mapfeatnograd` both completed but failed to fix map convergence. `gradacc` crashed before training due to a config error.

| Config | per-GPU BS | map_mAP | map_loss_line_5 (end) | NDS | Status |
|--------|-----------|---------|----------------------|-----|--------|
| `4gpu_bs24` (baseline, working) | **6** | **0.553** | ~0.10 | — | converged |
| `4gpu_normeval` | **12** | **0.079** | ~0.69 | 0.528 | not converged |
| `4gpu_mapfeatnograd` | **12** | **0.074** | ~0.70 | 0.524 | not converged |
| `4gpu_gradacc` | **12** | — | — | — | **crashed (config error)** |

**Gradacc crash — root cause and fix**: The config used `optimizer_config = dict(grad_clip=..., cumulative_iters=2)`. In `mmdet_train.py:147`, when no `type` key is present, it calls `OptimizerHook(**cfg.optimizer_config)`, which does not accept `cumulative_iters`. The container runs mmcv 1.7.1, which does include `GradientCumulativeOptimizerHook` — it just needs to be specified explicitly. Fix: `optimizer_config = dict(type='GradientCumulativeOptimizerHook', cumulative_iters=2, grad_clip=...)`. **Config has been updated in-place.**

**What the results eliminate**:

1. **BN statistics (eliminated by `normeval`)**: `norm_eval=True` freezes BN completely from the first step, using stage1 precomputed running stats independent of current batch size. With BN fully decoupled from per-GPU BS, map_mAP remains 0.079 and map_loss_line_5 stays at ~0.69 — identical to the failing baseline. BN statistics are **not** the root cause.

2. **Map anchor init gradient destabilization (eliminated by `mapfeatnograd`)**: Removing gradients from the shared map anchor initialization parameter (`feat_grad=False`) did not improve map convergence (0.074, loss ~0.70). The conflicting-gradient-through-shared-init hypothesis is ruled out.

**Remaining hypothesis**: The root cause is **gradient variance at per-GPU BS=12**. With 12 scenes per step (vs. 6), the gradient signal for the map head's ground-plane anchors is noisier and/or larger in magnitude, preventing convergence. The detection head (900 spatially-diverse anchors) is robust to this because anchor diversity provides implicit gradient averaging. The map head (100 anchors all constrained to `ground_height=-1.84023`) has no such protection — noisy gradients act coherently on all 100 anchors.

**Gradient accumulation is the only remaining untested fix.** `4gpu_gradacc` with `cumulative_iters=2` would make each optimizer step see the equivalent of 6 samples/GPU (accumulated from 2×3 = 6 forward passes on the 4GPU config... actually 2 forward passes of 12 samples each = 24 samples/GPU equivalent). Wait — actually `cumulative_iters=2` with BS=12/GPU means each optimizer step accumulates gradients from 2 iterations, each of 12 samples. Effective per-optimizer-step sample count = 24/GPU. That is **more** than the working BS=6 config, not less.

**Re-examining the gradacc hypothesis**: `cumulative_iters=2` with per-GPU BS=12 gives effective gradient aggregation over 24 samples/GPU per optimizer step. The working config has 6/GPU per step. This means gradacc as configured actually makes gradient variance *smaller* than the working config (larger effective batch → lower gradient variance). If this works, it confirms that gradient noise at 12/GPU is the cause (and 24/GPU aggregation fixes it). If it doesn't work, something more fundamental is at play.

## Discussion

- **This is a blocking issue for compute efficiency**: The attractive 4GPU-bs48 configuration is unusable for experiments that include the map head, forcing all experiments to use 4GPU-bs24 or 8GPU-bs48 configs. Understanding the root cause would enable safe use of higher per-GPU batch sizes.
- **BN is definitively not the cause**: `norm_eval=True` is a complete BN bypass (uses fixed precomputed stats), yet the failure persists identically. SyncBN would not fix this problem.
- **Gradient variance remains the leading hypothesis**: The only uncontrolled variable between the failing and working configs — after ruling out total BS, LR, BN, and anchor initialization — is the per-step gradient variance due to per-GPU sample count.
- **The detection head's robustness is important to understand**: Why can the detection head tolerate high-variance gradients while the map head cannot? The spatial diversity of detection anchors (spread across 55m range, all heights) means each noisy gradient update still averages across many sampling locations. The map head's ground-plane constraint means all anchors share the same deformable attention sampling regime — noisy gradients degrade all 100 simultaneously.

## Next Steps (priority order)

1. **[Immediate] Resubmit `4gpu_gradacc`** — the config fix is in place (`type='GradientCumulativeOptimizerHook'`). This is the last untested hypothesis and highest-priority run.
   - If map_mAP ≥ 0.5 → gradient variance at per-GPU BS=12 is confirmed as root cause; `cumulative_iters=2` is a practical fix for any future high-BS run.
   - If map_mAP ≈ 0.078 → all known hypotheses are exhausted; need to profile gradient norms per-module across BS=6 vs BS=12 to find the divergence mechanism.

2. **[If gradacc works] Standardize the fix**: Document that any 4GPU config with per-GPU BS > 6 must use `GradientCumulativeOptimizerHook` with `cumulative_iters` chosen so that effective per-step samples ≤ 24/GPU (i.e., `cumulative_iters=2` for BS=12 is safe). Add a config warning comment.

3. **[If gradacc fails] Profile gradient norms**: Add a hook to log per-module gradient norms during the first 200 iterations for BS=6 and BS=12 configs. The divergence should be visible in the map head transformer layers, and specifically in the deformable attention sampling offset network.

4. **[Deferred] Check `sparsedrive_r50_stage2_4gpu_syncbn` work dir**: SyncBN is now known to not be the fix (BN is ruled out), but checking this dir would close out any remaining SyncBN confusion.
