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

## Results

**Root cause hypothesis: backbone BN statistics at per-GPU BS=12**

With `norm_eval=False` and standard (non-sync) BatchNorm, each GPU independently estimates running statistics from its local batch. At BS=12/GPU vs. BS=6/GPU, the BN statistics diverge differently.

The detection head is robust to this shift: it uses ~900 spatially diverse 3D anchors that sample features from many locations — any single BN-induced feature corruption affects only a subset. The map head is uniquely vulnerable:
- All 100 map anchors are constrained to a **fixed ground plane** (`ground_height=-1.84023`) via `SparsePoint3DKeyPointsGenerator`
- Systematic BN-induced feature degradation at ground-plane projected coordinates collapses all 100 anchors simultaneously
- `feat_grad=True` in the map instance bank means the shared anchor initialization parameter receives gradient updates — this may amplify BN-induced feature drift when conflicting gradients arrive from 12 diverse scenes per step

**Pending results**: All four ablation configs (`4gpu_gradacc`, `4gpu_normeval`, `4gpu_mapfeatnograd`) are running as of 2026-04-01.

**Expected outcomes**:
- If `gradacc` works → per-GPU BS confirmed as root cause; `cumulative_iters=2` is a practical fix
- If `normeval` works but `gradacc` doesn't → BN statistics are the cause; production fix is SyncBN
- If `mapfeatnograd` works but `normeval` doesn't → shared anchor init parameter being destabilized by diverse gradients at larger BS

## Discussion

- **This is a blocking issue for compute efficiency**: The attractive 4GPU-bs48 configuration is unusable for experiments that include the map head, forcing all experiments to use 4GPU-bs24 or 8GPU-bs48 configs. Understanding the root cause would enable safe use of higher per-GPU batch sizes.
- **Practical short-term fix regardless of root cause**: Use `cumulative_iters=2` (gradient accumulation) on the failing bs48 config. This replicates per-GPU BS=6 for both BN statistics and gradient updates, and costs nothing in compute — it runs 2 forward passes per optimizer step.
- **SyncBN would be the clean architectural fix if BN is confirmed**: Replacing `norm_cfg=dict(type='BN')` with `norm_cfg=dict(type='SyncBN')` in the backbone pools statistics across all 4 GPUs (effective batch = 48), making per-GPU BS irrelevant to BN. The existing `syncbn` work dir (`sparsedrive_r50_stage2_4gpu_syncbn`, 2026-02-16) may have results to consult.
- **The detection head's robustness is important to understand**: Why can the detection head tolerate BN-shifted features while the map head cannot? The spatial diversity of detection anchors (spread across 55m range, all heights) vs. the map head's ground-plane constraint is the most credible explanation.

## Future Work

- Read the results from `4gpu_gradacc`, `4gpu_normeval`, and `4gpu_mapfeatnograd` once available to confirm the root cause.
- Check the `sparsedrive_r50_stage2_4gpu_syncbn` work dir for any map performance data — if SyncBN is already tested, this may close the investigation immediately.
- Once confirmed, standardize the fix (either `cumulative_iters=2` or SyncBN) and apply to all future 4GPU high-BS configs.
- Document the per-GPU BS sensitivity as a configuration warning in the project README to prevent future accidental use of failing configs.
