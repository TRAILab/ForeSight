# Inference Latency Breakdown — Baseline SparseDrive vs Ours

## Setup

- Device: RTX 3090, fp32, batch=1, single sample per iteration
- Container: `foresight` (torch 1.13)
- Dataset: nuScenes val
- Iterations: 95 timed after 5 warmup unless noted
- Tools used:
  - `tools/benchmark.py` — FPS + per-module FLOPs (mmcv `add_flops_counting_methods`)
  - `tools/benchmark_stages.py` — direct cuda-event timing of backbone / neck / head / total
  - `tools/benchmark_egoonly.py` — ego-only inference variant (det/map heads stubbed)
  - `tools/benchmark_components.py` — sub-component cuda-event hooks + phase markers
  - `tools/plot_latency_breakdown.py` — paper figure
- Profiler-hooks add ~10–15% overhead to absolute ms vs no-hook timing; **percentages within each run are accurate**.
- Final paper figure: `reports/latency_breakdown.{pdf,png}` (different normalization — see note at the end).

## Variants

1. **Baseline** — `sparsedrive_r50_stage2_4gpu_bs24.py` + `ckpt/sparsedrive_stage2.pth`. Full SparseDrive: backbone → FPN → det_head + map_head + coupled motion-planning head with collision rescorer.
2. **3-layer minS2 ego-only** — `_tmp_minS2_decoder3_latency.py` (copy of minS2 with planner `*6 → *3`) + `ckpt/sparsedrive_stage2_minS2.pth`. Det/map heads stubbed at inference, only ego token enters the planner, conflict head off, rescorer vestigial.
3. **Baseline ego-only** — Baseline architecture + the same ego-only patches. Planner reduces to temp_gnn + ffn (gnn / cross_gnn nulled), single refine at end.

## Headline comparison

| Variant | Total ms | FPS | % of baseline | Speedup |
|---|---:|---:|---:|---:|
| Baseline (full) | 166.7 | 6.0 | 100% | 1× |
| 3-layer minS2 ego-only | 77.2 | 12.8 | 46.3% | 2.2× |
| Baseline ego-only | 46.8 | 21.4 | 28.1% | 3.6× |

Backbone+neck floor on this device ≈ 20 ms (≈ 50 fps) — once perception heads + agent tokens are removed, you bump into the encoder cost.

## Baseline (166.7 ms)

| Component | ms | % of total |
|---|---:|---:|
| Backbone + neck | 15.0 | **9.0%** |
| Detection decoder (`det_head.forward`) | 41.9 | **25.1%** |
| Mapping decoder (`map_head.forward`) | 29.8 | **17.9%** |
| Coupled motion+plan head (`motion_plan_head.forward`) | 32.3 | **19.4%** |
| Collision rescorer (`planning_decoder.decode`) | 42.3 | **25.3%** |
| Other (motion_decoder + glue) | ~5 | ~3% |

### Motion vs plan within the coupled head

Baseline's coupled decoder runs all 61 tokens (50 det + 10 map + 1 ego) through identical attention layers; the motion vs planning split happens only at the final refine head. So:
- **Motion-only path** (agents only, ego stripped) ≈ **32 ms / 19%** — full motion_plan_head; the marginal cost of 1 ego token in 61-token attention is negligible.
- **Plan-only path** (ego only) ≈ **5–10 ms / 3–6%** — N=1 self-attention is launch-overhead-bound, not compute-bound.

These are estimates; direct ablation hits flash-attn empty-K/V crashes in baseline (no `skip_perception_kv`) and shape mismatches (hard-coded `num_anchor + 1`). Confirming would need ~50 lines of forward-patching to make the coupled head ego-only-safe — the **baseline ego-only** variant below is the cleaner data point.

## 3-layer minS2 ego-only (77.2 ms)

### Major buckets

| Component | ms | % of total |
|---|---:|---:|
| Backbone + neck | 21.0 | **27.2%** |
| Planner (`motion_plan_head.forward`) | 53.4 | **69.3%** |
| Det / map heads (stubbed) | ~0.2 | ~0.2% |
| Post-process / glue | ~3 | ~4% |

### Planner internals — per-op (cumulative across stages)

| Component | ms | % of total |
|---|---:|---:|
| `det.anchor_encoder` (called 7×/iter — ego, temp, per-stage plan_anchor_box) | **14.3** | **18.3%** |
| `planner.deformable[3 stages]` | 6.3 | 8.2% |
| `planner.refine[3 stages]` | 4.1 | 5.3% (incl. plan/motion cls/reg MLPs ~2.7) |
| `planner.temp_gnn[3 stages]` | 2.9 | 3.7% |
| `planner.ffn[3 stages]` | 1.0 | 1.4% |
| `planner.gnn`, `planner.cross_gnn` (nulled by `skip_perception_kv`) | 0 | 0% |
| `plan_anchor_encoder` | 1.0 | 1.3% |
| `iq.ego_feature_encoder` (1×1 conv on cam-0 feat map) | 1.0 | 1.3% |
| `motion_anchor_encoder` | 0.7 | 0.8% |
| `fc_before` + `fc_after` (decouple_attn glue) | 0.6 | 0.7% |
| Conflict classifier MLP (proxy: `plan_cls_branch`) | ~0.2 | ~0.3% |
| Tensor-manipulation residual (sineembed, anchor box construction, expand/cat/reshape glue) | ~22 | ~28% |

### Planner internals — phase intervals

| Phase | ms | % of total |
|---|---:|---:|
| `mph_setup` (before stage 0) — anchor init + encode + iq.get | 14.9 | **18.5%** |
| `stage 0` (full op_loop + post-refine plan_anchor rebuild) | 16.0 | **19.9%** |
| `stage 1` | 14.4 | 17.9% |
| `stage 2` (excl. its final refine) | 8.0 | 9.9% |
| `last refine + post-loop` | ~3 | ~3.7% |

The setup phase alone is ~18% — most of it is the `det.anchor_encoder` calls (chunky 4-loop SparseBox3DEncoder with `decouple_attn=True`) plus tensor manipulation that doesn't go through any single hookable Module.

### Rough rule of thumb for the planner
**~1/3 anchor encoding · ~1/3 transformer ops · ~1/3 framework + tensor glue.**

## Baseline ego-only (46.8 ms)

Same architecture as baseline, with det/map heads stubbed, gnn/cross_gnn layers nulled, num_det/num_map=0, ego_only_planning=True, rescorer off.

| Component | ms | % of total |
|---|---:|---:|
| Backbone + neck | 19.9 | **42.5%** |
| Planner (`motion_plan_head.forward`) | 23.4 | **50.0%** |
| Det / map heads (stubbed) | ~0.1 | ~0.3% |
| Rescorer (off) | ~0.9 | ~1.9% |

### Planner internals (23.4 ms)

| Phase | ms | % of total |
|---|---:|---:|
| `mph_setup` | 13.7 | 29.3% |
| `stage 0` (full) | 2.2 | 4.6% |
| `stage 1` (full) | 2.1 | 4.4% |
| `stage 2` (excl. final refine) | 2.0 | 4.3% |
| `last refine + post` | ~3 | ~6% |

Each baseline ego-only stage is only ~2 ms because there's no per-stage deformable, no per-stage refine, no per-stage `plan_anchor_box` rebuild — just `temp_gnn → ffn`. Contrast 3-layer minS2 ego-only's ~15 ms per stage.

## What's surprising

1. **Backbone is only ~9% of baseline latency**, not the bottleneck. FLOPs say backbone+neck = 78% of compute, but on a 3090 the head dominates wall-clock because of `DeformableFeatureAggregation`'s many small launch-overhead-bound CUDA kernels.
2. **The collision rescorer is 25% of baseline** — CPU-side per-mode collision-mask computation against the 50 predicted agent trajectories. Removing perception agents removes the need for this entirely.
3. **`det.anchor_encoder` is 18% of the 3-layer ego-only run** — called 7 times per iteration (ego, temp, and once per stage for `plan_anchor_box`). Each call passes a small tensor through a 4-loop linear network (`SparseBox3DEncoder`, `out_loops=4` due to `decouple_attn=True`). Eager small calls beat the encoder up.
4. **The "rest" of the planner is launch-overhead-bound at N=1.** ~22 ms (~28%) of the 3-layer ego-only is sineembed, plan-anchor-box construction, and expand/cat/reshape glue spread across ~30 small CUDA dispatches per stage.

## Easy latency-reduction options

Ordered by impact / ease ratio. Several are drop-in for an inference-time deployment of the existing trained checkpoint.

1. **Drop perception heads at inference** — biggest lever. Already implemented as a runtime patch in `tools/benchmark_egoonly.py` / `tools/benchmark_components.py --ego-only`. **−71% latency on baseline architecture (166.7 → 46.8 ms, 3.6× FPS)**. No retraining needed for the planner if it's already trained without depending on perception tokens (e.g. minS2-style with `skip_perception_kv`); for vanilla baseline the plan output may degrade since the trained planner was conditioned on real perception activations.

2. **Drop the collision rescorer** — 25% of baseline by itself. Trivial code change (`use_rescore=False`). Only safe when there are no agent trajectories to collide with, or when an alternative rescore mechanism (e.g., learned conflict head) is in place. Saves ~42 ms on baseline.

3. **Halve the planner decoder layers** — 6 → 3 layers in the minS2 ego-only path went from 113.5 → 70.2 ms (-38%, 2.2× → 2.7× over baseline). For the trained 6-layer ckpt, dropping the last 3 layers would still load layers 0–2 with trained weights; quality impact unknown without retraining.

4. **Batch `det.anchor_encoder(plan_anchor_box)` across stages** — currently called 3× per iter for the same encoder with similar inputs. Batching collapses to 1× call, saving ~5 ms (~6% of 3-layer ego-only). Code change inside `motion_planning_head.forward`.

5. **Cache `det.anchor_encoder(ego_anchor)` and `(temp_anchor)`** — the ego anchor is constant across frames; temporal anchors only need to be re-projected when the ego pose changes (which they already are inside `instance_queue.get`). Encode once, reuse. Saves another ~3 ms (~4%).

6. **Switch planner `temp_graph_model` to `MultiheadFlashAttention`** — currently plain `MultiheadAttention`. Even at small seqlen (1×4), small but free wins. Same module pattern as gnn / cross_gnn already use.

7. **Fuse `gen_sineembed_for_position` + `_build_planning_anchor_boxes_multi`** — the anchor-box construction across (modes × waypoints) is a long chain of small tensor ops. A single fused implementation (or `torch.compile`) would collapse the launch overhead. Estimate: 3–5 ms savings.

8. **CUDA graph capture for the whole forward** — ego-only inference at batch=1 is launch-bound; CUDA graphs eliminate per-kernel dispatch overhead. Estimated 30–50% savings on the 22 ms tensor-manipulation residual. Requires shape stability (no dynamic anchor counts).

9. **Disable the per-stage planning_deformable_instfeat path** — minS2's per-stage deformable readout for plan_mode_query (modes × waypoints = 36 anchors per call) costs ~6.3 ms across 3 stages. If the laststage-only variant is sufficient, drop the earlier stages. Saves 4–5 ms in 3-layer minS2 ego-only.

10. **Use fp16 for inference** — easy global toggle; ~1.5–2× compute savings on Ampere tensor cores. Backbone is already fp16-capable; the planner's small dispatches benefit less but it's free.

### Combined upper-bound estimate

Stacking 1 (perception removal) + 2 (rescorer off) + 8 (CUDA graph) on the baseline architecture, an aggressive deployment could plausibly land at **~30 ms / ~33 fps**, a 5.5× over baseline. The encoder floor (~12 ms with fp16 + Ampere) is the hard limit on this device.

## Files / artifacts

- Final paper figure: `reports/latency_breakdown.{pdf,png}` (note: the figure uses a normalization that yields a 2.3× speedup based on a slightly different ms accounting from the user; the profiler measurements above use the wall-clock totals from `tools/benchmark_components.py`).
- Profiler scripts: `tools/benchmark.py`, `tools/benchmark_stages.py`, `tools/benchmark_egoonly.py`, `tools/benchmark_components.py`, `tools/plot_latency_breakdown.py`.
- Per-run logs: `/tmp/latency_bench/*.log`.
- 3-layer config: `projects/configs/_tmp_minS2_decoder3_latency.py`.
- minS2 ckpt (copied from killarney): `ckpt/sparsedrive_stage2_minS2.pth`.
