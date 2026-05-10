# beyondsight_reb — predonly revival + temporal prior decoder

## Intro

Two ablations submitted together on branch `sd_bs`:

1. **`predonly_revival`** — modernizes the GT-perception oracle (predonly) setup
   that was previously stuck on the SparseDrive baseline motion head. Adds the
   architectural extensions that proved out elsewhere (cumulative refinement,
   multimode-deformable, depth 6) plus two new code paths specific to motion:
   per-waypoint deformable sampling and persistent mode-mode self-attention.
   Goal: push the upper bound on agent prediction when perception is perfect.

2. **`tpd` (temporal prior decoder)** — adds a small pre-decoder warmup pass
   on the detection head that cross-attends cached temporal queries to the
   previous frame's *motion* outputs. Targets the question: does feeding
   motion-prediction context back into detection's temporal mechanism improve
   downstream prediction (without changing perception heads)?

Both share the new evaluation metrics added in this batch:
`all_*` aggregate over every detection class, and `top1_ade_err` for car /
pedestrian / all (top-1-mode ADE alongside the existing top1-FDE / minADE /
minFDE / brier-minFDE / miss-rate).

Companion baseline jobs (predonly_deform, sparsedrive_r50_stage2_4gpu) already
exist; both new variants are A/B against those.

## Method

### A. predonly_revival (`sparsedrive_r50_stage2_4gpu_bs24_predonly_revival.py`)

Extends `predonly_deform` with five changes; full and simple variants share
all but the two motion-architecture extensions.

**Per-stage motion ops** (×6, replacing the old 3-stage):

```
[temp_gnn, gnn, norm, motion_self_attn, norm, deformable, norm, ffn, norm, refine] * 6
```

The `gnn` agent-agent SA is restored (had been dropped from earlier predonly
sketches), `motion_self_attn` is new.

**New code paths in `motion_planning_head.py`:**

- `motion_deformable_waypoints=[1, 3, 5, 7, 11]` — when set, the multimode
  deformable cross-attends per (mode, K) waypoint instead of per mode at the
  endpoint only. Helper `_build_motion_waypoint_anchors_all_modes` produces
  `(B, N, M, K, 11)` boxes by repeating the anchor box at each predicted
  trajectory waypoint with heading from local segment direction. The
  attended `(B, N, M, K, D)` is mean-pooled over K back to per-mode features
  before mode-confidence aggregation.
- `motion_self_attn` op — persistent SA across the `fut_mode` modes per agent.
  Operates on `motion_mode_query` `(B, N, M, D)`, reshapes to `(B·N, M, D)`,
  runs SA with positional embedding from the live per-mode endpoint
  (`motion_endpoint_anchor_all[..., :2]` through `motion_anchor_encoder ∘
  gen_sineembed_for_position`), reshapes back. Trailing `norm` LN-normalizes
  the mode-query (mirrors the existing `plan_self_attn` pattern).
- `motion_cumulative_refinement=True` — each refine adds the previous stage's
  detached prediction to the new prediction.

**`GTSparseDriveHead` change:**

- `instance_feature_init="class_embed"` — replaces the legacy zero-init
  `instance_feature` with a learnable `nn.Embedding(num_classes, embed_dims)`
  looked up by GT class for valid slots (padding stays zero). Trainable
  module owned by the head; the rest of `det_head` remains frozen.

**Sizing:**

- `instance_bank.num_anchor=200` (was 900) — nuScenes train max GT/frame is
  160 (p99=110, mean=35). 200 has comfortable headroom and gives ≈4.5×
  compute reduction in the deformable (queries drop from 27000→6000 with
  per-waypoint K=5).
- `num_temp_instances=134` (≈ 200 × ⅔, matching the original 600/900 ratio).
- The kmeans file `kmeans_det_900.npy` is reused — `InstanceBank` truncates
  via `min(len(anchor), num_anchor)`. GT path overrides anchors anyway, so
  the truncated kmeans values are never read in this config.

**Lower-risk fallback (`..._predonly_revival_simple.py`):** drops
`motion_deformable_waypoints` and `motion_self_attn` (and the corresponding
op + config block). Keeps everything else (cumulative, multimode-deform
endpoint, depth 6, num_anchor=200, gnn, class_embed). Tests whether the
simpler stack already captures most of the gain.

### B. tpd (`sparsedrive_r50_stage2_4gpu_tpd.py`)

Detection-side ablation on the vanilla full-stack config (all heads enabled,
no GT perception). Adds a five-op warmup pass that runs before the main
decoder:

```
det_head.temporal_warmup_order = [temp_gnn, norm, ffn, norm, refine]
```

The warmup operates on `temp_instance_feature` (the previous-frame top-`num_temp_instances`
cached agent features, returned by `InstanceBank.get`). Its `temp_gnn`
cross-attends:

- **Q** = `temp_instance_feature` (`(B, 600, 256)`), positionally embedded
  via `anchor_encoder(temp_anchor)`.
- **K/V** = `cached_motion_feature` — **new state** in `InstanceBank`,
  populated each frame from the motion head's mode-weighted aggregation:
  `softmax(motion_classification[-1]) · motion_mode_query` summed over modes,
  then top-k-selected by the same indices `cache()` used so motion features
  align per-agent with cached det features.
- **K_pos** = `anchor_encoder` of an "endpoint-positioned" anchor —
  `cached_anchor.clone()` with XY overridden by `cached_motion_endpoint`
  (top-1 mode's predicted final-step XY in lidar frame). The K side is
  positionally indexed at *where the agent is predicted to be* on the
  current frame, not where it was last frame.

This gives the warmup access to forward-looking context the main decoder
itself cannot see (the main decoder's `temp_gnn` only attends to single-frame
cached det state). On the first frame `cached_motion_feature` is `None` and
the op is skipped — `refine` at the end of the warmup keeps grad flowing
through the warmup params anyway.

**Plumbing changes:**

- `InstanceBank`:
  - New state: `cached_motion_feature`, `cached_motion_endpoint`,
    `_last_topk_indices`.
  - `cache()` saves the top-k indices it picked.
  - New `cache_motion(motion_feature, motion_endpoint)` selects with
    `_last_topk_indices` and stores. No-op if motion is None.
  - `reset()` clears the new fields.
- `MotionPlanningHead.forward`:
  - After the decoder loop, computes mode-weighted `motion_feature_agg`
    `(B, N, D)` and top-1 endpoint `motion_endpoint` `(B, N, 2)`. Endpoint
    converts agent-frame trajectories to lidar via `_agent2lidar` when
    `motion_target_in_agent_frame`, then adds `det_anchors[..., :2]` to
    land in absolute lidar XY.
  - Both attached to `motion_output` dict.
- `SparseDriveHead.forward`: after the motion head returns, calls
  `det_head.instance_bank.cache_motion(...)`.
- `Sparse4DHead`:
  - `_temp_gnn_with_layer(layer, q_feat, q_anchor_embed, kv_feat,
    kv_anchor_embed)` mirrors `_gnn_with_layer` but for cross-attention and
    reuses the warmup-only `warmup_fc_before/after` projections.
  - New `warmup_temp_graph_model` config slot — separate attention weights
    from the main decoder's `temp_graph_model` (cleaner A/B; no parameter
    sharing surprises).
  - Warmup forward loop gains the `temp_gnn` op handler.

### Eval metric additions

- `motion_utils.prediction_metrics` returns 6-tuple now —
  `(minade, minfde, mr, top1_ade, top1_fde, brier)`. `top1_ade =
  dist[top1_idx, :].mean()` (vs the existing `top1_fde = dist[top1_idx, -1]`).
- `MotionMetricData` adds `top1_ade_err` field; `serialize` /
  `deserialize` / `no_predictions` / `random_md` updated. Backwards-compatible
  deserialize: missing key falls back to `min_ade_err`.
- `accumulate` now accepts `class_name='all'` — uses local helper
  `_matches_class` that skips per-class filtering when `'all'`.
- `MotionEval.evaluate`: `class_names = ['car', 'pedestrian', 'all']`,
  `MOTION_TP_METRICS += ['top1_ade_err']`. `OccludedMotionEval` and
  `AllMotionEval` inherit `evaluate` so they pick up the same change.

After eval, each `MotionEval` mode (`/det/all` and `/occluded` and
`/all-objects`) emits per-class:

```
{cls}_min_ade_err, {cls}_min_fde_err, {cls}_miss_rate_err,
{cls}_top1_ade_err, {cls}_top1_fde_err, {cls}_brier_min_fde_err, {cls}_EPA
```

with `cls ∈ {car, pedestrian, all}`.

## Submitted runs

| Run | Server | Job ID | Config | Notes |
|---|---|---|---|---|
| predonly_revival | DGX | 3793 | `..._bs24_predonly_revival.py` | full impl: cumulative + per-waypoint multimode-deform + mode-mode SA + class_embed init + num_anchor=200 |
| predonly_revival_simple | DGX | 3794 | `..._bs24_predonly_revival_simple.py` | drops per-waypoint and motion_self_attn; keeps cumulative + multimode endpoint + class_embed + 200 |
| tpd | Killarney | 3496058 | `..._stage2_4gpu_tpd.py` | full-stack vanilla + temporal_warmup_order=[temp_gnn, norm, ffn, norm, refine] with motion-cache K/V |

All three: trainval, 10 epochs, 4 GPUs, deterministic.

## Pending verification

- DGX job 3793 first 153 iters showed total loss falling from 41.6 → 14.3,
  per-stage motion losses dropping evenly across all 6 refines, grad norm
  15–21 (well under 25 clip), memory 14.2 GB/GPU, ETA ~4.7 h. Healthy.
- DGX job 3794 (simple) queued behind 3793.
- Killarney job 3496058 (tpd) running on `kn027`.

What's *not* yet validated end-to-end on real data:
- Per-waypoint deformable CUDA kernel call (DAF) with K=5-expanded query
  batch — synthetic shape plumbing OK, kernel unverified.
- TPD `temp_gnn` flash-attention forward — wired in fp16 build, not
  exercised on synthetic data (dtype quirk under bare `.half()` outside
  `Fp16OptimizerHook`). Real training will surface any issue.
- `cache_motion()` alignment confirmed against `motion_feature.gather(1, idx)`
  in unit smoke; correct in the static case. Cross-frame projection of
  `cached_motion_endpoint` is *not* applied by `anchor_handler.anchor_projection`
  — the endpoint is treated as "predicted current-frame XY" without further
  projection. Acceptable since prediction error dominates the small
  per-frame projection delta, but worth revisiting if results disappoint.

## Open questions

- **predonly_revival full vs simple**: the simple variant is a clean control
  for the per-waypoint + mode-mode SA additions. If `simple` reaches similar
  motion ADE/FDE, the two new code paths aren't pulling weight in the GT
  setting and shouldn't be ported into headline configs.
- **TPD value-add over plain `temporal_warmup_order=[gnn, norm, ffn, refine]`**:
  TPD's distinguishing claim is "K/V comes from forward-looking motion, not
  from the temporal queries themselves." If a self-attn warmup baseline lands
  similar numbers, the motion-cache CA isn't the source of the gain. Worth
  queueing a side-by-side `..._tpd_self_only.py` once we see TPD numbers.
- **Top-1 ADE drift over training**: `top1_ade` will trail `min_ade` heavily
  early in training (mode-classification head untrained → confidence noise
  → top-1 random). Watch the gap close — it's a classifier-quality proxy.
- **`all_*` aggregate utility**: useful for comparing perception-quality vs
  prediction-quality bottlenecks across the long-tail classes (truck, bus,
  cyclist, etc.) that aren't in the current `car`/`pedestrian` reporting.
  If `all_min_ade` stays much higher than `car_min_ade` even with strong
  models, the long-tail is the bottleneck.

## Next

- Wait for first-epoch evals on all three jobs (~4–5 h on DGX, similar on
  Killarney) to get an initial read on motion metrics.
- Decide based on `revival` vs `revival_simple` whether per-waypoint
  + motion_self_attn graduate to other configs.
- Decide based on `tpd` vs vanilla baseline whether the motion-cache CA
  is worth the added complexity in `InstanceBank` + `MotionPlanningHead`
  + dispatcher.
