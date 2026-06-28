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

## Outcomes (2026-05-10)

### First batch — all three crashed, root causes identified
- **DGX 3793 (predonly_revival)**: trained the full schedule (Iter 11720/11720,
  4 h 41 m wall) and saved `work_dirs/.../iter_11720.pth`. Crashed during the
  post-training validation pass with `KeyError: 'gt_map_labels'` on the first
  val batch. Root cause: `GTSparseDriveHead.forward` called
  `_build_gt_map_output` whenever `map_head` existed, but `test_pipeline`
  drops `gt_map_*` keys (use_map=False).
- **DGX 3794 (predonly_revival_simple)**: same outcome — trained fully
  (Iter 11720/11720, 2 h 54 m), saved ckpt, same KeyError on val.
- **Killarney 3496058 (tpd)**: crashed at iter 0 (10 m wall) with
  `RuntimeError: Given normalized_shape=[512], expected input with shape
  [*, 512], but got input of size [12, 600, 256]`. Two root causes uncovered:
    1. `warmup_ffn` was wired with `in_channels=embed_dims*2` while every
       other working `temporal_warmup_order` config uses `in_channels=embed_dims`
       — `AsymmetricFFN.pre_norm` builds `LN(in_channels)` so this expected
       512 against a 256-dim feature.
    2. Even after the LN fix, the `temp_gnn` warmup op `continue`d when
       `cached_motion_feature is None` (first frame / before any motion cache),
       leaving `warmup_temp_graph_model` params disconnected from the loss
       → DDP raised "Expected to have finished reduction…" on iter 1.

### Fixes applied (committed on `sd_bs`)
- `projects/configs/sparsedrive_r50_stage2_4gpu_tpd.py`: `warmup_ffn`
  `in_channels=embed_dims`.
- `projects/mmdet3d_plugin/models/gt_sparse_drive_head.py`: gate
  `_build_gt_map_output` on `"gt_map_labels" in metas` so eval falls through
  to `map_output=None` (already handled by `MotionPlanningHead`'s `cross_gnn`).
- `projects/mmdet3d_plugin/models/detection3d/detection3d_head.py`: when the
  motion cache is unavailable, the `temp_gnn` warmup op falls back to
  self-attn (Q=K/V=w_feat) instead of `continue`-ing, keeping
  `warmup_temp_graph_model` params connected without `find_unused_parameters`.

### Smoke test (2026-05-10, post-fix)
- **DGX 3796 (tpd smoke)**: ran clean past iter 153/5860 — total loss
  22.04 → 18.32 → 17.26, motion_loss_reg 4.23 → 2.57 → 1.66, no DDP errors,
  grad_norm 32–55 (clip will fire), 2.7 s/iter, ETA 4:21. Cancelled after
  smoke; full run resubmitted on Killarney.

### Resubmissions
- **Killarney 3498999 (tpd, full)** — RUNNING. As of last survey, at
  Iter 4437/5860 (~76%), ETA ~56 min, total loss 12.8–13.4,
  motion_loss_reg ~0.8, no DDP errors.
- **DGX 3804 (predonly_revival full eval) — COMPLETED (12 m 59 s).**
- **DGX 3805 (predonly_revival_simple eval) — COMPLETED (11 m 19 s).**
  Both unblocked by commit `0997406b` (motion_decoder forwards
  `num_output`) plus the `num_output=200` config setting.

### Recovered metrics (2026-05-10) — top-1 prediction only

predonly is the GT-perception oracle; the experiment goal is to push the
top-1 prediction ceiling when perception is exact. Reporting only the
metric the experiment is designed to move: `top1_ade` / `top1_fde` per
class. Multimodal min_ade/min_fde, miss_rate, EPA, and planning numbers
are tracked in the run logs but not headline here.

> **Class-lumping caveat.** `motion_utils.motion_name_mapping`
> (motion_utils.py:30) collapses the 10 detection classes into three
> motion-eval labels *before* metric accumulation:
> `car` ← {car, truck, construction_vehicle, bus, trailer, motorcycle,
> bicycle}; `pedestrian` ← {pedestrian}; `barrier` ← {barrier,
> traffic_cone}. So `car_top1_*` in the table below averages over cars
> *and* the four heavy-vehicle classes *and* both two-wheelers; it is not
> a car-only metric. `all_*` is the union of all three lumps (i.e. every
> motion-evaluable class, including stationary barriers/cones). A
> finer-grained breakdown (Vehicle / Pedestrian / Movable kept as the
> existing lumps; horizon and motion-behavior bins added) is being built
> in `tools/detailed_prediction_eval.py`.

| Class | Metric | full (3804) | simple (3805) | Δ (full − simple) |
|---|---|---:|---:|---:|
| car | top1_ade | **0.9584** | 0.9877 | −0.0293 |
| car | top1_fde | **2.2043** | 2.2736 | −0.0693 |
| pedestrian | top1_ade | 0.4478 | **0.4433** | +0.0044 |
| pedestrian | top1_fde | 0.9345 | **0.9258** | +0.0087 |
| all | top1_ade | **0.6641** | 0.6795 | −0.0155 |
| all | top1_fde | **1.4958** | 1.5339 | −0.0381 |

Bold = better.

### Confirmed Findings (predonly_revival full vs simple, top-1)

> **Superseded (2026-06-27):** the consistent Killarney re-eval batch
> (`3502404`/`3502405`, see "Predonly follow-up batch" under Outcomes
> 2026-05-11) shows full and simple **tie** (0.9523 vs 0.9529). The
> "full wins" read below was an artifact of comparing across two separate
> DGX eval runs (3804 vs 3805). Keep for history; trust the re-eval batch.

- **Full wins on car and on the all-class aggregate.** car_top1_ade
  −0.029, car_top1_fde −0.069; all_top1_ade −0.016, all_top1_fde −0.038.
  Mode classification (which the top-1 path is driven by) picks a
  meaningfully better mode in the full config.
- **Pedestrian flips.** simple wins ped by ≈0.004–0.009 on both metrics.
  Pedestrian trajectories are shorter-horizon and lower-variance, so the
  extra per-waypoint deformable + mode-mode SA capacity isn't adding
  signal there — possibly slight overfitting on the larger module.
- **Net read on top-1**: the two motion additions (per-waypoint
  deformable + persistent mode-mode SA) help the car-dominated long-tail
  classes' mode classification but neither isolation is established. We
  don't yet know which of the two is responsible, or whether one of them
  is doing the work and the other is dead weight.

Eval recovery note: the topk-out-of-range issue was
`SparseBox3DMotionDecoder` defaulting `num_output=300` against the
predonly `num_anchor=200`. Fix is `0997406b` + the predonly config addition
`motion_decoder=dict(type="SparseBox3DMotionDecoder", num_output=200)`.

### Other failed launches (2026-05-10)
- **DGX 3795 (tpd train)** — pre-fix attempt, ~1.5 m wall. DDP
  "Expected to have finished reduction…" with parameter indices 644 645
  646 647 697 698 unused on rank 1 — same `temp_gnn`/`warmup_temp_graph_model`
  disconnection that was later patched by the self-attn fallback in
  e79d0238. Subsumed by 3796 (smoke pass) and 3498999 (full run in flight).

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

## Outcomes (2026-05-11) — predonly follow-up batch + detailed eval

Two things landed after the 2026-05-10 top-1 numbers: (1) the 6-variant
follow-up batch cloned from `revival_simple` (commit `0d4094bd`), and (2) a
new offline detailed-prediction-eval framework (commit `c8e18ef1`) that dumps
raw per-class motion predictions and scores deterministic **L2 / Yaw / BEV-IoU
/ HitRate**, broken down by class lump (Vehicle / Pedestrian / Movable),
forecast horizon (0–2 / 2–4 / 4–6 s), and GT-trajectory maneuver (Stationary
/ Straight / Turning). This is a *different* metric family than the
`top1_ade`/`top1_fde` above — it scores the single deterministic trajectory,
not the multimodal min/top-1 selection.

### Detailed-eval headline (All-agent, μ / P90)

`.tex` sources in `reports/`. Baseline `bs24` = full-stack real perception
(not an oracle), included only as a floor.

| Run | L2 (m) | Yaw (°) | BEV IoU | Hit <2m / <1m |
|---|---:|---:|---:|---:|
| bs24 baseline (real perception) | 1.401 / 2.914 | 47.9 | 0.374 | 0.863 / 0.685 |
| revival **full** (`k3502404`) | 0.816 / 1.656 | 21.6 | 0.650 | 0.912 / 0.862 |
| revival **simple** | 0.837 / 1.680 | 21.6 | 0.649 | 0.911 / 0.863 |
| **modesk1_noclass** (`k3502409`) | **0.653 / 1.350** | **20.7** | **0.668** | **0.925 / 0.878** |

Per-class mean L2 (Vehicle / Pedestrian / Movable):
full `1.195 / 0.570 / 0.097`, simple `1.237 / 0.559 / 0.095`,
modesk1_noclass `0.943 / 0.479 / 0.088`.

### Confirmed findings (detailed eval)

- **Single-mode (`modesk1_noclass`) wins decisively on deterministic L2** —
  0.653 vs full 0.816 / simple 0.837 (−0.16 to −0.18 m, well above the
  ~0.012 noise floor), and it wins on *every* metric and *every* class lump,
  **including pedestrian** (0.479 vs full 0.570 / simple 0.559). It is
  `fut_mode=1` with `motion_loss_cls.loss_weight=0.0` — no mode-classifier.
- **Mechanism**: the 2026-05-10 top-1 analysis pegged the ceiling on the
  mode classifier (top1 trailed min_ade because argmax picked the wrong mode).
  Collapsing to one mode removes the classifier from the loop entirely, so the
  single trained trajectory is a better *deterministic* prediction than the
  6-mode top-1. This sidesteps — rather than solves — the classifier
  bottleneck flagged in "Next" item (2).
- **Pedestrian flip is gone under single-mode.** The full-vs-simple ped flip
  was a mode-classification artifact; with no classifier it disappears.
- **full ≈ simple confirmed on the new metric** (0.816 vs 0.837, full
  marginally better) — consistent with the top-1 ranking, so the new
  framework corroborates the earlier numbers.

### Occluded breakdown (revival full, `..._occluded_27.tex` / `..._reweighted.tex`)

Occluded-agent deterministic L2 is ~27% worse than visible (All 1.039 vs
0.816), concentrated in Vehicles (1.621 vs 1.195) and at long horizon
(4–6 s Straight Vehicle blows up to 6.9 m). Pedestrian/Movable degrade much
less. Per-(class × horizon × maneuver) table is in
`predonly_revival_full_detailed_pred_eval_occluded_27.tex`.

### Detection track — tpd + fusion (Killarney, recovered 2026-06-27)

All four ran to completion in May (~5 h 16–22 m each, clean `Done`) on
Killarney; logs recovered now that access is restored.
Full-stack vanilla (real perception). Baseline = `SparseDrive default`
(NDS 0.523 / mAP 0.413 / mAP_normal 0.553) in `results_summary.md`.

| Run | Job | NDS | mAP | mAP_norm | L2 | CR | car_EPA | car_minADE | all_minADE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (bs24) | — | 0.523 | 0.413 | 0.553 | — | — | 0.492 | — | — |
| **tpd** | 3498999 | **0.5226** | **0.4135** | **0.5677** | **0.5987** | 0.101% | **0.4906** | **0.6355** | **0.5537** |
| fuseA | 3500449 | 0.5165 | 0.4098 | 0.5589 | 0.5983 | 0.111% | 0.4827 | 0.6717 | 0.5749 |
| fuseB | 3500450 | 0.5213 | 0.4057 | 0.5558 | 0.6394 | 0.104% | 0.4820 | 0.6724 | 0.5722 |
| fuseC | 3500451 | 0.5173 | 0.3987 | 0.5569 | 0.6273 | 0.142% | 0.4740 | 0.6356 | 0.5616 |

**Confirmed findings (detection track):**

- **TPD is detection-neutral, not detection-positive.** tpd NDS 0.5226 /
  mAP 0.4135 sit right on the vanilla baseline (0.523 / 0.413) — within noise.
  The distinguishing TPD claim (forward-looking motion-cache K/V into the
  detection warmup) does **not** move detection mAP/NDS. The one bright spot is
  mAP_normal +0.015 (0.5677 vs 0.553), and it doesn't regress planning
  (L2 0.599, CR 0.101%) or motion (best car/all minADE of the four).
- **No fusion strategy beats injection-only TPD — clean negative.** All three
  fusion variants are *below* tpd on both NDS (0.5165 / 0.5213 / 0.5173) and
  mAP (0.4098 / 0.4057 / 0.3987), and fuseB/fuseC also regress planning L2
  (0.639 / 0.627 vs 0.599) and motion (car_minADE 0.67 vs 0.64). Folding warmup
  features into the main decoder's last refine *hurts* rather than helps. The
  report's primary question — does any fusion beat injection-only TPD on
  detection — is answered **no**.

### Predonly follow-up batch — top-1 isolation (Killarney, recovered 2026-06-27)

All 6 follow-up variants + a re-eval of full/simple completed on Killarney
(jobs `3502404–3502411`, all clean `Done`). **Critically, these were all
scored in one consistent eval batch** — the same framework that dumps
`motion_predictions.pkl` — so they are directly comparable to each other in a
way the original DGX (3804) vs DGX (3805) numbers were not.

`top1_ade_err` (lower = better), the metric this track targets. `min_ade`
shown for context (multimodal coverage, ~flat across the 6-mode variants).

| Variant | Job | car_top1 | all_top1 | car_min_ade | What it adds vs `simple` |
|---|---|---:|---:|---:|---|
| simple | 3502405 | 0.9529 | 0.6601 | 0.3578 | neither (control) |
| full | 3502404 | 0.9523 | 0.6603 | 0.3603 | per-waypoint **+** mode-SA |
| modeSAonly | 3502406 | 0.9690 | 0.6696 | 0.3585 | mode-mode SA only |
| **perwaypointonly** | 3502407 | **0.9361** | **0.6503** | 0.3597 | per-waypoint deformable only |
| classw2x | 3502410 | 0.9246 | 0.6425 | 0.3629 | motion_loss_cls ×2 |
| longer20ep | 3502411 | 0.9224 | 0.6382 | 0.3512 | 20 epochs |
| modesk4 | 3502408 | 0.9113 | 0.6332 | 0.4175 | fut_mode=4 |
| **modesk1_noclass** | 3502409 | **0.7545** | **0.5305** | 0.7545 | fut_mode=1, no cls loss |

**Confirmed findings (top-1 isolation):**

- **The original "full beats simple" headline was eval-batch noise.** In the
  consistent re-eval, full (0.9523) and simple (0.9529) **tie** (Δ 0.0006,
  far under the 0.012 noise floor) — not the −0.029 car gap the DGX batch
  showed. The 2026-05-10 "Confirmed Findings" section below is superseded by
  this batch.
- **2×2 isolates the mechanism: per-waypoint deformable helps, mode-mode SA
  hurts, and in `full` they cancel.** `perwaypointonly` 0.9361 beats simple
  by −0.017 car (above noise); `modeSAonly` 0.9690 is *worse* than simple by
  +0.016. `full` lands back at simple because the productive per-waypoint gain
  is canceled by the harmful mode-SA. → **graduate per-waypoint deformable,
  drop mode-mode SA.** This answers "Next" item (1) decisively.
- **Cheap classifier knobs help top-1, as predicted.** classw2x (0.9246) and
  longer20ep (0.9224) both beat simple while `min_ade` stays flat (~0.36) —
  confirming top-1 is mode-classifier-bottlenecked, not regression-bottlenecked
  ("Next" items 2 & 4).
- **Fewer modes trade coverage for top-1.** modesk4 (0.9113) and especially
  modesk1_noclass (0.7545) give the best top-1, but modesk1's `min_ade`
  collapses to 0.7545 (single mode → min == top1, no multimodal benefit).
  Consistent with the detailed-eval L2 result above where modesk1_noclass won
  deterministic L2. The choice is explicitly top-1/deterministic quality vs
  multimodal coverage.

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

### Follow-ups to push top-1 prediction further (predonly track)

Goal here is to raise `top1_ade`/`top1_fde` per class. Full beats simple
on car + all but flips on pedestrian, and we don't know which of the two
motion additions (per-waypoint deformable, mode-mode SA) is doing the work.
Concrete follow-ups, ordered by expected value:

1. **Isolate the two motion additions.** Two new configs cloned from
   `predonly_revival_simple`:
   - `..._revival_modeSAonly.py` — add `motion_self_attn` op + module
     back (no per-waypoint deformable). Tests whether mode-mode SA alone
     reproduces the car/all top-1 gain.
   - `..._revival_perwaypointonly.py` — add `motion_deformable_waypoints
     =[1,3,5,7,11]` back (no mode-mode SA). Tests the deformable side.
   Run both 10-epoch on Killarney; we want a 2×2 against full + simple.
   Whichever of the two reproduces the car_top1 gain alone graduates;
   the other gets dropped.
2. **Mode-classification loss reweight.** `top1_ade` is bottlenecked by
   the mode classifier — `min_ade` (argmax-free) was nearly identical
   between full and simple. Try `motion_loss_cls` weight 2× and 4× the
   default in `..._revival_simple_clsw2x.py` /`..._clsw4x.py`. Cheapest
   ablation: only the loss config changes, same modules.
3. **More modes.** Current `fut_mode=6`. Try `fut_mode=10` (and resize
   the kmeans motion anchor file accordingly). More modes = better
   coverage = mode classifier picks a closer-to-GT mode in argmax.
   Higher risk because the kmeans-anchor regeneration touches the data
   pipeline.
4. **Train longer.** Top-1 ADE is the slowest-converging metric (mode
   classification trains after regression stabilizes). Run
   `predonly_revival_simple` 20 epochs instead of 10 and watch whether
   the top1/min_ade gap closes. Establishes whether the current 10-epoch
   schedule is the limiter.
5. **Fix the pedestrian flip.** Pedestrian top-1 is *worse* with the
   richer motion stack. Likely the larger module overfits on a class
   with low intrinsic mode-diversity. A per-class mode count, or a
   pedestrian-specific reduced-capacity branch, could fix it — but
   that's a structural change and should wait until (1)–(2) land.

For (1)–(3), the noise floor for top1_ade per class is ~0.012 (single-run
deltas below that are not interpretable). Targeted improvements should
aim ≥0.02.

### Pending — detection track (TPD + 3 fusion variants)

For the TPD and TPD↔main fusion experiments the goal is detection
quality with no regression on full task set. Full metric set will be
populated when the runs land:

- **Killarney 3498999 (tpd)** and **3500449/50/51 (fuseA/B/C)** — **DONE,
  metrics recovered 2026-06-27** (see "Detection track" table in Outcomes
  2026-05-11). Result: TPD is detection-neutral vs the vanilla baseline, and
  none of the three fusion strategies beats injection-only TPD — both the
  primary (fusion > injection?) and the implicit (TPD > baseline?) questions
  came back negative.

**Status (resurvey 2026-06-27):** detection track is closed — TPD neutral,
fusion a clean negative; do not port fusion forward. The predonly track has
moved on to the single-mode result (`modesk1_noclass`), which is the current
front-runner on deterministic L2.
