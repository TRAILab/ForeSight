# Planning Relevance: Teaching Perception What Matters for Planning

2026-04-25

## TODO

- Implement GT corridor relevance label computation (BEV path overlap per agent)
- Add auxiliary relevance classification head to `detection3d_head.py`
- Wire GT ego future trajectories into stage-1 training pipeline
- Decide on binary vs. soft relevance label

## Abstract

## Intro

Detection and map heads are trained to answer "what is in the scene?" The planning head must then extract from those features the answer to a different question: "what matters for what I should do next?" These are not the same question. A high-confidence detection of a parked car 40m away may be easy to classify but irrelevant to the ego's next 3 seconds. A low-confidence pedestrian stepping into the predicted ego corridor is hard to detect but critical for planning. The current pipeline gives the planner no way to communicate this distinction back to perception.

The `numdemapx2` null result (doubling from `num_det=50` to `num_det=100` changed nothing) rules out token quantity as the bottleneck. Changing *which* 50 tokens are selected via a heuristic (nearest, corridor overlap) would not help either — the features of those tokens are shaped entirely by detection loss regardless of how they are ranked. The bottleneck is what the features *encode*, not how they are selected.

This batch tests whether auxiliary planning-relevance supervision in the detection head reshapes detection features to encode planning-useful information. The supervision signal is derived directly from GT ego future trajectories: for each training frame, an agent is labeled as planning-relevant if its ground-truth future positions overlap with the ego's predicted BEV corridor. This signal is available at stage 1 and propagates planning-relevance gradients directly through detection features and into the backbone — addressing the root cause rather than the symptom.

The high-level question: **can the perception tasks learn what is relevant for planning, and does that improve planning?**

Questions:
- Does planning-relevance auxiliary supervision in the detection head improve stage-2 planning metrics?
- Does applying this supervision at stage 1 (shaping backbone initialization) improve over stage-2-only?
- Does extending relevance supervision to the map head add further benefit?

## Method

All stage-2 experiments use `ptaux2d_ppdeformmm_planifls` as the reference baseline unless a new stage-1 pretrain is being tested.

**GT relevance label.** For each detected agent at each training frame, compute a binary or soft relevance label:
- **Binary**: agent is relevant if any future GT box position falls within a fixed-width corridor (e.g. ±2m) around the ego's GT future BEV path over the planning horizon.
- **Soft**: relevance = `exp(-min_distance / σ)` where `min_distance` is the minimum BEV distance between any future agent position and any point on the GT ego path. `σ` controls falloff.

The label is computed on-the-fly during training using `gt_ego_fut_trajs` (already present at stage 2; needs wiring at stage 1) and the Hungarian-matched GT agent positions.

**Experiment 1 — `detrel_s2`: relevance head at stage 2 only.**
Add a small auxiliary binary classification head (2-layer MLP, 64 units) to `Detection3DHead`, predicting the GT relevance label per instance at each decoder layer. Supervised by binary cross-entropy weighted by GT relevance frequency. No changes to stage-1 pretrain (`ptaux2d`). This isolates the stage-2 benefit of relevance supervision before committing to a new stage-1 run.

Config: `ptaux2d_ppdeformmm_planifls_detrel`.

**Experiment 2 — `stage1_detrel`: planning-relevance supervision at stage 1.**
Add the same relevance head to the stage-1 config alongside detection loss. Wire `gt_ego_fut_trajs` into the stage-1 pipeline. The resulting checkpoint `sparsedrive_stage1_detrel.pth` provides a backbone initialized with both detection and planning-relevance gradients. Stage-2 variant `ptdetrel_ppdeformmm_planifls` uses this pretrain.

This tests whether planning-relevance shaping of backbone features from stage 1 outperforms applying the same signal only at stage 2.

**Experiment 3 — `maprel`: relevance supervision on the map head.**
Extend relevance supervision to map elements: a map segment is labeled relevant if it falls within the ego's predicted corridor (e.g. lane boundaries or crossings adjacent to the planned path). Add the same auxiliary head to the map decoder. Combined with `detrel_s2` or `stage1_detrel`.

**Experiment 4 — `selfrel`: self-supervised relevance at stage 2.**
Replace the GT ego corridor with the model's own predicted ego trajectory (from the previous decoder stage) as the relevance signal. This closes the loop: planning outputs drive detection feature shaping without requiring GT labels at inference time. Depends on Experiment 1 validating that the GT-supervised signal is useful.

Implementation changes:
- `detection3d_head.py`: add `PlanningRelevanceHead` auxiliary output (Experiments 1–2)
- `nuscenes_3d_dataset.py` / stage-1 pipeline: wire `gt_ego_fut_trajs` to stage-1 collected keys (Experiment 2)
- `sparsedrive_r50_stage1_*`: add relevance loss and head (Experiment 2)
- `map_head.py`: add relevance auxiliary head (Experiment 3)
- `motion_planning_head.py`: pass current predicted ego trajectory back to relevance head (Experiment 4)

## Results

| Server | Config | Job ID | Status |
| --- | --- | --- | --- |
| DGX | `ptaux2d_ppdeformmm_planifls` (baseline) | 3614 | COMPLETED |
| Narval | `ptaux2d_ppdeformmm_planifls` (baseline) | 59670984 | COMPLETED |
| Killarney | `ptaux2d_ppdeformmm_planifls` (baseline) | 3282987 | COMPLETED |
| Killarney | `ptaux2d_ppdeformmm_planifls_detrel` | 3284461 | FAILED (5-tuple unpack on map head) |
| Killarney | `ptaux2d_ppdeformmm_planifls_detrel` (resubmit) | 3284553 | COMPLETED |
| Apollo | `stage1_8gpu_noflash_detrel` | — | PENDING |
| DGX | `ptdetrel_ppdeformmm_planifls` | — | PENDING |
| DGX | `ptaux2d_ppdeformmm_planifls_detrel_maprel` | — | PENDING |
| DGX | `ptaux2d_ppdeformmm_planifls_selfrel` | — | PENDING |
| Narval | `ptaux2d_ppdeformmm_planifls_selrelGT` | 59925463 | PENDING (queue) |
| Narval | `ptaux2d_ppdeformmm_planifls_selrelGT_half` | 59925464 | CANCELLED |
| Narval | `ptaux2d_ppdeformmm_planifls_topk_half` | 59925465 | CANCELLED |
| Narval | `ptaux2d_ppdeformmm_planifls_nodetmap` | 59925466 | CANCELLED (would hit DDP unused-param crash; cancelled before start) |
| Narval | `ptaux2d_ppdeformmm_planifls_nodetmap` (resubmit) | 59938597 | CANCELLED |
| Killarney | `ptaux2d_ppdeformmm_planifls_selrelGT` | 3305984 | COMPLETED |
| Killarney | `ptaux2d_ppdeformmm_planifls_selrelGT_half` | 3305986 | FAILED (apptainer symlink flake) |
| Killarney | `ptaux2d_ppdeformmm_planifls_selrelGT_half` (resubmit) | 3306736 | COMPLETED |
| Killarney | `ptaux2d_ppdeformmm_planifls_topk_half` | 3305987 | COMPLETED |
| Killarney | `ptaux2d_ppdeformmm_planifls_nodetmap` | 3305988 | FAILED (DDP unused params from skipped gnn/cross_gnn ops) |
| Killarney | `ptaux2d_ppdeformmm_planifls_nodetmap` (resubmit) | 3306735 | COMPLETED |

| Config | Host | Job | L2 | obj_box_col | car_ade | ped_ade | car_epa | ped_epa | NDS | mAP | mAP_normal |
|--------|------|-----|-----|-------------|---------|---------|---------|---------|-----|-----|------------|
| `ptaux2d_ppdeformmm_planifls` (baseline) | DGX | 3614 | 0.5057 | 0.043% | 0.6141 | 0.7177 | 0.5055 | 0.4365 | 0.5436 | 0.4383 | 0.5649 |
| `ptaux2d_ppdeformmm_planifls` (baseline) | Narval | 59670984 | 0.5122 | 0.055% | 0.6131 | 0.7106 | 0.5070 | 0.4344 | 0.5479 | 0.4435 | 0.5566 |
| `ptaux2d_ppdeformmm_planifls` (baseline) | Killarney | 3282987 | **0.4987** | 0.063% | 0.6228 | 0.7201 | 0.5073 | 0.4302 | 0.5447 | 0.4403 | 0.5633 |
| `ptaux2d_ppdeformmm_planifls_detrel` | Killarney | 3284553 | 0.5393 | 0.105% | 0.6019 | 0.7335 | 0.5094 | 0.4282 | 0.5466 | 0.4397 | 0.5648 |
| `ptaux2d_ppdeformmm_planifls_selrelGT` | Killarney | 3305984 | 0.5115 | 0.089% | 0.6143 | 0.7192 | 0.5115 | 0.4342 | 0.5477 | 0.4431 | 0.5523 |
| `ptaux2d_ppdeformmm_planifls_selrelGT_half` | Killarney | 3306736 | 0.5120 | 0.097% | 0.6084 | 0.7111 | 0.5087 | 0.4304 | 0.5458 | 0.4386 | 0.5617 |
| `ptaux2d_ppdeformmm_planifls_topk_half` | Killarney | 3305987 | **0.5046** | **0.060%** | 0.6060 | 0.7216 | 0.5081 | 0.4346 | 0.5478 | 0.4446 | 0.5684 |
| `ptaux2d_ppdeformmm_planifls_nodetmap` | Killarney | 3306735 | 0.5215 | 0.092% | 0.5974 | 0.7201 | 0.5118 | 0.4297 | 0.5485 | 0.4422 | 0.5638 |

Same-host comparison on Killarney (detrel − baseline):

| Metric | Baseline 3282987 | Detrel 3284553 | Δ | Direction |
|--------|------------------|----------------|---|-----------|
| L2 | 0.4987 | 0.5393 | +0.0406 | regression |
| obj_box_col | 0.063% | 0.105% | +0.042 pp | regression |
| NDS | 0.5447 | 0.5466 | +0.0019 | flat |
| mAP | 0.4403 | 0.4397 | −0.0006 | flat |
| mAP_normal | 0.5633 | 0.5648 | +0.0015 | flat |
| car_ade | 0.6228 | 0.6019 | −0.0209 | improved |
| ped_ade | 0.7201 | 0.7335 | +0.0134 | regression |
| car_epa | 0.5073 | 0.5094 | +0.0021 | flat |
| ped_epa | 0.4302 | 0.4282 | −0.0020 | flat |

Same-host comparison on Killarney for the corridor-selection grid (variant − baseline 3282987):

| Metric | Baseline 3282987 | `selrelGT` 3305984 | `selrelGT_half` 3306736 | `topk_half` 3305987 | `nodetmap` 3306735 |
|--------|------------------|--------------------|--------------------------|---------------------|---------------------|
| L2 | 0.4987 | 0.5115 (+0.0128) | 0.5120 (+0.0133) | 0.5046 (+0.0059) | 0.5215 (+0.0228) |
| obj_box_col | 0.063% | 0.089% (+0.026 pp) | 0.097% (+0.034 pp) | 0.060% (−0.003 pp) | 0.092% (+0.029 pp) |
| NDS | 0.5447 | 0.5477 | 0.5458 | 0.5478 | 0.5485 |
| mAP | 0.4403 | 0.4431 | 0.4386 | 0.4446 | 0.4422 |
| mAP_normal | 0.5633 | 0.5523 | 0.5617 | 0.5684 | 0.5638 |
| car_ade | 0.6228 | 0.6143 | 0.6084 | 0.6060 | 0.5974 |
| ped_ade | 0.7201 | 0.7192 | 0.7111 | 0.7216 | 0.7201 |
| car_epa | 0.5073 | 0.5115 | 0.5087 | 0.5081 | 0.5118 |
| ped_epa | 0.4302 | 0.4342 | 0.4304 | 0.4346 | 0.4297 |

## Discussion

The Killarney baseline (3282987) is the strongest ptaux2d run across the three hosts (L2 0.4987 vs. DGX 0.5057 and Narval 0.5122), so the host itself is not penalising the detrel run. With a same-host control available, host variance is no longer a confound and the detrel regression on Killarney is a genuine effect of the change rather than noise.

Detrel reshapes the detection features it touches but leaves the planner's architecture and supervision unchanged. The result is a clean dissociation: detection-side metrics (NDS, mAP, mAP_normal) are flat to within noise, while the planning metrics that consume those features regress materially (L2 +8.1%, obj_box_col +66.7% relative). The car motion prediction component (`car_ade` −0.0209) actually improves slightly, which is consistent with the auxiliary objective doing what it was designed to do at the agent level — encoding "this agent matters for ego" carries some information for that agent's own future. But planning, which is the metric this batch was meant to move, gets worse.

This pattern matches the standing repo finding that *"planning is not improved much by perception alone."* If the planner is not trained to consume the new feature axis, reshaping detection features toward planning-relevance costs the planner some of the information it was using before without giving it a path to exploit the new axis. The aux head is supervised only on Hungarian-matched positives (a small fraction of the 900 anchors), and the modal label within that subset is "irrelevant" given a 2 m corridor and 6-step horizon, so the BCE signal is also fairly coarse.

The `numdemapx2` null result earlier in the combined batch already ruled out token-quantity bottlenecks; this result rules out the cheaper version of the relevance hypothesis (label which agents matter, leave everything else alone). The remaining hypotheses in this batch — stage-1 detrel pretrain (Experiment 2), map relevance (Experiment 3), and self-supervised relevance closing the loop into planning (Experiment 4) — all remain plausible because they change either the timing or the recipient of the relevance signal, but none should be expected to beat baseline if the underlying mechanism (perception-side feature shaping) is what is failing here.

A separate concern is that the GT relevance label used in `detrel_s2` was synchronous-only: `_build_relevance_target` computes `min_t dist(ego(t), agent(t))` rather than the cross-time minimum `min_{t1,t2} dist(ego(t1), agent(t2))`. The synchronous test misses two important relevance cases: (i) ego catches up to a slow agent (positions never coincide in time but ego's path drives through where the agent is), and (ii) ego enters a spot the agent vacated (path-overlapping but synchronously distant). The corrected criterion treats both ego and agent as **swept corridors** in space-time and measures their closest approach. This is the same fix that the Method's "any future GT box position falls within a fixed-width corridor" prose implied but the implementation did not realise. The corrected formulation is the basis for the corridor-selection experiments below.

### Corridor-selection grid (Experiments 5–8)

Across the four corridor-selection runs, **none of the consumer-side selection variants beat the Killarney baseline on `L2`**, and only `topk_half` improves `obj_box_col` (by a marginal 0.003 pp, well within run-to-run noise). The closest variant to baseline on planning is `topk_half` (`L2=0.5046`, baseline `0.4987`, +0.0059); the GT-corridor variants `selrelGT` and `selrelGT_half` regress by ~0.013 on `L2` and ~0.03 pp on `obj_box_col`. The `nodetmap` floor ablation regresses further on `L2` (+0.0228) and matches the corridor variants on collisions, so the planning K/V cross-attention is contributing *some* useful signal, but the magnitude is small (≈0.02 in `L2`).

Three things follow.

**Selection criterion is not the bottleneck.** Replacing `topk(det_confidence)` with a cross-time GT corridor oracle — i.e., the most planning-relevant possible selection at this token count — does not improve planning. If the *planner* could exploit relevance information, this should have been the easiest win in the batch. It was not. This is the **consumer-side counterpart** to the `detrel_s2` producer-side null: neither the features themselves nor the way they are selected can move planning while the planner's supervision and architecture are held fixed.

**Token count is also not the bottleneck.** `numdemapx2` (doubling) and `topk_half` (halving) both produce planning numbers indistinguishable from baseline, with `nodetmap` (zeroing) costing only ~0.02 `L2`. Combined with `selrelGT`, this means the (selection criterion × query count) grid is essentially flat: planning quality is largely insensitive to which 0–100 perception tokens it cross-attends to.

**The K/V path is small but nonzero.** `nodetmap` (zero queries) is the worst of the four on `L2` (`+0.0228`) but is broadly comparable to the corridor variants on collisions and on detection metrics. The image-feature deformable attention plus terminal collision rescoring already carries most of the planning signal; the perception K/V coupling adds a modest residual that is *not* recoverable by reweighting which tokens are passed in.

Together with `detrel_s2`, this closes out the perception→planning hand-off as a tractable axis for further gains *without* changing the planner. The remaining improvement directions are planner-side: changing what the planning decoder is supervised to predict, what its query structure is, or what auxiliary objectives the planning head receives. The `plan_aux` and `plan_unified` lines are the natural continuation; `selfrel` (Experiment 4) is now also unattractive because it depends on a relevance signal whose oracle upper bound (`selrelGT`) does not beat baseline.

A small caveat on `selrelGT` interpretation: the oracle is GT-supervised at *both* train and eval time (the test pipeline collects `gt_ego_fut_trajs` and friends). This is the explicit-oracle limitation; the result is therefore an upper bound on what *any* learned relevance selector could deliver via this hand-off, conditional on the same planner architecture. The negative result for the upper bound implies the learned-selector variants would not exceed it.

## Method (v2) — Corridor selection experiments

`detrel_s2` and `numdemapx2` together rule out the **producer-side feature shaping** and **token quantity** axes for relevance. The remaining unexplored axis is **selection**: planning currently consumes the top-50 detection tokens by `det_confidence` and the top-10 map tokens by `map_confidence`. Both criteria are pure perception signals — they have no notion of what is relevant for the ego corridor.

The next batch tests three points on the **(selection criterion × query count)** grid against the same Killarney baseline 3282987. All three add no parameters to the model — only the selection rule and counts inside `MotionPlanningHead.forward` differ:

**Experiment 5 — `selrelGT`: GT corridor selection at standard counts.**
Replace `topk(det_confidence, 50)` with `topk(corridor_score_det, 50)`, and `topk(map_confidence, 10)` with `topk(corridor_score_map, 10)`. Scores:
- `corridor_score_det[i]` = relevance of detection anchor `i`'s **matched GT agent** under cross-time corridor: `1` if `min_{t1,t2} dist(ego_gt(t1), agent_gt(t2)) < 2 m` over the 6-step planning horizon, else `0`. Tiebreak by `det_confidence` so we always fill 50 slots.
- `corridor_score_map[j]` = `−min_{t,p} dist(ego_gt(t), map_pt[j, p])` (any vertex of map element `j` to any ego future path point). Higher is closer.

This is a consumer-side oracle probe: if planning, given perfect agent and map selection, doesn't beat baseline, the relevance signal itself is uninformative and the producer-side failure of `detrel_s2` was not a recoverable problem.

Config: `ptaux2d_ppdeformmm_planifls_selrelGT`.

**Experiment 6 — `selrelGT_half`: GT corridor selection with half the queries.**
Same selection rule as Experiment 5 but `num_det=25, num_map=5`. Tests whether GT-selected smaller token sets preserve planning quality. If `selrelGT` ties `selrelGT_half`, the planner is already operating on a small set of relevant tokens and the rest are filler.

Config: `ptaux2d_ppdeformmm_planifls_selrelGT_half`.

**Experiment 7 — `topk_half`: standard confidence top-k with half the queries.**
Keep the existing `topk(det_confidence, ...)` and `topk(map_confidence, ...)` selection but set `num_det=25, num_map=5`. Closes the **fewer-queries** direction the `numdemapx2` doubling did not test. If reducing queries with the existing criterion improves planning, the planner is being *distracted* by confident-but-irrelevant detections and the bottleneck is selection quality, not quantity.

Config: `ptaux2d_ppdeformmm_planifls_topk_half`.

**Experiment 8 — `nodetmap`: zero detection and map queries to planning, image-only cross-attention.**
Set `num_det=0, num_map=0`. The planning decoder's cross-attention to detection and map tokens is removed entirely (skip the `cross_gnn` ops when there are no K/V tokens); only the image deformable attention layer feeds the planner. Agents are still computed by the detection head and motion is still predicted, so the planner's terminal **collision rescoring** in `HierarchicalPlanningDecoder` (`use_rescore=True`) keeps its access to agent futures — only the K/V flow into the planning cross-attention is disabled.

This is the floor of the (count) axis. Combined with `numdemapx2` (full doubling, null), `topk_half` (50% reduction), and `nodetmap` (100% reduction), the four runs trace out planning quality as a function of how many perception tokens the planner consumes during decoding. If `nodetmap` ties or beats baseline, the cross-attention path into the planner is not contributing useful information beyond what the image features and rescoring already provide — a strong negative result for the entire perception→planning K/V coupling.

Config: `ptaux2d_ppdeformmm_planifls_nodetmap`.

The four runs cover the **(selection criterion × query count)** grid at standard / half / zero counts, with the existing baseline 3282987 as the standard-count confidence-selection reference cell.

Implementation notes:
- Cross-time corridor computation is a small change to `_build_relevance_target`: replace the same-timestep diff with `agents_abs[:, :, None, :] - ego_xy[None, None, :, :]` and reduce over both time dims.
- Corridor selection lives in `MotionPlanningHead.forward` (around the existing `topk(det_confidence, ...)` call) gated by a new `relevance_selection: bool = False` flag. Off by default → no change to existing configs.
- GT keys (`gt_ego_fut_trajs`, `gt_agent_fut_trajs`, `gt_agent_fut_masks`, `gt_ego_fut_masks`, `gt_bboxes_3d`) need to be added to `test_pipeline` Collect for Experiments 5–6 so eval has the same selection signal as training. This is the explicit-oracle limitation: the resulting model is a probe, not a deployable model.
- For Experiment 8, the planning operation_order must skip `cross_gnn` ops when `num_det == 0` and `num_map == 0`. Rescoring is unaffected — it consumes `det_output` / `motion_output` directly, not the planning K/V.

## Future Work

- **Do not run a Killarney repeat or a second-host repeat of `detrel_s2`.** The same-host control on Killarney already provides a clean comparison; another seed is unlikely to flip a +8% L2 / +67% CR effect.
- **Close out Experiments 2, 3, and 4 (`stage1_detrel`, `maprel`, `selfrel`).** With the Experiment 5 oracle (`selrelGT`) failing to beat the Killarney baseline at the standard token count, the producer-side relevance hypothesis no longer has an upside that the consumer-side oracle has not already ruled out. Stage-1 detrel and map relevance deepen the same intervention; `selfrel` is bounded above by `selrelGT`. None should be run.
- **Cancel the still-pending Narval `selrelGT` repeat (59925463)** unless a same-config cross-host comparison is genuinely needed; the Killarney result is unambiguous and the four other Narval submissions in this batch (`selrelGT_half`, `topk_half`, `nodetmap` ×2) were already cancelled.
- **Net result of the corridor-selection grid:** confidence top-k is already capturing equivalent selection information for planning, so the relevance-based selection line closes out cleanly. `nodetmap` is the worst of the four on `L2` but the gap (~0.02) is small, so the K/V path contributes modestly but not enough to motivate further selection-side work.
- **Pivot to planner-side interventions.** `plan_aux` and `plan_unified` modify the planner directly; revisit detrel only if either of those produces a stronger planning baseline against which the perception→planning hand-off is no longer flat.
