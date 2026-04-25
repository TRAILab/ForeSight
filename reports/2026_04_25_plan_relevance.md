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

| Config | Host | Job | L2 | obj_box_col | car_ade | ped_ade | car_epa | ped_epa | NDS | mAP | mAP_normal |
|--------|------|-----|-----|-------------|---------|---------|---------|---------|-----|-----|------------|
| `ptaux2d_ppdeformmm_planifls` (baseline) | DGX | 3614 | 0.5057 | 0.043% | 0.6141 | 0.7177 | 0.5055 | 0.4365 | 0.5436 | 0.4383 | 0.5649 |
| `ptaux2d_ppdeformmm_planifls` (baseline) | Narval | 59670984 | 0.5122 | 0.055% | 0.6131 | 0.7106 | 0.5070 | 0.4344 | 0.5479 | 0.4435 | 0.5566 |
| `ptaux2d_ppdeformmm_planifls` (baseline) | Killarney | 3282987 | **0.4987** | **0.063%** | 0.6228 | 0.7201 | 0.5073 | 0.4302 | 0.5447 | 0.4403 | 0.5633 |
| `ptaux2d_ppdeformmm_planifls_detrel` | Killarney | 3284553 | 0.5393 | 0.105% | 0.6019 | 0.7335 | 0.5094 | 0.4282 | 0.5466 | 0.4397 | 0.5648 |

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

## Discussion

The Killarney baseline (3282987) is the strongest ptaux2d run across the three hosts (L2 0.4987 vs. DGX 0.5057 and Narval 0.5122), so the host itself is not penalising the detrel run. With a same-host control available, host variance is no longer a confound and the detrel regression on Killarney is a genuine effect of the change rather than noise.

Detrel reshapes the detection features it touches but leaves the planner's architecture and supervision unchanged. The result is a clean dissociation: detection-side metrics (NDS, mAP, mAP_normal) are flat to within noise, while the planning metrics that consume those features regress materially (L2 +8.1%, obj_box_col +66.7% relative). The car motion prediction component (`car_ade` −0.0209) actually improves slightly, which is consistent with the auxiliary objective doing what it was designed to do at the agent level — encoding "this agent matters for ego" carries some information for that agent's own future. But planning, which is the metric this batch was meant to move, gets worse.

This pattern matches the standing repo finding that *"planning is not improved much by perception alone."* If the planner is not trained to consume the new feature axis, reshaping detection features toward planning-relevance costs the planner some of the information it was using before without giving it a path to exploit the new axis. The aux head is supervised only on Hungarian-matched positives (a small fraction of the 900 anchors), and the modal label within that subset is "irrelevant" given a 2 m corridor and 6-step horizon, so the BCE signal is also fairly coarse.

The `numdemapx2` null result earlier in the combined batch already ruled out token-quantity bottlenecks; this result rules out the cheaper version of the relevance hypothesis (label which agents matter, leave everything else alone). The remaining hypotheses in this batch — stage-1 detrel pretrain (Experiment 2), map relevance (Experiment 3), and self-supervised relevance closing the loop into planning (Experiment 4) — all remain plausible because they change either the timing or the recipient of the relevance signal, but none should be expected to beat baseline if the underlying mechanism (perception-side feature shaping) is what is failing here.

## Future Work

- **Do not run a Killarney repeat or a second-host repeat of `detrel_s2`.** The same-host control on Killarney already provides a clean comparison; another seed is unlikely to flip a +8% L2 / +67% CR effect.
- **Cheap diagnostic before any architectural follow-up**: rerun `detrel_s2` with `loss_relevance.loss_weight=0.05` (vs. 0.2). If the regression scales monotonically with weight, this is feature-shaping interference; if the regression is flat, the aux head is mis-specified and the corridor / horizon should be revisited.
- **Skip Experiments 2 and 3 unless Experiment 4 (`selfrel`) shows a real effect.** Stage-1 detrel and map relevance both deepen the same intervention that already failed at stage 2; they are unlikely to reverse the sign of the effect.
- **Prioritise `selfrel`**: replacing the GT corridor with the planner's own predicted ego trajectory means the relevance signal is generated by the consumer, which is the missing piece in the current setup. This is the only Experiment in this batch that does not assume "perception-side reshape alone helps planning."
- **Combine with `plan_aux` and `plan_unified` findings**: those batches modify the planner directly; if either shows a real planning gain, revisit detrel on top of that stronger planner before declaring the relevance hypothesis dead.
