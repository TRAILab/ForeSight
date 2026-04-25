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
| Killarney | `ptaux2d_ppdeformmm_planifls_detrel` | — | PENDING |
| Apollo | `stage1_8gpu_noflash_detrel` | — | PENDING |
| DGX | `ptdetrel_ppdeformmm_planifls` | — | PENDING |
| DGX | `ptaux2d_ppdeformmm_planifls_detrel_maprel` | — | PENDING |
| DGX | `ptaux2d_ppdeformmm_planifls_selfrel` | — | PENDING |

| Config | L2 | obj_box_col | car_ade | ped_ade | car_epa | ped_epa | NDS | mAP |
|--------|-----|-------------|---------|---------|---------|---------|-----|-----|
| `ptaux2d_ppdeformmm_planifls` (baseline, DGX 3614) | 0.5057 | 0.043% | 0.6141 | 0.7177 | 0.5055 | 0.4365 | 0.5436 | 0.4383 |
| `detrel_s2` | — | — | — | — | — | — | — | — |
| `ptdetrel` (stage-1 pretrain) | — | — | — | — | — | — | — | — |
| `detrel_maprel` | — | — | — | — | — | — | — | — |
| `selfrel` | — | — | — | — | — | — | — | — |

## Discussion

## Future Work

- If `detrel_s2` helps: run Narval repeat to confirm, then proceed to stage-1 pretrain.
- If stage-1 `ptdetrel` beats `ptaux2d`: planning-relevance shaping of backbone features is the dominant effect; combine `detrel` + `aux2d` at stage 1 to test additivity.
- If `selfrel` works: this closes the planning→perception feedback loop without GT labels at inference, strengthening the paper claim.
- Analyse which agents get high relevance scores vs. high detection confidence — do they diverge? In which scenarios? This is the core evidence for the paper's thesis.
- Combine with `plan_aux` and `plan_unified` findings once all three mature.
