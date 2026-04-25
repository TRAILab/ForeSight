# Unified Planning: Co-Evolving Detection and Planning Tokens

2026-04-25

## TODO

- Implement all-det cross-attention option in `MotionPlanningHead`
- Implement bidirectional planning→detection cross-attention
- Add GT ego future trajectories to stage-1 training pipeline

## Abstract

## Intro

In the current SparseDrive architecture, detection tokens are fully computed before planning sees them. `MotionPlanningHead` reads a frozen slice of `det_output` via confidence-based top-k selection; planning queries cross-attend to those tokens but cannot influence how they were computed. This is a one-way dependency: detection informs planning, but planning never informs detection.

The hypothesis is that planning quality is limited by this unidirectional, late coupling. If planning queries could interact with detection tokens during their computation — not just after — detection representations could be shaped by what is useful for the ego decision. This batch explores a progressive series of changes that loosen the sequential dependency, starting from the cheapest intervention (stage-1 planning supervision) and building toward bidirectional detection–planning interaction at each transformer layer.

Questions:
- Does supervising the backbone with planning loss at stage 1 improve stage-2 planning beyond `ptaux2d`?
- Does removing the top-k bottleneck (planning attending to all 900 detection tokens) improve planning?
- Does bidirectional coupling — planning queries writing back into detection token representations — further improve planning?

## Method

All stage-2 experiments use `ptaux2d_ppdeformmm_planifls` (DGX 3614) as the reference baseline unless a new stage-1 pretrain is being tested.

**Experiment 1 — `stage1_planifls`: planning supervision at stage 1.**
Enable `with_motion_plan=True` in the stage-1 config with the full `ppdeformmm_planifls` planning head architecture. Wire `gt_ego_fut_trajs` into the stage-1 pipeline. The resulting checkpoint `sparsedrive_stage1_planifls.pth` provides a backbone that has received planning gradients from the start, not just detection. Stage-2 variant `ptplan1_ppdeformmm_planifls` uses this pretrain with no other changes.

This tests whether planning-gradient-aligned backbone initialization outperforms detection-only initialization (`ptaux2d`). Can also be combined with `aux2d` stage-1 supervision to test additivity.

**Experiment 2 — `alldet`: planning attends to all detection tokens.**
Remove the `topk(det_confidence, self.num_det, ...)` call in `MotionPlanningHead`. Replace with direct cross-attention from planning queries to all 900 detection instance features. The cross-attention mechanism already handles variable-length inputs; this is a one-line change in the forward pass. No new parameters.

This removes the selection bottleneck entirely. If planning performance does not improve, the bottleneck is not selection — it is feature quality or the one-way token flow. If it does improve, confidence-based top-k was actively discarding planning-relevant agents.

Config: `ptaux2d_ppdeformmm_planifls_alldet`.

**Experiment 3 — `bidir`: bidirectional planning–detection coupling.**
After each planning cross-attention step (planning queries attending to detection features), run a reverse cross-attention: detection instance features attend to the current planning query state and update their representations. This is a lightweight bidirectional coupling — planning state can reshape detection features within the same forward pass. The additional reverse cross-attention layer adds one `AttentionLayer` per planning decoder stage.

This tests whether the unidirectional flow is the binding constraint. If detection features updated by planning state produce better planning outputs, it confirms that planning needs to write back into perception, motivating the full ego-query integration.

Config: `ptaux2d_ppdeformmm_planifls_bidir`.

**Experiment 4 — `egoquery`: ego query in the detection transformer.**
Add ego planning queries (one per mode, `ego_fut_mode=3`) to the instance bank inside `Detection3DHead`. Ego queries participate in all sparse4D self-attention and deformable attention layers alongside the 900 detection instance queries. At each layer, ego queries attend to all agent queries and to image features directly; agent queries attend back to ego queries. A trajectory output head on ego queries produces planning predictions. Planning loss and detection loss are jointly computed in the same forward pass.

`MotionPlanningHead` is retained as a lightweight refinement stage on top of ego query outputs (preserving the `planifls` endpoint attention mechanism). Requires changes to `instance_bank.py`, `detection3d_head.py`, and `sparsedrive_head.py`. Gated on Experiments 1–3 providing directional validation.

Config: `ptaux2d_egoquery_planifls`.

Implementation changes:
- `nuscenes_3d_dataset.py` / stage-1 pipeline: wire `gt_ego_fut_trajs` to stage-1 collected keys (Experiment 1)
- `sparsedrive_r50_stage1_*`: set `with_motion_plan=True`, add planning head config (Experiment 1)
- `motion_planning_head.py`: replace `topk()` with full-instance cross-attention (Experiment 2)
- `motion_planning_head.py`: add reverse cross-attention block after each planning layer (Experiment 3)
- `instance_bank.py`, `detection3d_head.py`, `sparsedrive_head.py`: add ego query slots and trajectory head (Experiment 4)

## Results

| Server | Config | Job ID | Status |
| --- | --- | --- | --- |
| Apollo | `stage1_8gpu_noflash_planifls` | — | PENDING |
| DGX | `ptplan1_ppdeformmm_planifls` | — | PENDING |
| DGX | `ptaux2d_ppdeformmm_planifls_alldet` | — | PENDING |
| DGX | `ptaux2d_ppdeformmm_planifls_bidir` | — | PENDING |
| DGX | `ptaux2d_egoquery_planifls` | — | PENDING |

| Config | L2 | obj_box_col | car_ade | ped_ade | car_epa | ped_epa | NDS | mAP |
|--------|-----|-------------|---------|---------|---------|---------|-----|-----|
| `ptaux2d_ppdeformmm_planifls` (baseline, DGX 3614) | 0.5057 | 0.043% | 0.6141 | 0.7177 | 0.5055 | 0.4365 | 0.5436 | 0.4383 |
| `ptplan1_ppdeformmm_planifls` | — | — | — | — | — | — | — | — |
| `ptaux2d_ppdeformmm_planifls_alldet` | — | — | — | — | — | — | — | — |
| `ptaux2d_ppdeformmm_planifls_bidir` | — | — | — | — | — | — | — | — |
| `ptaux2d_egoquery_planifls` | — | — | — | — | — | — | — | — |

## Discussion

## Future Work

- If `ptplan1` beats `ptaux2d`: combine planning + aux2d at stage 1 to test additivity; the stage-1 gradient alignment story becomes central to the paper.
- If `alldet` does nothing: the selection bottleneck is not binding; the information is not in the detection features regardless of how many you see. Motivates `plan_relevance` (better detection features) as the primary lever.
- If `alldet` helps: confidence-based top-k was discarding relevant agents; combine with `detrel` from `plan_relevance` to test whether better features and better access are additive.
- If `bidir` helps over `alldet`: the unidirectional flow is a real constraint; proceed to `egoquery`.
- Combine with `plan_relevance` and `plan_aux` findings once all three mature.
