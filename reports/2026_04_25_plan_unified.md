# Unified Planning: Co-Evolving Detection and Planning Tokens

2026-04-25

## TODO

- [x] Implement all-det cross-attention option in `MotionPlanningHead`
- [x] Implement bidirectional planning→detection cross-attention
- Add GT ego future trajectories to stage-1 training pipeline (deferred — Exp 1 not in current scope)

## Abstract

## Intro

In the current SparseDrive architecture, detection tokens are fully computed before planning sees them. `MotionPlanningHead` reads a frozen slice of `det_output` via confidence-based top-k selection; planning queries cross-attend to those tokens but cannot influence how they were computed. This is a one-way dependency: detection informs planning, but planning never informs detection.

The hypothesis is that planning quality is limited by this unidirectional, late coupling. If planning queries could interact with detection tokens during their computation — not just after — detection representations could be shaped by what is useful for the ego decision. This batch explores a progressive series of changes that loosen the sequential dependency, starting from the cheapest intervention (stage-1 planning supervision) and building toward bidirectional detection–planning interaction at each transformer layer.

Questions:
- Does supervising the backbone with planning loss at stage 1 improve stage-2 planning beyond `ptaux2d`?
- Does removing the top-k bottleneck (planning attending to all 900 detection tokens) improve planning?
- Does bidirectional coupling — planning queries writing back into detection token representations — further improve planning?

## Method

Current scope: Experiments 2 (`alldet`) and 3 (`bidir`) on killarney, branch `sd_combined`. Experiments 1 (`stage1_planifls`) and 4 (`egoquery`) are described below for narrative continuity but are deferred to a later batch.

All stage-2 experiments use `ptaux2d_ppdeformmm_planifls` (DGX 3614) as the reference baseline unless a new stage-1 pretrain is being tested.

**Experiment 1 — `stage1_planifls`: planning supervision at stage 1.**
Enable `with_motion_plan=True` in the stage-1 config with the full `ppdeformmm_planifls` planning head architecture. Wire `gt_ego_fut_trajs` into the stage-1 pipeline. The resulting checkpoint `sparsedrive_stage1_planifls.pth` provides a backbone that has received planning gradients from the start, not just detection. Stage-2 variant `ptplan1_ppdeformmm_planifls` uses this pretrain with no other changes.

This tests whether planning-gradient-aligned backbone initialization outperforms detection-only initialization (`ptaux2d`). Can also be combined with `aux2d` stage-1 supervision to test additivity.

**Experiment 2 — `alldet`: planning attends to all detection tokens.**
Add a `use_alldet_kv` flag to `MotionPlanningHead`. When set, the `gnn` op's keys/values become the full instance set (`instance_feature[:, :num_anchor + 1]` = 900 detection tokens + ego, no DN tokens) instead of the `topk(det_confidence, num_det=50, ...)` selection. The existing cross-attention mechanism handles the variable length; no new parameters.

This removes the selection bottleneck entirely. If planning performance does not improve, the bottleneck is not selection — it is feature quality or the one-way token flow. If it does improve, confidence-based top-k was actively discarding planning-relevant agents.

Config: `ptaux2d_ppdeformmm_planifls_alldet`.

**Experiment 3 — `bidir`: bidirectional planning–detection coupling.**
After each planning `gnn` op (planning queries attending to detection features), insert a new `rev_gnn` op: detection instance features attend to the current planning query state (`plan_mode_query + ego instance feature + ego anchor embed`) and update their representations. This is a lightweight bidirectional coupling — planning state can reshape detection features within the same forward pass. The additional reverse cross-attention layer adds one `MultiheadFlashAttention` per planning decoder stage (3 total). `bidir` keeps the standard top-50 detection K/V (does not stack with `alldet`), isolating the reverse-flow effect.

This tests whether the unidirectional flow is the binding constraint. If detection features updated by planning state produce better planning outputs, it confirms that planning needs to write back into perception, motivating the full ego-query integration.

Config: `ptaux2d_ppdeformmm_planifls_bidir`.

**Experiment 4 — `egoquery`: ego query in the detection transformer.**
Add ego planning queries (one per mode, `ego_fut_mode=3`) to the instance bank inside `Detection3DHead`. Ego queries participate in all sparse4D self-attention and deformable attention layers alongside the 900 detection instance queries. At each layer, ego queries attend to all agent queries and to image features directly; agent queries attend back to ego queries. A trajectory output head on ego queries produces planning predictions. Planning loss and detection loss are jointly computed in the same forward pass.

`MotionPlanningHead` is retained as a lightweight refinement stage on top of ego query outputs (preserving the `planifls` endpoint attention mechanism). Requires changes to `instance_bank.py`, `detection3d_head.py`, and `sparsedrive_head.py`. Gated on Experiments 1–3 providing directional validation.

Config: `ptaux2d_egoquery_planifls`.

Implementation changes (current scope):
- `motion_planning_head.py`: add `use_alldet_kv` flag — the `gnn` op uses `instance_feature[:, :num_anchor + 1]` as K/V when enabled (Experiment 2)
- `motion_planning_head.py`: add `bidir_planning` flag and `rev_gnn` op handler — reverse cross-attention from agent features to planning state (Experiment 3)
- `motion_planning_head.py`: add `rev_graph_model` cfg slot in `op_config_map` (Experiment 3)

Deferred (Experiments 1, 4):
- `nuscenes_3d_dataset.py` / stage-1 pipeline: wire `gt_ego_fut_trajs` to stage-1 collected keys (Experiment 1)
- `sparsedrive_r50_stage1_*`: set `with_motion_plan=True`, add planning head config (Experiment 1)
- `instance_bank.py`, `detection3d_head.py`, `sparsedrive_head.py`: add ego query slots and trajectory head (Experiment 4)

## Results

| Server | Config | Job ID | Status |
| --- | --- | --- | --- |
| Apollo | `stage1_8gpu_noflash_planifls` | — | DEFERRED |
| DGX | `ptplan1_ppdeformmm_planifls` | — | DEFERRED |
| Killarney | `ptaux2d_ppdeformmm_planifls` (baseline) | 3282987 | COMPLETED |
| Killarney | `ptaux2d_ppdeformmm_planifls_alldet` | 3284640 | COMPLETED |
| Killarney | `ptaux2d_ppdeformmm_planifls_bidir` | 3284641 | COMPLETED |
| DGX | `ptaux2d_egoquery_planifls` | — | DEFERRED |

| Config | L2 | obj_box_col | car_ade | ped_ade | car_epa | ped_epa | NDS | mAP |
|--------|-----|-------------|---------|---------|---------|---------|-----|-----|
| `ptaux2d_ppdeformmm_planifls` (baseline, DGX 3614) | 0.5057 | 0.043% | 0.6141 | 0.7177 | 0.5055 | 0.4365 | 0.5436 | 0.4383 |
| `ptaux2d_ppdeformmm_planifls` (baseline, Killarney 3282987) | 0.4987 | 0.063% | 0.6228 | 0.7201 | 0.5073 | 0.4302 | 0.5447 | 0.4403 |
| `ptplan1_ppdeformmm_planifls` | — | — | — | — | — | — | — | — |
| `ptaux2d_ppdeformmm_planifls_alldet` (Killarney 3284640) | 0.5018 | 0.084% | 0.6031 | 0.7119 | 0.5090 | 0.4270 | 0.5477 | 0.4404 |
| `ptaux2d_ppdeformmm_planifls_bidir` (Killarney 3284641) | 0.5378 | 0.057% | 0.6069 | 0.7227 | 0.5134 | 0.4280 | 0.5442 | 0.4386 |
| `ptaux2d_egoquery_planifls` | — | — | — | — | — | — | — | — |

## Discussion

Both experiments are compared against the **server-matched** killarney baseline (3282987) rather than the DGX 3614 baseline, because the cross-server L2 gap (0.5057 → 0.4987) is comparable to the per-experiment effect size and would otherwise mask the result.

**`alldet` (job 3284640): no planning improvement.** L2 moves from 0.4987 → 0.5018 (+0.0031, within run-to-run noise) and `obj_box_col` worsens from 0.063% → 0.084%. Motion-prediction metrics tick up slightly (`car_ade` 0.6228 → 0.6031, `ped_ade` 0.7201 → 0.7119) and detection metrics are essentially unchanged. Removing the top-50 detection bottleneck and giving planning queries direct cross-attention to all 900 detection tokens does not improve planning. This rules out token-quantity / selection-heuristic as the binding constraint at the K/V layer — the selection is not throwing away planning-relevant information that planning could otherwise use. Consistent with the earlier `numdemapx2` null result (50 → 100 had no effect): the bottleneck is not how many detection tokens planning sees, it is what those tokens encode.

**`bidir` (job 3284641): planning regresses.** L2 0.4987 → 0.5378 (+0.039, clearly outside noise); `obj_box_col` improves marginally (0.063% → 0.057%); detection and motion metrics are flat. Adding reverse cross-attention so detection features attend to the current planning state per decoder stage hurts more than it helps. Hypothesis: writing planning information into the agent token slice destabilizes the detection-side refine targets the same tokens are still being supervised against — the agent representation has to serve both detection regression/classification and a planning-conditioned write, and the joint objective drifts. The fact that collision rate did not regress alongside L2 also suggests the planning head is producing more conservative trajectories rather than learning a better policy.

**Combined reading.** Both interventions targeted the *coupling layer* between detection and planning (more access via `alldet`, two-way flow via `bidir`) and neither improved planning. Together with the prior `numdemapx2` null, this is now a fairly strong signal that the binding constraint is not the planning ↔ detection interface — it is upstream, in what detection features encode. The `plan_relevance` direction (auxiliary planning-relevance supervision shaping detection features through their own loss) is a more promising lever, since it changes the features themselves rather than how planning accesses them.

## Future Work

- The `egoquery` experiment (Experiment 4) was conditioned on `bidir` providing directional validation. With `bidir` regressing, the cleanest next motivation for ego-query integration would now have to come from `plan_relevance` showing that planning-shaped detection features help — pursue `egoquery` only after that signal exists.
- Stage-1 planning supervision (`ptplan1`, Experiment 1) is still worth running: it shapes the *backbone* with planning gradients rather than coupling at the decoder. It is mechanistically distinct from `alldet`/`bidir` and the null results here do not predict its outcome.
- If a second run of `bidir` reproduces the L2 regression, ablate the placement: try `rev_gnn` *only* in the last decoder stage rather than all 3, and try detaching the planning state in the K/V (so the reverse attention conditions on but does not gradient-couple back to planning). Either may recover baseline.
- Combine `alldet` with `detrel` from `plan_relevance`: if relevance supervision improves the *features*, then giving planning access to all 900 of those improved features may finally produce additive gains. The current null suggests testing access and feature quality together rather than in isolation.
