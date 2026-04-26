# Unified Planning Architectures

2026-04-25

## TODO

- [x] Implement `alldet` (use_alldet_kv flag in MotionPlanningHead)
- [x] Implement `bidir` (rev_gnn op)
- [x] Re-implement perception detach flag as `detach_perception` ctor arg in `MotionPlanningHead` (stop-grad on det+map output dicts at top of `forward`)
- [x] Verified `gt_ego_fut_trajs/masks/cmd` already collected by `stage1_8gpu_noflash`
- [ ] **Arm A** (Killarney): stage-2 from existing `joint_detach` stage-1 ckpt — **highest priority, run first**
- [ ] **Arm B**: new joint stage-1 on Apollo (modernized recipe) + stage-2 follow-up on Killarney
- [ ] Implement map promotion (map tokens as queries in planning decoder self-attention) — cheap MVE
- [ ] Egoquery — only if Arm A/B or map promotion shows directional signal first

## Abstract

Tracks experiments on architectural couplings between planning and the rest of the perception stack. Existing baseline `MotionPlanningHead` already shares self-attention across (ego ∪ agents) via `temp_gnn`/`gnn`/`cross_gnn`, so the design space remaining is: extend that sharing back into the detection stage (egoquery), include map tokens as queries (map promotion), or align the backbone with planning gradients from stage 1 (stage1_planifls). Two prior experiments — `alldet` (planning attends to all 900 detection tokens) and `bidir` (reverse cross-attention into agent tokens) — completed and were null/regressive, ruling out the coupling-layer hypothesis. The remaining live experiments are stage-1 planning supervision (highest expected impact, mechanism-distinct from anything tried) and map promotion (cheap MVE).

## Intro

The current SparseDrive architecture computes detection tokens fully before planning sees them. `MotionPlanningHead` reads `det_output` via confidence-based top-k selection and concatenates the ego token into the resulting set, which then goes through shared self-attention via `temp_gnn`, `gnn`, and `cross_gnn`. Within the planning head, ego↔agent coupling at the feature level is therefore already present. What is *not* present:

- Planning gradients reaching the perception backbone before stage 2 — the backbone is shaped by detection (and optionally aux 2D) loss only at stage 1
- Sharing of the planning-head self-attention back into the detection stage — ego queries don't exist inside `Detection3DHead`
- Map tokens as queries in the planning decoder — map is currently K/V-only via cross-attention

Two prior experiments tested coupling at the existing planning ↔ detection interface and both failed:

- **`alldet` (Killarney 3284640)**: planning queries cross-attend to all 900 detection tokens instead of top-50. Null result on planning (L2 +0.003, within noise).
- **`bidir` (Killarney 3284641)**: detection token features attend back to planning state via a reverse cross-attention in the planning decoder. L2 regressed by 0.039 — likely because writing planning information into agent token slots destabilized the detection-side regression those slots are still supervised against.

Combined with the earlier `numdemapx2` null and `detrel` regression in `plan_relevance`, the binding constraint is not the planning ↔ detection interface itself. The remaining mechanism-distinct interventions are upstream (stage-1 backbone alignment) or laterally extending coupling (egoquery into detection, map promotion in planning).

Questions:
- Does aligning the backbone with planning gradients at stage 1 improve stage-2 planning beyond `ptaux2d`? (highest expected impact)
- Does promoting map tokens to queries in the planning decoder help, given that ego↔agent sharing is already there?
- If either provides directional validation, does extending shared self-attention into the detection stage (egoquery) compound?

## Method

All stage-2 experiments use `ptaux2d_ppdeformmm_planifls_planinstfeat_laststage` (current best, L2=0.522 / CR=0.047%) as the reference baseline, with same-server (Killarney 3282987) baseline numbers used for direct comparison.

### Completed (negative controls)

**`alldet` — planning attends to all 900 detection tokens.** `use_alldet_kv` flag in `MotionPlanningHead` swaps the `gnn` op's K/V from `topk(det_confidence, num_det=50, ...)` to `instance_feature[:, :num_anchor + 1]`. Null result; rules out token-quantity / selection bottleneck.

**`bidir` — reverse cross-attention from agent tokens to planning state.** `rev_gnn` op inserts a `MultiheadFlashAttention` after each `gnn`, with agent features as queries and `(plan_mode_query + ego_feature + ego_anchor_embed)` as K/V. Regressed L2 by 0.039; rules out shared-token bidirectional coupling.

### Live experiments (ordered by expected impact)

**Experiment 1 — `stage1_planifls`: planning supervision at stage 1.** *(Highest expected impact: 0.01-0.03 L2.)* Enable `with_motion_plan=True` in the stage-1 config so the backbone receives planning gradients from the start, not just detection. Mechanism-distinct from `alldet`/`bidir`/`detrel`/`planaux_*` — those all targeted decoder-stage couplings or output supervision. Backbone-level alignment has historically moved things on this codebase: `ptaux2d` was a major contributor to going from L2=0.636 to 0.522. This is the planning analog.

*Prior attempt — `joint_detach` (Apollo, finished 2026-03-23, never followed up).* A stage-1 run with motion+plan and `detach_det=True` was already trained for 100 epochs from `stage1_8gpu_noflash`. Despite the flag name, `detach_det` actually detached both det AND map outputs before the planning head, so planning loss only reached the backbone via the planning head's own deformable cross-attention to image features (not via the det/map heads' instance features). Final stage-1 eval: L2=0.6825, NDS=0.5216, mAP=0.4038, mAP_normal=0.5704. Perception regressed ~1pt NDS / ~0.7pt map_normal vs the matched `noflash` baseline (NDS=0.5307, mAP_normal=0.5774) — the cost of the extra objective on shared backbone capacity. The checkpoint exists at `/home/spapais/ForeSight/work_dirs/sparsedrive_r50_stage1_8gpu_noflash_joint_detach/iter_117200.pth` but was never used as a stage-2 init, so the actual hypothesis (stage-1 planning supervision → better stage-2 L2) was never tested.

The `detach_det` flag has since been removed from the codebase and needs re-implementation (cleaner name: `detach_perception`).

*Two-arm plan.*

**Arm A — stage-2 from the existing joint_detach checkpoint.** Zero stage-1 cost. Stage-2 clones `sparsedrive_r50_stage2_4gpu_bs24_planpredtrajdeformmm.py` (without `planinstfeat_laststage`, since the joint_detach stage-1 head was vanilla — laststage params would init fresh) and changes `load_from` to the staged ckpt. **Comparison anchor:** plain `planpredtrajdeformmm` baseline L2=0.5420 (avg DGX/Narval). Win = joint stage-1 helps planning even on the no-aux2d, no-modern-head base.

**Arm B — new joint stage-1 + matching stage-2.** Modernized recipe vs joint_detach, on the same `noflash` base (intentional: minimize confounders, also faster — 32h vs 42h on aux2d). Three deltas from joint_detach:
1. **LR = 4e-4** (not joint_detach's 1.5e-4 outlier; matches every other current `stage1_8gpu*` config).
2. **Modern planning head**: `planning_cumulative_refinement=True`, `motion_cumulative_refinement=True`, `planning_deformable=True`, `motion_deformable_multimode=True` — matches the `planpredtrajdeformmm` recipe so stage-2 warm-init is dense.
3. **`detach_perception` flag** (re-implemented), defaults to detaching both det and map outputs (mirroring the original joint_detach semantic).

Same as joint_detach: `noflash` base (no aux2d), 100ep × 8 GPU, backbone lr_mult=0.5, det+map+motion_plan task config.

**Why no aux2d?** The combined report (2026-04-21) shows aux2d is the planning-best stage-1 base, but adds a strong confounder for this experiment (aux2d shapes the backbone too) and 10 extra hours per stage-1 run. Single-delta comparison is cleaner. If joint stage-1 lands on plain `noflash`, follow-up tests joint-on-aux2d for compounding.

Configs: `sparsedrive_r50_stage1_8gpu_noflash_joint.py` (Arm B stage 1, Apollo), `sparsedrive_r50_stage2_4gpu_bs24_ptjointdetach_planpredtrajdeformmm.py` (Arm A stage 2, Killarney; expects `ckpt/sparsedrive_stage1_joint_detach.pth` on the run host), `sparsedrive_r50_stage2_4gpu_bs24_ptjoint_planpredtrajdeformmm.py` (Arm B stage 2, Killarney; expects `ckpt/sparsedrive_stage1_joint.pth`).

**Experiment 2 — `map_promotion`: map tokens as queries in planning decoder.** *(Cheap MVE: 0-0.005 L2 expected.)* The current planning decoder uses ego ∪ agents as queries with map as K/V-only via cross-attention. Promote map tokens to queries by including them in the self-attention population: queries = ego ∪ agents ∪ map (local copies, not propagated back to map head — avoids the `bidir`-style supervision conflict). The map tokens become plan-aware (and ego-aware, agent-aware) within the planning forward pass.

This is the cheapest test of "extend the existing shared self-attention to additional populations." Bounded to `motion_planning_head.py`. Implementation: append top-K map K/V to the query stream before `gnn`, separate the map slice after `refine` so map outputs don't feed into trajectory regression.

Config: `sparsedrive_r50_stage2_4gpu_bs24_ptaux2d_ppdeformmm_planifls_planinstfeat_laststage_mappromote.py`.

**Experiment 3 — `egoquery`: ego queries inside detection transformer.** *(Gated on Exp 1 or Exp 2 providing directional signal: 0-0.015 L2 expected.)* Add ego planning queries (one per mode, `ego_fut_mode=6`) to the instance bank inside `Detection3DHead`. Ego queries participate in detection self-attention, deformable attention to image features, and (separately-supervised) trajectory regression. Agent queries continue under detection loss; ego queries under planning loss; no shared token serves two losses. `MotionPlanningHead` retained as a lightweight refinement stage on top of ego query outputs.

Requires changes to `instance_bank.py`, `detection3d_head.py`, `sparsedrive_head.py`. Largest implementation cost in the batch; only justified if Exp 1 or Exp 2 lands.

Config: `sparsedrive_r50_stage2_4gpu_bs24_ptaux2d_egoquery_planifls.py`.

### Implementation summary

Already done:
- `motion_planning_head.py`: `use_alldet_kv` flag, `rev_gnn` op handler, `rev_graph_model` cfg slot

To implement (Exp 1):
- `motion_planning_head.py`: re-add `detach_perception` flag (stop-grad on det+map outputs before planning head consumes them; mirrors original joint_detach `detach_det=True` semantic)
- `sparsedrive_r50_stage1_8gpu_noflash_joint.py`: clone `noflash`, set `with_motion_plan=True` + `detach_perception=True`, add modern planning head (deformmm flags), keep lr=4e-4
- Verify `gt_ego_fut_trajs/masks/cmd` in stage-1 `Collect` keys (already present in aux2d variant — confirm for plain `noflash`)
- Stage-2 configs: clone `planpredtrajdeformmm.py`, swap `load_from` to the joint stage-1 ckpts (Arm A: existing joint_detach; Arm B: new joint)

To implement (Exp 2):
- `motion_planning_head.py`: add `with_map_promotion` flag; append map K/V to instance feature stream as queries; slice off map outputs after refine

To implement (Exp 3, if gated):
- `instance_bank.py`, `detection3d_head.py`, `sparsedrive_head.py`: add ego query slots and trajectory output head

## Results

| Server | Config | Job ID | Status |
| --- | --- | --- | --- |
| Killarney | `ptaux2d_ppdeformmm_planifls` (baseline) | 3282987 | COMPLETED |
| Killarney | `ptaux2d_ppdeformmm_planifls_alldet` | 3284640 | COMPLETED — null |
| Killarney | `ptaux2d_ppdeformmm_planifls_bidir` | 3284641 | COMPLETED — regression |
| Apollo | `stage1_8gpu_noflash_joint_detach` (prior, vanilla head, lr=1.5e-4) | — | COMPLETED 2026-03-23 — never used as stage-2 init |
| Killarney | `ptjointdetach_planpredtrajdeformmm` | 3301176 | SUBMITTED — Arm A stage 2 (uses prior joint_detach ckpt) |
| Apollo | `stage1_8gpu_noflash_joint` (modern recipe, bs=48, lr=3e-4) | tmux:armb_s1 | RUNNING — Arm B stage 1 (peak ~21GB/GPU; ETA ~38h) |
| Killarney | `ptjoint_planpredtrajdeformmm` | — | PLANNED — Arm B stage 2 (after Arm B stage 1) |
| TBD | `ptaux2d_ppdeformmm_planifls_planinstfeat_laststage_mappromote` | — | PLANNED — Exp 2 |
| TBD | `ptaux2d_egoquery_planifls` | — | GATED on Exp 1/2 |

| Config | L2 | obj_box_col | car_ade | ped_ade | car_epa | ped_epa | NDS | mAP |
|--------|-----|-------------|---------|---------|---------|---------|-----|-----|
| `ptaux2d_ppdeformmm_planifls` (DGX 3614, baseline) | 0.5057 | 0.043% | 0.6141 | 0.7177 | 0.5055 | 0.4365 | 0.5436 | 0.4383 |
| `ptaux2d_ppdeformmm_planifls` (Killarney 3282987, server-matched baseline) | 0.4987 | 0.063% | 0.6228 | 0.7201 | 0.5073 | 0.4302 | 0.5447 | 0.4403 |
| `ptaux2d_ppdeformmm_planifls_alldet` (Killarney 3284640) | 0.5018 | 0.084% | 0.6031 | 0.7119 | 0.5090 | 0.4270 | 0.5477 | 0.4404 |
| `ptaux2d_ppdeformmm_planifls_bidir` (Killarney 3284641) | 0.5378 | 0.057% | 0.6069 | 0.7227 | 0.5134 | 0.4280 | 0.5442 | 0.4386 |
| `stage1_8gpu_noflash_joint_detach` (stage-1 eval only) | 0.6825 | 0.142% | — | — | 0.4950 | 0.4019 | 0.5216 | 0.4038 |
| `ptjointdetach_planpredtrajdeformmm` (Arm A) | — | — | — | — | — | — | — | — |
| `ptjoint_planpredtrajdeformmm` (Arm B) | — | — | — | — | — | — | — | — |
| `ptaux2d_ppdeformmm_planifls_planinstfeat_laststage_mappromote` | — | — | — | — | — | — | — | — |

## Discussion

Both completed experiments compared against the **server-matched** Killarney baseline (3282987) to control for the cross-server L2 gap (0.5057 → 0.4987, comparable to per-experiment effect size).

**`alldet`: no planning improvement.** L2 0.4987 → 0.5018 (within noise); `obj_box_col` worsens 0.063% → 0.084%. Removing the top-50 detection bottleneck and giving planning queries direct cross-attention to all 900 detection tokens does not improve planning. Combined with the earlier `numdemapx2` null (50 → 100), the bottleneck is not how many detection tokens planning sees — it is what those tokens encode.

**`bidir`: planning regresses.** L2 0.4987 → 0.5378 (clearly outside noise); `obj_box_col` improves marginally; detection metrics flat. Hypothesis: writing planning information into agent token slots destabilizes the detection-side refine targets the same tokens are still supervised against. The agent representation has to serve both detection regression/classification and a planning-conditioned write, and the joint objective drifts.

**Combined reading.** Both interventions targeted the *coupling layer* between detection and planning at decoder stage 2 (more access via `alldet`, two-way flow via `bidir`) and neither improved planning. Together with the prior `numdemapx2` null and `detrel` regression in `plan_relevance`, this is a strong signal that the binding constraint is not the planning ↔ detection interface or perception-side reshape with planning-uninformed supervision. The remaining mechanism-distinct directions are *upstream* (stage-1 backbone alignment with planning gradients) or *laterally extending the existing shared self-attention* to additional populations (map promotion, egoquery).

## Plan and Future Work

Ordered by expected impact / cost ratio:

1. **Arm A first (free signal).** Stage-2 from the existing `joint_detach` ckpt costs only the stage-2 training (~12h). If positive vs `planpredtrajdeformmm` L2=0.5420, validates the mechanism cheaply and motivates Arm B. If null/regressive, it still constrains the hypothesis at low cost.
2. **Arm B (modernized stage 1) after Arm A.** ~32h stage-1 + ~12h stage-2. The modern recipe (lr=4e-4, deformmm planning head, re-implemented detach) tests whether the joint-stage-1 idea works with the current architecture, not the obsolete one.
3. **Exp 2 (`map_promotion`) opportunistically.** Bounded to `motion_planning_head.py`, ~1-2 days. Cheap diagnostic of whether the existing shared self-attention story extends to additional populations.
4. **Egoquery (Exp 3) gated on directional signal from Arms A/B or Exp 2.** Largest implementation cost; defer until at least one cheaper experiment shows traction.

If Arm A or Arm B lands strongly and Exp 2 is null, the architectural story is "backbone alignment matters; decoder-side coupling does not." If both land, egoquery becomes well-motivated. If both fail, the entire "coupling-via-architecture" hypothesis is exhausted on this baseline and the contribution shifts to a comprehensive negative paper anchored by the four configurations.

If Arm A lands but Arm B does not (or vice versa), the recipe deltas (lr, planning head architecture) become the next thing to ablate.

Open follow-ups (lower priority):
- If `bidir` regression should be further ablated: try `rev_gnn` only at the last decoder stage; try detaching the planning state in K/V (no gradient back to planning). Either may recover baseline.
- Combine `alldet` with `detrel`-style supervision: if relevance reshapes the features, more access to better features may yet be additive. Currently low priority because both component experiments were null/regressive.
