# Plan Scoring Beyond Hard Heuristic Feasibility

2026-04-26

## TODO

- [x] Run cheap diagnostic: disable rescore on current best, eval — bounds inference-time headroom (Killarney 3296430, `ptaux2d_ppdeformmm_planifls_norescore`)
- [ ] Implement soft collision loss as training-loss term — **highest expected impact deployment** (in progress: collision-only first cut on `planpredtrajdeformmm` baseline)
- [ ] If first cut lands: extend cost to drivable / comfort / progress and λ sweep
- [ ] If diagnostic shows headroom: extend rescore from hard binary → soft multi-cost selection
- [ ] If both deployments help individually: train + select with the same cost (joint A+B) for compounding
- [ ] Defer learned scorer / value head and iterative refinement until first analytic results land

## Abstract

Tracks experiments that move the trajectory scoring step beyond the existing hard-binary feasibility filter (`HierarchicalPlanningDecoder.rescore()`). The primitive applies `-9999` to colliding modes and is `use_rescore=True` by default in every config — the current best (L2=0.522 / CR=0.047%) is already with this filter on. The research direction is *what comes after a hard-binary heuristic*: soft analytic multi-cost (collision + drivable + comfort + progress), learned scoring, multi-stage scoring, and — most importantly — scoring as a training-time supervision signal. The training-time deployment is the higher-expected-impact half of the contribution; inference-time is bounded by what the existing primitive is leaving on the table.

## Intro

The current planning decoder produces 6 (or 18, cmd-conditional) trajectory candidates per sample. At inference, `HierarchicalPlanningDecoder.rescore()` builds ego BEV boxes along plan waypoints, builds agent boxes along the argmax motion mode per agent, runs a hard corner-in-box collision check, and applies a `-9999` score offset to colliding modes. Non-colliding modes are not differentiated. If all modes collide, the offset is suppressed (degenerate fallback). The selected mode is the argmax of the rescored logits.

This is a *primitive* scorer: hard-binary, single-stage, non-learned, top-1 motion mode, collision-only. The contribution direction is to extend it along several axes:

- **Soft cost** instead of binary — distance-based, differentiable
- **Multi-objective** — collision + drivable + comfort + progress, not just collision
- **Multi-mode aggregation** — weighted by motion confidence rather than top-1
- **Training deployment** — the same cost as a gradient signal back through the planner, not just an inference-time selector
- **Learned scorer** — trained to predict trajectory quality from rollout outcomes (later)

The "inference vs training" distinction is artificial — both are uses of the same scoring machinery, with a gradient at training and an argmin at inference. The interesting empirical question is whether each deployment helps independently and whether they compound.

Questions:
- How much headroom does the existing hard-binary primitive leave on the table at inference time? (cheap diagnostic)
- Does a soft cost as a training-time loss directly shape trajectory generation toward feasibility / drivable / comfort? (mechanistically distinct from the failed `planaux_*` classification heads)
- Do training-time and inference-time deployments compound when applied jointly?

## Method

All experiments use `ptaux2d_ppdeformmm_planifls_planinstfeat_laststage` (current best, L2=0.522 / CR=0.047%) as the reference baseline. Same-server (Killarney 3282987) baseline used for direct comparison.

### Cost function

Single multi-objective cost over a trajectory `τ ∈ ℝ^{T×3}` (XY + heading) given scene context (detected agents, drivable mask, goal direction):

```
C(τ; scene) = w_col · C_col(τ, agents) + w_drv · C_drv(τ, drivable_mask)
            + w_cmf · C_cmf(τ) + w_pgr · C_pgr(τ, goal)
```

Each term differentiable:
- **Collision** `C_col`: `Σ_t Σ_a exp(-min_dist(τ_t, agent_t,a)² / σ²)` — soft, distance-based; aggregated over motion modes weighted by motion confidence (not top-1)
- **Drivable** `C_drv`: `grid_sample` of `(1 - drivable_mask)` along τ — penalty for off-road waypoints (uses rasterization machinery from `planaux_da`)
- **Comfort** `C_cmf`: curvature + jerk on τ
- **Progress** `C_pgr`: dot product of ego displacement with `gt_ego_fut_cmd` direction

### Live experiments (ordered by expected impact / cost)

**Experiment 0 — `rescore_disable_diag`: diagnostic.** *(Cheapest in roadmap: ~1 hour, no retraining.)* Set `use_rescore=False` in the current best config; re-evaluate the existing checkpoint. Measures how much the hard-binary feasibility filter is contributing to the current 0.522 / 0.047%. CR delta bounds the headroom available for richer inference-time scoring.

Outcome interpretations:
- CR degrades substantially (e.g. 0.047% → >0.08%): existing primitive is doing real work; richer scoring has plausible additional headroom → proceed to Exp 2 (soft cost selection)
- CR barely changes (e.g. 0.047% → 0.055%): existing primitive rarely fires; soft / multi-cost extensions unlikely to help much at inference → focus on Exp 1 (training-time deployment) instead

Config: existing best with `use_rescore=False`. No retraining.

**Experiment 1 — `softcost_train` (first cut: collision-only): cost as training-time loss.** *(Highest expected impact: 0.005-0.02 L2, larger CR effect.)* Add a soft collision cost as a loss term to the standard plan loss:

```
L_plan_total = L_plan_imitation (existing L1+CE) + λ_col · C_col(plan_reg, agents)
C_col = mean over (modes, t, agents) of exp(-||ego_pos_t - agent_pos_t||² / σ²)
```

First cut starts from the *plain* `planpredtrajdeformmm` baseline (not `_planifls_planinstfeat_laststage`) to keep the change isolated. Collision is the only term that directly targets `obj_box_col`; comfort doesn't show up in any tracked metric and progress is already implicit in the imitation target, so both are deferred. Drivable is deferred as an additive component if collision lands.

Agent footprints come from GT (`gt_bboxes_3d` t=0 + cumsum(`gt_agent_fut_trajs`)), reusing the machinery from `_loss_planning_conflict`. Predicted-detection variant (`det_output`) is a follow-up if collision lands; using GT first cleanly isolates whether the cost shape itself helps before introducing detector-noise gradient. Cost is applied to all 18 plan modes per decoder stage. σ=2.0m matches the existing `conflict_threshold`. λ_col=0.2 matches existing aux-loss weights.

This is mechanistically distinct from `planaux_da` and `planaux_conf` (which predicted the property via classification heads). Here the cost is computed directly from the trajectory geometry and backpropped, rather than predicted as an auxiliary task.

Config: `sparsedrive_r50_stage2_4gpu_bs24_planpredtrajdeformmm_softcostcol.py` — base + `plan_softcost_collision_enable=True`, `plan_softcost_collision_weight=0.2`, `plan_softcost_collision_sigma=2.0`.

Sub-ablations (only if first cut lands):
- Add drivable / comfort / progress terms (separately, then jointly)
- Predicted-agent variant (`det_output` instead of GT) to match inference distribution
- Critic weight sweep (`λ_col` ∈ {0.05, 0.2, 1.0})

**Experiment 2 — `softcost_infer`: replace hard binary rescore with soft multi-cost.** *(Conditional on Exp 0 showing headroom.)* Replace `HierarchicalPlanningDecoder.rescore()` hard collision check with the soft multi-objective cost C; selection becomes argmin over continuous scores rather than mask-out colliding modes. Aggregate over all motion modes weighted by motion confidence rather than top-1. Extend to all 18 plan modes (not cmd-filtered 6) before final cmd selection.

Implementation bounded to `motion/decoder.py`. No retraining required (uses the existing best checkpoint).

Sub-ablations:
- Hard binary (existing) → soft single-cost (collision only) → soft multi-cost — isolates the marginal value of cost design
- Top-1 motion mode → confidence-weighted motion modes — isolates the marginal value of mode aggregation

Config: same checkpoint as best, with new decoder cost configuration.

**Experiment 3 — `softcost_joint`: A + B compounding.** *(Conditional on both Exp 1 and Exp 2 landing.)* Train with the soft cost as a loss (Exp 1) AND select with the soft cost at inference (Exp 2). Tests whether the two deployments compound additively.

Config: `sparsedrive_r50_stage2_4gpu_bs24_ptaux2d_ppdeformmm_planifls_planinstfeat_laststage_softcost.py` with the new decoder selector enabled.

### Future variants (deferred)

- **Learned scorer / value head**: replace analytic cost with a trained value head; train on rollout outcomes. Requires more infrastructure; only justified if analytic cost shows real signal first.
- **Hybrid analytic prior + learned residual**: analytic cost provides strong prior; learned head corrects systematic errors.
- **Two-stage scoring**: existing hard binary as feasibility filter, soft cost ranking among feasible candidates.
- **Iterative refinement at inference**: gradient descent on the scorer's output to refine the top candidate.

### Implementation summary

Already done:
- `HierarchicalPlanningDecoder.rescore()` — hard binary feasibility filter (existing; `use_rescore=True` default)
- Drivable mask rasterization in `vectorize.py` — built for `planaux_da`, reusable here
- Multi-modal motion outputs from `MotionPlanningHead` — already produced

To implement (Exp 1 — training):
- New module / function in `motion_planning_head.py` for soft multi-objective cost C
- New loss term `_loss_planning_softcost` wired into `forward` / `loss`
- Cost-component weights exposed in config

To implement (Exp 2 — inference):
- Extend `HierarchicalPlanningDecoder.rescore()` with soft cost variant
- Aggregate over all motion modes (not top-1)
- Score combination weights exposed in config

To implement (Exp 3 — joint):
- Same as Exp 1 + Exp 2; just enable both flags simultaneously

## Results

| Server | Config | Job ID | Status |
| --- | --- | --- | --- |
| Killarney | `ptaux2d_ppdeformmm_planifls_planinstfeat_laststage` (baseline, with `use_rescore=True`) | TBD | RECORDED — best |
| TBD | `..._planinstfeat_laststage` re-eval with `use_rescore=False` | — | PLANNED — Exp 0 diagnostic |
| TBD | `..._planinstfeat_laststage_softcost` (training loss) | — | PLANNED — Exp 1 |
| TBD | `..._planinstfeat_laststage` with soft-cost rescore (no retrain) | — | CONDITIONAL — Exp 2 |
| TBD | `..._planinstfeat_laststage_softcost` with soft-cost rescore (joint) | — | CONDITIONAL — Exp 3 |

| Config | L2 | obj_box_col | car_ade | ped_ade | car_epa | ped_epa | NDS | mAP | Notes |
|--------|-----|-------------|---------|---------|---------|---------|-----|-----|-------|
| `ptaux2d_ppdeformmm_planifls_planinstfeat_laststage` (current best, with rescore) | 0.522 | 0.047% | 0.633 | — | — | — | 0.524 | 0.415 | reference |
| Same checkpoint, `use_rescore=False` | — | — | — | — | — | — | — | — | Exp 0 diagnostic |
| `..._softcost` (training loss) | — | — | — | — | — | — | — | — | Exp 1 |
| Best checkpoint + soft-cost rescore | — | — | — | — | — | — | — | — | Exp 2 |
| `..._softcost` + soft-cost rescore | — | — | — | — | — | — | — | — | Exp 3 |

## Discussion

Pending experiments. Key interpretation questions to answer:

- **Exp 0 diagnostic**: how much of the current 0.047% CR is from the existing hard-binary primitive vs. from the model's mode classification?
- **Exp 1 (training loss)**: does the soft cost shape trajectory generation in a way `planaux_*` classification heads couldn't?
- **Exp 2 (inference scoring)**: does cost-based selection beat hard-binary feasibility filtering, conditional on the existing primitive leaving headroom?
- **Exp 3 (joint A+B)**: do the two deployments compound, or do they overlap (one captures most of the available signal)?

## Plan and Future Work

Ordered by expected impact / cost ratio:

1. **Run Exp 0 (rescore disable diagnostic) immediately.** ~1 hour, no retraining. Single config flag flip + re-evaluate the best checkpoint. Result determines whether Exp 2 is worth pursuing.
2. **Implement Exp 1 (soft cost as training loss) in parallel.** ~1-2 days for implementation, ~12h for training. This is the higher-expected-impact deployment and doesn't depend on Exp 0's outcome.
3. **Run Exp 2 (soft cost as inference selector) only if Exp 0 shows headroom.** ~1 day implementation, no retraining. Uses the existing best checkpoint.
4. **Run Exp 3 (joint A+B) only if both Exp 1 and Exp 2 land independently.** Tests compounding.

Deferred until at least one of Exp 1 / Exp 2 shows signal:
- Learned scorer / value head — replace analytic cost with a trained value function
- Hybrid analytic + learned residual
- Iterative refinement at inference
- Two-stage scoring (hard feasibility filter + soft ranking among feasible)

If Exp 1 lands and Exp 2 doesn't: the contribution is "training-time scoring as supervision" — a clean direct-loss alternative to the failed `planaux_*` aux-head approach.

If Exp 2 lands and Exp 1 doesn't: the contribution is narrower — a richer inference-time selector that improves on the existing primitive without changing how the planner is trained.

If both land and compound: the full dual-deployment story (one cost function, two consumers, additive) is the contribution.

If neither lands: the binding constraint is not at the scoring layer; redirect to the unified architectures direction (`plan_unified`) and the closed-loop alignment direction (`plan_closedloop` to be created).
