# Plan Scoring Beyond Hard Heuristic Feasibility

2026-04-26

## TODO

- [x] Run cheap diagnostic: disable rescore on current best, eval — bounds inference-time headroom (Killarney 3302063, `ptaux2d_ppdeformmm_planifls_norescore`: L2=0.5008, CR=0.106% vs 0.4988/0.063% with rescore → +0.043pp CR headroom)
- [x] Implement soft collision loss as training-loss term — first cut NULL (Killarney 3296906, `planpredtrajdeformmm_softcostcol`: L2=0.5515, CR=0.050% vs 0.542/0.047% plain baseline avg)
- [x] Exp 2 (inference selector, 3-point `w_col` sweep at σ=2m on Killarney ckpt synced to narval) — NULL: catastrophic L2 regression at every `w_col` (0.6667 / 0.7205 / 0.7492 vs 0.4988 baseline), CR never reaches hard rescore's 0.063%. Saturating-Gaussian-on-SDF kernel is "always on" and dominates plan_cls (Narval 59920744/45/46).
- [x] Exp 1 retry (`softcostcol_v2`, training, geometry-aware SDF, λ=0.05, τ=0.5m) — NULL: L2=0.5481, CR=0.059% (Narval 59920383). Geometry + normalization fixes did not lift result out of v1's regime; both v1 and v2 mildly regress vs plain `planpredtrajdeformmm` baseline.
- [ ] Joint A+B (Exp 3) — DROPPED; both legs failed independently.
- [ ] **Next: learned scorer with stationary label** — a per-(mode) BCE-supervised collision-feasibility head where the label is computed against a fixed reference (plan-anchor template *or* GT ego trajectory), not against `cumsum(plan_reg)`. Decouples supervision from the planner's own current iteration, recovers sharpness via the binary label. Existing `planaux_conf` infrastructure can be reused for the head wiring; the change is in label construction (and using the head's logits at inference, not just as aux loss). See "Plan and Future Work" for the design sketch.
- [ ] Defer iterative refinement at inference / hybrid analytic-prior + learned-residual until the learned scorer lands.

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
| Killarney | `ptaux2d_ppdeformmm_planifls_norescore` re-eval (Exp 0, attempt 1) | 3296430 | FAILED — config missing at job start (manual scp lost across git operations) |
| Killarney | `ptaux2d_ppdeformmm_planifls_norescore` re-eval (Exp 0, attempt 2, 12h limit) | 3301041 | CANCELLED — superseded by 3302063 |
| Killarney | `ptaux2d_ppdeformmm_planifls_norescore` re-eval (Exp 0, attempt 3, 3h limit) | 3302063 | COMPLETED — Exp 0 |
| Killarney | `planpredtrajdeformmm_softcostcol` (training loss, collision-only) | 3296906 | COMPLETED — Exp 1 |
| Narval | `planpredtrajdeformmm_softcostcol_v2` (Exp 1 retry, geometry-aware SDF) | 59920383 | COMPLETED — Exp 1 retry |
| Narval | `ptaux2d_ppdeformmm_planifls_softrescore_w3_s2` (Exp 2a, attempt 1) | 59920405 | FAILED — ckpt path unreachable in container (used `/home/spapais/scratch/...` which is not bind-mounted; only `/scratch/spapais/ForeSight/work_dirs` is mapped to `/workspace/ForeSight/work_dirs`) |
| Narval | `ptaux2d_ppdeformmm_planifls_softrescore_w10_s2` (Exp 2b, attempt 1) | 59920406 | FAILED — same ckpt path issue |
| Narval | `ptaux2d_ppdeformmm_planifls_softrescore_w30_s2` (Exp 2c, attempt 1) | 59920407 | CANCELLED — caught early, would have failed for same reason |
| Narval | `ptaux2d_ppdeformmm_planifls_softrescore_w3_s2` (Exp 2a, attempt 2) | 59920744 | COMPLETED — Exp 2a |
| Narval | `ptaux2d_ppdeformmm_planifls_softrescore_w10_s2` (Exp 2b, attempt 2) | 59920745 | COMPLETED — Exp 2b |
| Narval | `ptaux2d_ppdeformmm_planifls_softrescore_w30_s2` (Exp 2c, attempt 2) | 59920746 | COMPLETED — Exp 2c |
| TBD | `..._planinstfeat_laststage_softcost` with soft-cost rescore (joint) | — | DEPRIORITIZED — Exp 1 v1 didn't land |

| Config | L2 | obj_box_col | car_ade | ped_ade | car_epa | ped_epa | NDS | mAP | Notes |
|--------|-----|-------------|---------|---------|---------|---------|-----|-----|-------|
| `ptaux2d_ppdeformmm_planifls_planinstfeat_laststage` (cross-server best, with rescore) | 0.522 | 0.047% | 0.633 | — | — | — | 0.524 | 0.415 | reference (DGX/Narval avg from `2026_04_02_plan_refine.md`) |
| `ptaux2d_ppdeformmm_planifls` Killarney baseline (with rescore) | 0.4988 | 0.063% | — | — | — | — | — | — | same-server reference for Exp 0 (recorded in `0000_00_00_research_findings.md`) |
| `planpredtrajdeformmm` (cross-server avg, plain baseline for Exp 1) | 0.542 | 0.047% | 0.626 | — | — | — | 0.526 | 0.414 | from `2026_04_02_plan_refine.md`; no Killarney run |
| `ptaux2d_ppdeformmm_planifls_norescore` (same checkpoint, `use_rescore=False`) | **0.5008** | **0.106%** | 0.6231 | 0.7196 | 0.5068 | 0.4297 | 0.5453 | 0.4399 | Exp 0 — Killarney 3302063; mAP_normal=0.5633 |
| `planpredtrajdeformmm_softcostcol` (training loss, collision-only) | **0.5515** | **0.050%** | 0.6271 | 0.7344 | 0.4935 | 0.4111 | 0.5257 | 0.4142 | Exp 1 v1 — Killarney 3296906; mAP_normal=0.5537 |
| `planpredtrajdeformmm_softcostcol_v2` (training, SDF-corners, λ=0.05, τ=0.5m) | **0.5481** | **0.059%** | 0.6359 | 0.7326 | 0.4880 | 0.4133 | 0.5266 | 0.4176 | Exp 1 retry — Narval 59920383; mAP_normal=0.5566 |
| `ptaux2d_ppdeformmm_planifls_softrescore_w3_s2` (Killarney ckpt, σ=2m) | **0.6667** | **0.067%** | — | — | — | — | — | — | Exp 2a — Narval 59920744; planning-only eval |
| `ptaux2d_ppdeformmm_planifls_softrescore_w10_s2` (Killarney ckpt, σ=2m) | **0.7205** | **0.072%** | — | — | — | — | — | — | Exp 2b — Narval 59920745 |
| `ptaux2d_ppdeformmm_planifls_softrescore_w30_s2` (Killarney ckpt, σ=2m) | **0.7492** | **0.079%** | — | — | — | — | — | — | Exp 2c — Narval 59920746 |
| `..._softcost` + soft-cost rescore | — | — | — | — | — | — | — | — | Exp 3 (deprioritized) |

## Discussion

**Exp 0 (`norescore`, Killarney 3302063): rescore is doing real work; Exp 2 cleared.** Re-evaluating the Killarney `ptaux2d_ppdeformmm_planifls` checkpoint with `use_rescore=False` gave `L2=0.5008 / obj_box_col=0.106%`. Versus the same-server with-rescore baseline (`L2=0.4988 / obj_box_col=0.063%`), L2 is essentially unchanged (+0.002, noise) but CR jumps `+0.043pp` (~+68% relative). Detection metrics (`NDS=0.5453`, `mAP=0.4399`) match what the with-rescore checkpoint would produce — rescore touches the planning decoder selector only. By the plan's outcome rule (CR delta > 0.08% absolute reading: `0.063% → 0.106%` is well past that), the existing hard-binary primitive is suppressing real collisions, so a richer inference-time selector (Exp 2) has plausible additional headroom.

**Exp 1 (`softcostcol`, Killarney 3296906): NULL / mild regression.** Adding a soft Gaussian collision cost on `plan_reg` as a training-loss term (λ=0.2, σ=2.0m, GT-derived agent positions, all 18 plan modes per decoder stage) gave `L2=0.5515 / obj_box_col=0.050%` from a `planpredtrajdeformmm` base. Reference plain `planpredtrajdeformmm` (cross-server avg from `2026_04_02_plan_refine.md`) is `L2=0.542 / 0.047%`. So softcost is L2 +0.010 (~+1.8% relative) and CR essentially flat. Detection unchanged (`NDS=0.5257`, `mAP=0.4142` — within noise of every other plan-head perturbation). The cost is NOT lifting feasibility geometry above what imitation already captures, and is mildly hurting L2 — most likely because the cost gradient pushes ego trajectories away from agents the human did not need to dodge, weakening imitation alignment.

Caveat on Exp 1: cross-server comparison. We do not have a same-server Killarney baseline for plain `planpredtrajdeformmm`. The cross-server delta is small enough that single-seed noise could absorb it; but the lack of any positive movement on either L2 or CR is strong enough evidence to deprioritize the training-loss deployment.

Joint reading: the inference-side scorer is doing real work, but adding the same cost as a training signal didn't help. This is consistent with the broader pattern in `0000_00_00_research_findings.md` — every architectural perturbation to the planner head has regressed planning. Inference-side scoring is the only direction in this report that's still "live" given current evidence.

Open questions still pending:
- **Exp 2 (inference scoring)**: does soft multi-cost selection improve on the hard-binary feasibility filter at the same checkpoint, given the 0.043pp headroom Exp 0 just demonstrated?
- **Exp 3 (joint A+B)**: now unlikely to be informative since the training-side leg did not land independently.

**Exp 1 retry (`softcostcol_v2`, Narval 59920383): NULL.** Geometry-aware retry with min-of-4-ego-corner SDF to agent oriented bbox, closest-agent-only per (mode, t), softplus(−min_sdf/τ=0.5m), λ_col=0.05, ego heading from atan2 trajectory tangent (matching `rescore.get_yaw`). Reached `L2=0.5481, obj_box_col=0.059%` from a `planpredtrajdeformmm` base; v1 was `L2=0.5515 / 0.050%`; plain `planpredtrajdeformmm` cross-server avg is `L2=0.542 / 0.047%`. So v2 vs v1 is L2 −0.003 (noise) traded for CR +0.009pp (noise). v2 vs plain is L2 +0.006 / CR +0.012pp — same regression direction as v1, smaller magnitude. Detection unchanged (`NDS=0.5266, mAP=0.4176, mAP_normal=0.5566`). Caveat: cross-server comparison vs plain (no narval `planpredtrajdeformmm` baseline). The geometry + normalization fixes did not lift the result out of v1's regime. Combined with the Exp 2 sweep, both training-time and inference-time deployments of an analytic collision cost have failed independently.

**Exp 2 (inference selector sweep, Narval 59920744/45/46): catastrophic L2 regression.** Three-way sweep `w_col ∈ {3, 10, 30}` at `σ=2m` on the same Killarney `iter_11720.pth` ckpt that Exp 0 disabled rescore on. Results monotone in `w_col`:

| | L2 | obj_box_col |
|---|---|---|
| hard rescore (existing) | 0.4988 | 0.063% |
| no rescore (Exp 0) | 0.5008 | 0.106% |
| soft `w_col=3`  (Exp 2a) | **0.6667** | 0.067% |
| soft `w_col=10` (Exp 2b) | **0.7205** | 0.072% |
| soft `w_col=30` (Exp 2c) | **0.7492** | 0.079% |

L2 increases ~33–50% relative for every `w_col`; CR is between hard and norescore but never reaches hard's `0.063%`, even at large `w_col`. Detection irrelevant (rescore touches selector only).

Diagnosis: the saturating Gaussian `exp(−clamp(min_sdf, 0)² / σ²)` is *always on* — every mode gets nonzero penalty proportional to closest-agent SDF, regardless of whether it actually collides. With σ=2m, sdf=2m still gives 0.37 penalty per timestep × 6 = 2.2 total, times `w_col` swamps the plan_cls logits (roughly [−3, +3] range). Selection collapses to "mode farthest from agents," which often deviates from the imitation target — humans routinely drive close to parked cars or follow lead vehicles. Hard rescore only fires on binary corner-in-box, so >95 % of modes are untouched and `plan_cls` drives selection — that's why hard works. The CR-versus-`w_col` U-shape (best at `w_col=3`) suggests the cost still helps a little when its weight is low enough not to dominate plan_cls; but the kernel never thresholds, so even very large `w_col` doesn't recover hard's CR (modes that *would* be hard-masked also see neighbouring-mode penalties, distorting the relative ordering).

Joint reading after both legs: analytic collision costs in their tested forms (Gaussian on point distance, Gaussian-on-SDF, softplus-on-SDF) do not improve planning at either training or inference time. The cost shape itself is the binding constraint — neither tested kernel is sharp enough to mimic the binary corner-in-box semantic of the metric while remaining differentiable / continuous. The remaining live direction is the **learned scorer / value head** (deferred earlier in this report) with a *stationary* label decoupled from the planner's own predictions — sharpness comes from BCE supervision, not from kernel hand-design.

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

**Update after Exp 0 + Exp 1 (2026-04-26):** Exp 0 cleared the headroom bar; Exp 1 first cut was null but the implementation has known fidelity issues. Run Exp 2 and a fixed Exp 1 retry **in parallel** rather than sequentially — they share no resources and disconfirm or confirm independent hypotheses.

**Exp 2 (inference selector) — anchor to Exp 0's checkpoint.**
- Base: `sparsedrive_r50_stage2_4gpu_bs24_ptaux2d_ppdeformmm_planifls.py` (the same checkpoint Exp 0 disabled rescore on; gives a clean three-way same-checkpoint comparison: hard rescore = 0.4988/0.063%, no rescore = 0.5008/0.106%, soft rescore = ?).
- Implementation in `projects/mmdet3d_plugin/models/motion/decoder.py` `HierarchicalPlanningDecoder.rescore()`: replace `-9999` mask of colliding modes with a continuous score offset `−w_col · C_col(mode)`, where `C_col` is a soft Gaussian on the existing oriented-bbox-corner geometry (per-mode summed over agents/timesteps; min over agents per waypoint to avoid the dilution problem from Exp 1). Selection becomes argmax over `cls_logits + (−w_col · C_col)`. No retraining required.
- σ ≈ 2m, w_col swept over a few values offline if needed; eval only on the trainval val set.

**Exp 1 retry (training loss) — fix the implementation issues the first cut exposed.**
- Base: same as first cut (`sparsedrive_r50_stage2_4gpu_bs24_planpredtrajdeformmm.py`).
- Geometry: **min-of-4-ego-corners signed-distance** to each agent's oriented bbox. Build agent box from `gt_bboxes_3d` (`x, y, w, l, sin_yaw, cos_yaw`); rotate ego corner into agent's local frame; SDF = `sqrt(max(qx,0)² + max(qy,0)²) + min(max(qx,qy), 0)`. Per (mode, agent, t), take `min` over the 4 ego corners → matches the corner-in-box semantic of `obj_box_col`. Penalty per pair = `softplus(−min_corner_sdf / τ)`, τ ≈ 0.5 m. Single-point-SDF (ego center only) is a strictly weaker signal — discarded.
- Ego heading at each waypoint comes from the trajectory tangent: `heading[t] = atan2(cumxy[t+1] − cumxy[t−1])` central difference, forward/backward diff at endpoints. Differentiable; reflects the planner's own predicted motion direction.
- Ego footprint `(L_ego, W_ego)` matches the constants the existing `HierarchicalPlanningDecoder.rescore()` uses so the loss and the inference geometry agree exactly.
- Aggregation: closest-agent-only per (mode, t) — `min_a softplus(−sdf/τ)` — then mean over `(M_total, T_ego)`. Drops the per-scene-agent-count dilution from the first cut.
- Scope: keep all 18 modes for a first attempt (isolate "geometry + normalization fixes" from "scope change"); cmd-indexed-only is a follow-up if this is still flat.
- λ_col=0.05 (down from 0.2) since the larger weight visibly hurt L2 last time.

If neither Exp 2 nor the fixed Exp 1 lands, redirect to the unified-architecture (`plan_unified`) and closed-loop alignment (`plan_closedloop` to be created) directions; the binding constraint isn't at the scoring layer.

**Implementation + submission (2026-04-27).**

Code (committed `966adee`):
- `projects/mmdet3d_plugin/models/motion/decoder.py` — `HierarchicalPlanningDecoder` gains `use_rescore_soft / rescore_soft_w_col / rescore_soft_sigma`. New `rescore_soft()` method reuses the same ego/motion 7-d box construction as `rescore`, then takes `min` over (4 ego corners, agents, motion modes) per (ego_mode, t), saturating Gaussian `exp(-clamp(min_sdf, 0)² / σ²)`, sums over t, applies `−w_col · C_col` offset to `plan_cls`. Low-conf agents (`det_confidence < 0.5`) are pushed to SDF=1e6 to drop them from the min. The forward 0.5m offset is applied with correct `[..., 0]/[..., 1]` indexing; the long-standing `[0]/[1]` row-indexing oddity in the hard variant is left alone to preserve baseline behaviour.
- `projects/mmdet3d_plugin/models/motion/motion_planning_head.py` — `_loss_planning_softcost_collision` dispatches to a new `_loss_planning_softcost_collision_sdf` when `plan_softcost_geometry='sdf_corners'`. The new path computes ego heading from the trajectory tangent (atan2 central diff with the same static-distance guard as `rescore.get_yaw`), applies a +0.5 m forward offset, builds 4 ego corners with the rescore footprint, computes SDF to each GT agent's oriented bbox, takes `min` over the 4 corners and then over agents (closest-agent-only) per (mode, t), and penalises with `softplus(−min_sdf / τ)` averaged over `(M=18, T=6)`.

Sweep choice for Exp 2 (eval-only): a 3-point `w_col ∈ {3, 10, 30}` sweep at fixed `σ=2 m` on a single shared checkpoint. The soft selector behaviour scales with `w_col` — at very large values it converges to the hard rescore (CR ≤ 0.063 %), at very small values it converges to no rescore (CR → 0.106 %); the curve localises the operating point.

Configs (4 new):
- `..._planpredtrajdeformmm_softcostcol_v2.py` — Exp 1 retry train. `plan_softcost_geometry='sdf_corners'`, `plan_softcost_collision_weight=0.05`, `plan_softcost_collision_tau=0.5`, σ kept at 2 m for symmetry with the inference selector but unused on the SDF path.
- `..._planifls_softrescore_w{3,10,30}_s2.py` — Exp 2 evals. Hard rescore off, soft rescore on, `rescore_soft_sigma=2.0`, `rescore_soft_w_col` swept. `eval_mode = {with_planning=True}` only — det/track/map/motion metric computation skipped (forward pass still runs since rescore_soft consumes `det_output` + `motion_output`). Saves ~10–20 min of post-processing per run.

Workflow note: Exp 2 evals reuse the *Killarney* `iter_11720.pth` ckpt for a strict same-checkpoint comparison against Exp 0's `0.5008 / 0.106%` and the Killarney baseline's `0.4988 / 0.063%`. Ckpt scp'd Killarney → local → narval to `work_dirs/sparsedrive_r50_stage2_4gpu_bs24_ptaux2d_ppdeformmm_planifls_killarney/iter_11720.pth` so it does not clobber narval's separately-trained planifls ckpt at the canonical work_dir.
