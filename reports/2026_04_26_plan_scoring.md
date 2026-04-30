# Plan Scoring Beyond Hard Heuristic Feasibility

2026-04-26

## TODO

- [x] Run cheap diagnostic: disable rescore on current best, eval — bounds inference-time headroom (Killarney 3302063, `ptaux2d_ppdeformmm_planifls_norescore`: L2=0.5008, CR=0.106% vs 0.4988/0.063% with rescore → +0.043pp CR headroom)
- [x] Implement soft collision loss as training-loss term — first cut NULL (Killarney 3296906, `planpredtrajdeformmm_softcostcol`: L2=0.5515, CR=0.050% vs 0.542/0.047% plain baseline avg)
- [x] Exp 2 (inference selector, 3-point `w_col` sweep at σ=2m on Killarney ckpt synced to narval) — NULL: catastrophic L2 regression at every `w_col` (0.6667 / 0.7205 / 0.7492 vs 0.4988 baseline), CR never reaches hard rescore's 0.063%. Saturating-Gaussian-on-SDF kernel is "always on" and dominates plan_cls (Narval 59920744/45/46).
- [x] Exp 1 retry (`softcostcol_v2`, training, geometry-aware SDF, λ=0.05, τ=0.5m) — NULL: L2=0.5481, CR=0.059% (Narval 59920383). Geometry + normalization fixes did not lift result out of v1's regime; both v1 and v2 mildly regress vs plain `planpredtrajdeformmm` baseline.
- [ ] Joint A+B (Exp 3) — DROPPED; both legs failed independently.
- [x] First swing at learned scorer (`planaux_conf_anchorlabel`, Trillium 472465) — **catastrophic regression** (L2=0.7692, CR=1.028%). Diagnosed in "Discussion" below: head trained well (acc=0.977) but inference selector swamped `plan_cls`, and the label was a 2m proximity proxy on plan-anchor trajectories rather than the actual collision metric.
- [ ] **Cancel DGX 3659** — duplicate of Trillium 472465, no informational value from confirming a known-bad regression.
- [ ] **Next: `planaux_conf_evalmatch` (Exp 4) — eval-aligned learned rescore.** Re-do the learned scorer with three structural fixes: (1) BCE label = exact replication of `planning_eval.py`'s `obj_box_col` per-(mode, anchor) computation against GT agent futures, (2) inference selector mirrors hard rescore (`-999` mask + all-collide fallback) instead of soft penalty, (3) head is trained decoupled from inference (no rescore active during training). See "Exp 4 design" below.
- [x] Exp 4–7: evalmatch v3, threshold/aggregation eval-only sweep, evalmatchmode v4, v4 followups + hybrid_or — see Discussion sections below. **Conclusion on K/V-on:** v4 ties hard rescore on CR via `hybrid_or` but never beats it; v4 alone lands at L2=0.5328 / CR=0.080% (vs hard rescore 0.4988 / 0.063%).
- [ ] **Exp 8 (running): can hard rescore be removed on the K/V-off paper architecture?** Two configs in parallel on Killarney: (Option 1) `_evalmatchmode` (3366620) — K/V-off + v4 learned scorer; (Option 2) `_distillrescore` (3366621) — K/V-off + `plan_cls` distilled from `rescore()` collide flag at training, `use_rescore=False` at inference. See "Exp 8" below.
- [ ] Defer iterative refinement at inference / hybrid analytic-prior + learned-residual until Exp 8 results land.

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
| DGX | `ptaux2d_ppdeformmm_planifls_planaux_conf_anchorlabel` (learned scorer, anchor label, λ=0.05) | 3659 | SUBMITTED |
| Trillium | `ptaux2d_ppdeformmm_planifls_planaux_conf_anchorlabel` (replicate of DGX 3659) | 472465 | COMPLETED — Exp learned-scorer |
| TBD | `..._planinstfeat_laststage_softcost` with soft-cost rescore (joint) | — | DEPRIORITIZED — Exp 1 v1 didn't land |
| Killarney | `_laststage_nodetmap_decoder6_planwp_evalmatchmode` (Exp 8 Option 1: K/V-off + v4) | 3366620 | SUBMITTED |
| Killarney | `_laststage_nodetmap_decoder6_planwp_distillrescore` (Exp 8 Option 2: K/V-off + plan_cls distill) | 3366621 | SUBMITTED |

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
| `ptaux2d_ppdeformmm_planifls_planaux_conf_anchorlabel` (learned scorer, anchor label, λ=0.05) | **0.7692** | **1.028%** | 0.6003 | 0.7256 | 0.5169 | 0.4269 | 0.5433 | 0.4404 | Trillium 472465; mAP_normal=0.5607 — large planning regression (L2 +0.27, CR +0.98pp vs `_planifls_laststage` reference) |

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

**Learned scorer v1 (`planaux_conf_anchorlabel`, Trillium 472465): catastrophic regression.** First swing at the deferred learned-scorer direction. Auxiliary BCE head over plan modes with stationary label = "anchor-trajectory passes within 2m of GT agent's GT-future trajectory at any (t_a, t_e) pair", λ_planaux=0.05, used in inference selector with `w_col=10.0` additive penalty on `plan_cls`. Result: `L2=0.7692 / obj_box_col=1.028%` from a `ptaux2d_ppdeformmm_planifls` base — `+0.27 L2 / +0.98pp CR` versus the `_planinstfeat_laststage` reference (0.522 / 0.047%) and far worse than the same Killarney `ptaux2d_ppdeformmm_planifls` baseline with hard rescore (`0.4988 / 0.063%`). Detection metrics (`NDS=0.5433, mAP=0.4404`) are within noise of the perception-only baseline — the regression is entirely on the planner.

**Diagnosis (from training-time diagnostics in the run log):** the head trained well — by end of training, `acc_05=0.977`, `pos_logit_mean=0.59`, `neg_logit_mean=0.035`, BCE loss converged 0.020 → 0.005. Supervision worked as designed. Failure is on two simultaneous defects:

1. **Inference selector swamped `plan_cls`.** `rescore_learned()` (decoder.py:460–500) does `max` over high-confidence det anchors → `sigmoid` → `-w_col · score`. With `w_col=10`, trained positives produce `−5.9` per-mode penalty; `plan_cls` natural range is `[−3, +3]`. The penalty is 2× the signal it's supposed to bias, so selection collapses to "minimum max-over-agents conflict score along the *fixed anchor path*" — which throws away every mode-preference the planner learned. This is the same failure shape as Exp 2's soft analytic rescore (`L2=0.67–0.75`).
2. **Label was a 2m point-proximity proxy on the *anchor* trajectory, not the eval's collision criterion on the *predicted* trajectory.** The label used `min_t1,t2 ||GT_agent_pos[t1] − plan_anchor_pos[t2]||² < 4m²` — non-time-aligned, point-to-point, computed against fixed k-means anchor centroids. The `_planinstfeat_laststage` architecture deliberately moves the prediction *away* from the anchor at the final stage, so the head learned about a function that's only loosely correlated with what gets evaluated. The actual eval (`planning_eval.py:13–33`) is **time-aligned, polygon-on-polygon, on the *predicted* ego trajectory**. Different primitive entirely.

The `acc_05=0.977` was real but on the wrong target.

### Exp 4 design (`planaux_conf_evalmatch`)

Goal: replace heuristic `rescore` with a learned approximator of the **eval metric** itself. The learned scorer is the only inference-side direction left after analytic costs (Exp 2) and proxy-label learning (Exp v1) both failed; it's also the only architectural lever that can in principle exceed `rescore`'s ceiling, because the head sees richer features (image-attended `instance_feature[anchor]`, plan-attended `plan_query[mode]`) than the geometric check ever has access to.

**Architecture.** Reuse the existing `plan_conflict_branch` (`motion_blocks.py:84–90`) — it's already pairwise: `MLP([agent_feature_j, plan_query_m]) → conflict logit per (anchor, mode)`. Output shape `(bs, num_anchor, M_total=18)` per decoder stage. **No architectural change needed**; the failure was purely on label semantics and inference selector.

**Label = exact replication of `planning_eval.py`'s `obj_box_col` computation, restricted to the t=0-detected agent set.** New `conflict_label_source='evalmatch'` mode in `_loss_planning_conflict`:
- Build predicted ego box per `(m, t)` with `H=4.084, W=1.85, h=1.56`, `0.5m` forward offset along `get_yaw(plan_reg.cumsum(-2))`. Mirrors `planning_eval.py:76–89` verbatim.
- Build GT ego box per `t` from `gt_ego_fut_trajs.cumsum(-2)` with the same geometry (the eval's `gt_box_coll` is what gets subtracted from `box_coll`).
- Build GT agent box per `(a, t)` for each Hungarian-matched detected anchor: `xy = gt_bboxes_3d[gt_idx, :2] + gt_agent_fut_trajs[gt_idx].cumsum(-2)[t]`, GT WLH from `gt_bboxes_3d`, `yaw_t` from trajectory tangent with `gt_bboxes_3d[gt_idx, YAW]` as start_yaw.
- Polygon-polygon intersection per `(m, a, t)` via vectorized PyTorch SAT (Shapely is too slow for label-gen at training time; SAT is numerically equivalent for these convex rectangles).
- Apply `gt_agent_fut_masks` to skip invisible-future timesteps.
- Reduce: `label[m, a] = any_t(pred_coll[m, a, t] AND NOT gt_coll[a, t])` — the eval's `box_coll AND NOT gt_box_coll` rule (planning_eval.py:103) ports verbatim.
- `pos_weight` calibrated from first 200 iters of training (likely 50–200 given expected `pos_rate ≪ 1%`).
- Loss applied **per decoder stage**, using each stage's own `plan_reg` and `motion_reg` to avoid future-information leakage from final stage to early stages.

**Caveat (documented but accepted for v1):** `fut_boxes[t]` in the eval includes agents that may not have been present at `t=0` (drove into the scene). The per-anchor label can only cover `t=0`-detected agents. This is the same blind spot `rescore` already has (it also only scores against `t=0`-detected anchors), so we don't lose anything *relative to the function we're replacing*. Worth measuring on the val set: compute `obj_box_col` using only `t=0`-keyed agents and compare to the eval's full number; if the gap is < 0.005pp, ignore. Otherwise add a scene-level head as Exp 5.

**Inference selector (`rescore_learned_hard`, replaces `rescore_learned`).** Mirrors `rescore()` (decoder.py:221–309) byte-for-byte:
- For each `(ego_mode, anchor)`: `p = sigmoid(conflict_logit)`.
- Filter low-confidence anchors: `p[det_conf < 0.5] = 0` (matches `rescore`'s `score_thresh=0.5`).
- Per ego mode: `col[m] = (p[:, :, m] > 0.5).any(dim=anchor)` — same `any(anchor)` reduction as `rescore`.
- All-collide fallback: `col[col.all(dim=mode)] = False` — graceful degradation (decoder.py:305–306).
- `plan_cls += col.float() * -999` — same hard mask, no soft swamping.

**Training schedule.** `use_rescore=False, use_rescore_learned=False, use_rescore_learned_hard=False, with_conflict_head=True` during training. The head is supervised by BCE only; `plan_cls` trains as the project-best baseline does, with no rescore active. At eval-time, flip `use_rescore_learned_hard=True`. This decoupling is the third structural fix: in v1, the soft selector was active from iter 0 and `plan_cls` tried to compensate, compounding the regression.

**Validation harness (must run before full 12h training).** Take an existing `ptaux2d_ppdeformmm_planifls` checkpoint; freeze everything except the conflict head; fine-tune for 1 epoch on the new label. Then on val:
- **Head-vs-eval F1**: per-(mode, anchor) `(sigmoid > 0.5)` against the GT-derived label. Target ≥ 0.9.
- **Selector-vs-rescore agreement**: how often `rescore_learned_hard()` picks the same mode as `rescore()`. Target ≥ 0.85 (allowed to disagree when head has better info).
- **Selector chosen-mode collision rate per `obj_box_col` eval**: this is the headline. Compare to rescore's number on the same checkpoint. **If lower → head is using its information advantage and beating the heuristic.**

**Expected outcomes:**
- Parity (within 0.005 L2 / 0.005pp CR of the rescore baseline `0.4988 / 0.063%`): paper headline is "learned approximator of the collision eval is plug-compatible with the heuristic; opens upgrade path via richer features." Keep going.
- **Beats** rescore baseline: paper headline is "we can replace the heuristic feasibility filter with a learned head that exceeds it on its own metric." This is the win condition.
- Regresses by > 0.01 L2 or > 0.02pp CR: the per-anchor formulation is fundamentally limited. Investigate the post-`t=0` agent gap, fall back to the scene-level head (Exp 5).

**Compute.** ~12h on Killarney (single 4×L40S node, full stage-2 training from `sparsedrive_stage1.pth` with `ptaux2d` pretrain). Validation harness ~2h on top of an existing checkpoint. Total: ≤ 14h compute commitment. Cancel DGX 3659 in parallel — it adds zero information.

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

## Exp 5: threshold sweep + selector aggregation on the evalmatch checkpoint (eval-only)

After v3 (`evalmatch`) landed at `L2=0.5355 / CR=0.178%` — better than v1 anchorlabel but ~3× the hard-rescore CR — training-time diagnostics showed the head was well-separated in logit space (mean P(pos)=0.88, mean P(neg)=0.019, `acc_05=0.983`) but operating at a poorly-chosen threshold. The default `T=0.5` threshold combined with `any(anchor)` reduction over ~50 anchors and a base rate of ~0.5% positives means an anchor-level FP rate of ~1.7% cascades into ~57% of modes getting falsely flagged. Exp 5 tested two cheap, eval-only fixes against the existing checkpoint.

**Code (committed `826279f`).**
- `HierarchicalPlanningDecoder` gains `rescore_learned_hard_aggregation` ∈ {`any`, `topk`, `detweighted`} and `rescore_learned_hard_topk_k`. `topk` requires k anchors above the prob threshold; `detweighted` multiplies prob by `det_confidence` and skips the hard det-conf cutoff.

**Submission (DGX 2-GPU eval-only, 10–15 min each; planning-only `eval_mode`).** All re-use Killarney `iter_11720.pth` scp'd to DGX as `work_dirs/..._evalmatch_killarney/iter_11720.pth`.

| # | Config suffix | Param | DGX job | L2 | CR |
|---|---|---|---:|---:|---:|
| (control) | `_evalmatch` | T=0.50, any | Killarney 3314427 | 0.5355 | 0.178% |
| 1 | `_evalmatch_t70` | T=0.70, any | 3675 | 0.5379 | 0.158% |
| 2 | `_evalmatch_t85` | T=0.85, any | 3676 | 0.5413 | 0.142% |
| 3 | `_evalmatch_t90` | T=0.90, any | 3677 | **0.5399** | **0.122%** |
| 4 | `_evalmatch_t95` | T=0.95, any | 3678 | 0.5315 | 0.156% |
| 5 | `_evalmatch_t99` | T=0.99, any | 3679 | 0.5227 | 0.201% |
| 6 | `_evalmatch_t85_topk2` | T=0.85, k=2 | 3680 | 0.5284 | 0.159% |
| 7 | `_evalmatch_t85_topk3` | T=0.85, k=3 | 3681 | 0.5247 | 0.173% |
| 8 | `_evalmatch_t85_detweighted` | T=0.85, det-weighted | 3682 | 0.5370 | **0.119%** |

Reference: hard rescore `0.4988 / 0.063%`; no rescore `0.5008 / 0.106%`.

**Findings.**
- **Threshold sweep is a U-curve in CR.** Tightening the threshold from 0.50 → 0.90 monotonically improves CR (0.178% → 0.122%, −31%), then collapses past 0.90 as too few modes are rejected. L2 trades inversely: tighter T → fewer rejections → L2 closer to no-rescore (best L2 at T=0.99: 0.5227). This matches the well-separated logit picture but the residual ~0.06pp gap to hard rescore (0.063% vs 0.122%) shows threshold tuning alone is not enough.
- **Top-k aggregation does not help.** Requiring k=2 or k=3 anchors above threshold per mode worsens CR vs `any` (0.159% / 0.173% vs 0.142% at T=0.85). False positives are *not* concentrated at single anchors per mode — multiple anchors per mode flag together, so top-k misses true positives faster than it filters out FPs. This in turn implies the head is over-confident on a *spatial cluster* of anchors near each spurious-collision mode, not on isolated noise spikes.
- **Det-weighting is the strongest eval-only fix.** Multiplying per-(anchor, mode) prob by `det_confidence` and removing the hard det-conf gate gives `0.5370 / 0.119%` — best CR among Exp-5 variants, slightly better than the T=0.90 sweep result. Low-det-conf anchors do contribute noticeably to FPs; soft-weighting is a better gate than the binary 0.5 threshold.
- **None match hard rescore.** Best CR among Exp 5 variants is 0.119% (det-weighted) — still ~2× hard rescore's 0.063%, and L2 is ~0.04 worse than hard rescore in every variant. The eval-only operating-point fixes are real but bounded.

## Exp 6: per-mode aggregated training (`evalmatch_mode`)

The decisive failure mode of v3 (`evalmatch`) was train/eval aggregation mismatch: training supervised per-(anchor, mode) BCE, but the inference selector reduces with `any(anchor)`, so per-anchor FPs cascade into per-mode false-collide flags. Exp 6 fixes this directly by making the training objective match the inference computation.

**Code (committed `826279f`).**
- `MotionPlanningHead` gains `conflict_label_source='evalmatch_mode'` and `conflict_smooth_max_tau` (default 5.0).
- Label generation reuses `_fill_evalmatch_labels` (per-(anchor, mode) polygon-on-polygon SAT replicating the eval). Then the per-(anchor, mode) target/weight are collapsed:
  - **Per-mode label**: `mode_label[m] = max_{matched_a} target[a, m]` — exact `any(anchor)` reduction.
  - **Per-mode prediction**: `score[m] = (1/τ) · logsumexp(τ · logit[a, m])` over matched anchors only. With τ=5, this is a tight smooth-max approximation; differentiable through anchors and matches the `max(anchor)` decision the selector makes at inference.
  - BCE applied on per-mode (logit, label); `pos_weight='auto'` recalibrated against the new (much higher) base rate.
- All other knobs identical to `evalmatch` (loss weight 0.10, threshold 2.0, full evalmatch label geometry).

**Result.** Killarney 3319213, full stage-2 retrain, 7.6h on 4×L40S.

| Run | L2 | CR | NDS | mAP | mAP_normal | car_ade | ped_ade | car_epa | ped_epa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hard rescore baseline | 0.4988 | 0.063% | 0.5447 | 0.4403 | 0.5633 | 0.6228 | 0.7201 | 0.5073 | 0.4302 |
| `evalmatch` (v3) | 0.5355 | 0.178% | 0.5472 | 0.4428 | 0.5600 | 0.5966 | 0.7340 | 0.5082 | 0.4239 |
| `evalmatch_mode` (v4) | **0.5328** | **0.080%** | 0.5434 | 0.4376 | 0.5587 | 0.6033 | 0.7271 | 0.5142 | 0.4267 |

**Classifier diagnostics (training-time, last 10% averaged).**

| metric | v3 `evalmatch` | v4 `evalmatch_mode` |
|---|---:|---:|
| `acc_05` | 0.9830 | 0.9489 |
| predict-all-neg baseline | 0.9953 | 0.9237 |
| acc lift over baseline | −0.0123 | **+0.0252** |
| mean P(pos) | 0.883 | 0.614 |
| mean P(neg) | 0.019 | 0.065 |
| `pos_rate` | 0.0047 | **0.0763** |
| F1 (recall=1 bound) | 0.36 | **0.75** |

The `pos_rate` jumps 16× because the per-mode aggregation changes the base-rate problem: the per-(anchor, mode) label is dominated by trivial negatives (most anchors aren't relevant to most modes); the per-mode label has many more positives because *some anchor* is usually relevant somewhere in a colliding mode. Predict-all-negative accuracy drops from 0.9953 to 0.9237, and the classifier now beats it by 2.5pp instead of trailing it by 1.2pp. Implied F1 jumps from ~0.36 to ~0.75 — roughly 2× improvement on the genuine classification task.

**Findings.**
- **Train/eval aggregation alignment is the single most impactful fix.** CR drops from 0.178% → 0.080% (−55%), more than the entire threshold + aggregation eval-only batch combined. This is the first learned-scorer to come within ~0.02pp of hard rescore (0.063%) on CR.
- **L2 still trails hard rescore by ~0.034.** Even at matched aggregation, the head doesn't quite recover the L2 quality of the geometric check. The remaining gap is consistent with the head occasionally rejecting an L2-best mode that hard rescore would have accepted (or vice versa). Whether a follow-up threshold sweep on the v4 ckpt closes this further is open.
- **Detection metrics drift slightly negative** (NDS 0.5447 → 0.5434, mAP 0.4403 → 0.4376) — within run-to-run noise (`±0.005 NDS`) but consistent across the v3 and v4 evalmatch runs vs the hard-rescore baseline. The conflict-head loss adds gradient pressure to the planning decoder; this is a small cost.

**Implications for paper architecture.** v4 promotes the learned scorer from "null path" to "viable but slightly worse than hard rescore." If the v1-paper claim is the no-K/V Stage 2 architecture with optional learned cost replacement (per `2026_04_28_paper_plan.md`), v4 is now a defensible variant: CR within ~0.02pp of hard rescore, L2 within ~0.03, and the inference path is fully decoupled from `det_output`-derived motion futures. The "hard rescore is essential" finding is also weaker now — it's still better, but not by a wide margin once the train/eval objective is aligned.

**Followups.**
- Threshold sweep on v4 ckpt at T ∈ {0.50, 0.70, 0.85, 0.90, 0.95}: cheap eval-only batch (2h DGX). Likely to localise a better operating point given the much-improved classifier separation.
- Selector aggregation variants (top-k, detweighted) on v4: same setup as Exp 5 but on the new ckpt; tests whether the alignment fix changes the optimal aggregation regime.
- Hybrid: stack v4 selector on top of hard rescore (take the OR of their veto decisions) — answers whether the head adds *any* signal hard rescore misses.

## Exp 7: v4 followups — threshold + aggregation + hybrid_or (eval-only)

Run in parallel on DGX (3683–3691) and Killarney (3324524–3324532); 2-GPU each, 2:59:00 timelimit, planning-only `eval_mode`. Both clusters returned identical metrics (Killarney finished first end-to-end; DGX duplicates 3688–3691 cancelled). All re-use the v4 (`evalmatchmode`) `iter_11720.pth` ckpt (Killarney 3319213).

**Code (committed `0d354d6`).** Added `use_rescore_hybrid_or` to `HierarchicalPlanningDecoder`: applies hard `rescore()` then `rescore_learned_hard()` in sequence on the same `plan_cls`. Each writes `-999` to colliding modes, so the per-mode collide flag is the OR of both selectors.

**Results.**

| # | Variant | DGX job | Killarney job | L2 | CR |
|---|---|---:|---:|---:|---:|
| (control) | `_evalmatchmode` (training-end eval) | — | 3319213 | 0.5328 | 0.080% |
| 1 | T=0.50 any | 3683 | 3324524 | 0.5331 | 0.082% |
| 2 | T=0.70 any | 3684 | 3324525 | **0.5316** | 0.081% |
| 3 | T=0.85 any | 3685 | 3324526 | 0.5343 | 0.093% |
| 4 | T=0.90 any | 3686 | 3324527 | 0.5335 | 0.085% |
| 5 | T=0.95 any | 3687 | 3324528 | 0.5334 | 0.080% |
| 6 | T=0.85 topk2 | 3688† | 3324529 | 0.5350 | 0.092% |
| 7 | T=0.85 topk3 | 3689† | 3324530 | 0.5325 | 0.101% |
| 8 | T=0.85 detweighted | 3690† | 3324531 | 0.5320 | 0.097% |
| 9 | T=0.85 hybrid_or | 3691† | 3324532 | 0.5330 | **0.063%** |

†Cancelled before completion; metrics from Killarney.

References: hard rescore `0.4988 / 0.063%`; no rescore `0.5008 / 0.106%`; v3 evalmatch `0.5355 / 0.178%`.

**Findings.**
- **Threshold sweep on v4 is essentially flat.** All 5 thresholds land within 0.003 L2 and ~0.013pp CR. Contrasts sharply with the v3 sweep (T 0.5→0.9 cut CR 0.178→0.122%, −31%). The per-mode aggregation training fix already pushed positive logits to a sensible default operating point, so threshold tuning has nothing left to do. Confirms that v3's threshold sensitivity was an aggregation-mismatch symptom, not a real calibration gap.
- **Top-k and det-weighted no longer help** (and slightly regress). With the aggregation already aligned at training time, FP cascading is gone, and the smoothing effect of top-k or det-conf weighting only filters out true positives.
- **Hybrid_or matches hard rescore CR exactly: 0.063%.** This is the headline result. L2 stays at v4's level (0.5330), not hard rescore's (0.4988), because OR-ing rejection sets only adds rejections; the hybrid selector keeps every veto from either side, including v4's FPs.
- **Implied conclusion: the learned head's correct rejections are a subset of hard rescore's.** If v4 caught any unique collisions, hybrid_or would have CR < 0.063% (lower than either alone). Equality at 0.063% means hard rescore's rejection set is sufficient — the head doesn't see anything hard rescore misses on this architecture and dataset. The L2 cost (0.5330 vs 0.4988 = +0.034) is purely from v4's FPs that wouldn't have been rejected by hard rescore.
- **Net for the paper: the learned scorer is dominated by hard rescore.** v4 is a viable variant if hard rescore is unavailable (e.g., a future architecture with no `det_output`-based motion futures), but it's not a Pareto improvement over the geometric check. The "hard rescore is essential" claim from earlier reports is now strengthened, not weakened: v4 was the strongest-engineered learned alternative and still failed to find unique signal.

**Implications for paper architecture.** Drop the "learned cost replaces rescore" option from `2026_04_28_paper_plan.md`'s Stage-2 architecture choices (line 145). Keep the lightweight motion head + hard rescore. The learned scorer thread closes here unless a different formulation appears (per-(mode, agent) residual, scene-level head with post-t=0 agents, distillation from a planner with access to GT futures).

## Exp 8: Removing rescore on the K/V-off architecture (parallel pair)

After Exp 7 closed the "learned scorer beats hard rescore on K/V-on" line, two questions remain for the paper-plan K/V-off Stage 2 (`_laststage_nodetmap_decoder6_planwp`, current best L2=0.5150 / CR=0.068%): (a) does v4's `hybrid_or` subset relationship to hard rescore hold on the K/V-off trajectory distribution (where the planner produces different mode preferences without det K/V), and (b) is there a training-time route to remove rescore that we never tried — namely, distill `rescore()`'s collide flag into `plan_cls` directly so the planner internalises the constraint and `use_rescore=False` at inference becomes a no-op?

**Code (committed `00ae4e1`).**
- `HierarchicalPlanningDecoder.rescore()` factored: `compute_rescore_collision_mask()` returns the per-(sample, ego_mode) bool collide flag (with the all-collide fallback applied) without applying the -999 score offset. `rescore()` is the thin wrapper that adds the offset.
- `MotionPlanningHead` gains `plan_distill_rescore_enable` and `plan_distill_rescore_weight`. New `_loss_planning_distill_rescore()` calls `compute_rescore_collision_mask` on detached cmd-indexed plan_reg/motion/det outputs; pushes plan_cls of flagged modes toward 0 via BCE-with-logits (target=0, weight=collide_mask). Per-iter diagnostics: `plan_distill_collide_rate`, `plan_distill_flagged_prob_mean`. Loss applied at every decoder stage.
- `loss()` / `loss_planning()` signatures now thread `det_output` through (kept optional; backwards-compat default `None`).

### Configs (both based on the K/V-off best, `_laststage_nodetmap_decoder6_planwp`)

**Option 1 — `_evalmatchmode` (Killarney 3366620).**
- Reuses the `evalmatch_mode` BCE conflict head + smooth-max(τ=5) per-mode aggregation that landed v4 (`_planaux_conf_evalmatchmode`) at L2=0.5328 / CR=0.080% on K/V-on.
- Inference selector: `use_rescore=False, use_rescore_learned_hard=True` (T=0.5, agg=`any`).
- Architecture deviation: `num_det=50` (vs the strict K/V-off baseline's 0) so the conflict head's pairwise MLP has agent features. K/V into the planner is still nulled via `skip_perception_kv=True` (the planner's `gnn`/`cross_gnn` ops never see the agent tokens), so this is K/V-off in the same sense as the parent — only the conflict head's pairwise MLP reads det features. `with_conflict_head=True` set in both `motion_plan_head` and `refine_layer` (the latter is what actually instantiates `plan_conflict_branch`).
- Tests: do v4's rejections still subset hard rescore's, or does the K/V-off trajectory distribution surface unique signal the head can catch?

**Option 2 — `_distillrescore` (Killarney 3366621).**
- Strict K/V-off base (`num_det=0, num_map=0, skip_perception_kv=True`) + `plan_distill_rescore_enable=True, plan_distill_rescore_weight=0.05`.
- Inference selector: `use_rescore=False`, no learned head — the planner's own logits encode the constraint after distillation training.
- The det head is still trained (perception-as-supervisor), so `det_output` exists at training time for distill label generation; just not consumed as planner K/V or in the conflict head's pairwise MLP.
- Tests: can the planner be trained to make hard rescore a no-op? Failure mode if any: `plan_cls` distillation may compete with imitation L1, regressing L2 (analytic `softcostcol_*` precedent suggests this is the prior).

### Reference points (same baseline family for direct comparison)

| Architecture | Selector | L2 | CR | Source |
|---|---|---|---|---|
| K/V-on `_planifls` | hard rescore | 0.4988 | 0.063% | Killarney baseline |
| K/V-on `_planifls` | v4 `evalmatchmode` | 0.5328 | 0.080% | Killarney 3319213 |
| K/V-on `_planifls` | hybrid_or (hard ∪ v4) | 0.5330 | 0.063% | Exp 7, DGX 3691 / Killarney 3324532 |
| K/V-off `_decoder6_planwp` | hard rescore | 0.5150 | 0.068% | Killarney 3319227 (current best K/V-off) |
| K/V-off `_decoder6_planwp` | v4 `evalmatchmode` (Opt 1) | ? | ? | **Killarney 3366620 (this exp)** |
| K/V-off `_decoder6_planwp` | distilled plan_cls (Opt 2) | ? | ? | **Killarney 3366621 (this exp)** |

### Expected outcomes / decision tree

- **Option 1 lands at parity with K/V-off + hard rescore (within ±0.005 L2 / ±0.01pp CR).** v4 generalises to K/V-off; paper has a clean "drop hard rescore on K/V-off architecture, learned head suffices" headline.
- **Option 2 lands at parity.** Stronger result: the planner's own logits encode the constraint with no additional inference machinery. Distillation is a viable way to remove rescore *and* the conflict head from inference.
- **Both regress (L2 +0.02–0.05, CR +0.02–0.05pp).** Hard rescore is structurally load-bearing — the geometric check sees something neither learned-head nor distilled-logits replicate. The paper keeps hard rescore in the K/V-off architecture (consistent with `2026_04_28_paper_plan.md` decision #4).
- **Only Option 2 lands (Option 1 regresses).** Distillation through the existing planner head turns out to be a stronger route than pairwise per-anchor learning. Suggests the bottleneck on K/V-on was the conflict head's per-anchor formulation, not the supervision signal itself.
- **Only Option 1 lands (Option 2 regresses).** Distillation interferes with imitation. Stick with the conflict head + learned-hard inference selector on the K/V-off architecture.
