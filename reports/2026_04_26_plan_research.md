# Planning Research Directions: Beyond Incremental Architecture Fixes

2026-04-26

## TODO

- Decide which 1-2 directions to commit to as the primary research thesis
- Audit existing codebase for re-usable building blocks per direction (rollout primitives, uncertainty heads, eval harness)
- Identify minimum viable experiment per direction to validate feasibility before committing

## Abstract

ForeSight has reached L2=0.522 / CR=0.047% on nuScenes via the `plan_refine` line of work, with five recent attempts at perception-side / interface-side improvements (`detrel`, `planaux_da`, `planaux_conf`, `alldet`, `bidir`) all failing. The roadmap in `plan_gaps.md` covers near-term incremental experiments. This document scopes six bigger research directions that could carry a paper or thesis chapter, ranked by distinctiveness from current literature and tractability from ForeSight's sparse-token starting point. The strongest candidates are conditional joint prediction (#1) and differentiable scene rollout (#2), both of which reframe the architecture rather than tweak it; closed-loop alignment (#4) is the most honest framing of where the field needs to go.

## Intro

The four reference papers (HiP-AD, DiffusionDrive, DeMo, DriveTransformer) plus ForeSight's plan_refine line have largely converged on the same recipe: trajectory-conditioned attention + multi-modal output + iterative refinement + some form of stage coupling. The competitive surface within this recipe is narrow — the differences between SOTA papers are 0.01-0.03 L2 and a few PDMS points. The recent five-experiment failure on ForeSight suggests we are approaching the empirical ceiling of this paradigm on open-loop nuScenes metrics.

The interesting research questions are upstream of this saturation. Each of the directions below changes the *factorization* of the problem (what's predicted from what), the *supervision* (what signal trains the model), or the *inference* (what compute happens at test time) — rather than yet another decoder block.

Question:
- Among directions that step beyond incremental architecture fixes, which are research-worthy *and* tractable from ForeSight's starting point, and which is the strongest single thesis?

## Method

Each direction below is described as: *core thesis*, *concrete architecture sketch*, *building blocks already in ForeSight*, *what would need to be built*, *evaluation*, *risk*.

### 1. Conditional joint prediction (interactive futures)

**Thesis.** The current factorization — motion head predicts agent futures *independently* of ego, planner consumes them as fixed inputs — assumes agents do not react to the ego. They do. The right factorization is `p(agent_futures | ego_trajectory_candidate)`, with planning scored against the conditional agent rollouts.

**Architecture sketch.**
- Add ego candidate trajectory waypoints as conditioning input to the motion head's cross-attention
- Motion head produces `K × num_agent × num_modes × num_steps` futures, one set per ego candidate
- Plan loss: ego trajectory minimizes a function of the *conditional* agent set, not the marginal one
- Training: condition motion head on GT ego (one path); at inference, condition on each candidate

**Building blocks present.** Multi-modal motion head, multi-modal plan head, instance bank shared across both heads. The `motion_loss_cache.indices` mapping already aligns predicted to GT agents.

**What's new.** Cross-attention pathway from ego candidate waypoints into motion queries; conditional-supervision plumbing; interaction-quality eval.

**Evaluation.** Standard L2/CR + interaction metrics: do agents predicted under different ego candidates differ in plausible ways? Does the planner produce less conservative plans without sacrificing safety?

**Risk.** Compute cost (K parallel motion forwards); may overfit to GT-conditioned motion training; conditional prediction is harder than marginal — could regress agent metrics.

**Position vs. literature.** M2I, GameFormer, MotionLM do this for forecasting; few E2E driving papers do. ForeSight's shared instance bank is unusually well-suited because the conditioning signal can flow through existing query-token plumbing.

### 2. Differentiable scene rollout as planning supervision

**Thesis.** plan_aux showed that *predicting* a property of a trajectory doesn't move planning. The pivot: roll the *whole scene* forward N steps using the model's own motion predictions, and supervise the planner on differentiable cost terms over the rollout (collision, progress, comfort). The model's predictions become its own training signal — closer to model-based policy optimization than imitation.

**Architecture sketch.**
- Existing motion head produces agent futures
- Plan head produces ego candidates
- Rollout step: at each future timestep, advance both ego and agents along their predicted trajectories
- Cost terms (all differentiable):
  - Soft collision: `exp(-dist² / σ²)` between ego footprint and each agent footprint at each step
  - Drivable: `grid_sample` on the drivable mask (same machinery as `planaux_da`) along ego waypoints
  - Progress: dot product of ego displacement with goal direction (from `gt_ego_fut_cmd`)
  - Comfort: curvature / jerk penalties on ego trajectory
- Total rollout cost backprops through plan head and partially through motion head

**Building blocks present.** Multi-modal motion outputs, plan_anchor, drivable mask rasterization (from `planaux_da`), motion_loss_cache.

**What's new.** Differentiable rollout module; soft cost functions; cost weighting schedule.

**Evaluation.** L2/CR + cost-component breakdown (which terms drive the gain). Compare to plan_aux-style aux head supervision as ablation.

**Risk.** Differentiable approximations to discrete events (collision detection, road boundary) can be unstable; rollout horizon trades training cost vs. signal density. Imitation prior may need to be retained to avoid degeneracies.

**Position vs. literature.** Closer to MARC, MARL, model-based RL than current E2E driving. DiffusionDrive's diffusion is a different kind of self-supervision; this is more direct.

### 3. Inference-time search and test-time compute

**Important context: ForeSight already has a primitive form of this.** The `HierarchicalPlanningDecoder.rescore()` function is a **feasibility filter**: it builds ego BEV boxes along plan waypoints, builds agent boxes along the argmax motion mode, runs a hard corner-in-box collision check, and applies a `-9999` score offset to colliding modes. Non-colliding modes are not differentiated. If all modes collide, the offset is suppressed (degenerate fallback).

`use_rescore=True` is the default in every config, including all `plan_refine` configs. The current best (0.522 / 0.047%) is therefore *with* this hard-binary feasibility filter. Search is not a new axis to introduce; it is an existing low-resolution component to extend. The contribution to claim is *richer / differentiable / training-coupled* search beyond hard-binary feasibility filtering — not search itself.

**Thesis.** Replace the existing binary feasibility filter with a continuous cost-based scoring function: soft collision (distance-based), drivable margin (`grid_sample` on a rasterized mask), comfort (curvature, jerk), aggregated over multiple motion modes weighted by motion confidence. This makes search **differentiate between feasible modes**, not just **filter out infeasible ones** — and produces a cost function that doubles as a training signal (#2's rollout cost).

**Architecture sketch.**
- Replace hard corner-in-box collision with `exp(-min_dist² / σ²)` soft cost
- Aggregate across motion modes weighted by `motion_cls`, not argmax
- Add drivable cost via `grid_sample` (machinery from `planaux_da`)
- Add comfort terms (curvature, jerk on plan_reg)
- Score = weighted sum of cost terms; select argmin
- Optional: extend to all 18 plan modes rather than the cmd-conditional 6
- Optional: gradient-descent refinement of the top candidate at test time

**Building blocks present.** Existing `rescore()` function with ego/agent box construction, collision geometry, decoder integration. Drivable mask rasterization from `planaux_da`. Multi-mode motion outputs. Most of what's needed is already wired — the change is replacing the hard collision check and adding cost terms.

**What's new.** Soft cost formulation, multi-cost aggregation, motion-mode weighting, possibly value-function training, latency-quality tradeoff study.

**Evaluation.** L2/CR vs. inference-time compute curve. Crucially: ablate against (a) `use_rescore=False` (no search), (b) current hard-binary rescore (existing baseline), (c) soft single-cost (collision only), (d) soft multi-cost (collision + drivable + comfort). The ablation chain isolates the contribution of each extension over the existing primitive.

**Risk.** Real-time constraints. Also: the existing hard-binary rescore may already be capturing most of the available search gain — the headroom for richer scoring is unknown until the diagnostic MVE runs (see Future Work).

**Position vs. literature.** DiffusionDrive is the only paper doing implicit search (denoising). Making explicit cost-based search modular and training-coupled — and ablating it against feasibility filtering — is the contribution.

### 4. Closed-loop alignment and the metric problem

**Thesis.** Open-loop nuScenes L2 is saturating because the metric stopped discriminating useful improvements ~0.05 L2 ago. The five-experiment failure may be partly a metric problem, not just a model problem. Two complementary moves:

**A. Pivot to closed-loop benchmarks.**
- Port ForeSight to NAVSIM (PDMS metric, 88.1 for DiffusionDrive, room above)
- Port to Bench2Drive (DS, SR; HiP-AD at 86.77 DS, DriveTransformer at 63.46 — wide spread that open-loop nuScenes can't see)
- Re-evaluate plan_refine improvements: do they transfer to closed-loop? Which mechanisms hold up?

**B. Better open-loop metrics on nuScenes.**
- Counterfactual scenario perturbation: noise on agent positions, missing detections, map errors
- Mode-distribution evaluation: does the planner produce diverse plausible trajectories or collapse to one?
- Severity-weighted CR: not all collisions are equal; weight by closing speed / safety margin
- Coverage-style trajectory eval (recall against the set of plausibly-good plans)

**Building blocks present.** Existing eval harness; mostly an integration / porting effort.

**What's new.** NAVSIM / Bench2Drive eval scripts; perturbation harness; better metric design.

**Evaluation.** This *is* the contribution: a systematic study of which open-loop signals predict closed-loop performance, and which model-side improvements transfer.

**Risk.** Closed-loop sims are slow and brittle. Bench2Drive needs CARLA. NAVSIM is more tractable. Engineering-heavy contribution.

**Position vs. literature.** Almost no E2E driving paper does both nuScenes and closed-loop evaluation thoroughly. The ones that do (DiffusionDrive) report selectively. A rigorous open-loop-vs-closed-loop study is a paper by itself.

### 5. Uncertainty as a first-class signal end-to-end

**Thesis.** Detection scores, motion logits, planning logits are all uncalibrated point estimates of confidence. Planning consumes them as fixed inputs. There is no mechanism for "be more cautious near low-confidence agents." Propagating *calibrated uncertainty* through the entire stack and using it as a planning input — and as a search signal in #3 — is something no current E2E paper does well.

**Architecture sketch.**
- Detection head: predict box uncertainty (mean + per-dim variance, or learned ensemble)
- Motion head: Laplace NLL (mean + per-step scale; from DeMo) instead of L1
- Planning head: condition cross-attention on agent uncertainty (high-uncertainty agents attended differently)
- Plan output: explicit per-waypoint uncertainty
- Calibration loss: NLL + ECE term

**Building blocks present.** Some quality estimation in detection (`SparseBox3DRefinementModule.with_quality_estimation`); mode logits exist.

**What's new.** Probabilistic outputs at every stage; uncertainty-conditioned attention; calibration evaluation.

**Evaluation.** Calibration (ECE, reliability diagrams) + standard metrics + safety metrics on uncertain scenarios.

**Risk.** Uncertainty quantification is finicky; calibration is not the same as what makes safer plans. May require careful loss balancing to avoid the model collapsing to high-uncertainty everywhere or underconfident everywhere.

**Position vs. literature.** DeMo uses Laplace NLL but only as a forecasting loss, not as a planning input. No E2E driving paper threads uncertainty through detection → motion → planning end-to-end.

### 6. Self-supervised perception for E2E driving

**Thesis.** Current pipeline: ImageNet pretrain → stage-1 detection-supervised → stage-2 fine-tune. This is small-data thinking. Massive amounts of unlabeled driving video exist; current E2E pipelines don't use it. There is no E2E driving analog to LLM pretraining. The hypothesis: the gap between current E2E driving and human-level driving can be closed in significant part by scaling self-supervision.

**Architecture sketch.**
- Stage 0 (new): self-supervised pretraining on unlabeled multi-camera + lidar data
  - Multi-view consistency: image features should align across cameras given known calibration
  - BEV completion: occlude regions of the BEV, predict from camera features
  - Future-frame feature prediction: predict t+1 features from t
  - Optional: contrastive (temporally adjacent frames as positives)
- Stage 1: detection + map supervision on labeled subset (existing)
- Stage 2: planning fine-tune (existing)

**Building blocks present.** Existing backbone, FPN, BEV / camera projections, temporal frame infrastructure.

**What's new.** SSL objectives; data loading for unlabeled video; pretraining infrastructure.

**Evaluation.** Downstream detection / planning metrics under: (a) same-data-budget comparison vs. detection pretrain; (b) extra-data scaling (does adding unlabeled data help?).

**Risk.** SSL objective design is hard; may not transfer; ambitious data infrastructure investment.

**Position vs. literature.** Most E2E driving papers don't address scaling. A few perception papers do SSL (e.g., SimMIM-style). Bridging SSL → E2E driving is the contribution.

### 7. Task-separated query populations in a shared decoder

**Thesis.** Most E2E driving stacks sit at one of two extremes: fully sequential / unidirectional (ForeSight, SparseDrive), or fully unified everywhere (DriveTransformer). The plan_unified failures (`bidir`, `alldet`, `detrel`) narrow the design space in a useful way and motivate a middle ground: *separate query slots for each task (ego, agents, map)*, each with its own loss, processed through *shared transformer layers*. Detection slots are protected from planning interference (the bidir failure tells us shared tokens fail), but the shared layers receive gradients from both objectives, and planning gets its own representational space from layer 1.

**Architecture sketch.**
- **Final form (`egoquery`)**: ego planning queries (one per mode, `ego_fut_mode = 6`) added to the detection transformer's instance bank from layer 1
- Ego queries participate in detection self-attention, deformable attention to image features, and (separately-supervised) trajectory regression
- Agent queries continue to be supervised by detection loss; ego queries by planning loss; map queries by map loss — *no shared token serves two losses*
- `MotionPlanningHead` retained as a lightweight refinement stage on top of ego query outputs (preserves the planinstfeat endpoint attention)
- **Stage-1 component**: with planning queries in the decoder from stage 1, the backbone receives planning gradients from the start (this is `stage1_planifls` realised at the architecture level)

**Building blocks present.** Existing instance bank, deformable attention, plan_anchor / k-means infrastructure. The plan_refine endpoint attention machinery transfers.

**What's new.** Ego query slots in `instance_bank.py` and `detection3d_head.py`; trajectory output head on ego queries; loss routing so ego queries see only planning supervision; stage-1 config that includes the ego query population.

**Evaluation.** Standard L2/CR + careful ablation against the existing failure set: bidir (token-shared coupling), alldet (access without coupling), detrel (perception-side reshape). The four together populate the design space rigorously.

**Risk.** This is the most architectural change in the roadmap. If it also fails, the entire "coupling-via-architecture" hypothesis is exhausted, and the contribution is a comprehensive negative paper rather than a positive one. Either outcome is publishable, but the positive one is more interesting.

**Position vs. literature.** DriveTransformer goes further (all queries, all layers, no architectural separation). The proposal here is *task-separated representations with shared computation*, which is mechanistically distinct and has a clean comparison story against both DriveTransformer (more separation) and ForeSight current (less coupling).

**Intermediate (MVE before committing to full egoquery).** Promote det/map tokens from keys-only to *queries* inside the existing planning decoder. Keep them as **local copies** that don't propagate back to detection / map heads — this avoids the bidir-style supervision conflict. Run self-attention over the union (ego/plan modes + det copies + map copies) within the planning head. Deformable cross-attention to image features still happens. Plan trajectories still emerge from the ego/plan slots only.

This tests whether the "shared decoder with separate token populations" idea has any traction at all, before committing to the much larger surgery of putting ego queries into the detection transformer. Implementation is bounded to `motion_planning_head.py` — no changes to detection / map / instance bank.

Expected effect:
- Strongest plausible win on map: map tokens become plan-aware, which lets them adapt to "lanes ahead" vs. "lanes behind" without forcing all the adaptation onto the plan query side
- Modest effect on det: det tokens already self-attend in the detection head; re-running here with plan-mode context adds only the plan-conditioning information
- No risk of bidir-style regression because the original supervised tokens are untouched

If the intermediate helps, the full egoquery story becomes "this works, and the shared-layer version starting at stage 1 is the natural extension." If the intermediate is null, the small egoquery promise weakens and direction #7 becomes lower priority.

## Results

Empty — this is a roadmap document; results land in per-direction follow-up reports if and when they are committed to.

| Direction | Status |
| --- | --- |
| 1. Conditional joint prediction | PROPOSED |
| 2. Differentiable scene rollout | PROPOSED |
| 3. Inference-time search | PROPOSED |
| 4. Closed-loop alignment | PROPOSED |
| 5. Uncertainty as signal | PROPOSED |
| 6. Self-supervised pretraining | PROPOSED |
| 7. Task-separated query populations | PROPOSED |

## Discussion

Ranking by distinctiveness from current literature × tractability from ForeSight:

| # | Distinctiveness | Tractability | Strongest if... |
|---|---|---|---|
| **1. Conditional joint prediction** | High | Medium | The thesis is that the field has the wrong factorization |
| **2. Differentiable rollout** | High | Medium | Direct supervision wins where aux-head supervision (`planaux`) failed |
| **3. Inference-time search** | Medium | High | Test-time compute is the next axis of scaling |
| **4. Closed-loop alignment** | Medium | Medium-Low (engineering) | The honest framing — open-loop is saturated |
| **5. Uncertainty end-to-end** | Medium | High | Pairs naturally with #3; fills a real literature gap |
| **6. SSL pretraining** | High | Low (data infra) | Long bet; thesis is that scaling fixes E2E driving |
| **7. Task-separated query populations** | Medium-High | Medium | The plan_unified failures already narrow the design space; egoquery is the one configuration not ruled out |

Pairwise compounding: **#1 + #2** is a natural pairing (conditional motion produces the rollout that #2 supervises against). **#2 + #3** is a natural pairing (rollout cost is also the inference-time scoring function). **#3 + #5** is a natural pairing (search uses uncertainty to know what to search over). **#7 + stage-1 planning supervision** is a natural pairing — the architecture only delivers its full value if planning gradients reach the backbone from the start. The **best two-direction thesis depends on the failure mode of the existing baseline**: if the planner is supervision-limited, **#2 + #3** (model-based rollout + search). If the planner is architecture-limited, **#7** carrying its own ablation set (bidir/alldet/detrel as negative controls) plus stage-1 planning.

Recommended primary thesis:
- `[#2 differentiable rollout + #3 inference-time search, anchored by #4 closed-loop eval]`

Why:
- Mechanically distinct from everything in the recent failure set
- Builds on existing strong baseline (plan_refine) rather than replacing it
- Has clear closed-loop motivation (#4) — addresses the metric saturation problem
- Honest about what the field needs (compute scaling, model-based supervision) rather than another L2 0.01

## Future Work

Next-step actions if a direction is committed to:

- **#1 minimum viable experiment**: condition motion head on GT ego trajectory at training time only (no inference change). If conditional motion improves agent forecasting, the conditional pathway is viable; proceed to inference-time conditioning on candidates.
- **#2 minimum viable experiment**: replace `planaux_conf` aux head with a direct soft-collision loss along the trajectory, no rollout yet. Sanity-checks the differentiable cost formulation before building the rollout module.
- **#3 minimum viable experiment (revised given existing rescore)**: the current best already runs with `use_rescore=True` (hard-binary feasibility filter that applies `-9999` to colliding modes). The actually-informative diagnostic is the *opposite*: re-evaluate the current best checkpoint with `use_rescore=False`. If CR degrades substantially (e.g. 0.047% → >0.08%), the existing filter is doing real work and richer cost-based scoring has plausible additional headroom. If CR barely changes, the existing filter rarely fires and the soft / multi-cost extensions are unlikely to help — redirect toward #7 (architecture) or #4 (closed-loop). One-config eval, no retraining.
- **#4 minimum viable experiment**: port a single ForeSight checkpoint to NAVSIM eval. Compare PDMS to nuScenes L2 ranking across a few plan_refine configs. If they disagree, the metric pivot is justified.
- **#5 minimum viable experiment**: add Laplace NLL to plan output (mean + per-step scale). Check calibration without changing anything else.
- **#6 minimum viable experiment**: train a single SSL objective (multi-view consistency) on the existing nuScenes train split as a stage-0 pretrain. Compare downstream detection vs. ImageNet pretrain.
- **#7 minimum viable experiment**: in `motion_planning_head.py`, promote local copies of det and map tokens to queries; run self-attention over (ego/plan ∪ det copies ∪ map copies) within the planning decoder. Det/map originals untouched, no detection/map supervision conflict. If it helps, the design space points toward full egoquery; if null, egoquery becomes lower priority.

Pick one or two MVEs in the next 2-4 weeks before committing to a full direction. The goal of MVEs is to falsify the assumption fastest, not to produce the contribution.
