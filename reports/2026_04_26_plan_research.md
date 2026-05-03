# Planning Research Directions

2026-04-26 (revised 2026-05-02)

## Role in minS2

The four research directions below are now scoped through the **minS2** north star (see `reports/2026_04_28_paper_plan.md`): stage-2 with everything off except ego queries + planning task/loss. Translated:

- **#1 Plan scoring** — `evalmatchmode` (v4) + Stream B1's image-at-det conflict input are bundled into minS2 (Killarney 3397346, in flight). This direction is *implemented* in the paper headline; the remaining work is reproducibility + a clean "v4 vs hard rescore" ablation row.
- **#2 Unified planning architectures / stage-1 planning supervision** — egoonly stage-1 (planning loss + motion-keepalive at stage 1 only) is the most direct test. Apollo `joint_nodetach` (in flight) closes the long-flagged non-detach hypothesis. Stage-1 candidates are evaluated by their **stage-2 transfer through minS2** going forward.
- **#3 Closed-loop alignment (NavSim)** — orthogonal to minS2 architecture but is the planned cross-benchmark generalization study for the paper.
- **#4 SSL** — deferred. Not on the minS2 critical path.

## TODO

- Kick off #1 (plan scoring) and #2 (unified planning architectures) immediately
- Begin #3 (closed-loop alignment) once #1 and #2 produce first results
- Defer #4 (SSL) until at least one of the above lands

## Abstract

ForeSight has reached L2=0.522 / CR=0.047% on nuScenes via the `plan_refine` line of work, with five recent attempts at perception-side / interface-side improvements (`detrel`, `planaux_da`, `planaux_conf`, `alldet`, `bidir`) all regressing or null. This document scopes four research directions committed to going forward, each with multiple solution angles and an honest novelty assessment. **Plan scoring** (#1, especially as a training-time loss) and **unified planning architectures** (#2, especially stage-1 planning supervision) are the highest expected-impact directions. **Closed-loop alignment** (#3) is the most honest framing of where the field is — open-loop nuScenes is saturating and the real headroom may be on NAVSIM / Bench2Drive. **SSL for E2E driving** (#4) is a long bet with high variance. Several other directions were considered and scoped out; they are listed at the end.

## Intro

The four reference papers (HiP-AD, DiffusionDrive, DeMo, DriveTransformer) plus ForeSight's plan_refine line have largely converged on the same recipe: trajectory-conditioned attention + multi-modal output + iterative refinement + some form of stage coupling. The competitive surface within this recipe is narrow — differences between SOTA papers are 0.01-0.03 L2 and a few PDMS points. The recent five-experiment failure on ForeSight suggests we are approaching the empirical ceiling of this paradigm on open-loop nuScenes metrics.

The interesting research questions are upstream of this saturation. The four directions below change the *supervision* (what signal trains the model), the *architecture* (where representations live and how they couple), the *evaluation* (which metric is measured), or the *pretraining* (what data shapes the backbone) — rather than yet another decoder block.

Honest framing: at 0.522 / 0.047%, even the most generous gain estimates per experiment are small in absolute terms. Real progress measured *on the current metric* may not be available; the best path may be either changing the metric (#3) or making mechanism-distinct backbone-level changes (#2 stage-1) that haven't been tried yet.

Question:
- Which directions are most likely to move performance, given five recent failures and a saturating metric?

## Method

Each direction below is described as: *core thesis*, *solutions to investigate*, *prior work and honest novelty*, *expected impact*, *building blocks present in ForeSight*, *risk*, *minimum viable experiment*.

### 1. Plan scoring beyond hard heuristic feasibility

**Thesis.** The current plan_refine baseline already scores plans at inference (`HierarchicalPlanningDecoder.rescore()` — a hard-binary feasibility filter applying `-9999` to colliding modes). It is a primitive, single-stage, non-learned scorer. The research direction is *what comes after a hard-binary heuristic scorer*: richer cost design, learned scoring, multi-stage scoring, scoring as training supervision. The "inference vs training" distinction collapses — both are uses of the same scoring machinery, with a gradient at training time and an argmin at inference.

**Solutions to investigate.**
- **Soft analytic multi-cost** extending the existing rescore: `exp(-dist²/σ²)` collision + drivable + comfort + progress, aggregated over motion modes by confidence
- **Learned scorer / value head**: trained to predict trajectory quality from rollout outcomes (analog of an actor-critic value)
- **Hybrid: analytic prior + learned residual**: the analytic cost provides a strong prior; the learned head corrects systematic errors
- **Two-stage scoring**: hard feasibility filter (existing rescore) followed by a learned ranking among feasible candidates
- **Score as differentiable training loss** (the higher-expected-impact deployment): gradient flows from the scorer back through the planner. Direct soft cost on trajectory geometry — mechanistically distinct from `plan_aux`'s failed classification heads
- **Iterative refinement at inference**: gradient descent on the scorer's output to refine the top candidate

**Prior work and honest novelty.** Cost-based motion planning is decades old in classical robotics. Learned cost functions / inverse RL have been around for fifteen years. Reward modeling is the LLM analog (RLHF, DPO). Within E2E driving, DiffusionDrive's denoising is one form of inference-time critic; PDM in NAVSIM uses analytic post-hoc scoring. The genuine novelty here is narrower: the **combination** of (a) extending the existing primitive (not greenfield), (b) deployment as both selector and supervisor, (c) ablating the chain rigorously (no scorer → hard binary → soft analytic → learned → joint A+B). The cost design itself is unlikely to be novel; the end-to-end sparse-token integration and ablation may be.

**Expected impact (honest).** Inference-time deployment (selector) is bounded by what the existing hard-binary rescore is leaving on the table — likely small (0-0.005 L2, modest CR). Training-time deployment (supervisor) is potentially larger because it directly shapes trajectory generation toward feasibility / drivable / comfort — plausibly 0.005-0.02 L2 with a stronger CR effect. The training-time deployment is the part to prioritize.

**Building blocks present.** Existing `rescore()` (hard binary), drivable mask machinery from `planaux_da`, multi-modal motion outputs, `motion_loss_cache.indices`, `gt_ego_fut_cmd`.

**Risk.** Soft costs as training loss risk gaming (planner learns to optimize the cost rather than imitate humans) — requires careful imitation prior weighting. Differentiable approximations to discrete events can be unstable.

**MVE (cheap diagnostic).** Disable rescore (`use_rescore=False`) on the current best checkpoint and re-evaluate. The CR delta isolates how much the existing primitive contributes and bounds the headroom for richer inference-time scoring.

**Bigger first experiment (training-side, recommended in parallel).** Add a soft collision loss directly on the trajectory output — agent footprints from `det_output` × ego footprints from `plan_reg`, soft distance cost summed over the horizon, weighted into the standard plan loss. This tests the higher-expected-impact deployment.

### 2. Unified planning architectures

**Thesis.** Most E2E driving stacks sit at one of two extremes: fully sequential / unidirectional (UniAD, VAD, SparseDrive) or fully unified everywhere (DriveTransformer). ForeSight is somewhere in between — the planning decoder *already* shares self-attention across (ego ∪ agents) via `temp_gnn`, `gnn`, and `cross_gnn` (line 707 of `motion_planning_head.py`). What it does *not* do: extend that sharing further upstream (into the detection stage), include map tokens as queries in the planning decoder, or propagate planning gradients back into the perception backbone. The plan_unified failures (`bidir`, `alldet`, `detrel`) and plan_refine successes constrain the design space empirically. This direction asks: **what architectural couplings between planning and the rest of the stack actually help, given that within-planning-head ego↔agent coupling is already present?**

**Solutions to investigate.**
- **Stage-1 planning supervision** (`stage1_planifls`): planning queries inside the *stage-1* decoder so backbone receives planning gradients from start of training — mechanism-distinct from any inference-time architecture change. **Highest expected-impact mechanism in this direction**: backbone-level alignment historically moved things on this codebase (`ptaux2d` itself, which gave the bulk of the L2 gain to 0.522, was a backbone alignment via auxiliary 2D supervision). This is the analog with planning as the alignment signal.
- **Egoquery in detection** (DriveTransformer-style): ego planning queries co-located with detection queries inside the *detection transformer* from layer 1, separate losses, shared layers — extends the (ego ∪ agents) shared self-attention from the planning head back into the detection head
- **Map promotion**: add map tokens as queries (not just K/V) to the existing planning decoder's self-attention, so map tokens become plan-aware — small extension of what's already there
- **Cross-task attention bridges**: explicit cross-attention between detection-head agent tokens and planning-head ego tokens at chosen layers, rather than full token sharing
- **Recursive coupling**: detection → plan → re-detect (planning-aware re-encoding) → re-plan
- **Shared-backbone with task-conditioned attention**: detection and planning attend to the same image features but with task-specific attention patterns

**Prior work and honest novelty.** DriveTransformer occupies the "fully unified" end. UniAD / VAD / SparseDrive occupy the "sequential" end. The middle is sparse. The genuine contribution is not "we propose a new unification" but **a rigorous study of the architectural design space, anchored by the existing plan_unified ablations**. The five configurations (current ForeSight = ego↔agent sharing in planning head only; bidir; alldet; egoquery; stage1_planifls) populate the design space cleanly. Either positive (one configuration wins meaningfully) or negative (no architectural coupling helps over the strong baseline) is publishable.

**Expected impact (honest).** Stage-1 planning supervision is the highest-expected-impact single experiment in the entire roadmap — backbone-level interventions have a strong track record on this codebase, and this one is mechanism-distinct from anything in the failed plan_unified set. Plausibly 0.01-0.03 L2. Egoquery is harder to predict; DriveTransformer's full unification doesn't dominate HiP-AD on closed-loop, suggesting unification alone isn't a magic bullet — plausibly 0-0.015 L2 with modal outcome being null. Map promotion is the cheapest variant but the most modest expected effect (0-0.005 L2).

**Building blocks present.** Existing ego↔agent shared self-attention in `MotionPlanningHead`. Existing instance bank, deformable attention, plan_anchor / k-means infrastructure, plan_refine endpoint attention machinery. The four plan_unified failed configurations as built-in negative controls.

**Risk.** This is the most surgical change set. If all configurations fail, the entire "coupling-via-architecture" hypothesis is exhausted, and the contribution becomes a comprehensive negative paper.

**MVE (cheap).** Map promotion in the planning decoder — add map tokens as queries (not just K/V) to the existing self-attention. Bounded to `motion_planning_head.py`.

**Bigger first experiment (highest-impact, recommended in parallel).** Stage-1 planning supervision (`stage1_planifls`). The infrastructure is mostly built from `plan_unified` Experiment 1 — needs `with_motion_plan=True` in the stage-1 config and `gt_ego_fut_trajs` wired into stage-1 keys. Resulting checkpoint feeds into a stage-2 `ptplan1` variant.

### 3. Closed-loop alignment and the metric problem

**Thesis.** Open-loop nuScenes L2 is saturating — the metric stopped discriminating useful improvements ~0.05 L2 ago. The five-experiment failure may be partly a metric problem, not just a model problem. Two complementary directions: pivot to closed-loop benchmarks, and design better open-loop metrics that correlate with closed-loop.

**Solutions to investigate.**
- **NAVSIM port** (PDMS scoring; DiffusionDrive at 88.1, headroom above)
- **Bench2Drive port** (CARLA-based; HiP-AD at 86.77 DS, DriveTransformer at 63.46 — wide spread)
- **Counterfactual scenario perturbation on nuScenes**: noise on agent positions, missing detections, map errors; measure plan robustness
- **Mode-distribution evaluation**: does the planner produce diverse plausible trajectories or collapse to one mode?
- **Severity-weighted CR**: not all collisions are equal; weight by closing speed / safety margin
- **Coverage-style trajectory eval**: recall against the set of plausibly-good plans

**Prior work and honest novelty.** Closed-loop evaluation is what NAVSIM and Bench2Drive exist for; multiple papers (DiffusionDrive, HiP-AD, ORION, SimLingo) report on them. The pivot itself is not novel. The genuine contribution would be **a systematic comparison: which open-loop signals predict closed-loop performance, and which model-side improvements transfer**. This is the kind of paper that's rare not because the questions are hard but because nobody bothers to do it cleanly.

**Expected impact (honest).** Doesn't move open-loop L2 / CR at all — pivots the *measurement*, not the model. May reveal that current rankings are wrong, may justify why open-loop attempts plateau. Indispensable for direction-setting; zero direct effect on the metric you currently report. The biggest *real* progress available may live here, but only because the current metric is saturated.

**Building blocks present.** Existing eval harness; integration / porting effort.

**Risk.** Engineering-heavy. Closed-loop sims are slow and brittle. Bench2Drive needs CARLA infrastructure. NAVSIM is more tractable.

**MVE.** Port a single ForeSight checkpoint to NAVSIM eval. Compute PDMS for the current best plus 2-3 plan_refine variants. If the open-loop and closed-loop rankings disagree, the metric pivot is justified.

### 4. Self-supervised pretraining for E2E driving

**Thesis.** Current pipeline: ImageNet pretrain → stage-1 detection-supervised → stage-2 fine-tune. This is small-data thinking. Massive amounts of unlabeled driving video exist; current E2E pipelines don't use them. The hypothesis: scaling self-supervision can close part of the gap. SSL should be **for the entire E2E driving stack, not just perception** — objectives chosen to produce features useful for planning specifically, not just for downstream detection.

**Solutions to investigate.**
- **SSL on perception** (well-known): multi-view consistency, BEV completion, masked image modelling
- **SSL on motion**: future-frame feature prediction, agent trajectory completion from partial history, scene dynamics modelling
- **SSL on planning**: trajectory completion (mask part of a human demo trajectory, predict the rest), counterfactual trajectory plausibility
- **World model pretraining**: predict future BEV / scene state from past + action; this gives a planning-relevant latent (GAIA-1, DriveDreamer, Wayve direction)
- **Imitation from unlabeled human driving**: large-scale uncurated driving video as pretrain data
- **Counterfactual data augmentation**: perturb scenes, predict outcomes; teach the model to reason about alternatives

**Prior work and honest novelty.** Perception SSL is heavily studied (SimMIM, MAE, MoCo for images; BEVFormer-style multi-view consistency). World models for driving are an active area (GAIA-1, DriveDreamer, Vista, Wayve's research). Future-frame prediction in BEV (FIERY, BEVerse). The genuine novelty for ForeSight would be **SSL objectives chosen specifically for planning utility, not perception transfer** — most existing SSL work measures perception downstream metrics, not planning. A clean comparison of "what SSL objective transfers best to planning" is open.

**Expected impact (honest).** Highest variance in the roadmap. If it works, it could be the largest single-experiment gain — perception SSL has shown 1-5% gains in detection; planning-targeted SSL could plausibly add similar. Could also be zero if the objective doesn't transfer. Long timeline.

**Building blocks present.** Existing backbone, FPN, BEV / camera projections, temporal frame infrastructure.

**Risk.** SSL objective design is hard; transfer is unpredictable; ambitious data infrastructure investment. Likely the longest-bet direction in the roadmap.

**MVE.** Train a single SSL objective (multi-view consistency or BEV completion) on the existing nuScenes train split as a stage-0 pretrain. Compare downstream planning metrics vs. ImageNet pretrain. If even one objective shows planning transfer, the direction is viable.

## Considered and dropped

The following directions were scoped but dropped from active commitment. They remain available if future results redirect priorities.

- **Joint output distribution over ego and agent futures.** The planning head already shares features across (ego ∪ agents) via `temp_gnn`, `gnn`, and `cross_gnn`. The gap is at the output level — explicit mode coupling, joint autoregressive sampling, scenario latent factorization, or joint-consistency loss. Honest assessment: the implicit joint distribution from shared features likely captures most of the available signal; explicit output coupling is a small expected effect on planning metrics (forecasting metrics may improve more). Dropped because the expected impact is small relative to #1 / #2.
- **Uncertainty propagated end-to-end.** Threading calibrated uncertainty (Laplace NLL, ECE-calibrated detection) through detection → motion → planning is a real gap in E2E driving literature. Dropped because calibration rarely moves headline metrics directly; most useful as an enabler for #1's scorer cost weighting (high-uncertainty agents → wider collision margin). Pick up only if #1 lands and uncertainty becomes a natural cost-weight extension.

## Results

Empty — this is a roadmap document; results land in per-direction follow-up reports.

| Direction | Status |
| --- | --- |
| 1. Plan scoring beyond hard heuristic feasibility | COMMITTED — immediate |
| 2. Unified planning architectures | COMMITTED — immediate |
| 3. Closed-loop alignment | COMMITTED — follow-up |
| 4. SSL for E2E driving | COMMITTED — follow-up |

## Discussion

Honest assessment of expected impact on open-loop nuScenes L2 / CR:

| # | Mechanism | Expected impact (L2) | Expected impact (CR) | Why |
|---|---|---|---|---|
| **2 stage-1 planning supervision** | Backbone alignment with planning gradients | 0.01–0.03 | Modest | Mechanism-distinct from all five recent failures; backbone-level changes (e.g. `ptaux2d`) have moved things historically on this codebase |
| **1 plan scoring as training loss** | Direct soft cost on trajectory geometry | 0.005–0.02 | Larger | Mechanistically distinct from `plan_aux` (which predicted properties); shapes generation directly toward feasibility |
| **2 egoquery** | Ego in detection transformer from layer 1 | 0–0.015 | Modest | Modal outcome is null; full unification not dominant in literature (DriveTransformer doesn't beat HiP-AD on closed-loop) |
| **4 SSL** | Stage-0 pretraining for planning | 0 to 0.05 | Variable | Highest variance in roadmap; could be the biggest gain or zero |
| **1 plan scoring as inference selector** | Replace hard binary with soft cost | 0–0.005 | Marginal | Bounded by what existing rescore is leaving on the table; diagnostic MVE quantifies headroom |
| **2 map promotion** | Map tokens as planning-head queries | 0–0.005 | Marginal | Capacity change; map already accessible via cross-attention |
| **3 closed-loop pivot** | Switch metric to NAVSIM / Bench2Drive | N/A on current metric | N/A | Doesn't move the model; may reveal that current rankings are misleading |

**Honest summary.** We are at 0.522 / 0.047% on a saturating metric. Even the most generous gain estimates per experiment are small in absolute terms. The two highest-expected-impact mechanisms — **stage-1 planning supervision (within #2)** and **plan scoring as training loss (within #1)** — are both mechanism-distinct from the recent failure set and have plausible mid-range gains. The biggest *real* progress available may be at the metric level (#3); the model-side gains are bounded by where nuScenes can still discriminate.

Pairwise compounding:

- **#1 + #2**: training-time scorer signal compounds with backbone-level alignment — the planner is shaped both by direct cost supervision and by features encoding planning-aware information from stage 1
- **#1 + #3**: the scorer is itself a closed-loop-shaped metric; if the scorer predicts closed-loop performance better than open-loop L2, that's a finding
- **#2 + #4**: SSL pretraining feeds into stage-1 planning supervision — both are backbone-level interventions and would naturally combine in a stage-0 → stage-1 → stage-2 pipeline

Recommended primary thesis:
- `[#1 plan scoring + #2 unified planning architectures, anchored by #3 closed-loop eval]`

The thesis is honest about the limited headroom available on open-loop nuScenes and the need for both supervision-side (#1) and architecture-side (#2) interventions, anchored by a metric pivot (#3) that establishes whether any of the gains generalize.

## Future Work

Minimum viable experiments to falsify each direction's premise as cheaply as possible, plus the bigger first experiment per direction:

- **#1 cheap MVE**: disable rescore (`use_rescore=False`), re-evaluate current best. CR delta bounds the headroom for richer inference-time scoring.
- **#1 bigger first**: soft collision loss as a training-loss term added to the standard plan loss. Tests the higher-expected-impact deployment (training-time scorer).
- **#2 cheap MVE**: map promotion in planning head (map tokens added as queries to existing ego ∪ agents self-attention). Bounded to `motion_planning_head.py`.
- **#2 bigger first**: stage-1 planning supervision (`stage1_planifls`). Reuses `plan_unified` Experiment 1 infrastructure; produces stage-1 checkpoint for stage-2 `ptplan1` variant. **Highest expected-impact single experiment in the roadmap.**
- **#3 MVE**: NAVSIM port — port a single ForeSight checkpoint, compute PDMS for current best + 2-3 plan_refine variants. If open-loop and closed-loop rankings disagree, the metric pivot is justified.
- **#4 MVE**: train one SSL objective (multi-view consistency or BEV completion) as stage-0 pretrain; compare downstream planning vs. ImageNet pretrain.

Suggested ordering:

1. **Now (parallel)**: #1 cheap diagnostic + #2 stage-1 planning supervision + #2 cheap map-promotion MVE. Three experiments across two directions, bounded scope each.
2. **As soon as bandwidth allows**: #1 bigger first (training-time soft cost). Depends on #1 diagnostic informing whether training-time loss carries the win or whether inference-time scoring also has headroom.
3. **In parallel with #1/#2 follow-ups**: #3 NAVSIM port begins. Engineering-heavy, doesn't compete with model experiments.
4. **After at least one of #1, #2, #3 lands**: #4 SSL — long bet, longer timeline, higher variance.

Pattern: cheapest falsification first, then progressively heavier commitments only if cheap tests show traction. The MVE goal is to falsify the assumption fastest, not to produce the contribution.
