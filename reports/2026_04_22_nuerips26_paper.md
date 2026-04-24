# NeurIPS 2025 Paper Direction: Planning-Critical Representation Learning

2026-04-22

## Summary

The strongest paper direction is not "we combined several good tricks and beat baseline." That is not a sufficiently strong learning claim for NeurIPS. The better angle is a representation-learning story:

**Perception improvements do not transfer monotonically to planning. End-to-end driving needs a planning-critical scene representation, not just a stronger global perception representation.**

Our current experiments already support this:

- `ppdeform` improves planning over the stage-2 baseline.
- `ppdeform_planifls` further improves planning over `ppdeform`.
- `ptaux2d + ppdeform` improves planning strongly over `ppdeform`.
- `rot3dv2` improves detection and sometimes planning from the public pretrain.
- But the full stack `ptaux2d + ppdeform_planifls + rot3dv2` is **not additive on planning**: it improves detection, but regresses planning relative to `ptaux2d + ppdeform_planifls`.

This suggests the bottleneck is not simply feature quality. The real issue is whether the model learns to extract the subset of scene information that is relevant for ego planning.

## Core Thesis

The paper should argue:

**Planning requires a sparse, planning-sufficient representation of the scene. Detection salience is not the same as planning relevance.**

The current architecture already hints at this problem. Agent interaction is already bottlenecked through top-`num_det` selection, but that selection is driven by detection confidence rather than planning importance. A top-confidence object is not necessarily important for ego planning, while a lower-confidence pedestrian near the ego path may be critical.

This leads to the proposed paper direction:

**Learn a planning-critical downsampling of scene tokens or agents, conditioned on ego planning needs.**

## Proposed Method Direction

The most practical first version is:

### Learned planning-relevance agent selection

Instead of selecting agent tokens purely by detection confidence, learn a relevance score for each detected agent based on its value for ego planning.

Possible inputs to the relevance scorer:

- agent feature
- relative pose to ego
- overlap with the predicted ego corridor
- predicted agent future modes
- uncertainty
- map context
- current planning query or candidate ego future

Then use this learned score to:

- select a sparse top-k set of agents, or
- apply a soft weighting over agent tokens before planning interaction

This is the cleanest entry point because the codebase already uses top-k selection and planning-agent interaction blocks.

### Stronger extension

The more ambitious version is:

**Future-conditioned planning relevance**

Token or agent importance should depend on the candidate ego future. Different possible ego trajectories care about different surrounding agents and different image/map evidence. This would make the bottleneck dynamic rather than static.

That gives a stronger claim:

**The right representation for planning is not a global scene summary, but a future-conditioned sparse scene abstraction.**

## Why This Matches Existing Results

This direction explains the current non-additivity:

- `ptaux2d` improves global representation quality.
- `rot3dv2` improves geometric robustness and detection quality.
- `planifls` improves how planning consumes scene information.
- But stronger perception alone does not guarantee stronger planning.

The likely reason is that planning still needs the right subset of tokens, queries, or interactions. If improved perception is diluted into planning-irrelevant context, the planner may get better detection metrics without better ego decisions.

This makes the existing experiments useful as motivation:

- `aux2d` shows that representation quality can matter a lot for planning.
- `planifls` shows that the planning query interface matters a lot.
- `rot3dv2` shows that some perception gains transfer weakly or inconsistently to planning.

Together, these support a broader learning claim about selective transfer from perception to planning.

## Strong Candidate Paper Framing

The best high-level framing is:

**End-to-end planning is bottlenecked not by raw perception quality alone, but by whether the model learns a planning-critical representation of the scene.**

Equivalent title directions:

- `Perception Salience Is Not Planning Relevance`
- `Planning-Critical Scene Abstractions for End-to-End Driving`
- `Learning Sparse Planning-Relevant Representations for Autonomous Driving`
- `Future-Conditioned Token Selection for End-to-End Planning`

## Recommended Experimental Program

To turn this into a NeurIPS-quality paper, we need more than final planning numbers. The paper should include a controlled study of representation bottlenecks.

### Baselines

- all agents / current dense setup
- top-confidence top-k agents
- nearest-k agents by distance
- heuristic risk-based k (e.g. corridor overlap / TTC)
- learned planning-relevance top-k

### Core evaluation metrics

- `L2`
- `obj_box_col`
- `car_ade`, `ped_ade`
- `car_epa`, `ped_epa`
- `NDS`, `mAP`
- runtime / memory, if compression becomes part of the claim

### Important scenario slices

- turning scenes
- intersections
- dense traffic
- pedestrian-heavy scenes
- occluded scenes
- collision-heavy subsets

### Analysis needed for a strong paper

- how often learned selection differs from top-confidence selection
- whether selected agents are closer to the ego future path
- whether low-confidence but planning-critical agents are recovered
- robustness as k is reduced
- whether token selection explains why aux2d transfers better than rot3dv2 in planning

## What Would Make This Paper Strong

The method alone is not enough. The paper needs a clean learning claim and analysis.

The strongest possible claim is:

**A small learned subset of scene elements is sufficient, and often better, for ego planning than the full or confidence-ranked scene representation.**

That becomes especially compelling if we show:

1. A learned planning-relevance bottleneck improves planning over confidence-based selection.
2. The learned bottleneck preserves or improves performance with fewer tokens.
3. The selected tokens align with future conflict regions or ego-path interactions.
4. The bottleneck explains why some perception improvements transfer to planning and others do not.

## What Should Not Be The Main Story

These are useful supporting results, but not strong enough as the paper's main idea:

- aux2d improves SparseDrive
- rot3d fix improves augmentation
- the best config gets lower L2
- combining several tricks gives better results

Those belong in the motivation and baseline sections, not as the central claim.

## Immediate Recommendation

The best next step is to implement:

**A learned planning-relevance scorer for agent selection, replacing or augmenting top-confidence top-k selection.**

This is the shortest path to a paper-quality result because:

- it fits the current codebase
- it directly tests the new hypothesis
- it gives a clean comparison against existing heuristics
- it can later be extended to future-conditioned or token-level selection

## Bottom Line

The most promising NeurIPS paper direction is:

**Learn a sparse, planning-critical scene representation for driving, and show that planning relevance is different from detection salience.**

The current combined experiments already provide the motivation:

- stronger perception is helpful but not uniformly transferable to planning
- planning-query design matters
- the remaining bottleneck is likely selective representation learning for ego planning

That is a much stronger and more general paper story than another report about combining architecture improvements.
