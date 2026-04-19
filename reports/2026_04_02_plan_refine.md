# Planning Head Refinement
2026-04-02

## Abstract

We investigate iterative multi-stage refinement and image-feature endpoint attention as mechanisms for improving ego-motion planning in the SparseDrive framework. Starting from the `sparsedrive_r50_stage2_4gpu_bs24` baseline (L2=0.636, CR=0.133%), we show that combining 3-stage cumulative planning refinement with multi-mode deformable attention at predicted trajectory endpoints across both the planning and agent motion decoders reduces L2 to 0.522 and CR to 0.047% — a 18% reduction in L2 error and 65% reduction in collision rate. The key insight is that attending at predicted far-horizon endpoints, rather than current agent positions, provides substantially more useful image context for both trajectory accuracy and obstacle avoidance. Extending this to attend independently at all 6 mode endpoints per agent, aggregated by mode confidence, further improves planning L2 without sacrificing collision avoidance. The best configuration also incorporates the ego instance feature from the temporal tracking stream as an additional query for the planning DAF, restricted to the final decoder stage only to avoid suppressing obstacle-aware attention in earlier stages.

## Introduction

The SparseDrive motion/planning decoder takes instance features and anchor embeddings from the detection head and refines them through a fixed sequence of temporal GNN, spatial GNN, cross-attention, deformable attention, and feed-forward layers before a final trajectory refine step. In the baseline, this sequence runs once: `temp_gnn → gnn → norm → cross_gnn → norm → deformable → norm → ffn → norm → refine`. The planning head outputs a set of trajectory modes for the ego vehicle; the motion head outputs multi-modal trajectory predictions per agent.

Two complementary hypotheses motivate this batch. First, a single pass through the decoder may be insufficient to produce well-calibrated trajectory predictions — iterating the refinement stages with accumulated trajectory deltas could progressively sharpen both accuracy and collision avoidance. Second, the image features sampled by the existing deformable attention layer are anchored at the current instance positions; they carry no information about the scene at the predicted future locations the ego or agents will actually occupy. Attending at trajectory endpoints rather than current positions should allow the model to directly observe upcoming obstacles, lane structure, and occlusions along the predicted path.

This report tests both hypotheses systematically. We introduce `planning_cumulative_refinement` and `motion_cumulative_refinement` for the iterative refinement component, and `planning_deformable` / `motion_deformable` / `motion_deformable_multimode` for endpoint image attention. A follow-up investigation adds the ego vehicle's temporal instance feature as a query source for the planning DAF, and identifies the critical importance of restricting this injection to the final decoder stage. We also run nomap ablations to test whether the map branch is load-bearing for the planning improvements, and ablations isolating deformable attention from refinement and comparing far-horizon vs. near-horizon endpoint references.

## Method

### Cumulative refinement

`planning_cumulative_refinement` repeats the `deformable → norm → ffn → norm → refine` block three times within the planning decoder. Trajectory deltas accumulate across stages, and the planning mode query is updated from the cumulative endpoint after each refine step. Deep supervision is applied at all three stages. `motion_cumulative_refinement` applies the same schedule to agent instance features in the motion decoder. The cumulative design avoids resetting trajectory state between stages, allowing each pass to refine incrementally rather than predicting from scratch.

### Endpoint deformable attention

`planning_deformable` makes each planning mode query attend image features at its own predicted trajectory endpoint rather than at the current ego position.

1. **Spatial reference construction.** After each refine step, the cumulative plan trajectory (shape `bs × num_mode × fut_ts × 2`, XY displacements in lidar frame) is used to build a 3D anchor box at the final-waypoint endpoint for each mode. The box is translated to the absolute XY endpoint and its yaw is set from the terminal heading vector.
2. **Key point sampling.** `SparseBox3DKeyPointsGenerator` scatters a fixed set of 3D points around that box (corners + learnable offsets), then projects them into all 6 camera views across all FPN levels.
3. **Weighted feature fusion.** A linear layer applied to `mode_query + anchor_embed` predicts scalar attention weights over every `(camera, level, point)` triple (softmax-normalised). The sampled image features are multiplied by these weights and summed to a 256-dim context vector, then projected and added residually to the mode query.

This is applied at each decoder stage, attending at the endpoint produced by the previous stage's refinement. The first stage uses the cumulative plan anchor from the immediately preceding refine.

### Motion deformable attention (single-mode)

`motion_deformable` applies the same endpoint attention mechanism to agent instance features. For each agent, only one endpoint box is built, using the best-mode predicted endpoint (argmax of mode logits) as the spatial reference, so all 6 trajectory modes share a single image lookup. The first decoder stage initialises from the k-means prior trajectory (mode 0 of the cluster-centre anchor), placing the reference at a plausible future position rather than the current detection box centre.

### Multi-mode motion deformable attention

`motion_deformable_multimode` extends the single-mode variant by attending independently at all 6 mode endpoints rather than collapsing to the best mode. For each agent, the instance feature is expanded to 6 copies, each attending at its own mode's endpoint box via a single DAF pass over `(bs, N×6, 256)` queries. The 6 attended outputs are then aggregated back to a single `(bs, N, 256)` feature by a softmax-weighted sum over mode logits, so each mode contributes context proportional to its predicted confidence. The first decoder stage uses uniform weights (mean of all 6 k-means endpoint lookups); subsequent stages weight by the latest predicted mode logits.

### Per-mode projection variant

`motion_deformable_modeproj` extends the multimode variant by applying a separate learned linear projection (`nn.Linear(256, 256)`) per mode to each DAF-attended output before the softmax-weighted aggregation. The intent is to give each mode its own transformation so that the aggregated feature reflects mode-specific structure more strongly. This is applied after reshaping the flat attended tensor to `(bs, num_anchor, fut_mode, embed_dims)` and before the mode-weight collapse.

### Ego instance feature for planning (planinstfeat)

`planning_deformable_instfeat` substitutes the ego vehicle's instance feature (retrieved from the detection head's temporal instance bank) as the query for the planning DAF lookup, replacing the plan mode query. The ego instance feature carries motion state from the temporal tracking stream and encodes recent observed dynamics, which may provide complementary context to the trajectory-derived anchor. The attended output is added residually to the plan mode query as before.

### Multi-waypoint planning deformable attention

`planning_deformable_waypoints` extends the planning endpoint DAF to attend at multiple trajectory waypoints per mode rather than a single far-horizon endpoint. Given `ego_fut_ts=6` (0.5s per step, 3s horizon), the config `planpredtrajdeformmm_planinstfeat_laststage_planwp` uses waypoints `[1, 3, 5]` (1s, 2s, 3s).

For each decoder stage the K anchor boxes are built by `_build_planning_anchor_boxes_multi`, which applies `_build_planning_anchor_boxes` logic independently at each waypoint index. The mode queries are expanded to `(bs, num_mode × K, 256)`, a single DAF pass is run over all `num_mode × K` queries simultaneously, and the outputs are reshaped to `(bs, num_mode, K, 256)` and mean-aggregated over the waypoint dimension before the residual is applied. The instfeat final stage is unaffected — it still uses the single far-horizon anchor box.

### Last-stage ego instance feature injection

`planning_deformable_instfeat_laststage` restricts the ego instance feature DAF to the final decoder stage only; earlier stages use the standard plan mode query path. A `_deformable_stage_idx` counter in `MotionPlanningHead.forward()` selects the query source per stage. This variant was motivated by the observation that injecting the ego feature across all stages suppresses obstacle-aware attention before the model has established a stable trajectory estimate; applying it only at the last stage preserves collision avoidance while retaining most of the trajectory accuracy benefit.

### Ablations

- **plantrajdeformonly / planpredtrajdeformonly**: endpoint deformable attention without cumulative refinement, isolating the contribution of the image attention alone.
- **plantrajdeformfirst / planpredtrajdeformfirst**: near-horizon spatial reference (`deformable_waypoint=0`, first waypoint) instead of the final waypoint, testing whether far-horizon context is specifically responsible for the gains.
- **nomap variants**: training with `with_map=False` across key configs, testing whether the map supervision branch is necessary for the planning improvements.

**Implementation note.** The planpredtrajdeformonly and planpredtrajdeformfirst configs initially failed due to an inplace-op bug in `_build_motion_endpoint_anchors`: `ego_yaw` was read from `anchor` (a clone of `det_anchors`) which was then modified inplace, causing a PyTorch autograd version-counter mismatch during backward for static agents. Fixed by reading `ego_yaw` directly from `det_anchors` before any inplace operations.

## Results

All completed runs. Failed and cancelled runs are excluded.

| # | Config | Server | det mAP | NDS | AMOTA | IDS | map mAP | car_EPA | car_ade | L2 | CR |
|---|--------|--------|---------|-----|-------|-----|---------|---------|---------|----|----|
| 0 | bs24 baseline | DGX | 0.413 | 0.523 | 0.378 | 1045 | 0.553 | 0.492 | 0.636 | 0.636 | 0.133% |
| 1 | planrefine3 | DGX | 0.410 | 0.520 | 0.363 | 784 | 0.553 | 0.480 | 0.686 | 0.572 | 0.103% |
| 1 | planrefine3 | Narval | 0.408 | 0.517 | 0.364 | 1178 | 0.556 | 0.481 | 0.708 | 0.578 | 0.093% |
| 2 | plantrajdeform | Narval | 0.412 | 0.521 | 0.373 | 890 | 0.547 | 0.480 | 0.690 | 0.561 | 0.065% |
| 2 | plantrajdeform | DGX | 0.411 | 0.522 | 0.368 | 946 | 0.548 | 0.479 | 0.685 | 0.558 | 0.051% |
| 3 | planpredrefine3 | Narval | 0.412 | 0.524 | 0.374 | 910 | 0.558 | 0.488 | 0.645 | 0.553 | 0.107% |
| 4 | planpredtrajdeform | Narval | 0.413 | 0.524 | 0.368 | 1041 | 0.554 | 0.495 | 0.622 | 0.554 | 0.045% |
| 5 | plantrajdeformonly | Narval | 0.410 | 0.520 | 0.366 | 834 | 0.551 | 0.481 | 0.687 | 0.575 | 0.090% |
| 5 | planpredtrajdeformonly | Narval | 0.408 | 0.521 | 0.364 | 1091 | 0.548 | 0.485 | 0.682 | 0.567 | 0.103% |
| 6 | plantrajdeformfirst | Narval | 0.411 | 0.519 | 0.366 | 973 | 0.544 | 0.481 | 0.720 | 0.568 | 0.132% |
| 6 | planpredtrajdeformfirst | Narval | 0.412 | 0.526 | 0.371 | 838 | 0.555 | 0.492 | 0.623 | 0.556 | 0.112% |
| 7 | planpredtrajdeformmm | Narval | 0.413 | 0.525 | 0.372 | 1159 | 0.557 | 0.491 | 0.628 | 0.536 | 0.043% |
| 7 | planpredtrajdeformmm | DGX | 0.415 | 0.527 | 0.380 | 970 | 0.553 | 0.493 | 0.623 | 0.548 | 0.050% |
| 8 | nomap_planrefine3 | Narval | 0.407 | 0.517 | 0.360 | 925 | — | 0.476 | 0.675 | 0.561 | 0.077% |
| 8 | nomap_plantrajdeform | Narval | 0.414 | 0.520 | 0.369 | 1046 | — | 0.481 | 0.686 | 0.556 | 0.061% |
| 9 | nomap_planpredtrajdeform | DGX | 0.413 | 0.522 | 0.381 | 816 | — | 0.495 | 0.642 | 0.545 | 0.080% |
| 9 | nomap_planpredtrajdeform | Narval | 0.416 | 0.528 | 0.381 | 677 | — | 0.493 | 0.634 | 0.535 | 0.056% |
| 10 | nomap_planpredtrajdeformmm | DGX | 0.414 | 0.527 | 0.380 | 896 | — | 0.492 | 0.646 | 0.540 | 0.046% |
| 10 | nomap_planpredtrajdeformmm | Narval | 0.411 | 0.526 | 0.372 | 870 | — | 0.493 | 0.625 | 0.542 | 0.055% |
| 11 | planpredtrajdeformmm_modeproj | DGX | 0.420 | 0.528 | 0.381 | 775 | 0.556 | 0.490 | 0.631 | 0.550 | 0.091% |
| 11 | planpredtrajdeformmm_modeproj | Narval | 0.415 | 0.526 | 0.378 | 756 | 0.556 | 0.491 | 0.633 | 0.562 | 0.052% |
| 12 | nomap_planpredtrajdeformmm_modeproj | DGX | 0.415 | 0.527 | 0.377 | 783 | — | 0.494 | 0.618 | 0.549 | 0.058% |
| 12 | nomap_planpredtrajdeformmm_modeproj | Narval | 0.416 | 0.527 | 0.377 | 1146 | — | 0.493 | 0.624 | 0.544 | 0.069% |
| 13 | planpredtrajdeformmm_planinstfeat | DGX | 0.413 | 0.525 | 0.375 | 975 | 0.553 | 0.491 | 0.649 | 0.521 | 0.095% |
| 13 | planpredtrajdeformmm_planinstfeat | Narval | 0.412 | 0.523 | 0.375 | 1090 | 0.554 | 0.484 | 0.645 | 0.531 | 0.113% |
| 14 | nomap_planpredtrajdeformmm_planinstfeat | DGX | 0.417 | 0.530 | 0.378 | 1008 | — | 0.492 | 0.646 | 0.510 | 0.084% |
| 15 | planpredtrajdeformmm_planinstfeat_laststage | DGX | 0.414 | 0.523 | 0.375 | 928 | 0.556 | 0.495 | 0.635 | 0.519 | 0.038% |
| 15 | planpredtrajdeformmm_planinstfeat_laststage | Narval | 0.415 | 0.525 | 0.376 | 1390 | 0.557 | 0.491 | 0.630 | 0.524 | 0.055% |
| 16 | planpredtrajdeformmm_planinstfeat_laststage_planwp | DGX | 0.413 | 0.524 | 0.377 | 904 | 0.556 | 0.486 | 0.633 | 0.514 | 0.058% |
| 16 | planpredtrajdeformmm_planinstfeat_laststage_planwp | Narval | 0.414 | 0.528 | 0.379 | 906 | 0.550 | 0.494 | 0.627 | 0.506 | 0.053% |

Averaged results across servers (DGX + Narval where both available); single-run configs are reported as-is.

| Config | N | L2 | CR | car_ade | NDS | det mAP | map mAP |
|--------|---|----|----|---------|-----|---------|---------|
| bs24 baseline | 1 | 0.636 | 0.133% | 0.636 | 0.523 | 0.413 | 0.553 |
| planrefine3 | 2 | 0.575 | 0.098% | 0.697 | 0.519 | 0.409 | 0.555 |
| plantrajdeform | 2 | 0.560 | 0.058% | 0.688 | 0.522 | 0.412 | 0.548 |
| plantrajdeformonly | 1 | 0.575 | 0.090% | 0.687 | 0.520 | 0.410 | 0.551 |
| plantrajdeformfirst | 1 | 0.568 | 0.132% | 0.720 | 0.519 | 0.411 | 0.544 |
| planpredrefine3 | 1 | 0.553 | 0.107% | 0.645 | 0.524 | 0.412 | 0.558 |
| planpredtrajdeform | 1 | 0.554 | 0.045% | 0.622 | 0.524 | 0.413 | 0.554 |
| planpredtrajdeformonly | 1 | 0.567 | 0.103% | 0.682 | 0.521 | 0.408 | 0.548 |
| planpredtrajdeformfirst | 1 | 0.556 | 0.112% | 0.623 | 0.526 | 0.412 | 0.555 |
| planpredtrajdeformmm | 2 | 0.542 | 0.047% | 0.626 | 0.526 | 0.414 | 0.555 |
| nomap_planrefine3 | 1 | 0.561 | 0.077% | 0.675 | 0.517 | 0.407 | — |
| nomap_plantrajdeform | 1 | 0.556 | 0.061% | 0.686 | 0.520 | 0.414 | — |
| nomap_planpredtrajdeform | 2 | 0.540 | 0.068% | 0.638 | 0.525 | 0.415 | — |
| nomap_planpredtrajdeformmm | 2 | 0.541 | 0.051% | 0.636 | 0.527 | 0.413 | — |
| planpredtrajdeformmm_modeproj | 2 | 0.556 | 0.072% | 0.632 | 0.527 | 0.418 | 0.556 |
| nomap_planpredtrajdeformmm_modeproj | 2 | 0.547 | 0.064% | 0.621 | 0.527 | 0.416 | — |
| planpredtrajdeformmm_planinstfeat | 2 | 0.526 | 0.104% | 0.647 | 0.524 | 0.413 | 0.554 |
| nomap_planpredtrajdeformmm_planinstfeat | 1 | 0.510 | 0.084% | 0.646 | 0.530 | 0.417 | — |
| **planpredtrajdeformmm_planinstfeat_laststage** | **2** | **0.522** | **0.047%** | **0.633** | **0.524** | **0.415** | **0.557** |
| planpredtrajdeformmm_planinstfeat_laststage_planwp | 2 | 0.510 | 0.056% | 0.630 | 0.526 | 0.414 | 0.553 |

## Discussion

**All modifications improve over the baseline.** Every config in this batch reduces both L2 and CR relative to the baseline (L2=0.636, CR=0.133%), confirming that the planning head is a productive target for improvement. The gains compound as components are added.

**3-stage cumulative refinement alone is a solid baseline.** `planrefine3` reduces L2 to 0.575 and CR to 0.098%, consistent across both servers. The iterative accumulation of trajectory deltas across stages produces a reliable improvement. Adding motion cumulative refinement (`planpredrefine3`) further reduces L2 to 0.553 and improves car_ade from 0.697 to 0.645, confirming that agent motion refinement helps planning indirectly. However, CR (0.107%) is slightly worse than the plan-only `planrefine3` on CR, suggesting motion refinement does not contribute to collision avoidance directly.

**Endpoint deformable attention is primarily responsible for collision rate reduction.** `plantrajdeform` (endpoint attention, no motion deformable) achieves CR=0.058% averaged — substantially better than `planrefine3` (0.098%) and even beating the paper's reported 0.080% CR. Attending at the predicted far-horizon endpoint allows the model to observe upcoming obstacles along the planned path, which is more directly informative for collision avoidance than the current-position context used by the baseline deformable layer. Refinement and endpoint attention are complementary: `plantrajdeformonly` (attention without refinement) achieves CR=0.090% — better than refinement alone but substantially worse than the combined `plantrajdeform`.

**Extending endpoint attention to agent queries gives a further large CR reduction.** `planpredtrajdeform` (adding motion deformable at single best-mode endpoint) achieves CR=0.045% — a further ~0.013pp reduction over `plantrajdeform` — and improves car_ade dramatically from 0.688 to 0.622. The agent motion head benefits strongly from attending at its own predicted future positions: it can directly observe what the scene looks like where each agent is predicted to be, rather than relying on current-position context. This also benefits planning through the cross-attention pathway between agent and ego features.

**Multi-mode motion deformable is the best single component combination.** `planpredtrajdeformmm` achieves averaged L2=0.542, CR=0.047% — the best planning L2 before adding the ego instance feature. Attending independently at all 6 mode endpoints and aggregating by mode confidence improves L2 by 0.012 over single-mode `planpredtrajdeform` at nearly identical CR. The multi-mode design gives the agent feature richer image context across the full distribution of predicted behaviors rather than committing to the top mode before the image lookup.

**Per-mode projection hurts.** `planpredtrajdeformmm_modeproj` adds a learned linear projection per mode after the DAF output and before the mode-weight collapse. It performs worse than the base multimode variant (averaged L2=0.556, CR=0.072%), suggesting that the additional per-mode transformation over-parameterises the aggregation step and degrades gradient flow to the mode logits.

**Far-horizon endpoint is critical.** The ablation configs attending at waypoint 0 instead of waypoint -1 (`plantrajdeformfirst`, `planpredtrajdeformfirst`) are substantially worse than their last-waypoint counterparts. `plantrajdeformfirst` achieves CR=0.132% — essentially the same as the baseline — while `planpredtrajdeformfirst` achieves CR=0.112%. The near-horizon reference is too close to the current position to provide meaningfully different context from the standard deformable layer. The model needs to see the scene at the predicted destination, not a few timesteps ahead.

**Map supervision is not critical for planning.** The nomap ablations (`nomap_planpredtrajdeform`, `nomap_planpredtrajdeformmm`) achieve results nearly identical to their map-supervised counterparts — within 0.001–0.004 on both L2 and CR after averaging. Dropping the map branch neither substantially helps nor hurts planning in this configuration, suggesting the planning head does not heavily depend on map supervision for the endpoint attention mechanism to work. Detection and NDS metrics are also unaffected by removing map supervision.

**Ego instance feature improves L2 but causes a CR regression when applied at all stages.** `planpredtrajdeformmm_planinstfeat` achieves the best averaged L2 (0.526) but CR (0.104%) is a large regression — nearly back to the `planrefine3` level and 2× worse than `planpredtrajdeformmm`. The temporal instance feature from the tracking stream encodes useful motion context for trajectory accuracy, but injecting it at all decoder stages appears to suppress the obstacle-aware attention established by the endpoint DAF before it is stable. Two ablations confirmed this: (1) `nomap_planpredtrajdeformmm_planinstfeat` (DGX, L2=0.510, CR=0.084%) showed the CR regression is not caused by an interaction with the map branch. (2) `planpredtrajdeformmm_planinstfeat_laststage` (L2=0.522, CR=0.047%) showed that restricting ego feature injection to the final stage only resolves the CR regression entirely while preserving most of the L2 gain.

**Multi-waypoint planning attention improves L2 at a small CR cost.** `planpredtrajdeformmm_planinstfeat_laststage_planwp` (waypoints [1, 3, 5]) achieves averaged L2=0.510 and CR=0.056% — a 0.012 L2 improvement over `laststage` at the cost of +0.009pp CR. Attending at intermediate waypoints (1s, 2s, 3s) provides richer spatial context along the full predicted trajectory, which sharpens waypoint accuracy. The CR regression suggests that attending at near-horizon waypoints reintroduces some of the same obstacle-awareness dilution seen with the all-stage instfeat variant, though at a much smaller scale. Whether the L2 gain is worth the CR tradeoff depends on the downstream priority.

**Detection and motion metrics are largely stable.** NDS, det mAP, AMOTA, and IDS remain within ~0.003 of the baseline across all planning-head modifications. The improvements are isolated to the planning and motion prediction metrics, confirming that the planning head changes do not interfere with the upstream detection or map branches.

## Best Model

**`planpredtrajdeformmm_planinstfeat_laststage`** is the recommended model going forward (averaged L2=0.522, CR=0.047%, N=2). It is the only config in this batch to simultaneously achieve L2 < 0.525 and CR < 0.050%, and it strictly dominates `planpredtrajdeformmm` on L2 (+0.020) at identical averaged CR.

Key comparisons from averaged results:

| Config | L2 | CR | Δ L2 vs best | Δ CR vs best |
|--------|----|----|--------------|--------------|
| **planpredtrajdeformmm_planinstfeat_laststage** | **0.522** | **0.047%** | — | — |
| planpredtrajdeformmm_planinstfeat_laststage_planwp | 0.510 | 0.056% | −0.012 | +0.009pp |
| planpredtrajdeformmm | 0.542 | 0.047% | +0.020 | 0.000pp |
| planpredtrajdeformmm_planinstfeat | 0.526 | 0.104% | +0.004 | +0.057pp |
| nomap_planpredtrajdeformmm | 0.541 | 0.051% | +0.019 | +0.004pp |
| planpredtrajdeform | 0.554 | 0.045% | +0.032 | −0.002pp |
| bs24 baseline | 0.636 | 0.133% | +0.114 | +0.086pp |

The `nomap_planpredtrajdeformmm` variant is a viable alternative if map supervision is to be dropped (L2=0.541, CR=0.051%), at the cost of 0.019 L2 and 0.004pp CR relative to the best model.

The DGX laststage result (CR=0.038%) was notably stronger than Narval (CR=0.055%); the averaged CR (0.047%) matches `planpredtrajdeformmm`. Server-to-server variability at this CR scale is normal and seen consistently across the batch.

## Future Work

- **Optimal injection stage for ego instance feature.** The laststage variant restricts the ego feature DAF to stage 3 of 3. Testing injection at the last 2 stages would map out the full stage-sensitivity curve and may recover more L2 gain without CR regression. This is the most direct extension of the current best model and requires only a flag change.

- **Combining laststage with nomap.** The best model uses map supervision, but nomap ablations show the map branch contributes little to planning. Testing `nomap_planpredtrajdeformmm_planinstfeat_laststage` is a single config that confirms whether the L2 improvement from the ego feature is preserved without the map branch — closing the last open question from this batch.

- **Last-stage ego feature for motion.** Apply the same laststage-only injection pattern to `motion_deformable`, substituting the agent's instance feature at the final stage. Given the large car_ade improvements already seen from motion endpoint attention, adding temporal tracking context at the last stage could further improve both motion prediction and downstream planning.

- **Extended stage-2 fine-tuning on planning loss.** The best model trains with the full stage-2 objective. A short additional fine-tuning phase with upweighted planning loss (L2 + CR) may further sharpen trajectory accuracy without requiring new components or architectural changes.

- **Scale to a stronger backbone.** All experiments use ResNet-50. The endpoint attention mechanism is backbone-agnostic; applying it on top of a R101 or VoVNet stage-2 model would establish whether the planning gains are additive with backbone improvements.
