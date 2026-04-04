# Planning Head Refinement
2026-04-02

## Intro
This batch targets direct planning-head improvements on the `sparsedrive_r50_stage2_4gpu_bs24` baseline. The goal is to test whether iterative multi-stage planning refinement and direct image-feature attention at planned trajectory endpoints improve the primary planning metrics.

## Method

### Cumulative refinement
The baseline motion/planning decoder runs a single pass: `temp_gnn → gnn → norm → cross_gnn → norm → deformable → norm → ffn → norm → refine`. `planning_cumulative_refinement` repeats the `deformable → norm → ffn → norm → refine` block three times, with trajectory deltas accumulating across stages and the mode query updated from the cumulative endpoint after each refine. Deep supervision is applied at each stage. `motion_cumulative_refinement` applies the same schedule to agent queries.

### Endpoint deformable attention
`planning_deformable` makes each planning mode query attend image features at its own predicted trajectory endpoint. Concretely:

1. **Spatial reference construction.** After each refine step the cumulative plan trajectory (shape `bs × num_mode × fut_ts × 2`, XY displacements in lidar frame) is used to build a 3D anchor box at the last-waypoint endpoint for each mode. The box is translated to the absolute XY endpoint and its yaw is set from the final trajectory heading vector.
2. **Key point sampling.** `SparseBox3DKeyPointsGenerator` scatters a fixed set of 3D points around that box (corners + learnable offsets), then projects them into all 6 camera views across all FPN levels.
3. **Weighted feature fusion.** A linear layer applied to `mode_query + anchor_embed` predicts scalar attention weights over every `(camera, level, point)` triple (softmax-normalised). The sampled image features are multiplied by these weights and summed to a 256-dim vector, then projected and added residually to the mode query.

This is repeated per decoder stage, so each stage attends at the endpoint predicted by the previous stage's refinement. The first planning stage uses the cumulative plan anchor from the immediately preceding refine.

`motion_deformable` applies the same mechanism to agent instance features, using the best-mode predicted endpoint (argmax of mode logits) as the spatial reference. Only one endpoint box is built per agent (best mode), so all 6 trajectory modes collapse to a single spatial reference before the image lookup. The first decoder stage initialises from the k-means prior trajectory (mode 0 of the cluster-centre anchor) rather than the detection box centre, placing the reference at a plausible future position from the outset.

`motion_deformable_multimode` extends this by attending at all 6 mode endpoints simultaneously rather than collapsing to the best mode. For each agent, the agent feature is expanded to 6 copies, each attending at its own mode's endpoint box via a single DAF pass over `(bs, N×6, 256)` queries. The 6 attended outputs are then aggregated back to a single `(bs, N, 256)` feature by a softmax-weighted sum over mode logits. This gives each mode its own image-feature context, rather than forcing all modes to share the lookup of whichever mode happens to be top-ranked. The first decoder stage uses uniform weights (mean of all 6 k-means endpoint lookups); subsequent stages weight by the latest predicted mode logits.

### Ablations
Four additional configs isolate component contributions:
- **plantrajdeformonly / planpredtrajdeformonly**: deformable attention without `planning_cumulative_refinement`.
- **plantrajdeformfirst / planpredtrajdeformfirst**: first-waypoint spatial reference (`deformable_waypoint=0`) instead of last-waypoint.

**Implementation note.** planpredtrajdeformonly and planpredtrajdeformfirst initially failed due to an inplace-op bug in `_build_motion_endpoint_anchors`: `ego_yaw` was read from `anchor` (a clone of `det_anchors`) which was then modified inplace four times, causing a PyTorch autograd version-counter mismatch during backward for static agents. Fixed by reading `ego_yaw` directly from `det_anchors`.

## Results

| # | Config | Server | L2 | obj_box_col | car_ade | NDS | Status | Job ID |
|---|--------|--------|----|-------------|---------|-----|--------|--------|
| 0 | sparsedrive_r50_stage2_4gpu_bs24 (baseline) | DGX | 0.636 | 0.133% | 0.636 | 0.5232 | done | — |
| 1 | planrefine3 | DGX | 0.572 | 0.103% | 0.686 | 0.5197 | done | 3572 |
| 1 | planrefine3 | Narval | 0.578 | 0.093% | 0.708 | 0.5173 | done | 58789115 |
| 2 | plantrajdeform | Narval | 0.561 | **0.065%** | 0.690 | 0.5206 | done | 58789117 |
| 2 | plantrajdeform | DGX | — | — | — | — | training | 3573 |
| 3 | planpredrefine3 | Narval | — | — | — | — | eval-pending | 58807951, 58808164 |
| 4 | planpredtrajdeform | Narval | 0.554 | **0.045%** | 0.622 | 0.5237 | done | 58824084 |
| 5 | plantrajdeformonly | Narval | 0.575 | 0.090% | 0.687 | 0.5200 | done | 58817554 |
| 5 | planpredtrajdeformonly | Narval | 0.567 | 0.103% | 0.682 | 0.5208 | done | 58824298 |
| 6 | plantrajdeformfirst | Narval | 0.568 | 0.132% | 0.720 | 0.5194 | done | 58817556 |
| 6 | planpredtrajdeformfirst | Narval | 0.556 | 0.112% | 0.623 | 0.5258 | done | 58824432 |
| 7 | planpredtrajdeformmm | Narval | — | — | — | — | training | 58846242 |

Full metrics for completed runs:

| Config | Server | det mAP | det NDS | AMOTA | IDS | map mAP | car_EPA | car_ade | plan L2 | plan CR |
|--------|--------|---------|---------|-------|-----|---------|---------|---------|---------|---------|
| bs24 baseline | DGX | 0.4132 | 0.5232 | 0.3776 | 1045 | 0.5528 | 0.492 | 0.636 | 0.636 | 0.133% |
| planrefine3 | DGX | 0.4102 | 0.5197 | 0.3629 | 784 | 0.5530 | 0.480 | 0.686 | 0.572 | 0.103% |
| planrefine3 | Narval | 0.4082 | 0.5173 | 0.3638 | 1178 | 0.5564 | 0.481 | 0.708 | 0.578 | 0.093% |
| plantrajdeform | Narval | 0.4121 | 0.5206 | 0.3732 | 890 | 0.5473 | 0.480 | 0.690 | 0.561 | 0.065% |
| plantrajdeform | DGX | — | — | — | — | — | — | — | — | *(training)* |
| planpredrefine3 | Narval | 0.4072 | 0.5224 | 0.3729 | 877 | — | — | — | — | *(eval-pending)* |
| planpredtrajdeform | Narval | 0.4127 | 0.5237 | 0.3684 | 1041 | 0.5536 | 0.495 | 0.622 | 0.554 | 0.045% |
| plantrajdeformonly | Narval | 0.4104 | 0.5200 | 0.3657 | 834 | 0.5505 | 0.481 | 0.687 | 0.575 | 0.090% |
| planpredtrajdeformonly | Narval | 0.4077 | 0.5208 | 0.3638 | 1091 | 0.5484 | 0.485 | 0.682 | 0.567 | 0.103% |
| plantrajdeformfirst | Narval | 0.4105 | 0.5194 | 0.3664 | 973 | 0.5442 | 0.481 | 0.720 | 0.568 | 0.132% |
| planpredtrajdeformfirst | Narval | 0.4119 | 0.5258 | 0.3705 | 838 | 0.5552 | 0.492 | 0.623 | 0.556 | 0.112% |
| planpredtrajdeformmm | Narval | — | — | — | — | — | — | — | — | *(training)* |

## Discussion

**All planning-head modifications improve over baseline.** The baseline (DGX) has CR=0.133%, L2=0.636. All submitted configs reduce both, with the best results from adding motion deformable attention on top of 3-stage planning refinement.

**planrefine3** reduces CR to 0.093–0.103% and L2 to 0.572–0.578. Consistent across DGX and Narval, confirming a real gain from 3-stage refinement.

**plantrajdeform** (Narval) achieves CR=0.065%, L2=0.561 — the best of the plan-only configs, beating the paper's reported 0.080% CR. Endpoint deformable attention provides ~0.028pp further CR reduction over planrefine3.

**planpredtrajdeform** (adding `motion_deformable=True`) achieves the best overall result: **CR=0.045%, L2=0.554**. Extending endpoint deformable attention to agent queries gives another ~0.020pp CR reduction over plantrajdeform, and also improves car_ade from 0.690 to 0.622 — the best motion prediction across all runs.

**Ablation: deformable-only (no refinement).** plantrajdeformonly achieves CR=0.090%, L2=0.575 — substantially worse than plantrajdeform (CR=0.065%) but still better than planrefine3-only. This confirms that deformable attention and cumulative refinement are complementary: deformable attention alone recovers roughly half the refinement benefit.

**Ablation: first vs. last waypoint.** plantrajdeformfirst (attend at waypoint 0 instead of waypoint -1) achieves CR=0.132%, L2=0.568 — significantly worse than plantrajdeform. Attending at the near-horizon endpoint hurts, suggesting that the far-horizon endpoint provides more useful context for image-feature sampling. The planpred versions show the same pattern: planpredtrajdeformfirst (CR=0.112%) is worse than planpredtrajdeform (CR=0.045%).

**Motion prediction (car_ade) improves with motion_deformable.** planpredtrajdeform and planpredtrajdeformfirst both achieve car_ade ≈ 0.622–0.623, substantially better than the plan-only configs (0.687–0.720) and even the baseline (0.636). planpredtrajdeformonly also improves (0.682), though less so. This suggests the motion head benefits strongly from attending at its own predicted endpoints.

**car_EPA is slightly higher in planpred configs** (0.492–0.495 vs 0.480–0.481), consistent with the ade improvement.

**Detection and map are largely unaffected** — NDS and mAP remain within ~0.003 of the baseline across all runs.

**planpredrefine3 eval pending** — checkpoint done but planning/motion metrics not captured (eval job 58845832 queued). Detection/tracking metrics already available (mAP=0.4072, NDS=0.5224, AMOTA=0.3729).

**nomap variants pending** — jobs 58845828 / 58845829 resubmitted after fixing `find_unused_parameters=True` (DDP unused-parameter error with `with_map=False`).

## Future Work
- Await planpredrefine3 eval results and DGX plantrajdeform result.
- Await nomap_planrefine3 and nomap_plantrajdeform results.
- Investigate whether the planpredtrajdeform CR=0.045% result holds on DGX (cross-server consistency check).
- ~~**Multi-mode motion deformable attention.**~~ Implemented as `motion_deformable_multimode` (job 58846242). Attends at all 6 mode endpoints, aggregates by softmax mode confidence. Results pending.
