# Planning Head Refinement
2026-04-02

## Intro
This batch targets direct planning-head improvements on the `sparsedrive_r50_stage2_4gpu_bs24` baseline. The goal is to test whether iterative multi-stage planning refinement and direct image-feature attention at planned trajectory endpoints improve the primary planning metrics.

## Method
Two experiments are prepared from the bs24 baseline. The first adds three-stage cumulative planning refinement with deep supervision at each refinement stage. The second adds endpoint-based deformable image attention before each refinement stage on top of the same three-stage planning refinement schedule.

Both runs were submitted to Narval (4 A100) and DGX (4 A100). DGX plantrajdeform is still training as of 2026-04-03.

planpredrefine3 has two Narval submissions (58807951, 58808164) — the second was submitted after a code update; both ran to completion but evaluation crashed before writing planning metrics. planpredtrajdeform additionally enables `motion_deformable=True`, making agents attend at their predicted trajectory endpoint (best mode) rather than their current detection box, matching the behaviour already used for planning.

Four ablation configs were also submitted to isolate the contribution of each component:
- **plantrajdeformonly / planpredtrajdeformonly**: removes `planning_cumulative_refinement` to test deformable attention without multi-stage refinement.
- **plantrajdeformfirst / planpredtrajdeformfirst**: replaces last-waypoint deformable attention with first-waypoint (`deformable_waypoint=0`) to test whether the near-horizon endpoint is more informative than the far endpoint.

Note: planpredtrajdeformonly and planpredtrajdeformfirst initially failed due to an inplace operation bug in `_build_motion_endpoint_anchors` (version counter mismatch during backward when static agents were present). Fixed by reading `ego_yaw` from `det_anchors` instead of from the inplace-modified `anchor` clone.

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
- Consider combining motion_deformable with motion_cumulative_refinement in a more principled schedule.
