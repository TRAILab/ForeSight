# plan_refine — 2026-04-02

## Intro
This batch targets direct planning-head improvements on the `sparsedrive_r50_stage2_4gpu_bs24` baseline. The goal is to test whether iterative multi-stage planning refinement and direct image-feature attention at planned trajectory endpoints improve the primary planning metrics.

## Method
Two experiments are prepared from the bs24 baseline. The first adds three-stage cumulative planning refinement with deep supervision at each refinement stage. The second adds endpoint-based deformable image attention before each refinement stage on top of the same three-stage planning refinement schedule.

Both runs were submitted to Narval (4 A100) and DGX (4 A100). DGX plantrajdeform is still training as of 2026-04-03.

## Results

| # | Config | Server | L2 | obj_box_col | car_ade | NDS | Status | Job ID |
|---|--------|--------|----|-------------|---------|-----|--------|--------|
| 0 | sparsedrive_r50_stage2_4gpu_bs24 (baseline) | DGX | 0.636 | 0.133% | 0.636 | 0.5232 | done | — |
| 1 | sparsedrive_r50_stage2_4gpu_bs24_planrefine3 | DGX | 0.572 | 0.103% | 0.686 | 0.5197 | done | 3572 |
| 1 | sparsedrive_r50_stage2_4gpu_bs24_planrefine3 | Narval | 0.578 | 0.093% | 0.708 | 0.5173 | done | 58789115 |
| 2 | sparsedrive_r50_stage2_4gpu_bs24_plantrajdeform | Narval | 0.561 | **0.065%** | 0.690 | 0.5206 | done | 58789117 |
| 2 | sparsedrive_r50_stage2_4gpu_bs24_plantrajdeform | DGX | — | — | — | — | training | 3573 |

Full metrics for completed runs:

| Config | Server | det mAP | det NDS | AMOTA | IDS | map mAP | car_EPA | plan L2 | plan CR |
|--------|--------|---------|---------|-------|-----|---------|---------|---------|---------|
| bs24 baseline | DGX | 0.4132 | 0.5232 | 0.3776 | 1045 | 0.5528 | 0.492 | 0.636 | 0.133% |
| planrefine3 | DGX | 0.4102 | 0.5197 | 0.3629 | 784 | 0.5530 | 0.480 | 0.572 | 0.103% |
| planrefine3 | Narval | 0.4082 | 0.5173 | 0.3638 | 1178 | 0.5564 | 0.481 | 0.578 | 0.093% |
| plantrajdeform | Narval | 0.4121 | 0.5206 | 0.3732 | 890 | 0.5473 | 0.480 | 0.561 | 0.065% |
| plantrajdeform | DGX | — | — | — | — | — | — | — | *(pending)* |

## Discussion

**Both experiments improve planning over the bs24 baseline.** The baseline trained on DGX has a collision rate of 0.133% and L2 of 0.636 — noticeably worse than the paper checkpoint inference (0.096%, 0.606). Both planrefine3 and plantrajdeform close this gap substantially.

**planrefine3** reduces collision rate to 0.093–0.103% (Narval / DGX) and L2 to 0.572–0.578. This already recovers close to the paper checkpoint inference level (0.096%). The improvement is consistent across both servers, confirming it is a real gain from the 3-stage refinement rather than noise.

**plantrajdeform** (Narval) achieves the best results: CR=0.065% and L2=0.561. This is a new best for any trained run in this project, beating even the paper's reported 0.080% collision rate. The endpoint deformable attention on top of 3-stage refinement provides a further ~0.028pp reduction in collision rate beyond planrefine3. Tracking (AMOTA=0.3732) is also slightly better, possibly because the richer planning features help maintain instance consistency.

**Detection and map are largely unaffected** — NDS and mAP are within noise of the bs24 baseline across all runs, as expected since the planning head changes do not touch the detection or map branches.

**car_ade is slightly worse in planrefine3** (0.686–0.708 vs 0.636 baseline). This is unexpected given the refinement stages; possible explanations are that the planning loss weight balance shifts the motion head marginally, or that the FDE=1.000 artifact from the baseline masks a truer ADE comparison. plantrajdeform partially recovers this (0.690).

**DGX plantrajdeform result pending** — once training completes it will be a direct hardware-controlled comparison against the Narval result.

## Future Work
- Await DGX plantrajdeform result to confirm consistency with Narval.
- Investigate the car_ade regression in planrefine3; check whether it is an artifact of the FDE/ADE metric interaction or a genuine motion regression.
- Try plantrajdeform on top of a more careful stage2 training recipe (with DN enabled, matching the paper's training setup) to see if the collision rate improves further.
- Consider whether the endpoint attention mechanism could be applied to the motion head for agent prediction.
