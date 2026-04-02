# plan_refine — 2026-04-02

## Intro
This batch targets direct planning-head improvements on the `sparsedrive_r50_stage2_4gpu_bs24` baseline. The goal is to test whether iterative multi-stage planning refinement and direct image-feature attention at planned trajectory endpoints improve the primary planning metrics.

## Method
Two experiments are prepared from the bs24 baseline. The first adds three-stage cumulative planning refinement with deep supervision at each refinement stage. The second adds endpoint-based deformable image attention before each refinement stage on top of the same three-stage planning refinement schedule.

## Results

| # | Config | L2 | obj_box_col | car_ade | NDS | Status | Notes | Job ID |
|---|--------|----|-------------|---------|-----|--------|-------|--------|
| 1 | sparsedrive_r50_stage2_4gpu_bs24_planrefine3 | — | — | — | — | submitted | bs24 baseline + 3-stage cumulative planning refinement | narval: 58789115; dgx: 3572 |
| 2 | sparsedrive_r50_stage2_4gpu_bs24_plantrajdeform | — | — | — | — | submitted | planrefine3 + endpoint deformable image attention | narval: 58789117; dgx: 3573 |

## Discussion
_(filled at the end)_

## Future Work
_(filled at the end)_
