# planinstfeat CR Regression Diagnosis — 2026-04-14

## Intro
`planpredtrajdeformmm_planinstfeat` achieves the best averaged L2=0.526 across the planning-head refinement batch, but its CR=0.104% is nearly 2× worse than `planpredtrajdeformmm` (0.047%). This batch diagnoses the root cause with two targeted ablations.

## Method

### Exp 1 — nomap_planpredtrajdeformmm_planinstfeat
Same as `planpredtrajdeformmm_planinstfeat` but with map supervision disabled (`with_map=False`). Tests whether the CR regression is an interaction between the map branch and the ego instance feature. Precedent: nomap variants in the prior batch were near-identical to map-supervised ones on planning metrics, so if CR recovers here the instfeat + map interaction is the culprit.

### Exp 2 — planpredtrajdeformmm_planinstfeat_laststage
Same as `planpredtrajdeformmm_planinstfeat` but the ego instance feature DAF path is applied only at the final decoder stage (`planning_deformable_instfeat_laststage=True`). In the earlier stages the regular `plan_mode_query` path is used instead. Tests whether injecting the ego feature in early stages suppresses obstacle-aware attention before it is established, causing the CR regression.

Requires a small code change to `motion_planning_head.py`: track the deformable stage count and gate instfeat on the last stage only.

## Results

| # | Config | Server | L2 | obj_box_col | car_ade | NDS | Status | Job ID |
|---|--------|--------|----|-------------|---------|-----|--------|--------|
| 1 | nomap_planpredtrajdeformmm_planinstfeat | DGX | — | — | — | — | pending | — |
| 1 | nomap_planpredtrajdeformmm_planinstfeat | Narval | — | — | — | — | pending | — |
| 2 | planpredtrajdeformmm_planinstfeat_laststage | DGX | — | — | — | — | pending | — |
| 2 | planpredtrajdeformmm_planinstfeat_laststage | Narval | — | — | — | — | pending | — |

## Discussion
_(filled at the end)_

## Future Work
_(filled at the end)_
