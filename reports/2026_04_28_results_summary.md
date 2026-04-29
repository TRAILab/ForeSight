# Results Summary for Paper Plan

Date: 2026-04-29

Scope: concise summary of recent results relevant to `reports/2026_04_28_paper_plan.md`.
Primary metrics are planning `L2` and `CR=obj_box_col` (lower is better).

## Core Planning Stack

Baseline: SparseDrive default stage-2 planning.

| Run | N | L2 | CR | car_ade | ped_ade | car_epa | ped_epa | NDS | Det mAP | Map mAP | Takeaway |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SparseDrive default | 1 | 0.636 | 0.133% | 0.636 | - | 0.492 | - | 0.523 | 0.413 | 0.553 | Baseline |
| `planpredtrajdeformmm` | 2 | 0.542 | 0.047% | 0.625 | 0.725 | 0.492 | 0.414 | 0.526 | 0.414 | 0.555 | Image deformable planning is positive |
| `planpredtrajdeformmm_planinstfeat_laststage` | 2 | 0.522 | 0.047% | 0.632 | 0.723 | 0.493 | 0.410 | 0.524 | 0.414 | 0.556 | Last-stage ego instance feature is positive |
| `planpredtrajdeformmm_planinstfeat_laststage_planwp` | 2 | 0.510 | 0.056% | 0.630 | 0.726 | 0.490 | 0.414 | 0.526 | 0.413 | 0.553 | Better L2, worse CR |
| `ptaux2d_ppdeformmm_planifls` | 2 | 0.509 | 0.049% | 0.614 | 0.714 | 0.506 | 0.435 | 0.546 | 0.441 | 0.561 | Best balanced recent stack |

## Perception K/V Interface

Baseline: Killarney same-host `ptaux2d_ppdeformmm_planifls` normal eval.

| Run | N | L2 | CR | car_ade | ped_ade | car_epa | ped_epa | NDS | Det mAP | Map mAP | Takeaway |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | 1 | 0.4988 | 0.063% | 0.6228 | 0.7201 | 0.5073 | 0.4302 | 0.5447 | 0.4403 | 0.5633 | Control |
| `alldet` | 1 | 0.5018 | 0.084% | 0.6031 | 0.7119 | 0.5090 | 0.4270 | 0.5477 | 0.4404 | - | More det tokens does not help planning |
| `topk_half` | 1 | 0.5046 | 0.060% | 0.6060 | 0.7216 | 0.5081 | 0.4346 | 0.5478 | 0.4446 | 0.5684 | Half K/V is essentially tied |
| `selrelGT` | 1 | 0.5115 | 0.089% | 0.6143 | 0.7192 | 0.5115 | 0.4342 | 0.5477 | 0.4431 | 0.5523 | GT-relevance selection regresses |
| `nodetmap` | 1 | 0.5215 | 0.092% | 0.5974 | 0.7201 | 0.5118 | 0.4297 | 0.5485 | 0.4422 | 0.5638 | Removing K/V is only a small regression here |
| `detrel` | 1 | 0.5393 | 0.105% | 0.6019 | 0.7335 | 0.5094 | 0.4282 | 0.5466 | 0.4397 | 0.5648 | Reshaped relevance path regresses |
| `bidir` | 1 | 0.5378 | 0.057% | 0.6069 | 0.7227 | 0.5134 | 0.4280 | 0.5442 | 0.4386 | - | Bidirectional coupling does not help |

## Laststage No-K/V Checks

Baseline: Killarney same-host `ptaux2d_ppdeformmm_planifls_laststage`.

| Run | N | L2 | CR | car_ade | ped_ade | car_epa | ped_epa | NDS | Det mAP | Map mAP | Takeaway |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | 1 | 0.5302 | 0.054% | 0.6194 | 0.7331 | 0.4942 | 0.4124 | 0.5248 | 0.4133 | 0.5580 | Control |
| `laststage_pts12` | 1 | 0.5270 | 0.071% | 0.6285 | 0.7263 | 0.4935 | 0.4202 | 0.5225 | 0.4102 | 0.5590 | More points does not solve CR |
| `laststage_nodetmap` | 1 | 0.5153 | 0.086% | 0.6235 | 0.7236 | 0.4900 | 0.4169 | 0.5230 | 0.4109 | 0.5596 | L2 improves, CR worsens |
| `laststage_nodetkv` | 1 | 0.5466 | 0.125% | 0.6171 | 0.7195 | 0.4914 | 0.4161 | 0.5255 | 0.4134 | 0.5526 | Removing det K/V (keep map) regresses |
| `laststage_nomapkv` | 1 | 0.5246 | 0.047% | 0.6239 | 0.7092 | 0.4943 | 0.4185 | 0.5262 | 0.4123 | 0.5549 | Removing map K/V (keep det) is roughly tied |

## Laststage Rescore Variants

Baseline: Killarney same-host `..._planinstfeat_laststage` with hard rescore.

| Run | N | L2 | CR | car_ade | ped_ade | car_epa | ped_epa | NDS | Det mAP | Map mAP | Takeaway |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline (hard rescore) | 1 | 0.5302 | 0.054% | 0.6194 | 0.7331 | 0.4942 | 0.4124 | 0.5248 | 0.4133 | 0.5580 | Control |
| `laststage_norescore` | 1 | 0.5188 | 0.104% | 0.6335 | 0.7189 | 0.4940 | 0.4122 | 0.5235 | 0.4131 | 0.5555 | Same pattern as planifls: rescore mostly helps CR |
| `laststage_motionconf` | 1 | 0.5184 | 0.084% | 0.6344 | 0.7182 | 0.4938 | 0.4113 | 0.5227 | 0.4129 | 0.5554 | Motion-conf rescore in-between hard and none |
| `laststage_nodetmap_norescore` | 1 | 0.5131 | 0.089% | 0.6209 | 0.7188 | 0.4890 | 0.4136 | 0.5231 | 0.4105 | 0.5597 | Stacking nodetmap+norescore does not amplify trade |

## Pretraining Signal and Stage 1 Alignment

Baseline: joint-detach stage-1 evaluation.

| Run | N | L2 | CR | car_ade | ped_ade | NDS | Det mAP | Map mAP | Takeaway |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `stage1_8gpu_noflash_joint_detach` | 1 | 0.6825 | 0.142% | - | - | 0.5216 | 0.4038 | - | Baseline joint-detach signal |
| `stage1_8gpu_noflash_joint` | 1 | 0.6428 | 0.104% | 0.6571 | 0.7286 | 0.5216 | 0.4051 | 0.5816 | Non-detach is better than detach, still not strong |
| `ptjointdetach_planpredtrajdeformmm` | 1 | 0.6669 | 0.089% | 0.6336 | 0.7322 | 0.5261 | 0.4092 | - | Modern stage-2 from detach regresses |
| `ptjointdetach` matched legacy | 1 | 0.6945 | 0.161% | 0.6460 | 0.7177 | 0.5231 | 0.4165 | - | Matched legacy also regresses |
| `ptjoint_planpredtrajdeformmm` | 1 | 0.6948 | 0.124% | 0.7429 | 0.7469 | 0.4411 | 0.3517 | 0.4785 | 0.3713 | 0.5318 | Modern-head detach joint stage-1 → modern stage-2 also fails. **Caveat:** stage-1 config still has `detach_perception=True`; not a true non-detach test. |

Note: Killarney job `3314785` is a separate modified planning-eval diagnostic on the same checkpoint, not a duplicate of the normal Apollo eval above.

## Map and Perception-Head Removal

Baseline: current best `planpredtrajdeformmm_planinstfeat_laststage`.

| Run | N | L2 | CR | car_ade | ped_ade | car_epa | ped_epa | NDS | Det mAP | Map mAP | Takeaway |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Current best | 2 | 0.522 | 0.047% | 0.632 | 0.723 | 0.493 | 0.410 | 0.524 | 0.414 | 0.556 | Baseline |
| Stage-1 no-map, stage-2 with map | 1 | 0.5203 | 0.048% | 0.6300 | 0.6849 | 0.5205 | 0.4561 | 0.5502 | 0.4471 | 0.2411 | Planning survives, map head breaks |
| Stage-1 with map, stage-2 no-map (`laststage_s2nomap`) | 1 | 0.5266 | 0.069% | 0.6210 | 0.7176 | 0.4979 | 0.4158 | 0.5286 | 0.4150 | - | Dropping map at stage-2 is essentially tied with laststage baseline |
| Both stages no-map | 1 | 6.612 | 3.605% | - | - | - | - | - | - | - | Catastrophic |

## Mode and Planner Architecture Negatives

Baseline: Killarney same-host `ptaux2d_ppdeformmm_planifls`.

| Run | N | L2 | CR | car_ade | ped_ade | car_epa | ped_epa | NDS | Det mAP | Map mAP | Takeaway |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | 1 | 0.4988 | 0.063% | 0.6228 | 0.7202 | 0.5073 | 0.4302 | 0.5447 | 0.4403 | 0.5633 | Control |
| `modes12` | 1 | 0.5278 | 0.082% | 0.6162 | 0.7119 | 0.5109 | 0.4402 | 0.5431 | 0.4408 | 0.5632 | More modes regress |
| `modes20diverse` | 1 | 0.5361 | 0.119% | 0.6085 | 0.7184 | 0.5085 | 0.4360 | 0.5469 | 0.4399 | 0.5574 | Diverse mode bag regresses |
| `noagg` | 1 | 0.5246 | 0.091% | 0.6144 | 0.7211 | 0.5122 | 0.4279 | 0.5427 | 0.4396 | 0.5510 | Removing aggregation regresses |
| `modetime36q` | 1 | 0.5578 | 0.099% | 0.6009 | 0.7121 | 0.5117 | 0.4338 | 0.5454 | 0.4403 | 0.5582 | Larger time-query setup regresses |
| `modesspeedstrat` | 1 | 0.5548 | 0.070% | 0.6102 | 0.7096 | 0.5108 | 0.4422 | 0.5494 | 0.4405 | 0.5647 | Speed-stratified modes regress |

## Rescoring and Safety

Baseline: hard-rescore `ptaux2d_ppdeformmm_planifls`.

| Run | N | L2 | CR | car_ade | ped_ade | car_epa | ped_epa | NDS | Det mAP | Map mAP | Takeaway |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Hard rescore baseline | 1 | 0.4988 | 0.063% | 0.6228 | 0.7201 | 0.5073 | 0.4302 | 0.5447 | 0.4403 | 0.5633 | Control |
| No rescore | 1 | 0.5008 | 0.106% | 0.6231 | 0.7196 | 0.5068 | 0.4297 | 0.5453 | 0.4399 | 0.5633 | Hard rescore mainly improves CR |
| `motionconf` rescore | 1 | 0.5036 | 0.089% | 0.6141 | 0.7193 | 0.5071 | 0.4363 | 0.5443 | 0.4403 | 0.5649 | Motion-conf rescore lands between hard and none on CR |
| `softcostcol` | 1 | 0.5515 | 0.050% | 0.6271 | 0.7344 | 0.4935 | 0.4111 | 0.5257 | 0.4142 | 0.5537 | Training loss did not help |
| `softcostcol_v2` | 1 | 0.5481 | 0.059% | 0.6359 | 0.7326 | 0.4880 | 0.4133 | 0.5266 | 0.4176 | 0.5566 | Geometry retry still null |
| Analytic soft rescore w3 | 1 | 0.6667 | 0.067% | - | - | - | - | - | - | - | L2 collapses |
| Analytic soft rescore w10 | 1 | 0.7205 | 0.072% | - | - | - | - | - | - | - | L2 collapses |
| Analytic soft rescore w30 | 1 | 0.7492 | 0.079% | - | - | - | - | - | - | - | L2 collapses |
| `planaux_conf_anchorlabel` | 1 | 0.7692 | 1.028% | 0.6003 | 0.7256 | 0.5169 | 0.4269 | 0.5433 | 0.4404 | 0.5607 | Learned scorer failed |
| `planaux_conf_evalmatch` | 1 | 0.5355 | 0.178% | 0.5966 | 0.7340 | 0.5082 | 0.4239 | 0.5472 | 0.4428 | 0.5600 | Eval-match recovers most of the regression but still worse than hard rescore |

## Run-to-Run Noise

Estimate: duplicated DGX/Narval pairs from the recent planning reports. Values below are absolute pair deltas; CR is in percentage points. Missing April 2 `ped_ade`/`ped_epa` values were pulled directly from DGX and Narval logs on 2026-04-28.

| Pair | Source | Hosts | L2 | CR pp | car_ade | ped_ade | car_epa | ped_epa | NDS | Det mAP | Map mAP |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `planrefine3` | 2026-04-02 | DGX/Narval | 0.0060 | 0.010 | 0.0224 | 0.0148 | 0.0009 | 0.0021 | 0.0024 | 0.0020 | 0.0034 |
| `plantrajdeform` | 2026-04-02 | DGX/Narval | 0.0027 | 0.014 | 0.0048 | 0.0067 | 0.0015 | 0.0009 | 0.0015 | 0.0015 | 0.0003 |
| `planpredtrajdeformmm` | 2026-04-02 | DGX/Narval | 0.0120 | 0.007 | 0.0045 | 0.0067 | 0.0022 | 0.0030 | 0.0019 | 0.0019 | 0.0045 |
| `nomap_planpredtrajdeform` | 2026-04-02 | DGX/Narval | 0.0102 | 0.024 | 0.0084 | 0.0122 | 0.0016 | 0.0051 | 0.0051 | 0.0024 | - |
| `nomap_planpredtrajdeformmm` | 2026-04-02 | DGX/Narval | 0.0020 | 0.009 | 0.0214 | 0.0124 | 0.0016 | 0.0066 | 0.0008 | 0.0031 | - |
| `planpredtrajdeformmm_modeproj` | 2026-04-02 | DGX/Narval | 0.0119 | 0.039 | 0.0025 | 0.0071 | 0.0009 | 0.0003 | 0.0020 | 0.0043 | 0.0046 |
| `nomap_planpredtrajdeformmm_modeproj` | 2026-04-02 | DGX/Narval | 0.0058 | 0.011 | 0.0062 | 0.0029 | 0.0007 | 0.0007 | 0.0007 | 0.0010 | - |
| `planpredtrajdeformmm_planinstfeat` | 2026-04-02 | DGX/Narval | 0.0093 | 0.018 | 0.0039 | 0.0150 | 0.0065 | 0.0058 | 0.0023 | 0.0007 | 0.0018 |
| `planpredtrajdeformmm_planinstfeat_laststage` | 2026-04-02 | DGX/Narval | 0.0046 | 0.017 | 0.0046 | 0.0063 | 0.0039 | 0.0015 | 0.0015 | 0.0007 | 0.0014 |
| `planpredtrajdeformmm_planinstfeat_laststage_planwp` | 2026-04-02 | DGX/Narval | 0.0071 | 0.005 | 0.0057 | 0.0028 | 0.0082 | 0.0008 | 0.0037 | 0.0013 | 0.0067 |
| `stage2_ptaux2d_planpredtrajdeformmm` | 2026-04-21 ref | DGX/Narval | 0.0020 | 0.025 | - | - | 0.0010 | - | 0.0040 | 0.0050 | 0.0050 |
| `ppdeformmm_planifls_rot3dv2` | 2026-04-21 | DGX/Narval | 0.0047 | 0.004 | 0.0038 | 0.0189 | 0.0024 | 0.0078 | 0.0015 | 0.0023 | 0.0083 |
| `ptaux2d_ppdeformmm_planifls` | 2026-04-21 | DGX/Narval | 0.0065 | 0.012 | 0.0010 | 0.0071 | 0.0015 | 0.0021 | 0.0043 | 0.0052 | 0.0083 |
| `ptaux2p5d_ppdeformmm_planifls` | 2026-04-21 | DGX/Narval | 0.0131 | 0.009 | 0.0055 | 0.0136 | 0.0041 | 0.0077 | 0.0049 | 0.0043 | 0.0091 |

| Metric | Delta N | Mean abs delta | Median abs delta | Std dev of abs deltas | Max abs delta |
|---|---:|---:|---:|---:|---:|
| L2 | 14 | 0.0070 | 0.0062 | 0.0038 | 0.0131 |
| CR | 14 | 0.015 pp | 0.012 pp | 0.010 pp | 0.039 pp |
| car_ade | 13 | 0.0073 | 0.0048 | 0.0067 | 0.0224 |
| ped_ade | 13 | 0.0097 | 0.0071 | 0.0050 | 0.0189 |
| car_epa | 14 | 0.0026 | 0.0018 | 0.0023 | 0.0080 |
| ped_epa | 13 | 0.0034 | 0.0021 | 0.0028 | 0.0078 |
| NDS | 14 | 0.0026 | 0.0021 | 0.0015 | 0.0051 |
| Det mAP | 14 | 0.0026 | 0.0021 | 0.0016 | 0.0052 |
| Map mAP | 11 | 0.0049 | 0.0046 | 0.0030 | 0.0091 |

Relative deltas use `abs(delta) / pair_mean`.

| Metric | Delta N | Mean rel delta | Median rel delta | Std dev rel deltas | Max rel delta |
|---|---:|---:|---:|---:|---:|
| L2 | 14 | 1.31% | 1.17% | 0.71% | 2.54% |
| CR | 14 | 23.94% | 17.57% | 14.89% | 54.55% |
| car_ade | 13 | 1.12% | 0.73% | 1.00% | 3.37% |
| ped_ade | 13 | 1.34% | 0.99% | 0.70% | 2.68% |
| car_epa | 14 | 0.54% | 0.35% | 0.47% | 1.63% |
| ped_epa | 13 | 0.81% | 0.53% | 0.66% | 1.83% |
| NDS | 14 | 0.49% | 0.41% | 0.27% | 0.97% |
| Det mAP | 14 | 0.60% | 0.52% | 0.36% | 1.18% |
| Map mAP | 11 | 0.87% | 0.83% | 0.54% | 1.63% |

Practical threshold: single-run L2 differences below about `0.007` / `1.3%` and CR differences below about `0.015 pp` / `24% relative` should be treated as noise unless supported by matched-host controls or repeated runs. CR's relative percentage is large because the absolute CR values are very small.

## Confirmed Findings

- The planner does not appear bottlenecked by the perception decoder K/V interface. Token count, token selection, all-det tokens, relevance reshaping, and bidirectional coupling are null or regressive.
- Effects near `~0.007 L2` or `~0.015 pp CR` should be treated as noise. This makes `numdemapx2`, `topk_half`, and small same-host deltas weak/null evidence, not actionable wins.
- The robust positive path is image-feature access for planning: endpoint deformable attention, last-stage ego instance features, and aux2d pretraining give the strongest planning stack.
- Stronger perception pretraining does not automatically improve planning. Richer rot3d/2.5d/dnrot variants improve perception metrics but are neutral or worse for L2/CR.
- Removing K/V on the laststage stack improves L2 but worsens CR by more than the estimated noise floor. This is the sharpest current evidence that the K/V path is not needed for L2, while safety/collision behavior still needs care.
- Direct joint stage-1 planning supervision is not a confirmed win. Non-detach is better than detach, but the stage-2 transfers from joint-detach remain clearly worse than the main planning stack.
- Map evidence is nuanced: planning can survive stage-1 no-map pretraining if stage 2 restores map supervision, but the map head itself breaks and both-stage no-map is catastrophic.
- Hard collision rescore is still important. Removing it mostly hurts CR, while analytic soft rescore and learned scorer replacements failed. The new `motionconf` rescore lands between hard and none and is not a clear win.
- Disaggregating laststage K/V removal: dropping det K/V alone (`laststage_nodetkv`) regresses, while dropping map K/V alone (`laststage_nomapkv`) is essentially tied. Combining both (`laststage_nodetmap`) improves L2 but worsens CR. The map-side K/V is the dispensable one.
- Stage-2 nomap on top of normal stage-1 (`laststage_s2nomap`) is roughly tied with laststage baseline — dropping map supervision at stage-2 alone is not catastrophic, unlike dropping it at both stages.
- The "Arm B" joint stage-1 → modern stage-2 (`ptjoint_planpredtrajdeformmm`) regresses further than the legacy-head detach version on both planning and perception. **Caveat:** the Arm B stage-1 config still has `detach_perception=True`; the difference vs Arm A is the head architecture, not the gradient flow. The actual non-detach hypothesis (`detach_perception=False`) has not been tested yet.
- The learned scorer with eval-match (`planaux_conf_evalmatch`) recovers most of the catastrophic regression of `planaux_conf_anchorlabel` (L2 0.77 → 0.54) but is still worse than the hard-rescore baseline (L2 0.499); the matched-baseline eval setup matters but does not flip the result.
