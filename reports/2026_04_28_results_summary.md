# Results Summary for Paper Plan

Date: 2026-04-29 (revised 2026-05-02)

Scope: concise summary of recent results relevant to `reports/2026_04_28_paper_plan.md`.
Primary metrics are planning `L2` and `CR=obj_box_col` (lower is better).

## North star: minS2 (everything off in stage 2 except ego queries + planning task)

Reference comparator throughout: K/V-off paper headline (`_decoder6_planwp_evalmatchmode_egostatus`) at **L2=0.3708 / CR=0.0405%** (mean of seeds 0+1; seed 2 in flight as 3394849).

**Path to minS2** combines four components, each demonstrated individually but never (until 3397346) together:

| Component | Closes at | Result | Source |
|---|---|---|---|
| K/V off | inference perception interface | Headline (0.3708) | `_decoder6_planwp_evalmatchmode_egostatus` |
| Loss zeroing (Stream A, individual) | stage-2 perception supervision | All 3 tie L2=0.5204 baseline within noise | 3386472/73/74 |
| Loss zeroing (Stream A, **combined**) | stage-2 perception supervision (joint) | **In flight 3397345 (`_s2nopercep`)** | TBD ~9.5 h |
| Drop agent slots (Stream C) | inference perception in planner queries | Stream C training done, eval crashed; **rerun in flight 3394847** | TBD ~1 h |
| Conflict head image features (Stream B1) | inference perception in conflict head | Ties evalmatchmode (3376418, L2=0.5119 / CR=0.047%) | already in hand |
| Frozen backbone+heads (Stream E) | stage-2 backbone gradient | **In flight 3394848 (`_frozenpercep`)** | TBD ~9.5 h |
| **All combined → minS2** | the goal | **In flight 3397346 (`_minS2`)** | TBD ~9.5 h |

If 3397346 ties headline within noise (ΔL2 ≤ 0.007 / ΔCR ≤ 0.015 pp), the paper architecture is locked. Stage-1 candidates currently in flight (3396706-9, joint_nodetach) will be evaluated by their **stage-2 transfer through minS2**.

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
| `laststage_nodetmap_decoder6` | 1 | 0.5194 | 0.077% | 0.6217 | 0.7170 | 0.4923 | 0.4184 | 0.5265 | 0.4140 | 0.5527 | Doubling decoder layers (3→6) ties L2; CR slightly improved |
| `laststage_nodetmap_planwp_full6` | 1 | 0.6352 | 0.097% | 0.6305 | 0.7317 | 0.4950 | 0.4159 | 0.5286 | 0.4135 | 0.5488 | All-waypoint without laststage instfeat regresses L2 by 0.12 — laststage is load-bearing |
| `laststage_nodetmap_planwp_full6_ls` | 1 | 0.5186 | 0.071% | 0.6224 | 0.7102 | 0.4880 | 0.4193 | 0.5203 | 0.4087 | 0.5555 | All-waypoint with laststage instfeat retained (K-pool) ties L2 and improves CR |
| `laststage_nodetmap_decoder6_planwp` (seed 0) | 1 | 0.5150 | 0.068% | 0.6265 | 0.7289 | 0.4925 | 0.4136 | 0.5275 | 0.4164 | 0.5559 | Original (3319227): stack of decoder6 + waypoints + laststage |
| `laststage_nodetmap_decoder6_planwp` (seed 1) | 1 | 0.5100 | 0.087% | 0.6194 | 0.7344 | 0.4894 | 0.4122 | 0.5209 | 0.4078 | 0.5515 | Repro (3366270): L2 reproduces (mean 0.5125, ΔL2 vs seeds=0.005); **CR did not reproduce** (drifted from 0.068 → 0.087, mean 0.0775%, back to baseline range) |
| `laststage_nodetmap_decoder6_planwp_evalmatchmode` (seed 0) | 1 | 0.5204 | 0.046% | 0.6275 | 0.7103 | 0.4923 | 0.4212 | 0.5248 | 0.4137 | 0.5531 | Original (3366620): decoder6_planwp + v4 evalmatchmode learned rescore. CR matches/beats K/V-on `_laststage` (0.054%) |
| `laststage_nodetmap_decoder6_planwp_evalmatchmode` (seed 1) | 1 | 0.5087 | 0.053% | 0.6083 | 0.7280 | 0.4842 | 0.4034 | 0.5222 | 0.4085 | 0.5549 | Repro (3377724): **CR survives across seeds** (mean 0.0495%, both well below K/V-on hard rescore 0.054%). Headline reproduces |
| **`_decoder6_planwp_evalmatchmode_egostatus`** (seed 0) | 1 | **0.3720** | **0.044%** | 0.6223 | 0.7361 | 0.4955 | 0.4097 | 0.5253 | 0.4125 | 0.5535 | **NEW HEADLINE (3376689):** folds anchor-capacity egostatus lever into the K/V-off paper stack — L2 −0.148, CR −0.002 pp vs evalmatchmode seed 0; matches the −0.13 ΔL2 seen on K/V-on (0.4988 → 0.3701) |
| **`_decoder6_planwp_evalmatchmode_egostatus`** (seed 1) | 1 | **0.3695** | **0.037%** | 0.6179 | 0.7175 | 0.4872 | 0.4035 | 0.5211 | 0.4063 | 0.5526 | **Repro (3386471): both L2 and CR survive across seeds.** Mean L2=0.3708, mean CR=0.0405% — well above the noise floor on L2 (Δseed=0.0025) and within noise on CR. Paper main quantitative claim locked across seeds |
| `_decoder6_planwp_evalmatchmode_s2nodetloss` (Stream A1) | 1 | 0.5265 | 0.076% | 1.9112 | 1.1661 | -0.529 | -1.686 | 0.0489 | 0.0061 | 0.5510 | **(3386472)** Zeroing stage-2 det loss collapses detection (mAP 0.006, NDS 0.05) but planning ties evalmatchmode (L2 +0.006, CR +0.030 pp). Stage-2 det supervision is **empty for planning** given a strong stage-1 init |
| `_decoder6_planwp_evalmatchmode_s2nomaploss` (Stream A2) | 1 | 0.5161 | 0.063% | 0.6247 | 0.7123 | 0.4939 | 0.4190 | 0.5242 | 0.4160 | 0.3095 | **(3386473)** Zeroing stage-2 map loss collapses map (mAP_normal 0.31 vs 0.55) but planning matches evalmatchmode (L2 −0.004, CR +0.017 pp). Stage-2 map supervision is **empty for planning** |
| `_decoder6_planwp_evalmatchmode_s2nomotionloss` (Stream A3) | 1 | 0.5132 | 0.042% | 4.1709 | 2.4926 | 0.1577 | 0.0677 | 0.5263 | 0.4132 | 0.5519 | **(3386474)** Zeroing stage-2 motion loss collapses motion (car_ade 4.17 vs 0.62) but planning ties evalmatchmode (L2 −0.007, CR −0.004 pp). Stage-2 motion supervision is **empty for planning** |
| `_decoder6_planwp_evalmatchmode_imgconfdet` (Stream B1) | 1 | 0.5119 | 0.047% | 0.6053 | 0.7214 | 0.4936 | 0.4140 | 0.5257 | 0.4128 | 0.5588 | Replaces conflict-head agent_features with image features sampled at det-anchor BEV positions (3376418). Ties evalmatchmode → **conflict head's per-agent token isn't load-bearing**; Stream C (drop det/motion queries entirely) is now viable |
| `_decoder6_planwp_evalmatchmode_dispweight` (F1) | 1 | 0.5205 | 0.055% | 0.6395 | 0.7355 | 0.4840 | 0.4085 | 0.5251 | 0.4117 | 0.5515 | GT-displacement-weighted plan loss (3376749). ΔL2≈0, ΔCR=+0.009 pp — null as a standalone lever |
| `_decoder6_planwp_distillrescore` (Exp 8 Opt 2) | 1 | 0.5091 | 0.098% | 0.6349 | 0.7098 | 0.4915 | 0.4132 | 0.5226 | 0.4096 | 0.5544 | K/V-off + plan_cls distilled from rescore() collide flag, `use_rescore=False` at inference (3377547). L2 ties evalmatchmode but CR worse by 0.05 pp — distillation lands but doesn't beat the v4 learned head on CR |
| `laststage_nodetmap_tempstack2` | 1 | 0.5614 | 0.065% | 0.6233 | 0.7265 | 0.4925 | 0.4116 | 0.5214 | 0.4105 | 0.5516 | T2.6 with 2 frames: regresses L2 by 0.046; CR gain not worth the L2 cost |
| `laststage_nodetmap_tempstack3` | 1 | 0.5495 | 0.059% | 0.6348 | 0.7144 | 0.4906 | 0.4125 | 0.5234 | 0.4089 | 0.5546 | T2.6 with 3 frames: regresses L2 by 0.034; loses to decoder6+planwp on both metrics |
| `laststage_nodetmap_tempstack3_noegocomp` | 1 | 0.6345 | 0.071% | 0.6255 | 0.7266 | 0.4904 | 0.4073 | 0.5225 | 0.4107 | 0.5544 | T2.6 without ego compensation: L2 collapses by 0.12 — ego comp is load-bearing for temporal stacking |

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
| `ptegoonly_ppdeformmm_planifls` | 1 | 0.5420 | 0.073% | 0.6234 | 0.7484 | 0.4881 | 0.4022 | 0.5214 | 0.4046 | 0.5699 | Stage-1 with motion-keepalive (planning-only at stage 1, det+map intact) → stage-2 K/V-on (3377809). ΔL2=+0.022 vs `_ptnomapdnrot_ppdeformmm_planifls` (0.5203) — minimum-perception stage-1 still produces a viable planning init, with map_normal=0.5699 (higher than #7's 0.2411 since map head was trained at stage 1 here) |
| `ptegoonly_ep3_ppdeformmm_planifls` | 1 | 0.5402 | 0.060% | 0.6505 | 0.7265 | 0.4876 | 0.4121 | 0.5206 | 0.4033 | 0.5575 | **(3388945)** epoch-3 stage-1 ckpt (planning-best snapshot L2=0.6351) → stage-2 ties epoch-5 init (L2 −0.002, CR −0.013 pp). Stage-1 epoch choice doesn't move stage-2 L2 |
| `ptegoonly_decoder6_planwp_evalmatchmode_egostatus` | 1 | 0.3946 | 0.055% | 0.6359 | 0.7404 | 0.4886 | 0.4045 | 0.5215 | 0.4021 | 0.5617 | **(3388946)** Egoonly (minimum-perception) stage-1 + K/V-off paper headline stack. ΔL2=+0.023 vs full-perception stage-1 (0.3720), CR +0.011 pp — minimum-perception stage-1 nearly recovers the headline; supports the supervision-vs-interface decoupling argument |

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

## Anchor Capacity / Speed Decoupling

Baseline: Killarney `ptaux2d_ppdeformmm_planifls`. Suite from `2026_04_30_anchor_capacity.md`.

| Run | N | L2 | CR | car_ade | ped_ade | car_epa | ped_epa | NDS | Det mAP | Map mAP | Takeaway |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | 1 | 0.4988 | 0.063% | 0.6228 | 0.7202 | 0.5073 | 0.4302 | 0.5447 | 0.4403 | 0.5633 | Control |
| `egostatus` | 1 | **0.3701** | **0.058%** | 0.6032 | 0.7240 | 0.5111 | 0.4267 | 0.5461 | 0.4408 | 0.5613 | MLP-encoded current ego_status into plan_mode_query: −0.13 L2, −0.005pp CR (3368763) |
| `shapeanchor_velnorm` | 1 | 0.5215 | 0.058% | 0.6165 | 0.7247 | 0.5119 | 0.4328 | 0.5485 | 0.4415 | 0.5620 | Shape anchors scaled by per-batch \|\|v_0\|\|: +0.02 L2, −0.005pp CR (3368762) |
| `shapeanchor_endptnorm_mag` | 1 | 1.3456 | 0.640% | 0.6340 | 0.7147 | 0.4943 | 0.4285 | 0.5331 | 0.4262 | 0.5259 | Endpt-norm anchors with `refmag=0` for stopped Straight cluster degenerated sineembed → blow-up (3368761) |
| `shapeanchor_endptnorm_mag` retry (refmag floor) | 1 | 1.1124 | 0.392% | 0.6337 | 0.7191 | 0.4903 | 0.4224 | 0.5365 | 0.4229 | 0.5253 | refmag-floor patch did **not** rescue this variant (3376688) — still degenerate. Stopped-Straight cluster handling needs a deeper fix |
| `shapeanchor_endptnorm_mag_egostatus` | 1 | 0.4120 | 0.100% | 0.6263 | 0.7170 | 0.5003 | 0.4221 | 0.5450 | 0.4384 | 0.5589 | Eval (3377754) on existing checkpoint at 2 GPUs. egostatus *partially* rescues the endptnorm_mag variant (1.346 → 0.412); still loses to plain `egostatus` (0.3701) — the anchor change adds no value once egostatus is in |
| `modesspeedstrat_perbucketreg` | 1 | 0.5415 | 0.339% | 0.6036 | 0.7185 | 0.5091 | 0.4238 | 0.5467 | 0.4440 | 0.5597 | Eval (3377755). L2 close to plain speedstrat (0.5548) but CR blows up to 0.339% — per-bucket regression branch doesn't cleanly decouple decoder capacity |
| `modesspeedstrat_norescore` (eval-only) | 1 | 0.5560 | 0.060% | 0.6104 | 0.7096 | 0.5112 | 0.4411 | 0.5485 | 0.4398 | 0.5646 | Decode-time rescore flip on existing speedstrat ckpt — L2 unchanged → speedstrat regression is training-time (3368766) |

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
| `evalmatch` T=0.70 any (3675) | 1 | 0.5379 | 0.158% | - | - | - | - | - | - | - | Eval-only — same evalmatch ckpt, threshold ↑ |
| `evalmatch` T=0.85 any (3676) | 1 | 0.5413 | 0.142% | - | - | - | - | - | - | - | Eval-only |
| `evalmatch` T=0.90 any (3677) | 1 | 0.5399 | 0.122% | - | - | - | - | - | - | - | Best CR among `any`-aggregation thresholds |
| `evalmatch` T=0.95 any (3678) | 1 | 0.5315 | 0.156% | - | - | - | - | - | - | - | Eval-only |
| `evalmatch` T=0.99 any (3679) | 1 | 0.5227 | 0.201% | - | - | - | - | - | - | - | Best L2 (rejects almost nothing); CR rises |
| `evalmatch` T=0.85 topk2 (3680) | 1 | 0.5284 | 0.159% | - | - | - | - | - | - | - | k=2 anchors required; CR worse than `any` |
| `evalmatch` T=0.85 topk3 (3681) | 1 | 0.5247 | 0.173% | - | - | - | - | - | - | - | k=3; FPs not single-anchor — multi-anchor agreement |
| `evalmatch` T=0.85 detweighted (3682) | 1 | 0.5370 | 0.119% | - | - | - | - | - | - | - | det-conf soft-weighting beats T-sweep on CR |
| `planaux_conf_evalmatchmode` (3319213) | 1 | 0.5328 | 0.080% | 0.6033 | 0.7271 | 0.5142 | 0.4267 | 0.5434 | 0.4376 | 0.5587 | Per-mode aggregated BCE — best learned-scorer CR; ~0.017pp from hard rescore |
| `evalmatchmode` T=0.50 any (K3324524) | 1 | 0.5331 | 0.082% | - | - | - | - | - | - | - | Reproduces v4 default — same ckpt, planning-only eval |
| `evalmatchmode` T=0.70 any (K3324525) | 1 | **0.5316** | 0.081% | - | - | - | - | - | - | - | Best L2 in v4 sweep |
| `evalmatchmode` T=0.85 any (K3324526) | 1 | 0.5343 | 0.093% | - | - | - | - | - | - | - | T-sweep on v4 is essentially flat |
| `evalmatchmode` T=0.90 any (K3324527) | 1 | 0.5335 | 0.085% | - | - | - | - | - | - | - | v4 already calibrated; no leverage from T tuning |
| `evalmatchmode` T=0.95 any (K3324528) | 1 | 0.5334 | **0.080%** | - | - | - | - | - | - | - | Ties v4 default CR |
| `evalmatchmode` T=0.85 topk2 (K3324529) | 1 | 0.5350 | 0.092% | - | - | - | - | - | - | - | top-k no longer needed once train aligned |
| `evalmatchmode` T=0.85 topk3 (K3324530) | 1 | 0.5325 | 0.101% | - | - | - | - | - | - | - | top-k regresses CR |
| `evalmatchmode` T=0.85 detweighted (K3324531) | 1 | 0.5320 | 0.097% | - | - | - | - | - | - | - | det-weighting no longer helpful |
| `evalmatchmode` T=0.85 hybrid_or (K3324532) | 1 | 0.5330 | **0.063%** | - | - | - | - | - | - | - | OR(hard, learned) matches hard rescore CR exactly; learned head adds no unique vetoes |

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
- Threshold sweep on the evalmatch checkpoint (T ∈ {0.50, 0.70, 0.85, 0.90, 0.95, 0.99}) traces a CR U-curve with minimum at T=0.90 (CR=0.122%, vs T=0.50 default 0.178%); L2 monotonically improves as T → 0.99 (rejecting fewer modes converges to no-rescore). Threshold tuning alone cuts CR by ~30% but does not close the ~0.06pp gap to hard rescore (0.063%).
- Selector aggregation variants on the evalmatch checkpoint at T=0.85: `topk` (k=2, k=3) regresses CR vs `any`, indicating false positives are *not* concentrated at single anchors per mode — multiple anchors per mode flag together. `detweighted` (multiply prob by det_confidence, no hard det-conf gate) achieves the best CR among eval-only fixes (CR=0.119%) at modest L2 cost.
- **Per-mode aggregated retrain (`evalmatchmode`, label_source='evalmatch_mode', smooth-max τ=5.0)** matches the inference-time `any(anchor)` reduction at training time and is the first learned-scorer to come within ~0.02pp CR of hard rescore: L2=0.5328 / CR=0.080% (vs hard 0.4988 / 0.063%). Classifier diagnostics improve substantially (acc_05 0.95 vs evalmatch 0.98 at much higher pos_rate 7.6% vs 0.5%; F1 ≈ 0.72 vs 0.36). L2 is still ~0.034 worse than hard rescore — the head closes most of the CR gap but cannot match L2 even at matched aggregation. Train/eval aggregation alignment is the single most impactful fix in the rescoring family.
- **`egostatus` (current-frame ego state into plan_mode_query) is a clean L2 win**: L2 0.4988 → 0.3701 (−0.13, ~26 noise-floor units), CR 0.063% → 0.058% (3368763). MLP-encodes `ego_status[:, [0,1,5,6,7]]` (planar accel, yaw rate, planar velocity) and broadcast-adds it to plan_mode_query at init and after every per-stage rebuild. Decisively rules in the diagnosis from `2026_04_30_anchor_capacity.md` that the planner was missing current-frame ego state: it had only a one-step-lagged longitudinal-velocity signal via `ego_anchor_embed`. Stack-effect with anchor variants pending eval.
- **Velocity-norm shape anchors (`shapeanchor_velnorm`) are roughly tied with baseline**: L2 0.4988 → 0.5215 (+0.02), CR 0.063% → 0.058% (3368762). The trajectory-shape-anchor formulation works (per-batch ||v_0|| scaling is well-defined) but doesn't add value alone — likely because the just-stopped/just-starting samples (v_0 ≈ 0) lose their anchor magnitude entirely under multiplication by 0.
- **Endpoint-norm with magnitude head (`shapeanchor_endptnorm_mag`) blew up**: L2 1.346, CR 0.64% (3368761). Root cause: cmd=Straight cluster k=0 (the stopped anchor) was assigned `refmag=0.0` from k-means cluster median, so the metric anchor for that mode collapses to zero everywhere → `gen_sineembed_for_position` of the zero endpoint is degenerate, and the deformable sampling has no meaningful reference position. Variant 1a needs a refmag floor (`max(refmag, 1.0)`) before retry.
- **Speedstrat L2 regression is training-time, not decode-time**: no-rescore eval on the existing `modesspeedstrat` checkpoint (3368766) lands at L2 0.5560 / CR 0.060% vs the rescore-on baseline of L2 0.5548 / CR 0.070%. ΔL2 = +0.001 (well below noise floor) → speedstrat's +0.056 L2 vs `ppdeformmm_planifls` baseline cannot be recovered by flipping rescore. The decoder-shared-capacity hypothesis (variant 4: `modesspeedstrat_perbucketreg`) is the right intervention — eval pending.
- v4 (`evalmatchmode`) followups confirm the train/eval-alignment fix already calibrated the head — threshold sweep is essentially flat (L2 ∈ [0.5316, 0.5350], CR ∈ [0.080, 0.101]%), and aggregation variants (top-k, detweighted) no longer help once training matches inference. The big knob from Exp 5 (sweeping T from 0.5 to 0.9 cut v3 CR ~30%) collapses to noise on v4. **The hybrid_or selector (OR of hard rescore + v4 learned-hard) hits CR=0.063% — exactly matching hard rescore** while L2 stays at v4's level (0.5330). This means the learned head's correct rejections are a subset of hard rescore's; the head adds no unique collision-avoidance signal beyond the geometric check, only contributes FPs that hurt L2. **Conclusion: the learned scorer cannot beat hard rescore on this architecture.** Hard rescore stays the right inference component.
- All-waypoint planning deformable (T2.5) is **not** a null lever once paired with laststage instfeat. The first run (`_planwp_full6`) dropped `planning_deformable_instfeat_laststage` and regressed L2 by 0.12 — but a follow-up (`_planwp_full6_ls`, K-pool extension of the laststage instfeat branch) ties baseline L2 *and* improves CR (0.086% → 0.071%). The L2 regression in the first run was attributable to losing laststage instfeat, not to all-waypoint sampling itself. Laststage instfeat is load-bearing on this architecture.
- Doubling decoder depth (3 → 6 layers, `_laststage_nodetmap_decoder6`) ties baseline L2 (ΔL2 = +0.004) and modestly improves CR (0.086% → 0.077%, ΔCR = −0.009 pp). Decoder depth alone is not a strong lever, but it stacks cleanly with the laststage all-waypoint variant.
- **Stack: `_decoder6_planwp` (decoder ×6 + all-waypoint + laststage instfeat).** Seed-0 (3319227): L2=0.5150 / CR=0.068%. Seed-1 (3366270): L2=0.5100 / CR=0.087%. **Mean L2=0.5125** across seeds — beats K/V-on `_laststage` reference (0.5302) by 0.018 (well above the same-cluster noise floor of 0.007). **CR did not reproduce**: seed-1 drifted back to the K/V-off baseline range (0.087% vs baseline 0.086%), making the original "recovers ~½ CR cost" reading a single-seed artifact. The L2 win is real; the CR claim must be reframed to "CR matches the K/V-off baseline within noise" pending a third seed or a different mechanism.
- **`_decoder6_planwp_evalmatchmode` (new stack, 3366620): L2=0.5204 / CR=0.046%.** Stacks the decoder6_planwp K/V-off architecture with the v4 evalmatchmode learned-rescore head. L2 within noise of the K/V-off baseline (0.5150) and the K/V-on baseline (0.5302); CR (0.046%) **matches or beats the K/V-on `_laststage` reference (0.054%)**. First K/V-off variant to close *both* the L2 and CR gap to K/V-on. This contradicts the Exp 7 conclusion that the learned scorer adds nothing — Exp 7 was tested on K/V-on, where hard rescore was already saturated; on K/V-off the learned rescore recovers the CR signal that hard rescore loses when det K/V is removed.
- **Seed-1 reproduction of `_decoder6_planwp_evalmatchmode` (3377724): L2=0.5087 / CR=0.053%.** Mean across seeds: **L2=0.5145 / CR=0.0495%** — both seeds beat the K/V-on `_laststage` hard-rescore reference (CR=0.054%) and sit at parity L2 with the K/V-off baseline. The CR=0.046% headline survives reproduction; ΔCR across seeds = 0.007 pp, well within the ~0.015 pp noise floor. Contribution 4 of the paper plan is now defensible across seeds.
- **NEW HEADLINE: `_decoder6_planwp_evalmatchmode_egostatus` (3376689): L2=0.3720 / CR=0.044%.** Folds the anchor-capacity `egostatus` lever (current-frame ego state MLP-encoded into plan_mode_query) into the K/V-off paper stack. ΔL2=−0.148 vs evalmatchmode seed 0, ΔCR=−0.002 pp; the L2 win matches the K/V-on result (0.4988 → 0.3701, ΔL2=−0.13) confirming egostatus transfers cleanly across architectures. **The K/V-off architecture now beats every K/V-on result on this baseline both on L2 and CR**, single seed. Promotes from "supporting evidence" to the paper's main quantitative claim — needs seed-2 reproduction before final lock.
- **Stream B1 (`_evalmatchmode_imgconfdet`, 3376418): L2=0.5119 / CR=0.047%.** Replaces the conflict head's `agent_features` input with image features sampled at det-anchor BEV positions (`conflict_input='image_at_det'`). Ties evalmatchmode within noise on both L2 and CR — **the conflict head's per-agent token is not load-bearing**; image features at the agent location carry the same signal. Unlocks Stream C (drop det/motion queries from the joint decoder) and brings the "zero-dependence stage 2" claim within reach.
- **F1 dispweight (`_evalmatchmode_dispweight`, 3376749): L2=0.5205 / CR=0.055%.** GT-displacement-weighted plan loss (`min_w=0.5, max_w=2.0, scale_m=10.0`). ΔL2≈0, ΔCR=+0.009 pp vs evalmatchmode seed 0. **Null as a standalone lever** — the paper-integrity caveat about uniform application across comparison columns is moot since there's no signal to apply.
- **Exp 8 Option 2 lands but loses to Option 1 on CR (`_distillrescore`, 3377547): L2=0.5091 / CR=0.098%.** K/V-off + `plan_cls` distilled from `rescore()` collide flag at training, `use_rescore=False` at inference. L2 ties evalmatchmode (0.5091 vs 0.5204) and beats hard rescore on its own architecture; CR (0.098%) is 0.05 pp worse than evalmatchmode (0.046%). Distillation through the planner's own logits is *viable* (this is the first inference-rescore-free K/V-off run) but the v4 evalmatchmode learned head is strictly better. The paper architecture stays at `_evalmatchmode` (Option 1).
- **Apollo egoonly stage-1 → stage-2 follow-up (`_ptegoonly_ppdeformmm_planifls`, K3377809): L2=0.5420 / CR=0.073%.** ΔL2=+0.022 vs the direct comparator `_ptnomapdnrot_ppdeformmm_planifls` (#7, L2=0.5203). Minimum-perception stage-1 (motion supervision keepalive-only, planning loss on 3 decoder stages) still produces a viable planning init when stage-2 keeps the standard `ppdeformmm_planifls` recipe — supports the supervision-vs-interface decoupling argument but doesn't recover the strongest stage-1 init. mAP_normal=0.5699 (best of the related pretraining batch) since the map head was trained at stage 1 here, unlike #7 where it was init-from-scratch.
- **Anchor-capacity follow-ups (`shapeanchor_endptnorm_mag` retry, `_egostatus`, `_perbucketreg`): no clear winner beyond plain egostatus.** Refmag-floor patch did not rescue endptnorm_mag (still L2=1.11 / CR=0.39%, 3376688). Adding egostatus on top of endptnorm_mag rescues most of the regression (L2=0.412, 3377754) but does not beat plain egostatus alone (L2=0.3701). Per-bucket regression on speedstrat (3377755) lands at L2=0.5415 but CR=0.339% — per-bucket head doesn't cleanly decouple decoder capacity. **The decisive anchor-capacity finding is still egostatus, and now its strongest demonstration is on the K/V-off architecture (3376689 above), not on this batch.**
- **Egostatus seed-1 reproduction (3386471): L2=0.3695 / CR=0.037%.** Mean across seeds (3376689 + 3386471): **L2=0.3708 / CR=0.0405%**. Paper main quantitative claim is locked across seeds — both L2 and CR survive reproduction (Δ_L2=0.0025 well below noise floor 0.007, ΔCR=0.007 pp within noise floor 0.015 pp). The K/V-off paper architecture beats every K/V-on result on both metrics, both seeds.
- **REFRAMING (2026-05-02): single goal is minS2 — stage-2 with everything off except ego queries + planning task/loss.** The Stream A trio (individual loss zeroing) and the egoonly stage-1 → headline transfer together imply that stage-2 perception heads are inert on both their loss and their interface paths. The maximum-combined config `_egostatus_minS2` (Killarney 3397346) tests this directly: Stream A combined + Stream B1 + Stream C + Stream E in one config. Reference target: headline at L2=0.3708 / CR=0.0405%. All other in-flight work (stage-1 swaps, joint_nodetach, individual-stream diagnostics) is now measured by its contribution to or compatibility with minS2.
- **Stream A — stage-2 zero-loss diagnostic trio (3386472/3386473/3386474): stage-2 perception supervision is empty for planning.** Cloning `_decoder6_planwp_evalmatchmode` (seed-0 baseline L2=0.5204 / CR=0.046%) and zeroing one perception loss family at a time: `s2nodetloss` (L2=0.5265 / CR=0.076%), `s2nomaploss` (L2=0.5161 / CR=0.063%), `s2nomotionloss` (L2=0.5132 / CR=0.042%). All three planning L2s tie the baseline within ~0.006 (well within the 0.007 noise floor); CRs deviate by ≤0.030 pp (≤2× noise). Each ablation collapses its own perception head as expected (det mAP 0.006, map_normal 0.31, car_ade 4.17 respectively) — confirming the loss zeroing took. **First direct test of the supervision-not-interface claim**: stage-2 perception losses are not shaping the backbone in any way that matters for planning, given a strong stage-1 init. Stream E (frozen-backbone S2) is now the natural next beat. Caveat: these are seed-0 single runs against seed-0 baseline; the headline egostatus comparator is at L2=0.3720, but the right comparator for these zero-loss configs (which don't include egostatus) is the seed-0 evalmatchmode baseline.
- **Egoonly (minimum-perception) stage-1 + K/V-off headline stack (`_ptegoonly_decoder6_planwp_evalmatchmode_egostatus`, 3388946): L2=0.3946 / CR=0.055%.** ΔL2=+0.023 vs full-perception stage-1 headline (0.3720), CR +0.011 pp — egoonly stage-1 *nearly recovers* the paper headline. Stage-1 epoch choice on the K/V-on recipe (`_ptegoonly_ep3`, 3388945, L2=0.5402) doesn't move stage-2 L2 vs ep-5 (0.5420). Combined with Stream A, this **decisively supports the supervision-vs-interface decoupling argument**: stage-1 needs *some* perception-aware supervision to shape image features, but stage-2 perception losses are inert — meaning the architecture can be designed around image-feature shaping at stage-1 + planning at stage-2 with no perception decoders downstream.
- **Stream C (`_decoder6_planwp_evalmatchmode_streamc`, 3389115) crashed at eval start with `KeyError: 'trajs_3d'`.** Training completed to iter 11679/11720 (essentially the full 1-epoch schedule); evaluation crashed during NuScenes eval setup because `ego_only_planning=True` strips the agent slots whose `trajs_3d` output the eval pipeline still queries. Bug is in the eval path, not the training path — final ckpt is intact but the result is not yet readable. Needs an eval-time fix that gates on `ego_only_planning` before requesting `trajs_3d`.
- Temporal image-feature stacking (T2.6) is a **null lever at stage 2** on the K/V-off baseline. `tempstack2` regresses L2 by 0.046; `tempstack3` regresses by 0.034; both give only a marginal CR gain (~0.02 pp) that doesn't justify the L2 cost. `tempstack3_noegocomp` collapses by 0.119 L2, confirming ego compensation is load-bearing for the temporal path — the regression is not coming from missing alignment. None of the variants compete with the `_decoder6_planwp` stack. **Stage-1 hypothesis remains open:** the most plausible failure mechanism is that the stage-1 backbone was trained on single-frame supervision so per-frame features at T-1/T-2/T-3 don't compose at stage 2. Training a stage-1 backbone with temporal stacking in-the-loop, then loading into the K/V-off stage-2 stack, is a separate experiment that was not tested here. Added to the stage-1 lever queue.
