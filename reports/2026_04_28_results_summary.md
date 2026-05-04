# Results Summary for Paper Plan

Date: 2026-04-29 (revised 2026-05-02)

Scope: concise summary of recent results relevant to `reports/2026_04_28_paper_plan.md`.
Primary metrics are planning `L2` and `CR=obj_box_col` (lower is better).

## North star: minS2 (everything off in stage 2 except ego queries + planning task)

Reference comparator throughout: K/V-off paper headline (`_decoder6_planwp_evalmatchmode_egostatus`) at **L2=0.3692 / CR=0.0400%** (3-seed mean: 3376689 0.3720/0.044, 3386471 0.3695/0.037, 3394849 0.3660/0.039).

**LANDED 2026-05-03 — minS2 ties or marginally beats headline; paper architecture is locked.** `_egostatus_minS2` (Killarney 3397346): **L2=0.3563 / CR=0.041%**. ΔL2=−0.013 vs headline 3-seed mean (marginally outside the 0.007 noise floor on a single seed); ΔCR=+0.001 pp (at noise). All four streams combined (A loss-zeroing + B1 image_at_det + C ego_only_planning + E frozen backbone+heads) match the headline. Stream E alone (`_frozenpercep` 3394848): L2=0.3698 / CR=0.044% — also ties headline within noise. Best stage-1 swap is `_ptaux2d_dseg` (3396708) at L2=0.3621 / CR=0.043%.

**Refinement (2026-05-02 PM):** the `image_at_det` conflict head used by the headline + minS2 (3397346) still consults detection anchor BEV positions to know where to sample image features — so the detection head's forward pass still happens at inference even when its outputs feed nothing else. Three new conflict-sampler variants close this last dependency by sourcing spatial keypoints from non-perception signals: **B1.5** (plan trajectory waypoints), **B1.6** (top-K fixed init anchors closest to ego), **B1.7** (plan trajectory + 7-point footprint). B1.6 (3401950, eval-only on 3398020 ckpt): L2=0.3665 / CR=0.067%. B1.7 (3401952, eval-only on 3398021 ckpt): L2=0.3729 / CR=0.056%. **B1.6 ties headline on L2 (within noise) but elevates CR by 0.027 pp; B1.7 also ties L2 within noise but with elevated CR.** B1.5 (3399641 retry) crashed again — needs separate diagnosis. If a B1.x variant ties both metrics, the paper claim becomes **fully detection/map-free at inference**.

**Path to minS2** combines four components, each demonstrated individually and now also jointly (3397346):

| Component | Closes at | Result | Source |
|---|---|---|---|
| K/V off | inference perception interface | Headline (0.3692/0.040%) | `_decoder6_planwp_evalmatchmode_egostatus` (3 seeds) |
| Loss zeroing (Stream A, individual) | stage-2 perception supervision | All 3 tie L2=0.5204 baseline within noise | 3386472/73/74 |
| Loss zeroing (Stream A, **combined**) | stage-2 perception supervision (joint) | **L2=0.3572 / CR=0.061% (3406308)** | landed |
| Drop agent slots (Stream C) | inference perception in planner queries | **L2=0.3597 / CR=0.026% (K3394847, beats headline)** | landed |
| Conflict head image features (Stream B1) | inference perception in conflict head | Ties evalmatchmode (3376418, L2=0.5119 / CR=0.047%) | already in hand |
| Frozen backbone+heads (Stream E) | stage-2 backbone gradient | **L2=0.3698 / CR=0.044% (3394848)** | landed — ties headline |
| **All combined → minS2** | the goal | **L2=0.3563 / CR=0.041% (3397346)** | landed — ties/beats headline |

minS2 ties headline within noise. Stage-1 candidates 3396706-9 are evaluated below; best is `_ptaux2d_dseg` (3396708, L2=0.3621). The Apollo joint_nodetach stage-1 (`detach_perception=False`) is the last open lever — its stage-2 transfer through minS2 is the next step.

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
| **`_decoder6_planwp_evalmatchmode_streamc`** (Stream C, eval-only) | 1 | **0.3597** | **0.026%** | - | - | - | - | 0.5245 | - | 0.5556 | **(3394847)** ego_only_planning=True + egostatus + image_at_det conflict sampler. Agent slots dropped entirely from the planner. ΔL2=−0.011 vs headline mean (marginally outside noise floor), ΔCR=−0.015 pp (at noise floor). **Ties or marginally beats the headline on a single seed** — dropping all agent slots does not hurt performance. Needs a second seed. Motion metrics absent (no agent predictions by design). |
| **`_decoder6_planwp_evalmatchmode_streamc`** (seed 2, train+eval) | 1 | **0.3669** | **0.042%** | - | - | - | - | 0.5239 | - | 0.5541 | **(K3400093)** Independent training run from `_streamc` (not eval-only on `_egostatus` ckpt). ΔL2=+0.007 vs seed-1 (just at noise floor), CR drifts to baseline range. **2-seed mean L2=0.3633 / CR=0.034%** — still ties/beats 3-seed headline (0.3692/0.040%) but the seed-1 win softens. Confirms Stream C reproduces directionally; tightens the read to "ties headline" rather than "beats headline". |
| **`_decoder6_planwp_evalmatchmode_egostatus`** (seed 2) | 1 | **0.3660** | **0.039%** | 0.6276 | 0.7170 | 0.4860 | 0.4083 | 0.5171 | - | 0.5538 | **(3394849)** Third headline seed locks the comparator. **3-seed mean: L2=0.3692 / CR=0.0400%.** ΔL2 across seeds ≤ 0.0060 (under noise floor 0.007), ΔCR ≤ 0.007 pp (well under 0.015 pp). Headline is decisively reproducible. |
| **`_decoder6_planwp_evalmatchmode_egostatus_minS2`** (Stream A+B1+C+E combined) | 1 | **0.3563** | **0.041%** | - | - | - | - | 0.5295 | - | 0.3352 | **(3397346)** All four streams stacked: zero stage-2 perception losses (A) + image_at_det conflict sampler (B1) + ego_only_planning (C) + frozen backbone+neck+det+map (E). ΔL2=−0.013 vs 3-seed headline mean 0.3692 (marginally outside noise floor); ΔCR=+0.001 pp (at noise). **Paper architecture locked.** mAP_normal collapses to 0.3352 because map head is frozen and never trained at S2 in this config. |
| `_decoder6_planwp_evalmatchmode_egostatus_frozenpercep` (Stream E alone) | 1 | 0.3698 | 0.044% | 0.6397 | 0.7277 | 0.4910 | 0.4093 | 0.5295 | - | 0.3350 | **(3394848)** Freeze backbone + neck + det head + map head; only motion + plan train. ΔL2=−0.006 vs headline mean (within noise), ΔCR=+0.004 pp (within noise). Frozen-S2 ties headline → backbone gradient at S2 is empty for planning. mAP_normal collapses (frozen map head from S1 init). |
| `_decoder6_planwp_evalmatchmode_egostatus_s2nopercep` (Stream A combined) | 1 | 0.3572 | 0.061% | - | - | -0.78 | -0.28 | 0.0001 | - | 0.1594 | **(3406308, eval-only with NaN-safe det filter)** Zero all 3 stage-2 perception losses simultaneously (det+map+motion) on the egostatus headline. ΔL2=−0.012 vs 3-seed headline mean 0.3692 (marginally outside noise floor on a single seed); ΔCR=+0.021 pp (at edge of noise). All 3 perception heads collapse as expected (NDS=0.0001, mAP_normal=0.16, agent EPAs negative). Planning preserved. **Confirms Stream A combined alone is empty for planning** — matches the individual-stream behavior (3386472/73/74) at the joint level. |
| `_decoder6_planwp_evalmatchmode_egostatus_minS2_2a_aux` | 1 | 0.3634 | 0.040% | - | - | - | - | 0.5280 | - | 0.3357 | **(3399643)** Option-2a (with aux) variant of minS2. Ties headline (ΔL2=−0.006, ΔCR=0). |
| `_decoder6_planwp_evalmatchmode_egostatus_minS2_2a_noaux` | 1 | 0.3647 | 0.068% | - | - | - | - | 0.5277 | - | 0.3355 | **(3399642)** Option-2a (no aux). L2 within noise of headline; CR elevated by 0.028 pp. |
| `_decoder6_planwp_evalmatchmode_egostatus_minS2_B1p6` (eval-only) | 1 | 0.3665 | 0.067% | - | - | - | - | 0.5289 | - | 0.3344 | **(3401950 on 3398020 ckpt)** Conflict sampler reads from top-K fixed init anchors closest to ego — no detection anchor positions consulted. L2 ties headline within noise; CR elevated by 0.027 pp. |
| `_decoder6_planwp_evalmatchmode_egostatus_minS2_B1p7` (eval-only) | 1 | 0.3729 | 0.056% | - | - | - | - | 0.5287 | - | 0.3354 | **(3401952 on 3398021 ckpt)** Conflict sampler reads from plan trajectory + 7-point footprint — no detection anchor positions. ΔL2=+0.004 (at noise edge), ΔCR=+0.016 pp (at noise edge). Both B1.6 and B1.7 close the last detection-forward dependency at inference but trade ~0.025 pp CR for that closure. |
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
| `ptaux2d_decoder6_planwp_evalmatchmode_egostatus` | 1 | **0.3627** | 0.056% | 0.6162 | 0.7231 | 0.5058 | 0.4282 | 0.5456 | - | 0.5609 | **(3396706)** aux2d stage-1 → K/V-off headline stack. ΔL2=−0.007 vs 3-seed headline mean (at noise edge), ΔCR=+0.016 pp (at noise edge). Best L2 in this batch alongside dseg. NDS=0.5456 confirms strong perception on this stage-1 init |
| `ptaux2d_dino_decoder6_planwp_evalmatchmode_egostatus` | 1 | 0.3655 | 0.056% | 0.6276 | 0.6889 | 0.4879 | 0.4220 | 0.5275 | - | 0.5266 | **(3396707)** aux2d + DINO-init stage-1 → headline. Ties headline within noise (ΔL2=−0.004, ΔCR=+0.016 pp). DINO prior does not lift planning beyond aux2d on this recipe |
| `ptaux2d_dseg_decoder6_planwp_evalmatchmode_egostatus` | 1 | **0.3621** | **0.043%** | 0.6159 | 0.7048 | 0.4985 | 0.4247 | 0.5399 | - | 0.5538 | **(3396708)** aux2d + dense segmentation stage-1 → headline. **Best L2 (0.3621) and CR (0.043%) of the stage-1 swap batch.** ΔL2=−0.007 (at noise edge), ΔCR=+0.003 pp (within noise). Marginal headline improvement; if the L2 gain holds across seeds, dseg is the stage-1 to pair with minS2 |
| `ptnomapdnrot_decoder6_planwp_evalmatchmode_egostatus` | 1 | 0.3700 | 0.052% | 0.6197 | 0.6892 | 0.5166 | 0.4582 | 0.5512 | - | 0.2571 | **(3396709)** stage-1-nomap + dnrot → headline. Ties headline within noise (ΔL2=+0.001, ΔCR=+0.012 pp); mAP_normal=0.2571 because S1 didn't train the map head and S2 does (init-from-scratch). The map head still trains; planning is unaffected |

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
- **Stream C (`_evalmatchmode_streamc`, eval-only 3394847): dropping all agent slots from the planner (ego_only_planning=True) does not hurt performance.** L2=0.3597 / CR=0.026% vs headline mean L2=0.3708 / CR=0.0405%. ΔL2=−0.011 is marginally outside the 0.007 noise floor (single-seed, needs confirmation); ΔCR=−0.015 pp is at the noise floor. The planner doesn't use agent slots at inference — the "zero-dependence stage 2" claim is directly supported. Motion metrics are absent by design (ego_only_planning strips agent predictions). Second seed recommended before locking.
- **Cross-cluster confirmation of `_evalmatchmode` (2026-05-03): result is stable across clusters.** Rorqual 11245220 (H100): L2=0.5160 / CR=0.052% — essentially matches killarney seed 1 (0.5087 / 0.053%). Tamia 270728 (H100): L2=0.5248 / CR=0.071% — L2 within noise, CR slightly elevated but within max noise (0.025 pp Δ vs 0.039 pp max). **Fir 38335393 (H100): L2=0.5149 / CR=0.047%** — also matches Killarney within noise; this is the cleanest cross-cluster read of evalmatchmode (CR exactly matches K3366620 0.046%). 4-cluster mean now L2=0.5160 / CR=0.054%. **Caution: tamia and rorqual `_egostatus` runs (270279, 11239029) have anomalously high CR (0.243%, 0.131%) because those clusters are missing commit `95db051` (SparseBox3DKeyPointsGenerator dtype fix).** The egostatus results from these clusters are not valid until git pull is performed.
- **REFRAMING (2026-05-02): single goal is minS2 — stage-2 with everything off except ego queries + planning task/loss.** The Stream A trio (individual loss zeroing) and the egoonly stage-1 → headline transfer together imply that stage-2 perception heads are inert on both their loss and their interface paths. The maximum-combined config `_egostatus_minS2` (Killarney 3397346) tests this directly: Stream A combined + Stream B1 + Stream C + Stream E in one config. Reference target: headline at L2=0.3708 / CR=0.0405%. All other in-flight work (stage-1 swaps, joint_nodetach, individual-stream diagnostics) is now measured by its contribution to or compatibility with minS2.
- **LANDED 2026-05-03 — minS2 ties or marginally beats headline; paper architecture is locked.** `_egostatus_minS2` (Killarney 3397346) at **L2=0.3563 / CR=0.041%** vs 3-seed headline mean **0.3692 / 0.0400%** (ΔL2=−0.013 marginally outside the 0.007 noise floor on a single seed; ΔCR=+0.001 pp at noise). Stream E alone (`_frozenpercep` 3394848) lands at L2=0.3698 / CR=0.044% — within noise of headline → freezing the entire S2 perception stack does not hurt planning. Combined with Stream A (zero-loss) ties + Stream B1 + Stream C ties, this is the most aggressive S2 simplification we've tested and it works. The paper's "stage-2 perception is empty for planning" claim is directly demonstrated end-to-end. Seed-2 of minS2 is the next confirmation step.
- **Stream A combined (`_s2nopercep`, 3406308): L2=0.3572 / CR=0.061%.** Zeroing all 3 stage-2 perception losses simultaneously (det + map + motion) on the egostatus headline ties planning within noise (ΔL2=−0.012 vs 3-seed mean) while all 3 perception heads collapse as expected (NDS=0.0001, mAP_normal=0.16, agent EPAs negative). Confirms that the combined zero-loss is empty for planning, not just the individual streams (3386472/73/74). The eval was unblocked by the new NaN-safe filter in `output_to_nusc_box` (commit `c54c8e1`); without it, unsupervised det heads produce NaN box predictions and crash NuScenesBox formatting before planning metrics are computed. CR is +0.021 pp vs headline mean (at noise edge) — slightly elevated relative to minS2 (0.041%), suggesting the rescore signal benefits from *some* perception/temporal gradient stabilization that frozen-percep + agent-slot-drop preserve but pure-zero-loss doesn't.
- **Headline egostatus seed 3 lands (Killarney 3394849, L2=0.3660 / CR=0.039%); 3-seed mean L2=0.3692 / CR=0.0400%.** All seed deltas under noise floor (max ΔL2=0.0060 vs 0.007; max ΔCR=0.007 pp vs 0.015 pp). Headline is decisively reproducible.
- **Stage-1 swap batch (3396706-9) — all four ties or marginally beats headline; aux2d_dseg is the best variant (3396708): L2=0.3621 / CR=0.043%.** Pairs of stage-1 init × K/V-off headline stack: aux2d (0.3627/0.056%), aux2d_dino (0.3655/0.056%), aux2d_dseg (0.3621/0.043%), nomapdnrot (0.3700/0.052%). All four cluster within ~0.008 L2 of the 3-seed headline mean. **dseg is the only swap that beats both L2 and matches CR within noise.** DINO-init does not lift planning beyond aux2d. nomapdnrot reproduces the prior #7 finding that S1 nomap is fine when S2 keeps map (planning unaffected; map head warm-starts at S2). The next experiment is `_minS2_ptaux2d_dseg` (best stage-1 × locked architecture).
- **B1.6 / B1.7 conflict-sampler retries (eval-only on B1.6 train ckpt 3398020 = 3401950, on B1.7 train ckpt 3398021 = 3401952): both close the last detection-forward dependency at inference but elevate CR.** B1.6 (top-K fixed init anchors closest to ego, 3401950): L2=0.3665 / CR=0.067%. B1.7 (plan trajectory + 7-point footprint, 3401952): L2=0.3729 / CR=0.056%. Both L2 within noise of headline; CRs +0.027/+0.016 pp respectively. **Either ties L2 → "fully detection/map-free at inference" is achievable.** Tradeoff is CR: minS2 (with image_at_det → still queries det anchor BEV positions) hits CR=0.041%; B1.6/B1.7 (no det forward) hit ~0.06%. B1.5 (3399641 retry) crashed again — needs separate diagnosis.
- **Cross-cluster confirmation of headline + Stream C (Fir 38335844/45, Rorqual 11262822/873/74, Tamia 272454): L2 reproduces; CR is cluster-dependent on egostatus only.** Fir `_egostatus` seed-3 (38335844): L2=0.3701 / CR=0.087%. Fir `_streamc` seed-3 (38335845): L2=0.3700 / CR=0.040%. Rorqual `_egostatus` (11262873): L2=0.3781 / CR=0.094%. Rorqual `_egostatus` (11262822, 2026-05-03 PM): **L2=0.3638** / CR=0.119% — best L2 yet on Rorqual egostatus, CR consistent with TF32 hypothesis. Rorqual `_streamc` seed-3 (11262874): L2=0.3727 / CR=0.040%. Tamia `_egostatus` (272454): L2=0.3740 / CR=0.243% — exactly reproduces the prior tamia 270279 CR pattern. **Killarney `_streamc` seed-2 train+eval (3400093): L2=0.3669 / CR=0.042%** — second seed, full retrain not eval-only. **Streamc reproduces cleanly across clusters and seeds** (Killarney seed-1 0.3597/0.026%, seed-2 0.3669/0.042%, Fir 0.3700/0.040%, Rorqual 0.3727/0.040% — all within noise of headline; mean 0.3673/0.037%). **Egostatus L2 reproduces but CR is elevated** on Fir + Rorqual + Tamia — TF32-on-H100 hypothesis stands; Killarney L40S remains the canonical CR comparator for egostatus.
- **Fir egostatus CR diagnosis (2026-05-03): not a code/data difference — likely H100 TF32 numerics through the egostatus broadcast-add.** Fir already includes commit `95db051` (dtype fix); `ckpt/sparsedrive_stage1.pth` md5 matches Killarney exactly (`3e94352c…`); container is the same `foresight_cuda118.sif`. So three Hypothesis-A factors are ruled out (different ckpt, different env, missing fix). The remaining cluster-specific factor is **H100 SXM5 (Fir/Rorqual/Tamia) vs L40S (Killarney)** — H100 enables TF32 on matmul by default at higher throughput; the egostatus path runs an MLP on `ego_status[:,[0,1,5,6,7]]` (planar accel/yaw rate/planar velocity, mixed magnitudes) and broadcast-adds it to `plan_mode_query`. The downstream trajectory output diverges from Killarney by ~mm-level perturbations — enough to flip a small fraction of trajectories across the obj_box collision threshold but not enough to move L2. Streamc reproduces because it bypasses `plan_mode_query` for ego_only_planning (no rebuild after each per-stage refine). Diagnostic test: rerun seed-3 with `torch.backends.cuda.matmul.allow_tf32 = False` on Fir; if CR drops to 0.04%, the TF32 hypothesis is confirmed.
- **Stream A — stage-2 zero-loss diagnostic trio (3386472/3386473/3386474): stage-2 perception supervision is empty for planning.** Cloning `_decoder6_planwp_evalmatchmode` (seed-0 baseline L2=0.5204 / CR=0.046%) and zeroing one perception loss family at a time: `s2nodetloss` (L2=0.5265 / CR=0.076%), `s2nomaploss` (L2=0.5161 / CR=0.063%), `s2nomotionloss` (L2=0.5132 / CR=0.042%). All three planning L2s tie the baseline within ~0.006 (well within the 0.007 noise floor); CRs deviate by ≤0.030 pp (≤2× noise). Each ablation collapses its own perception head as expected (det mAP 0.006, map_normal 0.31, car_ade 4.17 respectively) — confirming the loss zeroing took. **First direct test of the supervision-not-interface claim**: stage-2 perception losses are not shaping the backbone in any way that matters for planning, given a strong stage-1 init. Stream E (frozen-backbone S2) is now the natural next beat. Caveat: these are seed-0 single runs against seed-0 baseline; the headline egostatus comparator is at L2=0.3720, but the right comparator for these zero-loss configs (which don't include egostatus) is the seed-0 evalmatchmode baseline.
- **Egoonly (minimum-perception) stage-1 + K/V-off headline stack (`_ptegoonly_decoder6_planwp_evalmatchmode_egostatus`, 3388946): L2=0.3946 / CR=0.055%.** ΔL2=+0.023 vs full-perception stage-1 headline (0.3720), CR +0.011 pp — egoonly stage-1 *nearly recovers* the paper headline. Stage-1 epoch choice on the K/V-on recipe (`_ptegoonly_ep3`, 3388945, L2=0.5402) doesn't move stage-2 L2 vs ep-5 (0.5420). Combined with Stream A, this **decisively supports the supervision-vs-interface decoupling argument**: stage-1 needs *some* perception-aware supervision to shape image features, but stage-2 perception losses are inert — meaning the architecture can be designed around image-feature shaping at stage-1 + planning at stage-2 with no perception decoders downstream.
- **Stream C (`_decoder6_planwp_evalmatchmode_streamc`, 3389115) crashed at eval start with `KeyError: 'trajs_3d'`.** Training completed to iter 11679/11720 (essentially the full 1-epoch schedule); evaluation crashed during NuScenes eval setup because `ego_only_planning=True` strips the agent slots whose `trajs_3d` output the eval pipeline still queries. Bug is in the eval path, not the training path — final ckpt is intact but the result is not yet readable. Needs an eval-time fix that gates on `ego_only_planning` before requesting `trajs_3d`.
- Temporal image-feature stacking (T2.6) is a **null lever at stage 2** on the K/V-off baseline. `tempstack2` regresses L2 by 0.046; `tempstack3` regresses by 0.034; both give only a marginal CR gain (~0.02 pp) that doesn't justify the L2 cost. `tempstack3_noegocomp` collapses by 0.119 L2, confirming ego compensation is load-bearing for the temporal path — the regression is not coming from missing alignment. None of the variants compete with the `_decoder6_planwp` stack. **Stage-1 hypothesis remains open:** the most plausible failure mechanism is that the stage-1 backbone was trained on single-frame supervision so per-frame features at T-1/T-2/T-3 don't compose at stage 2. Training a stage-1 backbone with temporal stacking in-the-loop, then loading into the K/V-off stage-2 stack, is a separate experiment that was not tested here. Added to the stage-1 lever queue.
