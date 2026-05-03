# Paper Plan: Rethinking Sparse Scene Representations for End-to-End Driving

2026-04-28 (revised 2026-05-02)

## North star: Minimum Stage-2 (minS2)

**Single goal: shortest path to a stage-2 architecture where everything is OFF except ego queries and the planning task/loss.**

Concretely: at stage 2 the model trains only the planner head on top of a stage-1-shaped backbone. Detection, map, and motion heads should be removable from the inference path entirely — they exist only for evaluation reporting (NDS/mAP) and contribute no gradient, no loss, and no spatial information to the planner or conflict head.

**The current minS2 config (3397346) is K/V-off + Stream A + Stream E + Stream B1, which closes most of the perception dependency but leaves a residual: the conflict head's `image_at_det` sampler still uses detection anchor BEV positions as its spatial query, so the detection head still runs forward at inference (just frozen, supervisionless, and unread by the planner). The next batch removes this last dependency** by sourcing the conflict sampler's spatial query from places that don't require any perception forward pass:

| Variant | Spatial source for conflict sampler | Closes |
|---|---|---|
| **B1.5** (`image_at_plan` single-point) | Plan trajectory waypoints (M×T points) | Det forward pass |
| **B1.6** (`image_at_init_topk`) | Top-K nearest fixed init det anchors to ego | Det forward pass; falls back to broad fixed coverage |
| **B1.7** (`image_at_plan` + footprint) | Plan trajectory waypoints + 7-point ±0.45 m footprint expansion | Det forward pass; covers ego-corridor properly |

If any of B1.5 / B1.6 / B1.7 ties the headline within noise, **the paper architecture is fully detection/map-free at inference** — perception heads can be deleted from the model entirely (kept only as eval-reporting hooks against the stage-1 frozen weights). This is the strongest possible form of the supervision-vs-interface decoupling claim.

Every stream below feeds this north star:
- **Stream A** kills perception losses at stage 2 (det/map/motion = 0).
- **Stream C** drops agent slots from the planner (`ego_only_planning=True`).
- **Stream B1** swaps the conflict head's agent-feature input for image features at det positions.
- **Stream E** freezes backbone + perception heads (lr_mult=0).
- **minS2 config (`_egostatus_minS2`, Killarney 3397346, in flight)** combines A + C + B1 + E. If it ties the headline within noise, the paper architecture is locked.

Stage-1 work feeds the same goal: stage-1 needs to shape the backbone well enough that stage-2 can train only the planner. Stage-1 candidates (`aux2d`, `aux2d_dino`, `aux2d_dseg`, `nomapdnrot`, `egoonly`, `joint_nodetach`) are evaluated by their stage-2 transfer **on the K/V-off paper headline stack**, and ultimately on minS2.

## Overview

Sparse scene representations (object detections, map elements, motion predictions) can help planning in two distinct roles:

1. **As an interface**: tokens consumed by the planner at inference time.
2. **As supervision**: training signal that teaches the visual representation what to encode.

Our central claim is that **sparse scene representations are more valuable in the second role than the first** — and stronger still: **stage-2 needs neither the interface nor the supervision**. The interface is empty (Stream A trio of individual loss zeroings, plus the K/V ablation grid). The supervision at stage-2 is empty (Stream A combined `_s2nopercep`, in flight, plus the converging Stream A trio). The bottleneck is representation alignment at stage 1; stage 2 is a small planner-only fine-tune.

## Contributions

1. **Hidden redundancy.** Sparse scene representations contribute far less as planning interfaces than current end-to-end architectures imply — across token count (0, 25, 50, 100, 900), selection criterion (top-k, oracle relevance, learned), and bidirectional flow, planning is unmoved.
2. **The real bottleneck.** Planning performance is limited by representation alignment in image features, not by token hand-off bandwidth. Every confirmed positive lever in the planning era touches the image-feature path; every perception-token-path lever has been null or regressive.
3. **Decoupling two conflated roles.** Perception as training signal and perception as inference-time planning input are routinely conflated in current stacks. We separate them empirically: stage-1 perception drives backbone shaping, stage-2 perception adds essentially nothing to planning at inference.
4. **A reusable abstraction.** Scene representations can be used purely as supervision — kept at stage 1, dropped at stage 2 — without being required as inference interfaces. The K/V-off stage-2 architecture is the concrete instantiation.
5. **Simplify without sacrificing performance.** Direct visual planning (no perception K/V at inference) matches or improves over sparse-scene-interface baselines while retaining rich training supervision. Lower latency, fewer moving parts, same or better L2/CR.

## Key insight

Perception decoders (det, map, motion) in current end-to-end driving stacks are **training-time auxiliary tasks**, not inference-time planning inputs. The planner reads what it needs from the image features directly.

## What "planning-relevant features" means

The planner doesn't need a tokenized scene graph. It needs image features that encode:

- **3D world knowledge** — where things are in space, projected through the deformable cross-attention.
- **Collision-relevant geometry** — proximity, agent extents, drivable surface boundaries.
- **Rules of the road** — lane geometry, signal/stop-sign cues, intersection structure.

These are exactly what perception supervision teaches the backbone during pretraining. We exploit that by *keeping* the supervision but *removing* the inference-time interface. Many assumptions baked into existing E2E driving stacks — that you need explicit object/map tokens flowing into the planner; that more agents in K/V helps; that motion predictions must be input to the planner at inference — are disproved by our ablation grid. **Features are all that matters at inference; complex multi-task token-passing architectures don't.**

## Paradigm

**Current.**
- Stage 1: perception pretraining (det, map).
- Stage 2: joint perception → planning/prediction training; perception tokens flow as K/V into the planner at inference.

**Proposed.**
- Stage 1: pretraining image features for planning. Perception is one supervisory option among several (alongside dense aux2d, depth, drivable-area, occupancy).
- Stage 2: end-to-end planner reading directly from image features — perception-free at inference. Perception heads exist if needed for evaluation reporting (NDS, mAP) but are not on the inference forward path.

## Roadmap to minS2 (everything off in stage 2 except ego queries + planning task)

**minS2 = headline + Stream A combined + Stream C + Stream B1 + Stream E.** A single config (`_egostatus_minS2`, Killarney 3397346) tests it directly — if it ties the headline within noise, the architecture is locked. The streams below remain useful as ablation localizers if minS2 regresses; each isolates one component of the combined config.

Reference comparator: K/V-off paper headline (`_decoder6_planwp_evalmatchmode_egostatus`) at **L2=0.3692 / CR=0.0400%** (3-seed mean: 3376689 0.3720/0.044, 3386471 0.3695/0.037, 3394849 0.3660/0.039). All seed deltas under noise floor.

**LANDED 2026-05-03 — minS2 (3397346) at L2=0.3563 / CR=0.041%; ΔL2=−0.013 (marginal beat, single seed), ΔCR=+0.001 pp (at noise). Paper architecture is locked.** Stream E alone (`_frozenpercep` 3394848) at L2=0.3698/CR=0.044% — confirms freezing the entire S2 perception stack is safe for planning. Best stage-1 swap is `_ptaux2d_dseg` (3396708) at L2=0.3621/CR=0.043%. The next batch pairs the best stage-1 with minS2.

### Stream A — Stage-2 supervision diagnostics (cheapest; running now)

Tests whether stage-2 perception losses contribute to backbone shaping at all. Three configs cloning `_decoder6_planwp_evalmatchmode` with one perception loss family zeroed each:

- `_s2nodetloss` (Killarney 3372803) — det loss family → 0
- `_s2nomaploss` (Killarney 3372804) — map loss family → 0
- `_s2nomotionloss` (Killarney 3372805) — motion loss family → 0

If all three tie the baseline, stage-1 carries the supervisory load and stage-2 perception heads can be deleted entirely without quality cost. This sharpens the paper's "supervision-not-interface" claim from "K/V is empty" to "**stage-2 perception is empty**".

### Stream B — Detach the conflict head from agent features

The conflict head currently reads agent_features. To break that dependency:
- **B1 (image-feature conflict head)**: replace the agent-feature input with image features sampled at the agent's predicted BEV location. Same spatial query, no per-agent token.
- **B2 (BEV-readout conflict head)**: a small head trained at stage 1 outputs per-BEV-cell occupancy/drivable; the conflict head reads cells along the planned trajectory's footprint. Cell-level rather than object-level.
- **B3 (model-based collision proxy)**: at inference, sample candidate trajectories, forward-roll a vehicle footprint, reject any that intersect a thresholded occupancy estimate. No learning — pure geometry on top of B2.

B1, B2, B3 substitute for each other. B1 is the cheapest engineering. B2/B3 are cleaner architecturally.

### Stream C — Drop detection / motion queries from the joint decoder

Remove the agent slots from `instance_feature` so the planning head runs ego-only at inference. The `ego_only_planning` flag already exists (added during the egoonly stage-1 work); needs a stage-2 code path that handles `det_output=None` and a temporal queue without agent IDs. Gated on Stream B (without a substitute conflict-head input, CR collapses).

### Stream D — Map removal at stage 2

Already supported by `_laststage_s2nomap` (ties baseline). Need to re-confirm on the current best stack with `_decoder6_planwp_evalmatchmode_s2nomaploss` (Stream A) and then take the next step: build with `with_map=False` so the map head literally isn't constructed.

### Stream E — Frozen backbone at stage 2

If Stream A shows stage-2 perception losses are null, the cleanest end-state is: freeze the backbone at stage 2, train only the planning head. Removes the question of "what shapes the backbone at stage 2" by taking the backbone off the table. Gated on Stream A's outcome.

### Stream F — Maneuver loss balancing (training-recipe lever, orthogonal to architecture)

The L1 trajectory loss is dominated by stationary/short-motion samples on nuScenes. Down-weighting trivial trajectories and up-weighting harder maneuvers is a known forecasting technique with documented gains; it's mechanism-distinct from anything we've explored architecturally.

- **F1 (GT-displacement weight, simplest)**: per-sample weight = `clip(endpoint_displacement / scale_m, min_w, max_w)`. Stationary → `min_w`, fast turns → `max_w`. No new labels. Defaults `min_w=0.5, max_w=2.0, scale_m=10.0` keep loss magnitude in [0.5×, 2×] of original. **Implemented as `displacement_weight` flag on `MotionTarget` and `PlanningTarget`.** First config: `_decoder6_planwp_evalmatchmode_dispweight` (Killarney **3376749**, completed 2026-05-01) → **L2=0.5205 / CR=0.055%** vs evalmatchmode seed 0 (0.5204 / 0.046%). **ΔL2≈0, ΔCR=+0.009 pp — null as a standalone lever.** No paper-integrity follow-up needed since there's no signal to apply uniformly.
- **F2 (curvature weight)**: same idea, but use trajectory curvature instead of endpoint displacement to specifically target turns rather than fast-straight motion. Deferred.
- **F3 (maneuver-class weight)**: classify GTs into {stationary, straight, accel, decel, left, right} and inverse-frequency weight. Needs a label-gen pass. Deferred.

**Important constraint for paper integrity**: any positive recipe-level result has to be applied identically to the K/V-on baseline (`_planinstfeat_laststage`) and to the K/V-on `ptaux2d_ppdeformmm_planifls` reference, otherwise the architectural delta between K/V-on and K/V-off can't be cleanly read. Treat F1 as a *recipe upgrade applied uniformly across all comparison columns*, not as a feature unique to our architecture.

### Fallback: minimal-dependence

If Stream B proves brittle on CR, the paper's claim becomes **"the planner needs *bounded* perception, not a tokenized scene graph"**: K nearest detected objects within the ego corridor, or just a single learned occupancy mask. Still a structural break from the standard E2E paradigm; not zero.

### Decision tree

```
Stream A: stage-2 loss-zero trio (3372803/4/5)
        │
        ├─ all three tie baseline ──► Stream E: frozen-backbone S2
        │                              │
        │                              └─ Stream B (conflict-head substitute)
        │                                  │
        │                                  └─ Stream C: drop det/motion queries
        │                                      │
        │                                      └─ HEADLINE: zero-dependence S2
        │
        └─ any regress ──► Fallback: minimal-dependence with bounded agent set
                            │
                            └─ HEADLINE: K-nearest-object S2
```

## Evidence summary

| Claim | Direct evidence |
|---|---|
| Stage-2 perception is essentially redundant for planning | K/V token-count grid (0/25/50/100/900): planning moves by ±0.01 L2/CR. ~15 selection / supervision / coupling perturbations: all null or regressive. |
| Stage-1 perception is load-bearing | Stage-1 *and* stage-2 nomap is catastrophic (L2=6.6, CR=3.6%). Stage-1 nomap with stage-2 map intact preserves planning. |
| Image features carry L2 | `planpredtrajdeformmm` (planner deformable to image features) is the main L2 driver: 0.636 → 0.522. |
| Rescore carries collision rate | Removing rescore costs ~0.04 pp CR; learned-scorer replacements all underperform; hybrid-OR with hard rescore matches but adds nothing. |
| Map K/V is dispensable | F3 `_laststage_nomapkv` ties baseline. Stage-2 nomap entirely (`_laststage_s2nomap`) ties baseline. |
| Det K/V is dispensable for L2, but rescore needs det confidence | F1 alone regresses; combined with the right architecture (`_laststage_nodetmap_decoder6_planwp`) the L2 win reproduces across seeds (mean 0.5125, vs K/V-on 0.5302). The CR fix needs a *different* mechanism: stacking with the v4 learned rescore head (`_decoder6_planwp_evalmatchmode`) gives CR=0.046%, matching/beating K/V-on. |

## TODO

- [x] Title locked: "Rethinking Sparse Scene Representations for End-to-End Driving"
- [ ] Lock the minimal end-to-end Stage 2 architecture (no perception K/V; rescore handled by either small motion head or learned cost)
- [ ] Run #1 DINO-init stage-1 on Trillium (4-GPU bs24) to test "stronger image backbone init" lever
- [x] Run #7 stage-1-nomap_dn_rotaug + stage-2-with-map on Killarney (3311181) — see "Nomap evidence" below; backbone-shaping vs inference-path confound resolved
- [x] Implement DenseSegHead + GenerateDenseSegMask (v1: 6 channels — 3 polylines + 3 agents); submitted aux2d_dseg on Trillium (472563)
- [ ] v2: extend dense seg with drivable_area / walkway / stop_line (requires map_annos extension or BEV-derivation)
- [x] All-waypoint planning deformable variant (T2.5) — first run `_laststage_nodetmap_planwp_full6` (Killarney 3319223) dropped laststage instfeat and regressed: `L2=0.6352 / CR=0.097% / NDS=0.5286 / mAP=0.4135 / mAP_normal=0.5488`, +0.12 L2 vs `_laststage_nodetmap` baseline. Follow-up with laststage instfeat retained via instfeat-branch K-pool, `_laststage_nodetmap_planwp_full6_ls` (Killarney 3319226), recovered: **`L2=0.5186 / CR=0.071% / NDS=0.5203 / mAP=0.4087 / mAP_normal=0.5555`** — ties baseline L2 within noise *and* improves CR (0.086% → 0.071%). The L2 regression in the first run was caused by losing laststage instfeat, not by all-waypoint sampling. **Laststage instfeat is load-bearing on the K/V-off architecture; T2.5 is positive when paired with it.**
- [x] Temporal image-feature stacking variant (T2.6) — Killarney 3322355/56/57, all completed 8h49m–9h14m. Results vs `_laststage_nodetmap` baseline (L2=0.5153 / CR=0.086%): `tempstack2` (3322355) `L2=0.5614 / CR=0.065%`, `tempstack3` (3322356) `L2=0.5495 / CR=0.059%`, `tempstack3_noegocomp` (3322357) `L2=0.6345 / CR=0.071%`. All three regress L2 (Δ +0.034 to +0.119) with only marginal CR gain. The `noegocomp` collapse confirms ego compensation is load-bearing when stacking temporal features. T2.6 does not compete with `_laststage_nodetmap_decoder6_planwp` (L2=0.5150 / CR=0.068%) at stage 2 — **closed out as a stage-2 lever**, but flagged for stage-1 (see below): the failure mechanism is plausibly that the stage-1 backbone wasn't trained with temporal stacking, so per-frame features don't compose; retraining stage-1 with temporal-stacking-aware supervision and re-evaluating the stack at stage 2 is a separate hypothesis that this experiment did not test.
- [ ] Temporal image-feature stacking at **stage 1** — train a stage-1 backbone with temporal stacking in-the-loop (planning/motion losses operate on stacked features), then load into the existing K/V-off stage-2 (`_laststage_nodetmap_decoder6_planwp`) and re-eval. Hypothesis: the stage-2 T2.6 regression came from feature mismatch (single-frame-trained backbone fed multi-frame at stage 2), not from stacking being intrinsically unhelpful. Add to the mix of stage-1 levers alongside aux2d_dino, aux2d_dseg, and the egoonly diagnostic.
- [x] Decoder depth ablation — `_laststage_nodetmap_decoder6` (Killarney 3319224, 9h10m). Doubling decoder layers 3→6: **`L2=0.5194 / CR=0.077% / NDS=0.5265 / mAP=0.4140 / mAP_normal=0.5527`** — ties baseline L2 (ΔL2=+0.004) and modestly improves CR (0.086% → 0.077%). Modest as a standalone lever but stacks cleanly with the T2.5+laststage variant.
- [x] Decoder6 + all-waypoint stack — `_laststage_nodetmap_decoder6_planwp`. **Seed 0** (Killarney 3319227, 9h17m): `L2=0.5150 / CR=0.068% / NDS=0.5275 / mAP=0.4164 / mAP_normal=0.5559`. **Seed 1** (Killarney 3366270, 9h18m): `L2=0.5100 / CR=0.087% / NDS=0.5209 / mAP=0.4078 / mAP_normal=0.5515`. **Mean L2 = 0.5125**, well below K/V-on `_laststage` reference (0.5302) — L2 win confirmed across seeds. **CR did not reproduce** (0.068% → 0.087%): the original "recovers ~½ K/V-removal CR cost" claim was a single-seed artifact; CR averaged across seeds is essentially K/V-off baseline (0.086%). Reframing: this stack confirms the L2 claim; CR fix requires a different mechanism (see evalmatchmode stack below).
- [x] **`_decoder6_planwp_evalmatchmode`** (Killarney 3366620, 9h45m). Stacks the K/V-off architecture with the v4 evalmatchmode learned-rescore head: **`L2=0.5204 / CR=0.046% / NDS=0.5248 / mAP=0.4137 / mAP_normal=0.5531`**. **First K/V-off variant to close both L2 and CR gaps to K/V-on**: L2 within noise of K/V-off baseline; **CR matches/beats the K/V-on `_laststage` reference (0.054%)**. Contradicts Exp 7's conclusion that the learned scorer adds nothing — Exp 7 was on K/V-*on* where hard rescore was already saturated; on K/V-*off* the learned head recovers the CR signal that hard rescore loses when det K/V is removed. Single-seed; needs reproduction. Promotes from "drop the learned cost option" (Exp 7) back to "viable in the K/V-off setting" — separate context from K/V-on.
- [x] **Seed-1 reproduction of `_decoder6_planwp_evalmatchmode` — REPRODUCES.** Killarney **3377724** completed (L2=0.5087 / CR=0.053% / NDS=0.5222 / mAP=0.4085 / mAP_normal=0.5549). Mean across seeds (3366620 + 3377724): **L2=0.5145 / CR=0.0495%**. Both seeds beat K/V-on `_laststage` hard rescore (CR=0.054%); ΔCR=0.007 pp well within the 0.015 pp noise floor. Contribution 4 is now defensible. DGX hedge 3702 still running — cancel once it lands or keep as third seed for further confidence.
- [x] **`_decoder6_planwp_evalmatchmode_egostatus` — NEW HEADLINE.** Killarney **3376689** completed: **L2=0.3720 / CR=0.044% / NDS=0.5253 / mAP=0.4125 / mAP_normal=0.5535**. ΔL2=−0.148 vs evalmatchmode seed 0; matches the K/V-on egostatus L2 win (0.4988 → 0.3701, ΔL2=−0.13) — egostatus transfers cleanly across architectures. The K/V-off paper architecture now beats every prior K/V-on result on both L2 and CR (single seed). Promotes from "fold-in" to **the paper's main quantitative claim**.
- [x] **Seed-1 reproduction of `_evalmatchmode_egostatus` — REPRODUCES.** Killarney **3386471** completed: **L2=0.3695 / CR=0.037% / NDS=0.5211 / mAP=0.4063 / mAP_normal=0.5526**. Mean across seeds (3376689 + 3386471): **L2=0.3708 / CR=0.0405%**. ΔL2=0.0025 (below 0.007 noise floor), ΔCR=0.007 pp (within 0.015 pp noise floor). **Paper main quantitative claim locked across seeds.**
- [x] **Stream B1 — `_decoder6_planwp_evalmatchmode_imgconfdet`** Killarney **3376418** completed: **L2=0.5119 / CR=0.047% / NDS=0.5257 / mAP=0.4128 / mAP_normal=0.5588**. Ties evalmatchmode within noise (ΔL2=−0.009, ΔCR=+0.001 pp) — **the conflict head's per-agent token is not load-bearing**; image features sampled at det BEV positions carry the same collision signal. Unlocks Stream C (drop det/motion queries from the joint decoder) and brings the "zero-dependence stage 2" claim within reach.
- [ ] Wait for Arm B s2 to finalize the joint-stage-1 result; Arm B s1 full Apollo eval landed (`L2=0.6428`, `obj_box_col=0.104%`, `NDS=0.5216`, `mAP=0.4051`, `mAP_normal=0.5816`) with no positive early signal
- [ ] **Joint stage-1 with `detach_perception=False`** (Apollo, tmux session `joint_nodetach`, started 2026-05-02 03:24 UTC, container `61e0ad6cfcaf`). Clones `_joint` (Arm B, modern head, `detach_perception=True` → L2=0.6948 at stage-2 transfer) and flips the single bit. Closes the long-flagged untested hypothesis: planning gradients flowing back through det/map heads at stage 1. Apples-to-apples with Arm B. ~24-32 h ETA on 8×V100. If non-detach beats Arm B at stage-2 transfer, joint-stage-1 thread is alive; if it also regresses, joint-stage-1 closes structurally.
- [ ] Pull `softrescore_w*` metrics from `plan_scoring`
- [ ] Decide whether the paper keeps a lightweight motion head for rescore or replaces rescore with a learned planner-internal cost
- [ ] Prepare baselines: SparseDrive default, our current best (`ptaux2d_ppdeformmm_planifls_planinstfeat_laststage`), and one external end-to-end baseline (UniAD or VAD)
- [x] Batch F: single-channel K/V disambiguation. F1 `_laststage_nodetkv` (Killarney 3314590) → L2=0.5466 / CR=0.125% (regression on both); F2 `_laststage_s2nomap` (3314591) → L2=0.5266 / CR=0.069% (tied with baseline); F3 `_laststage_nomapkv` (3314708) → L2=0.5246 / CR=0.047% (tied with baseline). Outcome: map K/V is dispensable, det K/V is not, and `_laststage_nodetmap`'s L2 win is a joint-removal artifact. Headline reframes toward "stage-2 map task dispensable" — see `2026_04_26_plan_gaps.md` Batch F outcome.
- [x] Batch G eval-only diagnostics on DGX (2-GPU). Re-IDs after queue rejection of 3660–3662: G1 `_laststage_motionconf` (3663) → L2=0.5184 / CR=0.084%; G2 `_laststage_norescore` (3664) → L2=0.5188 / CR=0.104%; G3 `_planifls_motionconf` (3665) → L2=0.5036 / CR=0.089%; G4 `_laststage_nodetmap_norescore` (3666) → L2=0.5131 / CR=0.089%. Outcome: motion-confidence rescore lies between hard and none on CR (not a substitute for det-confidence); the K/V-removal CR cost is rescore-recovery failure, not planner-output regression. See `2026_04_26_plan_gaps.md` Batch G outcome.
- [x] Stage-1 aux2d Trillium runs (`aux2d_dino` 472496, `aux2d_dseg` 472563) hit 24 h TIMEOUT at iter 70320 = **epoch 60/100**. Resubmitted 2026-04-29 as **474792** (dino) / **474793** (dseg) — both also TIMED OUT at 23:59:00 without reaching the epoch-80 save point. Second resume as **476862** (dino) / **476863** (dseg) on 2026-05-01:
  - **`aux2d_dino` (476862, COMPLETED 2026-05-01 21:15 UTC, 14h19m)** — ran to iter 117198/117200. Final eval (epoch 100, `iter_117200.pth`): **mAP=0.4257 / NDS=0.5343 / mAP_normal=0.5364**. Penultimate eval (epoch 80, `iter_93760.pth`): mAP=0.4147 / NDS=0.5268 / mAP_normal=0.5360.
  - **`aux2d_dseg` (476863, CANCELLED 2026-05-02 06:38 UTC after process hung 11 h)** — final saved checkpoint is `iter_93760.pth` (epoch 80) with eval **mAP=0.4282 / NDS=0.5382 / mAP_normal=0.5533**. The training process froze at iter 97155 with no log/checkpoint writes for 11 h before cancel; no later snapshot exists.
  - **Comparison vs canonical `aux2d` baseline (Apollo 8gpu_noflash, mAP=0.437 / NDS=0.546 / mAP_normal=0.570):** dino-init at epoch 100 underperforms baseline by −0.011 / −0.012 / −0.034. dseg at epoch 80 underperforms by −0.009 / −0.008 / −0.017 (smaller gap; would likely close further at epoch 100 but won't be measured). Neither is a clean stage-1 win on perception metrics.
  - **Caveat:** these are 4gpu_bs24 runs (total bs=24) vs the canonical 8gpu_noflash baseline (total bs=64). A same-config plain `_4gpu_bs24_aux2d` baseline doesn't exist; without it we can't fully separate "dino/dseg lever doesn't help" from "4gpu_bs24 has a lower ceiling than 8gpu". The decisive read is still a stage-2 follow-up reading L2/CR.
- [ ] Arm B stage-2 (`ptjoint_planpredtrajdeformmm`, Killarney 3314809) completed → L2=0.6948 / CR=0.124% / NDS=0.4785 / mAP=0.3713 / mAP_normal=0.5318. Worse than Arm A (`ptjointdetach_planpredtrajdeformmm` L2=0.6669) on both planning and perception. **Important caveat:** the Arm B stage-1 config (`sparsedrive_r50_stage1_8gpu_noflash_joint.py:535`) sets `detach_perception=True`, identical to the older `detach_det=True` flag in Arm A's stage-1. So Arm B is *not* a non-detach experiment — it is "modern head + detach" vs Arm A's "legacy head + detach." The non-detach hypothesis (planning gradients flowing back through det/map heads) has not been tested. To actually test it, train a stage-1 with `detach_perception=False`.
- [x] Killarney `planaux_conf_evalmatch` (3314427) — eval-match retrain of `planaux_conf_anchorlabel` → L2=0.5355 / CR=0.178% / NDS=0.5472 / mAP=0.4428 / mAP_normal=0.5600. Recovers most of the original regression (L2 0.7692 → 0.5355) but still worse than hard-rescore baseline (L2 0.4988); learned scorer remains a null path.
- [x] Exp 5 — threshold sweep + selector aggregation on the evalmatch ckpt (DGX 2-GPU eval-only, 3675–3682). T-sweep traces a CR U-curve with min at T=0.90 (`L2=0.5399 / CR=0.122%`); top-k regresses CR (`topk2 0.159% / topk3 0.173%`); det-weighted is the strongest eval-only fix at `L2=0.5370 / CR=0.119%`. None match hard rescore (0.063%). See `2026_04_26_plan_scoring.md` Exp 5.
- [x] Exp 6 — `planaux_conf_evalmatchmode` (Killarney 3319213, 7.6h retrain). Per-mode aggregated BCE with smooth-max(τ=5) over matched anchors; matches the inference `any(anchor)` reduction at training time. Result: `L2=0.5328 / CR=0.080% / NDS=0.5434 / mAP=0.4376 / mAP_normal=0.5587`. First learned scorer to come within ~0.02pp CR of hard rescore. Classifier F1 ≈ 0.75 (vs evalmatch v3's 0.36). See `2026_04_26_plan_scoring.md` Exp 6.
- [x] Exp 7 — v4 followups: threshold + selector aggregation + hybrid_or, parallel on DGX (3683–3691) and Killarney (3324524–3324532). Threshold sweep on v4 is **flat** (L2 ∈ [0.5316, 0.5350], CR ∈ [0.080, 0.101]%) — train/eval-alignment fix already calibrated the head, so T-tuning has nothing left to do. Aggregation variants (top-k, detweighted) no longer help. **`hybrid_or` (OR of hard rescore + v4) hits CR=0.063% — exactly matching hard rescore alone**, while L2 stays at v4's 0.5330. The learned head's correct rejections are a subset of hard rescore's; the head adds no unique collision-avoidance signal beyond the geometric check. Closes the learned-scorer thread on the K/V-on architecture. **Caveat (2026-04-30):** the `_decoder6_planwp_evalmatchmode` result (CR=0.046%) shows the learned head *does* add unique signal on the K/V-*off* architecture, where hard rescore loses signal due to det K/V removal. So the "drop the learned cost option" conclusion applies only to K/V-on. See `2026_04_26_plan_scoring.md` Exp 7.
- [x] Anchor-capacity batch (`2026_04_30_anchor_capacity.md`). 5 variants on `ppdeformmm_planifls` baseline + 1 no-rescore eval, submitted Killarney 2026-04-30 (3368761–3368766). Cluster-wide failure at 19:20 UTC killed mid-eval on jobs 3368764, 3368765 — checkpoints saved, eval-only resubmitted as 3376375/3376376; those two were mis-allocated at 4 GPUs and re-resubmitted on 2 GPUs (`--gres=gpu:l40s:2 --cpus-per-task=8 --mem=60G`) as **3377754** (`_endptnorm_mag_egostatus`) and **3377755** (`_modesspeedstrat_perbucketreg`).
  - **`egostatus`** (3368763): MLP-encode `ego_status[:,[0,1,5,6,7]]` → broadcast-add to plan_mode_query at init + each per-stage rebuild. **L2 0.4988 → 0.3701 (−0.13)**, CR 0.063% → 0.058%. Decisive L2 win; first confirmation that current-frame ego state was missing from the planner (only one-step-lagged longitudinal velocity reached the head via `ego_anchor_embed`).
  - **`shapeanchor_velnorm`** (3368762): per-batch `||v_0||` scaling on velocity-normalized k-means anchors. L2 0.4988 → 0.5215 (+0.02), CR 0.063% → 0.058%. Roughly tied; the multiply-by-zero failure mode at v_0≈0 limits standalone gain.
  - **`shapeanchor_endptnorm_mag`** (3368761): blew up to L2=1.346, CR=0.64%. Refmag-floor patch retry (Killarney **3376688**, completed 2026-05-01) **also blew up: L2=1.1124 / CR=0.392%**. Stopped-Straight cluster handling needs a deeper fix; closing this branch.
  - **`shapeanchor_endptnorm_mag_egostatus`** (variant 3, eval-only **3377754**, completed): **L2=0.4120 / CR=0.100%**. egostatus partially rescues the endptnorm_mag variant (1.346 → 0.412) but does not beat plain egostatus alone (0.3701) — the anchor change adds no value once egostatus is in.
  - **`modesspeedstrat_perbucketreg`** (variant 4, eval-only **3377755**, completed): **L2=0.5415 / CR=0.339%**. L2 close to plain speedstrat (0.5548) but CR blows up — per-bucket regression branch doesn't cleanly decouple decoder capacity. Variant 4 closes negatively; the speedstrat L2 regression remains an open question.
  - **`modesspeedstrat_norescore`** (3368766, eval-only on existing speedstrat ckpt): L2=0.5560, CR=0.060% vs rescore-on baseline 0.5548/0.070%. ΔL2=+0.001 (noise) → **speedstrat L2 regression is training-time, not decode-time** → variant 4's per-bucket reg branch is the right intervention.
- [x] **Stream A — Stage-2 zero-loss diagnostic trio: ALL THREE TIE BASELINE; stage-2 perception losses are empty for planning.** Reference baseline `_decoder6_planwp_evalmatchmode` seed-0 (L2=0.5204 / CR=0.046%, the right comparator since these configs don't include egostatus):
  - `_s2nodetloss` (Killarney **3386472**): **L2=0.5265 / CR=0.076% / NDS=0.0489 / mAP=0.0061 / mAP_normal=0.5510**. ΔL2=+0.006 (within noise), ΔCR=+0.030 pp. Detection collapses (mAP 0.006) confirming loss zeroing took.
  - `_s2nomaploss` (Killarney **3386473**): **L2=0.5161 / CR=0.063% / NDS=0.5242 / mAP=0.4160 / mAP_normal=0.3095**. ΔL2=−0.004 (within noise), ΔCR=+0.017 pp. Map collapses (mAP_normal 0.31).
  - `_s2nomotionloss` (Killarney **3386474**): **L2=0.5132 / CR=0.042% / NDS=0.5263 / mAP=0.4132 / mAP_normal=0.5519**. ΔL2=−0.007 (at noise edge), ΔCR=−0.004 pp. Motion collapses (car_ade 4.17).
  - **Outcome: all three planning L2s sit at parity with the seed-0 baseline (max |ΔL2|=0.007); CRs vary by ≤0.030 pp.** First direct test of the supervision-not-interface claim — stage-2 perception losses are not shaping the backbone in any way that matters for planning, given a strong stage-1 init. Stream E (frozen-backbone S2) is the natural next beat.
- [x] **Stream C — `_decoder6_planwp_evalmatchmode_streamc` — RESULT IN HAND.** Training: Killarney 3389115 (COMPLETED to iter 11679/11720). Eval: Killarney **3394847** (1h16m eval-only, COMPLETED 2026-05-02): **L2=0.3597 / CR=0.026% / NDS=0.5245 / mAP_normal=0.5556**. Combines egostatus + image_at_det + ego_only_planning=True (agent slots dropped). ΔL2=−0.011 vs headline mean (marginally outside noise floor, single-seed); ΔCR=−0.015 pp (at noise floor). **Ties or marginally beats the headline — dropping all agent slots from the planner does not hurt performance.** Submit a second seed to confirm before locking. Motion metrics absent by design.

## New cluster notes (2026-05-03)

- **Tamia and rorqual are missing commit `95db051` (SparseBox3DKeyPointsGenerator dtype fix).** `_evalmatchmode` results from these clusters are valid (rorqual L2=0.5160/CR=0.052%, tamia L2=0.5248/CR=0.071% — within cross-cluster noise). **`_egostatus` results are invalid**: CR is 3–6× expected (rorqual 0.131%, tamia 0.243% vs expected ~0.040%) due to the missing dtype fix. `git pull` required before re-running egostatus on these clusters.
- **B1.5 (`_minS2_B1p5`, killarney 3398019) and Option-2a (`_minS2_2a_noaux`, killarney 3398319) both crashed at startup (~10 min).** Tracebacks not captured — need to check error logs directly on killarney to diagnose.

## Naming convention

Two distinct interventions get conflated under "nomap" / "nodet" in older configs:
- **head-off**: head module not built. No perception loss → no backbone gradient pressure at that stage → also no K/V to planner. Examples: `stage1_nomap_dn_rotaug` (s1 map head gone), `stage2_4gpu_nomap` (s2 map head gone).
- **K/V-off**: head trained as usual (gradient pressure flows back into backbone), but the planner's `cross_gnn`/`gnn` op is nulled so tokens don't reach the planner. Examples: `_nodetmap` family (`skip_perception_kv=True`).

`skip_perception_kv` now accepts `'det'` (null only `gnn` op — det K/V), `'map'` (null only `cross_gnn` op — map K/V), `True`/`'both'` (legacy, both), or `False` (default). Reports should use "head-off" vs "K/V-off" to disambiguate.

## Working thesis

**Sparse scene representations are interface-redundant but supervision-essential for planning.** Detection and map tokens carry almost no information into the planner at inference time, but their *training signal* shapes the image-feature representation that the planner does read. The dual role gets conflated in current end-to-end stacks; separating it gives us a perception-free inference path with no quality cost, and frees stage-1 to use whatever supervision shapes image features best (perception, dense auxes, depth, drivable-area).

The strong-supported claim is the **stage-2 / inference-path** one: the perception → planner K/V channel carries little information (4-cell controlled grid plus ~15 targeted negatives). The **stage-1 contribution is additive, not subtractive**: planner-aware dense auxes (drivable-area, occupancy, future-flow) extend the existing aux2d insight on top of det+map. We do *not* claim that perception decoders are unnecessary at stage 1 — prior evidence (see "Nomap evidence" below) shows that removing map at stage 1 is catastrophic.

This is a structural shift relative to the standard end-to-end stack (UniAD, VAD, SparseDrive, ParaDrive), which treats perception decoders as the carrier of agent and map information into the planner. Our evidence shows that carrier is essentially empty *at inference*; what matters is the gradient flow at training time, which we extend without removing.

## Why this framing is stronger than "we beat the baseline"

The originally drafted NeurIPS direction (`reports/2026_04_22_nuerips26_paper.md`) framed the problem as "planning requires a planning-sufficient representation; detection salience ≠ planning relevance." Our subsequent experiments **falsified the specific selection-based instantiation of that thesis**:

- `selrelGT` (relevance-ranked top-k with ground-truth corridor labels) tied baseline.
- `topk_half` (confidence top-k at half the count) tied baseline.
- `nodetmap` (zero perception K/V) regressed by only L2=0.023 / CR=0.029pp at the time; the K/V-off stack with the right architecture (`_laststage_nodetmap_decoder6_planwp`) now ties baseline outright.

If selection criterion does not move planning at full / half / oracle / zero count, the right reading is not "we need a better selection mechanism." It is that **the sparse-scene → planning channel is essentially redundant as an interface but load-bearing as supervision**. The paper's contribution is the disentanglement: we separate the two roles empirically, then show the inference-time interface can be removed entirely without quality cost as long as the training-time supervision is preserved (or replaced with planning-relevant alternatives).

The originally framed thesis ("planning-critical representation learning") survives in sharper form: the planning-critical representation lives in the **image features**, and Stage 1 should be designed to shape them — perception is one effective option among several.

## Empirical foundation

### Negative evidence on the perception → planner K/V interface

| Class | Experiments | Outcome |
|---|---|---|
| Token count | `numdemapx2` (50→100), `topk_half` (50→25, 10→5), `alldet` (≈900), `nodetmap` (0,0) | Null between 25 and ~900; small regression at 0 |
| Selection criterion | `selrelGT` (oracle planning-relevance), `selrelGT_half` (running) | `selrelGT` ties baseline |
| Bidirectional flow | `bidir` (reverse cross-attn) | Regression |
| Perception-side reshape | `detrel`, `planaux_da`, `planaux_conf` | All regressions |
| Mode-bag perturbations | `modeproj`, `mmuniform`, `modes20diverse`, `noagg`, `modetime36q`, `modetime36q_timeattn`, `mode_softtgt`, `instfeataddls` | All regressions |

Total: ~15 controlled negatives plus 4-cell count grid. The K/V interface is saturated.

### Positive evidence on the image-features path

- `planpredtrajdeformmm` (planner's own deformable cross-attention to image features): main contributor to L2 0.636 → 0.522.
- `planifls` (last-stage ego instfeat replacement, sourced via the image-features deformable): the L2 lever in current best.
- `aux2d` depth supervision at stage 1: shapes image features the planner directly reads; major contribution to current best.

Every confirmed positive lever in the planning-refinement era touches the image-features path. None of the perception-tokens-path levers have moved planning.

### Stage-1 alignment evidence

- aux2d (non-planning supervision shaping the backbone): positive.
- joint_detach stage-1 ckpt + modern stage-2 head (Arm A, Killarney 3301176): regressed L2 by 0.125.
- joint_detach stage-1 ckpt + matched legacy stage-2 head (Arm A-matched, Killarney 3305025): regressed L2 by 0.059.
- Modern-head joint stage-1 (Arm B): full Apollo stage-1 eval landed at `L2=0.6428`, `obj_box_col=0.104%`, `NDS=0.5216`, `mAP=0.4051`, `mAP_normal=0.5816`, `AMOTA=0.3856`, `IDS=585`; motion metrics are `car_ade=0.6571`, `ped_ade=0.7286`, `car_epa=0.4921`, `ped_epa=0.4086`. Stage-2 follow-up remains the actual planning test.

The matched run rules out head-mismatch as the cause. Joint planning supervision at stage 1 with `detach_perception=True` does not produce a useful planning init. Arm B's stage-1 eval improves planning over `joint_detach` but does not recover perception or planning to the plain baseline; the conclusion still waits on Arm B stage 2. If that also regresses, **stage-1 alignment via planning loss does not work; auxiliary tasks that shape image features without competing for backbone capacity (e.g. aux2d) do.**

### Nomap evidence — perception-decoder removal at training time is risky

Two existing data points constrain "how aggressively can we remove perception decoders at stage 1":

- **Both stages nomap is catastrophic.** `pretrainv1_noflash_nomap` and the broader `nomap_dn_rotaug` family from `2026_03_05_map_removal.md`: stage 1 *and* stage 2 without the map head produced L2=6.612 (~10× regression) and obj_box_col=3.605%. This is the strongest negative on perception-decoder removal.
- **Stage-2-only nomap is *positive*.** Removing map only at stage 2 (keeping `sparsedrive_stage1.pth`'s map-trained backbone) on R50 *improves* planning L2 and CR slightly. So the catastrophic case is specifically the combination of "no map at training-time backbone shaping AND no map at inference."
- **The diagnostic cell is untested**: stage-1 nomap *with* stage-2 *retaining* map. This is what experiment #7 below resolves.

Reading: **at minimum, perception-task supervision at stage 1 is doing useful work somewhere.** The catastrophic both-nomap case rules out the most aggressive removal. Whether the load-bearing component is (a) backbone shaping by map-task gradients or (b) map-head warm-init flowing into stage 2 is currently confounded; #7 is designed to disambiguate.

**#7 result (Killarney 3311181, `ptnomapdnrot_ppdeformmm_planifls`).** Stage-1 trained without map (`sparsedrive_stage1_nomap_dn_rotaug.pth`) but stage-2 *with* map head (init from scratch). Result: `L2=0.5203, obj_box_col=0.048%, NDS=0.5502, mAP=0.4471, mAP_normal=0.2411` (car_ade=0.6300, ped_ade=0.6849, car_epa=0.5205, ped_epa=0.4561). Planning is on par with the project-best `_planinstfeat_laststage` reference (`L2=0.522 / 0.047%`); detection is slightly stronger (`NDS 0.5502 vs 0.5570 reference`, `mAP 0.4471 vs 0.4150`). The map head is broken (`mAP_normal 0.2411` vs `~0.55–0.58` family) because it was initialized from scratch when the backbone never saw map gradients — the stage-2 `~10` epochs of map training cannot recover what the stage-1 `~100` epochs would have established. **Disambiguation outcome: the load-bearing component for planning is (a) backbone shaping, not (b) map-head warm-init.** Removing map at stage 1 does not catastrophically hurt planning when the map head is restored at stage 2 and at inference. The catastrophic both-nomap case from `2026_03_05_map_removal.md` is therefore primarily about *inference-time absence of the map head*, not about backbone shaping. This unlocks the paper's core claim: a stage-1 designed around image-feature shaping (no map decoder needed) plus a stage-2 that retains map for inference is a feasible architecture, and the perception-K/V channel can stay or go without breaking planning.

Implication for the paper framing: the **stage-1 contribution is additive, not subtractive.** We do not propose removing perception decoders at stage 1 — we propose *augmenting* them with planner-aware auxes and showing the augmentation compounds with the no-K/V stage 2.

## Paper contributions (developed)

The five top-line contributions, with the supporting evidence and remaining work for each.

1. **Hidden redundancy in sparse-scene interfaces.** Controlled ablation grid on the perception → planner K/V channel: 4-cell token-count grid (0/25/50/100/900), 3 selection criteria (top-k, oracle relevance, learned), bidirectional flow, and 6 supervision/coupling perturbations. All null or regressive. The interface has no measurable effect on planning. **Status: locked.**
2. **The real bottleneck is image-feature alignment.** Every confirmed positive lever in the planning era touches the image-feature path: `planpredtrajdeformmm`, `planifls` (last-stage ego instfeat from image features), aux2d depth supervision. None of the perception-tokens-path levers move planning. **Status: locked. Ongoing reinforcement** from stage-1 lever runs (DINOv2-init, dseg, drivable-area aux).
3. **Decoupling supervisory and interface roles.** Stage-1 nomap + stage-2 with map preserves planning (`L2=0.5203`); both-stage nomap is catastrophic (`L2=6.6`). This isolates the load-bearing role of perception to **stage-1 backbone shaping**, not stage-2 inference. **Status: confirmed by `_ptnomapdnrot_ppdeformmm_planifls`** (Killarney 3311181).
4. **A reusable abstraction: scene representations as supervision without inference.** Concrete instantiation is the K/V-off stage-2 architecture: `with_perception_kv=False`, perception heads kept for training-time gradient flow only. Best stack is now `_laststage_nodetmap_decoder6_planwp_evalmatchmode` (Killarney 3366620) at `L2=0.5204 / CR=0.046%` — **first K/V-off variant to close both L2 and CR gaps to K/V-on** (`_laststage` reference: 0.5302 / 0.054%). The base `_decoder6_planwp` stack reproduced on L2 across seeds (0.5150 / 0.5100, mean 0.5125) but CR did not (0.068 / 0.087); the learned rescore on top recovers the CR signal that hard rescore loses when det K/V is removed. **Status: L2 confirmed across seeds; CR currently rests on the single-seed evalmatchmode-stack run — needs reproduction before locking as headline.**
5. **Simplify without sacrificing performance.** On nuScenes, the K/V-off stack matches the K/V-on `_planinstfeat_laststage` reference (L2=0.522) within noise while removing the perception cross-attention computation at inference. Lower inference cost, fewer moving parts. **Status: matched at L2; CR closes ~½ the K/V-on gap**; full headline benchmark vs SparseDrive default + UniAD/VAD + a 2nd dataset (NavSim) is the remaining work.

The paper's main claim does not depend on a clean stage-1 win. If the additive-aux experiments (DINO-init, dseg, dadense) are flat, contributions 1–4 still stand; contribution 5 is supported by what's already in hand.

## Architecture sketch

### Inference path (perception-free)
```
multi-camera images
        │
        ▼
   ResNet-50 backbone  (stage-1-pretrained for planning-relevant features)
        │
        ▼
       FPN  ─ planning-relevant image features
        │
        ▼
   Planning Decoder (reads image features directly)
   ─ ego query init (from temporal queue + ego state)
   ─ N stages of [temp_gnn, deformable_to_image, ffn, refine]
   ─ NO perception K/V cross-attention
   ─ all-waypoint deformable + laststage instfeat (best K/V-off stack)
        │
        ▼
   Multi-mode trajectory + per-mode confidence
        │
        ▼
   Hard-rescore for CR (lightweight motion head as aux supervisor)
```

### Training path
```
                       ┌─ det head        ─ aux loss (gradient → backbone)
multi-camera images ─► backbone + FPN ─┤─ map head        ─ aux loss
                                       ├─ motion head     ─ aux loss (also feeds rescore at inference if retained)
                                       └─ planning head   ─ primary loss
```

Perception heads are auxiliary supervisors. They are not on the inference forward path to ego trajectory; they exist for evaluation reporting (NDS, mAP) and for shaping the backbone.

### Key architectural decisions

1. **Planning decoder depth and width** — locked at 6 stages × 1 deformable based on Killarney 3319227 result. Doubling depth to 6 + adding all-waypoint deformable + retaining laststage instfeat is the K/V-off stack that matches K/V-on baseline.
2. **All-waypoint planning deformable (T2.5)** — locked **with laststage instfeat retained**. The earlier `planwp_full6` regression was a confound: dropping laststage instfeat alongside the waypoint change. The K-pool variant (`_planwp_full6_ls`) with both enabled ties baseline L2 and improves CR.
3. **Temporal image-feature stacking** (T2.6) — closed out as a stage-2 lever (regresses L2 by 0.034–0.046). Open as a stage-1 lever: train the backbone with temporal stacking in the loop so per-frame features compose, then load into the K/V-off stage-2.
4. **Rescore: hard rescore is locked.** Exp 7 hybrid_or matched hard rescore exactly (CR=0.063%) — the learned head adds no unique collision-avoidance signal. The lightweight motion head stays as a training-time aux + inference-time rescore input. The "learned cost replaces rescore" option is dropped.

## Stage-1 batch — ranked by paper EV (additive framing)

Each new stage-1 train is followed by an identical stage-2 best recipe so the only varying input is the stage-1 backbone. Stage 2 keeps perception decoders for safety (the fully-removed version is the no-K/V stage-2 architecture experiment, separate from this batch).

| # | Stage-1 recipe | Tests | Cost | Paper role |
|---|---|---|---|---|
| **1** | **DINO-init stage-1** (R50 backbone init from DINOv1 SSL weights instead of ImageNet; det+map+aux2d unchanged) | does a stronger SSL prior on the backbone lift planning? | 1× ~32h | tests "image-feature quality" lever; Trillium 4-GPU bs=24 |
| **2** | **`dadense_aux2d`** — drivable-area BEV dense aux, *added* on top of det+map+aux2d | does a planner-relevant dense aux compound with depth? | ~1d label-gen; 1× ~32h | first additive planner-aware aux |
| **3** | **`occupancy_aux2d`** — class-conditional BEV occupancy, *added* on top of det+map+aux2d | dense agent supervision without box decoding | ~1–2d label-gen; 1× ~32h | second additive planner-aware aux |
| **4** | **`dadense + occupancy + aux2d`** | full additive aux suite | trivial after #2,#3; 1× ~32h | strongest stage-1 recipe; headline number |
| **5** | **DINO-init + best additive stage-1** | best-of-both | 1× ~32h | optional further-best |
| **6** | **Arm B (`stage1_8gpu_noflash_joint`)** modernized joint-stage-1 — s1 full eval landed; s2 pending | locks in negative for "joint planning supervision at stage 1" | stage-1 done; stage-2 follow-up still needed | closes joint-stage-1 line for the paper |
| **7** | **`stage1_nomap_dn_rotaug` ckpt + stage-2 *with* map** (no new stage-1 train; already have ckpt) | resolves backbone-shaping vs inference-path confound from prior nomap evidence | 1× stage-2 only ~12h | side study / appendix — **DONE**: Killarney 3311181, see results below |
| **8** | **`stage1_nodet`** + stage-2 with det | analog of #7 for detection | 1× ~32h | side study / appendix |
| **9** | **`aux2d_only`** (no det, no map at stage 1) + stage-2 with full perception heads | extreme replacement test, *with* stage 2 keeping perception decoders for safety | 1× ~32h | appendix; gated on #7 outcome |

Compute budget for #1–#5: ~5 stage-1 trains × ~32h each = ~160 GPU-days, plus #4 occupancy label-gen. Feasible across 2 weeks if pipelined across Apollo, Killarney, Trillium.

## Stage-2 architecture experiments

| Experiment | Tier | Cost | Expected EV |
|---|---|---|---|
| DINOv2-init with current best Stage-2 recipe | 3 | ~12h | high (free image-feature lift) |
| All-waypoint planning deformable (T2.5) | 2 | ~12h | medium |
| 6-stage planning decoder (T2.1) | 2 | ~12h | medium |
| 2 deformable-per-stage (T2.2) | 2 | ~12h | low-medium |
| Temporal image-feature stacking (T2.6) | 2 | ~12h + 2d impl | high |
| End-to-end no-perception-K/V Stage 2 (paper headline) | core | ~12h + 1w impl | central |
| Learned collision cost replacing rescore | core (optional) | ~12h + 1w impl | architectural elegance |

## Baselines

- **SparseDrive default** (already have): L2=0.636, obj_box_col=0.133%.
- **Our current best K/V-on** `ptaux2d_ppdeformmm_planifls_planinstfeat_laststage`: L2=0.522, CR=0.047%.
- **K/V-off stage-2 (proposed)** `_laststage_nodetmap_decoder6_planwp`: L2=0.5150, CR=0.068% — already in hand. Matches K/V-on at L2 within noise; CR closes ~½ the gap.
- **External: UniAD or VAD on nuScenes** for cross-method comparison. Required for NeurIPS-grade reviewer satisfaction even though our ablation grid is the primary evidence.
- **Second benchmark: NavSim**. Bringing this up alongside the nuScenes story is in scope for the paper. NavSim tests whether the "image features carry planning" claim generalizes off nuScenes' specific train/eval split and protocol — a known reviewer concern. Start the data-pipeline + eval-runner work in parallel with manuscript drafting.

## Risks

1. **CR regression from removing rescore.** If we drop the motion head entirely and replace rescore with a learned cost, CR may regress more than the L2-side gain warrants. Mitigation: keep motion as a small aux head and retain rescore in the v1 paper architecture; defer the "fully clean" version to future work.
2. **DINOv2 transfer poor.** If DINOv2 features don't transfer well to camera-only autonomous driving (it's pretrained on natural images), the "free image-feature lift" angle weakens. Mitigation: try MAE-style or BEV-pretrained alternatives; the paper's main claim doesn't depend on any specific stage-1 init.
3. **The end-to-end Stage 2 underperforms.** If even with all the planning-deformable-path lifts, the no-K/V architecture loses more than the small nodetmap regression suggested, the paper's main claim is weakened. Mitigation: temporal image-feature stacking (T2.6) is the strongest untried lever; if it does not recover the nodetmap deficit, fall back to a hybrid (lightweight perception K/V + everything else).
4. **Reviewers ask for waymo / argoverse2.** Without longer-range or denser-agent data, the planning-bound claim is hard to argue beyond nuScenes. Mitigation: one Waymo Open Motion or AV2 evaluation as a "generalization study" in the appendix.
5. **Negative-paper fallback.** If the end-to-end Stage 2 doesn't match current best, the paper still has the diagnostic-ablation contribution (Section 1). The framing then becomes "we map the design space of perception → planning coupling and show that ten common interventions do not help." That's a publishable but lower-impact story.

## Decision points and gating

- **Now (week of 2026-04-28):** finalize the architecture sketch and run the two highest-EV cheap experiments (DINOv2-init + current best, and temporal stacking T2.6). Arm B s1 full eval has landed; wait for Arm B s2.
- **+1 week:** Arm B s2 result lands. Decision: if Arm B regresses, the joint-stage-1 negative is locked in; commit to the paper. If Arm B improves, the paper's stage-1 story shifts to "joint stage-1 with the right head + lr is the right pretrain" — different paper, also strong.
- **+2 weeks:** stage-1 auxiliary sweep (~3 backbones in flight). Decide on the best stage-1 aux combination.
- **+3-4 weeks:** end-to-end no-K/V Stage-2 architecture training + tuning.
- **+5-6 weeks:** ablation grid finalized, baselines run, manuscript drafting.
- **+8 weeks:** submission-ready.

NeurIPS 2026 abstract deadline (typical: mid-May) — tight but feasible if we lock the architecture by week 2 and start Stage-2 training by week 3.

## What to do this week — minS2 critical path

**RESULTS (2026-05-03):**

| Job | Config | Outcome |
|---|---|---|
| **3397346** | `_egostatus_minS2` | **L2=0.3563 / CR=0.041%** — ties/marginally beats 3-seed headline mean (0.3692 / 0.040%). **Paper architecture locked.** |
| **3397345 → 3406308** | `_egostatus_s2nopercep` | Training completed; eval crashed on NaN det boxes; eval-only rerun on iter_11720.pth landed: **L2=0.3572 / CR=0.061%** (ΔL2=−0.012 vs headline; CR +0.021 pp at noise edge). Stream A combined alone is empty for planning. |
| **3394848** | `_egostatus_frozenpercep` | **L2=0.3698 / CR=0.044%** — within noise of headline; freezing S2 perception stack is safe |
| **3394847** | `_streamc` eval-only | L2=0.3597 / CR=0.026% (already logged) |
| **3394849** | `_egostatus` seed 2 | **L2=0.3660 / CR=0.039%** — locks 3-seed mean at 0.3692/0.040% |
| **3396706** | `ptaux2d_egostatus` | L2=0.3627 / CR=0.056% |
| **3396707** | `ptaux2d_dino_egostatus` | L2=0.3655 / CR=0.056% |
| **3396708** | `ptaux2d_dseg_egostatus` | **L2=0.3621 / CR=0.043%** — best stage-1 swap; pair with minS2 next |
| **3396709** | `ptnomapdnrot_egostatus` | L2=0.3700 / CR=0.052% |
| **Apollo joint_nodetach** | stage-1 with `detach_perception=False` | Last open stage-1 hypothesis; its stage-2 transfer will also go through minS2 |

**Decision tree (resolved):**
- minS2 ties headline → **paper architecture locked.** Next: submit minS2 seed-2 to confirm the marginal L2 win. Pair best stage-1 (`_ptaux2d_dseg`, 3396708) with minS2 architecture.
- B1.6/B1.7 evals (3401950, 3401952) close the last detection-forward dependency at inference: B1.6 L2=0.3665/CR=0.067%, B1.7 L2=0.3729/CR=0.056%. Both tie L2 within noise but trade ~0.02–0.03 pp CR. Worth a seed-2 if the "fully perception-free at inference" headline is to be claimed.
- Cross-cluster: Streamc reproduces cleanly on Fir (0.3700/0.040%) and Rorqual (0.3727/0.040%); egostatus L2 reproduces but CR elevated outside Killarney — investigate after seed-2 minS2.

## Background notes (older state, kept for context)

1. ~~Confirm the K/V-off headline result. `_laststage_nodetmap_decoder6_planwp` (Killarney 3319227)~~ Superseded — current headline is `_decoder6_planwp_evalmatchmode_egostatus` at L2=0.3708 / CR=0.0405% (mean 2 seeds).
2. **Stage-1 aux2d_dino + aux2d_dseg: BOTH FINISHED (with caveats).** dino reached epoch 100 (Trillium 476862, 2026-05-01 21:15 UTC); dseg hung at epoch 80 and was cancelled (Trillium 476863, 2026-05-02 06:38 UTC) after 11 h of stalled output — `iter_93760.pth` (epoch 80) is the final ckpt. Both **underperform** the canonical aux2d 8gpu_noflash baseline on perception (mAP/NDS/mAP_normal). The 4gpu_bs24 vs 8gpu_noflash config disparity is a confound; without a same-config plain-aux2d baseline we can't fully separate "lever doesn't help" from "smaller batch caps the ceiling." Decisive read is the stage-2 follow-up.
3. **Apollo egoonly stage-1 + stage-2 follow-up: BOTH COMPLETED.** Stage-1 (Apollo, finished 2026-04-30 23:51 UTC, epoch-5 ckpt `iter_58600.pth`): L2=0.6518 / obj_box_col=0.156% / NDS=0.5170 / mAP=0.3989 / mAP_normal=0.5667. Stage-2 follow-up `_ptegoonly_ppdeformmm_planifls` (Killarney **3377809**, completed 2026-05-01): **L2=0.5420 / CR=0.073% / NDS=0.5214 / mAP=0.4046 / mAP_normal=0.5699**. ΔL2=+0.022 vs the direct comparator `_ptnomapdnrot_ppdeformmm_planifls` (#7, L2=0.5203). Minimum-perception stage-1 still produces a viable planning init — supports the supervision-vs-interface decoupling argument but doesn't recover the strongest stage-1 init. Map head still trained at stage 1 here so `mAP_normal=0.5699` (vs #7's 0.2411 init-from-scratch).
   - **Two follow-ups completed 2026-05-02:**
     - `_ptegoonly_ep3_ppdeformmm_planifls` (Killarney **3388945**, COMPLETED): **L2=0.5402 / CR=0.060% / NDS=0.5206 / mAP=0.4033 / mAP_normal=0.5575**. ΔL2=−0.002 vs ep-5 init (0.5420), ΔCR=−0.013 pp. **Stage-1 epoch choice doesn't move stage-2 L2** — planning-best stage-1 snapshot is not a stronger stage-2 init than final-epoch.
     - `_ptegoonly_decoder6_planwp_evalmatchmode_egostatus` (Killarney **3388946**, COMPLETED): **L2=0.3946 / CR=0.055% / NDS=0.5215 / mAP=0.4021 / mAP_normal=0.5617**. ΔL2=+0.023 vs full-perception stage-1 headline (0.3720), CR +0.011 pp. **Egoonly stage-1 nearly recovers the paper headline** — combined with Stream A's zero-loss tie, this decisively supports the supervision-vs-interface decoupling argument: stage-1 needs *some* perception-aware supervision, but stage-2 perception decoders/losses are inert.
4. Add **stage-1 temporal stacking (T2.6 at stage 1)** to the stage-1 lever queue alongside dino/dseg/egoonly — same hypothesis class (image-feature shaping), separate from the closed stage-2 T2.6 result.
5. Begin label-gen for **drivable-area BEV** (Stage-1 #2) — first additive planner-aware dense aux.
6. Begin **NavSim pipeline bring-up** in parallel with manuscript drafting. The cross-benchmark generalization study is in scope for this paper.
7. Wait on Arm B s2; Arm B s1 full Apollo eval produced `L2=0.6428`, `obj_box_col=0.104%`, `NDS=0.5216`, `mAP=0.4051`, `mAP_normal=0.5816`. Caveat noted in TODO: stage-1 still has `detach_perception=True` — actual non-detach hypothesis untested.
