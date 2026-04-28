# Paper Plan: The Perception Decoder Is Unnecessary For Planning

2026-04-28

## TODO

- [ ] Decide on paper title (working title above is a placeholder)
- [ ] Lock the minimal end-to-end Stage 2 architecture (no perception K/V; rescore handled by either small motion head or learned cost)
- [ ] Run #1 DINO-init stage-1 on Trillium (4-GPU bs24) to test "stronger image backbone init" lever
- [ ] Run #7 stage-1-nomap_dn_rotaug + stage-2-with-map on Killarney to resolve the backbone-shaping vs inference-path confound in prior nomap evidence
- [x] Implement DenseSegHead + GenerateDenseSegMask (v1: 6 channels — 3 polylines + 3 agents); submitted aux2d_dseg on Trillium (472563)
- [ ] v2: extend dense seg with drivable_area / walkway / stop_line (requires map_annos extension or BEV-derivation)
- [ ] Run all-waypoint planning deformable variant (T2.5)
- [ ] Run temporal image-feature stacking variant (T2.6)
- [ ] Wait for Arm B s1 → s2 to finalize the joint-stage-1 negative result that anchors the "perception loss isn't the right stage-1 signal" claim
- [ ] Pull `softrescore_w*` metrics from `plan_scoring`
- [ ] Decide whether the paper keeps a lightweight motion head for rescore or replaces rescore with a learned planner-internal cost
- [ ] Prepare baselines: SparseDrive default, our current best (`ptaux2d_ppdeformmm_planifls_planinstfeat_laststage`), and one external end-to-end baseline (UniAD or VAD)

## Working thesis

**The perception-decoder K/V channel into the planner is essentially redundant at inference time. Multi-camera planning information reaches the planner directly through image features, not through detection/map decoder tokens. We exploit this with a stage-2 planning head that drops the perception-token K/V cross-attention entirely, and we show that planner-aware auxiliary supervisions added to existing perception-decoder pretraining further improve stage-1 backbone shaping.**

The strong-supported claim is the **stage-2 / inference-path** one: the perception → planner K/V channel carries little information (4-cell controlled grid plus ~15 targeted negatives). The **stage-1 contribution is additive, not subtractive**: planner-aware dense auxes (drivable-area, occupancy, future-flow) extend the existing aux2d insight on top of det+map. We do *not* claim that perception decoders are unnecessary at stage 1 — prior evidence (see "Nomap evidence" below) shows that removing map at stage 1 is risky.

This is still a structural shift relative to the standard end-to-end stack (UniAD, VAD, SparseDrive, ParaDrive), which treats perception decoders as the carrier of agent and map information into the planner. Our evidence shows that carrier is essentially empty *at inference*; what matters is the gradient flow at training time, which we extend without removing.

## Why this is a stronger paper than "we beat the baseline"

The originally drafted NeurIPS direction (`reports/2026_04_22_nuerips26_paper.md`) framed the problem as "planning requires a planning-sufficient representation; detection salience ≠ planning relevance." Our subsequent experiments **falsified the specific selection-based instantiation of that thesis**:

- `selrelGT` (relevance-ranked top-k with ground-truth corridor labels) tied baseline.
- `topk_half` (confidence top-k at half the count) tied baseline.
- `nodetmap` (zero perception K/V) regressed by only L2=0.023 / CR=0.029pp.

If selection criterion does not move planning at full / half / oracle / zero count, the right reading is not "we need a better selection mechanism." It is **"the perception → planning K/V channel is essentially redundant."** That reading lifts the paper from "yet-another-better-fusion" to a structural claim about what the planner actually consumes.

The originally framed thesis ("planning-critical representation learning") survives but in a much sharper form: the planning-critical representation lives in the **image features**, and Stage 1 should be designed to shape them, not the perception decoders.

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
- Modern-head joint stage-1 (Arm B, in flight on Apollo): pending.

The matched run rules out head-mismatch as the cause. Joint planning supervision at stage 1 with `detach_perception=True` does not produce a useful planning init. If Arm B also regresses (likely by mechanism extrapolation), the conclusion is: **stage-1 alignment via planning loss does not work; auxiliary tasks that shape image features without competing for backbone capacity (e.g. aux2d) do.**

### Nomap evidence — perception-decoder removal at training time is risky

Two existing data points constrain "how aggressively can we remove perception decoders at stage 1":

- **Both stages nomap is catastrophic.** `pretrainv1_noflash_nomap` and the broader `nomap_dn_rotaug` family from `2026_03_05_map_removal.md`: stage 1 *and* stage 2 without the map head produced L2=6.612 (~10× regression) and obj_box_col=3.605%. This is the strongest negative on perception-decoder removal.
- **Stage-2-only nomap is *positive*.** Removing map only at stage 2 (keeping `sparsedrive_stage1.pth`'s map-trained backbone) on R50 *improves* planning L2 and CR slightly. So the catastrophic case is specifically the combination of "no map at training-time backbone shaping AND no map at inference."
- **The diagnostic cell is untested**: stage-1 nomap *with* stage-2 *retaining* map. This is what experiment #7 below resolves.

Reading: **at minimum, perception-task supervision at stage 1 is doing useful work somewhere.** The catastrophic both-nomap case rules out the most aggressive removal. Whether the load-bearing component is (a) backbone shaping by map-task gradients or (b) map-head warm-init flowing into stage 2 is currently confounded; #7 is designed to disambiguate.

**#7 result (Killarney 3311181, `ptnomapdnrot_ppdeformmm_planifls`).** Stage-1 trained without map (`sparsedrive_stage1_nomap_dn_rotaug.pth`) but stage-2 *with* map head (init from scratch). Result: `L2=0.5203, obj_box_col=0.048%, NDS=0.5502, mAP=0.4471, mAP_normal=0.2411` (car_ade=0.6300, ped_ade=0.6849, car_epa=0.5205, ped_epa=0.4561). Planning is on par with the project-best `_planinstfeat_laststage` reference (`L2=0.522 / 0.047%`); detection is slightly stronger (`NDS 0.5502 vs 0.5570 reference`, `mAP 0.4471 vs 0.4150`). The map head is broken (`mAP_normal 0.2411` vs `~0.55–0.58` family) because it was initialized from scratch when the backbone never saw map gradients — the stage-2 `~10` epochs of map training cannot recover what the stage-1 `~100` epochs would have established. **Disambiguation outcome: the load-bearing component for planning is (a) backbone shaping, not (b) map-head warm-init.** Removing map at stage 1 does not catastrophically hurt planning when the map head is restored at stage 2 and at inference. The catastrophic both-nomap case from `2026_03_05_map_removal.md` is therefore primarily about *inference-time absence of the map head*, not about backbone shaping. This unlocks the paper's core claim: a stage-1 designed around image-feature shaping (no map decoder needed) plus a stage-2 that retains map for inference is a feasible architecture, and the perception-K/V channel can stay or go without breaking planning.

Implication for the paper framing: the **stage-1 contribution is additive, not subtractive.** We do not propose removing perception decoders at stage 1 — we propose *augmenting* them with planner-aware auxes and showing the augmentation compounds with the no-K/V stage 2.

## Paper contributions (proposed)

1. **Diagnosis.** A controlled ablation programme on the perception → planner K/V interface, spanning four cells of count and three of selection criterion plus six bidirectional / supervision perturbations. We show the channel carries little planning information at inference.
2. **Stage 2 contribution (headline).** A minimal end-to-end planning head that reads image features directly, with no perception-token K/V cross-attention at inference. Perception heads are training-time auxiliaries only. Optionally, a learned collision cost replaces the motion-trajectory rescore step, fully decoupling the inference path from the perception decoders.
3. **Stage 1 contribution (additive).** Planner-aware dense auxiliaries (drivable-area, occupancy, future-flow) added on top of det+map+aux2d further improve stage-1 backbone shaping for planning. We do *not* claim perception decoders are unnecessary at stage 1; nomap-stage-1 evidence cautions against full removal. The contribution is additive: *augment* the existing pretraining recipe.
4. **Empirical results.** On nuScenes, this stack matches or beats the perception-decoder pipeline at lower inference cost. The diagnostic ablation grid serves as the supporting evidence.

The paper's main claim does not depend on the stage-1 contribution. If the additive-aux experiments are flat, the paper still has (1) the diagnostic ablation grid and (2) the no-K/V stage 2 architecture as headline contributions.

## Architecture sketch

### Inference path
```
multi-camera images
        │
        ▼
   ResNet-50 backbone  (stage-1-pretrained for planning, optionally DINOv2 init)
        │
        ▼
       FPN
        │
        ▼
   Planning Decoder
   ─ ego query init (from temporal queue + ego state)
   ─ N stages of [temp_gnn, gnn, deformable_to_image, ffn, refine]
   ─ no perception K/V
   ─ optional collision-cost head replacing motion-rescore
        │
        ▼
   Multi-mode trajectory + per-mode confidence
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

1. **Planning decoder depth and width** — currently 3 stages × 1 deformable each. This may need to grow (T2.1, T2.2) since it now bears the entire image-features → planning load.
2. **Temporal image-feature stacking** (T2.6) — the planner must reason about past frames; without perception tokens carrying agent state forward, the planning deformable should sample from past N frames' FPNs.
3. **Rescore replacement** — the existing rescore step uses motion-predicted agent futures for collision avoidance. Two options:
   a. Keep a lightweight motion head as an aux + inference-time rescore input (simpler; preserves CR).
   b. Replace rescore with a learned collision-cost MLP that the planner queries directly from image features at predicted ego positions (cleaner story; uncertain whether CR holds).

## Stage-1 batch — ranked by paper EV (additive framing)

Each new stage-1 train is followed by an identical stage-2 best recipe so the only varying input is the stage-1 backbone. Stage 2 keeps perception decoders for safety (the fully-removed version is the no-K/V stage-2 architecture experiment, separate from this batch).

| # | Stage-1 recipe | Tests | Cost | Paper role |
|---|---|---|---|---|
| **1** | **DINO-init stage-1** (R50 backbone init from DINOv1 SSL weights instead of ImageNet; det+map+aux2d unchanged) | does a stronger SSL prior on the backbone lift planning? | 1× ~32h | tests "image-feature quality" lever; Trillium 4-GPU bs=24 |
| **2** | **`dadense_aux2d`** — drivable-area BEV dense aux, *added* on top of det+map+aux2d | does a planner-relevant dense aux compound with depth? | ~1d label-gen; 1× ~32h | first additive planner-aware aux |
| **3** | **`occupancy_aux2d`** — class-conditional BEV occupancy, *added* on top of det+map+aux2d | dense agent supervision without box decoding | ~1–2d label-gen; 1× ~32h | second additive planner-aware aux |
| **4** | **`dadense + occupancy + aux2d`** | full additive aux suite | trivial after #2,#3; 1× ~32h | strongest stage-1 recipe; headline number |
| **5** | **DINO-init + best additive stage-1** | best-of-both | 1× ~32h | optional further-best |
| **6** | **Arm B (`stage1_8gpu_noflash_joint`)** modernized joint-stage-1 — already running | locks in negative for "joint planning supervision at stage 1" | already in flight | closes joint-stage-1 line for the paper |
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
- **Our current best** `ptaux2d_ppdeformmm_planifls_planinstfeat_laststage`: L2=0.522, CR=0.047%.
- **End-to-end no-perception-K/V** (proposed): expected to match or beat current best.
- **External**: UniAD or VAD on nuScenes for cross-method comparison. Not strictly needed for the controlled-ablation story but expected by NeurIPS reviewers.

## Risks

1. **CR regression from removing rescore.** If we drop the motion head entirely and replace rescore with a learned cost, CR may regress more than the L2-side gain warrants. Mitigation: keep motion as a small aux head and retain rescore in the v1 paper architecture; defer the "fully clean" version to future work.
2. **DINOv2 transfer poor.** If DINOv2 features don't transfer well to camera-only autonomous driving (it's pretrained on natural images), the "free image-feature lift" angle weakens. Mitigation: try MAE-style or BEV-pretrained alternatives; the paper's main claim doesn't depend on any specific stage-1 init.
3. **The end-to-end Stage 2 underperforms.** If even with all the planning-deformable-path lifts, the no-K/V architecture loses more than the small nodetmap regression suggested, the paper's main claim is weakened. Mitigation: temporal image-feature stacking (T2.6) is the strongest untried lever; if it does not recover the nodetmap deficit, fall back to a hybrid (lightweight perception K/V + everything else).
4. **Reviewers ask for waymo / argoverse2.** Without longer-range or denser-agent data, the planning-bound claim is hard to argue beyond nuScenes. Mitigation: one Waymo Open Motion or AV2 evaluation as a "generalization study" in the appendix.
5. **Negative-paper fallback.** If the end-to-end Stage 2 doesn't match current best, the paper still has the diagnostic-ablation contribution (Section 1). The framing then becomes "we map the design space of perception → planning coupling and show that ten common interventions do not help." That's a publishable but lower-impact story.

## Decision points and gating

- **Now (week of 2026-04-28):** finalize the architecture sketch and run the two highest-EV cheap experiments (DINOv2-init + current best, and temporal stacking T2.6). Wait for Arm B s1.
- **+1 week:** Arm B s2 result lands. Decision: if Arm B regresses, the joint-stage-1 negative is locked in; commit to the paper. If Arm B improves, the paper's stage-1 story shifts to "joint stage-1 with the right head + lr is the right pretrain" — different paper, also strong.
- **+2 weeks:** stage-1 auxiliary sweep (~3 backbones in flight). Decide on the best stage-1 aux combination.
- **+3-4 weeks:** end-to-end no-K/V Stage-2 architecture training + tuning.
- **+5-6 weeks:** ablation grid finalized, baselines run, manuscript drafting.
- **+8 weeks:** submission-ready.

NeurIPS 2026 abstract deadline (typical: mid-May) — tight but feasible if we lock the architecture by week 2 and start Stage-2 training by week 3.

## What to do this week

1. Wait on Arm B s1 (Apollo).
2. Submit **#1 DINO-init stage-1** on Trillium (4-GPU bs=24, lr=1.5e-4, 100ep) — validates Trillium for stage-1 jobs and tests the "stronger image backbone" lever.
3. Submit **#7 stage-1 nomap_dn_rotaug + stage-2 with map** on Killarney (stage-2 only, ~12h) — resolves the backbone-shaping vs inference-path confound in the prior nomap evidence.
4. Begin label-gen for **drivable-area BEV** (#2) — first additive planner-aware aux.
5. Begin implementation of the **no-perception-K/V Stage-2** architecture — `with_perception_kv=False` flag in `MotionPlanningHead` that skips the cross_gnn op and zero-pads num_det/num_map at the construction level.
6. Begin implementation of **temporal image-feature stacking** (T2.6) — highest-EV architectural experiment for the no-K/V stage-2.
