# Single-Flag Ablations on the Locked Paper Architecture (Jul 2026)

Rolling report for controlled, one-variable-at-a-time ablations against the
locked stage-2 anchors. Each entry isolates exactly one component of the paper
architecture, holding stage-1 init, batch size, learning rate, scorer, and
cluster fixed. Multi-component explorations belong in their own report; this
file is only for rows a reviewer could read as "X vs not-X."

## Standing anchors

All comparisons are against these r50 numbers unless stated otherwise. Both are
Killarney L40S.

| Anchor | L2 | CR | NDS | mAP_normal | Jobs |
| --- | ---: | ---: | ---: | ---: | --- |
| K/V-off headline (3-seed) | 0.3692 | 0.0400% | 0.5212 | 0.5533 | K3376689 / K3386471 / K3394849 |
| minS2 (2-seed) | 0.3644 | 0.046% | 0.5279 | 0.3363 | K3397346 / K3406306 |

Noise floors used throughout: **L2 0.007**, **CR 0.015 pp**, **NDS 0.0026**,
**mAP_normal 0.0049** (same-cluster, from `reports/2026_05_03_h100_reproduce.md`).

**Cluster discipline.** Every ablation in this report runs on Killarney L40S.
L2 is portable across clusters but CR is *not* portable on the `_egostatus`
family — Fir/Rorqual reads span 0.048–0.119% on the identical config, and
`reports/2026_07_25_r101_backbone_revisit.md` found an L2 A100/H100 gap of
0.0146 (2× the noise floor) on an identical minS2 config. A cross-cluster
ablation row is uninterpretable on CR and marginal on L2. Killarney's H100
partition is unusable for this codebase (`sm_90` kernel mismatch against
`foresight_cuda118.sif`), so L40S is both the canonical and the only option
there.

---

## Ablation 1 — Perception K/V on the inference path

### Question

Does re-enabling the perception → planner K/V channel, with **everything else
in the paper stack held fixed**, recover any planning quality?

### Why this row did not already exist

Three prior comparisons touch K/V, and none is a clean single-flag test at the
current operating point:

1. **`ppdeformmm_planifls` control** (`reports/2026_04_25_plan_relevance.md`) —
   K/V-on 0.4988 / 0.063% vs `nodetmap` 0.5215 / 0.092%. K/V-off *regressed*
   (+0.023 L2). Pre-`egostatus`, pre-`decoder6`, pre-learned-scorer.
2. **`_laststage` control** (Batch F, `reports/2026_04_26_plan_gaps.md`) —
   K/V-on 0.5302 / 0.054% vs `laststage_nodetmap` 0.5153 / 0.086%. Here K/V-off
   *improved* L2 by 0.015. Channel disaggregation from the same batch is the
   durable finding: `nodetkv` (drop det only) 0.5466 / 0.125% regresses,
   `nomapkv` (drop map only) 0.5246 / 0.047% ties. **Map K/V is dispensable;
   det K/V is not**, and `nodetmap`'s L2 win is a joint-removal artifact.
3. **Top-end stack-vs-stack** — K/V-on `_ptaux2d_ppdeformmm_planifls_egostatus`
   (K3368763) 0.3701 / 0.058% vs the K/V-off headline 0.3708 / 0.0405%
   (2-seed at the time). L2 ties (Δ=0.0007), CR favours K/V-off by 0.018 pp.

Note the sign flip between (1) and (2): the isolated K/V flag is worth roughly
±0.02 L2 depending on the control, i.e. small and control-dependent, not
cleanly positive.

Comparison (3) is the one the paper leans on, and it carries **five** unmatched
deltas, not one:

| | K/V-on comparator (K3368763) | K/V-off headline |
| --- | --- | --- |
| perception K/V | on (`num_map=10`) | off |
| scorer | hard rescore (`use_rescore=True`) | learned (`evalmatchmode`) |
| decoder stages | 3 | 6 |
| deformable sampling | endpoint (`ppdeformmm`) | all waypoints (`planwp`) |
| stage-1 init | `sparsedrive_stage1_aux2d.pth` | `sparsedrive_stage1.pth` |

That supports "our stack ties the best K/V-on stack" and nothing narrower. In
particular it cannot support "flipping K/V costs nothing at this operating
point," which is how Contribution (i) currently reads.

### Method

New configs, copied from their parents with only the K/V delta applied:

- `..._decoder6_planwp_evalmatchmode_egostatus_kvon` — headline arm
- `..._decoder6_planwp_evalmatchmode_egostatus_minS2_kvon` — minS2 arm

The delta is two lines in `motion_plan_head`:

```python
num_map=0,                 →  num_map=10,
skip_perception_kv=True,   →  skip_perception_kv=False,
```

Both are needed. `skip_perception_kv` nulls the built layers post-hoc
(`motion_planning_head.py:593-602`, `self.layers[i] = None`), so flipping it
restores real attention with no other code path involved; but with `num_map=0`
the restored `cross_gnn` would receive a zero-width map tensor
(`motion_planning_head.py:1461-1467`), leaving map K/V dead. The planner op
list already contains `gnn` / `cross_gnn` and both `graph_model` /
`cross_graph_model` are already configured in the parents, so nothing else
moves.

Held fixed against the anchors: learned scorer (`use_rescore=False` +
`use_rescore_learned_hard=True` + `conflict_label_source='evalmatch_mode'`),
`egostatus`, `decoder6`, `planwp`, `planinstfeat_laststage`, `num_det=50`,
`load_from='ckpt/sparsedrive_stage1.pth'`, bs24 / lr 1.5e-4, 10 epochs.

**Required code change (`motion_planning_head.py:1647-1663`).** `minS2` sets
`ego_only_planning=True`, which sliced the det K/V pool down to its last entry
(`instance_feature_selected[:, -1:]`) — the pool is `cat([top-k dets, ego])` at
`:1581`, so the slice keeps ego alone. That slice is an optimization valid only
when the `gnn` op is nulled; left unguarded it would have made the minS2 K/V-on
arm silently degenerate into ego attending to itself, producing a null result
for the wrong reason. It is now gated on the det-skip condition:

```python
if self.skip_perception_kv in (True, "both", "det"):
    instance_feature_selected = instance_feature_selected[:, -1:]
    anchor_embed_selected = anchor_embed_selected[:, -1:]
```

All 32 existing `ego_only_planning=True` configs set `skip_perception_kv` to
`True` or `'both'`, so this is a no-op for every config on record and cannot
perturb the standing anchors.

### Pre-registered reading

The K/V-on arms carry **+7.88M parameters** (92.45M → 100.33M headline;
92.69M → 100.58M minS2 — 6 stages × `graph_model` at `embed_dims*2=512` under
`decouple_attn_motion` plus `cross_graph_model` at 256). Therefore:

- **K/V-on ties or loses** → clean support for Contribution (i): more
  parameters *and* more information, no gain. The interface is empty.
- **K/V-on wins by more than 0.007 L2** → ambiguous between interface
  information and added capacity, and the paper claim must weaken. The
  existing `decoder6` result (3→6 stages ties L2, `_laststage_nodetmap` 0.5194
  vs 0.5153) is the evidence that this family is not capacity-starved, which
  mostly but not entirely defuses it.

Registering this before the runs land so the interpretation is not chosen after
seeing the numbers.

### Runs

| Server | Config | Seed | Job ID | Status |
| --- | --- | --- | --- | --- |
| Killarney L40S | `..._egostatus_kvon` | 0 | 4394494 | PENDING (submitted 2026-07-25) |
| Killarney L40S | `..._egostatus_kvon` | 1 | 4394495 | PENDING (submitted 2026-07-25) |
| Killarney L40S | `..._egostatus_minS2_kvon` | 0 | 4394496 | PENDING (submitted 2026-07-25) |

Seeds follow the established convention: same config, `--seed 1` for the
reproduction (precedent K3412422, K3424762). Seed 1 additionally gets
`--work-dir work_dirs/..._kvon_seed1` so the two seeds cannot collide on
`latest.pth` or the log files if they run concurrently — prior seed pairs were
run at different times and shared a work_dir. Each job also gets a distinct
`PORT` (28651 / 28652 / 28653), since `tools/dist_train.sh` defaults to 28651
and Killarney can co-schedule 4-GPU jobs on one node.

### Pre-submission verification

- All four configs (2 parents + 2 `_kvon`) build via `build_detector`.
- K/V layer liveness confirmed by inspecting `head.motion_plan_head.layers`:
  parents `gnn 0/6, cross_gnn 0/6`; `_kvon` arms `gnn 6/6, cross_gnn 6/6`.
- `num_det=50, num_map=10, skip_perception_kv=False` in both `_kvon` arms;
  `ego_only_planning` True in minS2 arm, False in headline arm.
- **Runtime probe of the K/V pool** (monkeypatched `MotionPlanningHead.graph_model`,
  1-GPU local run). Op index 1 is the `gnn` op; index 0 and 10 are `temp_gnn`
  from blocks 1 and 2:

  ```
  minS2_kvon     stage=1  query=(1,  1,256)  key=(1,51,256)  ego_only=True
  headline_kvon  stage=1  query=(1,901,256)  key=(1,51,256)  ego_only=False
  ```

  51 = 50 det tokens + ego in both arms, confirming the gating change works and
  that the minS2 arm is not silently degenerate.
- 1-GPU smoke trains on the local 3090, both arms, no shape errors. `minS2_kvon`:
  405 iterations, loss 7.9 → 6.8; det/map losses read 0.0000 as expected under
  minS2's zeroed weights, planning losses live across all 6 stages.
  `headline_kvon`: 19+ iterations, det/map/planning losses all live.

### Results — LANDED 2026-07-26

| Model | L2 | CR | NDS | mAP_normal | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| K/V-off headline (3-seed anchor) | 0.3692 | 0.0400% | 0.5212 | 0.5533 | reference |
| K/V-on headline seed 0 | 0.3717 | 0.063% | 0.5235 | 0.5567 | 4394494 |
| K/V-on headline seed 1 | 0.3626 | 0.045% | 0.5189 | 0.5484 | 4394495 |
| **K/V-on headline (2-seed mean)** | **0.3672** | **0.054%** | 0.5212 | 0.5526 | **ΔL2 = −0.0020** |
| minS2 (2-seed anchor) | 0.3644 | 0.046% | 0.5279 | 0.3363 | reference |
| **K/V-on minS2 seed 0** | **0.3608** | **0.049%** | 0.5279 | 0.3346 | **ΔL2 = −0.0036** |

### Discussion

**Both arms tie their anchors.** ΔL2 = −0.0020 (headline, 2-seed) and −0.0036
(minS2) against a 0.007 noise floor; ΔCR = +0.014 pp and +0.003 pp against a
0.015 pp floor. Every delta is inside noise, and the two L2 deltas point in the
*opposite* direction to the hypothesis that K/V carries information.

This matches the **pre-registered** "ties or loses" branch, recorded before the
runs landed: the K/V-on arms carry +7.88M parameters *and* the full det+map
token interface, and gain nothing. More capacity plus more information, no
effect.

This is the controlled row Contribution (i) never had. The three prior K/V
comparisons were either pre-`egostatus` (0.4988-band controls, where the flag
was worth ±0.02 depending on which control you picked) or stack-vs-stack with
five unmatched deltas. Here the only variable is the interface, with the learned
scorer, `egostatus`, `decoder6`, `planwp`, laststage instfeat, stage-1 init and
bs/lr all held fixed, on the same cluster as the anchors.

Perception metrics are unchanged in the headline arm (NDS 0.5212 vs 0.5212,
mAP_normal 0.5526 vs 0.5533), confirming the K/V restoration did not perturb the
perception heads — the planning result is not a side effect of a different
detector.

---

## Ablation 2 — Trainable backbone under minS2 (`lr_mult` 0.0 → 0.1)

### Question

Can the **planning loss alone** usefully adapt the backbone? Every measurement
on record conflates "backbone trains" with "perception losses supervise it,"
because the two have only ever been switched together.

### Status: never run

All 27 minS2-family configs set `img_backbone` `lr_mult=0.0`. The only ego-only
config above zero is `_streamc` (`lr_mult=0.1`), and it is *not* minS2 — its
loss weights are byte-identical to the headline, so its backbone is trained by
the perception losses in the usual way.

| | `_streamc` | minS2 | **Ablation 2** |
| --- | --- | --- | --- |
| `ego_only_planning` | True | True | True |
| backbone `lr_mult` | 0.1 | 0.0 | **0.1** |
| perception losses | live | all zeroed | all zeroed |
| result | 0.3673 / 0.037% (4 reads) | 0.3644 / 0.046% (2 seeds) | — |

Nothing has ever trained the backbone with planning as the only live loss.

### Why it was set to 0

Stream E, on a single run: `_egostatus_frozenpercep` (K3394848) landed at
**L2=0.3698 / CR=0.044%** vs the headline's 0.3692 / 0.0400%. Freezing the whole
stage-2 perception stack cost 0.0006 L2 — a hundredth of the noise floor — so
folding it into minS2 was free and maximized the "stage 2 is just a planner
fine-tune" claim. It was never revisited.

### Why revisit now

`reports/2026_07_25_r101_backbone_revisit.md` found r101 wins at `lr_mult=0.1`
and loses at `0.0`, and attributed it to "r101's features need stage-2
adaptation to become planning-useful." But under the headline, stage-2
adaptation *means perception-loss adaptation*. If planning-loss-only adaptation
also recovers r101's win, the story is backbone plasticity; if not, it is the
perception losses specifically. The Stream E claim cannot be stated precisely
until this is separated.

### Arms

Two, to separate backbone plasticity from the rest of the frozen stack:

- **2a** — backbone only: `img_backbone` `lr_mult=0.1`, neck/det/map stay 0.0.
- **2b** — whole stack: all four keys back to headline values (backbone 0.1,
  others default 1.0). The direct Stream E inverse under zeroed losses.

Note the grad-clip coupling (see Ablation 1's method): `grad_clip` is global and
mmcv's `clip_grads` filters on `requires_grad`, not on `lr_mult`, so unfreezing
changes which gradients count toward the norm even before any weight moves.

### Results — LANDED 2026-07-26

| Model | L2 | CR | NDS | mAP_normal | Job |
| --- | ---: | ---: | ---: | ---: | --- |
| minS2 (2-seed anchor) | 0.3644 | 0.046% | **0.5279** | **0.3363** | reference |
| **2a — backbone `lr_mult=0.1`** | **0.3627** | 0.037% | **0.4353** | 0.2391 | 4394722 |
| **2b — whole stack unfrozen** | **0.3661** | 0.060% | **0.0034** | 0.1273 | 4394723 |

### Discussion

**Planning is completely unmoved; perception is destroyed.** ΔL2 = −0.0017 (2a)
and +0.0017 (2b) — both an order of magnitude inside the noise floor. Meanwhile
NDS falls 0.5279 → 0.4353 → **0.0034**: with the perception losses zeroed, any
trainable perception stack drifts, and by 2b the detector is entirely gone.

**Framing note.** An earlier version of this section justified the freeze by
perception reportability. That is wrong for this paper: the contributions are
planning claims, and NDS / mAP_normal are diagnostics we report, not objectives
we optimise. The NDS collapse is therefore *not* a reason to prefer the freeze.
The argument has to be made on planning terms.

**On planning terms, 2a and 2b tie minS2 — they do not beat it.** 2a's 0.3627
sits *inside* minS2's own seed range (0.3563 / 0.3660 / 0.3724), and one minS2
seed is better than it. With planning a tie, the tiebreakers are simplicity and
compute, and those favour the freeze: ~23M fewer trainable parameters and "stage
2 is a planner fine-tune on a frozen backbone" is itself Contribution 5. Stream E
should read *"the freeze costs nothing for planning and buys a materially
simpler stage 2"* — not *"stage-2 backbone training is empty for planning"* and
not the perception-reportability argument.

**The substantive planning finding** is that the backbone can drift a long way —
far enough to take NDS from 0.5279 to 0.0034 — at **zero planning cost**. Read
together with Ablation 5 (removing the image readout costs 0.135), the picture is:

> The planner genuinely needs image features, but it does **not** need them to be
> perception-aligned.

That is a stronger and more useful claim than "backbone training is empty," and
it is squarely a planning result rather than a perception one.

**It also means `lr_mult=0.1` is unmotivated here.** That value was tuned for the
regime where perception losses hold the backbone in place. With those losses
zeroed, the backbone is free, and nothing establishes 0.1 as the right amount of
freedom — we have only shown it is harmless. Ablation 2c sweeps it.

### Ablation 2c — `lr_mult` sweep under minS2

Only `0.0` and `0.1` have ever been used across every r50 stage-2 config in the
repo. With the perception losses zeroed the backbone is unconstrained, so the
question is not "is training it harmless" (2a answered yes) but **"is there an
amount of freedom that actually helps planning."**

| Arm | backbone `lr_mult` | effective LR | Config |
| --- | ---: | ---: | --- |
| frozen (anchor) | 0.0 | 0 | `..._minS2` (3 seeds) |
| 2c-a | 0.05 | 7.5e-06 | `..._minS2_bblr0p05` |
| 2a / 2c-b | 0.1 | 1.5e-05 | `..._minS2_bblr0p1` (seed 1 added) |
| 2c-c | 0.2 | 3.0e-05 | `..._minS2_bblr0p2` |
| 2c-d | 0.5 | 7.5e-05 | `..._minS2_bblr0p5` |

Everything else identical: neck / det / map stay frozen at 0.0, planner at
1.5e-4, all perception losses zeroed. Verified by building the optimizer and
reading per-module LRs at every point.

A second seed at 0.1 is included because the existing 2a point is a single draw
sitting inside minS2's seed range — without it the sweep has no calibration for
how much of any trend is seed noise.

**Reading, registered in advance.** The frozen anchor is 0.3649 (3 seeds,
range 0.3563–0.3724). Given that spread, a sweep point only counts as a real
improvement if it lands **below ~0.356** — i.e. beats the best observed frozen
seed — and reproduces. Anything in 0.356–0.372 is inside the existing
distribution and should be reported as a tie regardless of where it falls in the
ordering. Expect NDS to degrade monotonically with `lr_mult`; that is a
diagnostic here, not a criterion.

**Bearing on the r101 asymmetry.** `2026_07_25_r101_backbone_revisit.md`
attributed r101's `lr_mult` sensitivity to "r101's features needing stage-2
adaptation to become planning-useful." On r50, planning-loss-only adaptation
does nothing (2a). That points the r101 explanation toward the *perception
losses* rather than backbone plasticity — but it is r50 evidence about an r101
claim, so the r101 `lr_mult` sweep in Future Work is still needed to close it.

---

## Ablation 3 — minS2 without ego status

### Question

What does the **locked paper architecture** score with no ego-status input?

### Status: never run on minS2

The headline arm of this A/B exists and is clean —
`..._decoder6_planwp_evalmatchmode.py` is the headline minus `egostatus`, same
stage-1 init, nothing else differing:

| Config | L2 | CR | Job |
| --- | ---: | ---: | --- |
| `..._decoder6_planwp_evalmatchmode` seed 0 | 0.5204 | 0.046% | K3366620 |
| `..._decoder6_planwp_evalmatchmode` seed 1 | 0.5087 | 0.053% | K3377724 |
| **mean (no ego)** | **0.5145** | **0.0495%** | |
| headline, 3-seed (ego) | 0.3692 | 0.0400% | |

**ΔL2 = 0.145** from that one input.

There is no `..._evalmatchmode_minS2.py`. Every minS2 config carries
`plan_ego_status_encode_enable=True` except two, and both bundle an extra
variant:

| Proxy | L2 | CR | Job | Contamination |
| --- | ---: | ---: | --- | --- |
| `_minS2_planmodeSA` (no ego) | 0.5238 | 0.056% | K3440406 | planmodeSA is a **known regression** — with ego, 0.3848 vs minS2's 0.3563 |
| `_minS2_B1p7_perstage` (no ego) | 0.5087 | 0.084% | K3444448 | B1.7 per-stage conflict sampler |
| `_ptdnrot3daux2p5d_..._minS2_B1p6` (no ego) | 0.5118 | 0.088% | K3437654 | different stage-1 **and** B1.6 |
| `planneronly_skeleton_noegostatus` | 0.5206 | 0.088% | K3445429 | different head implementation |

So minS2's no-ego number is bracketed at ~0.509–0.524 by four contaminated
reads and not measured. The planmodeSA proxy is the worst to lean on, since that
lever is known to cost ~0.03 L2 in the ego regime.

### Why it matters

minS2 is the locked paper architecture; the headline is only a comparator. Any
no-ego mirror table answering the AD-MLP / BEV-Planner critique of nuScenes
open-loop planning needs the locked architecture in it. Today we can say "K/V-off
holds parity without ego" (0.5145 vs the 0.5197 no-ego baseline) but cannot say
the same for minS2 without citing a config carrying a known-regressive lever.

### Arm

One line, matching how `_minS2_planmodeSA:412` expresses it:
`plan_ego_status_encode_enable=False`. Two seeds.

### Results — LANDED 2026-07-26

| Model | L2 | CR | NDS | Job |
| --- | ---: | ---: | ---: | --- |
| minS2 (2-seed anchor, ego) | 0.3644 | 0.046% | 0.5279 | reference |
| minS2 no-ego seed 0 | 0.5108 | 0.057% | 0.5287 | 4394720 |
| minS2 no-ego seed 1 | 0.5359 | 0.072% | 0.5291 | 4394721 |
| **minS2 no-ego (2-seed mean)** | **0.5233** | **0.0645%** | 0.5289 | **ΔL2 = +0.159** |

Also landed: **minS2 seed 2** (4394729) at **L2 = 0.3660 / CR = 0.047% /
NDS = 0.5258**, giving a 3-seed minS2 anchor of **0.3649 / 0.0463%** — the
2-seed value (0.3644) holds.

### Discussion

Ego status is worth **0.159 L2** on the locked architecture, closely matching the
0.145 measured on the headline (0.5145 → 0.3692). The dependence is a property
of the planner, not of a particular stage-2 configuration.

The minS2 no-ego number is now measured rather than bracketed by contaminated
proxies. It lands at 0.5233, above the previous 0.509–0.524 bracket's midpoint
and well above the `_minS2_B1p7_perstage` proxy (0.5087) that would have been
the natural stand-in.

**Seed spread is a caveat.** The two no-ego seeds differ by 0.025 L2 — about
3.6× the noise floor derived from the ego-regime runs, and far wider than the
ego-regime seed spread (0.3563 / 0.3724 / 0.3660). No-ego training appears
genuinely noisier, so **the 0.007 L2 noise floor should not be applied to no-ego
comparisons**; anything drawn from single no-ego seeds needs a wider band. This
affects how the 2×2 in Ablation 5 is read.

---

## Ablation 4 — Remove the learned collision rescore

### Question

Does the learned scorer still earn its place **now that `egostatus` is in**?

Its original justification was pre-`egostatus`: on K/V-off it drove CR from
0.068% (hard rescore, `_decoder6_planwp` 2-seed mean 0.0775%) down to 0.0495%,
recovering the CR signal hard rescore loses when det K/V is removed. But the
headline's CR is now **0.0400%** with `egostatus` folded in — at or below what
the scorer was introduced to achieve. It may now be redundant.

### A structural constraint worth recording

**minS2 cannot use hard rescore at all.** `decoder.py:295` `rescore()` takes
`motion_cls` / `motion_reg` — agent future trajectories — and under
`ego_only_planning=True` those are zero-width, so hard rescore degenerates to a
no-op. For minS2 the only choices are the learned scorer or no rescore. The
learned scorer is therefore structurally load-bearing for the locked
architecture in a way it is not for the headline.

### Arms

The conflict head does two things — an inference-time veto and a training-time
aux loss (`conflict_loss_weight=0.10`) — so they must be separated:

| Arm | Change | Isolates | Cost |
| --- | --- | --- | --- |
| **4a** | `use_rescore_learned_hard=False`, keep `with_conflict_head=True` | the inference-time veto only | **eval-only** on the existing ckpt |
| **4b** | `with_conflict_head=False` everywhere | veto + aux loss together | retrain |
| **4c** | `use_rescore=True` instead (hard rescore) | learned vs heuristic at the current operating point | **eval-only**, headline only |

4a and 4c are pure inference-path changes and run on existing checkpoints
(~1h each, `--time=2:59:00`), so this ablation is nearly free to start.
Precedent for eval-only scorer sweeps: Exp 5 in
`reports/2026_04_26_plan_scoring.md`. Only 4b needs a training slot.

Run 4a on both headline and minS2; 4c on the headline only, per the constraint
above. Prior no-rescore reference (pre-`egostatus`, K/V-off):
`_laststage_nodetmap_norescore` 0.5131 / 0.089%.

### Results — headline arm (LANDED 2026-07-26)

Paired eval, Fir H100, all three arms on the identical checkpoint
(`..._egostatus/iter_11720.pth`).

| Arm | Job | L2 | CR | ΔL2 | ΔCR |
| --- | --- | ---: | ---: | ---: | ---: |
| reference — learned veto ON | 51173713 | 0.3772 | 0.048% | — | — |
| **4a — no rescore at all** | 51173714 | **0.3758** | **0.053%** | −0.0014 | +0.005 pp |
| **4c — hard rescore instead** | 51173715 | **0.3812** | **0.059%** | +0.0040 | +0.011 pp |

Noise floors: L2 0.007, CR 0.015 pp. **Every delta is inside both.**

Validity check: the reference arm reproduces the previously recorded read on
this checkpoint (Fir 38378856: 0.3776 / 0.048%) to ΔL2 = 0.0004, confirming the
eval is deterministic and the paired design is sound.

**Reading: the inference-time collision veto contributes nothing measurable at
the current operating point.** Removing it entirely (4a) is within noise on both
metrics; substituting the heuristic (4c) is also within noise, marginally worse
on both. The learned scorer's original justification was CR recovery on K/V-off
— 0.068% (hard) → 0.046% (learned), pre-`egostatus`. With `egostatus` folded in,
the planner's raw CR is already ~0.05% before any veto is applied, so there is
no longer a gap for the scorer to close.

**Scope of the claim.** All three arms still carry the training-time conflict
loss (`conflict_loss_weight=0.10`); 4a disables only the inference veto. So the
supported statement is "the veto is empty," not "the conflict head is empty" —
the aux loss may still be shaping the planner. Single checkpoint, single seed.

**This triggers 4b**, which was held as conditional: remove
`with_conflict_head` entirely and retrain, separating the aux loss from the
veto. If 4b also ties, the conflict head can be deleted from the paper
architecture outright — a real simplification, and it would also dissolve the
minS2 hard-rescore constraint noted above, since minS2 would then need no
rescore mechanism at all.

The minS2 arm of 4a remains blocked until 4394729 produces a checkpoint.

---

## Ablation 5 — Remove the planner → image deformable readout

### Question

If the planner stops reading image features directly, and perception K/V is
already off, **what is left driving planning?**

### Why this is the most consequential row in this report

Contribution (ii) is "the planner reads directly from image features." The
mechanism is `planning_deformable` + the `deformable` ops in the planner op list
+ `planning_deformable_instfeat_laststage` + `planning_deformable_waypoints`.
Turning it off with K/V already off leaves the planner with only its ego query,
the `egostatus` encoding, and the temporal queue.

If that ties the headline, then the planner was never using scene information at
inference, and Contribution (ii) does not hold — the model would be an
ego-status predictor with a perception-shaped backbone attached. Combined with
Ablation 3's ΔL2 = 0.145 for ego status alone, this is the row a reviewer
running the AD-MLP argument will demand.

The last measurement of this lever is `planpredtrajdeformmm` at **0.636 → 0.522**
("main L2 driver", `2026_05_03_nuerips_paper_outline.md`) — but that predates
`egostatus`, which itself moved L2 by 0.13–0.15. The two have never been
measured against each other.

### Important caveat: this does not make the planner blind

The ego token is built from image features regardless —
`instance_queue.py:202-205` pools camera 0 (front), last FPN level, through
`ego_feature_encoder`. So Ablation 5 removes *multi-view deformable sampling at
trajectory waypoints*, not all visual input. A genuinely blind control would
additionally have to stub `ego_feature_encoder`. Worth adding as **5b** if 5a
ties, since only 5b distinguishes "no scene information" from "a front-camera
global embedding is sufficient."

### Design: run it as a 2×2 with Ablation 3

The two levers are the paper's whole planning signal, and their interaction is
the point. Built on **minS2**, since it is the locked architecture and three of
its four corners are already measured or in flight:

| minS2 | deformable ON | deformable OFF |
| --- | --- | --- |
| **egostatus ON** | 0.3644 (2-seed anchor) | **5a-minS2** |
| **egostatus OFF** | Ablation 3 (4394720/21, in flight) | **5c** |

The headline 2×2 has three of four corners covered as a by-product
(0.3692 anchor · 0.5145 no-ego 2-seed · 5a-headline); only
headline + no-ego + no-deformable is left open, and it is one further run if the
minS2 result makes it worth having.

`5c` is the floor of the whole architecture — neither ego status nor direct
image reading — and tells us what the temporal queue plus a front-cam embedding
are worth on their own.

### Arms

- **5a** — headline/minS2 minus the planner deformable: drop `deformable` +
  its `norm` from the op list, set `planning_deformable`,
  `planning_deformable_instfeat`, `planning_deformable_instfeat_laststage` to
  False, remove `planning_deformable_waypoints` and `motion_deformable_multimode`.
- **5b** *(conditional on 5a tying)* — 5a plus a stubbed `ego_feature_encoder`.
- **5c** — 5a with `plan_ego_status_encode_enable=False`. Completes the 2×2.

Expect DDP unused-parameter trouble on 5a: `deformable_model` would be built but
never called. Either drop the key from the config or follow the
`skip_perception_kv` pattern (`motion_planning_head.py:593-602`) and set the
layer to `None` so DDP does not see its params. *(Resolved: dropping the op from
`operation_order` means no layer is built at all; a backward probe confirmed
`UNUSED_PARAM_COUNT 0` on both arms, so no DDP handling was needed.)*

### Results — LANDED 2026-07-26

| Model | L2 | CR | NDS | Job | ΔL2 vs its anchor |
| --- | ---: | ---: | ---: | --- | ---: |
| headline anchor (3-seed) | 0.3692 | 0.040% | 0.5212 | — | — |
| **5a — headline, no deformable** | **0.4062** | 0.089% | 0.5251 | 4394757 | **+0.037** |
| minS2 anchor (3-seed) | 0.3649 | 0.046% | 0.5279 | — | — |
| **5a — minS2, no deformable** | **0.4996** | 0.123% | 0.5257 | 4394758 | **+0.135** |
| **5c — minS2, no ego + no deformable** | **0.6406** | 0.149% | 0.5274 | 4394759 | **+0.276** |

### Discussion

**Contribution (ii) survives.** This was the row ranked most likely to falsify a
headline claim, and it did not. Removing the planner's image readout costs
+0.037 L2 on the headline (5.3× the noise floor) and +0.135 on minS2 (19×). The
planner is genuinely using its direct image path; it is not an ego-status
predictor with a decorative deformable module attached.

**The completed minS2 2×2:**

| minS2 | deformable ON | deformable OFF |
| --- | ---: | ---: |
| **egostatus ON** | **0.3649** | **0.4996** |
| **egostatus OFF** | **0.5233** | **0.6406** |

Both levers are large and close to additive — ego status is worth 0.159 with the
deformable on and 0.141 with it off; the deformable is worth 0.135 with ego on
and 0.117 with ego off. The interaction term is ~0.018, small relative to either
main effect, so the two signals are largely complementary rather than redundant.

**The floor is the baseline.** Strip both and minS2 lands at 0.6406 — essentially
our trained SparseDrive baseline (0.636). With neither ego status nor direct
image reading, the temporal queue plus the front-camera ego embedding recover
nothing beyond where the whole project started. That is a useful sanity anchor:
the 0.36 headline is built from exactly these two ingredients, and removing both
returns the model to baseline.

**CR tracks L2 monotonically here** (0.046% → 0.123% → 0.149%), unlike most
levers in this codebase where CR moves independently. Losing scene access
degrades collision avoidance in proportion to trajectory quality, which is what
one would expect if the deformable readout is what supplies obstacle awareness.

Caveat: 5a and 5c are single seeds, and the Ablation 3 result shows no-ego
training is noisier than the ego regime. The 5c number in particular should be
treated as ±0.025 rather than ±0.007. The direction and magnitude are far too
large for this to matter to the conclusion.

---

## Ablation 6 — Close the ego-status loop inside the model

### Correction to the record

An earlier reading of `instance_queue.py:217` (`ego_anchor[..., VY] =
prev_ego_status[..., 6]`) was taken to mean every "no-ego" run still receives a
ground-truth ego velocity. **That is wrong.** `cache_planning` is called with
`plan_status` (`motion_planning_head.py:2357`), which is the *refine layer's
predicted* ego status (`:2263`, `:2285`), not `metas['ego_status']`.

Consequences, all of which strengthen rather than weaken the position:

- **SparseDrive is genuinely ego-status-free at inference.** GT `ego_status` is
  consumed only as an L1 training target for `plan_status_branch`
  (`plan_loss_status`, weight 1.0). The anchor write feeds back the model's own
  prediction.
- **Our no-ego runs are genuinely ego-free too.** The 0.5145 headline no-ego and
  0.5233 minS2 no-ego numbers stand without qualification.
- **The apples-to-apples comparison against SparseDrive is 0.5145 vs 0.636** —
  a 0.12 win from architecture alone, on equal ego-status footing. This should
  be stated explicitly in the paper before the text hardens around 0.3692, which
  is *not* comparable to UniAD / VAD / SparseDrive.

### Question

The model already predicts its own ego status at full loss weight. Can that
prediction replace the GT read in `plan_ego_status_encoder`, recovering part of
the 0.16 L2 ego gap without consuming ego status at inference?

### What already exists

| Mechanism | Where | Channel width |
| --- | --- | --- |
| `plan_status_branch(ego_feature + ego_anchor_embed)` → 10-D ego status | `motion_blocks.py:94,177` | trained at `plan_loss_status` weight 1.0 |
| prediction cached for next frame | `motion_planning_head.py:2357` | — |
| fed into next frame's ego anchor | `instance_queue.py:217` | **one scalar** (`[6]` → `VY`) |
| `temp_gnn` over past ego tokens | `instance_queue.py:138,142` → `:1863` | post-refinement ego feature, 4 frames |
| GT ego status → `plan_ego_status_encoder` | `:2340-2342` | 5-dim MLP, **current frame** |

So a visual ego-state estimate is already present and already fed back — but
through a one-scalar channel into an anchor slot. Ablation 6 widens it to the
same 5-dim MLP injection that `egostatus` uses, sourced from the model instead
of from CAN.

### Distinction from `egopred_lite` / `egopred_full`

Both existing predictors consume **raw `ego_status` history from `metas`**
(`:2364`), so neither is ego-free — they replace the current-frame read only.
`egopred_full` additionally bases its output on a constant-acceleration
extrapolation of that history, with the residual head zero-initialised
(`ego_state_estimator.py:67-68`), so its 0.4722 may be largely kinematic rather
than visual. Ablation 6 closes the loop entirely inside the model and would be
the first variant that touches no `metas['ego_status']` at inference.

### Method

New flag `plan_ego_status_source='predicted'` (default `'gt'`, preserving every
existing config). When set, `plan_ego_status_encoder` consumes `plan_status`
instead of `metas['ego_status']`.

Stage 0 has no current-frame prediction yet, so:

- **stage *i*** consumes stage *(i−1)*'s `plan_status` — matching the existing
  `planning_cumulative_refinement` pattern;
- **stage 0** bootstraps from the cached previous-frame `prev_ego_status`, which
  the queue already holds.

The pure previous-frame variant falls out as the stage-0 case, so only one
implementation is needed.

### Expected range and risk

Bounded by minS2 anchor **0.3649** (GT ego) and minS2 no-ego **0.5233**. The
open question is where in that 0.16 band it lands.

Tempering expectation: `plan_status` is already trained at weight 1.0 and
already feeds the anchor, so the visual estimate is *present in the model today*
— this widens the channel rather than adding information. The gain could be
0.15 or 0.02.

Real risk: **feedback instability.** `plan_status` is supervised on GT but at
inference its errors compound frame to frame. The anchor already does this for
one scalar; widening to five dimensions with a much stronger downstream effect
could drift. Watch `planning_loss_status` and the per-timestep L2 curve — drift
would show as degradation concentrated in the 4–6 s waypoints.

### Arms

- **6a** — `plan_ego_status_source='predicted'` on minS2, 2 seeds (no-ego
  regime is noisy; the Ablation 3 spread was 0.025).
- **6b** *(conditional on 6a)* — detach `plan_status` before the encoder, if 6a
  shows training instability from the gradient path through the status branch.

---

## Run log

Submitted 2026-07-25. Killarney has no per-user job cap and 100+ partially-free
L40S nodes, so these run concurrently rather than in waves.

| Ablation | Arm | Server | Job ID |
| --- | --- | --- | --- |
| 1 | K/V-on headline seed 0 | Killarney L40S | 4394494 |
| 1 | K/V-on headline seed 1 | Killarney L40S | 4394495 |
| 1 | K/V-on minS2 seed 0 | Killarney L40S | 4394496 |
| 3 | minS2 no-ego seed 0 | Killarney L40S | 4394720 |
| 3 | minS2 no-ego seed 1 | Killarney L40S | 4394721 |
| 2a | minS2 backbone `lr_mult=0.1` | Killarney L40S | 4394722 |
| 2b | minS2 fully unfrozen | Killarney L40S | 4394723 |
| — | minS2 seed 2 (anchor + ckpt for 4a) | Killarney L40S | 4394729 |
| 5a | no planner deformable, headline | Killarney L40S | 4394757 |
| 5a | no planner deformable, minS2 | Killarney L40S | 4394758 |
| 5c | no ego + no deformable, minS2 | Killarney L40S | 4394759 |
| 4 | reference eval (veto on) | Fir H100 | 51125562 |
| 4a | veto off | Fir H100 | 51125563 |
| 4c | hard rescore | Fir H100 | 51125564 |

Held back deliberately: **5b** (stub `ego_feature_encoder`) pending 5a, and
**4b** (remove the conflict head and its aux loss) pending 4a. Both are
conditional — their question only exists depending on what the upstream arm
returns.

### Checkpoint availability — corrects the "4a/4c are free" claim

Ablation 4 was planned as eval-only against the Killarney anchors. **Those
checkpoints no longer exist**: `/scratch/spapais/ForeSight/work_dirs` on
Killarney is 1.3 MB with zero `.pth` files — logs and plots were synced back and
the weights purged. Surviving r50 headline (`_egostatus`) checkpoints were found
on **Fir**, **Rorqual**, and **Tamia**; Tamia additionally has the no-ego
headline. **No plain r50 minS2 checkpoint exists on any host.**

Consequences:

- **4a / 4c on the headline run as planned**, on Fir against its existing
  checkpoint. Cluster portability is not a problem here because all three arms
  evaluate *the same weights on the same cluster* — the paired delta is the
  quantity of interest, not the absolute CR. A reference eval (51125562) is
  included so the comparison is against that exact checkpoint rather than the
  Killarney 3-seed anchor.
- **4a on minS2 is blocked** until a checkpoint exists. Job 4394729 retrains
  minS2 with `--seed 2` to produce one; it doubles as a third seed for the
  2-seed minS2 anchor, which is worth having independently.

Operational note: checkpoints are being purged from Killarney scratch between
sessions. Any ablation intended to be answered by eval-only reruns needs its
checkpoint preserved deliberately, or it silently becomes a 12h retrain.

## Did anything beat the anchors?

**No — not outside noise.** These were designed as falsification controls, not
as improvements, and every null is a confirmation. But several arms came in
numerically below their anchor, which is worth recording so nobody later mistakes
a seed draw for a result:

| Run | L2 | CR | vs anchor |
| --- | ---: | ---: | --- |
| K/V-on minS2 (4394496) | **0.3608** | 0.049% | −0.0041 vs minS2 0.3649 |
| K/V-on headline seed 1 (4394495) | 0.3626 | 0.045% | −0.0066 vs headline 0.3692 |
| 2a backbone `lr_mult=0.1` (4394722) | 0.3627 | **0.037%** | −0.0022 L2, −0.009 pp CR |
| minS2 seed 2 (4394729) | 0.3660 | 0.047% | +0.0011 |
| 2b unfrozen (4394723) | 0.3661 | 0.060% | +0.0012 |
| headline anchor (3-seed) | 0.3692 | 0.0400% | — |

Every delta is inside the 0.007 L2 / 0.015 pp CR floors. The lowest single
number in the batch is K/V-on minS2 at 0.3608 and the lowest CR is 2a at 0.037%
— **both single seeds, both configurations we have positive reasons not to
adopt** (K/V-on costs +7.88M params for nothing; 2a destroys NDS 0.5279 → 0.4353).
Chasing either would be seed-fishing.

The one substantive observation: five of six arms landing at 0.360–0.366 hints
the headline 3-seed anchor (0.3692) may be marginally pessimistic. The minS2
3-seed anchor (0.3649) sits closer to the batch centre and is the better
reference going forward.

## Priority

Ordered by what a reviewer is most likely to attack, not by cost:

1. **Ablation 5a** — if the planner doesn't need its image readout, Contribution
   (ii) is false and the paper's architecture section needs rewriting. Highest
   information per run.
2. **Ablation 3** — the locked architecture's no-ego number is the one row a
   no-ego mirror table cannot omit.
3. **Ablation 1** — in flight (4394494/4394495/4394496).
4. **Ablation 4a/4c** — eval-only, ~1h each, can run opportunistically against
   existing checkpoints while training slots are occupied.
5. **Ablation 2** — sharpens Stream E and feeds the r101 asymmetry, but does not
   threaten a headline claim.

## Future Work

- **Det-only / map-only split at the current operating point.** If Ablation 1
  shows any effect, re-run the Batch F disaggregation
  (`skip_perception_kv='map'` → det K/V only; `'det'` → map K/V only) on the
  headline stack. The `nodetkv` / `nomapkv` result is from the `_laststage` era
  and predates `egostatus`.
- **Capacity control.** If K/V-on wins, add a parameter-matched K/V-off arm
  (e.g. widened FFN or `decoder8`) to separate interface information from the
  +7.88M.
- **`lr_mult` sweep at stage 2** (0.0 / 0.1 / 0.5). Across all r50 stage-2
  configs only 0.0 and 0.1 were ever used; the r101 asymmetry in
  `reports/2026_07_25_r101_backbone_revisit.md` makes the intermediate values
  reviewer-relevant for the Stream E claim.
- **`norm_eval=False` control on r101 minS2** — isolates the BN-freezing
  confound from backbone size (carried over from the r101 report; belongs here
  as a single-flag row).
