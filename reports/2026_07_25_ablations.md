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

### Results

`[pending]`

| Model | L2 | CR | NDS | mAP_normal | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| K/V-off headline (3-seed anchor) | 0.3692 | 0.0400% | 0.5212 | 0.5533 | reference |
| K/V-on headline (seed 0) | `[ ]` | `[ ]` | `[ ]` | `[ ]` | |
| K/V-on headline (seed 1) | `[ ]` | `[ ]` | `[ ]` | `[ ]` | |
| minS2 (2-seed anchor) | 0.3644 | 0.046% | 0.5279 | 0.3363 | reference |
| K/V-on minS2 (seed 0) | `[ ]` | `[ ]` | `[ ]` | `[ ]` | |

### Discussion

`[pending]`

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
layer to `None` so DDP does not see its params.

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
| 4 | reference eval (veto on) | Fir H100 | 51125562 |
| 4a | veto off | Fir H100 | 51125563 |
| 4c | hard rescore | Fir H100 | 51125564 |

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
