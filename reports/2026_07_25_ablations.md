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
