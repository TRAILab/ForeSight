# planning_only_skeleton - 2026-05-06

## Intro

Step 1 of the planner redesign: fork a clean `PlanningOnlyHead` from
`MotionPlanningHead`, strip every motion / det-token / map-token code path,
and submit a parity check against the minS2 ego-only baseline before changing
the architecture in step 2 (joint self-attention etc.).

`MotionPlanningHead` has accumulated ~3700 lines of optional flags
(softtgt, distill_rescore, magnitude head, scene-query, kinematic motion,
agent-frame regression, time-axis attention, planning_temporal_stack, DN
groups, …). Most are gated off by minS2; the dispatch is still mixed into
the same `forward`. Continuing to extend that in place is too risky, so we
fork a slim head that does only what minS2's `ego_only_planning=True`
configuration does, and verify it reaches similar planning numbers before
introducing any new structure.

## Method

### New files

- `projects/mmdet3d_plugin/models/motion/planning_only_blocks.py`
  - `PlanningOnlyRefinementModule`: only `plan_cls`, `plan_reg`,
    `plan_status` (10-D), and `plan_conflict` branches. No `motion_cls` /
    `motion_reg`. Forward signature
    `(plan_query, ego_feature, ego_anchor_embed, agent_features=None) →
    (plan_cls, plan_reg, plan_status, plan_conflict)`.
- `projects/mmdet3d_plugin/models/motion/planning_only_head.py`
  - `PlanningOnlyHead`, registered under `HEADS`. Same forward / loss /
    post_process signatures as `MotionPlanningHead` so
    `SparseDriveHead.build_head(motion_plan_head)` dispatches it without
    code changes (verified by reading `sparsedrive_head.py` — the only
    contract is the head type's signature).
  - Geometry helpers (`_eval_get_yaw`, `_agent_get_yaw`,
    `_make_rect_corners_topdown`, `_rect_intersects_sat`) are imported
    from `motion_planning_head.py` to avoid duplication.
- `projects/configs/sparsedrive_r50_stage2_4gpu_bs24_planneronly_skeleton.py`
  - Copy of `..._egostatus_minS2.py` with `motion_plan_head.type` swapped
    and the matching subset of init args.

`projects/mmdet3d_plugin/models/motion/__init__.py` re-exports the new
classes so the registry sees them.

### What's preserved (vs minS2)

- `InstanceQueue` — driven by `det_output` so the queue's tracking state
  (`prev_confidence`, `prev_instance_id`, `metas`) stays consistent across
  frames. The agent slots in `temp_instance_feature` are sliced out
  immediately; only the ego entry feeds `temp_gnn`.
- `ego_feature_encoder` (front-cam FPN → ego token), inside `InstanceQueue`.
- `plan_anchor` k-means buffer + `plan_anchor_encoder`,
  `plan_ego_status_encoder` with indices `(0, 1, 5, 6, 7)`.
- `temp_gnn` with `decouple_attn=True` over the ego token only (single-token
  query against the ego temporal queue; matches minS2's
  `decouple_attn_motion=True`).
- Per-stage planning deformable. Stages 1–5 update `plan_mode_query` from
  multi-waypoint deformable attention (planning_deformable_waypoints =
  `[0,1,2,3,4,5]`, K-pool back to per-mode mean). The **last stage** replaces
  the deformable query with the ego `instance_feature` (mode-aggregated by
  softmax over the previous stage's plan classification), matching the minS2
  `planning_deformable_instfeat=True` + `_laststage=True` recipe.
- Refine: `plan_query = plan_mode_query.unsqueeze(1) +
  (instance_feature + anchor_embed)[:, 0:1].unsqueeze(2)` — additive
  broadcast preserved.
- `plan_reg += prev_plan_reg.detach()` cumulative refinement.
- Conflict head with `rescore_learned_hard` plumbing.
- `HierarchicalPlanningDecoder` post-processing
  (`use_rescore_learned_hard=True`, prob/score thresh `0.5`, agg `'any'`).

### Stripped

- `gnn` / `cross_gnn` / `rev_gnn`, top-K det/map selection,
  `instance_feature_selected`, `num_det`, `num_map`, `use_alldet_kv`,
  `skip_perception_kv`.
- All map-feature plumbing.
- `motion_anchor`, `motion_anchor_encoder`, `motion_mode_query`,
  `motion_deformable*`, `motion_decoder`, `MotionTarget`, agent-endpoint
  builders, `_agent2lidar`, `motion_target_in_agent_frame`.
- DN-agent and DN-ego tokens.
- `plan_self_attn`, `_use_instfeat` / `_laststage` / `_additive` /
  `mode_no_agg` branches in the deformable stage selector.
- `softtgt`, `plan_magnitude_head`, `softcost`, `distill_rescore`,
  `scene_query_decoder`, `ego_state_estimator`, `relevance_selection`,
  `planning_temporal_stack`, `plan_anchor_norm_mode`, time-axis attention,
  `_MotionPlanningAdapter` (input/planning embed_dims always equal here).

### `operation_order` and the conflict head

The skeleton uses the user-specified
`[temp_gnn, deformable, ffn, refine] × 6` op order. minS2 has
`[temp_gnn, gnn, norm, cross_gnn, norm, deformable, norm, ffn, norm,
refine] × 6` with `gnn`/`cross_gnn` nulled by `skip_perception_kv=True`.
Effective minS2 ops are
`[temp_gnn, LN, LN, deformable, LN, AsymFFN(LN+...), LN, refine]`. The
skeleton is `[temp_gnn, deformable, AsymFFN(LN+...), refine]` — i.e. **4
LayerNorms per stage are dropped**. AsymFFN's internal pre-norm is the only
LN that survives between ops. This is a deliberate simplification per the
step-1 spec; expect a small numerical deviation from minS2 because of it.

Conflict input is `image_at_plan` (det-free): we sample image features once
per forward at the static `plan_anchor` cumsum BEV positions
(`M*T = 18*6 = 108` queries), reuse across all six refines, and apply the
per-mode smooth-max BCE against the `evalmatch_mode` label. minS2 itself
uses `image_at_det` (det-anchor positions), but `image_at_det` requires
det-token Hungarian matching plumbing that we deliberately stripped, so the
skeleton commits to `image_at_plan` from the start. `rescore_learned_hard`
already handles per-mode-pooled conflict samplers (its
`det_compatible = (det_confidence.shape[-1] == num_anchor)` check skips
det-confidence filtering when shapes don't align), so inference rescore
works unchanged.

### Smoke test

Inside the `foresight` Docker image:

- Config loads; `motion_plan_head.type=PlanningOnlyHead`; 24 ops in
  `operation_order`, 6 deformable stages, 6 refine stages.
- `build_detector(cfg.model)` succeeds.
- Total 90,924,238 params; planner 15,248,356 params.

## Run

Single seed each, 4-GPU on Killarney L40S, 12 h wall, stage-2 from
`ckpt/sparsedrive_stage1.pth`. Backbone + det_head + map_head frozen
(`lr_mult=0.0`); det/map cls + reg losses zeroed (mirrors minS2 Stream E).

| # | Killarney job | Config stem | Notes |
|---|---|---|---|
| 1 | 3445407 | `sparsedrive_r50_stage2_4gpu_bs24_planneronly_skeleton` | `plan_ego_status_encode_enable=True`; conflict head on `image_at_plan`; parity check vs `..._egostatus_minS2.py`. **Landed L2=0.3714/CR=0.045% — parity passes** (vs minS2×B1.6 2-seed mean 0.3742/0.038%). |
| 2 | 3445429 | `sparsedrive_r50_stage2_4gpu_bs24_planneronly_skeleton_noegostatus` | `plan_ego_status_encode_enable=False`; A/B isolates the ego_status injection contribution; parity check vs `..._minS2.py` (no-egostatus baseline). **Landed L2=0.5206/CR=0.088% — parity passes** (vs no-ego 3-cluster mean 0.5197/0.068%): ΔL2=+0.001 (within noise), ΔCR=+0.020 pp (at noise edge). |

### Debug log — companion-config crashes

Three companion configs crashed during their first Killarney submission and
were debugged locally. All three were eval-time or DDP errors, not training
divergence:

| Config | First job (status) | Root cause | Fix | Rerun |
|---|---|---|---|---|
| `_egoplanner_scratch` | K3446865 (3h08m fail), K3447441 same class | `KeyError: 'final_planning'` — `EgoPlannerSparseDriveHead.post_process` double-wrapped the planning result in `img_bbox` (SparseDrive base already wraps). | Drop the outer `img_bbox` wrap; return `self.planner.post_process(...)` unchanged. | K3448826: L2=0.392 / CR=0.156% (only val/L2 + val/obj_box_col logged; perception heads stripped). Trains and evals end-to-end; metrics in their own neighborhood since this config is a from-scratch ego-planner without S1 init. |
| `_isolated` | K3446866 (3h34m fail) | `KeyError: 'scores'` in `format_map_results` — `task_config.with_map=False` (head removed) but `eval_mode.with_map=True`. | Set `eval_mode.with_map=False` to match `task_config`. | K3448827: L2=0.609 / CR=0.215%, NDS=0.0, car_EPA=−0.71 — eval pipeline runs to completion, but detection collapses (degenerate boxes); separate config bug yet to chase. |
| `_vanilla_planneronly`, `_vanilla_planneronly_scratch` | K3447441/3447442 (~10–11 min) | DDP `Parameter at index 157 marked ready twice` — `with_cp=True` + multi-use parameters (e.g. `plan_anchor_encoder` invoked once at init and again inside every refine stage) collides with `find_unused_parameters=True`. Diag confirmed 0 unused params. | Replace `find_unused_parameters=True` with `static_graph=True` (PyTorch's documented fix for this error class; safe given 0 unused). | K3453958 segfaulted (~11m, SIGSEGV at iter 0 on rank 1/2 — different failure class from the DDP error; static_graph plumbing landed but rerun crashes worker processes before training begins). K3453959 CANCELLED. The DDP root cause is patched; the segfault is the next thing to chase (likely an unrelated env/op issue surfacing once DDP no longer aborts first). |

Compare L2 / `obj_box_col` (and the secondary metrics) against the matching
minS2 baseline for each row. If the gap is small, step 2 lands joint self-
attention on top of this clean head; if not, the LN-vs-no-LN delta or the
`image_at_det → image_at_plan` switch is the first thing to investigate.

## Future plan

Once the skeleton hits parity with minS2, the head becomes a clean substrate
for three structural changes that today's `MotionPlanningHead` makes
expensive to attempt. Each is a single-experiment increment from the
preceding step.

### Step 2 — Joint [ego ⊕ modes] self-attention

Today the planner runs two parallel tracks: `ego_feature` (1 token) goes
through `temp_gnn → ffn`, `plan_mode_query` (M=18 tokens) goes through
`deformable`, and the only fusion is the additive broadcast at refine
(`plan_query = plan_mode_query + ego_feature + ego_anchor_embed`).
Consequences:

- **No mode ↔ mode interaction.** The 18 modes never see each other; mode
  confidences are produced independently from each token's own MLP read.
  This is the most plausible explanation for why `plan_cls` alone is
  poorly calibrated and the conflict head exists as a rescore prosthetic.
- **No mode-specific ego conditioning.** Ego sends the *same* vector to
  every mode via broadcast addition. The refine MLP has to disentangle
  ego state from anchor identity using the same feature dimensions.
- **No back-flow plan → ego in stages 1–5.** Only the last-stage
  deformable lets ego absorb trajectory-grounded scene context.

Step 2 unifies the two tracks. Tokens = `[ego, mode_1, ..., mode_18]`,
shape `(bs, 19, D)`. Per stage:

1. `temp_gnn` (ego only — modes have no natural temporal correspondence,
   keep the queue ego-only).
2. **`token_self_attn`** — `MultiheadAttention(D=256, heads=8)`,
   `Q = K = V = tokens`, `query_pos = type_embed + anchor_pos_embed`
   (ego anchor for ego, sineembed of waypoint for modes). Replaces the
   additive broadcast at refine.
3. `deformable` — same multi-waypoint sampler, applied per token at its
   own BEV position(s). Ego at ego origin (or front-of-car); modes at
   their 6 waypoints with mean-pool back to per-mode.
4. `ffn` (per token).
5. `refine` — emit `plan_cls` / `plan_reg` from mode tokens, `plan_status`
   from the ego token. **Drop the additive broadcast** — interaction is
   now learned by `token_self_attn`.

The skeleton's parity check makes it possible to attribute any L2 / CR
delta to this single architectural change instead of confounding it with
the strip-down from `MotionPlanningHead`.

### Step 3 — Unified deformable sampling primitive

Today, the planner runs **two separate** `DeformableFeatureAggregation`
instances with `SparseBox3DKeyPointsGenerator`:

- main planner deformable (per stage, ego-sized 11-D boxes at live
  trajectory waypoints, content-bearing query),
- B1.7 conflict deformable (once per forward, point-like 11-D boxes at
  static k-means waypoints, zero-vector query).

Both inherit detection-legacy machinery: an 11-D anchor with size/yaw
generates 13 keypoints (6 learnable + 7 box-corner offsets *scaled by box
dimensions*). For the planner this is bureaucratic packaging — there's no
"box" to talk about, only a sampling position. The unit-sized 11-D anchor
B1.7 builds is purely to keep using the detection sampler.

Step 3 replaces both with a **position-based** sampler:

- `PositionalKeyPointsGenerator(num_learnable_pts=N, offset_scale=σ)`.
  Input: `(bs, K, 3)` query positions. Output: `K · N` 3D points = query +
  N learnable offsets bounded by σ. No box-local scaling, no fixed corner
  pattern.
- Same projection / multi-cam aggregation as `DeformableFeatureAggregation`.
- Two practical knobs: `N` (controls the spatial footprint — large `N`
  reproduces B1.6's "broader area" effect via learning, no fixed library
  needed); `σ` (controls how far offsets can wander).

This collapses the full B1.x sampler taxonomy (B1, B1.6, B1.7, B1.8, B1.9)
into "what's `N` and where do the queries start." It also drops the 11-D
anchor format from the planner entirely; only the temporal queue retains
11-D for `InstanceQueue` legacy.

### Step 4 — Simplified, unified conflict head

Once main and conflict deformables share a primitive, the conflict head
no longer needs its own sampler pass. At the last stage:

- The planner deformable already produced per-(mode, waypoint) attended
  features `(bs, 18, 6, D)` *before* mean-pooling back to per-mode. This
  is the *exact* tensor the conflict head wants — image evidence at the
  current trajectory iterate, with the planner's own learned offsets.
- Pair `attended[:, m, t, :]` with `plan_query[-1][:, m, :]`, run the
  pairwise conflict MLP, get `conflict_logits (bs, 108, 18)` directly.
- Loss + `rescore_learned_hard` plumbing unchanged.

What this buys:

- **One deformable call per forward** instead of two (planner + conflict
  sampler). Roughly half the deformable cost in the post-decoder section.
- **Live trajectory grounding for free.** Today's B1.7 samples at the
  static k-means waypoints; the unified version samples at the
  cumsum-of-`plan_reg` waypoints because that's already what the planner
  did. This is exactly the per-stage-resampling lever discussed in
  `2026_05_06_conflict_perstage.md`, but without the cost — it falls out
  of unification.
- **Tied gradients.** Conflict and planner share keypoint offsets, so the
  same scene reads inform both `plan_reg`/`plan_cls` and the collision
  logit. Whether this is a feature (consistent scene grounding) or a bug
  (BCE on conflict pulls keypoints in a direction that hurts L2) is the
  empirical question step 4 answers.

If the tied-gradient regression is real, the fallback is to keep separate
sampler weights but *share the primitive and the query positions* — i.e.,
two `PositionalKeyPointsGenerator` instances both reading at live
trajectory waypoints. Still simpler than today's mix of box sizes and
static-vs-live anchor sources.

### Step 5 (optional) — Per-stage conflict refresh

If step 4 is undesirable for any reason (e.g., the unified head proves
unstable to train), a strictly cheaper intermediate is **B1.10**: keep
the standalone conflict deformable but recompute it inside the decoder
loop at the live `plan_anchor` cumsum each stage, and produce per-stage
conflict logits supervised by the existing per-stage BCE. This is the
prompt drafted in `2026_05_06_conflict_perstage.md`.

Order of preference: step 4 first (cleaner, cheaper). Step 5 is the
fallback if step 4 trades L2 for CR or trains unstably.

### Step 6 (orthogonal axis) — Predicted ego_status without ground-truth read

The `plan_ego_status_encode` head reads `ego_status[:, [0,1,5,6,7]]`
(planar accel, yaw rate, planar velocity) directly from sample
metadata. This contributes ~0.15 L2 (full-ego paper headline 0.37 vs
strict no-ego 0.52 on the K/V-off architecture) — uniformly across S1
swaps. It is also the single biggest reason the headline can't be
compared apples-to-apples against UniAD/VAD/SparseDrive baselines that
don't consume ego_status. The `egopred_lite` variant (history-only MLP
over past ego_status) lands at L2=0.4924 / CR=0.088% — recovers ~30 %
of the ego advantage but still touches ego_status from past frames.

Step 6 disconnects the ego signal from raw ego_status read entirely.
The ego encoder still emits a 5-dim vector consumed by the same
`plan_ego_status_encoder` MLP; what changes is the source of that
vector. The token redesign (step 2 — joint `[ego ⊕ modes]` self-
attention) makes the most natural variant fall out for free.

**Variant A — `egopred_visual` (history-free).** New head
`VisualEgoStateHead`: MLP over the ego token after step-2's
self-attention but before refine. Outputs the 5-dim ego_status surrogate
(accel_x, accel_y, yaw rate, vel_x, vel_y). Consumes only image
features (the ego token is constructed from front-cam FPN features in
`InstanceQueue.ego_feature_encoder`). Supervised by L1 on the raw
ego_status during training; at inference the prediction replaces the
ego_status read entirely. **Strictly no-ego**: zero ego_status
consumption at inference, zero past-frame ego_status during training
data assembly (could optionally drop ego_status from `Collect` keys
and load only when supervising the loss).

**Variant B — `egopred_attention` (architecture-native).** Skip the
separate VisualEgoStateHead. After step 2, the ego token is already
attended to image features through the unified positional deformable
(step 3) at the ego origin. Treat the ego token's `plan_status_branch`
output as the ego_status prediction directly. No new module, no extra
loss term — the existing `plan_status_branch` (which today regresses
ego_status anyway) becomes the prediction source, and the
`plan_ego_status_encoder` MLP is wired to its output instead of to
the GT meta. Cleanest: ego_status flows through the planner without
any external read path.

**Variant C — `egopred_kinematic` (preprocessing).** Compute current
ego_status from past trajectory deltas (velocity = Δposition / Δt,
accel = Δvelocity / Δt) at the data-pipeline stage. No model change;
no learned head. Still consumes past ego positions, so not strictly
"no ego_status" — but it removes the dependence on the raw sensor's
current-frame fields. Useful as a sanity-check baseline against the
learned variants.

**Recommended order**: A (cleanest, fastest to build, isolates the
visual contribution), then B if A wins on L2 (see if collapsing into
the existing plan_status_branch is "free"), then C as a baseline if
reviewers ask "did the model learn anything beyond kinematics?"

What this buys: a non-trivial datapoint between strict no-ego (~0.52)
and full-ego (~0.37) that uses no ego_status at inference. If
`egopred_visual` lands at ~0.45–0.48, it's a publishable mid-tier
baseline (and better than `egopred_lite` 0.49 because it drops past
ego_status too). If it lands at 0.52 (= strict no-ego), the ego signal
is cheaper to read directly than to predict, and the paper's no-ego
headline doesn't change.

This is orthogonal to steps 2–5. It can be built on top of the
skeleton or added after the joint-attention redesign — the variants A
and B specifically benefit from step 2's token unification because the
ego token then sees scene context through self-attention rather than
through a separate `temp_gnn` track.

## Design philosophy

Three principles the redesign is built around. Each one is the inverse
of a SparseDrive-inherited choice that makes today's planner harder to
read and harder to extend.

**1. Positions, not boxes.** The planner's queries are trajectory points.
Carrying around 11-D anchors with size, yaw, and velocity is a detection
abstraction that doesn't fit. Drop it everywhere except where temporal
matching genuinely needs anchor geometry (the `InstanceQueue`).

**2. One sampling primitive.** Today there's a main deformable and a
conflict deformable, with different anchor box conventions, different
weight modules, and a two-pass forward at inference. The unified planner
has one operation — *sample image features at a set of 3D positions* —
and uses it everywhere image features enter the planner. The B1.x sampler
zoo collapses into a single `(N, σ)` knob.

**3. Learned interactions over hand-wired fusion.** The current head
mixes ego, agents, modes, and time through ad-hoc additive broadcasts,
last-stage-special-cased deformables, and mode-aggregation softmaxes
keyed off `plan_cls`. The new head uses a single self-attention over a
small token set `[ego ⊕ M modes]` — every interaction is learned, every
stage runs the same operations, and there are no flags like
`_use_instfeat_laststage`, `_additive`, or `mode_no_agg` to keep the
forward path readable.

The headline architecture that comes out the other side reads as:

> *Ego token + M plan-mode tokens are jointly self-attended each stage,
> then each token samples image features at its own BEV position via a
> shared positional deformable. Per-mode classification and regression
> come from the mode tokens; per-frame ego-state regression comes from
> the ego token; conflict logits come from pairing the same per-(mode,
> waypoint) image features with the mode tokens. No detection, map,
> or motion tokens enter the planner at inference.*

That sentence is the paper figure. Today's code can't be described that
simply without lying.
