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

Single seed, 4-GPU on Killarney L40S, 12 h wall, stage-2 from
`ckpt/sparsedrive_stage1.pth`. Backbone + det_head + map_head frozen
(`lr_mult=0.0`); det/map cls + reg losses zeroed (mirrors minS2 Stream E).

| # | Config stem | Notes |
|---|---|---|
| 1 | `sparsedrive_r50_stage2_4gpu_bs24_planneronly_skeleton` | conflict head on `image_at_plan`; baseline parity check |

Compare L2 / `obj_box_col` (and the secondary metrics) against
`..._egostatus_minS2.py`. If the gap is small, step 2 lands joint self-
attention on top of this clean head; if not, the LN-vs-no-LN delta or the
`image_at_det → image_at_plan` switch is the first thing to investigate.
