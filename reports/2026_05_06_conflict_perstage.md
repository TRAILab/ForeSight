# conflict_perstage - 2026-05-06

## Intro

The current B1.7 conflict head (`conflict_input='image_at_plan'`) samples image
features once per forward pass at the **static** k-means `plan_anchor` (108
fixed BEV trajectories × T waypoints) and reuses those features across all 6
planning-decoder refine stages. The planner's `plan_anchor` is updated each
stage (cumsum of `plan_reg.detach()` from the previous stage), but the
collision evidence the conflict head reads stays anchored to the prior. This
is consistent with the observation that B1.7 ties minS2 on L2 but elevates CR
by ~0.025 pp (0.066 vs 0.041).

Hypothesis: re-sampling image features at the **live** `plan_anchor` each
refine stage gives non-stale collision evidence at the trajectory hypothesis
the planner is currently considering, and closes the residual CR gap.

## Method

Code change in `projects/mmdet3d_plugin/models/motion/motion_planning_head.py`:

- New constructor flag `conflict_resample_per_stage=False` (validated against
  `conflict_input='image_at_plan'`).
- When the flag is set, the pre-loop `image_at_plan` branch is skipped; instead,
  before each `refine` op, we build `(bs, M*T, 11)` sampler anchors from the
  live `plan_anchor` (already cumulative XY in lidar frame, `(bs, M, T, 2)`),
  zero-yaw / zero-size, and run the existing `conflict_image_sampler` to
  produce per-stage `conflict_image_features` that the refine call consumes.
- All other conflict params (`conflict_loss_weight=0.10`,
  `conflict_smooth_max_tau=5.0`, `conflict_label_source='evalmatch_mode'`, the
  learned-hard rescore thresholds) unchanged.

Two configs (single seed each, 4-GPU on Killarney, stage-2 from
`ckpt/sparsedrive_stage1.pth`):

| # | Config stem | Δ vs B1.7 baseline |
|---|---|---|
| 1 | `..._evalmatchmode_egostatus_minS2_B1p7_perstage` | `+conflict_resample_per_stage=True` |
| 2 | `..._evalmatchmode_minS2_B1p7_perstage` | `+conflict_resample_per_stage=True`, `plan_ego_status_encode_enable=False` |

Variant 2 also serves as the no-egostatus B1.7 reference (none currently exists in the repo).

## Results

Submitted 2026-05-06 on Killarney (single seed, 4-GPU, stage-2 from
`ckpt/sparsedrive_stage1.pth`):

| # | Config stem | Killarney job |
|---|---|---|
| 1 | `..._evalmatchmode_egostatus_minS2_B1p7_perstage` | K3444443 |
| 2 | `..._evalmatchmode_minS2_B1p7_perstage` | K3444448 |

_Metrics pending job completion._

## Discussion

_(pending)_

## Future Work

_(pending)_
