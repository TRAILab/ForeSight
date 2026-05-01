# Anchor Capacity Allocation: Speed–Maneuver Decoupling

2026-04-30

## TODO

- [x] Histogram `||gt_endpoint||` and `||v_0||` (`reports/anchor_norm_histograms*.png`) — picked ε=1.0m / ε_v=1.0m/s.
- [x] Generate `kmeans_plan_shape_endptnorm.npy` + refmag and `kmeans_plan_shape_velnorm.npy` (`reports/kmeans_plan_shape_*.png`).
- [x] Add per-mode magnitude head (`Linear(D, 1)` per decoder stage) + smooth-L1 magnitude loss for variant 1a/3.
- [x] Refactor `MotionPlanningHead.plan_anchor` from static buffer to per-batch tensor (`_get_initial_plan_anchor` helper).
- [x] Add `plan_ego_status_encoder` MLP (`ego_status[:, [0,1,5,6,7]]` → `plan_mode_query` additive injection) for variants 2/3.
- [x] Add per-bucket `plan_reg_branch` for variant 4 (modes 0-5 → low, 6-11 → high on speedstrat anchors).
- [x] Submit the 5-experiment parallel batch on Killarney `ptaux2d_ppdeformmm_planifls` baseline (3368761–3368765).
- [x] No-rescore eval on existing `modesspeedstrat` checkpoint (3368766) → L2=0.5560 vs 0.5548 with rescore: regression is **training-time**, not decode-time.
- [ ] Resubmitted eval (3376375 variant 3, 3376376 variant 4) — cluster-wide failure killed first eval phase mid-run.
- [ ] **Variant 1a refmag bug**: cmd=Straight stopped-cluster k=0 has `refmag=0.0` from k-means cluster median, which collapses the metric anchor to zero and degenerates `gen_sineembed_for_position`. Fix: clamp `refmag = max(refmag, 1.0)` in `gen_kmeans_plan_shape_endptnorm.py` (or at load time in `motion_planning_head.py:_get_initial_plan_anchor`), regenerate refmag, retry 1a + 3 if needed.
- Cheap follow-up: `plan_yaw` config (`plan_reg ts*2 → ts*3`, supervise yaw from `gt_ego_fut_yaw`) — independent lever, stackable with `egostatus`.

## Headline result (2026-04-30)

`egostatus` is the win: **L2 0.4988 → 0.3701 (−0.13, ~26 noise-floor units)**, CR 0.063% → 0.058% (Killarney 3368763). Confirms the "missing current-frame ego state" diagnosis from the integration audit above. Velnorm anchors alone are tied with baseline; endpt-norm with mag head needs a refmag floor before retry. Variant 3 (stack) and variant 4 (per-bucket speedstrat reg) eval pending.

## Abstract

The current per-cmd planning anchors collapse to a 1-D speed manifold: for cmd "Straight," all 6 k-means anchors are forward-only with lateral offsets under 0.16m, and for cmd "Right"/"Left" the curvature-vs-speed correlation makes the 6 modes a 1-D arc rather than a 2-D maneuver-speed plane. Every prior experiment that scaled or restructured this anchor bag regressed L2 (modes12, modes20diverse, modesspeedstrat, modetime36q) — see `2026_04_26_plan_gaps.md` Batches B/D — because they kept the same speed-collinear allocation. This report proposes a 5-experiment parallel batch comparing two anchor normalization variants (endpoint-magnitude with a learned magnitude head vs. velocity with `v_0` as the observed unnormalization factor), the speed-conditioning lever alone, the stacking experiment on the endpoint variant, and a per-bucket reg-branch follow-up on speedstrat as an independent lever.

## Intro

The Killarney `ptaux2d_ppdeformmm_planifls` baseline lands at L2=0.4988, obj_box_col=0.063%. Every architectural perturbation to the planning mode bag in the last two batches has regressed (full table in `2026_04_26_plan_gaps.md` Batches B and D):

- `modes12`, `modes20diverse`, `modesspeedstrat` — anchor count and structure
- `modetime36q`, `modetime36q_timeattn` — query factorization
- `noagg` — mode aggregation

The pattern is consistent enough that the failure mode is unlikely to be specific to any one variant. A plausible common cause: every variant inherits the *speed-collinear anchor manifold* of the original 6-mode k-means. Adding more anchors along the same manifold doesn't add representational capacity in the directions the planner actually needs (lateral within-lane variation, swerve, lane-change at constant cmd). This report verifies that allocation problem and proposes a fix that has not been tested.

Question:
- Can we decouple shape from magnitude in the planning anchors so each anchor represents a distinct maneuver template, freeing the model to predict speed as a separate continuous variable?

## Diagnosis: anchor capacity allocation

The k-means anchor file `data/kmeans/kmeans_plan_6.npy` has shape `(3, 6, 6, 2)` = (cmd, mode, ts, xy), in absolute LiDAR-frame meters at 0.5s steps over a 3s horizon (verified by inspecting `tools/data_converter/gen_kmeans_plan_12.py:22` which clusters on `gt_ego_fut_trajs.cumsum(axis=-2)`).

LiDAR-frame convention (verified from `tools/data_converter/nuscenes_converter.py:387-390`, where `ego_fut_trajs[-1][0] >= 2 → Turn Right`): **`x = right`, `y = forward`**. So a "go forward" anchor has `x ≈ 0` and `y > 0`.

### Endpoints at t=3s

| cmd | mode | endpoint (x_right, y_forward) | distance | implied avg speed |
|---|---|---|---|---|
| Straight | 0 | (+0.00, +0.63) | 0.63m | 0.2 m/s |
| Straight | 4 | (+0.04, +7.89) | 7.89m | 2.6 m/s |
| Straight | 2 | (+0.11, +15.05) | 15.05m | 5.0 m/s |
| Straight | 5 | (+0.13, +21.31) | 21.31m | 7.1 m/s |
| Straight | 1 | (+0.16, +28.06) | 28.07m | 9.4 m/s |
| Straight | 3 | (+0.15, +39.61) | 39.61m | 13.2 m/s |
| Right | 2 | (+4.82, +8.40) | 9.68m | 3.2 m/s |
| Right | 5 | (+5.76, +13.19) | 14.40m | 4.8 m/s |
| Right | 1 | (+4.33, +18.33) | 18.83m | 6.3 m/s |
| Right | 3 | (+3.47, +24.65) | 24.90m | 8.3 m/s |
| Right | 4 | (+3.70, +30.57) | 30.80m | 10.3 m/s |
| Right | 0 | (+3.14, +46.25) | 46.35m | 15.5 m/s |
| Left | 4 | (−4.33, +8.14) | 9.22m | 3.1 m/s |
| Left | 0 | (−5.89, +13.12) | 14.38m | 4.8 m/s |
| Left | 2 | (−3.95, +18.47) | 18.89m | 6.3 m/s |
| Left | 3 | (−3.43, +24.89) | 25.13m | 8.4 m/s |
| Left | 1 | (−3.58, +32.91) | 33.11m | 11.0 m/s |
| Left | 5 | (−2.88, +51.02) | 51.10m | 17.0 m/s |

### Reading

- **Straight: 6 anchors of pure speed variation.** All lateral offsets `|x| < 0.16m`. Six anchors covering 0.2 → 13.2 m/s of forward speed, *zero* representation of within-lane drift, lateral correction, or swerve. The entire mode budget for the most common cmd is wasted on a 1-D speed axis.
- **Right and Left: 1-D arc, not a 2-D maneuver-speed plane.** Lateral offset shrinks as forward distance grows (slow→sharp turn, fast→barely turning). Computed `atan2(x, y)` for Right cmd: 30° (slow) → 24° → 13° → 8° → 7° → 4° (fast). Speed and curvature are tied along a single curve. The model has no anchor for "fast sharp turn" or "slow gentle turn," and no orthogonal axis to express path-shape variation at constant turn rate.

### Why mode-count scaling didn't help

The Batch B/D failures (`modes12`, `modesspeedstrat`, `modes20diverse`) all ran k-means on the same `gt_ego_fut_trajs.cumsum` with the same allocation problem. Doubling K just doubles the density along the speed manifold; tripling it adds anchors that are *more* speed-redundant. `modesspeedstrat` got the best CR (0.070% vs 0.063% baseline) precisely because bucketing by speed at clustering time gave each bucket tighter per-anchor distributions — but the L2 cost (0.4988 → 0.5548) reflects that the decoder still has to discriminate between near-identical-shape anchors, just with the wrong axis (speed instead of maneuver) carrying the discrimination load.

## `ego_status` integration audit

Verified the existing pathway end-to-end. Summary: `ego_status` is **only weakly integrated** in the planning head, *not* through `plan_mode_query`.

### What's there

- `data['ego_status']` is a 9-dim vector loaded from CAN bus pose (`tools/data_converter/nuscenes_converter.py:432-434`):
  - `[0:3]` = acceleration (CAN ego frame: `x=forward, y=left`)
  - `[3:6]` = rotation rate (roll, pitch, yaw)
  - `[6:9]` = velocity (CAN ego frame: `vx=forward, vy=left`)
- It is used as a **supervision target** at every decoder stage. The head predicts a `plan_status` per layer; `plan_loss_status` regresses it against `data['ego_status']` (`motion_planning_head.py:1983`).
- The *previous* frame's *predicted* `plan_status` is cached by `instance_queue.cache_planning` (`instance_queue.py:211-212`) and on the next frame written into `ego_anchor[..., VY] = prev_ego_status[..., 6]` (`instance_queue.py:181`). The 11-dim `ego_anchor` is encoded by `anchor_encoder` into `ego_anchor_embed`, which is added to `plan_query` at the refine step (`motion_planning_head.py:1693`).

### Verified: no axis bug despite confusing index naming

The assignment `ego_anchor[..., VY] = prev_ego_status[..., 6]` looks like an index mismatch (CAN-frame `vx` written into LiDAR-frame `VY`), but the two frames are 90°-rotated about z:

- CAN ego frame: `x = forward`, `y = left`
- LiDAR frame: `x = right`, `y = forward` (verified via `nuscenes_converter.py:387-390` cmd derivation)

So CAN-frame `vx` (forward) physically equals LiDAR-frame `VY` (forward). The longitudinal velocity component lands on the right axis. **No bug.** The remaining lateral component (`ego_status[7]`, CAN `vy_left` = LiDAR `−VX`) is dropped from the temporal pathway — fine for lane-keeping, lossy during lane changes.

Minor: `augment.py:194-202` docstring claims `ego_status` is in the "LiDAR-aligned ego frame," which is wrong — it's in the CAN ego frame. The 2D z-rotation augmentation is still mathematically consistent (rotation about z is the same physical operation in either frame), so this is a comment-level mislabel only.

### What's missing

- **Current-frame `ego_status` never reaches the head as input.** Only the previous frame's predicted longitudinal velocity, lagged by one step, indirectly via `ego_anchor_embed`.
- **Acceleration, yaw rate, steering** are never input to the head — only used as supervision via `plan_loss_status`.
- **`plan_mode_query` itself is purely position-derived** (`plan_anchor_encoder(plan_pos)` at `motion_planning_head.py:1219`). No ego state injection.
- The temporal feedback uses *predicted* `plan_status`, so error compounds at frame 0 and during occlusion.

### Implication for this report

A magnitude head (proposed below) wants the current speed signal as input. Since current `ego_status` is fully absent from the head, this is a clean addition — not a duplication of existing pathways.

## Speed-invariance design space

Eight ways to remove or reallocate the speed-collinear anchor capacity. Ordered roughly cheapest-first.

### Anchor-side normalization

1. **Scale anchors by current speed (`v_0`)** — k-means on `traj / max(v_0, ε)`; runtime multiplies anchor by live `v_0`. Cheapest. Fragile when `v_0 ≈ 0` (stopped at light); needs a floor or a separate stopped branch.
2. **Endpoint-magnitude normalization** — k-means on `traj / ||traj_end||`, predict magnitude as a scalar head. Decouples shape from distance entirely; no `v_0` dependence. Adds one regression scalar + loss term. **Recommended (see below).**
3. **Per-step delta normalization** — cluster on `Δposition / Δt` (velocity sequences) instead of cumulative xy. Removes starting-position drift; the constant-speed prior becomes explicit-and-zero rather than implicit.
4. **Arc-length parameterization** — anchors describe `(lateral_offset, heading)` at fixed *distance* steps instead of fixed time. Decouples path geometry from timing entirely; needs a separate timing/speed head to reconstruct waypoints.
5. **Curvature/yaw-rate anchors** — cluster in a control space `(longitudinal accel, yaw rate)(t)`, reconstruct positions at inference by integrating against `(v_0, yaw_0)`. Strongest invariance; most code change.

### Decoder-side speed-awareness (no anchor change)

6. **Speed-conditioned `plan_mode_query`** — inject `(v_0, a_0, yaw_rate_0)` into the plan query (or its anchor embed) so the model can pick differently among existing 6 anchors based on speed regime. Cheap; addresses the diagnosed selection-side L2 problem in speedstrat. Diagnostic value > expected impact (the model already gets a lagged longitudinal-speed signal via `ego_anchor_embed`).
7. **Per-bucket reg branch on speedstrat checkpoint** — keep the 12 stratified anchors, give each speed bucket its own `plan_reg_branch` MLP. The original `plan_gaps.md` Batch D follow-up. Tests whether speedstrat's L2 was decoder-shared-capacity, not anchor-side.

### Hybrid (recommended direction)

8. **Shape anchors + separate magnitude head** — endpoint-normalized shape anchors (option 2) *plus* a small head emitting a magnitude scalar per mode. Cleanest decomposition; matches how classical planners separate path from velocity profile. **This is the proposed next experiment.**

## Floor selection for endpoint normalization

The endpoint-normalization formula is `traj_norm = traj / max(||end||, ε)`. The floor exists because a non-trivial fraction of GT trajectories have very small endpoints (parking, red light, traffic jam) where dividing by the raw endpoint produces meaningless or unstable normalized shapes.

**Approach**: histogram `||gt_endpoint||` across the train set first, then pick ε at the elbow rather than picking arbitrarily. This gets baked into `tools/data_converter/gen_kmeans_plan_shape.py` so the choice is recorded.

- If the histogram shows a clean elbow, set ε at that point.
- If the histogram is monotonically decreasing with no elbow, default to ε ≈ 1.0m (samples below ~1m of total motion don't carry useful shape information; lumping them into a single near-zero cluster is fine).
- Optional refinement: route any sample with `||end|| < threshold` to a dedicated "stop" mode rather than normalizing it. Cleaner theoretically; defer until #1 lands and we see whether the magnitude head struggles on stopped samples.

## Recommended parallel batch

Five-experiment ablation: two normalization variants (endpoint-magnitude vs velocity), the speed-conditioning lever alone, one stacking experiment on the endpoint variant, plus `modesspeedstrat`'s diagnosed-cause follow-up as an independent lever. All on `ptaux2d_ppdeformmm_planifls` Killarney baseline (L2=0.4988, CR=0.063%) for matched comparison. Each run targeted at ~12h.

| # | Config suffix | Mechanism | Isolates |
|---|---|---|---|
| 1a | `_shapeanchor_endptnorm_mag` | Endpoint-norm anchors (`traj / max(\|\|end\|\|, ε)`) + per-mode magnitude head + magnitude L1 loss, K=6/cmd, **no** `ego_status` injection | Shape/magnitude decoupling with *learned* magnitude |
| 1b | `_shapeanchor_velnorm` | Velocity-norm anchors (`traj / max(v_0, ε)`); at inference `final = shape * current_v_0`, **no magnitude head, no extra loss term**, K=6/cmd, no `ego_status` injection | Shape/magnitude decoupling with *observed* magnitude (free at inference) |
| 2 | `_egostatus` | Baseline anchors, **+** `ego_status` MLP-encoded into `plan_mode_query` (no anchor change, no magnitude head) | Speed conditioning alone — tests "is mode selection speed-blind" |
| 3 | `_shapeanchor_endptnorm_mag_egostatus` | 1a + 2 stacked | Stacking gain on endpoint variant: does conditioning help on top of decoupling? |
| 4 | `_modesspeedstrat_perbucketreg` | Existing speedstrat anchors + per-bucket `plan_reg_branch` MLP | Independent: addresses speedstrat's diagnosed L2 cause without touching anchor allocation |

### Endpoint-norm (1a) vs velocity-norm (1b) tradeoff

Both decouple speed from the anchor. They differ in the unnormalization factor:

| | Endpoint-norm (1a) | Velocity-norm (1b) |
|---|---|---|
| Anchor shape units | dimensionless (`m/m`) | seconds (`m / (m/s)`) |
| Unnormalization factor | `mag_pred` (predicted by head) | current `v_0` (observed from `ego_status`) |
| Extra head | per-stage `Linear(D, 1)` | none |
| Extra loss term | magnitude L1 | none |
| Failure mode | magnitude head poorly calibrated on long-tail samples | `v_0` is non-representative (e.g., briefly braked, momentarily stopped) |
| Code surface | larger (head + loss + per-batch anchor scaling) | smaller (per-batch anchor scaling only) |

Velocity-norm wins on simplicity. Endpoint-norm wins when `v_0` poorly predicts the trajectory's actual scale. We don't know which dominates; running both is the cleanest way to find out.

### Reading matrix

- **1a vs baseline** → endpoint-norm decoupling effect (with learned magnitude)
- **1b vs baseline** → velocity-norm decoupling effect (with observed magnitude)
- **1b vs 1a** → which normalization choice wins
- **2 vs baseline** → speed conditioning alone
- **3 vs 1a** → does conditioning add value on top of endpoint-norm decoupling?
- **3 vs 2** → does endpoint-norm decoupling add value on top of conditioning?
- **4 vs speedstrat baseline (L2=0.5548, CR=0.070%)** → does per-bucket reg fix the diagnosed L2 cause?

If 1b wins, the natural follow-up batch is `_shapeanchor_velnorm_egostatus` (test whether ego_status injection still helps even though `v_0` is already in the anchor scaling).

### Recipe

**k-means scripts**:
- `tools/data_converter/gen_kmeans_plan_shape_endptnorm.py`: input `gt_ego_fut_trajs.cumsum(axis=-2)`; histogram `||traj_end||`, pick ε at elbow (default 1.0 if monotonic); cluster `traj / max(||end||, ε)` per cmd; save `(3, 6, 6, 2)` + inspection plot.
- `tools/data_converter/gen_kmeans_plan_shape_velnorm.py`: same input; histogram `||v_0||` from `ego_status[6:8]`; cluster `traj / max(||v_0||, ε_v)` per cmd; save `(3, 6, 6, 2)` + inspection plot. Note: ε_v likely different from endpoint-norm ε since `v_0` distribution differs from `||end||` distribution.

**Magnitude head (1a / 3 only)**: per-decoder-stage `Linear(D, 1)` predicting `mag_pred` from refined `plan_query`. Final trajectory = `shape_anchor * mag_pred`. Loss adds smooth-L1 on `mag_pred` vs `||gt_end||` (smooth-L1 over raw L1 to handle the long magnitude tail; alternatively L1 on `log(mag)`).

**Per-batch anchor scaling (both 1a and 1b)**: `MotionPlanningHead.plan_anchor` becomes per-batch tensor in `forward()` rather than static buffer:
- 1a: `metric_anchor = shape_anchor * reference_magnitude_per_mode` (precomputed from k-means cluster medians) for deformable sampling positions; final trajectory uses learned `mag_pred`.
- 1b: `metric_anchor = shape_anchor * v_0` everywhere — sampling positions and final trajectory both use `v_0`. Simpler.

**`ego_status` injection (#2 / #3)**: `plan_ego_status_encoder: MLP(5 → D)` consuming `ego_status[:, [0, 1, 5, 6, 7]]` (planar accel + yaw rate + planar velocity). Output added to `plan_mode_query` at init (`motion_planning_head.py:1219`) and at every per-stage rebuild (`:1755`). Use GT `data['ego_status']`, not predicted `plan_status`.

### Risks

- Shape clusters for Straight could collapse hard. Inspect cluster centers before committing the K=6/cmd assumption.
- Magnitude head (1a/3) conflates speed with trajectory length; per-step magnitude profile is the fallback if needed.
- Velocity-norm (1b) anchor shapes have units of seconds, not meters — the deformable sampling code needs to multiply by `v_0` consistently before computing reference points. Per-batch anchor refactor is the bulk of the implementation work.
- `ego_status` is in CAN frame, `plan_mode_query` lives in LiDAR frame — let the MLP learn the rotation rather than pre-rotating (augmentation is consistent across both frames).

## Future Work

- **No-rescore eval of `modesspeedstrat` checkpoint** (Killarney 3302034) — single config flip, no retrain. Tells us whether speedstrat's L2 regression is decode-time selection or training-time supervision. Worth running concurrently with the batch above; informs whether #4 is even necessary.
- **`plan_yaw`** (independent, cheap) — `plan_reg ts*2 → ts*3`, supervise yaw from `gt_ego_fut_yaw`. Stabilizes derived endpoint anchor box yaw feeding deformable sampling. Stackable with whatever wins from the parallel batch.
- **Per-step magnitude profile** — only if #1 succeeds but accel/decel patterns are visible in the residual error.
- **Curvature/control-space anchors** (option 5) — research direction if shape/magnitude decoupling underperforms.
- **Fix `augment.py:196` docstring** — comment-level only, but the "LiDAR-aligned ego frame" claim is wrong (it's CAN ego frame).
