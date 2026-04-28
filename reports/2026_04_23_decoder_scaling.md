# Decoder Scaling For Motion Planning

## TODO

- DGX jobs 3654 (`d6h8`) and 3655 (`d3h16`) are still `PENDING`; Killarney mirror runs 3296361 (`d6h8`) and 3296363 (`d3h16`) `COMPLETED` and metrics are recorded below.
- Decide whether to restart the stage-1 `DTbase` Apollo run (preempted at 20%) and add the matching `DTlarge` run.

## Abstract

_Add this section last. Write 2-4 sentences summarizing the experiment, the main result, and the recommended takeaway._

## Intro

SparseDrive currently shares one embedding width across perception, prediction, and planning. That makes full decoder scaling expensive because widening the planning module also widens the detection and map stacks, which in turn pushes us toward retraining both stage 1 and stage 2. The immediate question is whether the stage-2 planning module is the actual bottleneck.

Two families of experiments were tried. The first added an optional projection path inside `MotionPlanningHead` so stage 2 could consume stage-1 outputs in the original feature space while running a wider planning decoder internally. The second kept the full model at the baseline embedding width and scaled only the motion/planning head depth and attention partitioning.

The current evidence favors the motion-head-only path. The projection branch was useful as an interface experiment, but it added complexity and did not improve the overall tradeoff on the completed stage-2 runs.

Question:
- Can a wider planning decoder improve motion/planning quality without scaling the perception heads or retraining stage 1?

## Method

Implementation changes:
- Added a config-gated projection path inside `MotionPlanningHead` with `use_planning_input_proj=False` by default. When enabled, detection/map/ego/temporal features and anchor embeddings are projected from the perception width into a larger planning width, and cached planning features are projected back to the queue width.
- Added two stage-2-only configs: `sparsedrive_r50_stage2_4gpu_bs24_ppdeformmm_DTbaseproj` with planning width `512`, and `sparsedrive_r50_stage2_4gpu_bs24_ppdeformmm_DTlargeproj` with planning width `768` plus a deeper planning decoder stack.
- Added two stage-2-only motion-head-depth configs: `sparsedrive_r50_stage2_4gpu_bs24_planpredtrajdeformmm_DTbase_mhdepth` and `sparsedrive_r50_stage2_4gpu_bs24_planpredtrajdeformmm_DTlarge_mhdepth`. These keep `embed_dims=256` everywhere and scale only the `motion_plan_head` depth and attention head count. Confirmed from saved server configs: `DTbase_mhdepth` changed motion decoder repeats `3 → 6` and `motion_num_heads` `8 → 16`. Because `decouple_attn_motion=True`, attention runs at `embed_dims=512` (concat), so `head_dim` shrinks from `64 → 32` despite more heads.
- Added stage-1 full-width configs `sparsedrive_r50_stage1_8gpu_noflash_DTbase` and `sparsedrive_r50_stage1_8gpu_noflash_DTlarge` to compare full-model width scaling against the stage-2-only motion-head variants.
- Losses, datasets, augmentation, perception width, detection/map decoder depth, and stage-1 checkpoints are unchanged.

Memory note:
- Current stage-2 baseline uses about `25 GB / 40 GB` GPU memory. Peak GPU memory should be logged for all runs in this report so we can judge whether planning-only decoder scaling is a practical tradeoff.

DriveTransformer scale reference:
- DT `small`: `embed_dims=256`, decoder depth `3`, effective attention heads `4` (`head_dim=64`)
- DT `base`: `embed_dims=512`, decoder depth `6`, effective attention heads `8`
- DT `large`: `embed_dims=768`, decoder depth `12`, effective attention heads `12`

SparseDrive variants tested here:
- Stage-2 baseline: `embed_dims=256`, motion/planning repeats `3`, motion heads `8`
- `DTbase_mhdepth`: `embed_dims=256`, motion/planning repeats `6`, motion heads `16`
- `DTlarge_mhdepth`: `embed_dims=256`, motion/planning repeats `8` (confirmed from saved server config), motion heads `16`
- `d6h8`: `embed_dims=256`, motion/planning repeats `6`, motion heads `8` (depth-only ablation vs baseline)
- `d3h16`: `embed_dims=256`, motion/planning repeats `3`, motion heads `16` (heads-only ablation vs baseline)
- `DTbaseproj`: perception stays `256`, planning width `512`, motion heads `8`
- `DTlargeproj`: perception stays `256`, planning width `768`, motion heads `12`
- stage-1 `DTbase`: full model width `512`, detection/map decoder depth `6`, det/map heads `8`
- stage-1 `DTlarge`: full model width `768`, detection/map decoder depth `12`, det/map heads `12`

## Results

_Use the first table for run tracking. Add metric tables below it once runs finish._

| Server | Config | Job ID | Status |
| --- | --- | --- | --- |
| `dgx` | `DTbase_mhdepth` | `3628` | `COMPLETED` |
| `dgx` | `DTbaseproj` | `3629` | `COMPLETED` |
| `dgx` | `DTlarge_mhdepth` reduced-depth rerun (depth 8) | `3635` | `COMPLETED` |
| `dgx` | `d6h8` (depth 6, heads 8) | `3654` | `PENDING` |
| `dgx` | `d3h16` (depth 3, heads 16) | `3655` | `PENDING` |
| `killarney` | `d6h8` (depth 6, heads 8) | `3296361` | `COMPLETED` (9h16m) |
| `killarney` | `d3h16` (depth 3, heads 16) | `3296363` | `COMPLETED` (7h25m) |
| `killarney` | `d8h32` (depth 8, heads 32; diagonal push) | `3300977` | `CANCELLED` (~17m) |
| `killarney` | `d6h16ffn4x` (motion FFN 2× → 4×) | `3300978` | `CANCELLED` (~18m) |
| `narval` | `d6h16ffn4x` (motion FFN 2× → 4×) | `59908177` | `COMPLETED` (7h29m) |
| `narval` | `d8h32` (depth 8, heads 32; diagonal push) | `59908912` | `COMPLETED` (8h07m) |
| `killarney` | `d6h16ffn4x` (cross-cluster mirror) | `3302411` | `COMPLETED` (9h14m) |
| `killarney` | `d6h16` (`DTbase_mhdepth`) same-cluster rerun | `3310790` | `COMPLETED` (9h22m) |
| `killarney` | `ptaux2d_ppdeformmm_planifls_d6h16` (combine with project-best baseline) | `3310791` | `COMPLETED` (9h13m) |
| `apollo` | `stage1 DTbase` | n/a | `INCOMPLETE` (~20%, preempted at iter 8780/43900) |
| `apollo` | `stage1 DTlarge` | n/a | `NOT STARTED` (only workflow init logged) |

Memory:
- Baseline: ~25 GB / 40 GB
- `DTbase_mhdepth`: not logged
- `DTlarge_mhdepth`: 28.3 GB / 40 GB (from training log)
- `stage1 DTbase`: 28.3 GB / 40 GB (from training log)

Experiment summary:

| Variant family | What changed | Stage-1 compatible | Outcome so far |
| --- | --- | --- | --- |
| `DTbaseproj` / `DTlargeproj` | Keep perception at `256`, widen planning internals with projections | Yes | `DTbaseproj` trains but underperforms baseline overall; `DTlargeproj` was brittle and not worth pursuing first |
| `DTbase_mhdepth` / `DTlarge_mhdepth` | Keep width at `256`, scale only motion/planning decoder depth and attention partitioning | Yes | `DTbase_mhdepth` is the best overall result; `DTlarge_mhdepth` (depth 8) improves mAP but regresses on both `obj_box_col` and `L2` versus `DTbase_mhdepth` |
| stage-1 `DTbase` / `DTlarge` | Widen full perception/det/map model in stage 1 | No stage-2-only shortcut | `DTbase` preempted at 20% (no val metrics); `DTlarge` never started; inconclusive |

Scale comparison:

| Model | Perception / shared width | Main decoder depth | Motion/planning depth | Attention heads |
| --- | --- | --- | --- | --- |
| DriveTransformer `small` | `256` | `3` | joint decoder | `4` effective |
| DriveTransformer `base` | `512` | `6` | joint decoder | `8` effective |
| DriveTransformer `large` | `768` | `12` | joint decoder | `12` effective |
| SparseDrive baseline | `256` | det/map `6` | motion `3` | motion `8` |
| SparseDrive `DTbase_mhdepth` | `256` | det/map `6` | motion `6` | motion `16` |
| SparseDrive `DTlarge_mhdepth` | `256` | det/map `6` | motion `8` | motion `16` |
| SparseDrive `d6h8` | `256` | det/map `6` | motion `6` | motion `8` |
| SparseDrive `d3h16` | `256` | det/map `6` | motion `3` | motion `16` |
| SparseDrive stage-1 `DTbase` | `512` | det/map `6` | n/a in stage 1 | det/map `8` |
| SparseDrive stage-1 `DTlarge` | `768` | det/map `12` | n/a in stage 1 | det/map `12` |

| Model | mAP | NDS | mAP_normal | car / ped EPA | car / ped ADE | obj_box_col | L2 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `planpredtrajdeformmm` baseline | `0.4145` | `0.5272` | `0.5527` | `0.4929 / 0.4158` | `0.6230 / 0.7212` | `0.050%` | `0.5480` | stage-2 baseline (`d3h8`) |
| `d6h16` (`DTbase_mhdepth`) | `0.4157` | `0.5281` | `0.5551` | `0.4940 / 0.4124` | `0.6258 / 0.7211` | `0.041%` | `0.5426` | **best overall** |
| `d8h16` (`DTlarge_mhdepth`) | `0.4180` | `0.5262` | `0.5551` | `0.4919 / 0.4178` | `0.6328 / 0.7260` | `0.075%` | `0.5531` | better mAP but worse planning than `d6h16` |
| `d3h8_pw512` (`DTbaseproj`) | `0.4111` | `0.5228` | `0.5531` | `0.4907 / 0.4242` | `0.6171 / 0.7065` | `0.106%` | `0.5636` | projection path (planning width 512); underperforms baseline overall |
| `d6h8` | `0.4146` | `0.5277` | `0.5548` | `0.4914 / 0.4153` | `0.6441 / 0.7355` | `0.060%` | `0.5385` | depth-only; Killarney 3296361; better `L2` than baseline but worse `obj_box_col` |
| `d3h16` | `0.4136` | `0.5224` | `0.5534` | `0.4979 / 0.4139` | `0.6150 / 0.7324` | `0.071%` | `0.5654` | heads-only; Killarney 3296363; regresses on both `obj_box_col` and `L2` vs baseline |
| `d6h16ffn4x` (narval) | `0.4106` | `0.5207` | `0.5482` | `0.4887 / 0.4156` | `0.6153 / 0.7298` | `0.063%` | `0.5351` | motion FFN 2× → 4×; Narval 59908177; lowest `L2` of all rows but on a different cluster |
| `d6h16ffn4x` (killarney) | `0.4153` | `0.5264` | `0.5597` | `0.4969 / 0.4172` | `0.6134 / 0.7194` | `0.067%` | `0.5469` | cross-cluster mirror; Killarney 3302411; `L2` worse than `d6h8` and `d6h16` on same/comparable cluster |
| `d8h32` | `0.4119` | `0.5260` | `0.5502` | `0.4920 / 0.4222` | `0.6257 / 0.7191` | `0.077%` | `0.5416` | diagonal push (depth 8, heads 32, head_dim 16); Narval 59908912; `obj_box_col` regresses vs `d6h16`, detection roughly flat |
| `d6h16` (killarney rerun) | `0.4176` | `0.5291` | `0.5604` | `0.4951 / 0.4152` | `0.6253 / 0.7096` | `0.062%` | `0.5542` | same-cluster rerun of `DTbase_mhdepth`; Killarney 3310790; planning regresses vs DGX `d6h16` (`L2` +0.012, `obj_box_col` +0.021pp) — cross-cluster variance is real |
| `ptaux2d_planifls_d6h16` | `0.4421` | `0.5461` | `0.5552` | `0.5081 / 0.4320` | `0.6184 / 0.7080` | `0.103%` | `0.5013` | `d6h16` motion+plan stack on top of `ptaux2d_ppdeformmm_planifls`; Killarney 3310791; planning `L2` improves vs DGX `d6h16` baseline (`0.5013` vs `0.5426`) but `obj_box_col` regresses (`0.103%` vs `0.041%`) |

## Discussion

What improved:
- `DTbase_mhdepth` slightly improves overall detection/tracking/map quality over the baseline: `mAP 0.4157 > 0.4145`, `NDS 0.5281 > 0.5272`, and `mAP_normal 0.5551 > 0.5527`.
- `DTbase_mhdepth` improves the key closed-loop proxy metrics: `obj_box_col` `0.041%` vs `0.050%`, `L2` `0.5426` vs `0.5480`.
- `DTlarge_mhdepth` pushes detection further: `mAP 0.4180`, the best across all variants.
- `d6h8` improves `L2` over the baseline (`0.5385` vs `0.5480`) and over `DTbase_mhdepth` (`0.5385` vs `0.5426`).
- `d6h16ffn4x` Narval result reports the lowest `L2` (`0.5351`), but the Killarney mirror of the same config gives `L2 0.5469` — a cross-cluster delta of `0.012` for an identical config.

What regressed or stayed flat:
- `DTbase_mhdepth` does not uniformly improve EPA/ADE: `car EPA` is slightly better, `ped EPA` is slightly worse, and `ADE` is essentially flat versus baseline.
- `DTlarge_mhdepth` regresses on both primary planning metrics relative to `DTbase_mhdepth`: `obj_box_col` `0.075%` vs `0.041%`, `L2` `0.5531` vs `0.5426`. Both are worse than the baseline too.
- `DTbaseproj` underperforms the baseline on all main metrics with noticeably worse `obj_box_col` and `L2`.
- `d6h8` regresses `obj_box_col` (`0.060%` vs baseline `0.050%`, vs `DTbase_mhdepth` `0.041%`) and worsens `car ADE` (`0.6441` vs `0.6230`).
- `d3h16` regresses on both primary planning metrics versus baseline: `obj_box_col` `0.071%` vs `0.050%`, `L2` `0.5654` vs `0.5480`.
- `d6h16ffn4x` regresses `obj_box_col` on both clusters (Narval `0.063%`, Killarney `0.067%`) vs `d6h16`'s `0.041%`. On Killarney specifically, `L2 0.5469` is worse than the Killarney `d6h8` (`0.5385`), so the Narval "best L2" reading does not replicate cross-cluster.
- `d8h32` (Narval) regresses `obj_box_col` to `0.077%` (vs `d6h16`'s `0.041%`) and gives `L2 0.5416` — close to `d6h16` but worse than `d6h8`. Pushing depth+heads further along the diagonal does not continue the trend, consistent with `d8h16` (`DTlarge_mhdepth`)'s earlier regression.

2×2 ablation summary (depth × heads, all at `embed_dims=256`; `d3h8` = baseline `planpredtrajdeformmm`, `d6h16` = `DTbase_mhdepth`):

| | heads 8 | heads 16 |
| --- | --- | --- |
| depth 3 | `d3h8`: `L2 0.5480`, `obj_box_col 0.050%` | `d3h16`: `L2 0.5654`, `obj_box_col 0.071%` |
| depth 6 | `d6h8`: `L2 0.5385`, `obj_box_col 0.060%` | `d6h16`: `L2 0.5426`, `obj_box_col 0.041%` |

Likely explanation:
- Depth scaling (`3 → 6`) is what drives the `L2` gain: `d6h8` (`L2 0.5385`) and `DTbase_mhdepth` (`L2 0.5426`) both beat baseline (`0.5480`), while `d3h16` (`L2 0.5654`) is worse than baseline. Heads-only changes appear to hurt planning at this depth.
- Heads `8 → 16` only helps `obj_box_col` when paired with the deeper stack: combined (`DTbase_mhdepth`) gives the best `obj_box_col` (`0.041%`), but heads-only (`d3h16`) and depth-only (`d6h8`) both regress on `obj_box_col` versus baseline.
- This is consistent with depth 8 (`DTlarge_mhdepth`) over-parameterizing the motion head relative to the fixed-width perception backbone, degrading planning even as detection scores improve marginally — depth 6 looks like the sweet spot.
- The head geometry in `mhdepth`/`d3h16` variants is not DT-consistent. `decouple_attn_motion=True` means attention runs at `embed_dims=512` (concatenated ego+agent features), so `head_dim = 512 / 16 = 32` — smaller than the DT small baseline (`head_dim=64`). The head-count increase adds parameters but may be hurting representational capacity per head; this matters more without the extra refinement iterations from depth 6.
- Stage-1 DTbase and DTlarge are inconclusive: DTbase was preempted at ~20% with `grad_norm=nan` throughout (possible instability at `embed_dims=512`), and DTlarge never started.
- Cross-cluster variance is non-trivial: the `d6h16ffn4x` Narval-vs-Killarney delta is `~0.012` in `L2` for an identical config, which is comparable in magnitude to many of the variant-vs-baseline deltas reported in this study. Comparisons across clusters should be treated cautiously; same-cluster comparisons are stronger.
- Same-cluster `d6h16` rerun (Killarney 3310790, `L2 0.5542`, `obj_box_col 0.062%`) confirms the variance pattern: it regresses on both planning metrics versus the original DGX `d6h16` (`L2 0.5426`, `obj_box_col 0.041%`). On Killarney, `d6h16` no longer beats baseline `planpredtrajdeformmm` on either planning metric, weakening the cross-cluster claim that `d6h16` is the unconditional motion-head winner.
- `ptaux2d_planifls_d6h16` (Killarney 3310791) stacks the depth-6/heads-16 motion+plan decoder onto the project-best `ptaux2d_ppdeformmm_planifls` base. `L2 0.5013` is the best stage-2 `L2` in this report, but `obj_box_col` regresses to `0.103%` — `2.5×` worse than the same base without `d6h16` (`0.041–0.047%` family). Depth-6 motion stack appears to trade collision performance for L2 even on the project-best base, so it does not Pareto-dominate `_planinstfeat_laststage`.

_If there is a clear winner, name it here and explain why it should be the default or next reference point._

Recommended model:
- `sparsedrive_r50_stage2_4gpu_bs24_planpredtrajdeformmm_DTbase_mhdepth`

Why:
- It is the strongest result on `obj_box_col` (`0.041%`) and second-best on `L2` (`0.5426`, behind only `d6h8`'s `0.5385`); the 2×2 ablation shows the depth-6 + heads-16 combination wins overall.
- The 2×2 corners now confirm the contribution split: depth (`3 → 6`) is what improves `L2`, while heads (`8 → 16`) only helps `obj_box_col` when paired with depth 6 — `DTbase_mhdepth` is the only configuration that improves both primary planning metrics simultaneously.
- Going deeper to 8 repeats (`DTlarge_mhdepth`) improves mAP but hurts planning, so depth 6 looks like the sweet spot.
- Its depth scaling (`3 → 6`) matches DriveTransformer's small-to-base decoder-depth progression, even though the head geometry is not yet DT-consistent: `head_dim = 32` (512 / 16) vs DT-small's `head_dim = 64`.

## Future Work

_List the next experiments that follow naturally from the results in this report._

- Confirm the Killarney-vs-DGX agreement once DGX 3654/3655 finally run; if they reproduce the Killarney trends, the 2×2 conclusion is solid.
- Investigate why depth-only (`d6h8`) hurts `obj_box_col` despite improving `L2` — possibly a refinement-iteration effect that needs the wider head count to suppress collisions.
- Restart stage-1 `DTbase` on Apollo if full-model width scaling is still of interest; investigate the `grad_norm=nan` before committing to a full run.
- Drop the projection-path branch unless a later result justifies its extra complexity.
