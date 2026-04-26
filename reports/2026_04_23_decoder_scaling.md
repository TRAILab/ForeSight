# Decoder Scaling For Motion Planning

## TODO

- Record results for `d6h8` (job 3654) and `d3h16` (job 3655) once complete and update the metrics table.
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
| `dgx` | `d6h8` (depth 6, heads 8) | `3654` | `RUNNING` |
| `dgx` | `d3h16` (depth 3, heads 16) | `3655` | `RUNNING` |
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
| `planpredtrajdeformmm` baseline | `0.4145` | `0.5272` | `0.5527` | `0.4929 / 0.4158` | `0.6230 / 0.7212` | `0.050%` | `0.5480` | stage-2 baseline |
| `DTbase_mhdepth` | `0.4157` | `0.5281` | `0.5551` | `0.4940 / 0.4124` | `0.6258 / 0.7211` | `0.041%` | `0.5426` | **best overall**; depth 6, heads 16 |
| `DTlarge_mhdepth` | `0.4180` | `0.5262` | `0.5551` | `0.4919 / 0.4178` | `0.6328 / 0.7260` | `0.075%` | `0.5531` | depth 8, heads 16; better mAP but worse planning than DTbase |
| `DTbaseproj` | `0.4111` | `0.5228` | `0.5531` | `0.4907 / 0.4242` | `0.6171 / 0.7065` | `0.106%` | `0.5636` | projection path underperforms baseline overall |

## Discussion

What improved:
- `DTbase_mhdepth` slightly improves overall detection/tracking/map quality over the baseline: `mAP 0.4157 > 0.4145`, `NDS 0.5281 > 0.5272`, and `mAP_normal 0.5551 > 0.5527`.
- `DTbase_mhdepth` improves the key closed-loop proxy metrics: `obj_box_col` `0.041%` vs `0.050%`, `L2` `0.5426` vs `0.5480`.
- `DTlarge_mhdepth` pushes detection further: `mAP 0.4180`, the best across all variants.

What regressed or stayed flat:
- `DTbase_mhdepth` does not uniformly improve EPA/ADE: `car EPA` is slightly better, `ped EPA` is slightly worse, and `ADE` is essentially flat versus baseline.
- `DTlarge_mhdepth` regresses on both primary planning metrics relative to `DTbase_mhdepth`: `obj_box_col` `0.075%` vs `0.041%`, `L2` `0.5531` vs `0.5426`. Both are worse than the baseline too.
- `DTbaseproj` underperforms the baseline on all main metrics with noticeably worse `obj_box_col` and `L2`.

Likely explanation:
- Depth 6 (`DTbase_mhdepth`) is the right point for planning capacity at `embed_dims=256`: it gives the planner more refinement iterations without overfitting or capacity imbalance. Depth 8 (`DTlarge_mhdepth`) appears to over-parameterize the motion head relative to the fixed-width perception backbone, degrading planning even as detection scores improve marginally.
- The head geometry in both `mhdepth` variants is not DT-consistent. `decouple_attn_motion=True` means attention runs at `embed_dims=512` (concatenated ego+agent features), so `head_dim = 512 / 16 = 32` — smaller than the DT small baseline (`head_dim=64`). The head-count increase adds parameters but may be hurting representational capacity per head.
- Stage-1 DTbase and DTlarge are inconclusive: DTbase was preempted at ~20% with `grad_norm=nan` throughout (possible instability at `embed_dims=512`), and DTlarge never started.

_If there is a clear winner, name it here and explain why it should be the default or next reference point._

Recommended model:
- `sparsedrive_r50_stage2_4gpu_bs24_planpredtrajdeformmm_DTbase_mhdepth`

Why:
- It is the strongest result on the primary planning metrics (`obj_box_col`, `L2`) and the cleanest implementation path. Going deeper to 8 repeats (`DTlarge_mhdepth`) improves mAP but hurts planning.
- Its depth scaling (`3 → 6`) matches DriveTransformer's small-to-base decoder-depth progression, even though the head geometry is not yet DT-consistent: `head_dim = 32` (512 / 16) vs DT-small's `head_dim = 64`.

## Future Work

_List the next experiments that follow naturally from the results in this report._

- Wait for `d6h8` and `d3h16` results to disentangle depth vs head-count contributions — these are the two missing corners of the 2×2 {depth 3,6} × {heads 8,16} ablation grid.
- Restart stage-1 `DTbase` on Apollo if full-model width scaling is still of interest; investigate the `grad_norm=nan` before committing to a full run.
- Drop the projection-path branch unless a later result justifies its extra complexity.
