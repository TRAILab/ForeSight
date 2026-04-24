# Decoder Scaling For Motion Planning

## TODO

- Record training stability, memory, and validation planning metrics for the active `DTlarge_mhdepth` run.
- Compare motion-head-only scaling against the stage-2 baseline and the projection variants.
- Decide whether the projection-path branch should be kept for further study or dropped in favor of motion-head-only scaling.
- If stage-1 scaling remains interesting, create corrected `DTbase` / `DTlarge` configs with widened detection anchor encoders from the start.

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
- Added two stage-2-only motion-head-depth configs: `sparsedrive_r50_stage2_4gpu_bs24_planpredtrajdeformmm_DTbase_mhdepth` and `sparsedrive_r50_stage2_4gpu_bs24_planpredtrajdeformmm_DTlarge_mhdepth`. These keep `embed_dims=256` everywhere and scale only the `motion_plan_head` depth and attention head count.
- Added stage-1 full-width configs `sparsedrive_r50_stage1_8gpu_noflash_DTbase` and `sparsedrive_r50_stage1_8gpu_noflash_DTlarge`. The first direct copies were invalid because the detection anchor encoder still output `256`; the corrected `DTbase` config makes the detection anchor encoder output match the widened model width.
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
- `DTlarge_mhdepth`: `embed_dims=256`, motion/planning repeats `8` or `12` depending on run, motion heads `16`
- `DTbaseproj`: perception stays `256`, planning width `512`
- `DTlargeproj`: perception stays `256`, planning width `768`
- stage-1 `DTbase`: full model width `512`, detection/map decoder depth `6`
- stage-1 `DTlarge`: full model width `768`, detection/map decoder depth `12`

## Results

_Use the first table for run tracking. Add metric tables below it once runs finish._

| Server | Config | Job ID | Status |
| --- | --- | --- | --- |
| `dgx` | `DTbase_mhdepth` | `3628` | `COMPLETED` |
| `dgx` | `DTbaseproj` | `3629` | `COMPLETED` |
| `dgx` | `DTlarge_mhdepth` reduced-depth rerun | `3635` | `RUNNING` |
| `apollo` | `stage1 DTbase` corrected anchor-encoder rerun | n/a | `RUNNING` |

_Add one or more tables here for the metrics that matter for the experiment. Keep the baseline and best variant easy to compare._

Memory to log:
- Peak GPU memory per run

Experiment summary:

| Variant family | What changed | Stage-1 compatible | Outcome so far |
| --- | --- | --- | --- |
| `DTbaseproj` / `DTlargeproj` | Keep perception at `256`, widen planning internals with projections | Yes | `DTbaseproj` trains but underperforms baseline overall; `DTlargeproj` was brittle and not worth pursuing first |
| `DTbase_mhdepth` / `DTlarge_mhdepth` | Keep width at `256`, scale only motion/planning decoder depth and attention partitioning | Yes | `DTbase_mhdepth` is the best result so far; `DTlarge_mhdepth` is still being tuned for memory/stability |
| stage-1 `DTbase` / `DTlarge` | Widen full perception/det/map model in stage 1 | No stage-2-only shortcut | First direct copies were invalid because the detection anchor encoder stayed at 256; corrected `DTbase` is now running on Apollo |

Scale comparison:

| Model | Perception / shared width | Main decoder depth | Motion/planning depth | Attention heads |
| --- | --- | --- | --- | --- |
| DriveTransformer `small` | `256` | `3` | joint decoder | `4` effective |
| DriveTransformer `base` | `512` | `6` | joint decoder | `8` effective |
| DriveTransformer `large` | `768` | `12` | joint decoder | `12` effective |
| SparseDrive baseline | `256` | det/map `6` | motion `3` | motion `8` |
| SparseDrive `DTbase_mhdepth` | `256` | det/map `6` | motion `6` | motion `16` |
| SparseDrive `DTlarge_mhdepth` | `256` | det/map `6` | motion `8` or `12` by run | motion `16` |
| SparseDrive stage-1 `DTbase` | `512` | det/map `6` | n/a in stage 1 | det/map `8` |
| SparseDrive stage-1 `DTlarge` | `768` | det/map `12` | n/a in stage 1 | det/map `12` |

| Model | mAP | NDS | mAP_normal | car / ped EPA | car / ped ADE | obj_box_col | L2 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `planpredtrajdeformmm` baseline | `0.4145` | `0.5272` | `0.5527` | `0.4929 / 0.4158` | `0.6230 / 0.7212` | `0.050%` | `0.5480` | stage-2 baseline |
| `DTbase_mhdepth` | `0.4157` | `0.5281` | `0.5551` | `0.4940 / 0.4124` | `0.6258 / 0.7211` | `0.041%` | `0.5426` | best overall so far |
| `DTbaseproj` | `0.4111` | `0.5228` | `0.5531` | `0.4907 / 0.4242` | `0.6171 / 0.7065` | `0.106%` | `0.5636` | projection path underperforms baseline overall |

## Discussion

_Summarize what changed and why it likely changed. Focus on actual evidence, not speculation alone._

[Short interpretation of the results.]

What improved:
- `DTbase_mhdepth` slightly improves overall detection/tracking/map quality over the baseline: `mAP 0.4157 > 0.4145`, `NDS 0.5281 > 0.5272`, and `mAP_normal 0.5551 > 0.5527`.
- `DTbase_mhdepth` also improves closed-loop proxy metrics with lower `obj_box_col` (`0.041%` vs `0.050%`) and lower `L2` (`0.5426` vs `0.5480`).

What regressed or stayed flat:
- `DTbase_mhdepth` does not uniformly improve planning metrics: `car EPA` is slightly better, `ped EPA` is slightly worse, and `ADE` is essentially flat versus baseline.
- `DTbaseproj` underperforms the baseline on the main aggregate metrics and has noticeably worse `obj_box_col` and `L2`.
- The stage-1 `DTbase` / `DTlarge` direct-copy configs were not valid widening experiments at first, because the detection anchor encoder stayed at `256` while the shared model width was increased.

Likely explanation:
- Scaling only the motion/planning decoder depth is a cleaner way to add stage-2 capacity than introducing a wider projected planning space. The motion-head-only variant preserves the baseline feature interface while giving the planner more iterative refinement depth.
- The current `mhdepth` comparison is depth-aligned with DriveTransformer, but not head-geometry-aligned. At `embed_dims=256`, a DT-style setting would use `4` heads, while the current SparseDrive motion-head experiments use `16`.

_If there is a clear winner, name it here and explain why it should be the default or next reference point._

Recommended model:
- `sparsedrive_r50_stage2_4gpu_bs24_planpredtrajdeformmm_DTbase_mhdepth`

Why:
- It is the strongest result so far on the main aggregate metrics and the cleanest implementation path. The projection variant adds complexity without improving the overall tradeoff.
- Its depth scaling (`3 -> 6`) matches DriveTransformer's small-to-base decoder-depth progression, even though the head geometry is not yet DT-consistent because SparseDrive width stayed fixed at `256`.

## Future Work

_List the next experiments that follow naturally from the results in this report._

- Finish the active `DTlarge_mhdepth` run and compare it directly against `DTbase_mhdepth`.
- Try a DT-style head-count ablation for motion-head-only scaling at `embed_dims=256`, i.e. use `4` heads instead of `16`.
- If stage-1 width scaling is retried, fix both `DTbase` and `DTlarge` configs by making the detection anchor encoder output match the widened `embed_dims`.
- Drop the projection-path branch unless a later result justifies its extra complexity.
