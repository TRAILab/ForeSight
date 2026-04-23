# Decoder Scaling For Motion Planning

## TODO

- Record training stability, memory, and validation planning metrics for the `DTbase_mhdepth` and `DTlarge_mhdepth` runs.
- Compare motion-head-only scaling against the stage-2 baseline and the projection variants.
- Decide whether the projection-path branch should be kept for further study or dropped in favor of motion-head-only scaling.

## Abstract

_Add this section last. Write 2-4 sentences summarizing the experiment, the main result, and the recommended takeaway._

## Intro

SparseDrive currently shares one embedding width across perception, prediction, and planning. That makes full decoder scaling expensive because widening the planning module also widens the detection and map stacks, which in turn pushes us toward retraining both stage 1 and stage 2. The immediate question is whether the stage-2 planning module is the actual bottleneck.

This experiment isolates planning-side decoder scaling by keeping the perception stack at the baseline width and adding an optional projection path inside `MotionPlanningHead`. That lets stage 2 consume stage-1 outputs in the original feature space while running a wider planning decoder internally.

After the first projection-path implementation proved brittle under distributed training, a second cleaner variant was added that keeps the full model at the baseline embedding width and scales only the motion/planning head depth and attention partitioning. That variant should preserve stage-1 checkpoint compatibility without adding a new width-conversion boundary.

Question:
- Can a wider planning decoder improve motion/planning quality without scaling the perception heads or retraining stage 1?

## Method

Implementation changes:
- Added a config-gated projection path inside `MotionPlanningHead` with `use_planning_input_proj=False` by default. When enabled, detection/map/ego/temporal features and anchor embeddings are projected from the perception width into a larger planning width, and cached planning features are projected back to the queue width.
- Added two stage-2-only configs: `sparsedrive_r50_stage2_4gpu_bs24_ppdeformmm_DTbaseproj` with planning width `512`, and `sparsedrive_r50_stage2_4gpu_bs24_ppdeformmm_DTlargeproj` with planning width `768` plus a deeper planning decoder stack.
- Added two stage-2-only motion-head-depth configs: `sparsedrive_r50_stage2_4gpu_bs24_planpredtrajdeformmm_DTbase_mhdepth` and `sparsedrive_r50_stage2_4gpu_bs24_planpredtrajdeformmm_DTlarge_mhdepth`. These keep `embed_dims=256` everywhere and scale only the `motion_plan_head` depth and attention head count.
- Losses, datasets, augmentation, perception width, detection/map decoder depth, and stage-1 checkpoints are unchanged.

Memory note:
- Current stage-2 baseline uses about `25 GB / 40 GB` GPU memory. Peak GPU memory should be logged for all runs in this report so we can judge whether planning-only decoder scaling is a practical tradeoff.

## Results

_Use the first table for run tracking. Add metric tables below it once runs finish._

| Server | Config | Job ID | Status |
| --- | --- | --- | --- |
| `dgx` | `DTlargeproj` | `3619` | `FAILED` |
| `dgx` | `DTbaseproj` | `3620` | `FAILED` |
| `dgx` | `DTbaseproj` retry | `3621` | `FAILED` |
| `dgx` | `DTlargeproj` retry | `3622` | `FAILED` |
| `dgx` | `DTbaseproj` adapter retry | `3623` | `CANCELLED` |
| `dgx` | `DTlargeproj` adapter retry | `3624` | `PENDING / superseded` |
| `dgx` | `DTbase_mhdepth` | `[pending submission]` | `[QUEUED]` |

_Add one or more tables here for the metrics that matter for the experiment. Keep the baseline and best variant easy to compare._

Memory to log:
- Peak GPU memory per run

| Model | Metric 1 | Metric 2 | Metric 3 | Notes |
| --- | --- | --- | --- | --- |
| `[baseline]` | `[x]` | `[y]` | `[z]` | `[short note]` |
| `[variant]` | `[x]` | `[y]` | `[z]` | `[short note]` |

## Discussion

_Summarize what changed and why it likely changed. Focus on actual evidence, not speculation alone._

[Short interpretation of the results.]

What improved:
- `[metric / behavior]`

What regressed or stayed flat:
- `[metric / behavior]`

Likely explanation:
- `[short explanation tied to the method]`

_If there is a clear winner, name it here and explain why it should be the default or next reference point._

Recommended model:
- `[config name]`

Why:
- `[best tradeoff or strongest result]`

## Future Work

_List the next experiments that follow naturally from the results in this report._

- `[next ablation or cleanup]`
- `[follow-up experiments]`
- `[dataset / metric / evaluation extension]`
