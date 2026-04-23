# Decoder Scaling For Motion Planning

## TODO

- Launch local stage-2 runs with 1gpu_bs2 variants for `DTbaseproj` and `DTlargeproj`.
- Launch stage-2 runs for `DTbaseproj` then `DTlargeproj` on dgx with 4gpu_bs24 variants.
- Record training stability, memory, and validation planning metrics.
- Compare against the stage-2 baseline and full-model scaling variants.

## Abstract

_Add this section last. Write 2-4 sentences summarizing the experiment, the main result, and the recommended takeaway._

## Intro

SparseDrive currently shares one embedding width across perception, prediction, and planning. That makes full decoder scaling expensive because widening the planning module also widens the detection and map stacks, which in turn pushes us toward retraining both stage 1 and stage 2. The immediate question is whether the stage-2 planning module is the actual bottleneck.

This experiment isolates planning-side decoder scaling by keeping the perception stack at the baseline width and adding an optional projection path inside `MotionPlanningHead`. That lets stage 2 consume stage-1 outputs in the original feature space while running a wider planning decoder internally.

Question:
- Can a wider planning decoder improve motion/planning quality without scaling the perception heads or retraining stage 1?

## Method

Implementation changes:
- Added a config-gated projection path inside `MotionPlanningHead` with `use_planning_input_proj=False` by default. When enabled, detection/map/ego/temporal features and anchor embeddings are projected from the perception width into a larger planning width, and cached planning features are projected back to the queue width.
- Added two stage-2-only configs: `sparsedrive_r50_stage2_4gpu_bs24_ppdeformmm_DTbaseproj` with planning width `512`, and `sparsedrive_r50_stage2_4gpu_bs24_ppdeformmm_DTlargeproj` with planning width `768` plus a deeper planning decoder stack.
- Losses, datasets, augmentation, perception width, detection/map decoder depth, and stage-1 checkpoints are unchanged.

Memory note:
- Current stage-2 baseline uses about `25 GB / 40 GB` GPU memory. Peak GPU memory should be logged for all runs in this report so we can judge whether planning-only decoder scaling is a practical tradeoff.

## Results

_Use the first table for run tracking. Add metric tables below it once runs finish._

| Server | Config | Job ID | Status |
| --- | --- | --- | --- |
| `[server]` | `[short config name]` | `[job id]` | `[RUNNING / COMPLETED / FAILED / CANCELLED]` |

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
