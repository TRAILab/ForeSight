# Occlusion Detection Experiments (Feb–Mar 2026)

## Intro

This document covers experiments evaluating SparseDrive's ability to detect occluded objects — agents that are partially or fully hidden from all cameras. The motivation is that undetected occluded agents represent invisible collision risks for the planning head. Several evaluation and training variants were tested: fully-hidden object evaluation (`occfh`), partially-occluded evaluation (`occp`), and variants that add occluded-object supervision to the training pipeline. Experiments ran between 2026-02-25 and 2026-03-20.

## Method

Three occlusion categories were defined and evaluated separately:
- **occfh** (fully hidden): objects not visible in any camera at the current frame
- **occfl** (fully or largely occluded): objects occluded beyond a visibility threshold
- **occp** (partially occluded): objects with some portion visible in at least one camera

**Evaluation-only variants** (standard checkpoint evaluated on occluded subsets):
- `stage2_4gpu_bs24_occfheval`, `occfleval`, `occpeval` — standard bs24 checkpoint, evaluated with custom occluded-object metrics

**Training+evaluation variants** (trained with occluded supervision):
- `stage2_4gpu_bs24_occfhtraineval`, `occfltraineval`, `occptraineval` — stage2 re-trained with explicit occluded object targets
- `stage1_8gpu_noflash_occptraineval` — stage1 re-trained with partial occlusion supervision

All compared against the standard bs24 baseline for detection performance on the full val set.

## Results

| Variant | occluded_det mAP | Notes |
|---------|-----------------|-------|
| bs24_occfheval (eval only) | ~0.000 | Fully hidden: essentially undetectable |
| bs24_occfleval (eval only) | ~0.000 | Fully/largely occluded: near-zero |
| bs24_occpeval (eval only) | marginal | Partial occlusion: minor improvement over zero |
| bs24_occfhtraineval | ~0.000 | Training supervision doesn't help for fully hidden |
| bs24_occptraineval | marginal | Partial occlusion training shows small gain |
| stage1_occptraineval | marginal | Stage1 occlusion supervision: marginal only |

Work dirs: `sparsedrive_r50_stage2_4gpu_bs24_occfheval` (2026-02-25), `sparsedrive_r50_stage2_4gpu_bs24_occpeval` (2026-02-26), `sparsedrive_r50_stage2_4gpu_bs24_occptraineval` (2026-02-26), `sparsedrive_r50_stage2_4gpu_bs24_occfhtraineval` (area 2026-03), `sparsedrive_r50_stage2_4gpu_bs24_occfleval` (2026-03-05), `sparsedrive_r50_stage1_8gpu_noflash_occptraineval` (2026-03-20).

## Discussion

- **Fully-hidden (occfh) objects are essentially undetectable by current SparseDrive**: The model has no mechanism to detect objects it has never seen in any camera. Camera-only systems are fundamentally limited here — without a sensor that penetrates occlusion (LIDAR, radar, or infrastructure sensors), fully-hidden agents cannot be detected from image features alone.
- **Partially-occluded objects (occp) show only marginal improvement**: Even with explicit partial-occlusion supervision during training, the model struggles to reliably detect partially-hidden agents beyond what it already detects from visible portions. The deformable cross-attention mechanism samples image features at predicted 2D projections — if the object is mostly occluded, there are few useful features at those locations.
- **Evaluation implementation may have issues**: The occluded detection metrics show inconsistencies between training supervision and evaluation, suggesting the occluded GT matching logic in the evaluation pipeline may not align with the training targets. This warrants investigation before drawing strong conclusions.
- **Temporal cache is insufficient for full occlusion**: The InstanceBank maintains a 600-instance temporal cache, but its entries decay exponentially (`confidence_decay=0.6`) and are not explicitly designed to track objects that disappear from all cameras. An object that goes behind a building will have its instance confidence decay to zero within 2-3 frames.
- **Principled architecture changes required**: Detecting occluded objects likely requires explicit memory/prediction mechanisms: (a) occupancy-based prediction that extrapolates where occluded agents should be, (b) uncertainty-aware detection that propagates occluded agent hypotheses through the temporal cache, or (c) infrastructure-level sensor fusion.

## Future Work

- Verify the occlusion evaluation metric implementation — ensure that GT occluded object matching uses the same visibility threshold convention in both training and evaluation.
- Test whether longer temporal cache (`num_temp_instances` increased from 600) improves partially-occluded detection by retaining more instance history.
- Explore an explicit occlusion memory: when a tracked instance's score drops due to occlusion (not because the object left the scene), maintain a "ghost" hypothesis in the instance bank with extrapolated position from last known velocity.
- Consider separating the occlusion detection problem from the standard detection pipeline — it may need dedicated architectural treatment (e.g., a separate occluded-agent head with extrapolation-based supervision).
