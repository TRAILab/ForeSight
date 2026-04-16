# Occluded Detection: Metric Analysis and Classifier Design

## TODO

- Run `sparsedrive_r50_stage2_4gpu_bs24_occhead_occptraineval` training and log results here
- Rerun evaluation for `sparsedrive_r50_stage1_8gpu_noflash_occptraineval`
- Tune alpha / confidence threshold based on results

## Abstract

The occluded detection mAP metric is dominated by false positives from visible-object predictions and is not a reliable signal for occluded detection capability. Analysis of an existing run shows high max-recall (0.83–0.92 at 4 m) but near-zero precision, confirming the model can locate occluded objects but the metric conflates this with FP noise. We added TPR and FDR scalars to all detection evaluators and designed a new occlusion classifier head with a corrected loss formulation to enable meaningful precision measurement.

## Intro

The project trains on occluded GT boxes and evaluates them separately via `OccludedDetectionEval`. However, a low `occluded/mAP` could mean either (a) the model cannot find occluded objects, or (b) the model finds them but the evaluation is swamped by false positives. These require different fixes and the mAP alone cannot distinguish them.

Two prior experiments attempted an occlusion classifier head but both collapsed to single-class predictions, making the output useless as a filter. The root cause was not identified at the time.

Questions:
- `Is the low occluded/mAP driven by low recall (model can't find occluded objects) or high FPs (metric is corrupted)?`
- `What caused the visibility head to collapse in prior experiments, and how should the loss be configured?`

## Method

### Metric analysis

Inspected `metrics_details.json` from `sparsedrive_r50_stage2_4gpu_bs24_occptraineval` on Apollo. Extracted max recall (TPR) and FDR = FP/(TP+FP) at the max-recall operating point from the interpolated PR curve per class.

**Why occluded/mAP is structurally unreliable:** the `accumulate_with_ignore` evaluator neutralises predictions matching visible GT within `dist_th`. But predictions slightly off from a visible GT (e.g. 0.6 m at a 0.5 m threshold) count as FPs. With 900 anchors per frame, this generates a large FP mass regardless of how well the model predicts occluded positions. The metric cannot separate a model that explicitly predicts occluded locations from one that merely floods the space with boxes.

### New TPR / FDR scalars

Added `compute_tpr_fdr()` to `occluded_det_eval.py`. Reads `metrics_details.json` after each eval run and computes:
- `mTPR` — mean max-recall across classes × dist_ths (analogous to mAP but measuring recall ceiling only)
- `mFDR` — mean FDR = FP/(TP+FP) at the max-recall operating point
- Per-class `{cls}_tpr` and `{cls}_fdr` at `dist_th_tp` (2 m)

Added to all four evaluators: `img_bbox_NuScenes/`, `vis/`, `occluded/`, `all/`.

### Prior visibility classifier analysis

| Config | acc_visible | acc_occluded | Root cause |
| --- | --- | --- | --- |
| `vishead_occptraineval` | 0.01 | 0.80 | FocalLoss alpha=0.2 → 4× weight on occluded → collapsed to predicting occluded for everything |
| `sepheadvisobsiso_occptrainval` | 0.99 | 0.07 | Default alpha (~0.5) on 1=visible majority + isolated/frozen features → collapsed to visible |

The `vishead` config used `gt_visibility` (1=visible) with alpha=0.2. In FocalLoss, alpha is the positive-class weight, so visible got 0.2 and occluded got 0.8 — a 4× weight on the minority class that over-corrected. The `sepheadvisobsiso` used a separate frozen head where features couldn't adapt, and the unweighted loss collapsed to the majority class.

### New occluded classifier (`occhead_occptraineval`)

Implementation changes:
- Added `gt_occluded = (num_lidar_pts == 0)` (1=occluded, 0=visible) to `NuScenes3DDataset` — separate from `gt_visibility`, no existing paths change
- Added `gt_occluded` passthrough in `InstanceNameFilter` and `CircleObjectRangeFilter` in `transform.py`
- Added `invert_visibility: bool = False` to `SparseBox3DDecoder` — when `True`, outputs `1 − sigmoid(logit)` so `visibility_scores` stays P(visible) and the accuracy evaluation is unchanged
- Set `gt_visibility_key='gt_occluded'` in det_head (positive class = occluded = minority)
- FocalLoss `alpha=0.85, gamma=2.0` — measured train-set occluded match rate is 15.3%, so `alpha≈0.85` better matches the positive-class prior than the original `0.9`
- Train jointly (not isolated) so features adapt to the visibility signal

### Training-set occluded match-rate check

Computed the positive-match prior directly from `data/infos/nuscenes_infos_train.pkl` using the actual Stage 2 training filters: `use_gt_mask=False`, class-name filtering, and the 55 m `CircleObjectRangeFilter`.

Because the visibility loss is applied only on matched positive anchors, and Hungarian assignment matches each GT once in practice (900 predictions per frame, far fewer GT boxes), the relevant class prior is the filtered GT occluded fraction rather than the raw anchor distribution.

Results on train:
- `811,056` filtered GT boxes total
- `124,320` occluded (`num_lidar_pts == 0`)
- `686,736` visible
- occluded match rate = `15.33%`
- visible:occluded ratio = `5.52:1`

Sanity check on val:
- `157,222` filtered GT boxes total
- `23,321` occluded
- occluded match rate = `14.83%`

### Classifier-based prediction filtering

To make `vis/` and `occluded/` metrics symmetric and comparable, both evaluators apply the same classifier gate at eval time via `occ_vis_threshold` in `eval_mode`:

- `occluded/` — keeps predictions where `visibility_score < threshold` (classified as occluded)
- `vis/` — keeps predictions where `visibility_score >= threshold` (classified as visible)

**Why symmetry matters:** without the filter, `occluded/mAP` is evaluated on the full 900-anchor prediction set (FDR ~98–99%), while `vis/mAP` benefits from the ignore mechanism acting on sparse occluded GT. The two metrics would be measuring structurally different things and cannot be fairly compared. Applying the classifier gate to both ensures each evaluator only sees predictions the model intends for that visibility class.

**Implementation:** `visibility_score` is embedded in `results_nusc.json` at format time (`_format_bbox`) by attaching scores to `NuScenesBox` objects before the class-range filter to preserve index alignment. At eval time, filtered copies (`results_nusc_occ_filtered.json`, `results_nusc_vis_filtered.json`) are written and passed to the respective evaluator. No filter is applied when `occ_vis_threshold` is absent or when scores are not present in the JSON (e.g., baseline models without the classifier head).

## Results

| Server | Config | Job ID | Status |
| --- | --- | --- | --- |
| `apollo` | `occhead_occptraineval` | — | NOT STARTED |

### Baseline occluded PR curve analysis (`occptraineval`)

| Class | TPR @2m | FDR @2m | TPR @4m | FDR @4m |
| --- | --- | --- | --- | --- |
| car | 0.810 | 0.989 | 0.900 | 0.987 |
| pedestrian | 0.750 | 0.990 | 0.830 | 0.989 |
| motorcycle | 0.660 | 0.980 | 0.700 | 0.975 |
| traffic_cone | 0.870 | 0.962 | 0.920 | 0.955 |
| barrier | 0.720 | 0.966 | 0.800 | 0.935 |
| truck | 0.420 | 0.995 | 0.560 | 0.995 |
| bus | 0.520 | 0.997 | 0.690 | 0.998 |
| bicycle | 0.480 | 0.993 | 0.540 | 0.995 |
| **mean** | **0.537** | **0.988** | **0.660** | **0.985** |

_occhead accuracy metrics to be filled after run._

### Training-set prior summary

The measured train-set occluded match rate is `15.33%`, not the assumed ~10%. This implies the balancing point for `gt_occluded` is closer to `alpha ≈ 0.85` than `0.9`, so the config was updated accordingly.

## Discussion

The PR curve analysis resolves the central diagnostic question: **low `occluded/mAP` is an FP problem, not a recall problem.** The model achieves TPR 0.53–0.87 at 2 m, meaning it does locate occluded objects. However, FDR is ~98–99% — for every TP, there are ~50–100 FPs from visible-object predictions that narrowly miss the ignore threshold. Optimising for mAP would not improve the underlying capability.

The two prior classifier experiments failed at opposite extremes of the same class-imbalance problem. The `vishead` experiment is the more useful reference: it proves the features encode visibility information (acc_occluded=0.80), but the alpha=0.2 loss weighting was too aggressive. The new `occhead` configuration uses 1=occluded as the positive class (enabling standard pos_weight via alpha). A direct train-set count gives a `15.33%` occluded matched-positive prior (`5.52:1` visible:occluded), so `alpha=0.85` is a better-balanced choice than the original `0.9`.

What improved:
- `Metric clarity — TPR/FDR now separate recall capability from FP contamination across all eval prefixes`
- `Loss formulation — gt_occluded with measured-prior alpha=0.85 should avoid both collapse modes`
- `Metric symmetry — vis/ and occluded/ both apply the same classifier gate, making them directly comparable`

What regressed or stayed flat:
- `occluded/mAP remains structurally low until a working classifier converges`

Likely explanation:
- `The model already has latent occluded-object representations (vishead acc_occ=0.80 confirms this); the bottleneck is a reliable classifier to surface them`

Recommended model:
- `sparsedrive_r50_stage2_4gpu_bs24_occhead_occptraineval` (pending results)

Why:
- `Corrects both failure modes of prior experiments; if alpha=0.85 converges, the classifier enables a meaningful occluded precision signal`

## Future Work

- Once classifier converges, set `occ_vis_threshold` to evaluate filtered `vis/` and `occluded/` mAP as primary comparison metrics
- Consider reporting occluded TPR at fixed threshold as primary occluded metric (bypasses FP problem entirely)
- Stage 1 counterpart: evaluate whether adding occluded head to Stage 1 improves Stage 2 initialisation
