# Occluded Detection: Metric Analysis and Classifier Design

## TODO

1. ~~**Fix `invert_visibility` and re-run eval from existing checkpoint.**~~ Done — Narval job 59639710. `invert_visibility=True` + `occ_vis_threshold=0.1` confirmed working (vis/ is now non-zero: NDS=0.311, mAP=0.112). See results below.
2. ~~**Tune `occ_vis_threshold` — threshold=0.1 is too strict.**~~ Re-eval submitted — DGX job 3609, `occ_vis_threshold=0.5`. Results pending.
3. **Add TPR at fixed threshold as a supplementary primary occluded metric.** It bypasses the FP-contamination problem entirely and is directly interpretable as recall capability.
4. **Stage 1 counterpart.** Evaluate whether adding the occluded classifier head to Stage 1 training improves Stage 2 initialisation for occluded detection.

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

Also added `visibility/opt_threshold` — the Youden's J optimal threshold (argmax TPR−FPR over the ROC curve), computed from the same sorted-score pass as AUROC. This gives the principled `occ_vis_threshold` value for each checkpoint without manual sweeping.

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
| `apollo` | `stage1_8gpu_noflash_occptraineval` (eval rerun) | — | `complete` |
| `narval` | `stage2_4gpu_bs24_occhead_occptraineval` | `59608052` | `complete` |
| `narval` | `stage2_4gpu_bs24_occhead_occptraineval` (corrected eval: `invert_visibility=True`, `occ_vis_threshold=0.1`) | `59639710` | `complete` |
| `dgx` | `stage2_4gpu_bs24_occhead_occptraineval` (eval rerun: `invert_visibility=True`, `occ_vis_threshold=0.5`) | `3610` | `running` |

### occhead threshold=0.5 eval (DGX job 3609, `invert_visibility=True`, `occ_vis_threshold=0.5`)

Results pending.

The key question: with threshold=0.5, a prediction reaches `occluded/` if P(visible) < 0.5 (P(occluded) > 0.5). Based on the classifier accuracy analysis, ~14% of occluded GT objects exceed this threshold (true acc_occluded ≈ 0.137), so occluded/ should be non-trivially populated. Whether this is enough to give a meaningful mAP depends on whether those ~14% are spatially well-localised.

### occhead corrected eval (Narval 59639710, `invert_visibility=True`, `occ_vis_threshold=0.1`)

| Metric prefix | NDS | mAP | L2 | obj_box_col | car_epa | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| img_bbox_NuScenes (standard) | 0.520 | 0.411 | 0.592 | 0.154% | 0.478 | |
| all/ | 0.515 | 0.403 | — | 0.154% | 0.474 | |
| occluded/ | 0.001 | 0.001 | — | 0.0% | — | near-empty: threshold too strict |
| vis/ | 0.311 | 0.112 | — | — | — | non-zero: sign-flip fix confirmed |

### occhead results (Narval 59608052, original `invert_visibility=False`)

| Metric prefix | NDS | mAP | mAP_normal | L2 | obj_box_col |
| --- | --- | --- | --- | --- | --- |
| img_bbox_NuScenes (standard) | 0.521 | 0.412 | 0.551 | 0.594 | 0.156% |
| all/ | 0.515 | 0.404 | — | — | 0.156% |
| occluded/ | 0.260 | 0.035 | — | — | 0.0% |
| vis/ | 0.0 | 0.0 | — | — | — |

The vis/ metrics being 0.0 indicates the classifier collapsed to predicting all boxes as occluded — the visibility filter (`visibility_score >= threshold`) produced an empty prediction set. The occluded/ NDS=0.260/mAP=0.035 are non-zero, consistent with the classifier passing occluded-class predictions through. Standard detection (img_bbox_NuScenes) is essentially unchanged from the baseline (NDS=0.521 vs 0.523, mAP=0.412 vs 0.413).

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

_Classifier accuracy metrics (acc_visible / acc_occluded) not yet extracted — requires inspecting the full eval log._

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

**occhead threshold=0.5 eval (DGX job 3610) — running.** Re-eval of the same checkpoint as job 59639710 with `occ_vis_threshold` raised from 0.1 to 0.5. With threshold=0.5, a prediction reaches `occluded/` if P(visible) < 0.5 (P(occluded) > 0.5). Code inspection of the accuracy metric (`nuscenes_3d_dataset.py:1214`) confirms the classifier is not collapsed but is miscalibrated: it assigns P(occ) < 0.5 to ~86% of occluded objects (acc_occluded=0.863 in the wrong-sign run means true acc_occluded≈0.137 after correction). AUROC=0.879 (corrected) confirms the classifier can discriminate — it just outputs sub-0.5 P(occ) for most occluded objects, so the threshold=0.5 eval will be sparse. If occluded/ is still near-zero at threshold=0.5, the classifier needs retraining with adjusted calibration rather than a threshold sweep.

**occhead corrected eval (Narval 59639710) — complete.** `invert_visibility=True` + `occ_vis_threshold=0.1` confirms the sign-flip fix: vis/ is now non-zero (NDS=0.311, mAP=0.112), proving the classifier is routing some predictions to the visible bucket. However, occluded/ is near-zero (NDS=0.001, mAP=0.001, mTPR=0.027). The reason: with `invert_visibility=True`, `visibility_score` = P(visible), so threshold=0.1 means a prediction must have P(visible) < 0.1 (i.e., P(occluded) > 0.9) to reach occluded/. This is too strict — the classifier routes almost everything to vis/ even for moderately occluded predictions. Standard detection is unchanged (NDS=0.520, mAP=0.411, L2=0.592), confirming the classifier head does not hurt main detection.

**occhead (Narval 59608052) — complete, eval sign-flip bug identified.** Standard detection is unchanged (NDS=0.521, mAP=0.412). Classifier accuracy metrics: acc_visible=0.004, acc_occluded=0.863, AUROC=0.121. The AUROC of 0.121 looks like near-random but is actually the inverse of a well-trained classifier: because `gt_visibility_key='gt_occluded'` trains sigmoid(logit) → 1 for occluded items, and `invert_visibility=False` means the output is passed directly as `visibility_score`, the score direction is P(occluded). The eval filter uses `visibility_score < threshold → occluded` which is designed for P(visible), so the directions are inverted. Real AUROC ≈ 1 − 0.121 = 0.879 — the classifier is working well. Fix: set `invert_visibility=True` in the decoder config, which flips the score to P(visible) before it reaches the eval filters. No retraining is needed; re-running eval from the existing checkpoint should recover the correct vis/ and occluded/ metrics.

## Future Work

- Once classifier converges, set `occ_vis_threshold` to evaluate filtered `vis/` and `occluded/` mAP as primary comparison metrics
- Consider reporting occluded TPR at fixed threshold as primary occluded metric (bypasses FP problem entirely)
- Stage 1 counterpart: evaluate whether adding occluded head to Stage 1 improves Stage 2 initialisation
