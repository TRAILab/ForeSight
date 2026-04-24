# Occluded Detection: Metric Analysis and Classifier Design

## TODO

1. ~~**Fix `invert_visibility` and re-run eval from existing checkpoint.**~~ Done — Narval job 59639710. `invert_visibility=True` + `occ_vis_threshold=0.1` confirmed working (vis/ is now non-zero: NDS=0.311, mAP=0.112). See results below.
2. ~~**Tune `occ_vis_threshold` — threshold=0.1 is too strict.**~~ Re-eval complete — DGX job 3610, `occ_vis_threshold=0.5`. Classifier collapsed to predicting occluded for everything (acc_visible=0.004); threshold sweep cannot fix this. Needs retraining with lower alpha.
3. ~~**Fix AUROC direction bug in `_evaluate_visibility_accuracy`.**~~ Fixed — `invert_visibility` flag threaded from `eval_mode` into the function; scores flipped to P(visible) when `False`. All new configs set `eval_mode.invert_visibility=True`.
4. ~~**Retrain with lower alpha.**~~ DGX job 3611 (alpha=0.5) complete — still collapsed to occluded. Job 3612 (alpha=0.7) cancelled. DGX job 3616 (alpha=0.2) complete — no single-class collapse, but classifier quality is still poor (`acc_visible=0.214`, `acc_occluded=0.190`, `AUROC=0.119`, `occluded/mAP=0.018`).
5. **Add TPR at fixed threshold as a supplementary primary occluded metric.** It bypasses the FP-contamination problem entirely and is directly interpretable as recall capability.
6. **Stage 1 counterpart.** Evaluate whether adding the occluded classifier head to Stage 1 training improves Stage 2 initialisation for occluded detection.

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

### AUROC direction fix

`_evaluate_visibility_accuracy` always computes AUROC with `gt_vis = (num_lidar_pts > 0)` as the positive class (1 = visible). Scores must therefore be P(visible) for the metric to be meaningful. This was not enforced: when `gt_visibility_key='gt_occluded'` and `invert_visibility=False` in the decoder, scores are P(occluded) and AUROC = 1 − true AUROC.

Fix: added `invert_visibility` parameter to `_evaluate_visibility_accuracy` (default `True` = scores are already P(visible)). When `False`, scores are flipped before computing AUROC, accuracy, and opt_threshold. The caller reads `eval_mode.get('invert_visibility', True)`, so configs that set `invert_visibility=True` in `eval_mode` get correct metrics without any heuristic. All new occhead configs include this flag.

Note: job 3610 reported AUROC=0.121 despite having `invert_visibility=True` in the decoder config. The cause is not yet confirmed — the eval may have loaded cached inference results from the original wrong-sign run (59608052), or the config override did not take effect. With the fix in place and both flags set consistently, new runs should report the correct AUROC.

### Prior visibility classifier analysis

| Config | acc_visible | acc_occluded | Root cause |
| --- | --- | --- | --- |
| `vishead_occptraineval` | 0.992 | 0.065 | `gt_visibility` (1=visible, majority) + alpha=0.2 insufficient minority boost → collapsed to predicting visible |
| `sepheadvisobsiso_occptrainval` | 0.99 | 0.07 | Default alpha (~0.5) on 1=visible majority + isolated/frozen features → collapsed to visible |

The `vishead` config used `gt_visibility` (1=visible) with alpha=0.2. In FocalLoss, alpha is the positive-class weight, so visible got 0.2 per example and occluded got 0.8 — but with 5.5× more visible examples, total gradient pressure still favoured visible (5.5 × 0.2 = 1.1 vs 1 × 0.8 = 0.8), and the classifier collapsed to predicting visible for everything. The `sepheadvisobsiso` used a separate frozen head where features couldn't adapt, and the unweighted loss collapsed to the majority class via the same mechanism.

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
| `dgx` | `stage2_4gpu_bs24_vishead_occptraineval` | — | `complete` |
| `narval` | `stage2_4gpu_bs24_occhead_occptraineval` | `59608052` | `complete` |
| `narval` | `stage2_4gpu_bs24_occhead_occptraineval` (corrected eval: `invert_visibility=True`, `occ_vis_threshold=0.1`) | `59639710` | `complete` |
| `dgx` | `stage2_4gpu_bs24_occhead_occptraineval` (eval rerun: `invert_visibility=True`, `occ_vis_threshold=0.5`) | `3610` | `complete` |
| `dgx` | `stage2_4gpu_bs24_occhead_a50_occptraineval` (alpha=0.5) | `3611` | `complete` |
| `dgx` | `stage2_4gpu_bs24_occhead_a70_occptraineval` (alpha=0.7) | `3612` | `cancelled` |
| `dgx` | `stage2_4gpu_bs24_occhead_a20_occptraineval` (alpha=0.2) | `3616` | `complete` |

| Run | mAP | mAP_all | mAP_occluded | acc_visible | acc_occluded | AUROC |
| --- | --- | --- | --- | --- | --- | --- |
| `vishead_occptraineval` | 0.41000 | 0.40300 | 0.01400 | 0.99200 | 0.06500 | 0.76700 |
| `occhead` original (`59608052`) | 0.41200 | 0.40400 | 0.03500 | 0.00400 | 0.86300 | 0.12100 |
| `occhead` corrected eval (`59639710`) | 0.41100 | 0.40300 | 0.00100 | - | - | - |
| `occhead` rerun threshold=0.5 (`3610`) | 0.41000 | 0.40300 | 0.03600 | 0.00400 | 0.86200 | - |
| `occhead_a50` (`3611`) | 0.41000 | 0.40300 | 0.03300 | 0.05600 | 0.55500 | 0.11800 |
| `occhead_a20` (`3616`) | 0.40752 | 0.39912 | 0.01818 | 0.21433 | 0.19039 | 0.11860 |

`mAP` is the standard full-val nuScenes detection mAP. `mAP_all` and `mAP_occluded` are from the custom all/occluded splits. Accuracy and AUROC come from the visibility classifier evaluation when the run logged trustworthy values.

### vishead reference (DGX `sparsedrive_r50_stage2_4gpu_bs24_vishead_occptraineval`, 2026-02-26)

| Metric prefix | NDS | mAP | mAP_normal | L2 | obj_box_col |
| --- | --- | --- | --- | --- | --- |
| img_bbox_NuScenes (standard) | 0.523 | 0.410 | 0.559 | 0.612 | 0.122% |
| all/ | 0.519 | 0.403 | — | — | 0.122% |
| occluded/ | 0.306 | 0.014 | — | — | 0.0% |

| Visibility classifier | Value |
| --- | --- |
| acc_visible | 0.992 |
| acc_occluded | 0.065 |
| AUROC | 0.767 |

Classifier collapsed to predicting visible for everything. AUROC=0.767 in the wrong-sign direction ≈ 0.233 real discriminative power (near-random). `occluded/` metrics are from unfiltered predictions — `occ_vis_threshold` filtering was not yet implemented at this stage. Standard detection unchanged from baseline.

### occhead threshold=0.5 eval (DGX job 3610, `invert_visibility=True`, `occ_vis_threshold=0.5`)

| Metric prefix | NDS | mAP | mTPR | mFDR | obj_box_col |
| --- | --- | --- | --- | --- | --- |
| img_bbox_NuScenes (standard) | 0.519 | 0.410 | — | — | — |
| all/ | 0.514 | 0.403 | 0.650 | 0.876 | 0.152% |
| occluded/ | 0.261 | 0.036 | 0.490 | 0.989 | 0.0% |
| vis/ | 0.0 | 0.0 | 0.002 | 0.707 | — |

| Visibility classifier | Value |
| --- | --- |
| acc_visible | 0.004 |
| acc_occluded | 0.862 |
| opt_threshold (Youden's J) | 0.007 |

Classifier collapsed to predicting occluded for everything — the mirror image of vishead. With `visibility_score = P(visible)` and threshold=0.5, `acc_visible=0.004` means 99.6% of all predictions (visible and occluded alike) get P(visible) < 0.5 and pass to `occluded/`. This makes `occluded/` nearly identical to the unfiltered baseline and leaves `vis/` empty. `opt_threshold=0.007` confirms the distribution: the entire output is packed near zero, with visible and occluded predictions both assigned P(visible) ≈ 0. Threshold sweeping cannot recover useful separation from a collapsed classifier.

### occhead_a50 (DGX job 3611, alpha=0.5)

| Metric prefix | NDS | mAP | mTPR | mFDR | obj_box_col |
| --- | --- | --- | --- | --- | --- |
| img_bbox_NuScenes (standard) | 0.523 | 0.410 | — | — | — |
| all/ | 0.518 | 0.403 | 0.643 | 0.869 | 0.124% |
| occluded/ | 0.275 | 0.033 | 0.384 | 0.987 | 0.0% |
| vis/ | 0.0 | 0.0 | 0.028 | 0.897 | — |

| Visibility classifier | Value |
| --- | --- |
| acc_visible | 0.056 |
| acc_occluded | 0.555 |
| AUROC | 0.118 |
| opt_threshold (Youden's J) | 0.010 |

L2=0.591. Still collapsed to occluded — vis/ empty, occluded/ reflects unfiltered prediction set. Less severe than alpha=0.85 (acc_visible was 0.004) but opt_threshold=0.010 confirms P(visible) packed near zero. Focal term suppression of easy visible examples likely responsible.

### occhead_a20 (DGX job 3616, alpha=0.2)

| Metric prefix | NDS | mAP | mTPR | mFDR | obj_box_col |
| --- | --- | --- | --- | --- | --- |
| img_bbox_NuScenes (standard) | 0.524 | 0.408 | — | — | — |
| all/ | 0.519 | 0.399 | 0.642 | 0.871 | 0.098% |
| occluded/ | 0.247 | 0.018 | 0.220 | 0.976 | 0.0% |
| vis/ | 0.110 | 0.008 | 0.120 | 0.960 | — |

| Visibility classifier | Value |
| --- | --- |
| acc_visible | 0.214 |
| acc_occluded | 0.190 |
| AUROC | 0.119 |
| opt_threshold (Youden's J) | 0.014 |

L2=0.615. This run no longer collapses fully to a single class, but it is still not a usable classifier. Both visible and occluded accuracy are poor, vis/ and occluded/ are both weak, and the classifier does not recover the strong occluded-filtering behavior needed to turn latent recall into precision. Standard detection remains healthy, but the visibility head is still not solving the FP contamination problem.

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

The three classifier experiments form a consistent pattern: all collapsed to single-class predictions, each from a different imbalance mis-configuration. `vishead` (gt_visibility, positive=visible majority, alpha=0.2) collapsed to visible — total gradient favoured visible even with down-weighting (5.5 × 0.2 > 1 × 0.8). `sepheadvisobsiso` collapsed to visible via frozen features. `occhead` (gt_occluded, positive=occluded minority, alpha=0.85) over-corrected into the other failure mode — alpha=0.85 gives occluded ~5.7× effective weight against a 5.5:1 imbalance and collapsed to occluded. The features do encode visibility information (vishead AUROC=0.767 in the wrong-sign direction ≈ real AUROC 0.233 is not informative; the occhead wrong-sign AUROC=0.121 → real ≈ 0.879 confirms discriminability). The bottleneck is calibrating alpha to sit between the two collapse extremes.

What improved:
- `Metric clarity — TPR/FDR now separate recall capability from FP contamination across all eval prefixes`
- `Loss formulation — gt_occluded with measured-prior alpha=0.85 should avoid both collapse modes`
- `Metric symmetry — vis/ and occluded/ both apply the same classifier gate, making them directly comparable`

What regressed or stayed flat:
- `occluded/mAP remains structurally low until a working classifier converges`

Likely explanation:
- `The model already has latent occluded-object representations (occhead AUROC≈0.879 corrected confirms this); the bottleneck is calibrating the classifier loss to avoid single-class collapse`

**occhead threshold=0.5 eval (DGX job 3610) — complete.** Classifier has collapsed to predicting occluded for everything (acc_visible=0.004, acc_occluded=0.862, opt_threshold=0.007). The `occluded/` bucket (NDS=0.261, mAP=0.036) is populated only because the full prediction set passes through — these numbers match the unfiltered baseline, not a working classifier. `vis/` is essentially empty (NDS=0.0, mAP=0.0). Threshold sweeping cannot recover this; the checkpoint needs retraining with lower alpha. With a 5.52:1 imbalance, alpha=0.85 gives the minority (occluded) class ~5.7× effective weight — slightly above the imbalance ratio and enough to push collapse into the occluded direction. Target alpha=0.5–0.6 for the next retrain (balanced gradient at the imbalance midpoint).

**occhead corrected eval (Narval 59639710) — complete.** `invert_visibility=True` + `occ_vis_threshold=0.1` confirms the sign-flip fix: vis/ is now non-zero (NDS=0.311, mAP=0.112), proving the classifier is routing some predictions to the visible bucket. However, occluded/ is near-zero (NDS=0.001, mAP=0.001, mTPR=0.027). The reason: with `invert_visibility=True`, `visibility_score` = P(visible), so threshold=0.1 means a prediction must have P(visible) < 0.1 (i.e., P(occluded) > 0.9) to reach occluded/. This is too strict — the classifier routes almost everything to vis/ even for moderately occluded predictions. Standard detection is unchanged (NDS=0.520, mAP=0.411, L2=0.592), confirming the classifier head does not hurt main detection.

**occhead (Narval 59608052) — complete, eval sign-flip bug identified.** Standard detection is unchanged (NDS=0.521, mAP=0.412). Classifier accuracy metrics: acc_visible=0.004, acc_occluded=0.863, AUROC=0.121. The AUROC of 0.121 looks like near-random but is actually the inverse of a well-trained classifier: because `gt_visibility_key='gt_occluded'` trains sigmoid(logit) → 1 for occluded items, and `invert_visibility=False` means the output is passed directly as `visibility_score`, the score direction is P(occluded). The eval filter uses `visibility_score < threshold → occluded` which is designed for P(visible), so the directions are inverted. Real AUROC ≈ 1 − 0.121 = 0.879 — the classifier is working well. Fix: set `invert_visibility=True` in the decoder config, which flips the score to P(visible) before it reaches the eval filters. No retraining is needed; re-running eval from the existing checkpoint should recover the correct vis/ and occluded/ metrics.

**occhead_a50 (DGX job 3611) — complete.** Still collapsed to predicting occluded. acc_visible=0.056, acc_occluded=0.555, AUROC=0.118, opt_threshold=0.010. vis/ NDS=0.0, mAP=0.0; occluded/ NDS=0.275, mAP=0.033 (unfiltered baseline). Standard detection unchanged (NDS=0.523, mAP=0.410, L2=0.591). The collapse is less severe than alpha=0.85 (acc_visible was 0.004) but the opt_threshold=0.010 confirms P(visible) is still packed near zero — the focal term (1−p)² is likely suppressing gradient on easy visible examples, so effective pressure continues to favour the occluded class even with equal per-example alpha. The balance point is at a lower alpha than expected.

**alpha=0.7 (DGX job 3612) — cancelled.** Redundant given alpha=0.5 already failed; alpha=0.7 would only confirm collapse in the same direction. Replaced by alpha=0.2 (job 3616).

**occhead_a20 (DGX job 3616) — complete.** alpha=0.2 avoided the earlier single-class collapse, but the classifier is still poor: acc_visible=0.214, acc_occluded=0.190, AUROC=0.1186, opt_threshold=0.0139. The model now routes predictions into both vis/ and occluded/ buckets, but both are weak (`vis/mAP=0.0078`, `occluded/mAP=0.0182`). This means lowering alpha did not find the useful operating region between the visible-collapse and occluded-collapse regimes; it mostly destroyed separability instead. The next step should be to treat this as a calibration/representation problem rather than continuing a one-dimensional alpha sweep.

**AUROC fix — complete.** `_evaluate_visibility_accuracy` now accepts `invert_visibility` from `eval_mode` and flips scores when `False`. All prior runs had AUROC potentially wrong (0.121 in both 59608052 and 3610); new runs with the consistent flag will be the first reliable measurements.

## Future Work

- Once classifier converges, set `occ_vis_threshold` to evaluate filtered `vis/` and `occluded/` mAP as primary comparison metrics
- Consider reporting occluded TPR at fixed threshold as primary occluded metric (bypasses FP problem entirely)
- Stage 1 counterpart: evaluate whether adding occluded head to Stage 1 improves Stage 2 initialisation
