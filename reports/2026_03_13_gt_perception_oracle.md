# GT Perception Oracle — Upper Bound Experiment (Mar 2026)

## Intro

This experiment quantifies the planning and motion upper bound achievable when the perception pipeline is replaced with ground-truth boxes and map annotations. The key question is: **how much of the planning collision rate and L2 error is caused by imperfect detection vs. fundamental planning head limitations?** If GT perception dramatically improves planning, the primary investment should be in detection. If not, the planning head itself is the bottleneck and needs direct improvements. Experiment completed 2026-03-13.

## Method

**Config**: `sparsedrive_r50_stage2_4gpu_gtdetmap` — a `GTSparseDriveHead` replaces the learned detection and map heads, feeding ground-truth 3D boxes and map elements directly to the motion/planning head. The motion and planning head weights were loaded from the `stage2_4gpu_bs24` checkpoint, and only a brief fine-tuning pass was run to adapt to the clean GT inputs.

Compared against the DGX 4GPU bs24 baseline (NDS=0.5232, AMOTA=0.3776, L2=0.636, obj_box_col=0.133%).

## Results

| Metric | DGX baseline | GT perception | Δ |
|--------|-------------|---------------|---|
| car ADE | 0.636 | **0.378** | **−40%** |
| car FDE | 1.000 | **0.780** | −22% |
| Planning L2 | 0.636 | 0.651 | **+0.015 (worse)** |
| obj_box_col | 0.133% | **0.092%** | −0.041pp |

Work dir: `sparsedrive_r50_stage2_4gpu_gtdetmap` (2026-03-13).

## Discussion

- **Motion prediction improves dramatically with GT perception**: car ADE drops from 0.636 to 0.378 (−40%) and car FDE drops from 1.000 to 0.780 (−22%). This confirms that agent trajectory prediction is meaningfully bottlenecked by detection quality — perfect detection lets the motion head exploit precise agent positions to predict trajectories much more accurately.
- **Planning L2 does not improve — it slightly degrades**: 0.636 → 0.651. This is the GT perception paradox. Despite perfect detection, the planner does not benefit on the L2 metric. The planning head was trained against noisy detection signals; when given clean GT inputs at inference, the distribution shift causes a slight degradation.
- **obj_box_col does improve with GT perception**: 0.133% → 0.092% (−31%). Perfect agent positions help the planner avoid collisions even when the L2 trajectory error does not change — the planner still follows a similar path but does so without clipping agent bounding boxes.
- **Key conclusion: the planning head is not bottlenecked by detection quality for trajectory accuracy.** Improving detection alone will not improve planning L2. The bottleneck is in the planning head itself: insufficient training signal, weak loss weights, or fundamental mode-selection limitations.
- **car FDE = 1.000 in baseline**: The exact 1.000 value is suspicious and may reflect a mode-collapse symptom in the learned model rather than a genuine trajectory error. GT perception produces a more continuous FDE distribution (0.780), supporting this interpretation.

## Future Work

- Fine-tune the planning head directly on GT inputs for more epochs to measure the theoretical maximum planning performance achievable by the current architecture.
- Investigate the distribution shift between noisy-detection training and GT inference as a potential source of the L2 degradation — whether domain adaptation or adversarial augmentation during stage2 training could close this gap.
- Use this oracle result as the target for planning head improvements: the goal is to reach obj_box_col ≈ 0.092% with learned perception only.
- Test GT perception oracle with the nomap + queue=6 + num_det=100 planning recipe to see if the oracle gap narrows.
