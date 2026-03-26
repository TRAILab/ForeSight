# AutoResearch Log — mar25
**Goal:** Improve val/L2 and val/obj_box_col
**Base config:** projects/configs/sparsedrive_r50_stage2_4gpu_nomap.py
**Session branch:** autoresearch/mar25
**Max experiments:** 5

## Baseline
- L2: 0.5911
- obj_box_col: 0.080% (0.0008)
- car_ade: 0.6189
- NDS: 0.5217
- mAP: 0.4131
- Config: sparsedrive_r50_stage2_4gpu_nomap.py (bs=48, queue=4, lr=3e-4)

## Prior Experiments (from DGX history, not on this branch)
| Config | L2 | obj_box_col | car_ade | NDS | Notes |
|---|---|---|---|---|---|
| nomap (baseline) | 0.5911 | 0.080% | 0.6189 | 0.5217 | bs48 |
| bs24_nomap | 0.5878 | 0.103% | 0.6301 | 0.5262 | smaller batch, better L2 but worse col |
| rotaug | 0.6213 | 0.165% | 0.6494 | 0.5234 | 3D rotation aug — both worse |
| anchorprop | 0.6066 | 0.114% | 0.7060 | 0.4990 | much worse |
| notempmotion | 0.7192 | 0.164% | 0.6288 | 0.5239 | no temporal motion — confirms critical |

**Key insights from history:**
- Temporal motion is essential (noplan/notempmotion both hurt badly)
- Rotation augmentation hurts both L2 and collision
- Smaller batch (24 vs 48) slightly helps L2 but worsens collision
- Anchor propagation hurts (velocity estimation degrades)

---

## [exp-001] auto_mar25_exp001_plan_loss_up — 2026-03-25
**Hypothesis:** Increasing planning regression loss (1.0→2.0) and cls loss (0.5→1.0) should directly reduce L2 by providing stronger supervision for trajectory regression and mode selection.
**Config changes:**
```python
model['head']['motion_plan_head']['plan_loss_reg']['loss_weight'] = 2.0
model['head']['motion_plan_head']['plan_loss_cls']['loss_weight'] = 1.0
```
**Job ID:** 3518
**Status:** keep

**Metrics:**
| Metric | Baseline | Best so far | This exp | Δ vs best |
|--------|----------|-------------|----------|-----------|
| L2 | 0.5911 | 0.5911 | 0.5902 | ↓ -0.0009 |
| obj_box_col | 0.080% | 0.080% | 0.084% | ↑ +0.004% |
| car_ade | 0.6189 | 0.6189 | 0.6405 | ↑ +0.0216 |
| NDS | 0.5217 | 0.5217 | 0.5239 | ↑ +0.0022 |

**Analysis:** Doubling plan loss weights had negligible effect on L2 (-0.0009, within noise). obj_box_col slightly worsened and car_ade degraded — the increased planning loss may be causing gradient interference with motion prediction. The planning network appears near-optimal given current detection/motion quality; the bottleneck is likely upstream feature quality. Next: try boosting motion loss weights to improve agent forecasting quality that feeds into the planner.

---

