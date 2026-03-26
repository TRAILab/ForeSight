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

## [exp-002] auto_mar25_exp002_motion_loss_up — 2026-03-26
**Hypothesis:** Increasing motion loss weights (0.2→0.5) should improve agent trajectory forecasting, giving the planner better context for collision avoidance.
**Config changes:**
```python
model['head']['motion_plan_head']['motion_loss_reg']['loss_weight'] = 0.5
model['head']['motion_plan_head']['motion_loss_cls']['loss_weight'] = 0.5
```
**Job ID:** 3519
**Status:** discard

**Metrics:**
| Metric | Baseline | Best so far | This exp | Δ vs best |
|--------|----------|-------------|----------|-----------|
| L2 | 0.5911 | 0.5902 | 0.6358 | ↑ +0.0456 |
| obj_box_col | 0.080% | 0.080% | 0.148% | ↑ +0.068% |
| car_ade | 0.6189 | 0.6405 | 0.6146 | ↓ -0.0259 |
| NDS | 0.5217 | 0.5239 | 0.5158 | ↓ -0.0081 |

**Analysis:** Strong negative result. Motion loss increase 0.2→0.5 dramatically hurts both L2 (+7.6%) and obj_box_col (+85% relative). Interestingly car_ade improved slightly (better motion forecasting), but planning quality collapsed — higher motion gradients appear to dominate and distort the planning head's learning. The 0.2 baseline motion loss weight seems well-calibrated. Do NOT increase motion loss weights. Next: try queue_length=6 for more temporal context.

---

## [exp-003] auto_mar25_exp003_queue6 — 2026-03-26
**Hypothesis:** Increasing temporal queue from 4 to 6 frames gives the model more historical ego-motion and agent state context, improving trajectory prediction and planning accuracy.
**Config changes:**
```python
queue_length = 6
model['head']['motion_plan_head']['instance_queue']['queue_length'] = queue_length
```
**Job ID:** 3520
**Status:** keep

**Metrics:**
| Metric | Baseline | Best so far | This exp | Δ vs best |
|--------|----------|-------------|----------|-----------|
| L2 | 0.5911 | 0.5902 | 0.5757 | ↓ -0.0145 (new best) |
| obj_box_col | 0.080% | 0.080% | 0.100% | ↑ +0.020% |
| car_ade | 0.6189 | 0.6405 | 0.6281 | ↓ -0.0124 |
| NDS | 0.5217 | 0.5239 | 0.5238 | ~ |

**Analysis:** Strong positive result for L2 — queue_length=6 reduces L2 by 2.6% absolute vs baseline (new best: 0.5757). More temporal context clearly helps the planner produce more accurate trajectories. obj_box_col regressed slightly (0.100% vs 0.080%) — additional frames may shift optimization balance. Next: exp-004 tests num_decoder=8; exp-005 will combine queue=6 with decoder=8 to target both metrics simultaneously.

---

## [exp-004] auto_mar25_exp004_decoder8 — 2026-03-26
**Hypothesis:** Increasing detection decoder depth from 6 to 8 layers produces richer instance features, improving detection quality and downstream motion/planning.
**Config changes:**
```python
num_decoder = 8
```
**Job ID:** 3521
**Status:** keep

**Metrics:**
| Metric | Baseline | Best so far | This exp | Δ vs best |
|--------|----------|-------------|----------|-----------|
| L2 | 0.5911 | 0.5757 | 0.5955 | ↑ +0.0198 vs best L2 |
| obj_box_col | 0.080% | 0.080% | 0.074% | ↓ -0.006% (new best col) |
| car_ade | 0.6189 | 0.6281 | 0.6189 | = baseline |
| NDS | 0.5217 | 0.5239 | 0.5239 | ~ |

**Analysis:** num_decoder=8 achieves best obj_box_col (0.074%, beating baseline 0.080%) and matching car_ade with baseline. However L2 is slightly worse than baseline (0.5955 vs 0.5911). The richer detection features help collision avoidance, but the larger model may need more epochs to converge on planning. Two clear winners emerge: queue=6 for L2, decoder=8 for obj_box_col — exp-005 combines both to target improvements on both metrics simultaneously.

---

## [exp-005] auto_mar25_exp005_queue6_dec8 — 2026-03-26
**Hypothesis:** Combining queue_length=6 (best L2) and num_decoder=8 (best obj_box_col) should yield improvements on both metrics simultaneously.
**Config changes:**
```python
queue_length = 6
model['head']['motion_plan_head']['instance_queue']['queue_length'] = queue_length
num_decoder = 8
```
**Job ID:** 3522
**Status:** discard

**Metrics:**
| Metric | Baseline | Best so far | This exp | Δ vs best |
|--------|----------|-------------|----------|-----------|
| L2 | 0.5911 | 0.5757 | 0.5934 | ↑ +0.0177 vs best L2 |
| obj_box_col | 0.080% | 0.074% | 0.126% | ↑ +0.052% vs best col |
| car_ade | 0.6189 | 0.6189 | 0.6292 | ↑ +0.0103 |
| NDS | 0.5217 | 0.5241 | 0.5241 | ~ |

**Analysis:** Negative synergy — combining queue=6 and decoder=8 produces results worse than either individually on both primary metrics. L2=0.5934 (worse than exp003's 0.5757) and obj_box_col=0.126% (far worse than exp004's 0.074% and baseline's 0.080%). The larger model (8 decoder layers) combined with longer temporal context (6 frames) likely exceeds what 10 training epochs can optimize effectively. The two changes compete for the same capacity budget. To get benefits of both, more training epochs (15-20) would likely be needed.

---

## Conclusions

**Session:** autoresearch/mar25 | **Date:** 2026-03-26 | **Goal:** Improve val/L2 and val/obj_box_col

### Final Results Table

| Exp | Config | L2 | obj_box_col | car_ade | NDS | Δ L2 | Δ col | Status |
|-----|--------|-----|-------------|---------|-----|------|-------|--------|
| baseline | nomap bs48 | 0.5911 | 0.080% | 0.6189 | 0.5217 | — | — | — |
| exp-001 | plan_loss_reg 1→2, cls 0.5→1 | 0.5902 | 0.084% | 0.6405 | 0.5239 | -0.0009 | +0.004% | keep |
| exp-002 | motion_loss 0.2→0.5 | 0.6358 | 0.148% | 0.6146 | 0.5158 | +0.0447 | +0.068% | **discard** |
| exp-003 | queue_length 4→6 | **0.5757** | 0.100% | 0.6281 | 0.5238 | **-0.0154** | +0.020% | keep ⭐ |
| exp-004 | num_decoder 6→8 | 0.5955 | **0.074%** | 0.6189 | 0.5239 | +0.0044 | **-0.006%** | keep ⭐ |
| exp-005 | queue6 + decoder8 | 0.5934 | 0.126% | 0.6292 | 0.5241 | +0.0023 | +0.046% | **discard** |

### Key Findings

1. **`queue_length=6` is the strongest single lever for L2** (exp-003): Reduces L2 by 0.0154 (2.6% relative) vs baseline. More temporal history gives the planner better ego-motion context. Config: `queue_length = 6` + `instance_queue queue_length = 6`.

2. **`num_decoder=8` is the strongest single lever for obj_box_col** (exp-004): Reduces collision rate from 0.080% to 0.074% (7.5% relative improvement). Richer detection features from deeper decoder improve collision awareness. NDS and car_ade unchanged.

3. **Loss weight changes are not effective levers** (exp-001, exp-002): Increasing plan loss had negligible effect on L2 (-0.0009). Increasing motion loss caused major regression on both metrics (+7.6% L2, +85% col). The default loss weights (plan_reg=1.0, motion_reg=0.2) appear well-calibrated.

4. **Combinations require more training epochs**: queue=6 + decoder=8 together showed negative synergy at 10 epochs — both metrics worse than individual bests. The increased model complexity needs longer training to converge.

### Recommended Next Steps

1. **Use `queue_length=6` as the new default** — clear L2 improvement with no architectural cost. Best config for L2: `auto_mar25_exp003_queue6`.

2. **Try `num_decoder=8` with 15+ epochs** — the collision improvement likely compounds with more training. The 10-epoch budget may be insufficient for the larger model.

3. **Try queue=6 + decoder=8 with 15 epochs** — the negative synergy may disappear with adequate training time. This combination has the highest ceiling.

4. **Avoid motion loss weight increases** — strong negative result, do not revisit.

5. **Explore dropout reduction** (0.1→0.05) or LR warmup tuning as next levers — loss weights and architecture depth are now better understood.



---

# AutoResearch Log — mar26
**Goal:** Improve val/L2 and val/obj_box_col
**Base config:** projects/configs/sparsedrive_r50_stage2_4gpu_nomap_queue6.py
**Session branch:** autoresearch/mar26
**Max experiments:** 5

## Context from mar25
- Best L2: 0.5757 (exp003, queue_length=6 — now baked into base config)
- Best col: 0.074% (exp004, num_decoder=8)
- Hard constraints: motion_loss >0.2 kills both metrics; queue6+decoder8 negative at 10 epochs
- Untested ideas on queue=6 base: extended training, num_det=100, confidence_decay, plan_loss_up

## Experiment Plan
1. exp001: Extended training 10→15 epochs (addresses known bottleneck B2)
2. exp002: Plan loss upweighting (plan_reg 1→2, plan_cls 0.5→1) on queue=6 base
3. exp003: num_det=100 (more agents to planner, untested)
4. exp004: confidence_decay=0.8 (better temporal tracking)
5. exp005: Best combo of above

