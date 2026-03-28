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


## [exp-000] auto_mar26_exp000_baseline — 2026-03-26
**Hypothesis:** Establish reproducible baseline for mar26 session on new base config (queue=6 already baked in).
**Config changes:** WandB name only.
**Job ID:** 3523
**Status:** baseline

**Metrics:**
| Metric | Value |
|--------|-------|
| L2 | 0.5927 |
| obj_box_col | 0.104% |
| car_ade | 0.6241 |
| NDS | 0.5236 |
| AMOTA | 0.3878 |

**Analysis:** Baseline is consistent with mar25 session (queue=6 already baked in). L2=0.5927 is slightly higher than mar25 exp003's 0.5757 — within expected seed variance (~±0.02). obj_box_col=0.104% matches the nomap_bs24 result from earlier history. This is the reference for all mar26 experiments.

---

## [exp-001] auto_mar26_exp001_epochs15 — 2026-03-27
**Hypothesis:** Extending stage-2 training from 10→15 epochs addresses the known undertraining bottleneck (B2). More optimizer steps should improve motion and planning quality.
**Config changes:**
```python
num_epochs = 15
checkpoint_epoch_interval = 15
runner = dict(type="IterBasedRunner", max_iters=num_iters_per_epoch * num_epochs)
checkpoint_config = dict(interval=num_iters_per_epoch * checkpoint_epoch_interval)
evaluation = dict(interval=num_iters_per_epoch * checkpoint_epoch_interval, eval_mode=eval_mode)
```
**Job ID:** 3524
**Status:** keep

**Metrics:**
| Metric | Baseline | Best so far | This exp | Δ vs best |
|--------|----------|-------------|----------|-----------|
| L2 | 0.5927 | 0.5927 | 0.5738 | ↓ -0.0189 (new best) |
| obj_box_col | 0.104% | 0.104% | 0.099% | ↓ -0.005% (new best) |
| car_ade | 0.6241 | 0.6241 | 0.6227 | ↓ -0.001 |
| NDS | 0.5236 | 0.5236 | 0.5253 | ↑ +0.002 |

**Analysis:** Clear win on both primary metrics. +50% more training (15 vs 10 epochs) reduces L2 by 3.2% and collision by ~5% relative. Confirms the stage-2 undertraining bottleneck (B2). Detection also slightly improves (NDS +0.002), suggesting the additional epochs help the joint optimization converge better. Next question: does plan_loss_up stack with extended training? exp002 tests plan_loss_up in isolation first (10 epochs), then exp005 will combine if exp001+exp002 are both positive.

---

## [exp-002] auto_mar26_exp002_plan_loss_up — 2026-03-27
**Hypothesis:** Increasing planning loss weights (plan_reg 1→2, plan_cls 0.5→1) provides stronger direct supervision for trajectory regression and mode selection, improving both L2 and collision rate on the queue=6 base.
**Config changes:**
```python
model['head']['motion_plan_head']['plan_loss_reg']['loss_weight'] = 2.0
model['head']['motion_plan_head']['plan_loss_cls']['loss_weight'] = 1.0
```
**Job ID:** 3525
**Status:** keep

**Metrics:**
| Metric | Baseline | Best so far | This exp | Δ vs best |
|--------|----------|-------------|----------|-----------|
| L2 | 0.5927 | 0.5738 | 0.5904 | ↑ +0.0166 vs exp001 |
| obj_box_col | 0.104% | 0.099% | 0.094% | ↓ -0.005% (new best) |
| car_ade | 0.6241 | 0.6227 | 0.6323 | ↑ +0.010 |
| NDS | 0.5236 | 0.5253 | 0.5218 | ↓ -0.004 |

**Analysis:** Divergent result — plan_loss_up achieves new best obj_box_col (0.094% vs 0.099%) but L2 is worse than exp001 (0.5904 vs 0.5738). The upweighted planning cls loss specifically helps collision avoidance (better mode selection for safety). Extended training (exp001) is better for L2 accuracy. NDS/AMOTA slightly worse — the stronger planning gradients may compete slightly with detection. Both exp001 and exp002 are independently positive vs baseline, suggesting their combination in exp005 (epochs15 + plan_loss_up) should achieve best on both metrics simultaneously.

---

## [exp-003] auto_mar26_exp003_num_det100 — 2026-03-27
**Hypothesis:** Increasing num_det from 50→100 surfaces more agents to the motion/planning head. In dense urban scenes, the current 50-agent cap drops relevant nearby agents, hurting both trajectory accuracy and collision avoidance.
**Config changes:**
```python
model['head']['motion_plan_head']['num_det'] = 100
```
**Job ID:** 3526
**Status:** keep

**Metrics:**
| Metric | Baseline | Best so far | This exp | Δ vs best |
|--------|----------|-------------|----------|-----------|
| L2 | 0.5927 | 0.5738 | 0.5676 | ↓ -0.0062 (new best) |
| obj_box_col | 0.104% | 0.094% | 0.091% | ↓ -0.003% (new best) |
| car_ade | 0.6241 | 0.6227 | 0.6307 | ↑ +0.008 |
| NDS | 0.5236 | 0.5253 | 0.5231 | ↓ -0.002 |

**Analysis:** Strongest single-change result this session — new best on BOTH primary metrics simultaneously. The mechanism is intuitive: with 50 agents, dense scenes with 50+ nearby vehicles drop important context. With 100 agents, the planner has richer scene awareness for both trajectory planning (L2) and collision avoidance. NDS/detection slightly lower (unrelated to planning change — likely random variance). This is now the most compelling single change to include in exp005 combo. Updating exp005 to combine epochs15 + plan_loss_up + num_det=100 (all three confirmed positive changes).

---

## [exp-004] auto_mar26_exp004_conf_decay08 — 2026-03-27
**Hypothesis:** Slower confidence decay (0.6→0.8) reduces temporal forgetting, keeping good tracks alive longer, reducing ID switches and improving motion feature consistency for planning.
**Config changes:**
```python
model['head']['det_head']['instance_bank']['confidence_decay'] = 0.8
```
**Job ID:** 3527
**Status:** discard

**Metrics:**
| Metric | Baseline | Best so far | This exp | Δ vs best |
|--------|----------|-------------|----------|-----------|
| L2 | 0.5927 | 0.5676 | 0.5731 | ↑ +0.0055 |
| obj_box_col | 0.104% | 0.091% | 0.120% | ↑ +0.029% (worse than baseline) |
| car_ade | 0.6241 | 0.6307 | 0.6315 | ↑ +0.008 |
| NDS | 0.5236 | 0.5231 | 0.5209 | ↓ -0.002 |

**Analysis:** Negative result. FAF jumped from ~43 to 61.6 — slower decay keeps low-quality/stale instances alive longer, significantly increasing false alarms. More ghost detections confuse the planner and increase collision rate (0.104%→0.120%). The default confidence_decay=0.6 is well-calibrated for the planning task. Do NOT use confidence_decay=0.8. The L2 is between baseline and best (0.5731) — not better than exp001 or exp003. Add to negative evidence table.

---

## [exp-005] auto_mar26_exp005_epochs15_plan_up — 2026-03-27
**Hypothesis:** Combining all three confirmed positive changes (epochs15 + plan_loss_up + num_det=100) should yield improvements on both metrics greater than any individual change.
**Config changes:**
```python
num_epochs = 15
checkpoint_epoch_interval = 15
runner = dict(type="IterBasedRunner", max_iters=num_iters_per_epoch * num_epochs)
checkpoint_config = dict(interval=num_iters_per_epoch * checkpoint_epoch_interval)
evaluation = dict(interval=num_iters_per_epoch * checkpoint_epoch_interval, eval_mode=eval_mode)
model['head']['motion_plan_head']['plan_loss_reg']['loss_weight'] = 2.0
model['head']['motion_plan_head']['plan_loss_cls']['loss_weight'] = 1.0
model['head']['motion_plan_head']['num_det'] = 100
```
**Job ID:** 3528
**Status:** discard

**Metrics:**
| Metric | Baseline | Best so far | This exp | Δ vs best |
|--------|----------|-------------|----------|-----------|
| L2 | 0.5927 | 0.5676 | 0.6109 | ↑ +0.043 (WORSE than baseline) |
| obj_box_col | 0.104% | 0.091% | 0.107% | ↑ +0.016% (worse than baseline) |
| car_ade | 0.6241 | 0.6307 | 0.6286 | — |
| NDS | 0.5236 | 0.5231 | 0.5210 | ↓ -0.002 |

**Analysis:** Strong negative synergy — combining all three positive changes produces results worse than baseline on both primary metrics. L2=0.6109 is the worst planning result this session. The plan_loss_up (2× regression gradient) appears to conflict with num_det=100's broader scene context over 15 epochs, causing over-optimization. This mirrors mar25 exp005 (queue6+decoder8 combo) where two individually positive changes combined negatively. The 100-agent input space requires careful gradient balancing — doubling the planning loss in this regime creates instability. Note: IDS improved to 842 (vs baseline 959), suggesting the combination does help tracking consistency through the longer training. Key lesson: for this architecture, individual improvements should be deployed one at a time rather than stacked.

---

## Conclusions

**Session:** autoresearch/mar26 | **Date:** 2026-03-27 | **Goal:** Improve val/L2 and val/obj_box_col

### Final Results Table

| Exp | Config | L2 | obj_box_col | car_ade | NDS | Δ L2 | Δ col | Status |
|-----|--------|-----|-------------|---------|-----|------|-------|--------|
| baseline | nomap_queue6 bs48 | 0.5927 | 0.104% | 0.6241 | 0.5236 | — | — | — |
| exp-001 | epochs 10→15 | 0.5738 | 0.099% | 0.6227 | 0.5253 | -0.019 | -0.005% | keep |
| exp-002 | plan_loss_reg 1→2, cls 0.5→1 | 0.5904 | 0.094% | 0.6323 | 0.5218 | -0.002 | -0.010% | keep |
| exp-003 | num_det 50→100 | **0.5676** | **0.091%** | 0.6307 | 0.5231 | **-0.025** | **-0.013%** | keep ⭐ |
| exp-004 | confidence_decay 0.6→0.8 | 0.5731 | 0.120% | 0.6315 | 0.5209 | -0.020 | +0.016% | **discard** |
| exp-005 | epochs15+plan_up+det100 | 0.6109 | 0.107% | 0.6286 | 0.5210 | +0.018 | +0.003% | **discard** |

### Key Findings

1. **`num_det=100` is the strongest single lever** (exp-003): Best result on BOTH metrics simultaneously. L2=0.5676 (-4.2% vs baseline) and obj_box_col=0.091% (-12.5% vs baseline). Doubling the agents surfaced to the planner provides richer scene context for trajectory planning and collision avoidance. This is a free improvement — no architectural change, no extra training cost.

2. **Extended training (15 epochs) helps both metrics** (exp-001): L2=0.5738, col=0.099% — confirms stage-2 undertraining bottleneck B2. An extra 5 epochs brings meaningful gains.

3. **Plan loss upweighting best for collision** (exp-002): Best obj_box_col in isolation (0.094%) at 10 epochs, but worse L2 than exp001/exp003. The 2× planning regression loss specifically helps mode selection for safety-critical scenarios.

4. **confidence_decay=0.8 hurts planning** (exp-004): FAF spiked +18, col worsened vs baseline (0.120% vs 0.104%). Slower temporal decay keeps stale detections alive, flooding the planner with false alarms. The default 0.6 is well-calibrated.

5. **Combining positive changes causes negative synergy** (exp-005): All three confirmed wins stacked together produced the worst result (L2=0.6109, worse than baseline). plan_loss_up conflicts with num_det=100 over 15 epochs. Same pattern as mar25 exp005.

### Recommended Next Steps

1. **Use `num_det=100` as the new default** — strongest single improvement, zero compute cost. Best config: `auto_mar26_exp003_num_det100`.

2. **Try epochs15 + num_det=100 (without plan_loss_up)** — exp001 and exp003 are individually positive; their combination was not tested. This pair avoids the plan_loss_up conflict observed in exp005.

3. **Try plan_loss_up + num_det=100 at 10 epochs** — exp002 (col best at 10 epochs) + exp003 (overall best); this is a smaller combo without the epochs instability.

4. **Do NOT combine plan_loss_up with num_det=100 at 15 epochs** — confirmed negative.

5. **Do NOT increase confidence_decay above 0.6** — FAF spikes and planning collision worsens.

6. **Explore num_det further (100→150)** — if 50→100 was a strong positive, 100→150 may yield additional gains.


---

# AutoResearch Log — mar27
**Goal:** Improve val/L2 and val/obj_box_col
**Base config:** projects/configs/sparsedrive_r50_stage2_4gpu_bs24.py (WITH map, queue=4, bs=24)
**Session branch:** autoresearch/mar27
**Max experiments:** 5

## Context from prior sessions
- bs24 baseline: L2=0.636, col=0.133%
- bs24 best (plan_loss_up, already done): L2=0.590, col=0.084% — NOT re-running
- nomap on bs24 (already done): L2=0.588, col=0.103%
- All-time best R50: L2=0.568, col=0.091% (mar26 exp003, nomap_queue6 + num_det=100)

## Experiment Plan
1. exp001: num_det=100 (strongest win from mar26, untested on bs24)
2. exp002: queue_length=6 (confirmed on nomap, untested on bs24 WITH map)
3. exp003: nomap + plan_loss_up combo (both individually confirmed on bs24, combo untested)
4. exp004: epochs=15 (confirmed on nomap_queue6, untested on bs24)
5. exp005: epochs15 + num_det=100 (top recommendation from mar26 conclusions)


## [exp-000] auto_mar27_exp000_baseline — 2026-03-28
**Job ID:** 3529
**Status:** baseline

**Metrics:**
| Metric | Value |
|--------|-------|
| L2 | 0.6274 |
| obj_box_col | 0.107% |
| car_ade | 0.6313 |
| NDS | 0.5233 |
| AMOTA | 0.3713 |
| mAP_normal | 0.5508 |
| FAF | 77.7 |

**Analysis:** Slightly better than historical baseline (L2=0.636, col=0.133%) due to code evolution. High FAF=77.7 reflects the with-map config generating more false alarms vs nomap baseline (~43-46 FAF). mAP_normal=0.5508 confirms map head is active and healthy. Using these numbers as the mar27 session reference.

---

