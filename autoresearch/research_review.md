# SparseDrive Research Review: Comprehensive Analysis and Improvement Roadmap

> Generated: 2026-03-25. Experimental findings updated: 2026-03-30.

---

## 1. Method Understanding

### What SparseDrive Does

SparseDrive is an end-to-end camera-only autonomous driving system that jointly performs 3D object detection, online HD map construction, multi-agent motion prediction, and ego-vehicle planning in a unified sparse-representation framework.

The core idea is to represent the scene as a sparse set of *instance queries* — each query is an anchor (learned position/size/heading priors, kmeans-initialized from the dataset) plus a 256-dim feature vector. These queries are updated through iterative deformable cross-attention to image features and self-attention among instances. This is far more efficient than dense BEV methods because only ~900 object anchors and 100 map anchors need to be maintained per frame.

### How the Three Tasks Interact

```
Image → ResNet50 → FPN (4 levels, 256-ch)
                    ↓
            ┌───── det_head (Sparse4DHead) ──────┐
            │     900 anchors, 6 decoders          │
            │     InstanceBank (600 temp cache)     │
            └──────→ det_output ──────────────────┐│
                    ↓                              ││
            ┌───── map_head (Sparse4DHead) ─────┐ ││
            │     100 anchors, 6 decoders         │ ││
            │     InstanceBank (33 temp cache)    │ ││
            └──────→ map_output ─────────────────│┘│
                                                 ↓  │
            ┌──── motion_plan_head ─────────────┐   │
            │    InstanceQueue (4 frames)         │◄─┘
            │    topk=50 agents, topk=10 map      │
            │    6 motion modes × 12 timesteps    │
            │    6 plan modes × 6 timesteps        │
            └──────────────────────────────────────┘
```

- **Stage 1** trains only det_head + map_head for 100 epochs. The instance bank's temporal cache trains the model to associate instances across frames and build spatially consistent features.
- **Stage 2** loads the stage-1 checkpoint and adds motion_plan_head for 10 epochs. The motion head uses det_output's instance features and indices (from the detection Hungarian matching) to supervise trajectory prediction. The planner reads ego status and map context.

### Most Important Losses

| Task | Loss | Weight | Comment |
|------|------|--------|---------|
| Det cls | FocalLoss (γ=2, α=0.25) | 2.0 | Dominant |
| Det reg | L1 (SparseBox3DLoss) | 0.25 | Balanced with aux |
| Depth aux | L1 | 0.2 | Auxiliary supervision only |
| Motion cls | FocalLoss | 0.2 | Low |
| Motion reg | L1 | 0.2 | Low |
| Plan cls | FocalLoss | 0.5 | Medium |
| Plan reg | L1 | 1.0 | Dominant for planning |
| Plan status | L1 | 1.0 | Ego kinematics |

### Likely Bottlenecks a Priori

1. **Recall**: ~0.5 across classes — half of GT objects are never detected. Undetected agents create invisible collision risks.
2. **Stage-2 duration**: 10 epochs is very short for the motion+planning sub-network.
3. **Trailer class**: AMOTA ≈ 0 — essentially complete failure on a major vehicle class.
4. **Denoising disabled**: `num_dn_groups=0` — known source of detection quality improvement.
5. **Motion loss scale**: motion cls/reg = 0.2 is very low relative to detection — likely undertrained.

---

## 2. Current Experiment Landscape

### Reproduced Baseline Results

**Config**: `sparsedrive_r50_stage2_4gpu_bs24.py` — 4 GPU, bs=24, lr=1.5e-4, backbone lr_mult=0.1, 10 epochs, loaded from `ckpt/sparsedrive_stage1.pth`. **Date run: 2026-02-16.**

Note: this config has `rot3d_range=[0,0]` (rotation augmentation disabled in data_aug_conf despite BBoxRotation appearing in the pipeline).

#### Detection (NuScenes)

| Metric | Value |
|--------|-------|
| NDS | **0.5232** |
| mAP | **0.4132** |
| mATE | 0.5537 |
| mASE | 0.2749 |
| mAOE | 0.5437 |
| mAVE | 0.2702 |
| mAAE | 0.1917 |

Per-class AP highlights:
- Car: AP@0.5=0.353, AP@4.0=0.832
- Trailer: AP@0.5=0.000, AP@1.0=0.030, AP@2.0=0.122, AP@4.0=0.251 — effectively non-functional at tight thresholds
- Construction vehicle: AP@0.5=0.000, AP@1.0=0.021 — similarly broken
- Traffic cone: AP@0.5=0.520, AP@4.0=0.790 — strongest class

#### Tracking

| Class | AMOTA | AMOTP | Recall | FAF |
|-------|-------|-------|--------|-----|
| Car | 0.617 | 0.875 | 0.670 | 125 |
| Pedestrian | 0.454 | 1.206 | 0.574 | 69 |
| Motorcycle | 0.408 | 1.245 | 0.506 | 15 |
| Bicycle | 0.406 | 1.199 | 0.540 | 24 |
| Truck | 0.366 | 1.262 | 0.514 | 46 |
| Bus | 0.391 | 1.307 | 0.517 | 20 |
| **Trailer** | **0.001** | 1.662 | **0.220** | **47** |
| **Overall** | **0.3776** | **1.2509** | **0.5058** | 49.4 |

Other tracking stats: IDS=1045, FRAG=626, MT=2488, ML=2261, MOTA=0.346, MOTP=0.629

#### HD Map

| Category | AP@0.5 | AP@1.0 | AP@1.5 | AP (avg) |
|----------|--------|--------|--------|----------|
| ped_crossing | 0.185 | 0.552 | 0.726 | 0.488 |
| divider | 0.355 | 0.632 | 0.752 | 0.580 |
| boundary | 0.290 | 0.674 | 0.810 | 0.591 |
| **mAP** | | | | **0.553** |

#### Motion Prediction

| Class | EPA | min_ADE | min_FDE | Miss Rate |
|-------|-----|---------|---------|-----------|
| Car | 0.492 | 0.636 | 1.000 | 0.133 |
| Pedestrian | 0.411 | 0.728 | 1.066 | 0.147 |

#### Planning

| Metric | 0.5s | 1.0s | 1.5s | 2.0s | 2.5s | 3.0s | **Avg** |
|--------|------|------|------|------|------|------|---------|
| obj_col | 0.742% | 0.713% | 0.684% | 0.659% | 0.649% | 0.638% | **0.670%** |
| obj_box_col | 0.000% | 0.020% | 0.052% | 0.098% | 0.168% | 0.283% | **0.133%** |
| L2 (m) | 0.197 | 0.310 | 0.444 | 0.603 | 0.787 | 0.995 | **0.636** |

**Reported paper numbers (SparseDrive-S, R50) for reference:**
- NDS: 0.5257, Planning collision: 0.097%

> The reproduced baseline is closely aligned with the paper for NDS (0.5232 vs 0.5257) and within expected variance. The `obj_box_col` of 0.133% is higher than the paper-reported 0.097%, which may reflect differences in evaluation protocol (occluded object handling) or the specific checkpoint used.

### Key Observations from Reproduced Results

- **obj_col = 0.670%**: This is the rate at which the *GT ego trajectory* itself collides with GT objects — it reflects unavoidable ground-truth collisions in the dataset, not model error. It serves as a lower bound on `obj_box_col`.
- **obj_box_col = 0.133%**: The predicted trajectory collision rate *beyond* what GT already collides with. This is the actual plannable gap — the paper claims 0.097%, so there is ~0.036pp gap worth addressing.
- **Trailer mAOE = 0.668**: Orientation error for trailers is very high, consistent with the `cls_allow_reverse` fix being needed.
- **Motorcycle/Bicycle mAOE ≈ 0.99–0.72**: Heading estimation is very poor for two-wheeled vehicles — worse than random for bicycles.
- **Motion car FDE = 1.000m**: Precisely 1.0 is suspicious — worth checking if this is a computation artifact or a mode-collapse symptom.

### Apollo 8GPU Baseline Results

**Config**: `sparsedrive_r50_stage1_8gpu_noflash.py` → `sparsedrive_r50_stage2_8gpu_noflash.py`. 8 GPU, bs=48, lr=3e-4, all MultiheadAttention (no flash), num_dn_groups=0. **Date run: 2026-02-13/15.**

| Metric | Value | vs. DGX bs24 |
|--------|-------|--------------|
| NDS | **0.5187** | -0.0045 |
| mAP | **0.4076** | -0.0056 |
| map mAP | **0.5471** | -0.0057 |
| AMOTA | **0.3714** | -0.0062 |
| IDS | **1088** | +43 |
| car ADE | **0.6148** | -0.0212 |
| car FDE | **0.9642** | -0.0357 |
| Planning L2 (avg) | **0.600** | -0.036 |
| obj_box_col | **0.104%** | -0.029pp |

> The Apollo 8GPU baseline is slightly weaker than the DGX bs24 baseline across all metrics. Likely causes: no FlashAttention vs Flash-enabled stage2, or minor config differences. Both are used as references depending on what an experiment was forked from.

---

## 3. Experimental Findings and Analysis

> Last updated: 2026-03-26. All results are single-seed val-set evaluations.

### 3.1 Two Baselines

All experiments fork from one of two baselines:
- **DGX 4GPU** (`stage2_4gpu_bs24`): NDS=0.5232, AMOTA=0.3776, map=0.5528, L2=0.636, obj_box_col=0.133%
- **Apollo 8GPU** (`stage2_8gpu_noflash`): NDS=0.5187, AMOTA=0.3714, map=0.5471, L2=0.600, obj_box_col=0.104%

The 4GPU bs=12/GPU config (`stage2_4gpu`) is a **known failure mode**: the map head fails to converge at per-GPU BS=12, yielding map mAP=0.078 vs 0.553 at per-GPU BS=6. Root cause under investigation (see Section 3.13). All 4GPU experiments must use bs=24 total until resolved.

---

### 3.2 Stage1 Detection Ablations (Apollo 8GPU)

100-epoch stage1 evaluations covering DN, nomap, and rotation augmentation.

| Config | NDS | mAP | AMOTA | IDS | Notes |
|--------|-----|-----|-------|-----|-------|
| **8gpu_noflash (baseline)** | 0.5307 | 0.4137 | 0.3973 | 535 | — |
| +DN | 0.5368 | 0.4227 | 0.4198 | 416 | +0.009 mAP, -119 IDS |
| +Nomap | 0.5415 | 0.4286 | 0.4165 | 614 | map head removed |
| +Nomap+DN | 0.5507 | 0.4365 | 0.4441 | 486 | best det w/ standard aug |
| +Nomap+Rotaug | 0.5583 | 0.4527 | 0.4351 | 516 | rotaug alone is strong |
| **+Nomap+DN+Rotaug** | **0.5620** | **0.4571** | **0.4534** | **464** | best stage1 config |

**Findings:**
- DN alone delivers a clean +0.009 mAP and cuts IDS by 22%.
- Removing map in stage1 (nomap) improves detection — less competing gradient.
- Rotation augmentation has the largest single-factor impact on detection (nomap+rotaug > nomap+DN).
- All three combined achieve the best detection checkpoint to date.

---

### 3.3 Stage2 with DN Pretrain (Apollo 8GPU)

**Config**: `stage2_8gpu_pretrainv2_noflash` — loads stage1_dn checkpoint, stage2 with `with_map=False`.

| Metric | Apollo baseline | DN stage2 | Δ |
|--------|----------------|-----------|---|
| NDS | 0.5187 | **0.5415** | +0.023 |
| mAP | 0.4076 | **0.4257** | +0.018 |
| AMOTA | 0.3714 | **0.4179** | +0.046 |
| IDS | 1088 | **425** | −663 |
| car ADE | 0.6148 | **0.6166** | +0.002 |
| Planning L2 | 0.600 | **0.602** | +0.002 |
| obj_box_col | 0.104% | **0.092%** | −0.012pp |

> DN training is a net positive: large detection and tracking improvements, slight collision improvement. Planning L2 is essentially unchanged. The stage2 uses `with_map=False` (no map head in stage2), which avoids the map sensitivity issue.

**Also evaluated** `stage2_4gpu_bs24_pt2` (DN stage1 → 4GPU bs24 stage2 with map):
NDS=0.5332, AMOTA=0.4180, IDS=520, L2=0.700, obj_box_col=0.133% — significantly worse planning (L2=0.700 vs 0.602), suggesting the map head in stage2 hurts planning when combined with DN pretrain.

---

### 3.4 Map Removal Experiments

**Condition A: Remove map from BOTH stages** (`stage2_8gpu_pretrainv1_noflash_nomap`)
Loads from stage1_nomap_dn_rotaug (no map ever trained):

| Metric | Apollo baseline | Nomap both stages | Δ |
|--------|----------------|-------------------|---|
| NDS | 0.5187 | **0.5593** | +0.041 |
| mAP | 0.4076 | **0.4550** | +0.047 |
| AMOTA | 0.3714 | **0.4535** | +0.082 |
| IDS | 1088 | **396** | −692 |
| car ADE | 0.6148 | **3.863** | **+3.25 (BROKEN)** |
| Planning L2 | 0.600 | **6.612** | **+6.01 (BROKEN)** |
| obj_box_col | 0.104% | **3.605%** | **+3.5pp (BROKEN)** |

**Detection dramatically improves when map is absent from both stages (no competing gradients). But motion and planning are completely non-functional.** The stage1 map training is not optional — it provides geometric representations that the planning head critically depends on.

**Condition B: Remove map from stage2 only** (standard stage1, `stage2_4gpu_nomap` / `stage2_4gpu_bs24_nomap`):

| Metric | DGX baseline | Nomap stage2 (bs=12) | Nomap stage2 (bs24) |
|--------|-------------|---------------------|---------------------|
| NDS | 0.5232 | 0.5217 | **0.5262** |
| mAP | 0.4132 | 0.4131 | **0.4154** |
| AMOTA | 0.3776 | **0.3787** | 0.3745 |
| IDS | 1045 | **785** | 853 |
| Planning L2 | 0.636 | **0.591** | **0.588** |
| obj_box_col | 0.133% | **0.080%** | **0.103%** |

**Removing map only from stage2 is beneficial for all tasks.** Detection/AMOTA holds, planning L2 improves by ~7%, collision rate improves. Hypothesis: the map head in stage2 competes for capacity with motion/planning gradients, but the map features from stage1 survive in shared instance representations.

---

### 3.5 GT Perception Oracle (Upper Bound)

**Config**: `stage2_4gpu_gtdetmap` — GTSparseDriveHead feeds ground-truth boxes and map to the motion/planning head.

| Metric | DGX baseline | GT perception | Δ |
|--------|-------------|---------------|---|
| car ADE | 0.636 | **0.378** | **−40%** |
| car FDE | 1.000 | **0.780** | −22% |
| Planning L2 | 0.636 | **0.651** | +0.015 (worse) |
| obj_box_col | 0.133% | **0.092%** | −0.041pp |

**Key finding: GT perception dramatically improves motion prediction (−40% ADE) but planning L2 does not improve — it slightly degrades.** This is the GT perception paradox:
1. The motion head can exploit perfect perception to predict agent trajectories much more accurately.
2. The planning head, however, does not automatically benefit — the planner's L2 error is not bottlenecked by detection quality alone.
3. obj_box_col does improve with GT perception (the perfect agent positions help collision avoidance), but not as much as the motion improvement would suggest.

**Implication**: Improving detection alone will not fix planning. The planning head needs its own improvements (better training signal, more training, or explicit safety objectives).

---

### 3.6 Prediction Pretraining Experiments (dual_head branch)

Both pretrainv3 (4GPU DGX) and pretrainv4 (8GPU Apollo) experiments attempted to pretrain the prediction/planning head before full stage2.

| Metric | DGX baseline | pretrainv3 (4GPU) | pretrainv4 (8GPU) |
|--------|-------------|-------------------|-------------------|
| NDS | 0.5232 | 0.4997 | 0.5128 |
| AMOTA | 0.3776 | 0.3129 | 0.3611 |
| IDS | 1045 | 1702 | 1035 |
| Planning L2 | 0.636 | 0.781 | 0.713 |
| obj_box_col | 0.133% | 0.189% | 0.163% |
| map mAP | 0.5528 | **0.065** | 0.554 |

**Both pretraining approaches degraded all metrics below baseline.** Pretrainv3 almost entirely broke the map head (map_mAP=0.065). Pretrainv4 is better but still worse than baseline on every metric. The pretraining strategy (likely separate head training on prediction before joint fine-tuning) introduced optimization conflicts that hurt the overall system.

---

### 3.7 R101 Backbone

**Config**: `stage2_4gpu` with ResNet101 (input 1408×512, nuim cascade-RCNN pretrained).

| Metric | R50 baseline | R101 stage2 | R101+nomap | Δ (R101 vs R50) |
|--------|-------------|-------------|------------|-----------------|
| NDS | 0.5232 | **0.5857** | **0.5936** | +0.063 |
| mAP | 0.4132 | **0.4954** | **0.4989** | +0.082 |
| map mAP | 0.5528 | **0.5506** | — | −0.002 |
| AMOTA | 0.3776 | **0.5020** | **0.5032** | +0.125 |
| IDS | 1045 | **586** | **651** | −459 |
| car ADE | 0.636 | **0.604** | **0.596** | −0.040 |
| Planning L2 | 0.636 | **0.598** | **0.592** | −0.044 |
| obj_box_col | 0.133% | **0.081%** | **0.103%** | −0.052pp |

**R101 delivers very large improvements across the board.** The backbone upgrade from R50→R101 is the single largest lever available: +0.063 NDS, +0.082 mAP, +0.125 AMOTA, −40% collision rate. The nomap variant of R101 further improves most metrics.

Stage1 ablations for R101:
- R101 base stage1: NDS=0.5855, mAP=0.4893, AMOTA=0.4979
- R101+DN stage1: NDS=0.5971, mAP=0.5038, AMOTA=0.5396, IDS=480

---

### 3.8 Occlusion Detection Experiments

Multiple variants trained/evaluated with occluded-object detection (occptrainval, occfheval, occfleval, etc.):
- The `occluded_det` metrics consistently show near-zero or very poor detection quality for occluded objects.
- The fully-hidden (occfh) category is essentially undetectable with the current model.
- Partially-occluded (occp) evaluation shows only marginal improvement over zero.

**Conclusion**: Current SparseDrive cannot detect objects that are not visible in any camera. The occluded detection task may require explicit memory/prediction mechanisms beyond the current temporal cache. The evaluation implementation may also have issues (inconsistent with training supervision).

---

### 3.9 Temporal Motion and Task Ablations (4GPU DGX)

| Config | NDS | mAP | AMOTA | IDS | L2 | obj_box_col |
|--------|-----|-----|-------|-----|-----|-------------|
| **bs24 (baseline)** | 0.5232 | 0.4132 | 0.3776 | 1045 | 0.636 | 0.133% |
| nomap | 0.5217 | 0.4131 | 0.3787 | 785 | 0.591 | 0.080% |
| bs24_nomap | 0.5262 | 0.4154 | 0.3745 | 853 | 0.588 | 0.103% |
| nomap_rotaug | 0.5234 | 0.4181 | 0.3676 | 882 | 0.621 | 0.160% |
| notempmotion | 0.5166 | 0.4092 | 0.3766 | 846 | 0.825 | 0.220% |
| nomap_notempmotion | 0.5239 | 0.4143 | 0.3772 | 638 | 0.719 | 0.160% |
| noplan | 0.5204 | 0.4089 | 0.3717 | 700 | — | — |
| nomap_noplan | 0.5232 | 0.4123 | 0.3698 | 1205 | — | — |
| bs24_nomap_sephead | 0.5224 | 0.4161 | 0.3670 | 864 | 0.629 | 0.200% |
| maplrdiv4 | 0.5231 | 0.4134 | 0.3658 | 724 | 0.587 | 0.120% |

**Key observations:**
- **Temporal motion features are critical**: `notempmotion` degrades planning L2 by 30% (0.636→0.825) and collision rate nearly doubles. Temporal context from prior frames is essential for planning.
- **Removing planning doesn't improve detection**: noplan has slightly worse NDS, ruling out planning gradient interference.
- **Rotation augmentation in stage2 doesn't help**: `nomap_rotaug` has similar or worse L2/collision vs `nomap` without rotaug. Rotaug is beneficial in stage1 but may interfere with stage2 optimization.
- **Separate head (sephead) degrades performance**: slightly worse across all metrics.
- **Map LR/4 kills map**: `maplrdiv4` achieves map_mAP=0.074 (vs 0.553 baseline) — map head is highly sensitive to its own learning rate.

---

### 3.10 Auto-Research Experiments — mar25 (March 2026)

Base config: `sparsedrive_r50_stage2_4gpu_nomap.py` (queue=4, bs=48). Baseline: L2=0.5911, col=0.080%.

| Config | NDS | AMOTA | L2 | obj_box_col | Status |
|--------|-----|-------|----|-------------|--------|
| nomap bs48 (baseline) | 0.5239 | 0.3741 | 0.5911 | 0.080% | baseline |
| exp001: plan_loss_reg 1→2, cls 0.5→1 | 0.5239 | 0.3741 | 0.5902 | 0.084% | keep |
| exp002: motion_loss 0.2→0.5 | 0.5158 | — | 0.6358 | 0.148% | discard |
| exp003: queue_length 4→6 | 0.5238 | — | **0.5757** | 0.100% | keep ⭐ |
| exp004: num_decoder 6→8 | 0.5239 | — | 0.5955 | **0.074%** | keep ⭐ |
| exp005: queue6 + decoder8 | 0.5241 | — | 0.5934 | 0.126% | discard |

**Mar25 key findings:** queue=6 is best for L2; decoder=8 is best for col; combinations negative at 10 epochs; motion_loss >0.2 kills both metrics.

---

### 3.11 Auto-Research Experiments — mar26 (March 2026)

Base config: `sparsedrive_r50_stage2_4gpu_nomap_queue6.py` (queue=6 already baked in, bs=48). Baseline: L2=0.5927, col=0.104%.

| Config | NDS | AMOTA | L2 | obj_box_col | IDS | Status |
|--------|-----|-------|----|-------------|-----|--------|
| nomap_queue6 (baseline) | 0.5236 | 0.3878 | 0.5927 | 0.104% | 959 | baseline |
| exp001: epochs 10→15 | 0.5253 | 0.3791 | 0.5738 | 0.099% | 960 | keep |
| exp002: plan_loss_reg 1→2, cls 0.5→1 | 0.5218 | 0.3691 | 0.5904 | **0.094%** | 1087 | keep |
| **exp003: num_det 50→100** | 0.5231 | 0.3712 | **0.5676** | **0.091%** | 1086 | **keep ⭐** |
| exp004: confidence_decay 0.6→0.8 | 0.5209 | 0.3725 | 0.5731 | 0.120% | 1018 | discard |
| exp005: epochs15+plan_up+det100 | 0.5210 | 0.3765 | 0.6109 | 0.107% | 842 | discard |

**Mar26 key findings:**
- `num_det=100` is the best single change: improves both L2 (-4.2%) and col (-12.5%) vs baseline simultaneously at zero compute cost.
- Extended training (15 epochs) helps both metrics individually.
- Plan loss upweighting best for col in isolation but not L2.
- confidence_decay=0.8 causes FAF spike (+18) and collision regression — default 0.6 is optimal.
- Combining all three positive changes (exp005) causes negative synergy — same pattern as mar25 exp005.

---

### 3.12 Summary Table

| Experiment | NDS | mAP | AMOTA | IDS | L2 | obj_box_col |
|-----------|-----|-----|-------|-----|-----|-------------|
| DGX bs24 (baseline) | 0.5232 | 0.4132 | 0.3776 | 1045 | 0.636 | 0.133% |
| Apollo 8GPU (baseline) | 0.5187 | 0.4076 | 0.3714 | 1088 | 0.600 | 0.104% |
| DN stage2 (Apollo 8GPU) | 0.5415 | 0.4257 | 0.4179 | 425 | 0.602 | 0.092% |
| Nomap stage2 (DGX bs24) | 0.5262 | 0.4154 | 0.3745 | 853 | 0.588 | 0.103% |
| Nomap both stages | 0.5593 | 0.4550 | 0.4535 | 396 | 6.612 | 3.605% |
| GT perception oracle | — | — | — | — | 0.651 | 0.092% |
| Pretrainv3 (4GPU) | 0.4997 | 0.3841 | 0.3129 | 1702 | 0.781 | 0.189% |
| Pretrainv4 (8GPU) | 0.5128 | 0.4047 | 0.3611 | 1035 | 0.713 | 0.163% |
| R101 stage2 (4GPU) | 0.5857 | 0.4954 | 0.5020 | 586 | 0.598 | 0.081% |
| R101+nomap (4GPU) | 0.5936 | 0.4989 | 0.5032 | 651 | 0.592 | 0.103% |
| mar25 exp003 queue=6 | 0.5238 | 0.4131 | — | — | 0.576 | 0.100% |
| mar25 exp004 decoder=8 | 0.5239 | 0.4131 | — | — | 0.596 | **0.074%** |
| **mar26 exp003 num_det=100** | 0.5231 | 0.4111 | 0.3712 | 1086 | **0.568** | **0.091%** |
| mar27 baseline (bs24+map+queue4) | 0.5233 | 0.4133 | 0.3713 | 990 | 0.627 | 0.107% |
| mar27 exp001 num_det=100 | 0.5258 | 0.4133 | 0.3751 | 577 | 0.616 | **0.091%** |
| mar27 exp002 queue=6 | 0.5262 | 0.4133 | 0.3751 | 995 | 0.623 | 0.147% |
| mar27 exp003 nomap+planup | — | — | — | — | crash | crash (DDP incompatibility) |
| mar27 exp003b plan_loss_up | 0.5271 | 0.4171 | 0.3836 | 691 | 0.622 | 0.110% |
| mar27 exp004 epochs=15 | 0.5255 | 0.4125 | 0.3829 | 778 | 0.635 | 0.143% |
| mar27 exp005 det100+planup | 0.5291 | 0.4139 | 0.3747 | 933 | 0.650 | 0.125% |

### 3.13 Map Head Failure at Per-GPU Batch Size > 6 (Investigation — April 2026)

#### Observed Problem

The `sparsedrive_r50_stage2_4gpu.py` config (total_batch_size=48, 4 GPUs → **12 samples/GPU**, lr=3e-4) produces catastrophically poor map performance despite detection performing normally. The `sparsedrive_r50_stage2_4gpu_bs24.py` config (total_batch_size=24, 4 GPUs → **6 samples/GPU**, lr=1.5e-4) works correctly.

| Config | per-GPU BS | LR | ped_crossing | divider | boundary | map mAP | map_loss_line_5 (end) |
|--------|-----------|-----|-------------|---------|----------|---------|----------------------|
| `4gpu` (bs48) | **12** | 3e-4 | 0.033 | 0.085 | 0.115 | **0.078** | ~0.70 (not converged) |
| `4gpu_maplrdiv4` (bs48, loss/4) | **12** | 3e-4 | 0.025 | 0.087 | 0.110 | **0.074** | — |
| `4gpu_bs24` | **6** | 1.5e-4 | 0.488 | 0.580 | 0.591 | **0.553** | ~0.10 (converged) |
| `8gpu_noflash` | **6** | 3e-4 | 0.483 | 0.584 | 0.575 | **0.547** | — |

The training losses confirm non-convergence: at iteration 51, `map_loss_line_0` is 0.81 for bs48 vs 0.26 for bs24, and it barely decreases throughout all 5860 iterations of training. Detection loss converges normally in both cases (det mAP ~0.41 for all).

#### Isolation of the Variable

The 8GPU config (total_batch_size=48, **6 samples/GPU**, lr=3e-4) achieves map mAP=0.547 — good performance with the **same total batch size and same LR** as the failing 4GPU config. This rules out total batch size and learning rate as root causes.

The only consistent predictor across all configs is **per-GPU batch size**:
- per-GPU BS = 6 → map works (4gpu_bs24, 8gpu_noflash)
- per-GPU BS = 12 → map fails (4gpu bs48, maplrdiv4)
- per-GPU BS = 16 → map also fails (stage1 4gpu, per user observation)

#### What Was Ruled Out

1. **LR too high**: The 8gpu config uses lr=3e-4 (same as failing 4gpu) and works fine.
2. **Total batch size**: The 8gpu config has total_batch_size=48 (same as failing 4gpu) and works fine.
3. **Map loss weight too high**: `maplrdiv4` divided both map loss weights by 4 — no improvement (0.074 vs 0.078).
4. **Temporal instance bank gradient flow**: `cache()` explicitly detaches all features before storing (`instance_feature = instance_feature.detach()`), so no gradients flow across the temporal boundary regardless of `feat_grad`.
5. **`feat_grad=True` on temporal features**: With `num_temp_instances=0` in stage1 (no temporal caching), `cache()` returns early — no temporal mechanism exists at all. Yet the issue still appears in stage1 at bs=16/GPU. This fully rules out temporal feature gradient accumulation as the cause.
6. **`feat_grad` itself as a temporal mechanism**: Code inspection confirmed `feat_grad` only controls whether `self.instance_feature` (a single `[num_anchor, embed_dims]` nn.Parameter used to initialize all queries) is trainable. It has no connection to temporal caching.

#### Most Likely Root Cause: BN Statistics

With `norm_eval=False` and standard (non-sync) BN in the backbone, each GPU independently estimates batch normalization running statistics from its local batch. Per-GPU BN statistics shift the backbone feature distribution when batch size changes from 6 to 12 samples per GPU.

The detection head is robust to this shift because it uses many redundant 3D anchors spread across diverse spatial locations (`feat_grad=False`, no sensitivity to feature initialization). The map head is uniquely sensitive because:
- Its deformable keypoints are all constrained to a **fixed ground plane** (`ground_height=-1.84023`) via `SparsePoint3DKeyPointsGenerator` — systematic BN-induced feature degradation at ground-plane projected coordinates collapses all 100 anchors simultaneously
- `feat_grad=True` means the shared anchor initialization parameter receives gradient updates, which may amplify BN-induced feature drift

#### Pending Experiments (running as of 2026-04-01)

Four ablation configs, all based on `sparsedrive_r50_stage2_4gpu.py` (bs48, 4GPU, per-GPU BS=12):

| Config | Change | Tests | Confidence |
|--------|--------|-------|------------|
| `4gpu_gradacc` | `cumulative_iters=2` | Directly replicates per-GPU BS=6 for both BN stats and gradient updates simultaneously | **Highest** |
| `4gpu_normeval` | `norm_eval=True` in backbone | Freezes BN running stats, isolates whether BN stats are the root cause | High |
| `4gpu_mapfeatnograd` | `feat_grad=False` in map instance bank | Freezes map anchor init parameter, tests whether its gradient updates destabilize training | Medium |

If `gradacc` works → per-GPU batch size confirmed as root cause, `cumulative_iters=2` is a practical fix.
If `normeval` works but `gradacc` doesn't → BN statistics are the cause; production fix is SyncBN.
If `mapfeatnograd` works but `normeval` doesn't → the shared anchor init parameter is being destabilized by conflicting gradients from diverse scenes at larger batch sizes.

#### Actionable Fix Once Confirmed

- **Short-term**: Use `cumulative_iters=2` to run 4GPU bs48 without map degradation, or keep using 4GPU bs24.
- **Proper fix if BN confirmed**: Replace `norm_cfg=dict(type="BN")` with `norm_cfg=dict(type="SyncBN")` in the backbone — this pools BN statistics across all GPUs (effective batch = 4×12=48), making per-GPU batch size irrelevant to BN.

---

### 3.14 Auto-Research Experiments — mar27 (March 2026)

**Base config:** `sparsedrive_r50_stage2_4gpu_bs24.py` (WITH map, queue=4, bs=24, lr=1.5e-4)
**Goal:** Improve L2 and obj_box_col on the bs24 with-map base config.
**Baseline:** L2=0.627, col=0.107%, IDS=990, FAF=77.7, mAP_normal=0.5508

| Exp | Change | L2 | col% | IDS | FAF | Status |
|-----|--------|-----|------|-----|-----|--------|
| exp001 | num_det 50→100 | **0.616** | **0.091%** | **577** | **44.8** | keep (best) |
| exp002 | queue 4→6 | 0.623 | 0.147% | 995 | 74.4 | discard |
| exp003 | nomap+planup | — | — | — | — | crash×3 (DDP) |
| exp003b | plan_loss_up only | 0.622 | 0.110% | 691 | 46.3 | discard |
| exp004 | epochs 10→15 | 0.635 | 0.143% | 778 | 45.2 | discard |
| exp005 | det100+planup | 0.650 | 0.125% | 933 | 69.9 | discard |

**Key takeaways:**
- **num_det=100 is the only reliable win** on bs24 with-map: all other changes hurt or are neutral.
- **with_map=False via runtime override is DDP-incompatible** in PyTorch 1.13 (map head weights remain registered). Requires a base config built without map head.
- **bs24 with-map is brittle**: map head gradient competition leaves no headroom for additional changes beyond num_det=100. queue=6, epochs=15, plan_loss_up, and combinations all degrade both metrics.
- **Negative synergy dominates**: even combining num_det=100 + plan_loss_up (both confirmed individual wins) produces the worst result (L2=0.650, IDS=933 — nearly back to baseline levels).
- **Config-specific findings**: epochs=15 helps on nomap (mar26 ✓) but hurts on with-map (mar27 ✗). Same for queue=6. The map head is the source of brittleness.

---

## 4. Likely Bottlenecks and Failure Modes

### B1. Trailer Catastrophic Failure (AMOTA ≈ 0)

**Evidence**: AMOTA = 0.001 (reproduced baseline), AP@0.5 = 0.000, AP@1.0 = 0.030. Recall = 0.220, FAF = 47. Construction vehicle is equally broken (AP@0.5 = 0.000).

**Root cause hypothesis**: Trailers are rare (2425 GT instances vs 58317 cars), elongated (typical 14m+), often partially occluded behind trucks, and lack a good kmeans anchor cluster given their rarity. The uniform 55m circular distance filter applies to all classes including trailers that are typically stationary or slow. The Hungarian assignment may consistently prefer assigning trailer anchors to background or short objects due to the large positional error. Additionally, the `cls_allow_reverse` hack for barriers suggests the system has known orientation issues for elongated objects — trailers likely suffer from the same problem.

**Impact on planning**: Undetected trailers are large stationary obstacles. Missing them is a major collision risk.

### B2. Stage-2 Under-Training (10 epochs)

**Evidence**: Stage 2 trains motion+planning for only 10 epochs. With `num_iters_per_epoch ≈ 586` iterations, stage 2 is only ~5,860 optimizer steps. The `exp001_plan_loss_up` experiment (simply upweighting planning loss) improved planning L2 from 0.636→0.590 and collision from 0.133%→0.084% without any architectural change — confirming the model is undertrained on planning.

**File**: `projects/configs/sparsedrive_small_stage2.py` — `num_epochs = 10`.

**Impact**: Motion and planning are undertrained. Upweighting planning loss is the quickest lever confirmed empirically.

### B3. Denoising Training Disabled in Baseline

**Evidence (empirical)**: Adding DN in stage1 (`stage1_8gpu_noflash_dn`) delivers +0.009 mAP, +0.022 AMOTA, −22% IDS vs. no-DN baseline. The full `stage2_8gpu_pretrainv2_noflash` (DN stage1 + stage2) achieves NDS=0.5415, AMOTA=0.4179, IDS=425, obj_box_col=0.092% — improvements on all metrics vs. baseline. DN is **confirmed as a net positive with no downside**.

**File**: `projects/mmdet3d_plugin/models/detection3d/target.py` — `get_dn_anchors()` and `update_dn()` are fully implemented.

### B4. Planning Head Bottlenecked Independently of Detection

**Evidence (empirical)**: The GT perception oracle shows that even with perfect detection, planning L2 does not improve (0.636→0.651 — slightly worse). The planner is not primarily bottlenecked by detection quality. This means improvements to detection alone will not fix planning.

**Implication**: Planning needs direct improvements: better loss formulation, more training, explicit safety objectives, or architectural changes to how the planner uses agent features.

### B4b. Motion Loss Scale Too Low

**Evidence**: motion_loss_cls=0.2, motion_loss_reg=0.2 vs plan_loss_reg=1.0, det_loss_cls=2.0. exp001 (plan_loss_up) showed that simply upweighting planning loss improves L2 and collision. exp002 (motion_loss_up) is running to test motion loss upweighting.

**File**: `projects/configs/sparsedrive_small_stage1.py` motion section.

### B5. High ID Switch Rate in Tracking

**Evidence**: 1045 total ID switches in the reproduced baseline (cars: 413, pedestrians: 561 — highest two). FRAG=626. The tracking uses instance_id matching in InstanceQueue with `tracking_threshold=0.2`. The confidence_decay=0.6 means temporal instance confidence halves every ~1.5 frames — fast enough to cause good tracks to die when detections momentarily dip below threshold.

**File**: `projects/mmdet3d_plugin/models/instance_bank.py` — `confidence_decay=0.6`.

**Impact**: ID switches break temporal feature consistency for motion prediction, since InstanceQueue matches by instance_id. Fragmented tracks → corrupted motion features.

### B6. Backbone LR in Stage 2 = 0.1 (Too Frozen)

**Evidence**: `img_backbone` LR multiplier drops from 0.5 (stage1) to 0.1 (stage2). This is appropriate for stability but may prevent the backbone from adapting its features to the new motion/planning supervision signal.

**File**: `projects/configs/sparsedrive_small_stage2.py` optimizer section.

### B7. Recall ~50% Directly Limits Planning Safety

**Evidence**: Overall recall = 0.506 (reproduced baseline). FN counts — car: 19253, pedestrian: 10833, truck: 4693. Over half of GT trucks and pedestrians are never detected. Since the planner only avoids agents present in `det_output`, undetected agents are invisible to the planner.

This is the fundamental perception-planning coupling: planning collision rate is not just about planning quality, it's limited by how many agents are surfaced to the planner.

### B8. 6-Mode Limitation for Motion and Planning

**Evidence**: `fut_mode=6, ego_fut_mode=6`. Mode selection uses best-of-N selection with FocalLoss on mode classification. With only 6 modes, the predicted distribution can't represent genuinely multi-modal futures (e.g., agent about to turn left vs. go straight at an intersection).

**File**: `data/kmeans/kmeans_motion_6.npy`, `kmeans_plan_6.npy`.

### B9. Rotation Augmentation Not in Main Recipe

**Evidence**: The `_rotaug` config variants exist but the main `sparsedrive_small_stage1/2.py` configs don't include 3D rotation augmentation as standard. There's also a known float64 bug in the rotation augmentation path (documented in MEMORY.md). This means models trained on the main recipe aren't using 3D rotation augmentation.

### B10. Map Head Failure at Per-GPU Batch Size > 6

**Evidence**: Systematic comparison across four configs (see Section 3.13) shows map mAP collapses from 0.553 to 0.078 when per-GPU batch size increases from 6 to 12, with all other variables controlled. Map loss never converges at bs=12/GPU. Detection is unaffected. The cause is isolated to per-GPU batch size — not total batch size, not LR, not loss weights, not temporal gradient flow (which is always detached in `cache()`). Most likely mechanism: backbone BN statistics differ at bs=12/GPU vs bs=6/GPU, degrading ground-plane projected feature quality that the map head's `SparsePoint3DKeyPointsGenerator` depends on. Experiments confirming the cause are pending (see Section 3.13).

**Note on `decouple_attn_map=False`**: Map head uses `decouple_attn=False` while det/motion use `True`. This architectural inconsistency may independently limit map quality but is not the cause of the batch-size-dependent failure.

### B11. Top-k Agent Selection = 50

**Evidence**: `num_det=50` in MotionPlanningHead. In crowded urban scenes with 50+ nearby agents, many relevant agents are dropped before motion/planning processing.

---

## 5. Improvement Ideas

### A. Fast Tuning Wins

**A1. Enable Denoising Training (num_dn_groups ≥ 5)**
- **Why**: DN-DETR shows consistent +1-2 NDS/mAP. Infrastructure fully exists. Configs like `sparsedrive_r50_stage1_8gpu_noflash_nomap_dn.py` already prototype this.
- **How**: Set `num_dn_groups=5` in sampler, verify `dn_noise_scale` and `max_dn_gt=32` settings.
- **Risk**: Can destabilize training if noise scale is too high; start with `dn_noise_scale=[1.0]*3+[0.5]*7`.
- **File**: `projects/configs/sparsedrive_small_stage1.py` — sampler dict.
- **Upside**: +0.5-2 NDS, improved recall.

**A2. Extend Stage-2 Training to 20-25 Epochs**
- **Why**: 10 epochs × 586 iter/epoch = 5,860 steps for motion+planning is insufficient. Extending to 20-25 epochs doubles the planning supervision.
- **How**: Set `num_epochs=20` in stage2 config, lower `min_lr_ratio` from 1e-3 to 1e-4.
- **Risk**: Low — this is pure additional compute.
- **File**: `projects/configs/sparsedrive_small_stage2.py`.
- **Upside**: Expected improvement in motion min_ADE, planning L2, and collision rate.

**A3. Raise Motion Loss Weights**
- **Why**: motion_cls=0.2, motion_reg=0.2 is 10× weaker than detection. Motion head needs stronger gradients to adapt instance features for trajectory prediction.
- **How**: Try motion_cls=0.5, motion_reg=0.5. Then try 1.0/1.0. Monitor for training instability.
- **File**: `projects/configs/sparsedrive_small_stage2.py` — motion loss weights.
- **Upside**: Improved min_ADE/FDE.

**A4. Lower confidence_decay (0.6 → 0.8)**
- **Why**: 0.6 decay means instances lose 40% of their temporal confidence each frame. This causes too-fast forgetting and contributes to high ID switches.
- **How**: Set `confidence_decay=0.8` in both InstanceBank instances.
- **File**: `projects/configs/sparsedrive_small_stage1.py`.
- **Risk**: May keep low-quality detections too long; monitor FAF rate.
- **Upside**: Reduced ID switches, better tracking consistency.

**A5. Tune cls_threshold_to_reg (0.05 → 0.02)**
- **Why**: Currently, only anchors with classification confidence > 0.05 receive regression gradients. Lowering it forces regression on more candidates, potentially improving recall.
- **File**: `projects/mmdet3d_plugin/models/detection3d/detection3d_head.py`, `cls_threshold_to_reg=0.05`.
- **Risk**: May increase false positives slightly.

**A6. Trailer-Specific Fixes**
- **Why**: Trailer AMOTA ≈ 0 is unacceptable. Trailer recall is only 21.5%.
- **How**:
  - Add `cls_allow_reverse` for trailers (same as barriers): `cls_allow_reverse=[class_names.index("barrier"), class_names.index("trailer")]`
  - Increase class-wise loss weight for trailer in `cls_wise_reg_weights`
  - Inspect whether any kmeans_det_900 clusters correspond to trailer shapes
- **File**: `projects/configs/sparsedrive_small_stage1.py`.
- **Upside**: Could recover 0.03-0.05 AMOTA from near-0.

**A7. Raise num_det in MotionPlanningHead (50 → 100)**
- **Why**: Only 50 agents are passed to motion/planning. In dense scenes this drops important agents.
- **How**: Change `num_det=50` to `num_det=100` in MotionPlanningHead config.
- **Risk**: Small compute increase.

**A8. Check and Fix Rotation Augmentation Float64 Bug**
- **Why**: The rotaug configs exist but the float64 issue may cause training instability or silent corruption.
- **How**: Verify the `dtype=trajs.dtype` fix in `augment.py` BBoxRotation, and check `NuScenesSparse4DAdaptor` for similar np.cos/sin upcasting.
- **File**: `projects/mmdet3d_plugin/datasets/pipelines/augment.py`, `transform.py`.
- **Upside**: Once fixed, enable rotaug in the main training recipe for free data diversity.

---

### B. Medium-Effort Method Improvements

**B1. Increase Motion/Planning Mode Count (6 → 12)**
- **Why**: 6 modes from kmeans is minimal. Brier score and miss rate both suffer from insufficient mode diversity. At intersections, 6 modes can't represent left/straight/right × slow/fast combinations.
- **How**:
  - Run `tools/kmeans/gen_motion_anchor.py` with `num_clusters=12`
  - Run `tools/kmeans/gen_plan_anchor.py` with `num_clusters=12`
  - Update config `fut_mode=12, ego_fut_mode=12`
- **Risk**: More modes → harder classification. Need to recheck post-processing.
- **Upside**: Reduced miss rate, better Brier-FDE.

**B2. Enable Decouple Attention for Map Head**
- **Why**: `decouple_attn_map=False` is inconsistent with detection. Decoupled attention allows separate key/value projections for position vs. content.
- **How**: Set `decouple_attn_map=True` in stage1 config.
- **Risk**: Architecture change; stage1 checkpoint not directly compatible. Requires full stage1 retraining.
- **Upside**: Possible +1-3pp map mAP.

**B3. Add Temporal Map Cache in Stage 1**
- **Why**: `num_temp_instances=0` for map in stage 1 means map has no temporal reasoning until stage 2. Map elements are highly static and would benefit from temporal consistency from the start.
- **How**: Set `num_temp_instances=33` for map in stage1.
- **File**: `projects/configs/sparsedrive_small_stage1.py`.
- **Risk**: Changes stage1 checkpoint format; requires full stage1 retraining.

**B4. Soft Collision Loss for Planning**
- **Why**: The planning head is trained with FocalLoss on mode classification + L1 regression but has no explicit penalty for trajectories that intersect GT agent boxes. The model learns to imitate trajectories but has no direct collision-avoidance signal.
- **How**: Add an auxiliary collision loss: for each predicted plan trajectory point, compute soft BEV overlap with detected/GT agent boxes and penalize.
- **Upside**: Direct optimization of the collision rate metric.
- **Risk**: Complex to implement; need to ensure it doesn't overwhelm trajectory accuracy.

**B5. Better Temporal DN (Temporal Denoising Groups)**
- **Why**: `num_temp_dn_groups=0` even when standard DN is enabled. Temporal DN adds noise to cached temporal anchors, regularizing the temp_gnn more aggressively.
- **How**: Set `num_temp_dn_groups=3` alongside `num_dn_groups=5`.
- **Upside**: Improved tracking consistency, fewer ID switches.

**B6. Instance Confidence Score Calibration**
- **Why**: The quality estimation (centerness × yaw-ness) used for score rescoring may have poorly calibrated magnitudes, causing over- or under-suppression.
- **How**: Analyze the distribution of centerness and yaw-ness scores on val. Apply learned temperature scaling or isotonic regression calibration.
- **Upside**: Better NMS and tracking threshold performance.

**B7. Map-Conditioned Planning Loss**
- **Why**: The planner receives map features via cross-attention but there's no explicit map-compliance loss ensuring the planned trajectory stays in the correct lane.
- **How**: Given the predicted ego trajectory and online map output, add a loss penalizing distance from nearest lane centerline or penalizing lane boundary crossings.

**B8. Address Recall Deficit via Auxiliary Supervision**
- **Why**: Only the final decoder layer (layer 6) produces the evaluation output. Intermediate layers produce auxiliary predictions used only for training losses. Ensuring high-quality intermediate supervision improves final recall.
- **How**: Consider DINO-style contrastive denoising with auxiliary layer supervision across all 6 decoder layers.

---

### C. Larger Conceptual Improvements

**C1. Gaussian/Distribution-Based Motion Prediction**
- **Why**: Current motion prediction outputs deterministic trajectory points (best mode by min_ADE). No explicit per-mode uncertainty representation. Planning with uncertain predictions requires explicit uncertainty for safety margins.
- **How**: Predict a Gaussian covariance per timestep alongside the trajectory. Use NLL loss instead of L1.
- **Upside**: Better Brier-FDE, principled collision avoidance.

**C2. Value-Based Planning / Collision Reranker**
- **Why**: The planner imitates GT ego trajectories (open-loop). This suffers from distribution shift at inference.
- **How**: Introduce a simple value function trained on collision/near-miss annotations that scores planned trajectories. Use it as a reranker over the 6 plan modes.
- **Upside**: Direct improvement on collision rate metric.

**C3. Longer-Horizon Planning (6 → 10 timesteps)**
- **Why**: Current ego_fut_ts=6 = 3 seconds. For highway merging or complex intersections, 3 seconds is too short.
- **How**: Extend `ego_fut_ts=10`, retrain stage2, regenerate planning anchors.
- **Upside**: Better anticipation of long-horizon hazards.

**C4. Visibility-Aware Feature Weighting**
- **Why**: Visibility scores are computed (`with_visibility=True`) but not used during motion training. Agents with low visibility should have their motion predictions down-weighted since GT trajectories for occluded agents are less reliable.
- **How**: Add visibility-dependent loss weighting in MotionTarget: `reg_weight *= visibility_score`.
- **Upside**: Better calibration of motion predictions for partially occluded agents.

**C5. Map-Ego Temporal Consistency**
- **Why**: The map is regenerated with only 33 temp instances in stage2. Map elements are highly static. A longer temporal cache (e.g., 10 frames) with a map consistency loss would greatly stabilize map outputs.

---

### D. Data-Centric Improvements

**D1. Trailer-Specific Scene Oversampling**
- **Why**: Trailer AMOTA ≈ 0. The dataset has only 2425 GT trailer instances vs 58317 car instances (24× imbalance).
- **How**: Implement class-balanced scene sampling: oversample scenes containing trailers by 5-10× during training.
- **File**: `projects/mmdet3d_plugin/datasets/nuscenes_3d_dataset.py`.
- **Risk**: May slightly reduce car performance.

**D2. Enable 3D Rotation Augmentation After Bug Fix**
- **Why**: `BBoxRotation` is implemented but disabled in the main training recipe. 3D rotation augmentation is one of the most effective augmentations for BEV detection.
- **How**: Fix float64 issue fully, then add to main stage1 config.
- **File**: `projects/configs/sparsedrive_small_stage1.py`, `augment.py`.

**D3. Hard Negative Mining for Detection**
- **Why**: FP rates are high (car: 6690 FP, truck: 1644 FP, trailer: 518 FP with near-zero TP). Many false positives come from geometrically similar but incorrect classes.
- **How**: Implement OHEM or focal loss alpha reweighting specifically for high-FP classes.

**D4. Night and Adverse Weather Augmentation**
- **Why**: The current augmentation only uses photometric distortion. More aggressive augmentation (gamma, blur, noise) would improve robustness to real-world conditions.

---

### E. Evaluation and Science Improvements

**E1. Per-Distance-Range Detection and Planning Breakdown**
- **Why**: Current metrics aggregate all distances up to 55m. Detection quality degrades significantly beyond 30m for camera-only systems. Per-range metrics (0-15m, 15-30m, 30-55m) reveal where the model actually fails.

**E2. Per-Class Motion Prediction Breakdown**
- **Why**: Current motion evaluation only reports car and pedestrian. Cyclists, motorcyclists, and trucks have very different motion profiles.

**E3. Seed Variance Reporting**
- **Why**: With only 10 epochs of stage2 training, variance between seeds could be significant (±0.5-1 AMOTA). All current experiments appear to be single-seed.
- **How**: Run 3 seeds of the final configuration. Report mean ± std.

**E4. Runtime vs. Accuracy Pareto Analysis**
- **Why**: SparseDrive-S reportedly runs at 9 FPS vs UniAD's 1.8 FPS. This efficiency claim should be benchmarked against model variants (num_decoder, embed_dims, num_modes).

**E5. Collision Rate Breakdown by Object Type**
- **Why**: The `check_ego_collisions.py` tool exists but collision breakdown by object class (car, pedestrian, cyclist, trailer) would identify which class is most dangerous.

**E6. GT Detection Oracle Upper Bound**
- **Why**: `sparsedrive_r50_stage2_4gpu_gtdetmap.py` configs exist. Running the full pipeline with GT detection as input quantifies the gap attributable to detection quality vs. planning quality. This is critical for determining where to invest effort.

---

## 6. Missing Baselines and Critical Comparisons

### M1. Standard Two-Stage SparseDrive Baseline (REQUIRED)

**Why**: To claim any improvement, you need a fully reproducible baseline matching the paper's reported numbers. The available work_dirs only show inference runs, not full training.

**What's needed**: A complete stage1 (100 epoch) + stage2 (10 epoch) run using `sparsedrive_small_stage1.py` + `sparsedrive_small_stage2.py` with fixed seed.

**Cost**: ~20-30 GPU-hours.

**Purpose**: Internal confidence + paper credibility.

### M2. GT Detection Oracle (VERY IMPORTANT)

**Why**: The gtdetmap configs exist. Running them quantifies how much of the planning/collision metric is bottlenecked by detection vs. planning itself. If GT detection dramatically improves collision rate, the primary investment should be in detection. If not, planning improvements are the priority.

**Cost**: Stage2 training only (~3-5 GPU-hours).

**Purpose**: Critical architectural insight; required for any paper claiming planning improvements.

### M3. Multi-Seed Baseline (REQUIRED FOR PAPER)

**Why**: Planning collision rate at 0.097% is a very small number. A ±0.01pp variance would change the narrative. Seed variance over 3 runs needed.

**Cost**: 3× baseline cost.

**Purpose**: Scientific defensibility.

### M2. ~~GT Detection Oracle~~ DONE

**Result**: `stage2_4gpu_gtdetmap` run. car ADE 0.636→0.378 (−40%), but planning L2 unchanged (0.636→0.651). **Finding: detection is NOT the planning bottleneck.** Focus should be on planning head improvements.

### M3. Multi-Seed Baseline

**Why**: Planning collision rate at 0.133% is a small number. Seed variance needed for scientific defensibility.

**Status**: Not yet run.

### M4. ~~DN vs. No-DN Ablation~~ DONE

**Result**: DN clearly improves detection (+0.009 mAP, +0.022 AMOTA, −22% IDS) with no planning degradation. DN is confirmed beneficial and should be in the main recipe.

### M5. Stage-2 Duration Ablation (10 vs. 20 vs. 30 epochs)

**Why**: Stage-2 duration is a free variable. exp001 shows planning is undertrained. Extending epochs may yield additional gains.

**Status**: Not yet run.

### M6. ~~Rotation Augmentation~~ PARTIALLY DONE

**Result**: Rotaug in stage1 is very effective (+0.044 mAP over nomap-only). Rotaug in stage2 alone (`nomap_rotaug`) does not help (L2 stays similar or worse). Should be part of stage1 recipe.

### M7. Constant Velocity / CTRV / CTRA Baselines

**Why**: `predonly_cv`, `predonly_ctrv`, `predonly_ctra`, `predonly_ca` configs exist. Essential to show the learned model beats physics priors.

**Status**: Not yet run.

### M8. R101 with DN + Nomap + Rotaug Full Stack

**Why**: Individual improvements are known. Combining DN + nomap (stage2) + rotaug with R101 backbone should deliver the best end-to-end result.

**Status**: Not yet run. High priority.

---

## 7. Prioritized Action Plan

> Updated 2026-03-30 based on empirical results.

### Confirmed Wins (run these immediately)

| Rank | Idea | Evidence | Expected Gain | Notes |
|------|------|----------|---------------|-------|
| 1 | **R101 backbone** | R101 stage2: NDS=0.5857 vs R50=0.5232 | +0.063 NDS, +0.125 AMOTA | Requires full retraining |
| 2 | **DN training in stage1** | DN stage2 AMOTA=0.4179 vs 0.3714 | +0.046 AMOTA, −22% IDS | Requires full retraining |
| 3 | **Nomap in stage2** | nomap bs24: L2=0.588 vs 0.636 | −7% L2, −22% collision | Use dedicated nomap base config, NOT override |
| 4 | **num_det=100** | mar26 exp003 + mar27 exp001: confirmed on 2 configs | −4.2% L2, −12.5% col, −42% IDS (zero compute cost) | **Universal — apply to all configs** |
| 5 | **queue=6 (nomap only)** | mar25 exp003 + nomap_queue6 base | −3% L2 on nomap | Do NOT use with map head active |
| 6 | **Extend stage2 to 15 epochs (nomap only)** | mar26 exp001: L2=0.574, col=0.099% | −3.2% L2, −5% col | Do NOT use with map head active |
| 7 | **Plan loss upweighting (nomap only)** | mar26 exp002: col=0.094%; mar25 history | best col in isolation on nomap | Negative synergy with num_det=100 |

### Best Current Recipe (empirical)

Based on all results, the best R50 configuration for planning is:

**Best confirmed config: `auto_mar26_exp003_num_det100`** (nomap_queue6 base + num_det=100)
- Results: **L2=0.5676, obj_box_col=0.091%**, NDS=0.5231, AMOTA=0.3712, IDS=1086
- Config: sparsedrive_r50_stage2_4gpu_nomap_queue6.py + `model['head']['motion_plan_head']['num_det'] = 100`

**Best bs24 with-map config: `auto_mar27_exp001_num_det100`**
- Results: L2=0.6159, obj_box_col=0.091%, NDS=0.5258, AMOTA=0.3751, IDS=577
- Note: with-map configs are brittle — only num_det=100 reliably improves them

**Full stack (not yet run):**
**Stage 1**: `R101 + DN (num_dn_groups=5, num_temp_dn_groups=3) + nomap + rotaug`
- Expected: NDS≈0.62+, AMOTA≈0.55+

**Stage 2**: `nomap + queue=6 + num_det=100 + 15 epochs`
- Expected: L2≈0.54-0.56, obj_box_col≈0.07-0.08%

### Ideas with Negative or Null Evidence (deprioritize)

| Idea | Result | Verdict |
|------|--------|---------|
| Prediction pretraining (pretrainv3/v4) | All metrics worse | Do not pursue |
| Separate prediction head (sephead) | Slightly worse | Do not pursue |
| Nomap in BOTH stages | Planning catastrophic failure | Never do this |
| Stage2 with map from DN pretrain (bs24_pt2) | L2=0.700 (worse) | Map head in stage2 hurts |
| Map LR reduction (maplrdiv4) | map_mAP collapses | Do not use |
| GT oracle → planning improvement | L2 unchanged | Detection not the bottleneck |
| motion_loss_reg/cls > 0.2 | L2 +7.6%, col +85% (mar25 exp002) | Never increase motion loss weights |
| confidence_decay=0.8 | FAF +18, col 0.120% vs 0.104% (mar26 exp004) | Do not increase confidence_decay above 0.6 |
| Combining plan_loss_up + num_det=100 + epochs15 | L2=0.611 worse than baseline (mar26 exp005) | Negative synergy — deploy changes one at a time |
| queue_length=6 with map head active (bs24) | col 0.091%→0.147%, IDS stays high (mar27 exp002) | Do NOT increase queue_length when map head is active |
| epochs=15 with map head active (bs24) | L2=0.635 and col=0.143% — WORSE than 10-epoch baseline (mar27 exp004) | Do NOT increase epochs on bs24 with-map — map/planning gradient competition compounds |
| with_map=False via config override on bs24 base | 3× DDP crash: map head weights remain registered, find_unused_parameters doesn't fix (mar27 exp003) | Do NOT set with_map=False via override — requires a base config built without map head |

### Remaining High-Value Ideas (untested)

| Idea | Expected Impact | Effort | Notes |
|------|----------------|--------|-------|
| epochs15 + num_det=100 on nomap_queue6 | High — two confirmed wins together on right base config | Low | Use nomap base, not bs24 |
| num_det=150 on nomap_queue6 | Medium — if 50→100 helped, may push further | Low | Zero compute cost |
| plan_loss_up + num_det=100 on nomap_queue6 | Medium — test if synergy holds on nomap (differs from bs24 with-map) | Low | Avoid on with-map configs |
| Trailer cls_allow_reverse + oversampling | Medium | Low | |
| Increase planning/motion modes 6→12 | Medium | Medium | |
| Soft collision auxiliary loss | High | High | |
| R101 + DN + nomap + num_det=100 + epochs15 full stack | Very High | High (full retraining) | Top priority |

---

## 8. Concrete Next Experiments (Execution Order)

### Immediate (high confidence wins on nomap base)

**Exp A: epochs15 + num_det=100 on nomap_queue6** ← top priority
- Stage2: nomap_queue6 base + num_det=100 + num_epochs=15
- Why: both confirmed individually on nomap; mar26 exp005 neg-synergy was because plan_loss_up was also included. Clean 2-way combo.
- Expected: L2≈0.55-0.56, col≈0.085-0.090%
- IMPORTANT: use nomap_queue6 base config, NOT bs24 override

**Exp B: num_det=150 on nomap_queue6**
- Stage2: nomap, queue=6, num_det=150
- Why: if 50→100 was a strong positive, 100→150 may compound further
- Expected: L2≈0.560, col≈0.085%

**Exp C: plan_loss_up + num_det=100 on nomap_queue6 (10 epochs)**
- Stage2: nomap, queue=6, num_det=100, plan_loss_reg=2.0, plan_loss_cls=1.0
- Why: mar27 showed negative synergy on bs24 with-map. Test on nomap where plan_loss_up was originally validated.
- Expected: col≈0.080-0.085%, L2≈0.560-0.570

**Exp D: R101 + DN + Nomap + Rotaug full stack**
- Stage1: R101, num_dn_groups=5, with_map=False, rot3d_range=[-0.3925, 0.3925]
- Stage2: nomap, num_det=100, epochs=15
- Expected: NDS≈0.60+, AMOTA≈0.55+, L2≈0.52-0.55, obj_box_col≈0.060-0.075%
- This is the most promising end-to-end experiment

### Short-term (untested ideas)

**Exp E: Trailer cls_allow_reverse**
- Add trailer to `cls_allow_reverse` list alongside barriers
- Expected: AMOTA recovery from ~0.001 to 0.05-0.10

**Exp F: Increased planning/motion modes (12 modes)**
- Regenerate kmeans anchors with k=12
- Retrain stage2 with num_det=100 recipe

### Science / Paper Requirements

**Exp F: Multi-seed variance (3 seeds of best config)**
- Run Exp A with seeds 42, 123, 456
- Report mean ± std for all metrics

**Exp G: CV/CTRV/CTRA baselines**
- Run existing predonly_cv, predonly_ctrv configs
- Needed to establish learned model beats physics priors

---

## 9. Open Questions / Missing Artifacts

1. **Optimal loss weights with num_det=100 on nomap**: plan_loss_up causes negative synergy with num_det=100 on bs24 with-map (mar27 exp005). Unclear whether this also holds on nomap_queue6 where plan_loss_up was originally validated. Exp C (Section 8) will answer this.

2. **Stage-2 duration with num_det=100 on nomap**: epochs15 alone helps on nomap (mar26 exp001). Does epochs15 + num_det=100 combine safely on nomap? Mar27 exp004 showed epochs15 HURTS on with-map. Exp A (Section 8) will answer this for nomap.

3. **num_det ceiling**: Is 100 the optimal or is 150+ still better? The improvement from 50→100 was strong enough to test 100→150.

4. **Rotation augmentation in stage2**: `nomap_rotaug` doesn't help in stage2. Understanding why (optimization conflict?) would clarify if rotaug should only go in stage1.

5. **Why planning L2 worsens with GT perception**: GT perception improves motion by 40% but planning L2 slightly degrades. This suggests the planner overfits to noisy detection signals during training. Worth investigating whether the planning head can better exploit clean GT features with fine-tuning.

6. **Occlusion detection mechanism**: Current system cannot detect occluded objects. What would a principled architecture look like? Possible directions: (a) explicit memory queries for extrapolated trajectories, (b) occupancy-based prediction for hidden areas, (c) uncertainty-aware detection that propagates occluded agent hypotheses.

7. **Anchor quality for trailers**: Trailer AMOTA ≈ 0 persists across all experiments. Unknown whether any of the 900 kmeans detection anchors correspond to trailer shapes. Visualize anchor clusters in BEV.

8. **CV/CTRV baselines**: No classical motion prediction baselines run yet. Required to claim learned model beats physics priors.

9. **Multi-seed variance**: All current experiments are single-seed. Paper claims require ±std reporting.

10. **R101 + best recipe**: The best combination (R101 + DN + nomap + rotaug + num_det=100 + epochs15) has not been run as a single experiment. This is the clearest path to a new state-of-the-art on this codebase.

---

## Special Questions (Updated)

**Is SparseDrive likely under-tuned?**
Yes, confirmed empirically. exp001 (simply upweighting planning loss) reduces collision rate from 0.133%→0.084% (−37%) without any architectural change. This is the clearest possible signal that the stage2 planning head is undertrained.

**Which changes are most likely to improve results quickly?**
1. Combine confirmed wins: R101 + DN + nomap stage2 + loss upweighting (none require new code)
2. Extend stage-2 to 20 epochs
3. Trailer cls_allow_reverse fix

**Which changes are most likely to improve collision rate specifically?**
1. Planning loss upweighting (confirmed: 0.133%→0.084%)
2. Nomap in stage2 (confirmed: 0.133%→0.080-0.103%)
3. Temporal motion features critical — do not remove (notempmotion degrades badly)
4. Soft collision auxiliary loss (not yet tried, high expected upside)
5. Better mode diversity (12 modes) for planning

**What is the planning bottleneck?**
Empirically confirmed (GT oracle experiment): **not detection**. GT perception doesn't improve planning L2. The bottleneck is in the planning head itself: insufficient training signal, weak loss weights, and possibly fundamental limitations of the mode-selection planning paradigm.

**Which improvements would be most convincing in a paper?**
1. R101 + DN + nomap + rotaug ablation table (clean, reproducible)
2. GT oracle gap analysis (planning bottleneck confirmed, not detection)
3. Soft collision loss (direct metric optimization)
4. Seed variance + statistical testing

**Which baselines and ablations are required before claiming a method improvement?**
1. ~~DN vs. no-DN~~ (DONE: confirmed beneficial)
2. ~~GT detection oracle~~ (DONE: planning not bottlenecked by detection)
3. CV/CTRV baselines for motion prediction
4. Multi-seed variance on best config
5. Ablation table of each component's individual contribution
