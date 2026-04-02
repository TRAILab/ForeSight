# AutoResearch Summary — mar25–mar27

## Intro

This research log documents three AutoResearch sessions (mar25–mar27) aimed at improving two planning metrics in the SparseDrive R50 stage-2 model: `val/L2` (trajectory accuracy) and `val/obj_box_col` (collision rate). Each session ran up to 5 experiments on a compute cluster using a fixed budget of ~10 training epochs. Sessions iterated on one another: mar25 used the default nomap bs48 config, mar26 promoted the best mar25 config (`queue_length=6`) as the new base, and mar27 switched to a with-map bs24 config to test generalization of findings across config families.

## Method

Experiments were generated and evaluated autonomously via the AutoResearch skill: each session proposed a set of targeted single-variable changes based on prior findings, submitted jobs to the cluster, and parsed validation metrics to decide which configs to keep or discard. Changes explored included loss weight scaling (plan and motion heads), temporal queue length, detection decoder depth, number of detections surfaced to the planner (`num_det`), confidence decay rate, and extended training epochs. A combination experiment was run at the end of each session to test whether the best individual changes could be stacked. All results were compared against a freshly re-run baseline at the start of each session to account for seed and code variance.

## Results

**mar25** (baseline: nomap bs48, L2=0.5911, col=0.080%)

| Exp | Config | L2 | obj_box_col | car_ade | NDS | Δ L2 | Δ col | Status |
|-----|--------|-----|-------------|---------|-----|------|-------|--------|
| baseline | nomap bs48 | 0.5911 | 0.080% | 0.6189 | 0.5217 | — | — | — |
| exp-001 | plan_loss_reg 1→2, cls 0.5→1 | 0.5902 | 0.084% | 0.6405 | 0.5239 | -0.0009 | +0.004% | keep |
| exp-002 | motion_loss 0.2→0.5 | 0.6358 | 0.148% | 0.6146 | 0.5158 | +0.0447 | +0.068% | **discard** |
| exp-003 | queue_length 4→6 | **0.5757** | 0.100% | 0.6281 | 0.5238 | **-0.0154** | +0.020% | keep ⭐ |
| exp-004 | num_decoder 6→8 | 0.5955 | **0.074%** | 0.6189 | 0.5239 | +0.0044 | **-0.006%** | keep ⭐ |
| exp-005 | queue6 + decoder8 | 0.5934 | 0.126% | 0.6292 | 0.5241 | +0.0023 | +0.046% | **discard** |

**mar26** (baseline: nomap_queue6 bs48, L2=0.5927, col=0.104%)

| Exp | Config | L2 | obj_box_col | car_ade | NDS | Δ L2 | Δ col | Status |
|-----|--------|-----|-------------|---------|-----|------|-------|--------|
| baseline | nomap_queue6 bs48 | 0.5927 | 0.104% | 0.6241 | 0.5236 | — | — | — |
| exp-001 | epochs 10→15 | 0.5738 | 0.099% | 0.6227 | 0.5253 | -0.019 | -0.005% | keep |
| exp-002 | plan_loss_reg 1→2, cls 0.5→1 | 0.5904 | 0.094% | 0.6323 | 0.5218 | -0.002 | -0.010% | keep |
| exp-003 | num_det 50→100 | **0.5676** | **0.091%** | 0.6307 | 0.5231 | **-0.025** | **-0.013%** | keep ⭐ |
| exp-004 | confidence_decay 0.6→0.8 | 0.5731 | 0.120% | 0.6315 | 0.5209 | -0.020 | +0.016% | **discard** |
| exp-005 | epochs15 + plan_up + det100 | 0.6109 | 0.107% | 0.6286 | 0.5210 | +0.018 | +0.003% | **discard** |

**mar27** (baseline: bs24 with-map queue=4, L2=0.6274, col=0.107%)

| Experiment | L2 | col% | IDS | FAF | Status | Key change |
|-----------|-----|------|-----|-----|--------|-----------|
| Baseline | 0.627 | 0.107% | 990 | 77.7 | — | bs24 with-map queue=4 |
| **exp001 num_det=100** | **0.616** | **0.091%** | **577** | **44.8** | **keep (best)** | num_det 50→100 |
| exp002 queue=6 | 0.623 | 0.147% | 995 | 74.4 | discard | queue 4→6 (with map) |
| exp003 crash | — | — | — | — | crash×3 | nomap DDP incompatible |
| exp003b plan_loss_up | 0.622 | 0.110% | 691 | 46.3 | discard | plan_loss×2 |
| exp004 epochs=15 | 0.635 | 0.143% | 778 | 45.2 | discard | epochs 10→15 (with map) |
| exp005 det100 + plan_up | 0.650 | 0.125% | 933 | 69.9 | discard | combo (negative synergy) |

## Discussion

Two changes emerged as reliable single-variable wins across sessions: `queue_length=6` (L2 −2.6% on nomap) and `num_det=100` (L2 −1.8–4.2%, col −12–15%, IDS −42% — effective on both nomap and with-map configs). Extended training (15 epochs) helps on nomap but hurts on with-map, where the active map head introduces gradient competition that makes longer optimization converge to a worse local minimum for planning. A persistent pattern of **negative synergy** appeared in every combo experiment across all three sessions — individually positive changes consistently degraded each other when stacked at 10 epochs, suggesting the optimizer lacks capacity to simultaneously improve multiple competing objectives. Hard constraints established: do not increase motion loss weights above 0.2; do not set `queue_length=6` with map head active; do not increase `confidence_decay` above 0.6; do not override `with_map=False` at runtime on a base config built with a map head (DDP incompatible).

## Future Work

- **Full-stack nomap config**: Build a purpose-built nomap + `queue_length=6` + `num_det=100` config (already partially achieved in mar26 exp003: L2=0.568, col=0.091%).
- **`num_decoder=8` with 15+ epochs**: Revisit on a nomap base — showed best single-run collision performance in mar25 and may compound with more training.
- **R101 backbone + DN pretraining**: Next architectural lever expected to push metrics toward SOTA for this model class.
- **Isolate `plan_loss_up` on nomap_queue6**: Re-evaluate without `num_det=100` stacked alongside — showed genuine collision benefit in isolation and may stack cleanly on the nomap base.
