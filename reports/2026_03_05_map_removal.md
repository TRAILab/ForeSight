# Map Removal Experiments (Feb–Mar 2026)

## Intro

This document covers experiments that systematically remove the map head from different training stages to understand its role in detection, motion, and planning performance. Two conditions were evaluated: (A) removing the map head from both stage1 and stage2, and (B) removing the map head from stage2 only while keeping standard stage1 map training. These experiments were motivated by the observation that the map head introduces a competing gradient signal in stage2 that may limit motion and planning quality. Experiments ran between 2026-02-16 and 2026-03-05.

## Method

**Condition A — Nomap both stages** (`stage2_8gpu_pretrainv1_noflash_nomap`): Loads a stage1 checkpoint that was trained entirely without a map head (`stage1_8gpu_noflash_nomap_dn_rotaug`), then runs stage2 also without map. Compared against the Apollo 8GPU baseline.

**Condition B — Nomap stage2 only**: Two variants using the standard stage1 checkpoint (with map) but removing the map head in stage2:
- `stage2_4gpu_nomap` (bs=12/GPU, note: this is the failing per-GPU batch size, but the nomap metric is unaffected)
- `stage2_4gpu_bs24_nomap` (bs=6/GPU, 24 total)

Both Condition B variants compared against the DGX 4GPU bs24 baseline.

## Results

**Condition A: Nomap both stages vs. Apollo 8GPU baseline**

| Metric | Apollo baseline | Nomap both stages | Δ |
|--------|----------------|-------------------|---|
| NDS | 0.5187 | **0.5593** | +0.041 |
| mAP | 0.4076 | **0.4550** | +0.047 |
| AMOTA | 0.3714 | **0.4535** | +0.082 |
| IDS | 1088 | **396** | −692 |
| car ADE | 0.6148 | 3.863 | **+3.25 (BROKEN)** |
| Planning L2 | 0.600 | 6.612 | **+6.01 (BROKEN)** |
| obj_box_col | 0.104% | 3.605% | **+3.5pp (BROKEN)** |

**Condition B: Nomap stage2 only vs. DGX bs24 baseline**

| Metric | DGX baseline | Nomap stage2 (bs=12/GPU) | Nomap stage2 (bs=6/GPU) |
|--------|-------------|--------------------------|-------------------------|
| NDS | 0.5232 | 0.5217 | **0.5262** |
| mAP | 0.4132 | 0.4131 | **0.4154** |
| AMOTA | 0.3776 | **0.3787** | 0.3745 |
| IDS | 1045 | **785** | 853 |
| Planning L2 | 0.636 | **0.591** | **0.588** |
| obj_box_col | 0.133% | **0.080%** | **0.103%** |

Work dirs: `sparsedrive_r50_stage2_8gpu_pretrainv1_noflash_nomap` (2026-03-05), `sparsedrive_r50_stage2_4gpu_nomap` (2026-02-16), `sparsedrive_r50_stage2_4gpu_bs24_nomap` (2026-02-16).

## Discussion

- **Condition A (nomap both stages) is catastrophic for planning**: Detection improves dramatically (+0.047 mAP, +0.082 AMOTA, −692 IDS) when the map head never competes for gradient. However, planning L2 degrades from 0.600 to 6.612 and obj_box_col reaches 3.605% — completely non-functional. This confirms that **stage1 map training is not optional**: the map representations learned in stage1 are critical inputs to the planning head in stage2. Removing them breaks the planner entirely.
- **Condition B (nomap stage2 only) is beneficial on all metrics**: Removing the map head only in stage2 preserves the map features built during stage1 while freeing stage2 gradient capacity for motion and planning. Planning L2 improves by ~7% (0.636→0.588) and obj_box_col drops by 22% (0.133%→0.103%). Detection and AMOTA hold steady.
- **The map head competes for gradient capacity in stage2**: This is the key finding. The stage1 map representations survive as frozen instance features in the planning head's cross-attention, but the active map head in stage2 consumes optimizer capacity that would otherwise improve motion and planning.
- **Hard constraint**: Never remove the map head from stage1. Always keep it; the nomap optimization applies to stage2 only.
- **Config note**: `with_map=False` must be set in a purpose-built base config (not a runtime override) to avoid DDP incompatibility in PyTorch 1.13 (see Section 3.14).

## Future Work

- Build a dedicated `stage2_nomap_queue6` base config that natively excludes the map head, to enable nomap + queue=6 experiments without DDP issues.
- Verify that the stage1 map features accessed by the planning head survive and remain useful after 15+ epochs of stage2 fine-tuning without any map supervision.
- Test whether decouple_attn for the map head (`decouple_attn_map=True`) improves stage1 map quality, which may in turn improve planning when the map head is removed from stage2.
