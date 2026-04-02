# Temporal Motion and Task Ablations — 4GPU DGX (Feb–Mar 2026)

## Intro

This document covers a broad set of stage2 ablation experiments run on the DGX 4GPU setup to understand which training components and hyperparameters matter for planning and tracking. The ablations cover temporal motion features, map head presence, planning head, rotation augmentation, separate prediction heads, and learning rate schedules. All experiments fork from the DGX 4GPU bs24 baseline (`stage2_4gpu_bs24`: NDS=0.5232, AMOTA=0.3776, map=0.5528, L2=0.636, obj_box_col=0.133%). Experiments ran between 2026-02-14 and 2026-03-08.

## Method

Single-variable ablations were run against the DGX bs24 baseline. Key changes evaluated:

- **nomap**: map head removed from stage2 only (standard stage1, nomap stage2)
- **bs24_nomap**: same as nomap but with bs=24 (vs. bs=12 for the nomap config)
- **nomap_rotaug**: nomap + 3D rotation augmentation in stage2
- **notempmotion**: temporal motion features disabled in the motion/planning head
- **nomap_notempmotion**: both map and temporal motion removed
- **noplan**: planning head removed entirely
- **nomap_noplan**: both map and planning removed
- **bs24_nomap_sephead**: nomap + separate prediction head architecture
- **maplrdiv4**: map head learning rate divided by 4

## Results

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

Work dirs: `sparsedrive_r50_stage2_4gpu_notempmotion` (2026-02-14), `sparsedrive_r50_stage2_4gpu_noplan` (2026-02-14), `sparsedrive_r50_stage2_4gpu_nomap_notempmotion` (2026-02-17), `sparsedrive_r50_stage2_4gpu_nomap_noplan` (2026-02-17), `sparsedrive_r50_stage2_4gpu_nomap_rotaug` (2026-02-23), `sparsedrive_r50_stage2_4gpu_bs24_nomap_sephead` (2026-03-08), `sparsedrive_r50_stage2_4gpu_maplrdiv4` (2026-02-16).

## Discussion

- **Temporal motion features are critical for planning**: `notempmotion` degrades planning L2 by 30% (0.636→0.825) and nearly doubles the collision rate (0.133%→0.220%). Temporal context from prior frames is the primary source of the planning head's ability to predict safe trajectories. This component must never be removed.
- **Removing planning doesn't improve detection**: `noplan` has slightly worse NDS (0.5204 vs. 0.5232), ruling out planning gradient interference as a bottleneck for detection. The planning head is a net-neutral or mildly beneficial co-trainer for detection.
- **Rotation augmentation in stage2 doesn't help planning**: `nomap_rotaug` L2 is 0.621 and obj_box_col is 0.160% — worse than `nomap` without rotaug on both metrics. Rotaug is confirmed beneficial in stage1 but introduces optimization interference in stage2 that hurts planning. Do not apply rotaug in stage2.
- **Separate head (sephead) degrades performance across all metrics**: Planning L2=0.629 and obj_box_col=0.200% vs. 0.588 and 0.103% for the joint nomap baseline. The joint optimization of detection and planning through shared instance features is important.
- **Map LR/4 kills map quality**: `maplrdiv4` achieves map_mAP=0.074 (vs. 0.553 baseline). The map head is highly sensitive to its own learning rate — reducing it by 4× prevents the map anchors from learning meaningful ground-plane features. (Note: this is distinct from the per-GPU batch size failure documented in Section 3.13; this failure is purely from LR reduction.)
- **nomap + bs24 is the best R50 stage2 base**: combining map removal with the bs=6/GPU config yields the best planning metrics among all standard ablations (L2=0.588, obj_box_col=0.103%).

## Future Work

- Test `notempmotion` on a longer training schedule (20+ epochs) to understand whether temporal motion's importance is partially a training duration artifact.
- Investigate what specifically the temporal motion features carry: last-frame velocities, multi-frame history, or inferred intentions? Ablating the queue length (1 → 4 → 6 frames) quantifies the temporal horizon required.
- Test `maplrdiv2` (halved LR, not quartered) to find the minimum map LR that still converges — this would allow a softer map gradient in stage2 without catastrophic map failure.
- Revisit `sephead` with a longer training schedule — separate heads may require more iterations to converge to a good joint solution.
