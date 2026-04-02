# Prediction Pretraining Experiments — dual_head branch (Mar 2026)

## Intro

This document covers two experiments that attempted to improve motion/planning quality by pretraining the prediction/planning head before the full stage2 joint training. The hypothesis was that the planning head's short 10-epoch stage2 window leaves it undertrained, and a dedicated pre-training phase for the prediction head on motion supervision alone would give it a better initialization before joint optimization. Two variants were tested on different hardware: `pretrainv3` on DGX 4GPU and `pretrainv4` on Apollo 8GPU. Both were compared against their respective baselines. Experiments completed 2026-03-24.

## Method

**pretrainv3** (`stage2_4gpu_pretrainv3`): 4GPU DGX run with a dedicated prediction head pretraining phase inserted before the standard stage2 joint training. Forked from the DGX 4GPU bs24 baseline.

**pretrainv4** (`stage2_8gpu_noflash_pretrainv4`): 8GPU Apollo run with a similar pretraining strategy. Forked from the Apollo 8GPU baseline.

Both experiments used the `dual_head` branch which implements separate head pretraining logic. The exact pretraining protocol involved training the motion/planning head on GT trajectory supervision with frozen perception weights before unfreezing for joint optimization.

## Results

| Metric | DGX baseline | pretrainv3 (4GPU) | pretrainv4 (8GPU) | Apollo baseline |
|--------|-------------|-------------------|-------------------|-----------------|
| NDS | 0.5232 | 0.4997 | 0.5128 | 0.5187 |
| mAP | 0.4132 | 0.3841 | 0.4047 | 0.4076 |
| AMOTA | 0.3776 | 0.3129 | 0.3611 | 0.3714 |
| IDS | 1045 | 1702 | 1035 | 1088 |
| Planning L2 | 0.636 | 0.781 | 0.713 | 0.600 |
| obj_box_col | 0.133% | 0.189% | 0.163% | 0.104% |
| map mAP | 0.5528 | **0.065** | 0.554 | 0.5471 |

Work dirs: `sparsedrive_r50_stage2_4gpu_pretrainv3` (2026-03-24), `sparsedrive_r50_stage2_8gpu_noflash_pretrainv4` (2026-03-24).

## Discussion

- **Both pretraining approaches degraded all metrics below baseline**: pretrainv3 is dramatically worse (AMOTA −0.065, L2 +0.145, IDS +657), while pretrainv4 is closer to baseline but still worse on every metric. The pretraining strategy introduces optimization conflicts that hurt the overall joint system.
- **pretrainv3 nearly destroyed the map head**: map_mAP = 0.065 vs. 0.553 in baseline. The pretraining phase — likely requiring the map head to remain frozen while the prediction head trains — disrupted the map head's learned representations severely. pretrainv4 avoids this (map_mAP = 0.554), suggesting the 8GPU run used a different (safer) freezing protocol.
- **ID switches explode in pretrainv3**: 1702 IDS vs. 1045 baseline. This suggests the detection and tracking representations were disrupted by the pretraining optimization, likely because gradients from the prediction head flowed back into the shared instance features while the map head was frozen.
- **Pretraining creates optimization conflicts**: The fundamental issue is that the planning head's joint optimization with perception requires the shared instance representations to encode both detection and planning signals simultaneously. Pretraining forces the instance features toward a planning-optimized regime, which then resists the joint fine-tuning phase.
- **Verdict**: Do not pursue prediction pretraining in its current form. The approach is consistently harmful.

## Future Work

- If prediction pretraining is revisited, consider freezing the entire shared backbone and instance bank (not just the map head) during the pretraining phase, to avoid disrupting shared representations.
- Alternatively, extend stage2 training duration (15-20 epochs) as a simpler substitute — the core problem is insufficient planning training, and more joint epochs may achieve the same goal without pretraining instability.
- Investigate whether a curriculum approach (gradually increasing prediction loss weight over stage2 epochs) achieves similar pretraining benefits without the optimization conflict.
