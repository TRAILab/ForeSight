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

