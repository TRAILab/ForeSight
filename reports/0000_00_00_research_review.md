# SparseDrive Research Notes

> Living document for technical conclusions, bottlenecks, and promising directions.

---

## Summary

- SparseDrive appears materially under-tuned, especially in stage 2.
- Planning is not mainly bottlenecked by detection quality.
- The strongest proven levers so far are a stronger backbone, denoising in stage 1, removing the map head in stage 2, and increasing `num_det`.
- The map head is unusually brittle, especially at larger per-GPU batch sizes.
- Last updated: 2026-04-18

---

## Confirmed Findings

- SparseDrive is under-tuned, especially in stage 2.
- Better detection helps motion more than planning.
- Stage-1 map supervision is useful, but stage-2 map training often interferes with motion and planning.
- The planning head needs direct improvement rather than relying on perception gains alone.
- Denoising in stage 1 is clearly beneficial.
- R101 is the single strongest proven lever so far.
- Full rotation augmentation (rot3dv2 — rotating map_geoms, agent/ego trajectories, ego_status vel/accel, and recomputing gt_ego_fut_cmd) helps in both stage 1 and stage 2. The prior stage-2 null result was caused by a label inconsistency: only gt_bboxes_3d was rotated, leaving map/motion/planning GT in the original frame. Fixing all fields gives L2 0.636→0.593 and NDS 0.5232→0.5315 in stage 2.
- Trailer performance is catastrophically weak.
- Tracking remains fragile enough to affect downstream temporal reasoning.
- Planning is under-trained, and simple planning-focused tuning already helps.
- GT perception improves motion much more than planning.
- Removing the map head only in stage 2 is usually beneficial.
- Removing map from both stages breaks motion and planning.
- Temporal motion features are critical.
- Increasing `num_det` is one of the best cheap improvements.
- Prediction pretraining did not help.
- Separate prediction heads did not help.
- Map training is much more brittle than detection.
- Larger per-GPU batch size appears to be a real failure mode for the map head.
- Occlusion handling remains fundamentally weak.
- Most current conclusions are still based on single-seed experiments.

---

## Core Bottlenecks

- **Stage-2 under-training**
  - Planning receives too little optimization.
  - Simple training and loss changes already improve planning metrics.
- **Planning bottleneck independent of detection**
  - Detection-only improvements will not be enough.
  - GT perception improves motion strongly but not planning L2.
- **Stage-2 map interference**
  - The map head appears to compete with motion and planning gradients.
  - Removing map only in stage 2 often helps, while removing it from both stages is catastrophic.
- **Trailer failure**
  - Large missed obstacles are a direct safety issue.
  - Trailer AP / AMOTA remain near-zero or extremely poor.
- **Weak temporal tracking consistency**
  - Motion and planning depend on stable identities and temporal context.
  - ID switches remain high and weakening temporal support causes large degradation.
- **Limited agent budget**
  - Relevant actors are dropped before motion and planning.
  - `num_det=100` gives consistent gains over smaller settings.
- **Map-head batch-size brittleness**
  - This creates misleading failures and blocks clean experiment scaling.
  - Map quality collapses at larger per-GPU batch sizes while detection stays normal.
- **Low mode capacity**
  - Six motion/planning modes are likely too few for complex multimodal futures.
  - The case is strong conceptually but not yet fully validated experimentally.
- **Occlusion blindness**
  - Detection quality on fully hidden actors is extremely weak.
  - Occlusion-focused evaluation remains near-zero.

---

## Promising Directions

Highest priority:
- **Run the strongest full recipe**
  - stronger backbone
  - denoising in stage 1
  - full rotation augmentation (rot3dv2) in both stages
  - nomap in stage 2
  - larger `num_det`
  - stage 2 loading from rot3dv2 stage-1 checkpoint (not public ckpt)
  - longer stage 2 on the correct nomap base
- **Extend stage-2 training on the nomap recipe**
  - Stage 2 still looks under-optimized.
  - This should be done only on the correct nomap base config.
- **Apply trailer-specific fixes**
  - `cls_allow_reverse`
  - class-specific weighting
  - trailer oversampling
  - inspect anchor coverage
- **Stabilize map training**
  - grad accumulation
  - `norm_eval=True`
  - SyncBN
  - freezing map feature initialization

Medium priority:
- Increase motion/planning mode count from 6 to 12.
- Add a soft collision auxiliary loss.
- Enable temporal denoising groups, not just static DN.
- Add per-class and per-range evaluation.
- Add a map-conditioned planning loss.

Longer-term research directions:
- Predict explicit uncertainty for motion.
- Add value-based or reranking-style planning.
- Introduce explicit occlusion memory or persistent hidden-actor state.
- Use visibility-aware training for motion and planning.

Avoid / treat carefully:
- Avoid removing map from both stages.
- Avoid prediction pretraining variants that already degraded performance.
- Avoid separate prediction heads that already degraded performance.
- Avoid naive `with_map=False` overrides on configs not built for it.
- Avoid queue / epoch changes on with-map recipes that already showed negative interactions.
- Do not assume perception gains will automatically improve planning.
- Treat map-head results from large per-GPU batch settings carefully.
- The old stage-2-only rotation augmentation (rot3dv1) null result was a bug, not a real finding — do not use it to argue against rotation augmentation in stage 2.
- Treat any claim based on a single seed carefully.
- Treat motion-loss upweighting carefully until its interaction with the best nomap recipe is clearer.

---

## Open Questions

- What is the best combined stage-2 nomap recipe once `num_det`, queue length, and epoch count are tuned together?
- Is the map batch-size failure fundamentally BN-related, or is map feature initialization also involved?
- Is 100 the right agent budget, or does performance keep improving at 150+?
- How much of trailer failure is anchor coverage vs orientation handling vs class imbalance?
- Why does GT perception help motion much more than planning?
- Should rotation augmentation live only in stage 1, or can it also be useful in stage 2 with a different setup?
- How much does increasing mode count help planning in practice?
- What is the right architectural mechanism for persistent hidden actors?
- How large is seed variance on the best current recipe?
- What still needs to be validated before claiming a method improvement?
  - multi-seed results on the best recipe
  - classical prediction baselines (CV / CTRV / CTRA)
  - strongest combined recipe run end-to-end
  - clean ablation table of each major component
  - better per-class / per-range breakdowns