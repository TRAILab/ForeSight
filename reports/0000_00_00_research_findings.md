# SparseDrive Research Findings

> Living document for experiment-backed conclusions only.

---

## Reproducibility And Baselines

- The released checkpoint reproduces the README numbers closely on our setup: NDS and mAP match within evaluation noise, and planning L2 stays near `0.61`.
- Our trained R50 baselines are slightly weaker than the released checkpoint, with the main gap in planning collision rate rather than detection.
- Trailer performance is catastrophically weak in the baseline runs: trailer AP@0.5 is effectively zero and trailer AMOTA is near zero.

---

## Confirmed Perception And Tracking Findings

- **R101 is the strongest confirmed perception/tracking lever.**
  R50 to R101 improves NDS from `0.5232` to `0.5857`, mAP from `0.4132` to `0.4954`, and AMOTA from `0.3776` to `0.5020`.
- **Removing the map head in stage 2 also helps R101.**
  `R101+nomap` improves further to `NDS=0.5936`, `mAP=0.4989`, `AMOTA=0.5032`, and `L2=0.592`.
- **Stage-1 denoising is a consistent win.**
  On R50 stage 1, DN improves mAP by `+0.009` and cuts IDS by 22%; on R101 stage 1, DN also improves NDS, mAP, and AMOTA.
- **Stage-1 rotation augmentation helps detection.**
  In the stage-1 ablations, rotaug is one of the strongest detection improvements and contributes to the best stage-1 checkpoint.
- **Removing the map head in stage 1 improves stage-1 detection but breaks the full pipeline.**
  Stage-1 nomap increases stage-1 detection metrics, but when map is removed from both stages, stage-2 planning becomes non-functional.
- **DN-pretrained stage 1 transfers well into stage 2 when stage 2 is nomap.**
  The Apollo DN-pretrained nomap run improves NDS, mAP, AMOTA, and IDS substantially while keeping planning L2 flat and slightly reducing collision rate.

---

## Confirmed Planning And Motion Findings

- **Direct planning-head changes work.**
  Every completed config in the planning-refinement batch improved both planning L2 and collision rate relative to the R50 bs24 baseline.
- **The best confirmed R50 planning model is `planpredtrajdeformmm_planinstfeat_laststage`.**
  Averaged over DGX and Narval, it reaches `L2=0.522` and `CR=0.047%`.
- **Endpoint deformable attention is the main collision-rate driver.**
  Attending at predicted far-horizon endpoints cuts collision rate much more than refinement alone.
- **Far-horizon endpoint attention matters; near-horizon does not.**
  The waypoint-0 variants are much worse than the final-waypoint variants, especially on collision rate.
- **Multi-mode motion deformable attention improves planning over single-mode motion attention.**
  `planpredtrajdeformmm` improves L2 relative to `planpredtrajdeform` while preserving similarly low collision rate.
- **Per-mode projection is harmful.**
  `planpredtrajdeformmm_modeproj` is worse than the base multimode variant on both L2 and collision rate.
- **Ego instance feature injection helps L2 only if restricted to the last decoder stage.**
  Injecting it at all stages improves L2 but causes a large collision-rate regression; last-stage-only injection preserves the L2 gain while restoring low collision rate.
- **Temporal motion features are critical.**
  Removing them degrades planning sharply: L2 rises from `0.636` to `0.825` and collision rate from `0.133%` to `0.220%`.
- **Planning is not improved much by perception alone.**
  The GT perception oracle improves car motion prediction strongly and reduces collision rate, but planning L2 does not improve.
- **Planning changes are largely isolated from upstream detection.**
  Across the planning-refinement batch, NDS, detection mAP, AMOTA, and IDS stay close to baseline while planning metrics improve.
- **Planning-relevance aux supervision on the detection head regresses planning without affecting detection.**
  On a same-host Killarney comparison against `ptaux2d_ppdeformmm_planifls`, adding a 2 m / 6-step planning-relevance BCE head to the detection refine module (`detrel_s2`) regressed L2 by 8.1% and `obj_box_col` by 66.7% relative, while NDS and mAP stayed flat. Reshaping detection features toward a planning-relevance axis the planner cannot consume costs planning information without compensating gain.

---

## Confirmed Map-Head Findings

- **Stage-1 map training is required for planning.**
  Removing the map head from both stages improves detection and tracking, but planning collapses to `L2=6.612` and `obj_box_col=3.605%`.
- **Removing the map head only in stage 2 is beneficial for baseline-style stage-2 training.**
  On R50, stage-2 nomap improves planning L2 and collision rate while keeping detection and AMOTA roughly stable.
- **Stage-2 map supervision is not required for the endpoint-attention planning mechanism to work.**
  The tested nomap planning-refinement variants perform close to their map-supervised counterparts.
- **The map head is brittle in ways the detection head is not.**
  Map LR divided by 4 destroys map quality, while detection remains comparatively stable.
- **Per-GPU batch size above 6 breaks map training.**
  This failure reproduces across map-training runs, and the working configurations all keep per-GPU batch size at 6.
- **BN freezing, disabling map anchor-init gradients, and gradient accumulation do not fix the batch-size failure.**
  The map failure persists after all three interventions, so none of them is the primary cause.

---

## Confirmed Stage-2 Recipe Findings

- **Stage-2 rotation augmentation does not help planning.**
  In the nomap stage-2 ablation, rotaug worsens both L2 and collision rate relative to nomap without rotaug.
- **Separate prediction heads do not help.**
  The sephead variant is worse than the joint nomap baseline across planning metrics.
- **Prediction pretraining is harmful in its tested forms.**
  Both pretraining runs degrade NDS, mAP, AMOTA, planning L2, and collision rate relative to baseline; one also destroys map mAP.
- **Increasing `num_det` from 50 to 100 is a reliable planning/tracking win.**
  In AutoResearch, `num_det=100` improved L2 and collision rate on both nomap and with-map recipes, and also cut IDS heavily in the with-map run.
- **Removing the detection top-k bottleneck entirely does not help planning.**
  In `plan_unified` `alldet` (Killarney 3284640), letting planning queries cross-attend to all 900 detection tokens (instead of `topk(det_confidence, 50)`) leaves L2 essentially unchanged vs. the server-matched baseline (0.4987 → 0.5018) and worsens collision rate (0.063% → 0.084%). With the prior `numdemapx2` null, this is consistent evidence that the planning ↔ detection interface is not the binding constraint — the bottleneck is what detection features encode, not how planning accesses them.
- **Adding bidirectional planning ↔ detection coupling at the decoder hurts planning.**
  In `plan_unified` `bidir` (Killarney 3284641), inserting a reverse cross-attention per decoder stage so detection features attend to current planning state regresses L2 from 0.4987 to 0.5378 (server-matched comparison), with no compensating gains elsewhere. Coupling detection and planning at the K/V layer in either direction does not improve planning.
- **`queue_length=6` helps on nomap but not with map.**
  In AutoResearch, queue length 6 improves L2 on nomap runs but hurts collision rate on with-map runs.
- **Longer stage-2 training helps on nomap but not with map.**
  Going from 10 to 15 epochs helps the nomap queue-6 recipe, but hurts planning on the with-map recipe.
- **Raising motion loss weight and increasing confidence decay are harmful in the tested recipes.**
  Both changes degraded planning in AutoResearch.
- **Naively stacking individually positive changes often fails.**
  Multiple combo experiments showed negative synergy even when the single changes were beneficial in isolation.
- **Auxiliary planning heads on `MotionPlanningRefinementModule` did not help at default settings.**
  At loss weight 0.2 applied to every refine stage, both `planaux_da` (per-waypoint drivable-area BCE; Killarney 3284570) and `planaux_conf` (per-anchor pos-weighted conflict BCE; Killarney 3284571) regressed against the same-server Killarney baseline (`L2=0.4988`, `obj_box_col=0.063%`): `planaux_da` reached `L2=0.5362`, `obj_box_col=0.086%`; `planaux_conf` reached `L2=0.5202`, `obj_box_col=0.124%`. Detection (NDS, mAP) was flat. Plausible causes — aux-loss interference at this weight/attachment, label noise in the boundary-flood-fill drivable mask, and recursive supervision in conflict (label depends on the model's own predicted ego trajectory) — none retested yet.

---

## Confirmed Occlusion Findings

- **Fully hidden objects are effectively undetectable in the current camera-only setup.**
  The dedicated occlusion experiments stayed near zero for fully hidden detection.
- **Partial-occlusion supervision gives only marginal gains in the current pipeline.**
- **The existing occluded mAP metric is dominated by false positives.**
  The April 16 metric analysis found high occluded TPR but extremely high FDR, so low occluded mAP is primarily an FP problem rather than pure recall failure.

---

## Evidence Quality Notes

- Most results are still single-seed unless a report explicitly includes repeats across DGX and Narval.
- The planning-refinement batch has the strongest replication within the report set because several key models were run on both servers.
