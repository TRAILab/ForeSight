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
- **Distance-softmax soft-target supervision over all plan modes hurts both L2 and collision.**
  On the Killarney baseline (`ptaux2d_ppdeformmm_planifls`, `L2=0.4988`, `obj_box_col=0.063%`), replacing winner-takes-all L1 with distance-softmax weighted L1 over all 6 cmd-indexed plan modes (`mode_softtgt`, Killarney 3293054) regressed L2 to `0.5823` (+16.7% relative) and `obj_box_col` to `0.208%` (~3.3× baseline). Detection (NDS, mAP) stayed flat. Spreading gradients across modes anchored to off-direction k-means initializations pulls the winning mode away from the GT trajectory; the winner-takes-all assignment in `MotionTarget`/`PlanningTarget` is doing useful work in the current configuration. Top-k or low-temperature variants would be the redirection if soft-target supervision is to be retried.
- **Replacing the deformably-attended ego token at last stage is load-bearing for planning L2.**
  Adding `planning_deformable_instfeat_additive` to the existing `_laststage` recipe — using `plan_mode_query + ego_feat_exp` as the deformable query and *not* overwriting the ego token in `instance_feature` (`instfeataddls`, Killarney 3293055) — regressed L2 from `0.4988` to `0.5601` (+12.3% relative) but improved `obj_box_col` to `0.047%` (−25% relative vs. Killarney baseline). The original replacement scheme reuses the deformably-attended ego feature inside the refine layer on top of `plan_mode_query`; that double-use is load-bearing for L2 and is not a correctness issue to fix.
- **Mode-confidence aggregation in the motion deformable multimode block is load-bearing.**
  Replacing `softmax(motion_classification[-1])` with uniform `1/fut_mode` weights when aggregating per-mode attended features (`mm_uniform`, Killarney 3293056) regressed L2 from `0.4988` to `0.5209` (+4.4% relative) and `obj_box_col` from `0.063%` to `0.133%` (~2.1× baseline). Detection unchanged. Mode logits — not just spatial diversity at the kp generator — drive the multimode gain. This makes calibration-style follow-ups (Laplace NLL, focal cls) more interesting than diversity-only changes.
- **Per-(mode, ts) plan queries with shared 2-dim MLP regress trajectory accuracy regardless of added temporal coupling.**
  `modetime36q` (Killarney 3293233, on `planifls` base) regressed L2 from `0.4988` to `0.5578` and `obj_box_col` from `0.063%` to `0.099%`. The targeted fix `modetime36q_timeattn` (Killarney 3296782, on `ppdeformmm` base, no `planifls`) added a per-mode time-axis self-attention block (one MHA + LN per planning deformable stage, with a learnable time positional embedding) after the standard-branch deformable cross-attention to restore intra-mode temporal coupling. It regressed *further* to L2=`0.5872`, obj_box_col=`0.096%`. Detection unchanged in both runs. The 2-dim MLP output projection — predicting per-(mode, ts) (x, y) without parameter sharing across timesteps for the *output* — loses representational capacity that a single per-mode MLP outputting all 12 dims at once provides; adding a 6-token MHA over the queries is too thin to recover this. A real DeMo-style fix would need separate mode and state decoders, not a small MHA bolt-on.
- **Removing `planifls` (laststage ego-instfeat replacement) is a net loss on this host.**
  `modetime36q_timeattn` was run on the `ppdeformmm` baseline (no `planifls`) to escape the suspected fragile local minimum. It reached L2=`0.5872` vs. the cross-host `ppdeformmm` reference of `~0.542`. Even accounting for run-to-run variance, the laststage replacement is providing a real L2 lever; removing it does not open up the optimization landscape, it just removes the lever.

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
- **The existing `joint_detach` stage-1 ckpt is a worse init for planning than plain det+map stage-1, regardless of stage-2 head.**
  Two head choices, both regress:
  - **Modern stage-2 head** (`plan_unified` Arm A, Killarney 3301176): joint_detach init + `planpredtrajdeformmm` recipe → L2=0.6669 / obj_box_col=0.089%, vs plain-`stage1.pth`-init baseline 0.5420 / 0.047% (+0.125 L2, ~2× CR). NDS=0.5261, mAP=0.4092 flat.
  - **Legacy stage-2 head, head-matched to stage 1** (`plan_unified` Arm A-matched, Killarney 3305025): joint_detach init + plain `stage2_4gpu_bs24` recipe → L2=0.6945 / obj_box_col=0.161%, vs plain-`stage1.pth`-init DGX legacy baseline 0.636 / 0.133% (+0.059 L2, ~+21% rel CR). NDS=0.5231, mAP=0.4165 flat.
  The matched run controls the head-mismatch confounder, so the regression is attributable to joint_detach's stage-1 ckpt itself. As trained — `detach_perception=True`, legacy planning head, outlier lr=1.5e-4, ~1 pt NDS / ~0.7 pt mAP_normal weaker stage-1 perception than plain noflash — joint stage-1 does not help planning. Says nothing about the joint-stage-1 idea with the modern recipe; the in-progress `stage1_8gpu_noflash_joint` (modernized recipe, Arm B on Apollo) is the remaining clean test. Side observation: the modern stage-2 head was 0.028 better than the legacy head even from this bad init — the modern recipe's ~0.094 L2 gain is robust to stage-1 quality.
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
- **Restructuring the planner's mode bag also regresses planning across every axis tested (Batch B `plan_gaps`).**
  All three Tier 2 perturbations of the planning mode structure regressed against the same-server Killarney baseline (`L2=0.4988`, `obj_box_col=0.063%`); detection metrics were flat:
  - `modes20diverse` (Killarney 3293230, `ego_fut_mode` 6 → 20 with new k-means anchors + Gaussian pairwise diversity reg, weight 0.05, sigma 5.0): `L2=0.5361`, `obj_box_col=0.119%` — worst CR (~1.9× baseline).
  - `noagg` (Killarney 3293232, additive blend of per-mode attended into `motion_mode_query` / `plan_mode_query`; uniform mean for the `instance_feature` agent / ego path): `L2=0.5246`, `obj_box_col=0.091%` — mildest regression but still net-negative.
  - `modetime36q` (Killarney 3293233, per-(mode, ts) plan queries with shared MLP, 6×6=36 queries per cmd): `L2=0.5578`, `obj_box_col=0.099%` — worst L2 in the batch.
  Across `plan_relevance`, `plan_aux`, `plan_unified`, and `plan_gaps` Batch B, every architectural perturbation to the planning head has regressed planning. The most likely remaining bottlenecks are invariants we have not perturbed: the stage-1 backbone, the supervision shape (winner-takes-all on a single cmd-indexed mode), or the fixed `HierarchicalPlanningDecoder.rescore` / cmd-selection step.
- **Per-cmd plan mode-count scaling 6 → 12 also regresses, with or without speed stratification (Batch D `plan_gaps`).**
  Two anchor-scaling experiments on the same Killarney `planifls` baseline (`L2=0.4988`, `obj_box_col=0.063%`); detection metrics flat:
  - `modes12` (Killarney 3302033, `ego_fut_mode` 6 → 12 with a single full-distribution `kmeans_plan_12.npy`, **no diversity reg**): `L2=0.5278`, `obj_box_col=0.082%`. Rules out "diversity reg alone caused `modes20diverse` to fail" — diversity-free 12 modes still regress L2. Best CR among the four 12+-mode variants for the no-strat case.
  - `modesspeedstrat` (Killarney 3302034, two 6-mode k-means files speed-bucketed at 3 m/s on `||ego_status[6:8]||`, concatenated to 12 modes/cmd via a list-input path in `MotionPlanningHead`, single shared `plan_reg_branch`): `L2=0.5548`, `obj_box_col=0.070%`. Best CR among all 12+-mode variants (only +0.007pp over baseline) but worst L2 of the four. Confirms that bucketed clustering keeps anchors tight against the data (CR benefit) but the shared reg branch + winner-takes-all supervision can't separate low-speed vs high-speed clusters at decode (L2 cost).
  HiP-AD's stratification recipe doesn't transfer one-to-one because they pair stratified anchors with per-group reg branches; our shared-branch port loses the precision benefit at the decoder. Further per-cmd mode-count scaling without per-group reg branches is now ruled out (three failures: modes20diverse, modes12, modesspeedstrat).
- **Soft collision cost as a training-loss term on `plan_reg` is also null at first-cut weights.**
  In `plan_scoring/softcost_train` Exp 1 (`planpredtrajdeformmm_softcostcol`, Killarney 3296906, λ_col=0.2, σ=2.0m, GT-derived agent positions, all 18 plan modes per decoder stage), adding a Gaussian collision potential `mean exp(-d²/σ²)` as a backprop term to the existing imitation loss reached `L2=0.5515`, `obj_box_col=0.050%`. Versus the cross-server plain `planpredtrajdeformmm` baseline (`L2=0.542 / 0.047%` from `2026_04_02_plan_refine.md`), this is L2 +0.010 (~+1.8% relative) and CR essentially flat. Detection unchanged (`NDS=0.5257`, `mAP=0.4142`). The cost gradient appears to weaken imitation alignment without lifting CR. Caveat: cross-server comparison; no Killarney baseline exists for plain `planpredtrajdeformmm`. Smaller-λ and cmd-indexed-scope variants are worth one shot each before retiring the training-time deployment of analytic costs.
- **The hard-binary `HierarchicalPlanningDecoder.rescore()` is doing real CR work; richer inference-time scoring has headroom.**
  In `plan_scoring/rescore_disable_diag` Exp 0 (`ptaux2d_ppdeformmm_planifls_norescore`, Killarney 3302063), re-evaluating the same Killarney `ptaux2d_ppdeformmm_planifls` checkpoint with `use_rescore=False` reached `L2=0.5008, obj_box_col=0.106%`. Versus the same-server with-rescore baseline (`L2=0.4988, obj_box_col=0.063%`), L2 is flat (+0.002, noise) but CR jumps `+0.043pp` (~+68% relative). Detection (`NDS=0.5453`, `mAP=0.4399`) is unaffected as expected — rescore touches the planning selector only. Soft multi-cost inference selection (Exp 2 in the report) is the cleared direction; joint A+B (Exp 3) is deprioritized since the training-time leg did not land.
- **Soft Gaussian-on-SDF as an inference selector catastrophically regresses L2; the kernel is "always on" and dominates `plan_cls`.**
  In `plan_scoring/softrescore_infer` Exp 2 (Narval 59920744/45/46), evaluating the same Killarney `iter_11720.pth` ckpt with `HierarchicalPlanningDecoder.rescore_soft()` (4-ego-corner SDF to predicted-agent oriented bbox; saturating Gaussian `exp(-clamp(min_sdf, 0)² / σ²)`, σ=2m; min over agents per waypoint, sum over t; `plan_cls += -w_col · C_col`) gave `L2=0.6667, CR=0.067%` at `w_col=3`; `L2=0.7205, CR=0.072%` at `w_col=10`; `L2=0.7492, CR=0.079%` at `w_col=30`. Hard rescore is `0.4988 / 0.063%`, no rescore is `0.5008 / 0.106%` on the same ckpt. L2 monotonically worsens with `w_col`; CR is between hard and norescore but never matches hard. The cost is nonzero for *every* mode based on closest-agent SDF (decay too slow at σ=2m: penalty=0.37 at sdf=2m × 6 timesteps = 2.2; even at `w_col=3` this swamps `plan_cls` ∈ ~[-3, +3] logits), so selection collapses to "mode farthest from agents" — which deviates from imitation since humans drive close to parked cars / lead vehicles. Hard rescore works because it only fires on binary corner-in-box, leaving >95% of modes untouched. Sharper kernels or learned scoring with binary supervision are needed; smooth analytic costs without thresholding do not work as inference selectors.
- **Geometry-aware soft collision cost as a training loss is also null at smaller weight.**
  In `plan_scoring/softcostcol_v2` Exp 1 retry (`planpredtrajdeformmm_softcostcol_v2`, Narval 59920383, `λ_col=0.05`, `τ=0.5m`, min-of-4-ego-corner SDF to GT agent oriented bbox, closest-agent-only per (mode, t), softplus(`-min_sdf/τ`), mean over (M=18, T=6), ego heading from `atan2` central diff matching `rescore.get_yaw`), reached `L2=0.5481, obj_box_col=0.059%` from a `planpredtrajdeformmm` base. Versus v1 first cut (`L2=0.5515 / 0.050%`, Killarney 3296906): L2 -0.003, CR +0.009pp — both within noise. Versus plain `planpredtrajdeformmm` cross-server avg (`L2=0.542 / 0.047%`): L2 +0.006, CR +0.012pp — same regression direction as v1. Detection unchanged (`NDS=0.5266, mAP=0.4176, mAP_normal=0.5566`). Geometry-aware fix (4-corner SDF, closest-agent-only, smaller λ) did not lift the result out of v1's regime. Combined with the Exp 2 inference-side null, both training-time and inference-time deployments of analytic collision costs (Gaussian on point distance, saturating-Gaussian-on-SDF, softplus-on-SDF) do not improve planning. The remaining live direction is a learned scorer with a stationary BCE label decoupled from the planner's own predictions.

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
