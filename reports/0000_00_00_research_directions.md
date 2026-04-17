# SparseDrive Research Directions

> Future work grounded in the current report set.

---

## Highest Priority

- **Combine the best planning head with the strongest backbone.**
  Run `R101` together with the best confirmed planning model family, starting from `planpredtrajdeformmm_planinstfeat_laststage`.
- **Lock in a clean nomap stage-2 recipe before larger combinations.**
  The strongest supported nomap ingredients so far are stage-1 DN, `num_det=100`, `queue_length=6`, and 15 epochs, but the reports also show strong negative synergy when changes are stacked casually.
- **Test `nomap_planpredtrajdeformmm_planinstfeat_laststage`.**
  This is the main unresolved planning ablation left by the refinement report.
- **Validate the best current recipes with multi-seed runs.**
  Most conclusions are still based on single-seed evidence outside the DGX/Narval planning-refinement repeats.

---

## Planning-Focused Directions

- **Map the best decoder stage schedule for ego instance feature injection.**
  The current best model uses last-stage-only injection; testing the last two stages is the most direct follow-up.
- **Test whether planning-head gains stack with longer stage-2 training on the right base.**
  Longer training helped nomap AutoResearch runs, but not with-map runs.
- **Run the multi-waypoint planning attention variant to completion and evaluate it.**
  The config exists in the planning report, but results were still pending there.
- **Try last-stage ego-feature injection for motion deformable attention as well.**
  This is the most directly motivated extension from the planning-refinement batch.

---

## Perception And Stage-2 Directions

- **Run the strongest full R101 stage-1 recipe.**
  DN is already confirmed on R101; the missing check is whether the full stage-1 stack transfers cleanly on the stronger backbone.
- **Test temporal denoising on top of `num_dn_groups=5`.**
  This is the main next DN follow-up repeated across the DN reports.
- **Benchmark R101 runtime against R50.**
  The accuracy gains are clear; the remaining practical question is deployment cost.
- **If root-cause diagnosis matters, do the scene-distribution check for the map batch-size failure.**
  Otherwise, accept per-GPU batch size `>6` as a hard constraint and avoid spending more time on it.

---

## Occlusion Directions

- **Run the new occlusion-classifier experiment and evaluate it with the new TPR/FDR metrics.**
  The April 16 report updates the metric framework and loss design, but the classifier run itself was still pending.
- **Use filtered `vis/` and `occluded/` metrics once the classifier converges.**
  That is the first path to a meaningful precision signal for occluded detection.
- **If occlusion work continues, prefer memory-based approaches over more of the same supervision.**
  The current reports support explicit occlusion memory or temporal persistence as the next architectural direction.

---

## Open Questions

- What is the best combined nomap stage-2 recipe once `num_det`, queue length, and training length are tuned together without creating negative synergy?
- Do the planning-head improvements transfer cleanly to `R101`, or do they need retuning?
- Does `nomap_planpredtrajdeformmm_planinstfeat_laststage` match the map-supervised best model?
- Is the map batch-size failure fundamentally scene-distribution sensitivity, or something else still hidden in the optimization?
- How much additional gain is available from temporal DN?
- Can a working occlusion classifier convert the strong occluded recall signal into usable precision?

---

## Avoid / Treat Carefully

- Avoid removing the map head from stage 1.
- Avoid stage-2 rotation augmentation in the current recipe family.
- Avoid separate prediction-head variants in the tested form.
- Avoid prediction pretraining in the tested form.
- Avoid per-GPU batch size above 6 whenever the map head is trained.
- Avoid raising motion loss weight above the tested baseline in current stage-2 recipes.
- Avoid increasing `confidence_decay` above `0.6` in current recipes.
- Avoid naive runtime `with_map=False` overrides on configs built with a map head.
- Treat large recipe combinations carefully; the reports repeatedly show negative interaction between individually positive changes.
