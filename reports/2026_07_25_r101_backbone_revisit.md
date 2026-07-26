# R101 Backbone Revisit under the minS2 Planning Architecture (Jul 2026)

## Abstract

The March 2026 R101 report concluded that ResNet-101 was "the single largest lever
available in this codebase," but measured that on the old coupled stage-2 head,
before the K/V-off and minS2 architectures existed. This report re-tests R101 on
both current stage-2 architectures at hyperparameters matched exactly to the r50
anchors. **R101 buys perception and not planning: +0.055 to +0.063 NDS on both
architectures, for a planning L2 delta of +0.0060 (headline, inside the 0.007
noise floor) to +0.0099 (minS2, 1.4× the floor).** Planning L2 is essentially
insensitive to backbone capacity here, mirroring the saturation already seen
across r50 stage-1 swaps. A secondary result: freezing the stage-2 backbone is
free on r101 as it was on r50 (0.3743 frozen vs 0.3752 trainable), so the paper's
Stream E claim now holds on a second backbone. Recommendation: **do not adopt r101
into minS2** — it costs ~2× compute for no planning gain. An unresolved
hyperparameter question is spun out into a separate run (see Future Work).

## Intro

`reports/2026_03_12_r101_backbone.md` measured r101 against the r50 stage-2
baseline and found large gains: +0.063 NDS, +0.082 mAP, +0.125 AMOTA, planning
L2 0.636 → 0.598. Those came from the coupled motion-planning head with detection
and map tokens on the inference path. Everything since — K/V-off, `egostatus`,
minS2 — has moved planning L2 into the 0.36 band, roughly 0.24 below where that
comparison sat, so the March recommendation ("r101 should be the backbone for all
serious performance-optimized experiments") was never tested against the
architecture the paper uses.

The prior concern was specific: minS2 freezes backbone, neck, and both perception
heads (`lr_mult=0.0`), so r101 can only help through the quality of the frozen
features inherited from stage 1. Every r50 stage-1 swap (`dn`, `aux2p5d`,
`rot3dv2`, `dnrot3daux2p5d`, `dseg`, `dsegv2`) moved planning L2 by at most 0.012,
suggesting planning L2 is near-saturated with respect to stage-1 feature quality.

Question:
- Does the r101 perception gain transfer to planning under the current stage-2
  architectures, and specifically under minS2 where the backbone is frozen?

## Method

Two stage-2 architectures, each as a straight backbone swap:

- **minS2** — the locked paper architecture. Backbone + neck + det + map frozen
  (`lr_mult=0.0`), stage-2 perception losses zeroed, `ego_only_planning=True`,
  `image_at_det` conflict sampler.
- **headline** — the K/V-off comparator `..._decoder6_planwp_evalmatchmode_egostatus`,
  which trains the backbone at `lr_mult=0.1`.

Each was run at two hyperparameter settings, and **this distinction turned out to
drive the entire interpretation**:

- `bs32` / `lr=3e-4` — r101-tuned, matching `sparsedrive_r101_stage2_4gpu.py`.
- `bs24` / `lr=1.5e-4` — **anchor-matched**, so the backbone is the only variable
  against the r50 results. These are the rows to trust.

The r101 delta is 9 lines, verbatim from `sparsedrive_r101_stage2_4gpu.py`:
`input_shape` (704,256) → (1408,512); `strides` [4,8,16,32] → [8,16,32,64];
backbone `depth` 50 → 101; `norm_eval` False → True; BN `requires_grad` True →
False plus nuImages cascade-RCNN `init_cfg`; FPN `start_level` 0 → 1;
`resize_lim` (0.40,0.47) → (0.80,0.94); `load_from` →
`ckpt/sparsedrive_r101_stage1.pth`.

Stage-1 init is the **plain** r101 stage-1 checkpoint (NDS 0.5855 / mAP 0.4893),
chosen as the matched control because the r50 anchors use the plain default
`sparsedrive_stage1.pth`. The stronger `sparsedrive_r101_stage1_dn.pth`
(NDS 0.5971 / mAP 0.5038) was deliberately not used, so results attribute to the
backbone rather than to a stage-1 recipe change.

Verified before submission: both configs build; the r101 stage-1 checkpoint loads
with 0 unexpected keys, all missing keys confined to `head.motion_plan_head.*`
(new at stage 2). Params 111.6M / 111.4M total, backbone 42.5M (vs r50 ~23.5M).

## Results

| Server | Config | Job ID | Status |
| --- | --- | --- | --- |
| DGX (A100-40GB) | `bs32 ..._minS2` | 3842 | COMPLETED (8:37:20) |
| Trillium (H100) | `bs32 ..._minS2` | 666578 | COMPLETED (5:05:02) |
| Trillium (H100) | `bs32 ..._egostatus` | 666579 | COMPLETED (7:42:14) |
| DGX (A100-40GB) | `bs24 ..._minS2` | 3844 | COMPLETED (9:04:51) |
| Trillium (H100) | `bs24 ..._minS2` | 670075 | COMPLETED (5:16:01) |
| Trillium (H100) | `bs24 ..._egostatus` | 672616 | COMPLETED (8:01:23) |
| DGX (A100-40GB) | `bs32 ..._egostatus` | 3843 | FAILED — CUDA OOM (capacity) |
| DGX (A100-40GB) | `bs24 ..._egostatus` | 3845 | FAILED — CUDA OOM (fragmentation) |
| Trillium (H100) | `bs24 ..._egostatus` | 670076 | FAILED — faulty GPU on `trig0016` |
| Killarney (H100) | `bs24` both | 4390356 / 4390357 | FAILED — `sm_90` kernel mismatch |

### Anchor-matched comparison (bs24 / lr 1.5e-4) — the load-bearing table

| Model | L2 | CR | NDS | det mAP | mAP_normal |
| --- | ---: | ---: | ---: | ---: | ---: |
| r50 minS2 (2-seed anchor) | 0.3644 | 0.046% | 0.5279 | – | 0.3363 |
| **r101 minS2** (2-cluster mean) | 0.3743 | 0.043% | **0.5832** | 0.4870 | 0.2510 |
| | *ΔL2 = +0.0099 (1.4× floor)* | *at noise* | *+0.055* | | *degenerate* |
| r50 headline (3-seed anchor) | 0.3692 | 0.0400% | 0.5212 | – | 0.5533 |
| **r101 headline** (672616, 1 seed) | 0.3752 | 0.036% | **0.5842** | 0.4879 | 0.5505 |
| | *ΔL2 = +0.0060 (inside floor)* | *at noise* | *+0.063* | | *tied* |

r101 minS2 2-cluster mean is 3844 (A100, 0.3742) and 670075 (H100, 0.3744).
`mAP_normal` under minS2 is degenerate in both backbones — the map head is frozen
from stage-1 init and never trained at stage 2 — so the 0.2510 vs 0.3363 gap is
not meaningful.

### Learning-rate sensitivity — the two architectures respond in opposite directions

| Architecture | lr 3e-4 | lr 1.5e-4 | Prefers |
| --- | ---: | ---: | --- |
| minS2 (planner is the only trainable module) | 0.3847 | **0.3743** | lower, by 0.010 |
| headline (backbone trains at `lr_mult=0.1`) | **0.3595** | 0.3752 | higher, by 0.016 |

minS2 at 3e-4 is the 2-cluster mean of 3842 (0.3920) and 666578 (0.3774);
headline at 3e-4 is 666579 (single seed, H100).

### Cross-cluster reproduction of the identical minS2 config

| Setting | A100 | H100 | Δ L2 | vs 0.007 floor |
| --- | ---: | ---: | ---: | --- |
| bs32 / lr 3e-4 | 0.3920 | 0.3774 | 0.0146 | 2.1× — fails |
| **bs24 / lr 1.5e-4** | **0.3742** | **0.3744** | **0.0002** | 0.03× — near-exact |

Perception metrics agreed tightly in both settings (NDS within 0.0013, det mAP
within 0.0009). The divergence was confined to planning L2 and only at the higher
LR.

### Per-timestep L2 and CR (1s → 6s, then average)

| Run | L2 curve | CR curve |
| --- | --- | --- |
| r101 minS2 bs24 (3844) | 0.106 / 0.162 / 0.238 / 0.339 / 0.466 / 0.622 → 0.3742 | 0.000 / 0.000 / 0.013 / 0.029 / 0.059 / 0.117 → 0.049% |
| r101 minS2 bs24 (670075) | 0.108 / 0.165 / 0.240 / 0.340 / 0.466 / 0.619 → 0.3744 | 0.000 / 0.000 / 0.000 / 0.010 / 0.023 / 0.098 → 0.036% |
| r101 headline bs24 (672616) | 0.107 / 0.164 / 0.240 / 0.340 / 0.467 / 0.621 → 0.3752 | 0.000 / 0.000 / 0.000 / 0.010 / 0.039 / 0.098 → 0.036% |
| r101 minS2 bs32 (3842) | 0.106 / 0.165 / 0.245 / 0.353 / 0.491 / 0.658 → 0.3920 | 0.000 / 0.000 / 0.013 / 0.029 / 0.055 / 0.114 → 0.048% |
| r101 headline bs32 (666579) | 0.106 / 0.160 / 0.231 / 0.326 / 0.447 / 0.593 → 0.3595 | 0.000 / 0.000 / 0.007 / 0.020 / 0.066 / 0.124 → 0.048% |

All runs are identical at 1s (0.106–0.108); the entire spread between
configurations lives in the 4–6s waypoints.

### r101 headline secondary metrics (666579, bs32)

det mAP 0.4921, AMOTA 0.5000, AMOTP 1.0872, IDS 631, car_EPA 0.5504,
ped_EPA 0.5287, car min-ADE 0.5905, ped min-ADE 0.6415. Against the r50 headline
anchor that is car_EPA +0.061, ped_EPA +0.122, car ADE −0.032, ped ADE −0.082.
The AMOTA/IDS direction matches the March report (r50 stage-2 baseline: AMOTA
0.3776, IDS 1045).

## Discussion

What improved:
- **Perception, consistently and substantially**: NDS +0.055 (minS2) and +0.063
  (headline), det mAP ≈ 0.487, AMOTA 0.500, IDS 631. Magnitudes match the March
  report, so r101's perception benefit is real and architecture-independent.
- **Motion metrics** under the headline: car_EPA +0.061, ped_EPA +0.122, car ADE
  −0.032, ped ADE −0.082 — the largest motion-metric gain from any single change
  in the planning era.

What regressed or stayed flat:
- **Planning L2 gained nothing.** +0.0060 on the headline (inside the noise floor)
  and +0.0099 on minS2 (1.4× the floor, so marginal and in the wrong direction).
  CR is at noise everywhere.

Likely explanation:
- Planning L2 in this codebase appears insensitive to backbone capacity, in the
  same way it is insensitive to stage-1 recipe (all six r50 stage-1 swaps moved it
  ≤0.012). The r101 features are better *for perception* — that shows up cleanly
  in NDS, mAP, AMOTA, and the agent motion metrics — but the planner does not
  convert them into better ego trajectories. This is the paper's
  "perception salience is not planning relevance" thesis appearing along the
  backbone-capacity axis rather than the token-interface axis, and it is arguably
  the most useful thing in this report.

**Stream E is confirmed on a second backbone.** Freezing the stage-2 backbone
costs 0.0009 L2 on r101 (frozen 0.3743 vs trainable 0.3752) and cost 0.0006 on
r50 (`_frozenpercep` 0.3698 vs headline 0.3692). Both are far inside noise. The
"stage-2 backbone training is empty for planning" claim therefore generalizes
beyond r50 at this schedule, which strengthens rather than qualifies it.

### Corrections to the first draft of this report (2026-07-25)

Two claims in the original draft were artifacts of unmatched hyperparameters and
are retracted:

1. **"R101 improves planning when the backbone trains and degrades it when
   frozen."** Retracted. That rested on comparing the bs32 headline (0.3595)
   against the bs32 minS2 (0.3847). At matched hyperparameters the two are within
   0.0009 of each other. The apparent asymmetry was the LR: the headline prefers
   3e-4 (its backbone sees 3e-5 at `lr_mult=0.1`) while minS2, whose only
   trainable module is the planner, prefers 1.5e-4. Opposite responses to the same
   knob produced a gap that looked architectural.
2. **"First instance of A100/H100 divergence on L2 (0.0146), so distrust
   cross-cluster L2 below ~0.015."** Retracted. At the matched LR the same two
   clusters agree to 0.0002. The divergence was run-to-run instability induced by
   the 3e-4 LR, not hardware numerics. `project_h100_repro` should **not** be
   extended to L2 on this evidence.

The general lesson: at 3e-4 this architecture is measurably less reproducible
(0.0146 spread on identical configs) than at 1.5e-4 (0.0002). Any future LR
increase should be validated for reproducibility, not just mean metric.

Recommended model:
- **None — do not adopt r101 into any current stage-2 architecture.**

Why:
- At matched hyperparameters r101 is at parity or marginally worse on both primary
  planning metrics, while costing roughly 2× the compute of r50 (8–9 h vs the r50
  anchors' runtimes, at 1408×512 vs 704×256) and requiring ≥48 GB GPUs for the
  headline variant. The perception gains are real but do not serve the paper's
  planning claims. If an r101 row is wanted in the paper, it belongs as evidence
  *for* the perception-vs-planning decoupling argument, not as a performance
  headline.

## Infrastructure findings

- **A100-40GB cannot train the r101 headline.** At 8 samples/GPU it OOMs on
  capacity (3843: 36.71 of 39.39 GiB). At 6 samples/GPU it OOMs on
  **fragmentation** (3845: only 20.70 GiB allocated but 34.96 GiB reserved, a
  single 4.64 GiB allocation failing). The latter is likely fixable with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` rather than a further batch
  reduction — untested. minS2 fits comfortably: 33.8 GB at 8/GPU, 25.9 GB at
  6/GPU. H100-80GB runs the headline without issue.
- **Killarney H100 is unusable for this codebase as configured.**
  `killarney_run.sh` binds `docker/foresight_cuda118.sif` (torch 2.0.1, compiled
  for L40S `sm_89`), the only `.sif` on that cluster, so H100 jobs die immediately
  with `CUDA error: no kernel image is available for execution on the device`. The
  four working H100 clusters all use `foresight_cuda118pytorch21.sif`. This is why
  every Killarney result on record is L40S. Fixing it means copying the ~11.5 GB
  pytorch21 image there and pointing the script at it for H100 submissions.
- **Trillium node `trig0016` has a faulty GPU** (`NVML: Failed to get usage(16):
  GPU requires reset`, ~14.6k occurrences). Job 670076 died on its first NCCL
  collective (`WorkNCCL(SeqNum=1, OpType=BROADCAST)` timeout) yet SLURM kept
  reporting RUNNING because the batch script wedged — it held 4 H100s for 5h16m
  producing nothing. Resubmitted as 672616 with `--exclude=trig0016`, which
  succeeded. Worth reporting to SciNet support.
- **Wall-time calibration.** All r101 stage-2 runs finished in 5–9 h
  (`bs24` is ~30 min *longer* than `bs32` — more iterations outweigh the lighter
  per-iteration cost). `11:59:00`, the default in both `trillium_run.sh` and
  `killarney_run.sh`, is sufficient; the 24 h overrides used early in this batch
  were unnecessary and cost queue priority.
- **Trillium schedules two 4-GPU jobs in parallel**; DGX (5 GPUs, 1 held by a
  long-running `lock_display` job) serializes them. Trillium is the better target
  for paired experiments.

## Future Work

- **`r50 headline @ lr 3e-4` — submitted as DGX 3846.** The best r101 planning
  number in this batch (0.3595) came from `headline + lr 3e-4`, and no r50
  headline has ever been trained at that LR: 220 of 248 stage-2 configs inherit
  `lr=1.5e-4` from the r50 baseline. Config
  `..._egostatus_lr3e4.py` differs from the 3-seed anchor by that one line. This
  distinguishes "r101 is better at 3e-4" from "the headline architecture prefers
  3e-4 regardless of backbone." **If the latter, the paper's main comparator
  (0.3692) is under-tuned and the LR axis is unswept**, which would touch every
  comparison in `2026_04_28_results_summary.md`. Worth resolving before the paper
  locks.
- **Second seed on the anchor-matched r101 headline** (672616 is single-seed). Low
  priority: it confirms a null result rather than deciding anything.
- **r101 + `_dn` stage-1** (`sparsedrive_r101_stage1_dn.pth`, NDS 0.5971) — only
  worth running if the perception-side r101 story is being developed for its own
  sake, since planning L2 is the metric that did not move.
- **Re-measure latency if any r101 row enters the paper.**
  `reports/2026_05_10_latency_breakdown.md` reports 166.7 ms → 77.2 ms at r50
  704×256; those speedup claims do not transfer to r101 1408×512 and would need
  `tools/benchmark_stages.py` rerun.
- **Update `reports/2026_03_12_r101_backbone.md`** with a pointer here. Its
  recommendation ("r101 should be the backbone for all serious
  performance-optimized experiments") is now known not to hold for planning under
  the current stage-2 architectures.
