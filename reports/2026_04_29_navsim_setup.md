# NavSim v1 Integration Plan

## Abstract

Plan and initial scaffolding for adding NavSim v1 (navtrain / navtest) as a second
training and evaluation track for SparseDrive, alongside the existing nuScenes
pipeline. Approach: vendor the official `navsim` Python package into `navsim/`
at the repo root, register a new `SparseDriveAgent` that wraps the existing SparseDrive head behind
NavSim's `AbstractAgent` interface, and run training and PDM-Score evaluation
through NavSim's Hydra runner. ForeSight's existing nuScenes runners
(`tools/dist_train.sh`, `tools/dist_test.sh`) stay untouched.

## Quickstart

```bash
# 1. Build the navsim docker image (once)
docker build --network=host -f docker/Dockerfile.navsim_cuda118pytorch21 \
    -t foresight_navsim:cuda118pytorch21 .

# 2. Smoke-test imports + forward pass + box conversion (no data needed)
docker run --gpus all --rm -v $PWD:/workspace/ForeSight -w /workspace/ForeSight \
    -e FORESIGHT_ROOT=/workspace/ForeSight \
    foresight_navsim:cuda118pytorch21 python scripts/navsim_smoke_test.py
# Expect: All 7 checks passed.

# 3. Download navmini data + nuplan maps (~12GB + 1GB; 5-10 min total)
NAVSIM_EXP_ROOT=/tmp/navsim_exp bash scripts/navsim_setup.sh download_navmini
cd data && wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip
unzip nuplan-maps-v1.1.zip && rm nuplan-maps-v1.1.zip

# 4. End-to-end smoke on a real scene
docker run --gpus all --rm -v $PWD:/workspace/ForeSight -w /workspace/ForeSight \
    -e FORESIGHT_ROOT=/workspace/ForeSight \
    -e OPENSCENE_DATA_ROOT=/workspace/ForeSight/data/openscene \
    -e NUPLAN_MAPS_ROOT=/workspace/ForeSight/data/nuplan-maps-v1.0 \
    foresight_navsim:cuda118pytorch21 python scripts/navsim_navmini_smoke.py
# Expect: prediction shape: (1, 8, 2)

# 5. Run a few Hydra training steps on navmini
docker run --gpus all --rm -v $PWD:/workspace/ForeSight -w /workspace/ForeSight \
    -e FORESIGHT_ROOT=/workspace/ForeSight \
    -e OPENSCENE_DATA_ROOT=/workspace/ForeSight/data/openscene \
    -e NUPLAN_MAPS_ROOT=/workspace/ForeSight/data/nuplan-maps-v1.0 \
    -e NAVSIM_EXP_ROOT=/tmp/navsim_exp \
    foresight_navsim:cuda118pytorch21 \
    bash scripts/navsim_train.sh navmini \
        'train_logs=[<your_logs>]' 'val_logs=[...]' \
        'dataloader.params.batch_size=1' 'dataloader.params.num_workers=0' \
        '~dataloader.params.prefetch_factor' 'dataloader.params.pin_memory=false' \
        'trainer.params.max_epochs=1' 'trainer.params.precision=32' \
        'trainer.params.strategy=auto' '+trainer.params.devices=1' \
        'trainer.params.limit_train_batches=5' 'trainer.params.limit_val_batches=0'
# Expect: Trainer.fit stopped: max_epochs=1 reached. With loss_step decreasing.
```

## Intro

NavSim is the leading benchmark for end-to-end planning on nuPlan-style data, and
its closed-loop PDM-Score metric is the field-standard comparator for VAD-,
SparseDrive-, and DiffusionDrive-class methods. ForeSight currently runs only on
nuScenes with open-loop L2 / collision metrics, which makes our results hard to
compare against the NavSim leaderboard.

Question:
- How do we add NavSim training (`navtrain`) and PDM-Score evaluation (`navtest`)
  to ForeSight with the smallest sustainable change to the existing SparseDrive
  pipeline?

## Why we vendor `navsim` instead of converting data

NavSim is not a dataset; it is a Hydra + PyTorch-Lightning runtime built on top of
nuPlan that owns:

- The `AbstractAgent` training interface (`navsim/agents/abstract_agent.py`)
- The per-token feature/target cache (`navsim/planning/training/dataset.py`)
- The PDM-Score closed-loop simulator (`navsim/evaluate/pdm_score.py` plus
  `navsim/planning/simulation/...`) which needs nuPlan's LQR + bicycle model

Trying to bend `tools/dist_train.sh` and `tools/dist_test.sh` to consume NavSim
data would either lose the metric cache (no PDM-Score) or duplicate the
closed-loop simulator from scratch. DiffusionDrive (CVPR 2025) ships exactly this
"vendor the package, add a new agent" pattern, and we follow it.

## Decisions (confirmed with user)

1. **Storage and execution targets**: local for smoke tests and debugging;
   Killarney for initial training and evaluation. DGX, Apollo, Narval, Trillium
   come later if needed. Killarney is geo-blocked but already wired up via
   `scripts/killarney_run.sh` and runs WandB offline.
2. **Scope, in order**:
   1. Stage-2 planning-only (no detection / map / motion losses) using a frozen
      or pretrained perception block, to land PDM-Score numbers fast.
   2. Stage-1 with detection and map auxiliary losses, after the planning-only
      path is green.
3. **Camera reduction**: keep 6 of NavSim's 8 cameras to match the nuScenes rig
   that SparseDrive expects. Drop `cam_l1` and `cam_r1` (the pure side-facing
   cameras). Mapping below.
4. **Anchors**: regenerate planning K-means anchors on `navtrain` futures rather
   than reusing DiffusionDrive's `kmeans_navsim_traj_20.npy`. We can borrow theirs
   for a baseline if our own anchors regress.
5. **Environment**: build a new Docker image branched from
   `docker/Dockerfile.cuda118pytorch21` (proven across all servers). Single image
   with a single `foresight` conda env that satisfies both stacks; no dual envs.

## Camera mapping (NavSim 8 → nuScenes 6)

| nuScenes name | NavSim name | Rationale |
| --- | --- | --- |
| `CAM_FRONT` | `cam_f0` | front |
| `CAM_FRONT_LEFT` | `cam_l0` | front-left |
| `CAM_BACK_LEFT` | `cam_l2` | rear-left |
| `CAM_FRONT_RIGHT` | `cam_r0` | front-right |
| `CAM_BACK_RIGHT` | `cam_r2` | rear-right |
| `CAM_BACK` | `cam_b0` | back |
| _dropped_ | `cam_l1` | pure left side |
| _dropped_ | `cam_r1` | pure right side |

This ordering matches `core/box3d.py` ordering and lets the existing
backbone + FPN + projection-mat code stay untouched.

## Trajectory / horizon mismatch

| Aspect | nuScenes (current) | NavSim | Fix |
| --- | --- | --- | --- |
| Future steps | 6 (3 s @ 0.5 s) | 8 (4 s @ 0.5 s) | bump `fut_ts=8` in `motion_planning_head.py`, regen anchors |
| History frames | 3–4 | 4 (`num_history_frames=4`) | OK |
| GT box dim | 9 (nuScenes) | 8 (nuPlan) | helper in feature builder |
| Driving cmd | 3-dim onehot | 4-dim onehot | extend `ego_status` dim |

## Phase plan

### Phase 0 — Environment + scaffolding (this PR)
- Vendor `navsim` package into `navsim/` (copied verbatim from
  DiffusionDrive's `navsim/`, which is the upstream `autonomousvision/navsim`
  package with the DiffusionDrive agent files added).
- New Docker image: `docker/Dockerfile.navsim_cuda118pytorch21` branched off
  `Dockerfile.cuda118pytorch21`; bump conda python to 3.9 (NavSim setup.py
  hard-pins `>=3.9`), keep torch 2.1.2 + mmcv 1.7.2 + flash-attn 2.3.2,
  add `requirement_navsim.txt` for nuplan-devkit + hydra + ray + lightning.
- `scripts/navsim_setup.sh`: env vars (`OPENSCENE_DATA_ROOT`, `NUPLAN_MAPS_ROOT`,
  `NAVSIM_DEVKIT_ROOT`, `NAVSIM_EXP_ROOT`) and download dispatch wrappers.

### Phase 1 — `SparseDriveAgent` skeleton
Files under `navsim/navsim/agents/sparsedrive/`:
- `sparsedrive_agent.py` — wraps SparseDrive head behind `AbstractAgent`.
- `sparsedrive_features.py` — 8→6 camera reduction, `lidar2img` /
  `projection_mat` from `sensor2lidar_*` + `cam_intrinsic`, `ego_status`,
  history stacking.
- `sparsedrive_targets.py` — future trajectory (8 poses), optional
  `gt_boxes` / `gt_labels` for det/motion auxiliary losses (nuPlan box → 11-dim
  SparseDrive anchor format).
- `sparsedrive_config.py` — dataclass with backbone / head / loss flags.
- `transfuser_config.py`-style YAML at
  `navsim/navsim/planning/script/config/common/agent/sparsedrive_agent.yaml`.

### Phase 2 — Head adaptation
- Add a `fut_ts` constructor argument to `motion_planning_head.py` so we can flip
  to 8 without touching the nuScenes path.
- Extend `ego_status_dim` for the 4-dim driving command.
- Regenerate planning K-means anchors on navtrain.

### Phase 3 — Train + evaluate
- `scripts/navsim_train.sh` wrapping
  `python navsim/navsim/planning/script/run_training.py
   agent=sparsedrive_agent train_test_split=navtrain ...`.
- `scripts/navsim_eval.sh` wrapping `run_pdm_score.py`.
- Killarney submission via `scripts/killarney_run.sh` (Apptainer + offline WandB).

### Phase 4 — Validation
- Smoke test on `navmini` first.
- Compare to DiffusionDrive's published 88.1 PDMS as a north star (do not expect
  to match; they are a tuned baseline).
- Sanity-check open-loop L2 / collision against our nuScenes numbers.

## Risks and open questions

- **Python 3.9 bump risk**: mmcv 1.7.2 builds from source so we are not blocked by
  wheel availability, but flash-attn 2.3.2 pre-builds for 3.9 are needed. If they
  are missing the build will fall back to source which is slow but works.
- **nuplan-devkit pins**: their `requirements.txt` pins `numpy==1.23.4`,
  `torch==2.0.1`, `pytorch-lightning` and a few SQLAlchemy / Shapely versions.
  We install with `--no-deps` for nuplan-devkit and supply a curated dependency
  set that is compatible with torch 2.1.2.
- **Storage on Killarney**: navtrain blobs are ~500 GB. Cache must live under
  `/scratch/spapais/...`; we will not put it under `/home`.
- **SparseDrive perception losses on nuPlan boxes**: 8-dim nuPlan box → 11-dim
  SparseDrive anchor format mapping is non-trivial. Phase 1 stays planning-only.

## Results

| Server | Config | Job ID | Status |
| --- | --- | --- | --- |
| local | smoke test (7 import/build/forward checks) | n/a | COMPLETED |
| local | navmini E2E smoke (real Scene through agent) | n/a | COMPLETED |
| local | navmini Hydra training step 1 (forward+loss+backward) | n/a | COMPLETED |
| local | navmini Hydra multi-step training (5 steps, loss decreases) | n/a | COMPLETED |
| killarney | navsim 7/7 smoke (.sif on /scratch, ops rebuilt) | 3365954 | COMPLETED |
| killarney | navmini 5-step Hydra training | 3365958 | COMPLETED |
| killarney + dgx | navtrain + navtest data download (~700 GB) | n/a | COMPLETED |
| killarney | navtest metric_cache build (12,146 entries) | 3371101 | COMPLETED |
| killarney | navtrain k-means anchors (real, plan-side only) | 3372479 | COMPLETED |
| killarney | navtrain dataset_cache (Ray-parallel, 103k entries) | 3386610 | COMPLETED |
| dgx | navtrain training, fp32, 500 batches | 3705 | COMPLETED (clean: loss 1.29 → 0.21) |
| dgx | navtrain training, fp32, 5000 batches | 3706 | NaN at step 4233/5000 (**blocker — see below**) |
| killarney | navtrain training, fp32, 2000 batches | 3387927 | NaN + cuda:0 vs cpu device crash (no stage-1 warmstart) |
| killarney | navtrain training, fp32, 5000 batches, stage-1 warmstart | 3388590 | PENDING (stage-1 ckpt + device fix — see 2026-05-01 update) |

### Smoke test (2026-04-29, local, GPU)

`docker run --gpus all ... foresight_navsim:cuda118pytorch21 python scripts/navsim_smoke_test.py`

```
[OK] import navsim base
[OK] import sparsedrive agent
[OK] import foresight plugin (registers mmcv heads)
[OK] build sparsedrive head from navsim variant config (model class: SparseDrive)
[OK] feature builder on synthetic AgentInput
       img shape: (6, 3, 256, 704)
       projection_mat shape: (6, 4, 4)
       ego_status shape: (8,)
[OK] forward pass on synthetic batched features (eval mode)
       trajectory type: Tensor
       trajectory shape: (1, 8, 2)
All 6 checks passed.
```

What this proves:
- The new conda env (python 3.9 + torch 2.1.2 + mmcv 1.7.2 + flash-attn 2.3.2 +
  nuplan-devkit + hydra + lightning) coexists in one image without import-time
  conflicts.
- `SparseDriveAgent` wiring (sys.path boost, plugin import, mmcv config →
  `build_detector`) works end-to-end with the navsim-variant config
  (`ego_fut_ts=8`, `num_driving_cmds=4`).
- The 8→6 camera reduction produces tensors with the shapes the head expects.
- A full forward pass through SparseDrive (backbone→FPN→det/map/motion/plan
  heads→decoder) returns a `(B, ego_fut_ts, 2)` ego trajectory when given
  zero-input synthetic features — proving the navsim variant config and the
  three additional `num_driving_cmds` plumbing edits (motion_planning_head,
  motion_blocks, decoder) all reshape consistently.

What this does NOT prove yet:
- Training-mode forward + loss against real navsim GT (the empty-GT path
  through det_head's DN sampler still hits a 10/12 shape mismatch in
  cls_wise_reg_weights). Needs the real nuPlan box→11-dim conversion or a
  with_det/with_map gate that doesn't break motion_plan_head's anchor_encoder
  dependency on det_head.
- PDM scoring — needs metric cache + nuplan maps + a real navtest pass.

### navmini E2E smoke (2026-04-29, local, GPU)

`docker run ... python scripts/navsim_navmini_smoke.py`

```
scene token: 8bc34517e08758ff
log: 2021.10.05.07.10.04_veh-52_01442_01802  map: sg-one-north
feature keys: img, projection_mat, image_wh, ego_status, gt_ego_fut_cmd
target keys:  gt_ego_fut_trajs, gt_ego_fut_masks, gt_bboxes_3d, gt_labels_3d
gt_bboxes_3d: (5, 11)  gt_labels_3d: (5,)
prediction shape: (1, 8, 2)
```

What this proves:
- The vendored navsim `SceneLoader` resolves the `${OPENSCENE_DATA_ROOT}/
  navsim_logs/<split>` and `${OPENSCENE_DATA_ROOT}/sensor_blobs/<split>`
  layout produced by our `scripts/navsim_setup.sh download_navmini`.
- The nuPlan annotations on a real scene convert cleanly to 5 SparseDrive
  11-dim boxes + class indices.
- The full SparseDrive pipeline (backbone → FPN → det/map/motion/plan
  heads → decoder) runs end-to-end on real navmini sensor data and
  produces an `(B, ego_fut_ts, 2)` ego trajectory.

What it does NOT prove yet:
- Training-mode forward + loss against real navmini GT. Needs auditing of
  the det/map sampler paths because the empty-GT injection that worked
  for synthetic features may collide with the real GT shapes via
  `cls_wise_reg_weights`.
- PDM-Score eval on navtest.

### Killarney smoke + navmini training (2026-04-30)

`foresight_navsim.sif` (12 GB) lives on Killarney at
`/scratch/spapais/ForeSight/docker/` because `/home` is 100% full. Same
constraint forced the navmini data and nuplan-maps onto `/scratch`:

```
/scratch/spapais/data/openscene/navsim_logs/mini/        12 GB
/scratch/spapais/data/nuplan-maps-v1.0/                  1.4 GB
/scratch/spapais/ForeSight/docker/foresight_navsim.sif   12 GB
```

`scripts/killarney_navsim_run.sh` defaults updated to point at these.

Setup gotchas on Killarney specifically:
- `deformable_aggregation_ext.cpython-39-*.so` checked into the repo on
  Killarney was from Feb 2026 (different torch ABI). Rebuilt inside the
  apptainer via
  `apptainer exec -c -e --pwd .../ops --bind=/home/spapais/ForeSight:/workspace/ForeSight $SIF python setup.py build_ext --inplace`.
  `setup.py develop` fails because the env's site-packages is read-only;
  `build_ext --inplace` is the right call.
- `tools/make_navsim_kmeans_placeholder.py` must be run once on Killarney
  to seed `data/kmeans/kmeans_motion_navsim_6.npy` and
  `kmeans_plan_navsim_6_cmd4.npy` (the .npy files are gitignored, not pushed).

Smoke test (job 3365954, 1× L40S, ~1 min): all 7 checks pass.

5-step navmini training (job 3365958, 1× L40S, ~16 s wall time for the
training loop):

```
Epoch 0: 100%|██████████| 5/5 [00:13<00:00, 0.36it/s]
Trainer.fit stopped: max_epochs=1 reached.
Step 1:  loss_step=1.76e+4    planning_loss_reg=2.50
Step 5:  loss_step=514        planning_loss_reg=0.806
Epoch:   loss_epoch=5.43e+3   planning_loss_reg=1.82  planning_loss_status=0.76
```

86.1 M params (85.9 M trainable) — same loss trajectory as local. Image,
data, code, and Hydra training pipeline all green on the cluster.

### First Hydra training step on navmini (2026-04-29)

`scripts/navsim_train.sh navmini ...` produces a complete training step
through the navsim Lightning runner with all four heads active:

```
Epoch 0:  33%|███▎      | 1/3 [00:01<00:02,  0.79it/s]
  train/det_loss_cls_0_step=2.13e+3  train/det_loss_box_0_step=17.9
  train/map_loss_cls_0_step=54.3     train/map_loss_line_0_step=0.0
  train/motion_loss_cls_0_step=0.0   train/motion_loss_reg_0_step=0.0
  train/planning_loss_cls_0_step=0.027 train/planning_loss_reg_0_step=2.51
  train/planning_loss_status_0_step=1.04
  train/loss_step=1.77e+4
```

Forward + loss + backward + logging all green. Loss magnitudes are noisy
because the training-mode placeholders (empty agent futures, empty map GT)
mean det/map heads see no real supervision yet.

Wiring fixes that landed to get here:
- `sparsedrive_agent.yaml` was pointing at the baseline nuScenes config; now
  points at `sparsedrive_r50_stage2_navsim_planonly.py`.
- Three additional hardcoded `3 *` driving-cmd literals replaced with
  `num_driving_cmds`: in `motion/target.py` (PlanningTarget.sample), in
  `motion/decoder.py` (HierarchicalPlanningDecoder.decode), and in
  `motion/motion_blocks.py` (MotionPlanningRefinementModule).
- Target builder produces RAW 9-dim boxes `[X,Y,Z,W,L,H,YAW,VX,VY]` (the
  head's `encode_reg_target` does the log/sin/cos encoding); not the
  pre-encoded 11-dim form I had originally.
- Feature builder returns a 10-dim `ego_status` matching the nuScenes
  layout `[acc_xyz, rot_rate_xyz, vel_xyz, steer]`; nuPlan ships only 2D
  acc + 2D vel so the unavailable channels are zero-padded.
- Agent injects `T_global_inv` along with `T_global` into `img_metas`
  for the temporal cache.

Multi-step training also works after switching the agent's
`T_global` / `T_global_inv` injections from CUDA torch tensors to CPU
numpy arrays — `instance_bank.get` calls `np.stack` on them and that
needed numpy. 5 training steps land cleanly with measurable loss decrease:
`loss_step` drops from 1.77e4 → 476 across the 5-batch run on a single
log; `planning_loss_reg` drops 2.51 → 0.80.

### nuPlan -> 11-dim box conversion (2026-04-29)

`navsim_boxes_to_sparsedrive()` in
`navsim/navsim/agents/sparsedrive/sparsedrive_features.py` maps nuPlan-style
`Annotations` to SparseDrive's 11-dim anchor format (`[X, Y, Z, log_W,
log_L, log_H, SIN_YAW, COS_YAW, VX, VY, VZ]`). Class remapping:

| nuPlan name | nuScenes index | nuScenes name |
| --- | --- | --- |
| `vehicle` | 0 | car |
| `pedestrian` | 8 | pedestrian |
| `bicycle` | 7 | bicycle |
| `traffic_cone` | 9 | traffic_cone |
| `barrier` | 5 | barrier |
| `czone_sign` | 5 | barrier (coalesced) |
| `generic_object` | 5 | barrier (coalesced) |
| `ego` | -1 | dropped |

Smoke test #7 verifies the shape, the ego-drop, and the log-WLH /
sin/cos-heading encoding.

### num_driving_cmds plumbing edits

The hardcoded literal `3` (number of nuScenes driving commands) appeared in
three places in the SparseDrive code that all needed an additive
`num_driving_cmds=3` constructor arg, plus the navsim variant config has to
forward it explicitly:
- `projects/mmdet3d_plugin/models/motion/motion_planning_head.py:1665`
- `projects/mmdet3d_plugin/models/motion/motion_blocks.py:115` and `:125`
- `projects/mmdet3d_plugin/models/motion/decoder.py:182,183`

All keep the nuScenes default of 3, so the existing nuScenes path is
untouched.

### Build issues encountered + fixes

1. `apt-get` failed inside `docker build` due to docker-default DNS being
   unreachable from the host. Fixed by passing `--network=host` to the build
   command (now noted in the Dockerfile header).
2. `mmcv 1.7.2` source build crashed with
   `ModuleNotFoundError: No module named 'pkg_resources'` because
   `pip install --upgrade setuptools` pulled setuptools 80, where
   `pkg_resources` is no longer top-level importable from a setup.py. Fixed by
   pinning `setuptools==65.5.1` (also navsim's own pin).
3. `deformable_aggregation_ext.cpython-39-*.so` left over on the host from a
   prior python-3.9 build had stale torch symbols, shadowed the in-image fresh
   build via the bind-mount. Fixed by rebuilding inside the new container; the
   3.8 .so on the host is left untouched so the original
   `Dockerfile.cuda118pytorch21` image keeps working.
4. NavSim variant config pointed at navsim-specific k-means anchors that don't
   exist yet. Added `tools/make_navsim_kmeans_placeholder.py` to write random
   placeholders with the correct shapes ((10, 6, 16, 2) motion, (4, 6, 8, 2)
   plan); real anchors must be regenerated from navtrain after data lands.

## Results table (PDM-Score on navtest, 12146 scenarios)

Higher is better for all sub-metrics.

| Job | Config | Epochs | PDMS | NAC | DAC | EP | TTC | Comf | DDC | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| _ref_ DiffusionDrive (published) | — | — | **88.1** | — | — | — | — | — | — | leaderboard north star |
| **3727 → 3746 eval** | `navsim_planonly` (baseline) fp32 bs=8 | 1 | **0.6502** | 0.9319 | 0.7893 | 0.5976 | 0.8518 | 0.9817 | 0.9207 | First real PDMS — image norm + timestamp + abs SE3 + photo-metric all in |
| **3751 → 3753 eval** | `navsim_planonly_minS2` fp32 bs=8 | 1 | **0.6489** | 0.9304 | 0.7610 | 0.6159 | 0.8454 | 0.9998 | 0.8795 | Net flat vs baseline. Redistributes errors: EP/Comf up (planner more confident), DAC/DDC down (worse on edges). 1 epoch likely too short to see minS2's full benefit |
| **3755 → 3756 eval** ❌ | + hflip shared-seed (REVERTED) | 1 | **0.3076** | 0.6695 | 0.4697 | 0.2957 | 0.5257 | 0.5297 | 0.7083 | -0.34 PDMS regression. Two bugs: (1) `compute_features` runs at PDM eval too, mirroring ~50% of test predictions; (2) T_global stays in unmirrored UTM but lidar coords mirror, breaking `instance_bank` warp. Reverted; will redesign with training-only gating + T_global mirror handling later |
| **3757 → 3758 eval** ❌ | + minS2 arch knobs (cumulative_refinement, deformable_*, multimode) | 1 | **0.5670** | 0.8833 | 0.7144 | 0.5597 | 0.7631 | 0.9954 | 0.8567 | -0.082 PDMS regression. Adds learnable params (per-layer refinement, deformable sampling at every waypoint) that need >1 epoch to settle. Reverted; could revisit at 10-epoch scale |
| **3759 → 3760 eval** | 10ep minS2 (promotion gate) | 10 | **0.6138** | 0.9091 | 0.7342 | 0.5877 | 0.8134 | 0.9962 | 0.8569 | Train loss dropped 0.183→0.103 but **all** PDMS sub-metrics regressed vs 1ep. Overfitting: planner has too few trainable params + open-loop L1 ≠ closed-loop PDMS. Pivot to stage-1 perception GT |
| **3765 → 3766 eval** | stage-1 1ep (det+map+motion + unfreeze, **placeholder anchors**) | 1 | **0.6498** | **0.9454** | 0.7621 | 0.5939 | **0.8756** | 0.9999 | **0.9203** | Real perception supervision lifts NAC +0.02, TTC +0.03, DDC +0.04 vs minS2. DAC stuck (placeholder anchors cap it). EP regressed -0.03. PDMS basically flat |
| **3768 → 3769 eval** | stage-1 1ep + **real anchors** (33k navtrain plan + 1.9M motion) | 1 | **0.6387** | 0.9189 | 0.7541 | 0.6193 | 0.8246 | 0.9988 | 0.8778 | Slight regression vs placeholder. EP restored to 0.62 (placeholder cost was real); but NAC/TTC/DDC lost their bumps. 1ep too short — 10ep is the cleaner test |
| **3770 (in flight)** | stage-1 10ep + real anchors | 10 | TBD | — | — | — | — | — | — | Overnight; the real promotion gate. nuScenes minS2 was tuned for 10ep |
| TBD | overnight: best 1-epoch config | 10 | TBD | — | — | — | — | — | — | Promotion gate after Tier-2 sweeps |

## Current State (2026-05-03, updated from 2026-05-01)

Everything from Phase 0 → Phase 4 first-checkpoint is in place. 4-GPU DDP
is working and the first full navtrain run is in flight on DGX.

**Working:**
- Image, .sif on DGX + Killarney, smoke 7/7 on both
- Full navtrain (700 GB) + navtest (220 GB) + nuplan-maps on DGX `/raid/home/spapais/datasets/` and Killarney `/home/spapais/projects/aip-swasland/datasets/`
- Apollo has data too but is unusable (V100s, no flash-attn 2.x support)
- Ray-parallel dataset_caching builds in ~46 min (vs 5 hr serial)
- navtest metric_cache built (Killarney, 12k entries, 3 GB) — ready for PDM-Score eval
- Real navtrain plan-side k-means anchors generated
- 500-batch training (`3705`) clean: loss 1.29 → 0.21, no NaN
- `sparsedrive_agent.yaml` now loads stage-1 ckpt via `foresight_pretrained`
- `sparsedrive_agent.forward` now moves all target tensors to `img.device` (fixes cuda:0 vs cpu device mismatch in planning loss)
- 4-GPU DDP fixed (`a69e848`): `DDPStrategy(find_unused_parameters=True, static_graph=True)` handles frozen det/map heads + gradient checkpointing

**In flight:**
- DGX job 3724: fp32, bs=2/GPU (8 total), grad_clip=1.0, 4×A100, stage-1 warmstart, `skip_perception_kv=True`, `num_map=0`. Step ~2700/10639 at epoch 0. Key milestone: pass step ~4300 (where job 3706 NaN'd) to confirm stability.

**Feature builder upgrades landed (2026-05-03, pending cache rebuild):**
- **ImageNet normalization** (was missing entirely — top suspect for
  training instability): `compute_features` now subtracts
  `mean=[123.675, 116.28, 103.53]` and divides by
  `std=[58.395, 57.12, 57.375]` after promoting uint8 RGB to float32 in
  [0, 255]. ResNet50 backbone now sees inputs in roughly the same
  distribution it was pretrained on instead of [0, 1]. On a real
  navmini token: per-cam `img.mean ≈ -0.4`, `img.std ≈ 0.9` (was 0.5
  and 0.3 before).
- **Real timestamps** plumbed via a new `EgoStatus.timestamp` field
  (microseconds; same source as `Frame.timestamp`), populated in both
  `AgentInput` construction paths. `compute_features` emits seconds.
  Replaces the agent's `img.new_zeros(bs)` placeholder which made
  `instance_bank.get`'s `|Δt| <= 2 s` gate always pass and the
  temporal warp fire on stale `cached_anchor` across scene boundaries.
- **Photo-metric distortion** (brightness/contrast/saturation/hue) on
  each cam image, mirroring the nuScenes
  `PhotoMetricDistortionMultiViewImage` constants. Cache-time only
  (one persisted draw per token), but adds meaningful color/lighting
  diversity at zero training-step cost. No GT touched, so safe under
  planning-only.
- `sparsedrive_features.py` emits per-history-frame `T_global` /
  `T_global_inv` (shape `(num_history, 4, 4)`) using **absolute** nuPlan
  global SE2 (z=0 flat-world promotion). To make this possible without
  changing the `compute_features(agent_input)` interface, vendored
  navsim's `EgoStatus` gained an optional `global_ego_pose` field
  populated alongside the local `ego_pose` in both `AgentInput`
  construction paths (`Scene.get_agent_input` for cache build,
  `AgentInput.from_scene_dict_list` for PDM eval). Cross-call warping in
  `instance_bank` / `InstanceQueue` is now consistent across sequential
  calls — both calls reference the same absolute frame, so
  `T_temp2cur = T_global_inv(curr) @ T_global(temp)` gives the real
  ego-motion compensation the temporal cache needs. The agent peels off
  the current-frame matrices for `img_metas` (replaces the identity
  placeholder). Float64 throughout to preserve UTM-scale translation
  precision; instance_bank's `cached_anchor.new_tensor` downcasts at the
  GPU boundary.
- `ego_status[5]` (rot_rate_z) is now filled from Δheading/0.5 s between
  the last two frames (heading delta is invariant to frame origin, so
  local pose is fine; wrapped into (-π, π] for safety). Other rot_rate
  channels stay zero. `ego_status[9]` (steer) stays zero too:
  `tire_steering_angle` is not exposed via `AgentInput.EgoStatus` — it
  only lives in nuPlan's full `EgoState` behind the scene loader.

**Stage-2 planning-only gap survey** (2026-05-03): a Plan agent compared
the navsim feature/agent path to the nuScenes pipeline. The big-impact
items above all land in this batch. Stage-1 perception items —
real det/map/motion GT plus a custom variable-length collate_fn — are
correctly out of scope for the current planning-only setup
(`with_det/with_map/with_motion_plan=True` only because
`motion_plan_head` consumes infra from `det_head`; det/map are frozen by
the agent and `compute_loss` filters their losses out of backprop;
`skip_perception_kv=True` and `num_map=0` already short-circuit any
runtime dependency on real perception cross-attn). Lidar depth
supervision (`MultiScaleDepthMapGenerator` + `depth_branch`) is the
remaining stage-2-safe gap; deferred to keep this batch focused.

**Deferred: horizontal-flip augmentation.** A first cut applied 50% hflip
inside `compute_features` (image LR flip, lidar2img X-column negation,
left↔right cmd swap), but `compute_targets` runs as a separate builder
that doesn't see the flip choice — `gt_ego_fut_trajs` would stay in
original coords on flipped tokens, breaking supervision symmetry. Doing
this properly needs a shared per-token seed (or per-cache flip flag) so
both builders make the same choice. Reverted for now; will revisit
alongside the on-the-fly augmentation discussion.

**Cache rebuild required before the next training run.** The feature
builder's cache key is `sparsedrive_feature.gz`; the existing DGX
`/raid/home/spapais/work_dirs/.../dataset_cache/` was built before these
keys existed and lacks `T_global` / `T_global_inv` plus the new
`ego_status[5]`. Re-run `scripts/run_dataset_caching` (Ray-parallel,
~46 min on Killarney) before the next `navtrain` job.

**Config alignment with nuScenes s2nopercep (2026-05-03):**
- Added `skip_perception_kv=True` and `num_map=0` to `sparsedrive_r50_stage2_navsim_planonly.py`
- This matches the nuScenes `_s2nopercep` config: no det/map K/V in planning cross-attention, map branch produces no cross-attn queries into the planner
- Previous run (3724) had `num_map=10` + no `skip_perception_kv` — frozen unsupervised map features were unnecessarily feeding the planner

**Next job (pending stability of 3724):**
- `batch_size=8/GPU` (32 total, 4× current), `skip_perception_kv=True`, `num_map=0`
- Expected epoch time: ~20 min → 100 epochs in ~1.4 days

## NaN Diagnosis (2026-05-01)

The empty-GT det/map heads compute massive (~50k summed) classification
losses that overflow fp16 immediately. fp32 sidesteps that, but a different
NaN appears around step 4233/5000 even with the head weights frozen and
their losses dropped from backprop.

Root cause is structural: `motion_plan_head` reaches into
`SparseDriveHead.det_head.anchor_encoder` and `.instance_bank`, so we
**must** keep `with_det=True` (head builds + runs forward) even though we
have no detection supervision. The det/map heads run forward against the
backbone (which is still updating), and their outputs feed motion_plan_head
via cross-attention. With no supervision, those features drift; eventually
some batch produces extreme cross-attention values and the planning loss
spikes through gradient clip into NaN.

**Secondary bug found (job 3387927):** `compute_targets()` returns CPU tensors
(`gt_ego_fut_trajs`, `gt_ego_fut_masks`). `motion_planning_head.py` derives
`device` from `gt_ego_fut_trajs`, so predictions on GPU vs targets on CPU
causes a `smooth_l1_loss(cuda:0, cpu)` RuntimeError. Fixed in session 2 by
moving all target tensors to `img.device` at the start of `forward`.

**Decision (with user, 2026-05-01):** Adopt **Option 2** — load a stage-1
nuScenes checkpoint to give det/map meaningful pretrained weights, freeze
them, skip their losses. Cross-attention then sees real (frozen) detection
features instead of noise.

## Iteration Plan (2026-05-03 PM)

**Strategy**: Tier-2 stage-2-safe knobs first, each as a 1-epoch + eval cycle
(~95 min/iter, ~60 min once `ray_distributed_torch` is validated). Promote
the best 1-epoch config to a 10-epoch overnight run. 100-epoch runs
deferred — too long for iteration (~3.3 days).

**Sub-metric weak spots from 3727 baseline** (PDMS=0.65):
- **DAC = 0.79** (worst, ~21% leave drivable area) → needs map awareness; will be fundamentally limited until stage-1 map GT is wired (Tier 3).
- **EP = 0.60** → too cautious / stalls; minS2's `plan_ego_status_encode` should help directly. Verify with 3750 result.
- Everything else (NAC 0.93, TTC 0.85, Comf 0.98, DDC 0.92) is healthy.

### Tier 2 — Stage-2-safe 1-epoch sweeps (current focus)

Numbered in the order we'll run them. Each row gates the next on a sane
PDMS (≥ baseline ± 1%):

1. **3750 minS2 baseline (in flight)** — 5-knob port of nuScenes minS2:
   `plan_ego_status_encode_enable`, `ego_only_planning`, zero motion losses,
   freeze backbone+perception via `lr_mult=0`, `use_rescore=False`, plus
   `eval_skip_map=True` for faster eval. Expected: ↑EP, similar everything else.
2. **+ hflip with shared per-token seed** — port the deferred hflip but with
   a hash-derived seed (e.g. `hash(scene_metadata.initial_token)` in the
   target builder, `hash(agent_input.cameras[-1].cam_f0.intrinsics.tobytes())`
   or similar shared content in the feature builder) so both builders pick
   the same flip choice → mirrored image AND mirrored gt_ego_fut_trajs.
   Doubles dataset diversity. Expected: ↑DAC modestly, ↑PDMS 1-3%.
3. **+ minS2 arch knobs we didn't first port**:
   - `planning_cumulative_refinement=True`, `motion_cumulative_refinement=True`
   - `planning_deformable=True`, `planning_deformable_instfeat=True`,
     `planning_deformable_instfeat_laststage=True`,
     `planning_deformable_waypoints=[0,1,2,3,4,5]`
   - `motion_deformable_multimode=True`
   - These are config-only knobs that already have plumbing in the head
     code (minS2 ships them on the same head). Expected: small wins.
4. **+ mixed precision done right** — needs `@force_fp32` decorators around
   mmcv focal_loss + careful dtype handling at the det/map loss boundary
   (3748 fp16 NaN'd, 3749 bf16 crashed on focal_loss CUDA kernel).
   ~1.5-2× training throughput once it works. ~2-4 hr engineering.
5. **+ ray_distributed_torch eval validation** — code committed but
   untested in distributed mode. Test on a checkpoint from #3 or #4. If
   it works, eval cycle drops 50→15 min.

### Tier 2 promotion gate

Run **10 epochs overnight on the best 1-epoch config**. ~9 hr wall.
This is where we expect the real step-change in PDMS — the nuScenes minS2
winner trained for 10 epochs.

### Tier 3 — Sub-metric-targeted (medium effort, multi-day)

Pursue once Tier 2 is exhausted or PDMS plateaus. Still stage-2-safe.

- **DAC fix via real map GT**. Build `Scene.map_api` → polylines extractor
  in current-ego frame, vectorize to `(N, 38, 20, 2)`, plug into a custom
  collate_fn for variable-N. Flip `num_map=10` and `skip_perception_kv=False`.
  This is a stage-1-shaped change but ships planning-only benefits.
- **Lidar depth aux loss**. `MultiScaleDepthMapGenerator`-equivalent in the
  feature builder; cache size grows. Stabilizes backbone features.
- **All 8 cameras**. Restore `cam_l1` / `cam_r1`. Touches backbone camera
  embed dim. Low impact unless side-camera coverage becomes a metric driver.

### Tier 4 — Stage-1 perception (real det/motion GT)

Heaviest lift; unlocks unfreezing perception heads. Needs:
- nuPlan `Annotations` → 11-dim SparseDrive boxes flowing through the
  target builder (the converter exists in `sparsedrive_features.py:92` but
  isn't called).
- Per-agent future trajectories via cross-frame `track_tokens` walk over
  `Scene.frames`.
- Custom `collate_fn` for variable-length per-batch GT lists.
- Re-enable det/map/motion losses + remove the agent's freeze.

### Tier 5 — Long-term

- Joint nuScenes + NavSim training (shared backbone, task-specific heads).
- Closed-loop fine-tuning with PDM-Score as reward (differentiable sim or RL).

## Eval-time speedups landed (untested in production)

Three optional opt-ins for the next eval cycle, committed in `a6843dd`:

1. `_eval_fp16` flag on `SparseDriveAgent` — wraps eval-mode forward in
   `torch.cuda.amp.autocast(dtype=torch.float16)`. Default True. ~17%
   faster on local A6000. Only the eval path is wrapped — training is
   unaffected.
2. `eval_skip_map` flag on `SparseDriveHead` — skips map_head's forward +
   post_process at eval when `num_map=0` makes it dead compute. Wired
   `True` in `navsim_planonly_minS2.py` config.
3. `worker=ray_distributed_torch` (new yaml + `worker_ray_torch.py`) —
   navsim-side subclass of `RayDistributedNoTorch` that registers GPUs
   with ray and stamps every dispatched task with `num_gpus_per_task`.
   Expected to enable true 4-GPU eval distribution. Untested in production
   (validate on first DGX use).

## DGX bash script gotchas (fixed)

- `navsim_train.sh` reads cache from `${NAVSIM_DATA_ROOT}/training_cache`
  but `navsim_cache.sh` writes to `${NAVSIM_EXP_ROOT}/training_cache`. Both
  now use `NAVSIM_EXP_ROOT` (`91802f1`).
- `navsim_eval.sh` sourced `navsim_setup.sh` without an empty arg, so its
  shifted positional args leaked into setup's case statement and the
  `*) Unknown subcommand` branch killed the job. Fixed to source with `""`
  (`94c72b3`).
- Agent's `os.getcwd()` based `ckpt_dir` lands ckpts in the repo root
  instead of the Hydra per-run dir. Workaround: ckpts at
  `/raid/home/spapais/ForeSight/checkpoints/sparsedrive-epoch=*-step=*.ckpt`,
  filter by mtime to disambiguate runs. Real fix: use `HydraConfig.get().runtime.output_dir`.
- nuplan-devkit's `worker_map` creates `Task(fn=fn)` without `num_gpus`,
  so `worker=ray_distributed` workers don't get CUDA → `t == DeviceType::
  CUDA INTERNAL ASSERT FAILED`. Worked around via `RayDistributedTorch`
  subclass (above).
- PDM-Score eval default `worker=single_machine_thread_pool` with all
  CPUs → 64 threads on 1 GPU → 38GB OOM. Sweet spot: ~8-32 threads on a
  40GB A100, with throughput plateauing past 8 (GPU contention dominates).
