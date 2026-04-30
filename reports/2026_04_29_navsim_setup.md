# NavSim v1 Integration Plan

## Abstract

Plan and initial scaffolding for adding NavSim v1 (navtrain / navtest) as a second
training and evaluation track for SparseDrive, alongside the existing nuScenes
pipeline. Approach: vendor the official `navsim` Python package into `navsim/`
at the repo root, register a new `SparseDriveAgent` that wraps the existing SparseDrive head behind
NavSim's `AbstractAgent` interface, and run training and PDM-Score evaluation
through NavSim's Hydra runner. ForeSight's existing nuScenes runners
(`tools/dist_train.sh`, `tools/dist_test.sh`) stay untouched.

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
| killarney | sparsedrive navmini training | n/a | PENDING |
| killarney | sparsedrive navtrain training | n/a | PENDING |

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

Known follow-ups blocking multi-step training:
- Step 2 hits `TypeError: can't convert cuda:0 device type tensor to numpy`
  inside the instance_bank temporal cache. The cache path expects T_global
  on CPU as numpy; the agent currently injects identity tensors on CUDA.
  Needs the feature builder to ship a real T_global from the scene's
  ego_pose (and probably as numpy) instead of a placeholder.

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

| Model | PDMS | NC | DAC | EP | TTC | Comf. | DDC | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| _baseline ref_ DiffusionDrive | 88.1 | — | — | — | — | — | — | published |
| ForeSight SparseDrive (planning-only) | — | — | — | — | — | — | — | TBD |
| ForeSight SparseDrive (full) | — | — | — | — | — | — | — | TBD |

## Discussion

Pending experiments. The expected default after Phase 4 will be the
planning-only SparseDrive agent on navtrain, which gives us a NavSim-comparable
PDMS number while keeping the rest of the codebase on its existing nuScenes
trajectory.

## Future Work

- Joint nuScenes + navtrain training (shared backbone, separate heads).
- Extend the SparseDrive head to consume all 8 NavSim cameras instead of dropping
  the side cams.
- Closed-loop fine-tuning using the PDM scorer as a reward.
