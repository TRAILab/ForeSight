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
| killarney | navtrain training, fp32, 2000 batches (in flight) | 3387639 cache | RUNNING |

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

| Model | PDMS | NC | DAC | EP | TTC | Comf. | DDC | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| _baseline ref_ DiffusionDrive | 88.1 | — | — | — | — | — | — | published |
| ForeSight SparseDrive (planning-only) | — | — | — | — | — | — | — | TBD |
| ForeSight SparseDrive (full) | — | — | — | — | — | — | — | TBD |

## Current State (2026-05-01, end of session)

Everything from Phase 0 → Phase 4 first-checkpoint is in place except the
training itself produces NaN before completing a useful run.

**Working:**
- Image, .sif on DGX + Killarney, smoke 7/7 on both
- Full navtrain (700 GB) + navtest (220 GB) + nuplan-maps on DGX `/raid/home/spapais/datasets/` and Killarney `/home/spapais/projects/aip-swasland/datasets/`
- Apollo has data too but is unusable (V100s, no flash-attn 2.x support)
- Ray-parallel dataset_caching builds in ~46 min (vs 5 hr serial)
- navtest metric_cache built (Killarney, 12k entries, 3 GB) — ready for PDM-Score eval
- Real navtrain plan-side k-means anchors generated
- 500-batch training (`3705`) clean: loss 1.29 → 0.21, no NaN

**Broken / blocker:**
- Training NaN's around step 4000–4500 even with fp32 + grad_clip=1.0 + frozen
  det/map weights + det/map losses excluded from backprop. See diagnosis in
  the next section.

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

**Decision (with user, 2026-05-01):** Adopt **Option 2** — load a stage-1
nuScenes checkpoint to give det/map meaningful pretrained weights, freeze
them, skip their losses. Cross-attention then sees real (frozen) detection
features instead of noise.

## Next Steps (clean session pickup)

1. **Wire stage-1 warm-start.** The local repo has
   `ckpt/sparsedrive_stage1.pth` (per CLAUDE.md the standard stage-2 init).
   - scp to DGX `/raid/home/spapais/ForeSight/ckpt/sparsedrive_stage1.pth`
     and Killarney `/home/spapais/ForeSight/ckpt/sparsedrive_stage1.pth`
   - Update `navsim/navsim/planning/script/config/common/agent/sparsedrive_agent.yaml`:
     ```
     foresight_pretrained: ${oc.env:FORESIGHT_ROOT}/ckpt/sparsedrive_stage1.pth
     ```
   - `SparseDriveAgent._load_pretrained` already does `strict=False`, so the
     `ego_fut_ts=8` / `num_driving_cmds=4` planning-head dim mismatches
     fall through and that head trains from scratch as intended.

2. **Confirm freeze logic still applies.** `SparseDriveAgent.__init__`
   already sets `requires_grad=False` on `head.det_head` + `head.map_head`.
   This combined with the warm-started weights means det/map are frozen at
   their stage-1-trained state — exactly what we want for stage-2 navsim.

3. **Resubmit training on DGX or Killarney**, same precision/strategy as
   `3705` (`fp32`, `grad_clip=1.0`, `bs=2`, single GPU). Now expect:
   - det/map cross-attention features are meaningful (not noise)
   - planning loss should stay stable past 5000 steps
   - Land a real ckpt via the `ModelCheckpoint` callback already wired into
     `SparseDriveAgent.get_training_callbacks`

4. **First navtest PDM-Score eval** once a ckpt exists:
   `bash scripts/navsim_eval.sh /path/to/ckpt navtest`. metric_cache is
   already built on Killarney at
   `/scratch/spapais/ForeSight/work_dirs/navsim/metric_cache`.

5. **If NaN still appears** after stage-1 warm-start: drop to LR=1e-5,
   grad_clip=0.5, OR fall back to **Option 1** (refactor `SparseDriveHead`
   to hoist `anchor_encoder` + `instance_bank` out of `det_head` so we can
   actually skip det/map forward).

## Future Work (deferred)

- Multi-GPU DDP: `find_unused_parameters_true` conflicts with gradient
  checkpointing. Need `DDPStrategy(static_graph=True, find_unused_parameters=True)`
  via custom strategy build. Single GPU works for now.
- Real `T_global` from `scene.frames[i].ego_status.ego_pose` in the feature
  builder (currently identity placeholders kill the temporal cache benefit).
- nuPlan `gt_agent_fut_trajs` from per-track futures across frames, so
  motion head loss is non-zero. Same path needed for the per-class
  motion-side k-means anchors.
- Padded variable-length detection GT or custom collate — to actually
  supervise det/map on navsim. Stage-1 navsim is the natural follow-up.
- Joint nuScenes + navtrain training (shared backbone, separate heads).
- Extend SparseDrive to consume all 8 NavSim cameras instead of dropping
  the side cams.
- Closed-loop fine-tuning using the PDM scorer as a reward.

## Discussion

Pending experiments. The expected default after Phase 4 will be the
stage-2 SparseDrive agent on navtrain (stage-1 nuScenes warm-start, frozen
det/map, planning supervision), which gives us a NavSim-comparable PDMS
number while keeping the rest of the codebase on its existing nuScenes
trajectory.
