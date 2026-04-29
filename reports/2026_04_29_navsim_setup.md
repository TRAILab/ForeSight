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
| local | smoke test (env build + 5 import/build checks) | n/a | COMPLETED |
| killarney | sparsedrive navmini smoke | n/a | PENDING |
| killarney | sparsedrive navtrain | n/a | PENDING |

### Smoke test (2026-04-29, local, no GPU needed)

`docker run ... foresight_navsim:cuda118pytorch21 python scripts/navsim_smoke_test.py`

```
[OK] import navsim base
[OK] import sparsedrive agent
[OK] import foresight plugin (registers mmcv heads)
[OK] build sparsedrive head from navsim variant config (model class: SparseDrive)
[OK] feature builder on synthetic AgentInput
       img shape: (6, 3, 256, 704)
       projection_mat shape: (6, 4, 4)
       ego_status shape: (8,)
All 5 checks passed.
```

What this proves:
- The new conda env (python 3.9 + torch 2.1.2 + mmcv 1.7.2 + flash-attn 2.3.2 +
  nuplan-devkit + hydra + lightning) coexists in one image without import-time
  conflicts.
- `SparseDriveAgent`'s wiring (`sys.path` boost, plugin import, mmcv config →
  `build_detector`) works end-to-end with the navsim-variant config
  (`ego_fut_ts=8`, `num_driving_cmds=4`).
- The 8→6 camera reduction in `SparseDriveFeatureBuilder` produces tensors with
  shapes the head expects.

What this does NOT prove yet:
- Forward pass through SparseDrive on real navsim images (no GPU on local).
- That `SparseDriveAgent.forward` reads the right key out of the head's output
  dict — TODO marker still live in the agent.
- Anything PDM-Score related — needs metric cache + nuplan maps.

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
