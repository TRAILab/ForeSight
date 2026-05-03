# Alliance Canada Cluster Reference — 2026-05-01

## GPU Resources

| Cluster | Site | GPU | VRAM | GPU/node | GPU nodes (12h) | Idle now | Max wall time | Account |
|---|---|---|---|---|---|---|---|---|
| narval | Narval (Montréal) | A100 SXM4 | 40 GB | 4 | 141 | ~76 (54%) | 7 d | rrg-swasland |
| trillium | Trillium (SciNet, Toronto) | H100 SXM5 | 80 GB | 4 | 62 | ~10 (16%) | 1 d | rrg-swasland |
| killarney (L40S) | Killarney (Victoria) | L40S PCIe | 48 GB | 4 | 126 | ~44 (35%) | 7 d | aip-swasland |
| killarney (H100) | Killarney (Victoria) | H100 SXM5 | 80 GB | **8** | 10 | ~1 (13%) | 7 d | aip-swasland |
| vulcan | Vulcan (UVic) | L40S PCIe | 48 GB | 4 | 213 | **~132 (62%)** | 7 d | aip-swasland |
| fir | Fir (SFU, Vancouver) | H100 SXM5 | 80 GB | 4 | ~80 | ~27 (34%) | ~122 d | def-swasland-ab |
| rorqual | Rorqual (Montréal) | H100 SXM5 | 80 GB | 4 | 93 | ~48 (52%) | 7 d | def-swasland-ab_gpu |
| tamia | Tamia (Toronto) | H100 SXM5 | 80 GB | 4 | 53 | ~14 (26%) | 1 d | aip-swasland |
| tamia (H200) | Tamia (Toronto) | H200 SXM5 | 141 GB | 8 | 12 | ~2 (18%) | 1 d | aip-swasland |
| nibi | Nibi | H100 | 80 GB | 4 | — | — | — | *no allocation* |

Node counts and idle fractions are for the 12 h tier; idle derived from `sinfo` CPU A/I/O/T fields at time of snapshot.  
Fir has ~36 additional H100 nodes currently draining/down (in maintenance).

---

## Queue & Estimated Time to Completion (2026-05-01)

Training speed estimates relative to H100 SXM5 (bandwidth-adjusted for attention-heavy workloads):
H100 SXM5 = 1.0×, A100 SXM4 40GB ≈ 0.45×, L40S PCIe ≈ 0.30×.
Assuming a full stage-2 run takes ~4.5 h on H100 (consistent with narval ~10 h on A100 fitting within 12 h wall time).

| Cluster | GPU | Queue: 12 h / 4 GPU | Queue: 3 h / 2–4 GPU | Est. run time | **Total: submit→done** |
|---|---|---|---|---|---|
| **tamia** | H100 80GB | **5s** (job 269932) | **4s** (job 269931, 4GPU) | ~4.5 h | **~4.5 h** |
| **fir** | H100 80GB | **7m56s** (job 38141983) | 24m27s (job 38141984, 2GPU) | ~4.5 h | **~4.6 h** |
| **rorqual** | H100 80GB | 18m06s (job 11230957) | **2m48s** (job 11230955, 2GPU) | ~4.5 h | **~4.8 h** |
| narval | A100 40GB | 2m30s (job 60026027, 3d ago) | 508m27s† (job 60019025, 4GPU) | ~10 h | ~10 h |
| trillium | H100 80GB | 411m17s (job 476927, 1d ago) | 46m40s (job 474041, 4GPU) | ~4.5 h | ~11.4 h |
| vulcan | L40S 48GB | 1m27s (job 4820052) | 26s (job 4820053, 2GPU) | ~15 h | ~15 h |
| killarney (L40S) | L40S 48GB | 95m57s (job 3386494, today) | 10m17s (job 3377755, 2GPU) | ~15 h | ~16.6 h |
| killarney (H100) | H100 80GB | — | — | ~4.5 h | — |

New-cluster queue times from `sacct Submit→Start` on first-round probe jobs (2026-05-01). Existing-cluster times from most-recent matching job in sacct cache.  
† Narval 3 h/4GPU outlier (508 min); the 12 h tier is far less contended.  
Queue time dominates for trillium; GPU speed dominates for vulcan/killarney L40S — both land near 15 h total despite opposite reasons.

---

## GPU Test Results (second-round probes, submitted 2026-05-01)

| Cluster | Job | GPUs seen | GPU memory | Scratch write | Apptainer (compute node) |
|---|---|---|---|---|---|
| vulcan | 4820187 | PENDING | — | — | — |
| fir | 38144177 | 4× H100 80GB HBM3 ✓ | 81559 MiB ✓ | OK ✓ | FAIL (module not available) |
| rorqual | 11231140 | PENDING | — | — | — |
| tamia | 269959 | 4× H100 80GB HBM3 ✓ | 81559 MiB ✓ | FAIL ✗ | FAIL (module not available) |

Notes:
- Fir compute nodes have `/etc/profile.d/modules.sh` but `module load apptainer/1.3.5` fails; apptainer will need to be invoked via CVMFS full path or a different init (same pattern as killarney).
- Tamia scratch FAIL: `/scratch/spapais` does not exist yet and `mkdir /scratch/spapais` likely requires project-group permissions. Need to create with: `mkdir -p /scratch/spapais`.
- Tamia apptainer: same module issue as fir. The CVMFS path approach should work.
- CVMFS apptainer binary (all Alliance Canada): `/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/apptainer/1.3.5/bin/apptainer`

---

## nuScenes Data Status

| Cluster | Data path | Format | Status |
|---|---|---|---|
| narval | `/home/spapais/projects/rrg-swasland/datasets/nuscenes/` | zip archives | Ready ✓ |
| trillium | `/home/spapais/projects/rrg-swasland/datasets/nuscenes/` | zip archives | Ready ✓ |
| killarney | `/home/spapais/projects/aip-swasland/datasets/nuscenes/` | zip archives | Ready ✓ |
| vulcan | — | — | Not present |
| fir | `/project/def-swasland-ab/datasets/nuscenes/` | pre-extracted | Partial ✓ (needs verification) |
| rorqual | — | — | Not present |
| tamia | — | — | Not present |

Fir's extracted dataset has: `samples/`, `v1.0-trainval/`, `maps/`, pkl infos, gt\_database. Missing `sweeps/` (not needed for camera-only model). Run script adaptation needed: bind mount directly instead of unzip loop.

---

## Setup Checklist by Cluster

### Existing (operational)
- **narval** ✓ — no action needed
- **trillium** ✓ — no action needed; GPU jobs must be submitted from `trillium_gpu` login node
- **killarney** ✓ — no action needed; apptainer via CVMFS full path in `killarney_run.sh`

### New clusters — set up tamia + fir + rorqual in parallel

All three share the same setup procedure. Cluster-specific differences are noted per cluster.

#### Step 0 — GitHub SSH key (one-time per cluster, do before clone)
On each cluster:
```bash
ssh-keygen -t ed25519 -C "spapais"   # accept defaults, no passphrase
cat ~/.ssh/id_ed25519.pub             # copy output
ssh-keyscan github.com >> ~/.ssh/known_hosts
```
Add the printed public key as a Deploy Key on the GitHub repo (Settings → Deploy keys, read-only is sufficient).

#### Step 1 — Create run script
Add `scripts/<cluster>_run.sh` to the local repo and push. See per-cluster notes below for account, GPU spec, wall time, and path differences.

#### Step 2 — Clone repo + checkout branch (after Step 0 + 1)
```bash
# tamia (SciNet home path is /home/s/spapais/)
ssh tamia "ssh-keyscan github.com >> ~/.ssh/known_hosts && git clone git@github.com:TRAILab/ForeSight.git /home/s/spapais/ForeSight && cd /home/s/spapais/ForeSight && git checkout sd_combined"

# fir / rorqual (standard Alliance path /home/spapais/)
ssh fir    "ssh-keyscan github.com >> ~/.ssh/known_hosts && git clone git@github.com:TRAILab/ForeSight.git /home/spapais/ForeSight && cd /home/spapais/ForeSight && git checkout sd_combined"
ssh rorqual "git clone git@github.com:TRAILab/ForeSight.git /home/spapais/ForeSight && cd /home/spapais/ForeSight && git checkout sd_combined"
```

#### Step 3 — Create scratch dirs and data dirs
```bash
ssh tamia   "mkdir -p /scratch/s/spapais/ForeSight/{logs,work_dirs,wandb} && mkdir -p /home/s/spapais/links/projects/aip-swasland/datasets/nuscenes && mkdir -p /home/s/spapais/ForeSight/{ckpt,data/{infos,kmeans,motion_stats,occlusions}}"
ssh fir     "mkdir -p /scratch/spapais/ForeSight/{logs,work_dirs,wandb} && mkdir -p /project/def-swasland-ab/datasets/nuscenes && mkdir -p /home/spapais/ForeSight/{ckpt,data/{infos,kmeans,motion_stats,occlusions}}"
ssh rorqual "mkdir -p /scratch/spapais/ForeSight/{logs,work_dirs,wandb} && mkdir -p /home/spapais/links/projects/def-swasland-ab/datasets/nuscenes && mkdir -p /home/spapais/ForeSight/{ckpt,data/{infos,kmeans,motion_stats,occlusions}}"
```

#### Step 4 — Copy container from killarney (~4 GB, run from killarney)
Use `foresight_cuda118pytorch21.sif` — the base `cuda118.sif` does not work on H100 SXM5 hardware (same issue as trillium).
```bash
ssh killarney "
  scp /home/spapais/ForeSight/docker/foresight_cuda118pytorch21.sif tamia:/home/s/spapais/ForeSight/docker/ &
  scp /home/spapais/ForeSight/docker/foresight_cuda118pytorch21.sif fir:/home/spapais/ForeSight/docker/ &
  scp /home/spapais/ForeSight/docker/foresight_cuda118pytorch21.sif rorqual:/home/spapais/ForeSight/docker/ &
  wait"
```

#### Step 5 — Copy nuScenes zip archives (run from narval, ~40 GB each)
Data is unzipped to `SLURM_TMPDIR` at job start for fast local access — do not use pre-extracted data.
```bash
ssh narval "
  nohup rsync -a /home/spapais/projects/rrg-swasland/datasets/nuscenes/ tamia:/home/s/spapais/links/projects/aip-swasland/datasets/nuscenes/ > /tmp/rsync_tamia.log 2>&1 &
  nohup rsync -a /home/spapais/projects/rrg-swasland/datasets/nuscenes/ fir:/project/def-swasland-ab/datasets/nuscenes2/ > /tmp/rsync_fir.log 2>&1 &
  nohup rsync -a /home/spapais/projects/rrg-swasland/datasets/nuscenes/ rorqual:/home/spapais/links/projects/def-swasland-ab/datasets/nuscenes/ > /tmp/rsync_rorqual.log 2>&1 &"
```

#### Step 6 — Build custom ops (one-time per cluster, after container is present)
```bash
ssh <cluster> "cd /home/[s/]spapais/ForeSight && \
  /cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/apptainer/1.3.5/bin/apptainer \
  exec --nv -c -e --pwd /workspace/ForeSight/ \
  --bind=/home/[s/]spapais/ForeSight:/workspace/ForeSight/ \
  /home/[s/]spapais/ForeSight/docker/foresight_cuda118pytorch21.sif \
  bash -c 'cd projects/mmdet3d_plugin/ops && python setup.py develop'"
```

#### Step 7 — Submit probe run (1 iter, after Steps 4–6 complete)
```bash
ssh tamia   "source ~/.bashrc && cd /home/s/spapais/ForeSight && sbatch --export=ALL scripts/tamia_run.sh bash ./tools/dist_train.sh projects/configs/sparsedrive_r50_stage2_4gpu_bs24_predonly.py 4 --deterministic"
ssh fir     "source ~/.bashrc && cd /home/spapais/ForeSight && sbatch --export=ALL scripts/fir_run.sh bash ./tools/dist_train.sh projects/configs/sparsedrive_r50_stage2_4gpu_bs24_predonly.py 4 --deterministic"
ssh rorqual "source ~/.bashrc && cd /home/spapais/ForeSight && sbatch --export=ALL scripts/rorqual_run.sh bash ./tools/dist_train.sh projects/configs/sparsedrive_r50_stage2_4gpu_bs24_predonly.py 4 --deterministic"
```

#### Status (2026-05-01)
- [x] Run scripts created (`scripts/tamia_run.sh`, `scripts/fir_run.sh`, `scripts/rorqual_run.sh`)
- [x] Branch pushed to origin
- [x] GitHub SSH keys generated and added (Step 0)
- [x] Repo cloned and on `sd_combined` (Steps 1–2)
- [x] Scratch dirs created (Step 3)
- [ ] Container copied (Step 4)
- [ ] nuScenes data copied (Step 5)
- [ ] Custom ops built (Step 6)
- [ ] Probe run submitted (Step 7)

#### Per-cluster notes

**Tamia** (aip-swasland) — ~4.5 h total, essentially zero queue
- Home path: `/home/s/spapais/` (SciNet convention, differs from all other clusters)
- `--gpus-per-node=h100:4`; `--time=23:59:00` (1 d max wall time); 4-GPU minimum
- DATA_DIR: `~/links/projects/aip-swasland/datasets/nuscenes/`

**Fir** (def-swasland-ab) — ~4.6 h total, 8 min queue
- DATA_DIR: `/project/def-swasland-ab/datasets/nuscenes2/` — original `nuscenes/` dir is owned by `nisarbar` (no write access), created `nuscenes2/` in the same parent
- `--gpus-per-node=4`; standard 12 h wall time

**Rorqual** (def-swasland-ab_gpu) — ~4.8 h total, 18 min queue
- `--partition=gpubase_bynode_b2`; account `def-swasland-ab_gpu`
- DATA_DIR: `~/links/projects/def-swasland-ab/datasets/nuscenes/`
- Both `rrg-swasland` and `def-swasland-ab` project dirs exist; script uses `def-swasland-ab` to match compute account

**Vulcan** (aip-swasland) — deprioritised
- L40S is ~3× slower than H100; total ~15 h regardless of near-instant queue
- Set up only if tamia/fir/rorqual are at capacity

---

## Recommendation: Set up Tamia + Fir + Rorqual in parallel

Ranking by estimated time from submit to completion (queue + training run):

| Rank | Cluster | Total (12 h job) | Why |
|---|---|---|---|
| 1 | **Tamia** | ~4.5 h | H100 + instant queue |
| 2 | **Fir** | ~4.6 h | H100 + low contention |
| 3 | **Rorqual** | ~4.8 h | H100 + moderate queue |
| — | narval | ~10 h | A100 is 2.2× slower |
| — | trillium | ~11.4 h | H100 hardware wasted on 7 h queue |
| — | vulcan | ~15 h | L40S is 3.3× slower than H100 |
| — | killarney L40S | ~16.6 h | L40S + long queue |

All three H100 clusters (tamia, fir, rorqual) land within 20 minutes of each other in total time — the GPU speed advantage dwarfs the queue differences between them. Setting them up in parallel gives redundancy and capacity without meaningful extra effort since the setup steps are nearly identical across all three.

Vulcan (L40S) is not worth prioritising — its 3× slower training means a 15 h total time regardless of its near-instant queue. Only fall back to it if the H100 trio is fully saturated.
