# ForeSight — Claude Context

ForeSight is an autonomous driving research project built on [SparseDrive](https://github.com/swc-17/SparseDrive). It extends SparseDrive with occlusion-aware evaluation, GT oracle heads, and rotation augmentation. Training and evaluation run on nuScenes.

## Model Architecture

- **Pipeline:** ResNet-50 → FPN → SparseDriveHead
- **SparseDriveHead** has three sub-heads: detection (`Sparse4DHead`), map, and motion/planning (`MotionPlanningHead`)
- **Two training stages:** Stage 1 (detection + map), Stage 2 (full model including motion/planning)
- **Configs** are Python files exec()'d by mmdet3d — appending lines at the end overrides earlier values

### Internal data formats
- `det_output`: `{instance_feature(bs,N,256), anchor_embed(bs,N,256), classification[list], prediction[list,11-dim], quality[list], instance_id(bs,N)}`
- 11-dim anchor: `[X, Y, Z, log_W, log_L, log_H, SIN_YAW, COS_YAW, VX, VY, VZ]`
- GT boxes from nuScenes: 9-dim decoded `[x, y, z, w, l, h, yaw, vx, vy]`

## Key Source Files

| File | Purpose |
|------|---------|
| `projects/mmdet3d_plugin/models/sparsedrive_head.py` | Main head dispatcher |
| `projects/mmdet3d_plugin/models/gt_sparse_drive_head.py` | GT oracle head |
| `projects/mmdet3d_plugin/models/detection3d/detection3d_head.py` | Sparse4DHead |
| `projects/mmdet3d_plugin/models/motion/motion_planning_head.py` | MotionPlanningHead |
| `projects/mmdet3d_plugin/models/motion/instance_queue.py` | Temporal tracking queue |
| `projects/mmdet3d_plugin/models/instance_bank.py` | Instance feature bank |
| `projects/mmdet3d_plugin/datasets/nuscenes_3d_dataset.py` | Dataset + evaluation |
| `projects/mmdet3d_plugin/datasets/evaluation/planning/planning_eval.py` | Planning metrics |
| `projects/mmdet3d_plugin/datasets/evaluation/det/occluded_det_eval.py` | Det eval with occlusion |
| `projects/mmdet3d_plugin/datasets/evaluation/motion/motion_eval_uniad.py` | Motion eval |
| `projects/mmdet3d_plugin/datasets/pipelines/augment.py` | BBoxRotation, traj_rotate |
| `projects/mmdet3d_plugin/datasets/pipelines/transform.py` | NuScenesSparse4DAdaptor |

## Key Metrics (nuScenes val)

| Metric | Direction | Description |
|--------|-----------|-------------|
| **L2** | ↓ lower | Ego planning L2 error (meters) — **primary** |
| **obj_box_col** | ↓ lower | Planning collision rate (%) — **primary** |
| car_ade / ped_ade | ↓ lower | Agent motion average displacement error |
| car_epa / ped_epa | ↑ higher | Motion end-point accuracy |
| NDS | ↑ higher | nuScenes detection score |
| mAP | ↑ higher | Detection mean AP |
| mAP_normal | ↑ higher | Map prediction mAP |

## Servers

| Key | DGX | Apollo | Narval |
|-----|-----|--------|-------------|
| **ssh_host** | `trail_dgx` | `apollo` | `narval` |
| **remote_repo** | `/raid/home/spapais/ForeSight` | `/home/spapais/ForeSight` | `/home/spapais/ForeSight` |
| **job_system** | SLURM | Docker (direct) | SLURM |
| **run_script** | `scripts/dgx_run.sh` | `scripts/apollo_run.sh` | `scripts/narval_run.sh` |
| **gpus** | 4 A100 | 8 V100 | 4 A100 |
| **slurm_log** | `logs/foresight-<JOB_ID>.log` | N/A | `logs/foresight-<JOB_ID>.log` |
| **work_dir_log** | `work_dirs/<stem>/*.log` | `work_dirs/<stem>/*.log` | `work_dirs/<stem>/*.log` |
| **wandb_mode** | online | online | offline |
| **vpn** | nmcli `utias-robotics` (persistent) | openconnect UW (persistent) | none |

### VPN
Both VPNs are kept always-on — never disconnect them. Use `/connect-server` to bring one up if it's down.
- **DGX:** `nmcli con show --active | grep -q utias-robotics || sudo nmcli con up id utias-robotics`
- **Apollo:** SSH-test first; if down, tell user to run `sudo openconnect -v vpn.uwaterloo.ca -u s2papais` (interactive — don't automate)
- **Narval:** no VPN, direct SSH

### Submit commands
```bash
# DGX
ssh trail_dgx "source ~/.bashrc && cd /raid/home/spapais/ForeSight && sbatch --export=ALL,WANDB_API_KEY=1cb0a37040ca089569cecda1c31722a24d56d3a4 scripts/dgx_run.sh <cmd>"

# Apollo (no SLURM, no job ID)
ssh apollo "cd /home/spapais/ForeSight && ./scripts/apollo_run.sh <cmd>"

# Narval
ssh narval "source ~/.bashrc && cd /home/spapais/ForeSight && sbatch --export=ALL,WANDB_API_KEY=1cb0a37040ca089569cecda1c31722a24d56d3a4 scripts/narval_run.sh <cmd>"
```

### Code sync (git)
Push from local, pull on server — no rsync needed for code:
```bash
git push && ssh <host> "cd <remote_repo> && git pull"
```

## Skills

| Skill | Purpose |
|-------|---------|
| `/connect-server --server dgx\|apollo\|cc` | Check/bring up VPN, verify SSH |
| `/sync-results [--server dgx\|apollo\|cc\|all]` | Pull work_dirs from server to local |
| `/submit-job --server <s> --config <path>` | Submit train/eval job |
| `/parse-metrics --server <s> --job <id> --config <stem>` | Parse metrics from log |
| `/check-runs [--server dgx\|cc\|all]` | List active SLURM jobs |
| `/autoresearch --goal "..." --max-experiments N` | Autonomous experiment loop |

## Experiment Hard Constraints

Results already in — never re-propose these:
- **No map removal** from both stages — planning catastrophically fails (L2: 0.600→6.61)
- **No pretrainv3/v4** prediction pretraining — degrades all metrics
- **No separate head** (sephead) — slightly worse across the board
- **No reduced map LR** — map_mAP collapses to ~0.07
- **No map head in stage2** when loaded from DN stage1 pretrain — L2 worsens to 0.700
- **No motion_loss_reg/cls > 0.2** — large regression on L2 and obj_box_col
- **No rotation augmentation** (rot3d_range) — hurts both L2 and obj_box_col
