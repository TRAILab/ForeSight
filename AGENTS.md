# ForeSight — Claude Context

ForeSight is an autonomous driving research project built on [SparseDrive](https://github.com/swc-17/SparseDrive). Training and evaluation run on nuScenes.

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
| `projects/mmdet3d_plugin/models/detection3d/detection3d_head.py` | Sparse4DHead |
| `projects/mmdet3d_plugin/models/motion/motion_planning_head.py` | MotionPlanningHead |
| `projects/mmdet3d_plugin/models/motion/instance_queue.py` | Temporal tracking queue |
| `projects/mmdet3d_plugin/models/instance_bank.py` | Instance feature bank |
| `projects/mmdet3d_plugin/datasets/nuscenes_3d_dataset.py` | Dataset + evaluation |

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
Both VPNs are kept always-on — never disconnect them. Use `/connect-server` to bring one up if it's down. Apollo VPN requires interactive login — tell the user to run it manually, never automate it.

### Code management (git)

Work locally on the code with git to manage code changes. Always ask for confirmation before committing and pushing changes. Use `git add <files>` to stage changes, `git commit -m "<message>"` to commit changes, and `git push` to push changes from local. Use `git pull` to pull changes to the servers.

```bash
git push && ssh <host> "cd <remote_repo> && git pull"
```


