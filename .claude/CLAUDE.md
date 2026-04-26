# ForeSight

Autonomous driving research codebase built on SparseDrive. Training and evaluation run on nuScenes.

## Verify Work
- Prefer the smallest check that verifies the change.
- For config edits, re-read the final config and confirm only the intended lines changed.
- For training or evaluation changes, prefer a targeted command or log-based verification over broad reruns.
- Report what was verified and what was not.

## Commands
- Local train: `./scripts/local_run.sh bash ./tools/dist_train.sh <config> <num_gpus> --deterministic`
- Local eval: `./scripts/local_run.sh bash ./tools/dist_test.sh <config> <ckpt> <num_gpus> --deterministic --eval bbox`
- DGX: `sbatch scripts/dgx_run.sh <cmd>`
- Apollo: `./scripts/apollo_run.sh <cmd>`
- Narval: `sbatch scripts/narval_run.sh <cmd>`
- Trillium: `sbatch scripts/trillium_run.sh <cmd>`
- Killarney: `sbatch scripts/killarney_run.sh <cmd>`
- Build custom ops after a fresh clone: `cd projects/mmdet3d_plugin/ops && python setup.py develop`

## Config Rules
- Configs live in `projects/configs/`.
- Edit existing configs in place. Do not append override blocks at the end when modifying an existing config.
- When creating a new config, copy the baseline config first and then modify it in place.
- Active research configs usually follow `sparsedrive_r50_stage2_4gpu_bs24_<variant>.py`.
- Standard stage 2 initialization uses `ckpt/sparsedrive_stage1.pth`.

## Metrics
- Primary planning metrics: `L2` and `obj_box_col` lower is better.
- Secondary metrics commonly discussed: `car_ade`, `ped_ade`, `car_epa`, `ped_epa`, `NDS`, `mAP`, `mAP_normal`.

## Key Files
- `projects/mmdet3d_plugin/models/sparsedrive_head.py`: main head dispatcher
- `projects/mmdet3d_plugin/models/detection3d/detection3d_head.py`: detection head
- `projects/mmdet3d_plugin/models/motion/motion_planning_head.py`: motion and planning head
- `projects/mmdet3d_plugin/models/motion/instance_queue.py`: temporal queue
- `projects/mmdet3d_plugin/models/instance_bank.py`: instance feature bank
- `projects/mmdet3d_plugin/datasets/nuscenes_3d_dataset.py`: dataset and evaluation
- `projects/mmdet3d_plugin/core/box3d.py`: anchor index constants

## Remote Access
- Never disconnect persistent VPNs.
- Do not add separate SSH precheck steps. Run the requested remote operation directly and handle failures if they occur.

### DGX
- Host: `trail_dgx`
- Repo: `/raid/home/spapais/ForeSight`
- VPN: `utias-robotics` via `nmcli`
- Check VPN: `nmcli con show --active | grep -q utias-robotics`
- If needed: `sudo nmcli con up id utias-robotics`
- Remote submission over SSH still needs interactive bash so `WANDB_API_KEY` is exported:
  `ssh trail_dgx "bash -i -c 'cd /raid/home/spapais/ForeSight && sbatch --export=ALL scripts/dgx_run.sh <cmd>'" 2>/dev/null`

### Apollo
- Host: `apollo`
- Repo: `/home/spapais/ForeSight`
- VPN: UW `openconnect`
- Never automate Apollo VPN login. If SSH is unreachable, tell the user to run:
  `sudo openconnect -v vpn.uwaterloo.ca -u s2papais`
- `scripts/apollo_run.sh` requires an interactive TTY. Use an interactive SSH session or `tmux`.

### Narval
- Host: `narval`
- Repo: `/home/spapais/ForeSight`
- No VPN required
- Remote submission over SSH:
  `ssh narval "source ~/.bashrc && cd /home/spapais/ForeSight && sbatch --export=ALL scripts/narval_run.sh <cmd>"`
- WandB runs in offline mode on Narval.

### Trillium
- CPU login host: `trillium` (`trillium.scinet.utoronto.ca`)
- GPU login host: `trillium_gpu` (`trillium-gpu.scinet.utoronto.ca`) — GPU jobs must be submitted from here
- Repo: `/home/spapais/ForeSight`
- No VPN required
- Home is read-only on compute nodes; output logs go to `/scratch/spapais/ForeSight/logs/`
- Eval must pass `--tmpdir /tmp/.dist_test` (the default `.dist_test` resolves under read-only `/workspace/ForeSight` and the rank-0 mkdir crashes after the full forward pass)
- GPU scheduling uses `--gpus-per-node=N` (not `--gres`); only 1 or 4 GPUs allowed per node
- Remote submission over SSH (must use GPU login node):
  `ssh trillium_gpu "source ~/.bashrc && cd /home/spapais/ForeSight && sbatch --export=ALL scripts/trillium_run.sh <cmd>"`
- WandB runs in offline mode on Trillium.

### Killarney
- Host: `killarney`
- Repo: `/home/spapais/ForeSight`
- No VPN required (geo-blocked; requires access from a Canadian institution network)
- SLURM requires module init; `--chdir=/scratch` is baked into the script:
  `ssh killarney "source /etc/profile.d/modules.sh && module load slurm/killarney/24.05.7 && sbatch --export=ALL /home/spapais/ForeSight/scripts/killarney_run.sh <cmd>"`
- Container runtime is apptainer via CVMFS (not a loadable module); called via full path in `killarney_run.sh`
- WandB runs in offline mode on Killarney.

## Git Workflow
- Work locally, then push and pull on the target server.
- Always ask for confirmation before committing or pushing.
- Do not switch remote branches automatically without user direction.
