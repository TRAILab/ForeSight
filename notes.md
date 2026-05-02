# Setup
mkdir ckpt
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
wget https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r101_fpn_1x_nuim/cascade_mask_rcnn_r101_fpn_1x_nuim_20201024_134804-45215b1e.pth -O ckpt/cascade_mask_rcnn_r101_fpn_1x_nuim_20201024_134804-45215b1e.pth
wget https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage1.pth -O ckpt/sparsedrive_stage1.pth
wget https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage2.pth -O ckpt/sparsedrive_stage2.pth
sh scripts/create_data.sh
sh scripts/kmeans.sh
sudo openconnect -v vpn.uwaterloo.ca -u s2papais
rsync -av ForeSight/ apollo:/home/spapais/ForeSight/

# Build images
docker build -f docker/Dockerfile -t foresight:latest .
sudo singularity build docker/foresight.sif docker-daemon://foresight:latest
sudo singularity build docker/foresight_cuda118.sif docker-daemon://foresight:latest
sudo singularity build docker/foresight_navsim.sif docker-daemon://foresight_navsim:cuda118pytorch21
singularity sif list docker/foresight_cuda118.sif # find squashfs partition ID                                           
singularity sif dump <id> docker/foresight_cuda118.sif > docker/foresight_cuda118.sqsh

# Sync binary artifacts to servers (code changes go via git push/pull)
# Apollo
sudo openconnect -v vpn.uwaterloo.ca -u s2papais
rsync -av docker ckpt apollo:/home/spapais/ForeSight/
# DGX
sudo nmcli con up id utias-robotics && rsync -av docker ckpt spapais@192.168.42.200:/raid/home/spapais/ForeSight/ && sudo nmcli con down id utias-robotics
# CC servers (Narval, Trillium, Killarney)
rsync -av docker ckpt spapais@narval.alliancecan.ca:/home/spapais/ForeSight/
rsync -av docker ckpt trillium:/home/spapais/ForeSight/
rsync -av docker ckpt killarney:/home/spapais/ForeSight/

# Test Stage 1
./scripts/local_run.sh bash ./tools/dist_test.sh projects/configs/sparsedrive_r50_stage1_1gpu.py ckpt/sparsedrive_stage1.pth 1 --deterministic --eval bbox
./scripts/apollo_run.sh bash ./tools/dist_test.sh projects/configs/sparsedrive_r50_stage1_8gpu_noflash.py ckpt/sparsedrive_stage1.pth 8 --deterministic --eval bbox
sbatch scripts/dgx_run.sh bash ./tools/dist_test.sh projects/configs/sparsedrive_r50_stage1_4gpu.py ckpt/sparsedrive_stage1.pth 4 --deterministic --eval bbox

# Test Stage 2
./scripts/local_run.sh bash ./tools/dist_test.sh projects/configs/sparsedrive_r50_stage2_1gpu_bs2.py ckpt/sparsedrive_stage2.pth 1 --deterministic --eval bbox
./scripts/apollo_run.sh bash ./tools/dist_test.sh projects/configs/sparsedrive_r50_stage2_8gpu_noflash.py ckpt/sparsedrive_stage2.pth 8 --deterministic --eval bbox
sbatch scripts/dgx_run.sh bash ./tools/dist_test.sh projects/configs/sparsedrive_r50_stage2_4gpu.py ckpt/sparsedrive_stage2.pth 4 --deterministic --eval bbox

# Train Stage 1
./scripts/local_run.sh bash ./tools/dist_train.sh projects/configs/sparsedrive_r50_stage1_1gpu.py 1 --deterministic
./scripts/apollo_run.sh bash ./tools/dist_train.sh projects/configs/sparsedrive_r50_stage1_8gpu_noflash.py 8 --deterministic
sbatch scripts/dgx_run.sh bash ./tools/dist_train.sh projects/configs/sparsedrive_r50_stage1_4gpu.py 4 --deterministic

# Train Stage 2
./scripts/local_run.sh bash ./tools/dist_train.sh projects/configs/sparsedrive_r50_stage2_1gpu_bs2.py 1 --deterministic
./scripts/apollo_run.sh bash ./tools/dist_train.sh projects/configs/sparsedrive_r50_stage2_8gpu_noflash.py 8 --deterministic
sbatch scripts/dgx_run.sh bash ./tools/dist_train.sh projects/configs/sparsedrive_r50_stage2_4gpu.py 4 --deterministic
sbatch scripts/narval_run.sh bash ./tools/dist_train.sh projects/configs/sparsedrive_r50_stage2_4gpu_nomap.py 4 --deterministic
sbatch --gres=gpu:a100:2 scripts/narval_run.sh bash ./tools/dist_train.sh projects/configs/sparsedrive_r50_stage2_2gpu_nomap.py 2 --deterministic

# Visualize
./scripts/apollo_run.sh scripts/visualize.sh

# Data generation
sbatch tools/dgx_run.sh python unitraj/inference.py --config-name=config_v1inference_3class
./scripts/local_run.sh python tools/data_converter/nuscenes_occlusion_converter.py convert --input data/infos/nuscenes_infos_val.pkl --output data/infos/nuscenes_infos_val_occ.pkl --predictions data/occlusions/nuscenes_predictions_trainval.npz

# Autoresearch
tmux new -s autoresearch (or) tmux attach -t autoresearch (to detach, press Ctrl+b then d or type "detach")
claude
/autoresearch --goal "improve val/L2 and val/obj_box_col" --base-config projects/configs/sparsedrive_r50_stage2_4gpu_nomap.py --max-experiments 5 --poll 30m
/autoresearch --goal "improve val/L2 and val/obj_box_col" --base-config projects/configs/sparsedrive_r50_stage2_4gpu_nomap_queue6.py --max-experiments 5 --poll 30m

# Results synchronization
sudo rsync -av --exclude='*.pkl' --exclude='*.pth' apollo:/home/spapais/ForeSight/work_dirs/ ./work_dirs/
sudo rsync -av --exclude='*.pkl' --exclude='*.pth' spapais@192.168.42.200:/raid/home/spapais/ForeSight/work_dirs/ ./work_dirs/
sudo rsync -av --exclude='*.pkl' --exclude='*.pth' spapais@narval.alliancecan.ca:/home/spapais/ForeSight/work_dirs/ ./work_dirs/
rsync -av --exclude='*.pkl' --exclude='*.pth' trillium:/scratch/spapais/ForeSight/work_dirs/ ./work_dirs/
rsync -av --exclude='*.pkl' --exclude='*.pth' killarney:/scratch/spapais/ForeSight/work_dirs/ ./work_dirs/

# Narval wandb sync
screen -S wandb-sync
bash scripts/narval_wandb_sync.sh (Ctrl+A D to detach)
