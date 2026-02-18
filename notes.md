# Setup
mkdir ckpt
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
wget https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r101_fpn_1x_nuim/cascade_mask_rcnn_r101_fpn_1x_nuim_20201024_134804-45215b1e.pth -O ckpt/cascade_mask_rcnn_r101_fpn_1x_nuim_20201024_134804-45215b1e.pth
wget https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage1.pth -O ckpt/sparsedrive_stage1.pth
wget https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage2.pth -O ckpt/sparsedrive_stage2.pth
sh scripts/create_data.sh
sh scripts/kmeans.sh
sudo openconnect -v vpn.uwaterloo.ca -u s2papais
rsync -av ForeSight/ spapais@129.97.163.137:/home/spapais/ForeSight/

# Build images
docker build -f docker/Dockerfile -t foresight:latest .
sudo singularity build docker/foresight.sif docker-daemon://foresight:latest

# Sync the code to the Apollo server
sudo openconnect -v vpn.uwaterloo.ca -u s2papais
rsync -av projects scripts docker tools ckpt data requirement.txt spapais@129.97.163.137:/home/spapais/ForeSight/
rsync -av --exclude='*.pyc' projects scripts spapais@129.97.163.137:/home/spapais/ForeSight/

# Sync the code to the DGX server
sudo nmcli con up id utias-robotics && rsync -av projects scripts docker tools ckpt data requirement.txt spapais@192.168.42.200:/raid/home/spapais/ForeSight/ && sudo nmcli con down id utias-robotics
sudo nmcli con up id utias-robotics && rsync -av --exclude='*.pyc' projects scripts spapais@192.168.42.200:/raid/home/spapais/ForeSight/ && sudo nmcli con down id utias-robotics

# Test Stage 1
./scripts/local_run.sh bash ./tools/dist_test.sh projects/configs/sparsedrive_r50_stage1_1gpu.py ckpt/sparsedrive_stage1.pth 1 --deterministic --eval bbox
./scripts/apollo_run.sh bash ./tools/dist_test.sh projects/configs/sparsedrive_r50_stage1_8gpu_noflash.py ckpt/sparsedrive_stage1.pth 8 --deterministic --eval bbox
sbatch scripts/dgx_run.sh bash ./tools/dist_test.sh projects/configs/sparsedrive_r50_stage1_4gpu.py ckpt/sparsedrive_stage1.pth 4 --deterministic --eval bbox

# Test Stage 2
./scripts/local_run.sh bash ./tools/dist_test.sh projects/configs/sparsedrive_r50_stage2_1gpu.py ckpt/sparsedrive_stage2.pth 1 --deterministic --eval bbox
./scripts/apollo_run.sh bash ./tools/dist_test.sh projects/configs/sparsedrive_r50_stage2_8gpu_noflash.py ckpt/sparsedrive_stage2.pth 8 --deterministic --eval bbox
sbatch scripts/dgx_run.sh bash ./tools/dist_test.sh projects/configs/sparsedrive_r50_stage2_4gpu.py ckpt/sparsedrive_stage2.pth 4 --deterministic --eval bbox

# Train Stage 1
./scripts/local_run.sh bash ./tools/dist_train.sh projects/configs/sparsedrive_r50_stage1_1gpu.py 1 --deterministic
./scripts/apollo_run.sh bash ./tools/dist_train.sh projects/configs/sparsedrive_r50_stage1_8gpu_noflash.py 8 --deterministic
sbatch scripts/dgx_run.sh bash ./tools/dist_train.sh projects/configs/sparsedrive_r50_stage1_4gpu.py 4 --deterministic

# Train Stage 2
./scripts/local_run.sh bash ./tools/dist_train.sh projects/configs/sparsedrive_r50_stage2_1gpu.py 1 --deterministic
./scripts/apollo_run.sh bash ./tools/dist_train.sh projects/configs/sparsedrive_r50_stage2_8gpu_noflash.py 8 --deterministic
sbatch scripts/dgx_run.sh bash ./tools/dist_train.sh projects/configs/sparsedrive_r50_stage2_4gpu.py 4 --deterministic

# Visualize
./scripts/apollo_run.sh scripts/visualize.sh
