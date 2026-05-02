#!/bin/bash
#SBATCH --job-name=foresight_navsim
#SBATCH --ntasks=1
#SBATCH --mem=440gb
#SBATCH --time=3-00:00:00
#SBATCH --output=./logs/%x-%j.log
#SBATCH --cpus-per-task=120
#SBATCH --gres=gpu:4
#SBATCH --mail-user="sandro.papais@robotics.utias.utoronto.ca"
#SBATCH --mail-type=END,FAIL
#
# NavSim DGX runner. Mirrors dgx_run.sh but:
#   - uses the foresight_navsim singularity image (python 3.9 + nuplan-devkit)
#   - mounts OpenScene + nuplan maps instead of nuScenes
#   - sets NAVSIM env vars
#
# Usage (from local):
#   ssh trail_dgx "bash -i -c 'cd /raid/home/spapais/ForeSight && \
#     sbatch --export=ALL scripts/dgx_navsim_run.sh \
#     bash /workspace/ForeSight/scripts/navsim_train.sh navmini'" 2>/dev/null

[[ -f ~/.bashrc ]] && source ~/.bashrc

CODE_DIR=/raid/home/spapais/ForeSight
OPENSCENE_HOST_DIR=${OPENSCENE_HOST_DIR:-/raid/home/spapais/datasets/openscene}
NUPLAN_MAPS_HOST_DIR=${NUPLAN_MAPS_HOST_DIR:-/raid/home/spapais/datasets/nuplan-maps-v1.0}
WORK_DIR=${CODE_DIR}/work_dirs

CMD=${@:-bash}

SIF_PATH=${SIF_PATH:-${CODE_DIR}/docker/foresight_navsim.sif}

# Skip openscene / nuplan-maps binds when the host dirs are missing so the
# unit smoke test still runs cleanly before data lands.
DATA_BINDS=""
[ -d "$OPENSCENE_HOST_DIR" ]   && DATA_BINDS+=" --bind=$OPENSCENE_HOST_DIR:/workspace/ForeSight/data/openscene"
[ -d "$NUPLAN_MAPS_HOST_DIR" ] && DATA_BINDS+=" --bind=$NUPLAN_MAPS_HOST_DIR:/workspace/ForeSight/data/nuplan-maps-v1.0"

CONTAINER_CMD="singularity exec --nv -e --pwd /workspace/ForeSight/ \
--env WANDB_API_KEY=$WANDB_API_KEY \
--env WANDB_MODE=${WANDB_MODE:-online} \
--env FORESIGHT_ROOT=/workspace/ForeSight \
--env NAVSIM_DEVKIT_ROOT=/workspace/ForeSight/navsim \
--env NAVSIM_EXP_ROOT=/workspace/ForeSight/work_dirs/navsim \
--env NAVSIM_DATA_ROOT=/workspace/ForeSight/data/navsim \
--env OPENSCENE_DATA_ROOT=/workspace/ForeSight/data/openscene \
--env NUPLAN_MAPS_ROOT=/workspace/ForeSight/data/nuplan-maps-v1.0 \
--bind=${CODE_DIR}:/workspace/ForeSight/ \
$DATA_BINDS \
$SIF_PATH"

echo "Running: $CMD"
$CONTAINER_CMD bash -c "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH && $CMD"
