#!/bin/bash
#SBATCH --job-name=foresight_navsim
#SBATCH --account=aip-swasland
#SBATCH --ntasks=1
#SBATCH --mem=120gb
#SBATCH --time=11:59:00
#SBATCH --output=/scratch/spapais/ForeSight/logs/%x-%j.log
#SBATCH --chdir=/scratch/spapais/ForeSight
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l40s:4
#SBATCH --mail-user="sandro.papais@robotics.utias.utoronto.ca"
#SBATCH --mail-type=END,FAIL
#
# NavSim Killarney runner. Mirrors killarney_run.sh but:
#   - uses the foresight_navsim apptainer image (python 3.9 + nuplan-devkit)
#   - mounts OpenScene + nuplan maps instead of nuScenes
#   - sets NAVSIM env vars
#
# Usage (from local):
#   ssh killarney "source /etc/profile.d/modules.sh && \
#                  module load slurm/killarney/24.05.7 && \
#                  sbatch --export=ALL /home/spapais/ForeSight/scripts/killarney_navsim_run.sh \
#                  bash /workspace/ForeSight/scripts/navsim_train.sh navmini"

# Parameters
TMP_DIR=$SLURM_TMPDIR/tmp
TMP_DATA_DIR=$SLURM_TMPDIR/data
OPENSCENE_HOST_DIR=${OPENSCENE_HOST_DIR:-/scratch/spapais/data/openscene}
NUPLAN_MAPS_HOST_DIR=${NUPLAN_MAPS_HOST_DIR:-/scratch/spapais/data/nuplan-maps-v1.0}
WORK_DIR=/scratch/spapais/ForeSight/work_dirs
WANDB_PERSIST_DIR=/scratch/spapais/ForeSight/wandb
CMD=${@:-bash}

[[ -f ~/.bashrc ]] && source ~/.bashrc

APPTAINER=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/apptainer/1.4.5/bin/apptainer
SIF_PATH=${SIF_PATH:-/scratch/spapais/ForeSight/docker/foresight_navsim.sif}

# Build optional data binds. Skip openscene / nuplan-maps when the host dirs
# are missing so the smoke test (no data needed) still runs cleanly.
DATA_BINDS=""
[ -d "$OPENSCENE_HOST_DIR" ]   && DATA_BINDS+=" --bind=$OPENSCENE_HOST_DIR:/workspace/ForeSight/data/openscene"
[ -d "$NUPLAN_MAPS_HOST_DIR" ] && DATA_BINDS+=" --bind=$NUPLAN_MAPS_HOST_DIR:/workspace/ForeSight/data/nuplan-maps-v1.0"

CONTAINER_CMD="$APPTAINER exec --nv -c -e --pwd /workspace/ForeSight/ \
--env WANDB_API_KEY=$WANDB_API_KEY \
--env WANDB_MODE=offline \
--env WANDB_DIR=/wandb \
--env TMPDIR=/tmp \
--env FORESIGHT_ROOT=/workspace/ForeSight \
--env NAVSIM_DEVKIT_ROOT=/workspace/ForeSight/navsim \
--env NAVSIM_EXP_ROOT=/workspace/ForeSight/work_dirs/navsim \
--env OPENSCENE_DATA_ROOT=/workspace/ForeSight/data/openscene \
--env NUPLAN_MAPS_ROOT=/workspace/ForeSight/data/nuplan-maps-v1.0 \
--bind=$TMP_DIR:/tmp \
--bind=$WANDB_PERSIST_DIR:/wandb \
--bind=/home/spapais/ForeSight:/workspace/ForeSight/ \
$DATA_BINDS \
--bind=$WORK_DIR:/workspace/ForeSight/work_dirs \
$SIF_PATH"
CONTAINER_CMD="$CONTAINER_CMD bash -c 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH && $CMD'"

mkdir -p $TMP_DATA_DIR $TMP_DIR $WORK_DIR $WANDB_PERSIST_DIR

source /etc/profile.d/modules.sh
module load slurm/killarney/24.05.7

SECONDS=0
echo "[$((SECONDS/60))m]: Running command"
echo "$CONTAINER_CMD"
eval $CONTAINER_CMD
echo "[$((SECONDS/60))m]: Done"
