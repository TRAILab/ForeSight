#!/bin/bash
# Apollo runner for the foresight_navsim singularity image.
#
# Apollo is interactive (no SLURM); runs the container directly.
# Use docker via the existing apollo_run.sh for the nuScenes path; this
# uses singularity (matches the killarney/dgx navsim pattern).
#
# Usage:
#   ./scripts/apollo_navsim_run.sh                              # interactive bash
#   ./scripts/apollo_navsim_run.sh bash scripts/navsim_train.sh navmini
#
# Override defaults with env vars:
#   SIF_PATH=/home/spapais/ForeSight/docker/foresight_navsim.sif
#   OPENSCENE_HOST_DIR=/home/spapais/data/openscene
#   NUPLAN_MAPS_HOST_DIR=/home/spapais/data/nuplan-maps-v1.0

[[ -f ~/.bashrc ]] && source ~/.bashrc

CODE_DIR=/home/spapais/ForeSight
OPENSCENE_HOST_DIR=${OPENSCENE_HOST_DIR:-/home/spapais/data/openscene}
NUPLAN_MAPS_HOST_DIR=${NUPLAN_MAPS_HOST_DIR:-/home/spapais/data/nuplan-maps-v1.0}
SIF_PATH=${SIF_PATH:-${CODE_DIR}/docker/foresight_navsim.sif}

CMD=${@:-bash}

# Skip data binds when host dirs are missing so the unit smoke runs cleanly.
DATA_BINDS=""
[ -d "$OPENSCENE_HOST_DIR" ]   && DATA_BINDS+=" --bind=$OPENSCENE_HOST_DIR:/workspace/ForeSight/data/openscene"
[ -d "$NUPLAN_MAPS_HOST_DIR" ] && DATA_BINDS+=" --bind=$NUPLAN_MAPS_HOST_DIR:/workspace/ForeSight/data/nuplan-maps-v1.0"

singularity exec --nv -e --pwd /workspace/ForeSight/ \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env WANDB_MODE=${WANDB_MODE:-online} \
    --env FORESIGHT_ROOT=/workspace/ForeSight \
    --env NAVSIM_DEVKIT_ROOT=/workspace/ForeSight/navsim \
    --env NAVSIM_EXP_ROOT=/workspace/ForeSight/work_dirs/navsim \
    --env OPENSCENE_DATA_ROOT=/workspace/ForeSight/data/openscene \
    --env NUPLAN_MAPS_ROOT=/workspace/ForeSight/data/nuplan-maps-v1.0 \
    --bind=${CODE_DIR}:/workspace/ForeSight/ \
    $DATA_BINDS \
    "$SIF_PATH" \
    bash -c "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH && $CMD"
