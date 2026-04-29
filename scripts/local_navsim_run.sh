#!/bin/bash
# Local docker runner for the NavSim image. Mirrors local_run.sh but mounts
# OpenScene blobs and nuPlan maps and uses the foresight_navsim image.
#
# Usage:
#   ./scripts/local_navsim_run.sh                          # interactive bash
#   ./scripts/local_navsim_run.sh bash scripts/navsim_train.sh navmini
#
# Override defaults with env vars:
#   IMAGE_NAME=foresight_navsim:cuda118pytorch21
#   OPENSCENE_HOST_DIR=/data/sets/openscene
#   NUPLAN_MAPS_HOST_DIR=/data/sets/nuplan-maps-v1.0

set -e

CODE_DIR=/home/trail/workspace/ForeSight
IMAGE_NAME=${IMAGE_NAME:-foresight_navsim:cuda118pytorch21}
WANDB_MODE_VALUE=${WANDB_MODE:-offline}

# Data roots on the host (override per machine)
OPENSCENE_HOST_DIR=${OPENSCENE_HOST_DIR:-/data/sets/openscene}
NUPLAN_MAPS_HOST_DIR=${NUPLAN_MAPS_HOST_DIR:-/data/sets/nuplan-maps-v1.0}

if [ "$#" -eq 0 ]; then
    CMD=(bash)
else
    CMD=("$@")
fi

# Build mount args only if the host dirs exist (so smoke tests work without data)
MOUNTS=(-v "$CODE_DIR:/workspace/ForeSight/")
[ -d "$OPENSCENE_HOST_DIR" ]    && MOUNTS+=(-v "$OPENSCENE_HOST_DIR:/workspace/ForeSight/data/openscene")
[ -d "$NUPLAN_MAPS_HOST_DIR" ]  && MOUNTS+=(-v "$NUPLAN_MAPS_HOST_DIR:/workspace/ForeSight/data/nuplan-maps-v1.0")

docker run --gpus all -it --rm --shm-size=16g \
    "${MOUNTS[@]}" \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env WANDB_MODE=$WANDB_MODE_VALUE \
    --env FORESIGHT_ROOT=/workspace/ForeSight \
    --env NAVSIM_DEVKIT_ROOT=/workspace/ForeSight/navsim \
    --env NAVSIM_EXP_ROOT=/workspace/ForeSight/work_dirs/navsim \
    --env OPENSCENE_DATA_ROOT=/workspace/ForeSight/data/openscene \
    --env NUPLAN_MAPS_ROOT=/workspace/ForeSight/data/nuplan-maps-v1.0 \
    -w /workspace/ForeSight/ \
    "$IMAGE_NAME" "${CMD[@]}"
