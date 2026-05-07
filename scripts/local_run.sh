#!/bin/bash
# Cuda 11.6 usage: ./scripts/local_run.sh [command]
# Cuda 11.8 usage: IMAGE_NAME=foresight_cuda118 ./scripts/local_run.sh [command]

# Directories
DATA_DIR=/data/sets/nuscenes
CODE_DIR=/home/trail/workspace/ForeSight
IMAGE_NAME=${IMAGE_NAME:-foresight}
WANDB_MODE_VALUE=${WANDB_MODE:-offline}

if [ "$#" -eq 0 ]; then
    CMD=(bash)
else
    CMD=("$@")
fi

# Run docker container
docker run --gpus all -i --rm --shm-size=16g \
    -v $DATA_DIR:/workspace/ForeSight/data/nuscenes \
    -v $CODE_DIR:/workspace/ForeSight/ \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env WANDB_MODE=$WANDB_MODE_VALUE \
    -w /workspace/ForeSight/ \
    $IMAGE_NAME "${CMD[@]}"
