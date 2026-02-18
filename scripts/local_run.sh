#!/bin/bash

# Directories
DATA_DIR=/data/sets/nuscenes
CODE_DIR=/home/trail/workspace/ForeSight
CMD=${@:-bash}

# Run docker container
docker run --gpus all -it --rm --shm-size=16g \
    -v $DATA_DIR:/workspace/ForeSight/data/nuscenes \
    -v $CODE_DIR:/workspace/ForeSight/ \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    -w /workspace/ForeSight/ \
    foresight:latest $CMD
