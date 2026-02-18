#!/bin/bash

# Directories
DATA_DIR=/scratch/hpc_nas/datasets/nuscenes/v1.0-trainval/
CODE_DIR=/home/spapais/ForeSight/
CMD=${@:-bash}

# Run docker container
docker run --gpus all -it --rm --shm-size=16g \
    -v $DATA_DIR:/workspace/ForeSight/data/nuscenes \
    -v $CODE_DIR:/workspace/ForeSight/ \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    -w /workspace/ForeSight/ \
    foresight:latest $CMD
