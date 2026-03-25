#!/bin/bash
#SBATCH --job-name=foresight
#SBATCH --ntasks=1
#SBATCH --mem=440gb # 440gb available
#SBATCH --time=3-00:00:00
#SBATCH --output=./logs/%x-%j.log
#SBATCH --cpus-per-task=120 # 120 cores available
#SBATCH --gres=gpu:4 # 4 gpus available
#SBATCH --mail-user="sandro.papais@robotics.utias.utoronto.ca"
#SBATCH --mail-type=END,FAIL

# Paths
DATA_DIR=/raid/datasets/nuscenes
CODE_DIR=/raid/home/spapais/ForeSight

# Default command
CMD=${@:-bash}

# Load env if needed (e.g. when submitted via non-interactive SSH)
[[ -f ~/.bashrc ]] && source ~/.bashrc

# Run
echo "Running: $CMD"
singularity exec --nv -e --pwd /workspace/ForeSight/ \
    --env="WANDB_API_KEY=$WANDB_API_KEY" \
    --bind=$CODE_DIR:/workspace/ForeSight/ \
    --bind=$DATA_DIR:/workspace/ForeSight/data/nuscenes \
    docker/foresight.sif $CMD
echo "Done"
