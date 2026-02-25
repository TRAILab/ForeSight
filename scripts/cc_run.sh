#!/bin/bash
#SBATCH --job-name=foresight
#SBATCH --account=rrg-swasland
#SBATCH --ntasks=1
#SBATCH --mem=120gb
#SBATCH --time=11:59:00 # 3 hours or 12 hours max recommended
#SBATCH --output=/home/spapais/ForeSight/logs/%x-%j.log
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:4 # gpu:h100:4 (trillium) or gpu:a100:4 (narval)
#SBATCH --mail-user="sandro.papais@robotics.utias.utoronto.ca"
#SBATCH --mail-type=END,FAIL

# Parameters
SERVER=narval
DATASET=nuscenes
NUM_GPUS=4
CFG_NAME=stream_petr_velforecast_vov_flash_800_bs2_seq_24e_2gpu

# Host paths
HOME_DIR=/home/spapais
TMP_DATA_DIR=$SLURM_TMPDIR/data
# TMP_DATA_DIR=/home/spapais/scratch/temp_data # Slurm unzip alternative
PROJ_DIR=$HOME_DIR/ForeSight
SING_IMG=docker/foresight.sif
if [ "$SERVER" = "graham" ]; then
    DATA_DIR=/home/spapais/projects/rrg-swasland/Datasets/nuscenes
    DATA_PKL_DIR=/home/spapais/projects/rrg-swasland/Datasets/nuscenes
fi
if [ "$SERVER" = "trillium" ]; then
    DATA_DIR=/home/spapais/projects/rrg-swasland/datasets/nuscenes/
    DATA_PKL_DIR=/home/spapais/datasets/nuscenes/
fi

# Command
CMD=${@:-bash}
WANDB_MODE='offline'
CONTAINER_CMD="apptainer exec --nv -c -e --pwd /workspace/ForeSight/ \
--env "WANDB_API_KEY=$WANDB_API_KEY"
--env "WANDB_MODE=$WANDB_MODE"
--bind=$PROJ_DIR:/workspace/ForeSight/ \
--bind=$TMP_DATA_DIR:/workspace/ForeSight/data/nuscenes \
docker/foresight.sif CMD
"

# Start script
SECONDS=0
echo "SLURM_JOB_ID=$SLURM_JOB_ID
CFG_NAME=$CFG_NAME
NUM_GPUS=$NUM_GPUS
"
# Extract dataset
echo "Extracting data"
if [ "$DATASET" = "nuscenes" ]; then
    for file in $DATA_DIR/*.zip; do
        duration=$SECONDS
        echo "[$((duration/3600))h$((duration%3600/60))m]: Unzipping $file to $TMP_DATA_DIR"
        unzip -qq $file -d $TMP_DATA_DIR
    done
fi
echo "Done extracting data"

# Run command
# echo "Debug mode: sleep engaged" && sleep 5d
module load StdEnv/2020
module load apptainer
duration=$SECONDS
echo "[$((duration/3600))h$(((duration%3600)/60))m]: Running command"
echo "$CONTAINER_CMD"
eval $CONTAINER_CMD
duration=$SECONDS
echo "[$((duration/3600))h$(((duration%3600)/60))m]: Done"