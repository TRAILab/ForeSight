#!/bin/bash
#SBATCH --job-name=foresight
#SBATCH --account=rrg-swasland
#SBATCH --ntasks=1
#SBATCH --mem=120gb
#SBATCH --time=11:59:00 # 3 hours or 12 hours max recommended
#SBATCH --output=/home/spapais/ForeSight/logs/%x-%j.log
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:2
#SBATCH --mail-user="sandro.papais@robotics.utias.utoronto.ca"
#SBATCH --mail-type=END,FAIL

# Parameters
TMP_DATA_DIR=$SLURM_TMPDIR/data
# TMP_DATA_DIR=/home/spapais/scratch/temp_data # Temporary data directory alternative
DATA_DIR=/home/spapais/projects/rrg-swasland/datasets/nuscenes/
CMD=${@:-bash}

# Command
CONTAINER_CMD="apptainer exec --nv -c -e --pwd /workspace/ForeSight/ \
--env "WANDB_API_KEY=$WANDB_API_KEY"
--env "WANDB_MODE=offline"
--bind=/home/spapais/ForeSight:/workspace/ForeSight/ \
--bind=$TMP_DATA_DIR:/workspace/ForeSight/data/nuscenes \
docker/foresight.sif CMD
"

# Extract dataset
SECONDS=0
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
# echo "Debug mode: sleep engaged" && sleep 5d # Uncomment to debug
module load StdEnv/2020
module load apptainer
duration=$SECONDS
echo "[$((duration/3600))h$(((duration%3600)/60))m]: Running command"
echo "$CONTAINER_CMD"
eval $CONTAINER_CMD
duration=$SECONDS
echo "[$((duration/3600))h$(((duration%3600)/60))m]: Done"