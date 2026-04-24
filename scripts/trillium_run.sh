#!/bin/bash
#SBATCH --job-name=foresight
#SBATCH --account=rrg-swasland
#SBATCH --ntasks=1
#SBATCH --time=11:59:00 # 3 hours or 12 hours max recommended
#SBATCH --output=/scratch/spapais/ForeSight/logs/%x-%j.log
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=4
#SBATCH --mail-user="sandro.papais@robotics.utias.utoronto.ca"
#SBATCH --mail-type=END,FAIL

# Parameters
TMP_DIR=$SLURM_TMPDIR/tmp
TMP_DATA_DIR=$SLURM_TMPDIR/data
# TMP_DATA_DIR=/home/spapais/scratch/temp_data # Temporary data directory alternative
DATA_DIR=/home/spapais/links/projects/rrg-swasland/datasets/nuscenes/
WORK_DIR=/scratch/spapais/ForeSight/work_dirs
CMD=${@:-bash}

# Load env if needed (e.g. when submitted via non-interactive SSH)
[[ -f ~/.bashrc ]] && source ~/.bashrc

# Command
CONTAINER_CMD="apptainer exec --nv -c -e --pwd /workspace/ForeSight/ \
--env "WANDB_API_KEY=$WANDB_API_KEY"
--env "WANDB_MODE=offline"
--env "TMPDIR=/tmp"
--bind=$TMP_DIR:/tmp \
--bind=/home/spapais/ForeSight:/workspace/ForeSight/ \
--bind=$TMP_DATA_DIR:/workspace/ForeSight/data/nuscenes \
--bind=$WORK_DIR:/workspace/ForeSight/work_dirs \
docker/foresight_cuda118.sif $CMD
"

# Extract dataset
SECONDS=0
echo "Extracting data"
mkdir -p $TMP_DATA_DIR $TMP_DIR $WORK_DIR
for file in $DATA_DIR/*.zip; do
    [[ "$file" == *sweeps* ]] && echo "Skipping $file (not needed for camera-only model)" && continue
    duration=$SECONDS
    echo "[$((duration/3600))h$((duration%3600/60))m]: Unzipping $file to $TMP_DATA_DIR"
    unzip -qq $file -d $TMP_DATA_DIR
done
echo "Done extracting data"

# Run command
# echo "Debug mode: sleep engaged" && sleep 5d # Uncomment to debug
module load StdEnv/2023
module load apptainer
duration=$SECONDS
echo "[$((duration/3600))h$(((duration%3600)/60))m]: Running command"
echo "$CONTAINER_CMD"
eval $CONTAINER_CMD
duration=$SECONDS
echo "[$((duration/3600))h$(((duration%3600)/60))m]: Done"