#!/bin/bash
#SBATCH --job-name=train_stream_petr_velforecast_vov_flash_800_bs2_seq_24e_2gpu    # Job name
#SBATCH --account=rrg-swasland
#SBATCH --ntasks=1                    # Run on n CPUs
#SBATCH --mem=120gb                     # Job memory request
#SBATCH --time=2:59:00               # Time limit hrs:min:sec
#SBATCH --output=/home/spapais/output/streampetr_jdmp/%x-%j.log   # Standard output and error log
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:2           # gpu:t4:4 (graham) or gpu:a100:1 (narval)
#SBATCH --mail-user="sandro.papais@robotics.utias.utoronto.ca"
#SBATCH --mail-type=ALL

# Parameters
SERVER=narval
DATASET=nuscenes
NUM_GPUS=2
CFG_NAME=stream_petr_velforecast_vov_flash_800_bs2_seq_24e_2gpu

# Host paths
HOME_DIR=/home/spapais
TMP_DATA_DIR=$SLURM_TMPDIR/data
# TMP_DATA_DIR=/home/spapais/scratch/temp_data # Slurm unzip alternative
PROJ_DIR=$HOME_DIR/StreamPETR-JDMP
OUT_DIR=$HOME_DIR/output/streampetr_jdmp
SING_IMG=/home/spapais/projects/rrg-swasland/spapais/singularity_images/streampetr.sif
if [ "$SERVER" = "graham" ]; then
    DATA_DIR=/home/spapais/projects/rrg-swasland/Datasets/nuscenes
    DATA_PKL_DIR=/home/spapais/projects/rrg-swasland/Datasets/nuscenes
fi
if [ "$SERVER" = "narval" ]; then
    DATA_DIR=/home/spapais/projects/rrg-swasland/datasets/nuscenes/
    DATA_PKL_DIR=/home/spapais/datasets/nuscenes/
fi

# Container paths
VOLUMES="--bind=$PROJ_DIR:/proj
         --bind=$TMP_DATA_DIR:/proj/data/nuscenes
         --bind=$OUT_DIR:/proj/output
        "
CFG_FILE=projects/configs/StreamPETR/$CFG_NAME.py
WRK_DIR=output/train_$CFG_NAME/

# Command
WANDB_MODE='offline'
BASE_CMD="./tools/dist_train.sh $CFG_FILE $NUM_GPUS --work-dir $WRK_DIR"
CONTAINER_CMD="apptainer exec --nv -c -e --pwd /proj/ \
--env "WANDB_API_KEY=$WANDB_API_KEY"
--env "WANDB_MODE=$WANDB_MODE"
$VOLUMES \
$SING_IMG \
$BASE_CMD
"

# Start script
SECONDS=0
echo "SLURM_JOB_ID=$SLURM_JOB_ID
CFG_NAME=$CFG_NAME
NUM_GPUS=$NUM_GPUS
"
# Extract dataset
echo "Extracting data"
if [ "$DATASET" = "nuscenes_mini" ]; then
    mkdir $TMP_DATA_DIR
    duration=$SECONDS
    file=$DATA_PKL_DIR/v1.0-mini.tgz
    echo "[$((duration/3600))h$((duration%3600/60))m]: Unzipping $file to $TMP_DATA_DIR"
    tar -xf $file -C $TMP_DATA_DIR
    duration=$SECONDS
    file=$DATA_DIR/maps.zip
    echo "[$((duration/3600))h$(((duration%3600)/60))m]: Unzipping $file to $TMP_DATA_DIR"
    unzip -o -qq $DATA_DIR/maps.zip -d $TMP_DATA_DIR
    duration=$SECONDS
    file=$DATA_PKL_DIR/nuscenes2d_mini_temporal_infos_train.pkl
    echo "[$((duration/3600))h$(((duration%3600)/60))m]: Copying $file to $TMP_DATA_DIR"
    cp $file $TMP_DATA_DIR
    duration=$SECONDS
    file=$DATA_PKL_DIR/nuscenes2d_mini_temporal_infos_val.pkl
    echo "[$((duration/3600))h$(((duration%3600)/60))m]: Copying $file to $TMP_DATA_DIR"
    cp $file $TMP_DATA_DIR
fi
if [ "$DATASET" = "nuscenes" ]; then
    for file in $DATA_DIR/*.zip; do
        duration=$SECONDS
        echo "[$((duration/3600))h$((duration%3600/60))m]: Unzipping $file to $TMP_DATA_DIR"
        unzip -qq $file -d $TMP_DATA_DIR
    done
    for file in $DATA_PKL_DIR/*.pkl; do
        duration=$SECONDS
        echo "[$((duration/3600))h$(((duration%3600)/60))m]: Copying $file to $TMP_DATA_DIR"
        cp $file $TMP_DATA_DIR
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