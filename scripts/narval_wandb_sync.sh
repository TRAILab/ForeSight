#!/usr/bin/env bash
# Syncs wandb offline runs from ForeSight work_dirs to wandb cloud.
# Run interactively on Narval (not via SLURM):
#   bash scripts/narval_wandb_sync.sh

REPO_DIR=/home/spapais/ForeSight
SING_IMG=$REPO_DIR/docker/foresight.sif
WANDB_DIR=$REPO_DIR/wandb
SLEEP_INTERVAL=600  # seconds between sync passes

module load StdEnv/2020
module load apptainer

echo "Starting wandb sync loop (interval: ${SLEEP_INTERVAL}s)"

while :
do
    date
    RUN_DIRS=("$WANDB_DIR"/offline-run-*/  "$WANDB_DIR"/offline-*/  )
    FOUND=0
    for RUN_DIR in "${RUN_DIRS[@]}"; do
        [[ -d "$RUN_DIR" ]] || continue
        FOUND=1
        echo "Syncing $RUN_DIR"
        apptainer --silent exec --nv -c -e --pwd /workspace/ForeSight/ \
            --env "WANDB_API_KEY=$WANDB_API_KEY" \
            --bind="$REPO_DIR":/workspace/ForeSight/ \
            "$SING_IMG" \
            wandb sync "$RUN_DIR"
    done
    [[ $FOUND -eq 0 ]] && echo "No offline wandb runs found in $WANDB_DIR"
    echo "Done syncing — sleeping for ${SLEEP_INTERVAL}s"
    sleep "$SLEEP_INTERVAL"
done
