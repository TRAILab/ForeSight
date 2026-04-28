#!/usr/bin/env bash
# Continuously syncs wandb offline runs to wandb cloud.
# Run interactively in tmux on a LOGIN node (compute nodes lack internet):
#   bash scripts/wandb_sync.sh [narval|trillium|killarney]
# Without arg, auto-detects from $(hostname).
# Overrides: REPO_DIR, WANDB_DIR, SING_IMG, APPTAINER, SLEEP_INTERVAL.

set -u

REPO_DIR=${REPO_DIR:-/home/spapais/ForeSight}
SLEEP_INTERVAL=${SLEEP_INTERVAL:-600}

HOST="${1:-}"
if [[ -z "$HOST" ]]; then
    case "$(hostname -f 2>/dev/null || hostname)" in
        *narval*)             HOST=narval ;;
        *trillium*)           HOST=trillium ;;
        *killarney*|*kln*)    HOST=killarney ;;
        *) echo "Could not auto-detect host; pass narval|trillium|killarney" >&2; exit 1 ;;
    esac
fi

case "$HOST" in
    narval)
        SING_IMG=${SING_IMG:-$REPO_DIR/docker/foresight.sif}
        WANDB_DIR=${WANDB_DIR:-$REPO_DIR/wandb}
        APPTAINER=${APPTAINER:-apptainer}
        module load StdEnv/2020
        module load apptainer
        ;;
    trillium)
        SING_IMG=${SING_IMG:-$REPO_DIR/docker/foresight_cuda118pytorch21.sif}
        WANDB_DIR=${WANDB_DIR:-/scratch/spapais/ForeSight/wandb}
        APPTAINER=${APPTAINER:-apptainer}
        module load StdEnv/2023
        module load apptainer
        ;;
    killarney)
        SING_IMG=${SING_IMG:-$REPO_DIR/docker/foresight_cuda118.sif}
        WANDB_DIR=${WANDB_DIR:-/scratch/spapais/ForeSight/wandb}
        APPTAINER=${APPTAINER:-/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/apptainer/1.4.5/bin/apptainer}
        source /etc/profile.d/modules.sh
        module load slurm/killarney/24.05.7
        ;;
    *) echo "Unknown host: $HOST (use narval|trillium|killarney)" >&2; exit 1 ;;
esac

echo "Host:        $HOST"
echo "REPO_DIR:    $REPO_DIR"
echo "WANDB_DIR:   $WANDB_DIR"
echo "SING_IMG:    $SING_IMG"
echo "APPTAINER:   $APPTAINER"
echo "Starting wandb sync loop (interval: ${SLEEP_INTERVAL}s)"

while :; do
    date
    if compgen -G "$WANDB_DIR/offline-*" > /dev/null; then
        echo "Syncing offline runs in $WANDB_DIR"
        "$APPTAINER" --silent exec -c -e \
            --env "WANDB_API_KEY=$WANDB_API_KEY" \
            --bind="$WANDB_DIR":/wandb_sync \
            "$SING_IMG" \
            wandb sync --sync-all /wandb_sync
    else
        echo "No offline wandb runs in $WANDB_DIR"
    fi
    echo "Done — sleeping ${SLEEP_INTERVAL}s"
    sleep "$SLEEP_INTERVAL"
done
