#!/usr/bin/env bash
# Periodically rsyncs offline wandb dirs from running SLURM jobs to a
# persistent location. Useful when wandb runs land on volatile storage
# (e.g. SLURM_TMPDIR / /dev/shm) — without this, the runs vanish at job end.
#
# Usage (run inside tmux on a login node):
#   bash scripts/wandb_rerrsync.sh <jobid> [<jobid>...]
#
# Overrides via env: DEST, INTERVAL.
# Source path is hardcoded for Trillium (SLURM_TMPDIR is RAM-backed at
# /dev/shm/slurm.<user>.<jid>/tmp). Edit src_for_job() for other clusters.

set -u

DEST=${DEST:-/scratch/spapais/ForeSight/wandb}
INTERVAL=${INTERVAL:-600}

src_for_job() {
    local jid=$1
    echo "/dev/shm/slurm.${USER}.${jid}/tmp/wandb"
}

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <jobid> [<jobid>...]" >&2
    exit 1
fi

JIDS=("$@")
mkdir -p "$DEST"
echo "DEST:     $DEST"
echo "INTERVAL: ${INTERVAL}s"
echo "Watching jobs: ${JIDS[*]}"

while :; do
    ANY_RUNNING=0
    for JID in "${JIDS[@]}"; do
        STATE=$(squeue -j "$JID" -h -o %T 2>/dev/null)
        if [[ "$STATE" == "RUNNING" ]]; then
            ANY_RUNNING=1
            SRC=$(src_for_job "$JID")
            echo "$(date) [job $JID] rsync from $SRC"
            srun --jobid="$JID" --overlap rsync -a "$SRC/" "$DEST/" 2>&1 | tail -3
        else
            echo "$(date) [job $JID] not running (state: ${STATE:-unknown})"
        fi
    done
    if [[ $ANY_RUNNING -eq 0 ]]; then
        echo "All watched jobs ended; exiting"
        break
    fi
    sleep "$INTERVAL"
done
