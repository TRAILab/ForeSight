#!/usr/bin/env bash
# Periodically rsyncs offline wandb dirs from running SLURM jobs to a
# persistent location. Useful when wandb runs land on volatile storage
# (e.g. SLURM_TMPDIR / /dev/shm) — without this, the runs vanish at job end.
#
# Usage (run inside tmux on a login node):
#   bash scripts/wandb_rerrsync.sh <jobid> [<jobid>...]
#
# Overrides via env: DEST, INTERVAL, SRC_TEMPLATE.
# SRC_TEMPLATE uses {jid} and {user} placeholders. Default matches Trillium
# (SLURM_TMPDIR is RAM-backed at /dev/shm/slurm.<user>.<jid>/tmp).

set -u

DEST=${DEST:-/scratch/spapais/ForeSight/wandb}
INTERVAL=${INTERVAL:-600}
SRC_TEMPLATE=${SRC_TEMPLATE:-/dev/shm/slurm.{user}.{jid}/tmp/wandb}

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
            SRC=${SRC_TEMPLATE//\{user\}/$USER}
            SRC=${SRC//\{jid\}/$JID}
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
