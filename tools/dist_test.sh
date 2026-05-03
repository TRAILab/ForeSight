#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29610}

# Default --tmpdir to /tmp/.dist_test if not specified.
# /tmp is always a fast local disk (bound to SLURM_TMPDIR/tmp on all clusters)
# and avoids read-only workspace issues on SciNet clusters (Trillium, Tamia).
[[ " ${@:4} " != *" --tmpdir "* ]] && TMPDIR_ARG="--tmpdir /tmp/.dist_test"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch $TMPDIR_ARG ${@:4}
