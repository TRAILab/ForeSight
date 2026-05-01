#!/bin/bash
# NavSim per-token feature/target cache build via Ray-parallel workers.
#
# This is the prerequisite for fast training. The default Dataset.cache_dataset
# in navsim is single-threaded and takes ~5 hr for navtrain on a single L40S;
# this script uses run_dataset_caching.py which Ray-parallels across all
# available CPUs and is 5-10x faster.
#
# Usage:
#   bash scripts/navsim_cache.sh                      # navtrain default
#   bash scripts/navsim_cache.sh navmini              # smoke
#   bash scripts/navsim_cache.sh navtrain max_epochs=20  # extra hydra overrides

set -e

SPLIT="${1:-navtrain}"
shift || true
EXTRA_ARGS="$@"

# Source navsim_setup.sh with explicit empty arg so its case statement does
# not consume our wrapper's positional args.
: "${NAVSIM_EXP_ROOT:=$(dirname "$(realpath "$0")")/..}/work_dirs/navsim"
if ! mkdir -p "${NAVSIM_EXP_ROOT}" 2>/dev/null; then
    NAVSIM_EXP_ROOT=/tmp/navsim_exp
    mkdir -p "${NAVSIM_EXP_ROOT}"
fi
export NAVSIM_EXP_ROOT
source "$(dirname "$0")/navsim_setup.sh" ""

case "${SPLIT}" in
    navtrain|trainval)  DATA_SPLIT=trainval ;;
    navmini|mini)       DATA_SPLIT=mini ;;
    navtest|test)       DATA_SPLIT=test ;;
    *)                  DATA_SPLIT=trainval ;;
esac

CACHE_PATH="${NAVSIM_EXP_ROOT}/training_cache/sparsedrive_${SPLIT}"
echo "Caching ${SPLIT} features to ${CACHE_PATH}"

python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_dataset_caching.py" \
    agent=sparsedrive_agent \
    experiment_name="cache_${SPLIT}" \
    train_test_split="${SPLIT}" \
    split="${DATA_SPLIT}" \
    cache_path="${CACHE_PATH}" \
    force_cache_computation=False \
    ${EXTRA_ARGS}
