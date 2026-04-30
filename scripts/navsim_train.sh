#!/bin/bash
# NavSim v1 training entry. Wraps navsim's run_training.py with our SparseDriveAgent.
#
# Usage:
#   bash scripts/navsim_train.sh                              # navtrain default
#   bash scripts/navsim_train.sh navmini                      # smoke test
#   bash scripts/navsim_train.sh navtrain max_epochs=20       # extra hydra overrides

set -e

SPLIT="${1:-navtrain}"
shift || true
EXTRA_ARGS="$@"

# Load env vars (sets OPENSCENE_DATA_ROOT etc.). Allow NAVSIM_EXP_ROOT to fall
# back to /tmp if the default work_dirs path is unwritable.
: "${NAVSIM_EXP_ROOT:=/home/trail/workspace/ForeSight/work_dirs/navsim}"
if ! mkdir -p "${NAVSIM_EXP_ROOT}" 2>/dev/null; then
    NAVSIM_EXP_ROOT=/tmp/navsim_exp
    mkdir -p "${NAVSIM_EXP_ROOT}"
fi
export NAVSIM_EXP_ROOT
# Source with an explicit empty arg so navsim_setup.sh's case statement does
# not consume our wrapper's positional args.
source "$(dirname "$0")/navsim_setup.sh" ""

# Match the right OpenScene data_split for the train_test_split selection.
case "${SPLIT}" in
    navtrain|trainval)  DATA_SPLIT=trainval ;;
    navmini|mini)       DATA_SPLIT=mini ;;
    navtest|test)       DATA_SPLIT=test ;;
    *)                  DATA_SPLIT=trainval ;;
esac

EXP_NAME="sparsedrive_${SPLIT}_$(date +%Y%m%d_%H%M%S)"

python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training.py" \
    agent=sparsedrive_agent \
    experiment_name="${EXP_NAME}" \
    train_test_split="${SPLIT}" \
    split="${DATA_SPLIT}" \
    cache_path="${NAVSIM_EXP_ROOT}/training_cache/sparsedrive_${SPLIT}/" \
    use_cache_without_dataset=False \
    force_cache_computation=False \
    ${EXTRA_ARGS}
