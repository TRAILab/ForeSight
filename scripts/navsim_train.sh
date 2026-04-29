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

# Load env vars (sets OPENSCENE_DATA_ROOT etc.)
source "$(dirname "$0")/navsim_setup.sh"

EXP_NAME="sparsedrive_${SPLIT}_$(date +%Y%m%d_%H%M%S)"

python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training.py" \
    agent=sparsedrive_agent \
    experiment_name="${EXP_NAME}" \
    train_test_split="${SPLIT}" \
    split=trainval \
    cache_path="${NAVSIM_EXP_ROOT}/training_cache/sparsedrive_${SPLIT}/" \
    use_cache_without_dataset=False \
    force_cache_computation=False \
    ${EXTRA_ARGS}
