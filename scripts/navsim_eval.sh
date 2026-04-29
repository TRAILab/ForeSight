#!/bin/bash
# NavSim v1 PDM-Score evaluation. Wraps navsim's run_pdm_score.py.
#
# Usage:
#   bash scripts/navsim_eval.sh /path/to/checkpoint.pth
#   bash scripts/navsim_eval.sh /path/to/checkpoint.pth navtest worker=ray_distributed
#
# Requires the metric cache to have been built first; see
#   bash scripts/navsim_setup.sh metric_cache

set -e

CKPT="${1:?usage: navsim_eval.sh <checkpoint_path> [split] [extra hydra args]}"
SPLIT="${2:-navtest}"
shift 2 || shift 1 || true
EXTRA_ARGS="$@"

source "$(dirname "$0")/navsim_setup.sh"

EXP_NAME="sparsedrive_${SPLIT}_eval_$(date +%Y%m%d_%H%M%S)"

python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score.py" \
    train_test_split="${SPLIT}" \
    agent=sparsedrive_agent \
    agent.checkpoint_path="${CKPT}" \
    worker=single_machine_thread_pool \
    metric_cache_path="${NAVSIM_EXP_ROOT}/metric_cache" \
    experiment_name="${EXP_NAME}" \
    ${EXTRA_ARGS}
