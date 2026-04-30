#!/bin/bash
# NavSim v1 data download + env-var setup helper.
#
# Usage:
#   source scripts/navsim_setup.sh                  # just export env vars
#   bash scripts/navsim_setup.sh download_maps      # individual stage
#   bash scripts/navsim_setup.sh download_test      # navtest sensor data
#   bash scripts/navsim_setup.sh download_navtrain  # ~500 GB, takes hours
#   bash scripts/navsim_setup.sh download_navmini   # small sanity split
#   bash scripts/navsim_setup.sh metric_cache       # build navtest metric cache
#   bash scripts/navsim_setup.sh all                # everything (NOT recommended)
#
# Layout produced under $OPENSCENE_DATA_ROOT (matches upstream navsim Hydra
# config that resolves paths as ${OPENSCENE_DATA_ROOT}/navsim_logs/<split>
# and ${OPENSCENE_DATA_ROOT}/sensor_blobs/<split>):
#   $OPENSCENE_DATA_ROOT/
#       navsim_logs/{trainval,test,mini}/       (meta pkls)
#       sensor_blobs/{trainval,test,mini}/      (camera + lidar blobs)
#   $NUPLAN_MAPS_ROOT/maps/

set -e

# ----- Paths (override before sourcing if you want a different layout) -----
: "${FORESIGHT_ROOT:=/home/trail/workspace/ForeSight}"
: "${NAVSIM_DEVKIT_ROOT:=${FORESIGHT_ROOT}/navsim}"
: "${NAVSIM_EXP_ROOT:=${FORESIGHT_ROOT}/work_dirs/navsim}"
: "${OPENSCENE_DATA_ROOT:=${FORESIGHT_ROOT}/data/openscene}"
: "${NUPLAN_MAPS_ROOT:=${FORESIGHT_ROOT}/data/nuplan-maps-v1.0}"

export FORESIGHT_ROOT NAVSIM_DEVKIT_ROOT NAVSIM_EXP_ROOT OPENSCENE_DATA_ROOT NUPLAN_MAPS_ROOT
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OPENSCENE_DATA_ROOT}" "${NUPLAN_MAPS_ROOT}" "${NAVSIM_EXP_ROOT}"

# ----- Sub-commands -----
download_maps() {
    cd "$(dirname "${NUPLAN_MAPS_ROOT}")"
    wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip
    unzip nuplan-maps-v1.1.zip
    rm nuplan-maps-v1.1.zip
}

download_test() {
    cd "${OPENSCENE_DATA_ROOT}"
    wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_test.tgz
    tar -xzf openscene_metadata_test.tgz && rm openscene_metadata_test.tgz
    for split in {0..31}; do
        wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_camera/openscene_sensor_test_camera_${split}.tgz
        tar -xzf openscene_sensor_test_camera_${split}.tgz && rm openscene_sensor_test_camera_${split}.tgz
    done
    for split in {0..31}; do
        wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_lidar/openscene_sensor_test_lidar_${split}.tgz
        tar -xzf openscene_sensor_test_lidar_${split}.tgz && rm openscene_sensor_test_lidar_${split}.tgz
    done
    mv openscene-v1.1/meta_datas test_navsim_logs
    mv openscene-v1.1/sensor_blobs test_sensor_blobs
    rm -r openscene-v1.1
}

download_navtrain() {
    cd "${OPENSCENE_DATA_ROOT}"
    wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz
    tar -xzf openscene_metadata_trainval.tgz && rm openscene_metadata_trainval.tgz
    mv openscene-v1.1/meta_datas trainval_navsim_logs
    rm -r openscene-v1.1
    mkdir -p trainval_sensor_blobs/trainval
    for split in {1..4}; do
        wget https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim/navtrain_current_${split}.tgz
        tar -xzf navtrain_current_${split}.tgz && rm navtrain_current_${split}.tgz
        rsync -rv current_split_${split}/* trainval_sensor_blobs/trainval && rm -r current_split_${split}
    done
    for split in {1..4}; do
        wget https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim/navtrain_history_${split}.tgz
        tar -xzf navtrain_history_${split}.tgz && rm navtrain_history_${split}.tgz
        rsync -rv history_split_${split}/* trainval_sensor_blobs/trainval && rm -r history_split_${split}
    done
}

download_navmini() {
    cd "${OPENSCENE_DATA_ROOT}"
    wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_mini.tgz
    tar -xzf openscene_metadata_mini.tgz && rm openscene_metadata_mini.tgz
    for split in {0..1}; do
        wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_${split}.tgz
        tar -xzf openscene_sensor_mini_camera_${split}.tgz && rm openscene_sensor_mini_camera_${split}.tgz
    done
    for split in {0..1}; do
        wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_mini_lidar/openscene_sensor_mini_lidar_${split}.tgz
        tar -xzf openscene_sensor_mini_lidar_${split}.tgz && rm openscene_sensor_mini_lidar_${split}.tgz
    done
    # Move into the upstream-compatible layout: navsim_logs/<split>/ + sensor_blobs/<split>/.
    mkdir -p navsim_logs sensor_blobs
    mv openscene-v1.1/meta_datas/mini navsim_logs/mini
    mv openscene-v1.1/sensor_blobs/mini sensor_blobs/mini
    rm -r openscene-v1.1
}

metric_cache() {
    python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_metric_caching.py" \
        train_test_split=navtest \
        cache.cache_path="${NAVSIM_EXP_ROOT}/metric_cache"
}

cmd="${1:-}"
case "${cmd}" in
    "")           echo "Env vars exported. Subcommand expected: download_maps | download_test | download_navtrain | download_navmini | metric_cache | all" ;;
    download_maps)       download_maps ;;
    download_test)       download_test ;;
    download_navtrain)   download_navtrain ;;
    download_navmini)    download_navmini ;;
    metric_cache)        metric_cache ;;
    all)                 download_maps && download_test && download_navtrain && metric_cache ;;
    *)                   echo "Unknown subcommand: ${cmd}" && exit 1 ;;
esac
