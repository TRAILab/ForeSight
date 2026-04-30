"""End-to-end smoke test on the navmini split.

Constructs a real navsim Scene from the downloaded navmini blobs, runs the
SparseDriveFeatureBuilder + SparseDriveTargetBuilder, and pushes a single
sample through the SparseDriveAgent in eval mode. Verifies the planning
trajectory tensor has the expected shape.

Run inside the foresight_navsim container:
    docker run --gpus all --rm \
        -v /home/trail/workspace/ForeSight:/workspace/ForeSight \
        -v /home/trail/workspace/ForeSight/data/openscene/openscene-v1.1:/workspace/ForeSight/data/openscene/openscene-v1.1 \
        -w /workspace/ForeSight \
        -e FORESIGHT_ROOT=/workspace/ForeSight \
        -e OPENSCENE_DATA_ROOT=/workspace/ForeSight/data/openscene \
        -e NUPLAN_MAPS_ROOT=/workspace/ForeSight/data/nuplan-maps-v1.0 \
        foresight_navsim:cuda118pytorch21 \
        python scripts/navsim_navmini_smoke.py
"""

from pathlib import Path
import os
import sys
import traceback

import torch


def main():
    sys.path.insert(0, "/workspace/ForeSight")
    os.environ.setdefault("FORESIGHT_ROOT", "/workspace/ForeSight")

    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import SceneLoader
    from navsim.agents.sparsedrive.sparsedrive_agent import SparseDriveAgent
    from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig

    openscene_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    # navmini layout (post download_navmini):
    #     openscene_root/navsim_logs/mini/<log>.pkl
    #     openscene_root/sensor_blobs/mini/<sensor data>
    data_path = openscene_root / "navsim_logs" / "mini"
    sensor_blobs_path = openscene_root / "sensor_blobs" / "mini"
    if not data_path.is_dir():
        print(f"ERROR: navmini logs dir not found at {data_path}", file=sys.stderr)
        sys.exit(1)
    if not sensor_blobs_path.is_dir():
        print(f"ERROR: navmini sensor blobs dir not found at {sensor_blobs_path}", file=sys.stderr)
        sys.exit(1)

    cfg = SparseDriveConfig(
        foresight_config="projects/configs/sparsedrive_r50_stage2_navsim_planonly.py",
        foresight_pretrained="",
    )
    print(f"Building agent (this loads the SparseDrive model)...")
    agent = SparseDriveAgent(cfg, lr=1e-4, checkpoint_path=None)
    if torch.cuda.is_available():
        agent = agent.cuda()
    agent.eval()

    # Only download_navmini's first 2 camera + lidar splits exist on disk.
    # The metadata covers more scenes; restrict to logs whose blobs we have.
    available_logs = [d.name for d in sensor_blobs_path.iterdir() if d.is_dir()]
    print(f"Loading navmini scenes for {len(available_logs)} logs with blobs on disk...")
    scene_filter = SceneFilter(
        num_history_frames=cfg.num_history_frames,
        num_future_frames=cfg.num_future_poses,
        log_names=available_logs,
        max_scenes=1,
    )
    scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    print(f"  {len(scene_loader)} scenes available; loading the first one")
    if len(scene_loader) == 0:
        print("ERROR: no scenes match the on-disk blobs", file=sys.stderr)
        sys.exit(1)
    token = scene_loader.tokens[0]
    scene = scene_loader.get_scene_from_token(token)
    print(f"  scene token: {token}")
    print(f"  log: {scene.scene_metadata.log_name}  map: {scene.scene_metadata.map_name}")

    print(f"Running feature + target builders...")
    feature_builder = agent.get_feature_builders()[0]
    target_builder = agent.get_target_builders()[0]
    features = feature_builder.compute_features(scene.get_agent_input())
    targets = target_builder.compute_targets(scene)
    print(f"  feature keys: {list(features.keys())}")
    print(f"  target keys:  {list(targets.keys())}")
    print(f"  gt_bboxes_3d: {tuple(targets['gt_bboxes_3d'].shape)}  "
          f"gt_labels_3d: {tuple(targets['gt_labels_3d'].shape)}")

    print(f"Running forward pass (eval mode)...")
    # Add a batch dim and move to device
    device = next(agent.parameters()).device
    batched = {k: v.unsqueeze(0).to(device) for k, v in features.items()}
    with torch.no_grad():
        prediction = agent.forward(batched, None)
    traj = prediction["trajectory"]
    print(f"  prediction shape: {tuple(traj.shape)}")

    print()
    print("All steps completed without error.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
