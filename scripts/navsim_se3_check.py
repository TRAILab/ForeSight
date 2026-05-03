"""Quick check that the new T_global / global_ego_pose plumbing produces
absolute SE3 matrices (not identity) on a real navmini scene, and that the
agent forward consumes them without error.

Run inside the foresight_navsim docker image with OPENSCENE_DATA_ROOT and
NUPLAN_MAPS_ROOT pointed at the navmini data.
"""
import os
import sys

import numpy as np
import torch

FORESIGHT_ROOT = os.environ.get("FORESIGHT_ROOT", "/workspace/ForeSight")
if FORESIGHT_ROOT not in sys.path:
    sys.path.insert(0, FORESIGHT_ROOT)
sys.path.insert(0, os.path.join(FORESIGHT_ROOT, "navsim"))

from pathlib import Path

from navsim.agents.sparsedrive.sparsedrive_agent import SparseDriveAgent
from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
from navsim.agents.sparsedrive.sparsedrive_features import (
    SparseDriveFeatureBuilder,
)
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader


def main() -> None:
    cfg = SparseDriveConfig(
        foresight_config="projects/configs/sparsedrive_r50_stage2_navsim_planonly.py",
        foresight_pretrained="",
    )

    openscene = Path(os.environ["OPENSCENE_DATA_ROOT"])
    sensor_blobs = openscene / "sensor_blobs/mini"
    available_logs = [d.name for d in sensor_blobs.iterdir() if d.is_dir()]
    loader = SceneLoader(
        sensor_blobs_path=sensor_blobs,
        data_path=openscene / "navsim_logs/mini",
        scene_filter=SceneFilter(
            num_history_frames=4,
            num_future_frames=10,
            log_names=available_logs,
            max_scenes=1,
        ),
        sensor_config=SparseDriveAgent(cfg, lr=1e-4).get_sensor_config(),
    )
    print(f"  {len(loader.tokens)} scenes available")
    token = loader.tokens[0]
    scene = loader.get_scene_from_token(token)
    agent_input = scene.get_agent_input()

    # Verify the new EgoStatus fields are populated.
    poses = [s.global_ego_pose for s in agent_input.ego_statuses]
    timestamps = [s.timestamp for s in agent_input.ego_statuses]
    assert all(p is not None for p in poses), "global_ego_pose missing from AgentInput"
    assert all(t is not None for t in timestamps), "timestamp missing from AgentInput"
    print(f"  global_ego_poses (4 frames): {[tuple(np.round(p, 2)) for p in poses]}")
    print(f"  timestamps (4 frames, μs):   {timestamps}")
    deltas_s = [(timestamps[i+1] - timestamps[i]) / 1e6 for i in range(len(timestamps)-1)]
    print(f"  Δt between frames (s):      {[round(d, 3) for d in deltas_s]}  (expect ~0.5)")

    # Build features and inspect T_global.
    builder = SparseDriveFeatureBuilder(cfg)
    feats = builder.compute_features(agent_input)
    print(f"  feature keys: {sorted(feats.keys())}")
    img = feats["img"].numpy()
    print(f"  img mean per cam: {np.round(img.mean(axis=(1,2,3)), 3)}  "
          f"(should be ~0 after ImageNet norm; was ~0.5 before)")
    print(f"  img std per cam:  {np.round(img.std(axis=(1,2,3)), 3)}  "
          f"(should be ~1 after ImageNet norm; was ~0.3 before)")
    print(f"  feature timestamp: {float(feats['timestamp']):.6f} s  "
          f"(should match Frame.timestamp/1e6)")
    Tg = feats["T_global"].numpy()
    Tg_inv = feats["T_global_inv"].numpy()
    print(f"  T_global shape: {Tg.shape}  T_global_inv shape: {Tg_inv.shape}")
    print(f"  T_global[-1] (current frame, absolute global SE3):")
    print(np.round(Tg[-1], 3))

    eye = np.eye(4)
    if np.allclose(Tg[-1], eye, atol=1e-3):
        print("  WARN: current T_global is identity — global_ego_pose did not plumb through")
    else:
        print("  OK: current T_global is non-identity (absolute global SE3 in use)")

    # Roundtrip
    assert np.allclose(Tg @ Tg_inv, np.broadcast_to(eye, Tg.shape), atol=1e-3)
    print("  OK: T_global @ T_global_inv == I (per-frame inverse round-trip)")

    # Push through the agent forward to make sure nothing breaks.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        agent = SparseDriveAgent(cfg, lr=1e-4, checkpoint_path=None).to(device)
        agent.eval()
        # Add batch dim and move to device
        batched = {k: v.unsqueeze(0).to(device) for k, v in feats.items()}
        with torch.no_grad():
            out = agent.forward(batched)
        print(f"  OK: agent.forward consumed T_global+T_global_inv, "
              f"trajectory shape {tuple(out['trajectory'].shape)}")
    else:
        print("  SKIP: no CUDA, skipping forward pass")


if __name__ == "__main__":
    main()
