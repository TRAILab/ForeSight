"""Phase 4 smoke test: verify the navsim+sparsedrive integration imports and
the SparseDriveAgent can be instantiated and produce a feature dict from a
synthetic AgentInput. Designed to run inside the foresight_navsim container.

Usage:
    cd /workspace/ForeSight && python scripts/navsim_smoke_test.py
"""

import sys
import traceback

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@check("import navsim base")
def t1():
    import navsim  # noqa: F401
    from navsim.agents.abstract_agent import AbstractAgent  # noqa: F401
    from navsim.common.dataclasses import (  # noqa: F401
        AgentInput,
        EgoStatus,
        Cameras,
        Camera,
        Lidar,
        SensorConfig,
    )


@check("import sparsedrive agent")
def t2():
    from navsim.agents.sparsedrive.sparsedrive_agent import SparseDriveAgent  # noqa: F401
    from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig  # noqa: F401
    from navsim.agents.sparsedrive.sparsedrive_features import (  # noqa: F401
        SparseDriveFeatureBuilder,
        SparseDriveTargetBuilder,
    )


@check("import foresight plugin (registers mmcv heads)")
def t3():
    sys.path.insert(0, "/workspace/ForeSight")
    import projects.mmdet3d_plugin  # noqa: F401


@check("build sparsedrive head from navsim variant config")
def t4():
    from mmcv import Config
    from mmdet.models import build_detector

    cfg = Config.fromfile(
        "/workspace/ForeSight/projects/configs/sparsedrive_r50_stage2_navsim_planonly.py"
    )
    model = build_detector(cfg.model)
    assert hasattr(model, "head"), "expected SparseDrive model to have .head"
    print(f"   model class: {type(model).__name__}")


@check("feature builder on synthetic AgentInput")
def t5():
    import numpy as np
    from navsim.common.dataclasses import (
        AgentInput,
        EgoStatus,
        Cameras,
        Camera,
    )
    from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
    from navsim.agents.sparsedrive.sparsedrive_features import SparseDriveFeatureBuilder

    def fake_camera():
        return Camera(
            image=np.zeros((1080, 1920, 3), dtype=np.uint8),
            sensor2lidar_rotation=np.eye(3, dtype=np.float32),
            sensor2lidar_translation=np.zeros(3, dtype=np.float32),
            intrinsics=np.array([[1545, 0, 960], [0, 1545, 540], [0, 0, 1]], dtype=np.float32),
            distortion=np.zeros(5, dtype=np.float32),
        )

    cams = Cameras(
        cam_f0=fake_camera(),
        cam_l0=fake_camera(),
        cam_l1=Camera(),  # will be skipped
        cam_l2=fake_camera(),
        cam_r0=fake_camera(),
        cam_r1=Camera(),  # will be skipped
        cam_r2=fake_camera(),
        cam_b0=fake_camera(),
    )
    ego = EgoStatus(
        ego_pose=np.zeros(3, dtype=np.float64),
        ego_velocity=np.array([3.0, 0.0], dtype=np.float32),
        ego_acceleration=np.array([0.0, 0.0], dtype=np.float32),
        driving_command=np.array([0, 1, 0, 0], dtype=np.int_),
    )
    agent_input = AgentInput(
        ego_statuses=[ego, ego, ego, ego],
        cameras=[cams, cams, cams, cams],
        lidars=[],
    )
    builder = SparseDriveFeatureBuilder(SparseDriveConfig())
    feats = builder.compute_features(agent_input)

    expected = {"img", "projection_mat", "image_wh", "ego_status", "gt_ego_fut_cmd"}
    assert set(feats.keys()) >= expected, f"missing keys: {expected - set(feats.keys())}"
    print(f"   img shape: {tuple(feats['img'].shape)}")
    print(f"   projection_mat shape: {tuple(feats['projection_mat'].shape)}")
    print(f"   ego_status shape: {tuple(feats['ego_status'].shape)}")


def main():
    print(f"Python: {sys.version.split()[0]}")
    print()
    failures = 0
    for name, fn in CHECKS:
        sys.stdout.write(f"[ ] {name} ... ")
        sys.stdout.flush()
        try:
            fn()
            print("OK")
        except Exception as e:
            print("FAIL")
            traceback.print_exc()
            failures += 1
    print()
    if failures:
        print(f"{failures}/{len(CHECKS)} checks failed.")
        sys.exit(1)
    print(f"All {len(CHECKS)} checks passed.")


if __name__ == "__main__":
    main()
