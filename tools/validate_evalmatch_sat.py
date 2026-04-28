"""Validation harness for the evalmatch label-gen geometry.

Runs three checks:
  1. PyTorch SAT vs Shapely polygon-polygon intersection on random box pairs.
  2. PyTorch SAT vs the eval's own ``check_collision`` (which uses Shapely)
     on synthetic ego-vs-agent box configurations.
  3. ``_eval_get_yaw`` parity with ``planning_eval.get_yaw`` on random trajs.

Pass criteria:
  - SAT vs Shapely: zero disagreements over 1000 random pairs.
  - SAT vs eval check_collision: zero disagreements over 200 synthetic pairs.
  - get_yaw parity: max element-wise abs delta < 1e-5.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import torch
from shapely.geometry import Polygon

from projects.mmdet3d_plugin.models.motion.motion_planning_head import (
    _eval_get_yaw,
    _make_rect_corners_topdown,
    _rect_intersects_sat,
)
from projects.mmdet3d_plugin.datasets.evaluation.planning.planning_eval import (
    check_collision as eval_check_collision,
    get_yaw as eval_get_yaw,
)


def _shapely_intersects(corners_a, corners_b):
    """corners_*: (4, 2) numpy. Returns bool."""
    pa = Polygon([(p[0], p[1]) for p in corners_a])
    pb = Polygon([(p[0], p[1]) for p in corners_b])
    return pa.intersects(pb)


def test_sat_vs_shapely(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    disagreements = 0
    for _ in range(n):
        # Two random boxes in a 20x20 area, dims 0.5-5m, yaw [-pi, pi].
        cx_a, cy_a = rng.uniform(-10, 10, size=2)
        W_a, L_a = rng.uniform(0.5, 5.0, size=2)
        yaw_a = rng.uniform(-np.pi, np.pi)
        cx_b, cy_b = rng.uniform(-10, 10, size=2)
        W_b, L_b = rng.uniform(0.5, 5.0, size=2)
        yaw_b = rng.uniform(-np.pi, np.pi)

        cx_a_t = torch.tensor([cx_a]).float()
        cy_a_t = torch.tensor([cy_a]).float()
        W_a_t = torch.tensor([W_a]).float()
        L_a_t = torch.tensor([L_a]).float()
        yaw_a_t = torch.tensor([yaw_a]).float()
        ca = _make_rect_corners_topdown(cx_a_t, cy_a_t, W_a_t, L_a_t, yaw_a_t)

        cx_b_t = torch.tensor([cx_b]).float()
        cy_b_t = torch.tensor([cy_b]).float()
        W_b_t = torch.tensor([W_b]).float()
        L_b_t = torch.tensor([L_b]).float()
        yaw_b_t = torch.tensor([yaw_b]).float()
        cb = _make_rect_corners_topdown(cx_b_t, cy_b_t, W_b_t, L_b_t, yaw_b_t)

        sat = _rect_intersects_sat(ca, cb).item()
        sh = _shapely_intersects(ca[0].numpy(), cb[0].numpy())
        if bool(sat) != bool(sh):
            disagreements += 1
    print(f"[SAT vs Shapely] disagreements: {disagreements}/{n}")
    return disagreements == 0


def test_sat_vs_eval_check_collision(n=200, seed=1):
    """Check the SAT impl against planning_eval.check_collision, which is the
    function we are ultimately replicating in the label."""
    rng = np.random.default_rng(seed)
    from projects.mmdet3d_plugin.core.box3d import X, Y, Z, W, L, H, YAW
    disagreements = 0
    for _ in range(n):
        # Build an "ego" box with eval dims and an "agent" box with random dims.
        ego_xy = rng.uniform(-3, 3, size=2)
        ego_yaw = rng.uniform(-np.pi, np.pi)
        agent_xy = rng.uniform(-5, 5, size=2)
        agent_W, agent_L = rng.uniform(1.0, 5.0, size=2)
        agent_yaw = rng.uniform(-np.pi, np.pi)

        # 7-dim box format [x, y, z, W, L, H, yaw].
        ego_box = torch.tensor(
            [ego_xy[0], ego_xy[1], 0.0, 4.084, 1.85, 1.56, ego_yaw],
            dtype=torch.float32,
        )
        agent_box = torch.tensor(
            [[agent_xy[0], agent_xy[1], 0.0, agent_W, agent_L, 1.7, agent_yaw]],
            dtype=torch.float32,
        )

        # Eval's check_collision adds 0.5m forward offset INSIDE the function
        # via mutation. Match that on the SAT side externally.
        eval_box = ego_box.clone()
        eval_agent = agent_box.clone()
        # Run the eval function (it mutates ego_box internally).
        eval_result = eval_check_collision(eval_box, eval_agent)

        # Build the SAT corners with the same offset applied.
        ego_yaw_t = torch.tensor([ego_yaw], dtype=torch.float32)
        ego_cx = torch.tensor([ego_xy[0] + 0.5 * np.cos(ego_yaw)], dtype=torch.float32)
        ego_cy = torch.tensor([ego_xy[1] + 0.5 * np.sin(ego_yaw)], dtype=torch.float32)
        ego_W_t = torch.tensor([4.084], dtype=torch.float32)
        ego_L_t = torch.tensor([1.85], dtype=torch.float32)
        ego_corners = _make_rect_corners_topdown(
            ego_cx, ego_cy, ego_W_t, ego_L_t, ego_yaw_t
        )
        ag_cx = torch.tensor([agent_xy[0]], dtype=torch.float32)
        ag_cy = torch.tensor([agent_xy[1]], dtype=torch.float32)
        ag_W_t = torch.tensor([agent_W], dtype=torch.float32)
        ag_L_t = torch.tensor([agent_L], dtype=torch.float32)
        ag_yaw_t = torch.tensor([agent_yaw], dtype=torch.float32)
        agent_corners = _make_rect_corners_topdown(
            ag_cx, ag_cy, ag_W_t, ag_L_t, ag_yaw_t
        )

        sat_result = _rect_intersects_sat(ego_corners, agent_corners).item()
        if bool(sat_result) != bool(eval_result):
            disagreements += 1
    print(f"[SAT vs eval check_collision] disagreements: {disagreements}/{n}")
    return disagreements == 0


def test_get_yaw_parity(n=100, T=6, seed=2):
    rng = np.random.default_rng(seed)
    max_delta = 0.0
    for _ in range(n):
        # Random absolute traj (post-cumsum).
        traj = rng.standard_normal(size=(T, 2)).cumsum(axis=0).astype(np.float32)
        traj_t = torch.from_numpy(traj)

        eval_yaw = eval_get_yaw(traj_t).numpy()
        sat_yaw = _eval_get_yaw(traj_t).numpy()

        # Eval's get_yaw returns (T,); ours also returns (T,) for unbatched input.
        delta = np.abs(eval_yaw - sat_yaw).max()
        max_delta = max(max_delta, float(delta))
    print(f"[get_yaw parity] max abs delta: {max_delta:.2e}")
    return max_delta < 1e-5


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    ok1 = test_sat_vs_shapely()
    ok2 = test_sat_vs_eval_check_collision()
    ok3 = test_get_yaw_parity()
    if not (ok1 and ok2 and ok3):
        print("\nFAILED — fix before running the heavier validation steps.")
        sys.exit(1)
    print("\nAll geometry checks passed.")
