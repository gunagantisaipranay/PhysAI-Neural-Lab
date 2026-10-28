"""Forward-kinematics checks for the built-in MuJoCo arm."""

import numpy as np
import pytest


pytest.importorskip("mujoco")
import mujoco

from physics.robot_scene import build_robot_scene


@pytest.mark.parametrize(
    ("joint_positions", "expected_tool_position"),
    [
        ([0.0, 0.0, 0.0], [1.0, 0.0, 0.34]),
        ([np.pi / 2, 0.0, 0.0], [0.0, 1.0, 0.34]),
        ([0.0, np.pi / 2, 0.0], [0.0, 0.0, -0.66]),
    ],
)
def test_builtin_arm_forward_kinematics_matches_known_tool_positions(
    joint_positions, expected_tool_position
):
    model, _ = build_robot_scene()
    data = mujoco.MjData(model)
    data.qpos[:3] = joint_positions
    mujoco.mj_forward(model, data)

    np.testing.assert_allclose(data.site_xpos[0], expected_tool_position, atol=1e-8)
