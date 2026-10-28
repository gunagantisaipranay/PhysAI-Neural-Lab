"""Generated-data training and held-out evaluation for robot IK models."""

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("sklearn")
import mujoco
from machine_learning.robot_ik_policy import predict_joint_target, train_ik_policy
from physics.robot_scene import Obstacle, build_robot_scene
from physics.robot_tasks import (
    evaluate_ik_rollouts,
    run_joint_target_episode,
    run_joint_trajectory_episode,
)

URDF_WITH_ORIGIN_ONLY_TOOL = b'''<robot name="origin_only">
  <link name="base"/>
  <link name="arm"><inertial><mass value="0.5"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
  <joint name="shoulder" type="revolute"><parent link="base"/><child link="arm"/><origin xyz="0 0 0.2"/><axis xyz="0 1 0"/><limit lower="-1.5" upper="1.5" effort="2" velocity="2"/></joint>
</robot>'''


def test_default_arm_policy_trains_and_evaluates_held_out_targets():
    model, _ = build_robot_scene()
    result = train_ik_policy(model, "end_effector", [0, 1, 2], samples=500, epochs=100, seed=23)

    assert result.train_samples == 400
    assert result.test_samples == 100
    assert 0.0 <= result.success_rate <= 1.0
    assert result.position_rmse_m < 0.15
    assert result.joint_rmse < 0.5
    assert result.loss_curve
    prediction = predict_joint_target(
        result,
        np.asarray(result.example_start),
        np.asarray(result.example_target),
    )
    assert prediction.shape == (3,)
    assert all(low <= value <= high for low, value, high in zip(result.joint_lows, prediction, result.joint_highs))
    rollouts = evaluate_ik_rollouts(model, result, "end_effector", [0, 1, 2])
    learned = [row for row in rollouts if row["controller"] == "learned IK"]
    expert = [row for row in rollouts if row["controller"] == "expert target"]
    assert len(rollouts) == 16
    assert sum(row["success"] for row in expert) >= 6
    assert sum(row["success"] for row in learned) >= 3


def test_ik_training_requires_an_existing_end_effector_site():
    model, _ = build_robot_scene()
    with pytest.raises(ValueError, match="does not exist"):
        train_ik_policy(model, "missing_site", [0, 1, 2], samples=200, epochs=2)


def test_urdf_without_site_gets_trainable_offset_tool_point():
    model, _ = build_robot_scene(
        model_filename="origin.urdf", model_content=URDF_WITH_ORIGIN_ONLY_TOOL
    )
    result = train_ik_policy(model, "physai_end_effector_arm", [0], samples=500, epochs=300)
    assert result.position_rmse_m < 0.08


def test_bounded_pd_episode_reaches_an_expert_joint_target():
    model, _ = build_robot_scene()
    start = np.asarray([0.35, 0.5, -0.7])
    goal = np.asarray([0.4, 0.45, -0.65])
    data = mujoco.MjData(model)
    data.qpos[:3] = goal
    mujoco.mj_forward(model, data)
    target = data.site_xpos[0].copy()
    episode = run_joint_target_episode(
        model, "end_effector", [0, 1, 2], start.tolist(), target.tolist(), goal.tolist()
    )
    assert episode["success"]
    assert episode["final_error_m"] < episode["initial_error_m"]


@pytest.mark.parametrize(
    ("start", "goal", "message"),
    [
        ([3.01, 0.0, 0.0], [0.0, 0.0, 0.0], "Start joint configuration"),
        ([0.0, 0.0, 0.0], [0.0, -2.81, 0.0], "Goal joint configuration"),
    ],
)
def test_joint_target_episode_rejects_configurations_outside_joint_limits(start, goal, message):
    model, _ = build_robot_scene()
    with pytest.raises(ValueError, match=message):
        run_joint_target_episode(
            model, "end_effector", [0, 1, 2], start, [0.8, 0.0, 0.4], goal
        )


def test_joint_target_episode_accepts_joint_limit_boundaries():
    model, _ = build_robot_scene()
    limits = np.asarray(model.jnt_range[:3], dtype=float)
    q = limits[:, 0]
    data = mujoco.MjData(model)
    data.qpos[:3] = q
    mujoco.mj_forward(model, data)

    episode = run_joint_target_episode(
        model, "end_effector", [0, 1, 2], q.tolist(), data.site_xpos[0].tolist(), q.tolist(),
        max_control_steps=1,
    )

    assert episode["control_steps"] >= 0


def test_joint_trajectory_tracks_multiple_waypoints():
    model, _ = build_robot_scene()
    start = [0.2, 0.25, -0.35]
    waypoints = [[0.25, 0.3, -0.4], [0.35, 0.4, -0.5]]

    episode = run_joint_trajectory_episode(model, "end_effector", [0, 1, 2], start, waypoints)

    assert episode["success"]
    assert not episode["obstacle_collision"]
    assert episode["waypoints_completed"] == 2
    assert episode["tracking_rmse_m"] < 0.08
    assert episode["final_joint_error"] < 0.1


def test_joint_trajectory_rejects_invalid_waypoint_configuration():
    model, _ = build_robot_scene()
    with pytest.raises(ValueError, match="Goal joint configuration"):
        run_joint_trajectory_episode(
            model, "end_effector", [0, 1, 2], [0.0, 0.0, 0.0], [[0.0, 2.81, 0.0]]
        )


def test_joint_trajectory_reports_collision_encountered_during_motion():
    start = np.asarray([0.0, 0.0, 0.0])
    goal = np.asarray([0.8, 0.4, -0.6])
    model, _ = build_robot_scene()
    data = mujoco.MjData(model)
    data.qpos[:3] = goal
    mujoco.mj_forward(model, data)
    obstacle = Obstacle("sphere", tuple(data.site_xpos[0]), (0.08, 0.08, 0.08))
    model, _ = build_robot_scene(obstacles=(obstacle,))

    episode = run_joint_trajectory_episode(
        model, "end_effector", [0, 1, 2], start.tolist(), [goal.tolist()]
    )

    assert episode["control_steps"] > 0
    assert episode["obstacle_collision"]
    assert not episode["success"]
    assert episode["waypoints_completed"] == 0


def test_obstacle_collision_is_reported_as_episode_failure():
    start = np.asarray([0.35, 0.5, -0.7])
    model, _ = build_robot_scene()
    data = mujoco.MjData(model)
    data.qpos[:3] = start
    mujoco.mj_forward(model, data)
    point = data.site_xpos[0].copy()
    obstacle = Obstacle("sphere", tuple(point), (0.12, 0.12, 0.12))
    model, _ = build_robot_scene(obstacles=(obstacle,))
    episode = run_joint_target_episode(
        model, "end_effector", [0, 1, 2], start.tolist(), point.tolist(), start.tolist()
    )
    assert episode["obstacle_collision"]
    assert not episode["success"]
