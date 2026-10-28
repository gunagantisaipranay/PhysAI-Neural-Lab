"""Tests for bounded target-reaching planning and control."""

import numpy as np
import pytest

pytest.importorskip("mujoco")

from physics.robot_reaching import (
    DEFAULT_FINAL_APPROACH_CLEARANCE_M,
    final_approach_target,
    plan_reach_target,
    pregrasp_target,
)
from physics.robot_simulation import MAX_STEP_COUNT, ManipulationSimulation


def test_pregrasp_target_uses_live_payload_position_and_is_finite():
    simulation = ManipulationSimulation.create()
    target = pregrasp_target(simulation.model, simulation.data, simulation.config)

    assert target.shape == (3,)
    assert np.isfinite(target).all()
    assert not np.array_equal(target, simulation.payload_position())


def test_reach_planner_returns_limited_joint_goal_for_reachable_target():
    simulation = ManipulationSimulation.create()
    target = pregrasp_target(simulation.model, simulation.data, simulation.config)

    plan = plan_reach_target(simulation.model, simulation.data, simulation.config, target)

    assert plan.goal_joint_positions.shape == (3,)
    assert plan.final_position.shape == (3,)
    assert plan.final_error_m <= 0.035
    for name, value in zip(simulation.config.arm_joints, plan.goal_joint_positions):
        import mujoco

        joint_id = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        low, high = simulation.model.jnt_range[joint_id]
        assert low <= value <= high


def test_invalid_and_unreachable_targets_are_rejected():
    simulation = ManipulationSimulation.create()
    with pytest.raises(ValueError, match="finite"):
        plan_reach_target(simulation.model, simulation.data, simulation.config, [np.nan, 0.0, 0.0])
    with pytest.raises(ValueError, match="not reachable"):
        plan_reach_target(simulation.model, simulation.data, simulation.config, [3.0, 3.0, 3.0])


def test_final_approach_clearance_must_be_positive_and_finite():
    simulation = ManipulationSimulation.create()
    with pytest.raises(ValueError, match="Final-approach clearance"):
        final_approach_target(simulation.model, simulation.data, simulation.config, clearance_m=0.0)


def test_pregrasp_target_keeps_safe_clearance_from_payload():
    simulation = ManipulationSimulation.create()
    target = pregrasp_target(simulation.model, simulation.data, simulation.config)

    assert target.shape == (3,)
    assert np.isfinite(target).all()
    distance = float(np.linalg.norm(target - simulation.payload_position()))
    payload_half_diagonal = float(np.sqrt(3.0) * 0.055)
    assert distance > payload_half_diagonal


def test_final_approach_target_moves_inside_grip_envelope():
    simulation = ManipulationSimulation.create()
    pregrasp = pregrasp_target(simulation.model, simulation.data, simulation.config)
    final = final_approach_target(simulation.model, simulation.data, simulation.config)

    assert final.shape == (3,)
    assert np.isfinite(final).all()
    pregrasp_distance = float(np.linalg.norm(pregrasp - simulation.payload_position()))
    final_distance = float(np.linalg.norm(final - simulation.payload_position()))
    assert final_distance < pregrasp_distance
    assert final_distance > DEFAULT_FINAL_APPROACH_CLEARANCE_M


def _final_approach_case(simulation: ManipulationSimulation, payload_position, approach_direction):
    import mujoco

    payload_body_id = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_BODY, "payload")
    tool_site_id = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_SITE, "tool_center")
    payload_geom_id = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_GEOM, "payload_geom")
    half_extents = np.asarray(simulation.model.geom_size[payload_geom_id], dtype=float)[:3]
    payload_pos = np.asarray(payload_position, dtype=float)
    direction = np.asarray(approach_direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    simulation.data.xpos[payload_body_id] = payload_pos
    simulation.data.site_xpos[tool_site_id] = payload_pos + direction * 0.2
    final = final_approach_target(simulation.model, simulation.data, simulation.config)
    ray_entries = [
        float(half_extents[axis]) / abs(float(direction[axis]))
        for axis in range(3)
        if abs(float(direction[axis])) > 1e-9
    ]
    return final, payload_pos, min(ray_entries)


def test_final_approach_target_is_outside_payload_for_axis_aligned_approach():
    simulation = ManipulationSimulation.create()
    final, payload_pos, entry_distance = _final_approach_case(
        simulation, (0.65, 0.42, 0.08), (1.0, 0.0, 0.0)
    )
    final_distance = float(np.linalg.norm(final - payload_pos))
    assert final_distance > entry_distance
    assert np.isclose(final_distance, entry_distance + DEFAULT_FINAL_APPROACH_CLEARANCE_M)


def test_final_approach_target_is_outside_payload_for_diagonal_approach():
    simulation = ManipulationSimulation.create()
    final, payload_pos, entry_distance = _final_approach_case(
        simulation, (0.65, 0.42, 0.08), (0.57735, 0.57735, 0.57735)
    )
    final_distance = float(np.linalg.norm(final - payload_pos))
    assert final_distance > entry_distance
    assert np.isclose(final_distance, entry_distance + DEFAULT_FINAL_APPROACH_CLEARANCE_M)


def test_reach_controller_steps_incrementally_and_keeps_gripper_open():
    simulation = ManipulationSimulation.create()
    simulation.start_reach_target()
    assert simulation.running
    assert simulation.state().controller_status == "reaching"

    simulation.step(5)
    state = simulation.state()

    assert 0.0 < simulation.simulation_time <= simulation.model.opt.timestep * 5
    assert state.controller_status in {"reaching", "reached", "timeout"}
    assert state.planned_arm_joint_goal is not None
    assert state.planner_final_error_m is not None
    assert np.isfinite(state.planned_arm_joint_goal).all()
    assert np.isfinite(state.planner_final_position).all()
    assert np.isfinite(state.planner_final_error_m)
    assert np.isfinite(state.raw_arm_controls).all()
    assert np.isfinite(state.clipped_arm_controls).all()
    assert state.arm_control_saturated.shape == (3,)
    for name in simulation.config.finger_actuators:
        import mujoco

        actuator_id = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert simulation.data.ctrl[actuator_id] == 0.0


def test_timeout_retains_finite_reach_diagnostics():
    simulation = ManipulationSimulation.create()
    simulation.start_reach_target()
    simulation._controller_steps = 999

    simulation.step(1)
    state = simulation.state()

    assert state.controller_status == "timeout"
    assert state.paused
    assert state.planned_arm_joint_goal is not None
    assert state.planner_final_error_m is not None
    assert np.isfinite(state.raw_arm_controls).all()
    assert np.isfinite(state.clipped_arm_controls).all()
    assert state.arm_control_saturated.dtype == bool


def test_reach_execution_respects_per_call_step_bound():
    simulation = ManipulationSimulation.create()
    simulation.start_reach_target()
    with pytest.raises(ValueError):
        simulation.step(MAX_STEP_COUNT + 1)
