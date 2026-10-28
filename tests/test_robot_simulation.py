"""Tests for the bounded manipulation simulation session."""

import numpy as np
import pytest

pytest.importorskip("mujoco")

from physics.robot_reaching import ReachPlan
from physics.robot_simulation import (
    MAX_STEP_COUNT,
    ManipulationSimulation,
    get_manipulation_simulation,
)


def test_session_constructs_with_finite_named_state():
    simulation = ManipulationSimulation.create()
    state = simulation.state()

    assert simulation.simulation_time == 0.0
    assert simulation.arm_joint_positions().shape == (3,)
    assert simulation.finger_joint_positions().shape == (2,)
    assert simulation.payload_position().shape == (3,)
    assert np.isfinite(simulation.arm_joint_positions()).all()
    assert np.isfinite(simulation.payload_position()).all()
    assert simulation.paused
    assert state.planned_arm_joint_goal is None
    assert state.planner_final_error_m is None
    assert np.isfinite(state.arm_control_ranges).all()
    assert np.isfinite(state.raw_arm_controls).all()
    assert np.isfinite(state.clipped_arm_controls).all()
    assert not state.arm_control_saturated.any()
    assert state.base_yaw_diagnostics == ()


def test_step_is_bounded_and_advances_when_running():
    simulation = ManipulationSimulation.create()
    simulation.set_paused(False)
    initial_time = simulation.simulation_time

    state = simulation.step(3)

    assert state.simulation_time > initial_time
    assert state.simulation_time == simulation.model.opt.timestep * 3


def test_reset_restores_deterministic_state_and_pause_status():
    simulation = ManipulationSimulation.create()
    initial = simulation.state()
    simulation.set_paused(False)
    simulation.step(2)

    simulation.reset()
    restored = simulation.state()

    assert restored.paused
    np.testing.assert_array_equal(restored.arm_joint_positions, initial.arm_joint_positions)
    np.testing.assert_array_equal(restored.finger_joint_positions, initial.finger_joint_positions)
    np.testing.assert_array_equal(restored.payload_position, initial.payload_position)
    assert restored.simulation_time == initial.simulation_time
    np.testing.assert_array_equal(restored.raw_arm_controls, np.zeros(3))
    np.testing.assert_array_equal(restored.clipped_arm_controls, np.zeros(3))
    assert not restored.arm_control_saturated.any()
    assert restored.base_yaw_diagnostics == ()


def test_base_yaw_diagnostics_record_pre_step_controller_values():
    import mujoco

    simulation = ManipulationSimulation.create()
    simulation.start_reach_target()

    state = simulation.step(1)
    assert len(state.base_yaw_diagnostics) == 1
    diagnostic = state.base_yaw_diagnostics[0]
    base_yaw_motor_id = mujoco.mj_name2id(
        simulation.model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "base_yaw_motor",
    )

    assert diagnostic.actuator_id == base_yaw_motor_id
    assert diagnostic.simulation_time == 0.0
    assert diagnostic.target_position_rad is not None
    assert diagnostic.reach_error_m is not None
    assert np.isfinite(
        [
            diagnostic.simulation_time,
            diagnostic.position_rad,
            diagnostic.velocity_rad_s,
            diagnostic.target_position_rad,
            diagnostic.bias_force_nm,
            diagnostic.actuator_moment,
            diagnostic.raw_control,
            diagnostic.clipped_control,
            diagnostic.reach_error_m,
        ]
    ).all()
    assert diagnostic.saturated == (
        diagnostic.raw_control < simulation.model.actuator_ctrlrange[base_yaw_motor_id, 0]
        or diagnostic.raw_control > simulation.model.actuator_ctrlrange[base_yaw_motor_id, 1]
    )


def test_base_yaw_diagnostic_history_is_bounded():
    simulation = ManipulationSimulation.create()
    simulation.start_reach_target()

    simulation.step(MAX_STEP_COUNT)

    diagnostics = simulation.state().base_yaw_diagnostics
    assert 0 < len(diagnostics) <= MAX_STEP_COUNT


@pytest.mark.parametrize("steps", [0, -1, MAX_STEP_COUNT + 1, True, 1.5])
def test_invalid_step_counts_are_rejected(steps):
    simulation = ManipulationSimulation.create()

    with pytest.raises((TypeError, ValueError)):
        simulation.step(steps)


def test_paused_step_does_not_advance_simulation():
    simulation = ManipulationSimulation.create()

    with pytest.raises(RuntimeError, match="paused"):
        simulation.step(5)
    assert simulation.simulation_time == 0.0
    assert simulation.paused


def test_session_state_reuses_valid_session_and_replaces_stale_state():
    session_state = {}
    first = get_manipulation_simulation(session_state)
    first.set_paused(False)
    second = get_manipulation_simulation(session_state)
    assert second is first
    assert second.running

    session_state["manipulation_simulation"] = object()
    replacement = get_manipulation_simulation(session_state)
    assert replacement is not first
    assert replacement.paused


def test_decorative_geoms_remain_collision_free():
    """Stage 1 visual-only decoration must stay out of collision geometry."""
    import mujoco

    simulation = ManipulationSimulation.create()
    model = simulation.model
    decorative_names = (
        "base_foot",
        "shoulder_cap",
        "wrist_flange",
        "left_finger_cap",
        "right_finger_cap",
    )
    for name in decorative_names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert geom_id >= 0, f"Manipulation scene is missing decorative geom '{name}'."
        assert int(model.geom_contype[geom_id]) == 0
        assert int(model.geom_conaffinity[geom_id]) == 0
    marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "destination_marker")
    assert marker_id >= 0, "Manipulation scene is missing the destination marker geom."
    assert int(model.geom_contype[marker_id]) == 0
    left_pad_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad")
    assert left_pad_id >= 0, "Manipulation scene is missing the left finger pad geom."
    assert int(model.geom_contype[left_pad_id]) == 1
    right_pad_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_pad")
    assert right_pad_id >= 0, "Manipulation scene is missing the right finger pad geom."
    assert int(model.geom_contype[right_pad_id]) == 1
    payload_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "payload_geom")
    assert payload_id >= 0, "Manipulation scene is missing the payload geom."
    assert int(model.geom_contype[payload_id]) == 1


def _at_target_reach_plan(simulation: ManipulationSimulation) -> ReachPlan:
    position = simulation.end_effector_position()
    return ReachPlan(
        target_position=position.copy(),
        goal_joint_positions=simulation.arm_joint_positions().copy(),
        final_position=position.copy(),
        final_error_m=0.0,
    )


def test_reach_flow_transitions_from_pregrasp_to_final_approach():
    simulation = ManipulationSimulation.create()
    simulation.start_reach_target()
    simulation._reach_plan = _at_target_reach_plan(simulation)

    state = simulation.step(1)

    assert state.phase == "final_approach"
    assert state.controller_status == "reaching"
    assert state.planned_arm_joint_goal is not None
    assert state.target_position is not None


def test_final_approach_completion_stops_at_final_approach_goal():
    simulation = ManipulationSimulation.create()
    simulation.start_reach_target()
    simulation._reach_plan = _at_target_reach_plan(simulation)
    simulation.step(1)
    assert simulation.state().phase == "final_approach"

    simulation._reach_plan = _at_target_reach_plan(simulation)
    final_state = simulation.step(1)

    assert final_state.phase == "final_approach"
    assert final_state.controller_status == "reached"
    assert final_state.paused
    assert final_state.reach_error_m is not None
    assert final_state.reach_error_m <= 0.035


def test_final_approach_failure_after_pregrasp_reports_status(monkeypatch):
    simulation = ManipulationSimulation.create()
    simulation.start_reach_target()
    simulation._reach_plan = _at_target_reach_plan(simulation)
    monkeypatch.setattr(
        "physics.robot_simulation.final_approach_target",
        lambda model, data, config, clearance_m=0.01: np.asarray([5.0, 5.0, 5.0]),
    )

    state = simulation.step(1)

    assert state.controller_status == "final_approach_failed"
    assert state.paused


def test_gripper_closure_respects_actuator_limits_and_progresses():
    import mujoco

    simulation = ManipulationSimulation.create()
    simulation.start_reach_target()
    simulation._reach_plan = _at_target_reach_plan(simulation)
    simulation.step(1)
    simulation._reach_plan = _at_target_reach_plan(simulation)
    simulation.step(1)
    simulation.start_closing()

    state = simulation.step(1)

    assert simulation._controller_steps >= 1
    for name in simulation.config.finger_actuators:
        actuator_id = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        low, high = simulation.model.actuator_ctrlrange[actuator_id]
        assert simulation.data.ctrl[actuator_id] >= low
        assert simulation.data.ctrl[actuator_id] <= high
    assert state.controller_status in {"closing", "grasped", "grasp_failed"}


def test_grasp_failure_after_bounded_closing_budget():
    simulation = ManipulationSimulation.create()
    simulation.start_reach_target()
    simulation._reach_plan = _at_target_reach_plan(simulation)
    simulation.step(1)
    simulation._reach_plan = _at_target_reach_plan(simulation)
    simulation.step(1)
    simulation.start_closing()
    simulation._controller_steps = MAX_STEP_COUNT - 1

    state = simulation.step(1)

    assert state.controller_status == "grasp_failed"
    assert state.paused


def test_grasp_success_requires_both_finger_pads_contact(monkeypatch):
    import mujoco

    simulation = ManipulationSimulation.create()
    simulation.start_reach_target()
    simulation._reach_plan = _at_target_reach_plan(simulation)
    simulation.step(1)
    simulation._reach_plan = _at_target_reach_plan(simulation)
    simulation.step(1)
    simulation.start_closing()
    left_pad_id = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad")
    right_pad_id = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_pad")
    monkeypatch.setattr(simulation, "_finger_payload_contact_pads", lambda: {left_pad_id, right_pad_id})

    state = simulation.step(1)

    assert state.controller_status == "grasped"
    assert state.phase == "grasped"
    assert state.paused


def test_grasp_failure_when_single_pad_contact_is_insufficient(monkeypatch):
    import mujoco

    simulation = ManipulationSimulation.create()
    simulation.start_reach_target()
    simulation._reach_plan = _at_target_reach_plan(simulation)
    simulation.step(1)
    simulation._reach_plan = _at_target_reach_plan(simulation)
    simulation.step(1)
    simulation.start_closing()
    left_pad_id = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad")
    monkeypatch.setattr(simulation, "_finger_payload_contact_pads", lambda: {left_pad_id})

    state = simulation.step(1)

    assert state.controller_status == "closing"
    assert not state.paused


def test_reset_clears_closing_state():
    simulation = ManipulationSimulation.create()
    simulation.start_reach_target()
    simulation._reach_plan = _at_target_reach_plan(simulation)
    simulation.step(1)
    simulation._reach_plan = _at_target_reach_plan(simulation)
    simulation.step(1)
    simulation.start_closing()
    simulation.reset()

    state = simulation.state()

    assert state.phase == "idle"
    assert state.controller_status == "idle"
    assert state.planned_arm_joint_goal is None
