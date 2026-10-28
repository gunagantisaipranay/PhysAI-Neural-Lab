"""Bounded simulation state for the standalone manipulation scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import MutableMapping

import mujoco
import numpy as np

from physics.robot_manipulation import (
    ManipulationSceneConfig,
    build_manipulation_scene,
)
from physics.robot_reaching import (
    DEFAULT_REACH_TOLERANCE_M,
    ReachPlan,
    actuator_for_joint,
    final_approach_target,
    plan_reach_target,
    pregrasp_target,
)


MAX_STEP_COUNT = 1000


@dataclass(frozen=True)
class BaseYawDiagnostic:
    """Pre-step base-yaw controller telemetry for one bounded simulation step."""

    simulation_time: float
    actuator_id: int
    position_rad: float
    velocity_rad_s: float
    target_position_rad: float | None
    bias_force_nm: float
    actuator_moment: float
    raw_control: float
    clipped_control: float
    saturated: bool
    reach_error_m: float | None


@dataclass(frozen=True)
class ManipulationState:
    """Finite state snapshot suitable for display or test assertions."""

    simulation_time: float
    arm_joint_positions: np.ndarray
    finger_joint_positions: np.ndarray
    payload_position: np.ndarray
    end_effector_position: np.ndarray
    target_position: np.ndarray | None
    reach_error_m: float | None
    controller_status: str
    planned_arm_joint_goal: np.ndarray | None
    planner_final_position: np.ndarray | None
    planner_final_error_m: float | None
    arm_actuator_ids: tuple[int, ...]
    arm_control_ranges: np.ndarray
    raw_arm_controls: np.ndarray
    clipped_arm_controls: np.ndarray
    arm_control_saturated: np.ndarray
    base_yaw_diagnostics: tuple[BaseYawDiagnostic, ...]
    paused: bool
    phase: str = "idle"


class ManipulationSimulation:
    """Own a manipulation model/data pair and advance it in bounded batches."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: ManipulationSceneConfig,
    ) -> None:
        self.model = model
        self.data = data
        self.config = config
        self._arm_qpos_addresses = self._joint_qpos_addresses(config.arm_joints)
        self._finger_qpos_addresses = self._joint_qpos_addresses(config.finger_joints)
        self._payload_body_id = self._body_id(config.payload_body)
        self._tool_site_id = self._site_id(config.tool_center_site)
        self._arm_joint_ids = self._joint_ids(config.arm_joints)
        self._arm_dofs = np.asarray(self.model.jnt_dofadr[self._arm_joint_ids], dtype=int)
        self._arm_actuators = [
            actuator_for_joint(self.model, self.data, int(joint_id))
            for joint_id in self._arm_joint_ids
        ]
        self._arm_actuator_ids = tuple(int(actuator_id) for actuator_id, _ in self._arm_actuators)
        self._arm_control_ranges = np.array(
            self.model.actuator_ctrlrange[list(self._arm_actuator_ids)],
            dtype=float,
            copy=True,
        )
        self._raw_arm_controls = np.zeros(len(self._arm_actuators), dtype=float)
        self._clipped_arm_controls = np.zeros(len(self._arm_actuators), dtype=float)
        self._arm_control_saturated = np.zeros(len(self._arm_actuators), dtype=bool)
        self._finger_joint_ids = self._joint_ids(config.finger_joints)
        self._finger_dofs = np.asarray(self.model.jnt_dofadr[self._finger_joint_ids], dtype=int)
        self._finger_actuators = [
            actuator_for_joint(self.model, self.data, int(joint_id))
            for joint_id in self._finger_joint_ids
        ]
        self._finger_closed_goals = np.zeros(len(self._finger_joint_ids), dtype=float)
        self._gripper_raw_controls = np.zeros(len(self._finger_actuators), dtype=float)
        self._gripper_clipped_controls = np.zeros(len(self._finger_actuators), dtype=float)
        self._gripper_control_saturated = np.zeros(len(self._finger_actuators), dtype=bool)
        self._base_yaw_diagnostics: list[BaseYawDiagnostic] = []
        self._phase = "idle"
        self._initial_qpos = np.array(data.qpos, dtype=float, copy=True)
        self._initial_qvel = np.array(data.qvel, dtype=float, copy=True)
        self._initial_time = float(data.time)
        self._paused = True
        self._reach_plan: ReachPlan | None = None
        self._controller_status = "idle"
        self._controller_steps = 0
        self._validate_state()

    @classmethod
    def create(cls) -> "ManipulationSimulation":
        """Build a fresh manipulation scene and initialize its simulation state."""
        model, data, config = build_manipulation_scene()
        return cls(model, data, config)

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def running(self) -> bool:
        return not self._paused

    @property
    def simulation_time(self) -> float:
        return float(self.data.time)

    def set_paused(self, paused: bool) -> None:
        """Enable or disable stepping; this does not advance the simulation."""
        if not isinstance(paused, (bool, np.bool_)):
            raise TypeError("paused must be a boolean.")
        self._paused = bool(paused)

    def reset(self) -> None:
        """Restore the exact deterministic initial qpos, qvel, and time."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._initial_qpos
        self.data.qvel[:] = self._initial_qvel
        self.data.time = self._initial_time
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._paused = True
        self._reach_plan = None
        self._controller_status = "idle"
        self._controller_steps = 0
        self._base_yaw_diagnostics.clear()
        self._reset_control_diagnostics()
        self._finger_closed_goals = np.zeros(len(self._finger_joint_ids), dtype=float)
        self._gripper_raw_controls.fill(0.0)
        self._gripper_clipped_controls.fill(0.0)
        self._gripper_control_saturated.fill(False)
        self._phase = "idle"

    def step(self, n_steps: int = 1) -> ManipulationState:
        """Advance by at most ``MAX_STEP_COUNT`` actual MuJoCo simulation steps."""
        if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)):
            raise TypeError("n_steps must be an integer.")
        if n_steps < 1 or n_steps > MAX_STEP_COUNT:
            raise ValueError(f"n_steps must be between 1 and {MAX_STEP_COUNT}.")
        if self._paused:
            raise RuntimeError("Simulation is paused; resume stepping before advancing it.")
        for _ in range(int(n_steps)):
            if self._phase == "closing" and self._controller_status == "closing":
                self._apply_gripper_control()
            else:
                self._apply_reach_control()
            mujoco.mj_step(self.model, self.data)
            if self._phase == "closing" and self._controller_status == "closing":
                self._update_grasp_status()
            else:
                self._update_reach_status()
        return self.state()

    def start_reach_target(self) -> ManipulationState:
        """Plan a pre-grasp target from the live payload and begin bounded reaching.

        The pre-grasp sequence then continues into the final approach
        automatically once the pre-grasp waypoint is reached.
        """
        self._phase = "pregrasp"
        target = pregrasp_target(self.model, self.data, self.config)
        self._reach_plan = plan_reach_target(self.model, self.data, self.config, target)
        self._controller_steps = 0
        self._controller_status = "reaching"
        self._paused = False
        self._set_gripper_open()
        return self.state()

    def start_closing(self) -> ManipulationState:
        """Begin bounded gripper closure after a successful final approach."""
        if self._phase != "final_approach" or self._controller_status != "reached":
            raise ValueError("Final approach must be reached before closing the gripper.")
        self._phase = "closing"
        self._controller_status = "closing"
        self._controller_steps = 0
        self._paused = False
        self._close_gripper()
        return self.state()

    def arm_joint_positions(self) -> np.ndarray:
        """Return current arm joint positions resolved from named joints."""
        return self._copy_positions(self._arm_qpos_addresses)

    def finger_joint_positions(self) -> np.ndarray:
        """Return current finger joint positions resolved from named joints."""
        return self._copy_positions(self._finger_qpos_addresses)

    def payload_position(self) -> np.ndarray:
        """Return the payload body's current world position."""
        return np.array(self.data.xpos[self._payload_body_id], dtype=float, copy=True)

    def end_effector_position(self) -> np.ndarray:
        """Return the current tool-center position in world coordinates."""
        return np.array(self.data.site_xpos[self._tool_site_id], dtype=float, copy=True)

    def state(self) -> ManipulationState:
        """Return a copy-based snapshot of the current simulation state."""
        return ManipulationState(
            simulation_time=self.simulation_time,
            arm_joint_positions=self.arm_joint_positions(),
            finger_joint_positions=self.finger_joint_positions(),
            payload_position=self.payload_position(),
            end_effector_position=self.end_effector_position(),
            target_position=(
                None if self._reach_plan is None
                else self._reach_plan.target_position.copy()
            ),
            reach_error_m=(
                None if self._reach_plan is None
                else float(np.linalg.norm(self.end_effector_position() - self._reach_plan.target_position))
            ),
            controller_status=self._controller_status,
            planned_arm_joint_goal=(
                None if self._reach_plan is None
                else self._reach_plan.goal_joint_positions.copy()
            ),
            planner_final_position=(
                None if self._reach_plan is None
                else self._reach_plan.final_position.copy()
            ),
            planner_final_error_m=(
                None if self._reach_plan is None
                else float(self._reach_plan.final_error_m)
            ),
            arm_actuator_ids=self._arm_actuator_ids,
            arm_control_ranges=self._arm_control_ranges.copy(),
            raw_arm_controls=self._raw_arm_controls.copy(),
            clipped_arm_controls=self._clipped_arm_controls.copy(),
            arm_control_saturated=self._arm_control_saturated.copy(),
            base_yaw_diagnostics=tuple(self._base_yaw_diagnostics),
            paused=self.paused,
            phase=self._phase,
        )

    def _joint_qpos_addresses(self, joint_names: tuple[str, ...]) -> np.ndarray:
        addresses = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Simulation scene is missing joint '{name}'.")
            addresses.append(int(self.model.jnt_qposadr[joint_id]))
        return np.asarray(addresses, dtype=int)

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Simulation scene is missing body '{name}'.")
        return body_id

    def _site_id(self, name: str) -> int:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"Simulation scene is missing site '{name}'.")
        return site_id

    def _joint_ids(self, names: tuple[str, ...]) -> np.ndarray:
        ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
        if any(joint_id < 0 for joint_id in ids):
            raise ValueError("Simulation scene is missing a required arm joint.")
        return np.asarray(ids, dtype=int)

    def _set_gripper_open(self) -> None:
        for name in self.config.finger_actuators:
            actuator_id = self._actuator_id(name)
            self.data.ctrl[actuator_id] = 0.0

    def _actuator_id(self, name: str) -> int:
        actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"Simulation scene is missing actuator '{name}'.")
        return actuator_id

    def _apply_reach_control(self) -> None:
        if self._reach_plan is None or self._controller_status != "reaching":
            return
        q = self.arm_joint_positions()
        qvel = self.data.qvel[self._arm_dofs]
        reach_error_m = float(
            np.linalg.norm(self.end_effector_position() - self._reach_plan.target_position)
        )
        torque = 18.0 * (self._reach_plan.goal_joint_positions - q) - 2.5 * qvel
        torque += self.data.qfrc_bias[self._arm_dofs]
        for index, (actuator_id, moment) in enumerate(self._arm_actuators):
            raw_control = float(torque[index] / moment)
            low, high = self._arm_control_ranges[index]
            clipped_control = (
                float(np.clip(raw_control, low, high))
                if self.model.actuator_ctrllimited[actuator_id]
                else raw_control
            )
            self._raw_arm_controls[index] = raw_control
            self._clipped_arm_controls[index] = clipped_control
            self._arm_control_saturated[index] = bool(
                self.model.actuator_ctrllimited[actuator_id]
                and (raw_control < low or raw_control > high)
            )
            self.data.ctrl[actuator_id] = clipped_control
            if index == 0:
                self._base_yaw_diagnostics.append(
                    BaseYawDiagnostic(
                        simulation_time=self.simulation_time,
                        actuator_id=int(actuator_id),
                        position_rad=float(q[index]),
                        velocity_rad_s=float(qvel[index]),
                        target_position_rad=float(self._reach_plan.goal_joint_positions[index]),
                        bias_force_nm=float(self.data.qfrc_bias[self._arm_dofs[index]]),
                        actuator_moment=float(moment),
                        raw_control=raw_control,
                        clipped_control=clipped_control,
                        saturated=self._arm_control_saturated[index],
                        reach_error_m=reach_error_m,
                    )
                )
                if len(self._base_yaw_diagnostics) > MAX_STEP_COUNT:
                    del self._base_yaw_diagnostics[:-MAX_STEP_COUNT]
        self._set_gripper_open()
        self._controller_steps += 1

    def _update_reach_status(self) -> None:
        if self._reach_plan is None or self._controller_status != "reaching":
            return
        error = float(np.linalg.norm(self.end_effector_position() - self._reach_plan.target_position))
        if error <= DEFAULT_REACH_TOLERANCE_M:
            if self._phase == "pregrasp":
                # Pre-grasp waypoint reached: transition automatically into the
                # final approach without requiring another user-triggered reach.
                self._phase = "final_approach"
                try:
                    final_target = final_approach_target(self.model, self.data, self.config)
                    self._reach_plan = plan_reach_target(self.model, self.data, self.config, final_target)
                except ValueError:
                    self._controller_status = "final_approach_failed"
                    self._paused = True
                    return
                self._controller_steps = 0
            else:
                self._controller_status = "reached"
                self._paused = True
            return
        if self._controller_steps >= MAX_STEP_COUNT:
            self._controller_status = "timeout"
            self._paused = True

    def _apply_gripper_control(self) -> None:
        if self._phase != "closing" or self._controller_status != "closing":
            return
        q = self._copy_positions(self._finger_qpos_addresses)
        qvel = self.data.qvel[self._finger_dofs]
        torque = 18.0 * (self._finger_closed_goals - q) - 2.5 * qvel
        torque += self.data.qfrc_bias[self._finger_dofs]
        for index, (actuator_id, moment) in enumerate(self._finger_actuators):
            raw_control = float(torque[index] / moment)
            low, high = self.model.actuator_ctrlrange[actuator_id]
            clipped_control = (
                float(np.clip(raw_control, low, high))
                if self.model.actuator_ctrllimited[actuator_id]
                else raw_control
            )
            self._gripper_raw_controls[index] = raw_control
            self._gripper_clipped_controls[index] = clipped_control
            self._gripper_control_saturated[index] = bool(
                self.model.actuator_ctrllimited[actuator_id]
                and (raw_control < low or raw_control > high)
            )
            self.data.ctrl[actuator_id] = clipped_control
        self._controller_steps += 1

    def _close_gripper(self) -> None:
        """Command both finger slides toward the closed configuration."""
        self._finger_closed_goals = np.zeros(len(self._finger_joint_ids), dtype=float)
        self._apply_gripper_control()

    def _finger_payload_contact_pads(self) -> set[int]:
        """Return the distinct finger-pad geom ids currently contacting the payload."""
        payload_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.config.payload_geom)
        finger_pad_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in self.config.finger_pad_geoms
        }
        contacted: set[int] = set()
        for index in range(int(self.data.ncon)):
            geom1 = int(self.data.contact[index].geom1)
            geom2 = int(self.data.contact[index].geom2)
            pad_id = None
            if geom1 in finger_pad_ids and geom2 == payload_geom_id:
                pad_id = geom1
            elif geom2 in finger_pad_ids and geom1 == payload_geom_id:
                pad_id = geom2
            if pad_id is not None and pad_id in finger_pad_ids:
                contacted.add(pad_id)
        return contacted

    def _update_grasp_status(self) -> None:
        if self._phase != "closing" or self._controller_status != "closing":
            return
        if self._controller_steps >= MAX_STEP_COUNT:
            self._controller_status = "grasp_failed"
            self._paused = True
            return
        expected_pads = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in self.config.finger_pad_geoms
        }
        if expected_pads and self._finger_payload_contact_pads() >= expected_pads:
            self._controller_status = "grasped"
            self._phase = "grasped"
            self._paused = True

    def _reset_control_diagnostics(self) -> None:
        self._raw_arm_controls.fill(0.0)
        self._clipped_arm_controls.fill(0.0)
        self._arm_control_saturated.fill(False)

    def _copy_positions(self, addresses: np.ndarray) -> np.ndarray:
        return np.array(self.data.qpos[addresses], dtype=float, copy=True)

    def _validate_state(self) -> None:
        if self.data.qpos.shape != self._initial_qpos.shape:
            raise ValueError("Simulation data qpos shape does not match the model.")
        if self.data.qvel.shape != self._initial_qvel.shape:
            raise ValueError("Simulation data qvel shape does not match the model.")
        if not np.isfinite(self._initial_qpos).all() or not np.isfinite(self._initial_qvel).all():
            raise ValueError("Simulation initial state must contain finite values.")


def get_manipulation_simulation(
    session_state: MutableMapping[str, object],
    key: str = "manipulation_simulation",
) -> ManipulationSimulation:
    """Return a valid persistent session object, replacing stale state safely."""
    existing = session_state.get(key)
    if isinstance(existing, ManipulationSimulation):
        return existing
    simulation = ManipulationSimulation.create()
    session_state[key] = simulation
    return simulation
