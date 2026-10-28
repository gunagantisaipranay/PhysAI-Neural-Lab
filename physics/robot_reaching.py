"""Bounded Jacobian target-reaching control for the manipulation scene."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from physics.robot_manipulation import ManipulationSceneConfig


MAX_IK_ITERATIONS = 200
MAX_JOINT_STEP_RAD = 0.12
DEFAULT_REACH_TOLERANCE_M = 0.035
DEFAULT_PREGRASP_CLEARANCE_M = 0.12
DEFAULT_FINAL_APPROACH_CLEARANCE_M = 0.02


@dataclass(frozen=True)
class ReachPlan:
    """A validated arm joint goal for a Cartesian tool-center target."""

    target_position: np.ndarray
    goal_joint_positions: np.ndarray
    final_position: np.ndarray
    final_error_m: float


def _named_id(model, object_type, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Manipulation scene is missing required component '{name}'.")
    return object_id


def _joint_data(model, config: ManipulationSceneConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    joint_ids = np.asarray(
        [_named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in config.arm_joints],
        dtype=int,
    )
    qadr = np.asarray(model.jnt_qposadr[joint_ids], dtype=int)
    dofs = np.asarray(model.jnt_dofadr[joint_ids], dtype=int)
    if not np.all(model.jnt_limited[joint_ids]):
        raise ValueError("All arm joints must have finite limits for target reaching.")
    lows = np.asarray(model.jnt_range[joint_ids, 0], dtype=float)
    highs = np.asarray(model.jnt_range[joint_ids, 1], dtype=float)
    if not np.isfinite(lows).all() or not np.isfinite(highs).all() or np.any(lows >= highs):
        raise ValueError("Arm joint limits must be finite and ordered.")
    return joint_ids, qadr, dofs, np.stack((lows, highs))


def pregrasp_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: ManipulationSceneConfig,
    clearance_m: float = DEFAULT_PREGRASP_CLEARANCE_M,
) -> np.ndarray:
    """Place a tool-center target just outside the payload along the current approach ray."""
    if not np.isfinite(clearance_m) or clearance_m <= 0.0:
        raise ValueError("Pre-grasp clearance must be positive and finite.")
    payload_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, config.payload_body)
    tool_id = _named_id(model, mujoco.mjtObj.mjOBJ_SITE, config.tool_center_site)
    payload = np.asarray(data.xpos[payload_id], dtype=float)
    tool = np.asarray(data.site_xpos[tool_id], dtype=float)
    direction = tool - payload
    distance = float(np.linalg.norm(direction))
    if not np.isfinite(payload).all() or not np.isfinite(tool).all():
        raise ValueError("Payload and tool-center positions must be finite.")
    if distance <= 1e-9:
        direction = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        direction = direction / distance
    # The payload half-diagonal is a conservative box envelope; the extra
    # clearance keeps the tool center outside the contact geometry.
    payload_half_diagonal = float(np.sqrt(3.0) * 0.055)
    return payload + direction * (payload_half_diagonal + clearance_m)


def final_approach_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: ManipulationSceneConfig,
    clearance_m: float = DEFAULT_FINAL_APPROACH_CLEARANCE_M,
) -> np.ndarray:
    """Place the tool-center final-approach target outside the payload solid.

    The final approach starts from the safe pre-grasp standoff and drives the
    open-gripper tool center toward the payload along the current approach ray.
    The target distance is the payload ray/box surface entry distance plus a
    validated approach clearance, so the final-approach waypoint is always
    outside the payload collision geometry (including diagonal approach rays).
    """
    if not np.isfinite(clearance_m) or clearance_m <= 0.0:
        raise ValueError("Final-approach clearance must be positive and finite.")
    payload_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, config.payload_body)
    tool_id = _named_id(model, mujoco.mjtObj.mjOBJ_SITE, config.tool_center_site)
    payload_geom_id = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, config.payload_geom)
    payload = np.asarray(data.xpos[payload_id], dtype=float)
    tool = np.asarray(data.site_xpos[tool_id], dtype=float)
    direction = tool - payload
    distance = float(np.linalg.norm(direction))
    if not np.isfinite(payload).all() or not np.isfinite(tool).all():
        raise ValueError("Payload and tool-center positions must be finite.")
    if distance <= 1e-9:
        direction = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        direction = direction / distance
    payload_half_extents = np.asarray(model.geom_size[payload_geom_id], dtype=float)[:3]
    if payload_half_extents.shape != (3,) or not np.isfinite(payload_half_extents).all():
        raise ValueError("Payload grip envelope must contain three finite half-extents.")
    ray_entries = [
        float(payload_half_extents[axis]) / abs(float(direction[axis]))
        for axis in range(3)
        if abs(float(direction[axis])) > 1e-9
    ]
    if not ray_entries:
        raise ValueError("Final-approach direction must have a finite component.")
    entry_distance = min(ray_entries)
    approach_distance = entry_distance + float(clearance_m)
    if not np.isfinite(approach_distance) or approach_distance <= 0.0:
        raise ValueError("Final-approach distance must be positive and finite.")
    return payload + direction * approach_distance


def plan_reach_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: ManipulationSceneConfig,
    target_position: np.ndarray | list[float],
    *,
    tolerance_m: float = DEFAULT_REACH_TOLERANCE_M,
) -> ReachPlan:
    """Solve a bounded local IK problem and reject targets outside its tolerance."""
    target = np.asarray(target_position, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("Reach target must contain three finite world coordinates.")
    if not np.isfinite(tolerance_m) or tolerance_m <= 0.0:
        raise ValueError("Reach tolerance must be positive and finite.")

    _, qadr, dofs, limits = _joint_data(model, config)
    site_id = _named_id(model, mujoco.mjtObj.mjOBJ_SITE, config.tool_center_site)
    probe = mujoco.MjData(model)
    probe.qpos[:] = data.qpos
    probe.qvel[:] = 0.0
    mujoco.mj_forward(model, probe)
    q = np.asarray(probe.qpos[qadr], dtype=float).copy()
    lows, highs = limits
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)

    for _ in range(MAX_IK_ITERATIONS):
        probe.qpos[qadr] = q
        mujoco.mj_forward(model, probe)
        error = target - probe.site_xpos[site_id]
        if float(np.linalg.norm(error)) <= tolerance_m:
            break
        mujoco.mj_jacSite(model, probe, jacp, jacr, site_id)
        jacobian = jacp[:, dofs]
        regularized = jacobian @ jacobian.T + np.eye(3) * 1e-4
        try:
            delta = jacobian.T @ np.linalg.solve(regularized, error)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Reach target has a singular local Jacobian.") from exc
        delta_norm = float(np.linalg.norm(delta))
        if not np.isfinite(delta).all() or delta_norm <= 1e-10:
            break
        if delta_norm > MAX_JOINT_STEP_RAD:
            delta *= MAX_JOINT_STEP_RAD / delta_norm
        q = np.clip(q + delta, lows, highs)

    probe.qpos[qadr] = q
    mujoco.mj_forward(model, probe)
    final_position = np.asarray(probe.site_xpos[site_id], dtype=float).copy()
    final_error = float(np.linalg.norm(target - final_position))
    if final_error > tolerance_m:
        raise ValueError(
            f"Reach target is not reachable within {tolerance_m:.3f} m "
            f"(best error {final_error:.3f} m)."
        )
    if np.any(q < lows - 1e-9) or np.any(q > highs + 1e-9):
        raise ValueError("Reach solution violates an arm joint limit.")
    return ReachPlan(target.copy(), q.copy(), final_position, final_error)


def actuator_for_joint(model: mujoco.MjModel, data: mujoco.MjData, joint_id: int) -> tuple[int, float]:
    """Resolve a direct motor actuator and its current joint moment."""
    dof = int(model.jnt_dofadr[joint_id])
    for actuator_id in range(model.nu):
        if (
            int(model.actuator_trntype[actuator_id]) != int(mujoco.mjtTrn.mjTRN_JOINT)
            or int(model.actuator_trnid[actuator_id, 0]) != joint_id
        ):
            continue
        row_start = int(data.moment_rowadr[actuator_id])
        row_count = int(data.moment_rownnz[actuator_id])
        for index in range(row_start, row_start + row_count):
            if int(data.moment_colind[index]) == dof:
                moment = float(data.actuator_moment[index])
                if abs(moment) > 1e-9:
                    return actuator_id, moment
    raise ValueError(f"Joint {joint_id} has no usable direct motor actuator.")
