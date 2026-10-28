"""Repeatable closed-loop joint-target tracking tasks in MuJoCo."""

from __future__ import annotations

import numpy as np

from machine_learning.robot_ik_policy import RobotIKTrainingResult, predict_joint_target


def _actuator_for_joint(model, data, joint_id: int) -> tuple[int, float, str]:
    import mujoco

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
            if int(data.moment_colind[index]) == dof and abs(float(data.actuator_moment[index])) > 1e-9:
                bias_type = int(model.actuator_biastype[actuator_id])
                if bias_type == int(mujoco.mjtBias.mjBIAS_NONE):
                    mode = "motor"
                elif (
                    bias_type == int(mujoco.mjtBias.mjBIAS_AFFINE)
                    and abs(float(model.actuator_biasprm[actuator_id, 1])) > 1e-9
                ):
                    mode = "position"
                else:
                    continue
                return actuator_id, float(data.actuator_moment[index]), mode
    raise ValueError(f"Joint {joint_id} has no usable direct motor actuator.")


def _prepare_joint_episode(model, mujoco, site_name: str, joint_ids: list[int], q_start: np.ndarray, q_goals: np.ndarray):
    """Validate joint configurations and prepare a reset MuJoCo control state."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"End-effector site '{site_name}' does not exist.")
    qadr = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=int)
    dofs = np.asarray([model.jnt_dofadr[joint_id] for joint_id in joint_ids], dtype=int)
    for index, joint_id in enumerate(joint_ids):
        if not model.jnt_limited[joint_id]:
            raise ValueError("Selected joints need finite model limits for safe episode execution.")
        low, high = model.jnt_range[joint_id]
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or str(joint_id)
        if not low <= q_start[index] <= high:
            raise ValueError(
                f"Start joint configuration for '{joint_name}' must be within [{low:g}, {high:g}]."
            )
        if np.any((q_goals[:, index] < low) | (q_goals[:, index] > high)):
            raise ValueError(
                f"Goal joint configuration for '{joint_name}' must be within [{low:g}, {high:g}]."
            )
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[qadr] = q_start
    mujoco.mj_forward(model, data)
    actuator_map = [_actuator_for_joint(model, data, joint_id) for joint_id in joint_ids]
    obstacle_ids = {
        geom_id for geom_id in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith("physai_obstacle_")
    }
    return site_id, qadr, dofs, data, actuator_map, obstacle_ids


def _obstacle_collision(data, obstacle_ids: set[int]) -> bool:
    return any(
        int(data.contact[index].geom1) in obstacle_ids
        or int(data.contact[index].geom2) in obstacle_ids
        for index in range(data.ncon)
    )


def _apply_joint_pd_control(model, data, qadr, dofs, q_goal, actuator_map, kp: float, kd: float) -> None:
    q = data.qpos[qadr]
    qvel = data.qvel[dofs]
    torque = kp * (q_goal - q) - kd * qvel + data.qfrc_bias[dofs]
    for index, (actuator_id, moment, mode) in enumerate(actuator_map):
        desired_control = float(q_goal[index]) if mode == "position" else float(torque[index] / moment)
        low, high = model.actuator_ctrlrange[actuator_id]
        if model.actuator_ctrllimited[actuator_id]:
            desired_control = float(np.clip(desired_control, low, high))
        data.ctrl[actuator_id] = desired_control


def run_joint_target_episode(
    model,
    site_name: str,
    joint_ids: list[int],
    start_q: list[float],
    target_xyz: list[float],
    goal_q: list[float],
    *,
    max_control_steps: int = 500,
    control_interval: int = 5,
    tolerance_m: float = 0.08,
    kp: float = 15.0,
    kd: float = 2.0,
) -> dict[str, float | int | bool]:
    """Use bounded motor effort to track a fixed joint target and measure it."""
    import mujoco

    if not joint_ids or len(joint_ids) != len(start_q) or len(joint_ids) != len(goal_q):
        raise ValueError("Joint ids, initial positions, and target positions must have matching nonzero lengths.")
    if max_control_steps < 1 or control_interval < 1 or tolerance_m <= 0:
        raise ValueError("Episode step counts and tolerance must be positive.")
    target = np.asarray(target_xyz, dtype=float)
    q_start = np.asarray(start_q, dtype=float)
    q_goal = np.asarray(goal_q, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all() or not np.isfinite(q_start).all() or not np.isfinite(q_goal).all():
        raise ValueError("Episode start, target, and goal values must be finite.")

    site_id, qadr, dofs, data, actuator_map, obstacle_ids = _prepare_joint_episode(
        model, mujoco, site_name, joint_ids, q_start, q_goal.reshape(1, -1)
    )

    def errors() -> tuple[float, bool]:
        distance = float(np.linalg.norm(data.site_xpos[site_id] - target))
        collision = _obstacle_collision(data, obstacle_ids)
        return distance, collision

    initial_error, initial_collision = errors()
    distance = initial_error
    collision = initial_collision
    steps = 0
    if collision:
        return {
            "initial_error_m": initial_error,
            "final_error_m": distance,
            "success": False,
            "obstacle_collision": True,
            "control_steps": 0,
            "final_joint_error": float(np.sqrt(np.mean((data.qpos[qadr] - q_goal) ** 2))),
            "contact_count": int(data.ncon),
        }
    for steps in range(1, max_control_steps + 1):
        _apply_joint_pd_control(model, data, qadr, dofs, q_goal, actuator_map, kp, kd)
        for _ in range(control_interval):
            mujoco.mj_step(model, data)
        distance, collision = errors()
        if distance <= tolerance_m or collision:
            break

    return {
        "initial_error_m": initial_error,
        "final_error_m": distance,
        "success": bool(distance <= tolerance_m and not collision),
        "obstacle_collision": bool(collision),
        "control_steps": steps,
        "final_joint_error": float(np.sqrt(np.mean((data.qpos[qadr] - q_goal) ** 2))),
        "contact_count": int(data.ncon),
    }


def run_joint_trajectory_episode(
    model,
    site_name: str,
    joint_ids: list[int],
    start_q: list[float],
    waypoint_q: list[list[float]],
    *,
    max_control_steps_per_waypoint: int = 500,
    control_interval: int = 5,
    tolerance_m: float = 0.08,
    kp: float = 15.0,
    kd: float = 2.0,
) -> dict[str, float | int | bool]:
    """Track joint-space waypoints and report Cartesian tracking and obstacle safety."""
    import mujoco

    if not joint_ids or len(joint_ids) != len(start_q):
        raise ValueError("Joint ids and initial positions must have matching nonzero lengths.")
    if max_control_steps_per_waypoint < 1 or control_interval < 1 or tolerance_m <= 0:
        raise ValueError("Episode step counts and tolerance must be positive.")
    q_start = np.asarray(start_q, dtype=float)
    waypoints = np.asarray(waypoint_q, dtype=float)
    if (
        q_start.shape != (len(joint_ids),)
        or waypoints.ndim != 2
        or waypoints.shape[0] == 0
        or waypoints.shape[1] != len(joint_ids)
        or not np.isfinite(q_start).all()
        or not np.isfinite(waypoints).all()
    ):
        raise ValueError("Trajectory start and waypoints must be finite joint configurations with matching dimensions.")

    site_id, qadr, dofs, data, actuator_map, obstacle_ids = _prepare_joint_episode(
        model, mujoco, site_name, joint_ids, q_start, waypoints
    )
    reference = mujoco.MjData(model)
    waypoint_targets = []
    for goal in waypoints:
        reference.qpos[qadr] = goal
        mujoco.mj_forward(model, reference)
        waypoint_targets.append(reference.site_xpos[site_id].copy())

    initial_collision = _obstacle_collision(data, obstacle_ids)
    if initial_collision:
        return {
            "success": False, "obstacle_collision": True, "control_steps": 0,
            "waypoints_completed": 0, "waypoint_count": len(waypoints),
            "tracking_rmse_m": float("nan"), "max_tracking_error_m": float("nan"),
            "final_joint_error": float(np.sqrt(np.mean((data.qpos[qadr] - waypoints[-1]) ** 2))),
            "contact_count": int(data.ncon),
        }

    errors, total_steps, completed, collision = [], 0, 0, False
    for goal, target in zip(waypoints, waypoint_targets):
        distance = float(np.linalg.norm(data.site_xpos[site_id] - target))
        for _ in range(max_control_steps_per_waypoint):
            _apply_joint_pd_control(model, data, qadr, dofs, goal, actuator_map, kp, kd)
            for _ in range(control_interval):
                mujoco.mj_step(model, data)
            total_steps += 1
            distance = float(np.linalg.norm(data.site_xpos[site_id] - target))
            collision = _obstacle_collision(data, obstacle_ids)
            if distance <= tolerance_m or collision:
                break
        errors.append(distance)
        if collision or distance > tolerance_m:
            break
        completed += 1

    error_array = np.asarray(errors, dtype=float)
    return {
        "success": bool(completed == len(waypoints) and not collision),
        "obstacle_collision": bool(collision),
        "control_steps": total_steps,
        "waypoints_completed": completed,
        "waypoint_count": len(waypoints),
        "tracking_rmse_m": float(np.sqrt(np.mean(error_array ** 2))),
        "max_tracking_error_m": float(np.max(error_array)),
        "final_joint_error": float(np.sqrt(np.mean((data.qpos[qadr] - waypoints[-1]) ** 2))),
        "contact_count": int(data.ncon),
    }


def evaluate_ik_rollouts(
    model,
    result: RobotIKTrainingResult,
    site_name: str,
    joint_ids: list[int],
    *,
    max_tasks: int = 8,
    max_control_steps: int = 500,
) -> list[dict[str, float | int | bool | str]]:
    """Compare learned and expert joint targets on held-out MuJoCo tasks."""
    rows = []
    for index, task in enumerate(result.heldout_tasks[:max_tasks]):
        learned_goal = predict_joint_target(result, np.asarray(task["start"]), np.asarray(task["target"]))
        for method, goal in (("learned IK", learned_goal), ("expert target", np.asarray(task["expert_goal"]))):
            episode = run_joint_target_episode(
                model,
                site_name,
                joint_ids,
                task["start"],
                task["target"],
                goal.tolist(),
                max_control_steps=max_control_steps,
            )
            rows.append({"task": index + 1, "controller": method, **episode})
    return rows
