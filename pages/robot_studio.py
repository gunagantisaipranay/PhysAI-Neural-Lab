"""Laptop-only 3D robot model and obstacle workspace."""

from __future__ import annotations

import hashlib

import mujoco
import numpy as np
import pandas as pd
import streamlit as st

from physics.robot_scene import (
    Obstacle,
    build_robot_scene,
    inspect_robot_model,
    render_robot_scene,
    supported_model_candidates,
)
from physics.robot_path_planning import plan_tool_path
from physics.robot_simulation import get_manipulation_simulation
from ui_theme import banner, metric_chip, note, page_title


def _obstacle_from_dict(item: dict) -> Obstacle:
    return Obstacle(
        shape=item["shape"],
        position=tuple(item["position"]),
        size=tuple(item["size"]),
    )


def _pose_limits(info: dict, index: int) -> tuple[float, float, float] | None:
    import mujoco

    joint_type = info["joint_types"][index]
    if joint_type not in {
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    }:
        return None
    lower, upper = info["joint_ranges"][index]
    if not info["joint_limited"][index] or lower >= upper:
        if joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
            return -3.14159, 3.14159, 0.05
        return -1.0, 1.0, 0.02
    step = 0.05 if joint_type == int(mujoco.mjtJoint.mjJNT_HINGE) else 0.02
    return lower, upper, step


def render() -> None:
    page_title("3D Robot Studio", "Inspect robot models and build obstacle scenes on your laptop")
    st.markdown(
        "Start with the built-in three-joint spatial arm, or load a robot model. Add static "
        "obstacles and a target marker, adjust joint poses, and inspect it in MuJoCo’s 3D simulation. "
        "Open Model Training to fit and evaluate a robot-specific joint-target model."
    )
    banner(
        "Model import accepts MuJoCo MJCF (`.xml` / `.mjcf`), URDF (`.urdf`), "
        "and bundled MuJoCo (`.mjz`) files. ZIP bundles can include a model and "
        "its relative mesh/texture assets. Review the source model's license "
        "before using it in a public demo."
    )

    if "robot_obstacles" not in st.session_state:
        st.session_state["robot_obstacles"] = []

    upload = None
    model_entry = None
    target_position = (0.65, 0.0, 0.7)
    with st.sidebar:
        st.markdown("### Robot and scene")
        upload = st.file_uploader(
            "Robot model or asset bundle",
            type=["xml", "mjcf", "urdf", "mjz", "zip"],
            help="Single MJCF/URDF/MJZ file, or ZIP containing a model and its mesh assets.",
        )
        if upload is not None:
            try:
                candidates = supported_model_candidates(upload.name, upload.getvalue())
                model_entry = st.selectbox("Model inside bundle", candidates) if len(candidates) > 1 else candidates[0]
                st.session_state["robot_model_upload"] = {
                    "filename": upload.name,
                    "content": upload.getvalue(),
                    "entry": model_entry,
                }
            except ValueError as exc:
                note(f"⚠ {exc}")
                upload = None
        if st.session_state.get("robot_model_upload") and st.button("Forget last imported robot"):
            st.session_state.pop("robot_model_upload", None)
            st.rerun()
        auto_motorize = st.checkbox(
            "Add virtual motors to unactuated hinge/slide joints",
            value=True,
            help="Simulation controls only. Torque limits are generic starting values, not specifications for a physical robot.",
        )

        with st.expander("Target position", expanded=True):
            target_x = st.slider("Target X (m)", -1.5, 1.5, 0.65, 0.05)
            target_y = st.slider("Target Y (m)", -1.5, 1.5, 0.0, 0.05)
            target_z = st.slider("Target Z (m)", 0.05, 1.5, 0.7, 0.05)
            target_position = (target_x, target_y, target_z)
            path_clearance = st.slider("Tool path clearance (m)", 0.02, 0.25, 0.07, 0.01)

        with st.form("robot_obstacle_form", clear_on_submit=True):
            st.markdown("### Add obstacle")
            shape = st.selectbox("Shape", ["box", "sphere", "cylinder", "capsule"])
            ox, oy, oz = st.columns(3)
            with ox:
                obstacle_x = st.number_input("X", -2.0, 2.0, 0.55, 0.05, key="ob_x")
            with oy:
                obstacle_y = st.number_input("Y", -2.0, 2.0, 0.0, 0.05, key="ob_y")
            with oz:
                obstacle_z = st.number_input("Z", 0.02, 2.0, 0.55, 0.05, key="ob_z")
            if shape == "box":
                sx, sy, sz = st.columns(3)
                with sx:
                    size_x = st.number_input("Half-width X", 0.03, 0.6, 0.12, 0.01)
                with sy:
                    size_y = st.number_input("Half-width Y", 0.03, 0.6, 0.12, 0.01)
                with sz:
                    size_z = st.number_input("Half-height Z", 0.03, 0.6, 0.18, 0.01)
                obstacle_size = (size_x, size_y, size_z)
            elif shape == "sphere":
                radius = st.number_input("Radius", 0.03, 0.6, 0.12, 0.01)
                obstacle_size = (radius, radius, radius)
            else:
                radius, half_length = st.columns(2)
                with radius:
                    obstacle_radius = st.number_input("Radius", 0.03, 0.6, 0.10, 0.01)
                with half_length:
                    obstacle_half_length = st.number_input("Half-length", 0.03, 0.8, 0.2, 0.01)
                obstacle_size = (obstacle_radius, obstacle_half_length, obstacle_half_length)
            add_obstacle = st.form_submit_button("Add obstacle")

        if add_obstacle:
            st.session_state["robot_obstacles"].append({
                "shape": shape,
                "position": (obstacle_x, obstacle_y, obstacle_z),
                "size": obstacle_size,
            })
        if st.button("Clear all obstacles", use_container_width=True):
            st.session_state["robot_obstacles"] = []
        if st.session_state["robot_obstacles"]:
            st.caption(f"{len(st.session_state['robot_obstacles'])} obstacle(s) in the scene.")

    obstacle_list = tuple(_obstacle_from_dict(item) for item in st.session_state["robot_obstacles"])
    try:
        model, data = build_robot_scene(
            model_filename=upload.name if upload is not None else None,
            model_content=upload.getvalue() if upload is not None else None,
            model_entry=model_entry,
            obstacles=obstacle_list,
            target_position=target_position,
            auto_motorize=bool(upload is not None and auto_motorize),
        )
        info = inspect_robot_model(model)
    except Exception as exc:  # noqa: BLE001 - model parsers surface format-specific exceptions
        note(f"⚠ Could not load this robot scene: {exc}")
        return

    if info["joint_count"]:
        st.markdown("### Manual joint pose")
        pose_columns = st.columns(min(4, int(info["joint_count"])))
        for index, name in enumerate(info["joints"]):
            limits = _pose_limits(info, index)
            if limits is None:
                continue
            low, high, step = limits
            with pose_columns[index % len(pose_columns)]:
                initial = float(data.qpos[info["joint_qpos_addresses"][index]])
                value = st.slider(
                    name,
                    min_value=float(low),
                    max_value=float(high),
                    value=float(np_clip(initial, low, high)),
                    step=float(step),
                    key=f"robot_pose_{name}_{index}",
                )
            data.qpos[info["joint_qpos_addresses"][index]] = value
        import mujoco
        mujoco.mj_forward(model, data)

    selected_site = None
    valid_path = None
    path_key = None
    if info["sites"]:
        st.markdown("### Collision-aware tool path")
        selected_site = st.selectbox("Tool point", info["sites"])
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, selected_site)
        start_point = data.site_xpos[site_id].copy()
        upload_digest = hashlib.sha256(upload.getvalue()).hexdigest() if upload is not None else "built-in-arm-v1"
        path_key = (
            upload_digest,
            model_entry,
            selected_site,
            tuple(np.round(start_point, 3)),
            tuple(round(float(value), 3) for value in target_position),
            tuple((item.shape, item.position, item.size) for item in obstacle_list),
            round(float(path_clearance), 3),
        )
        old_path = st.session_state.get("robot_planned_path") or {}
        path_col, clear_col = st.columns([3, 1])
        with path_col:
            plan_clicked = st.button("Plan tool route around obstacles", type="primary", use_container_width=True)
        with clear_col:
            clear_clicked = st.button("Clear route", use_container_width=True)
        if clear_clicked:
            st.session_state.pop("robot_planned_path", None)
            old_path = {}
        if plan_clicked:
            try:
                waypoints = plan_tool_path(
                    tuple(map(float, start_point)),
                    target_position,
                    obstacle_list,
                    clearance_m=float(path_clearance),
                )
                st.session_state["robot_planned_path"] = {"key": path_key, "points": waypoints.tolist()}
                old_path = st.session_state["robot_planned_path"]
            except ValueError as exc:
                st.session_state.pop("robot_planned_path", None)
                old_path = {}
                note(f"⚠ Could not plan a route: {exc}")
        if old_path.get("key") == path_key:
            valid_path = np.asarray(old_path["points"], dtype=float)
            path_length = float(np.linalg.norm(np.diff(valid_path, axis=0), axis=1).sum())
            metric_cols = st.columns(2)
            with metric_cols[0]:
                metric_chip(f"{path_length:.2f} m", "planned tool path length")
            with metric_cols[1]:
                metric_chip(str(len(valid_path) - 2), "intermediate waypoints")
            st.caption("The route keeps the tool point outside obstacle margins. It previews point motion only; the whole robot arm is not yet constrained to this path.")
        elif old_path:
            st.caption("The robot pose, target, or obstacles changed. Plan the route again to refresh it.")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_chip(str(info["body_count"]), "bodies")
    with m2:
        metric_chip(str(info["joint_count"]), "joints")
    with m3:
        metric_chip(str(info["actuator_count"]), "actuators")
    with m4:
        metric_chip(str(data.ncon), "contacts at current pose")

    if info["joint_count"] == 0:
        note("This model has no movable joints. It can be inspected in 3D, but needs articulated joints for robotics training.")
    elif info["actuator_count"] == 0:
        note("This model has movable joints but no actuators. Turn on virtual motors to simulate joint control.")
    elif upload is not None and auto_motorize:
        st.success("Model loaded with its declared and virtual actuators. Open Model Training to select an end effector and controlled joints.")
    else:
        st.success("Model loaded with actuators. Open Model Training to select an end effector and controlled joints.")

    image = render_robot_scene(model, data, path_waypoints=valid_path)
    st.markdown("### 3D simulation environment")
    st.image(
        image,
        caption="MuJoCo scene: robot, target, and collision-enabled obstacles",
        width="stretch",
    )

    with st.expander("Loaded model details"):
        st.write(f"**Model:** {info['model_name']}")
        st.write("**Joints:** " + (", ".join(info["joints"]) or "none"))
        st.write("**Actuators:** " + (", ".join(info["actuators"]) or "none"))
        st.write("**Sites:** " + (", ".join(info["sites"]) or "none"))
        if st.session_state["robot_obstacles"]:
            st.dataframe(pd.DataFrame(st.session_state["robot_obstacles"]), use_container_width=True, hide_index=True)

    st.markdown("### Manipulation simulation session")
    st.caption(
        "This separate MuJoCo scene provides bounded single-step simulation for the physical "
        "gripper and payload. It executes pre-grasp, final approach, and a bounded gripper "
        "closure with contact-based grasp verification. It does not execute payload attachment, "
        "lifting, or pick-and-place."
    )
    simulation = get_manipulation_simulation(st.session_state)
    control_columns = st.columns(4)
    with control_columns[0]:
        if st.button("Reset manipulation scene", use_container_width=True):
            simulation.reset()
    with control_columns[1]:
        pause_label = "Resume stepping" if simulation.paused else "Pause stepping"
        if st.button(pause_label, use_container_width=True):
            simulation.set_paused(not simulation.paused)
    with control_columns[2]:
        simulation_steps = st.number_input(
            "Steps",
            min_value=1,
            max_value=1000,
            value=10,
            step=1,
            key="manipulation_step_count",
        )
    with control_columns[3]:
        if st.button("Advance simulation", use_container_width=True):
            try:
                simulation.step(int(simulation_steps))
            except (RuntimeError, TypeError, ValueError) as exc:
                note(f"⚠ Could not advance manipulation simulation: {exc}")
    if st.button("Reach payload (pre-grasp)", type="primary", use_container_width=True):
        try:
            simulation.start_reach_target()
        except (RuntimeError, TypeError, ValueError) as exc:
            note(f"⚠ Could not plan reach target: {exc}")
    if st.button("Close gripper and verify grasp", type="secondary", use_container_width=True):
        try:
            simulation.start_closing()
        except (RuntimeError, TypeError, ValueError) as exc:
            note(f"⚠ Could not start gripper closure: {exc}")

    state = simulation.state()
    status = "Paused" if state.paused else "Running (step mode)"
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_chip(status, "simulation status")
    with m2:
        metric_chip(f"{state.simulation_time:.3f} s", "simulation time")
    with m3:
        metric_chip(
            "[" + ", ".join(f"{value:.3f}" for value in state.payload_position) + "]",
            "payload position (m)",
        )
    with m4:
        metric_chip(state.controller_status, "controller status")
    st.caption(
        "Phase: " + state.phase +
        " · End effector: [" + ", ".join(f"{value:.3f}" for value in state.end_effector_position) +
        "] · Target: " + (
            "not set" if state.target_position is None
            else "[" + ", ".join(f"{value:.3f}" for value in state.target_position) + "]"
        ) +
        " · Reach error: " + (
            "—" if state.reach_error_m is None else f"{state.reach_error_m:.3f} m"
        )
    )
    st.caption(
        "Arm q: [" + ", ".join(f"{value:.3f}" for value in state.arm_joint_positions) +
        "] · Finger q: [" + ", ".join(f"{value:.3f}" for value in state.finger_joint_positions) + "]"
    )
    with st.expander("Reach controller diagnostics", expanded=state.controller_status != "idle"):
        st.write(f"**Phase:** {state.phase}")
        st.write(
            "**Planned arm q:** " + (
                "not set" if state.planned_arm_joint_goal is None
                else "[" + ", ".join(f"{value:.4f}" for value in state.planned_arm_joint_goal) + "]"
            )
        )
        st.write(
            "**Planner final error:** " + (
                "not set" if state.planner_final_error_m is None
                else f"{state.planner_final_error_m:.4f} m"
            )
        )
        diagnostics = pd.DataFrame({
            "actuator_id": state.arm_actuator_ids,
            "raw_control": state.raw_arm_controls,
            "clipped_control": state.clipped_arm_controls,
            "control_low": state.arm_control_ranges[:, 0],
            "control_high": state.arm_control_ranges[:, 1],
            "saturated": state.arm_control_saturated,
        })
        st.dataframe(diagnostics, use_container_width=True, hide_index=True)
    manipulation_camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(manipulation_camera)
    manipulation_camera.lookat[:] = [0.42, 0.0, 0.30]
    manipulation_camera.distance = 2.2
    manipulation_camera.azimuth = 135
    manipulation_camera.elevation = -20
    simulation_image = render_robot_scene(simulation.model, simulation.data, camera=manipulation_camera)
    st.image(
        simulation_image,
        caption="Dedicated manipulation scene: current MuJoCo simulation state",
        width="stretch",
    )


def np_clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
