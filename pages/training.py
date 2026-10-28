"""Train and evaluate a robot-specific inverse-kinematics policy."""

from __future__ import annotations

import hashlib
from io import BytesIO
import joblib

import mujoco
import numpy as np
import pandas as pd
import streamlit as st

from machine_learning.robot_experiment import build_ik_experiment_record
from machine_learning.robot_ik_policy import predict_joint_target, train_ik_policy
from physics.robot_tasks import evaluate_ik_rollouts
from physics.robot_scene import (
    Obstacle,
    build_robot_scene,
    inspect_robot_model,
    render_robot_scene,
    supported_model_candidates,
)
from ui_theme import banner, metric_chip, note, page_title


def _direct_actuated_joints(model) -> list[int]:
    return [
        joint_id for joint_id in range(model.njnt)
        if int(model.jnt_type[joint_id]) in {
            int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
        }
        and bool(model.jnt_limited[joint_id])
        and any(
            int(model.actuator_trntype[actuator_id]) == int(mujoco.mjtTrn.mjTRN_JOINT)
            and int(model.actuator_trnid[actuator_id, 0]) == joint_id
            for actuator_id in range(model.nu)
        )
    ]


def render() -> None:
    page_title("Model Training", "Generate task data, train a robot-specific model, and measure held-out performance")
    banner(
        "This milestone trains a supervised inverse-kinematics model: current joint positions + a 3D target â†’ joint setpoints. "
        "Training examples and held-out evaluation targets are generated from MuJoCo. It is a learned kinematic mapper, "
        "not reinforcement learning or a hardware-ready torque controller."
    )

    with st.sidebar:
        st.markdown("### Training robot")
        uploaded = st.file_uploader(
            "Robot model or asset bundle",
            type=["xml", "mjcf", "urdf", "mjz", "zip"],
            key="training_robot_upload",
            help="Open-source MJCF/URDF models can be imported. Include meshes and textures in a ZIP bundle.",
        )
        saved_upload = st.session_state.get("robot_model_upload") or {}
        use_saved = st.checkbox(
            "Use last robot imported in Robot Workspace",
            value=bool(saved_upload),
            disabled=not bool(saved_upload),
        )
        use_workspace_model = uploaded is None and use_saved and bool(saved_upload)
        model_filename = uploaded.name if uploaded is not None else (
            saved_upload.get("filename") if use_workspace_model else None
        )
        model_bytes = uploaded.getvalue() if uploaded is not None else (
            saved_upload.get("content") if use_workspace_model else None
        )
        model_entry = None
        if uploaded is not None:
            try:
                choices = supported_model_candidates(uploaded.name, model_bytes)
                model_entry = st.selectbox("Model file", choices) if len(choices) > 1 else choices[0]
            except ValueError as exc:
                note(f"âš  {exc}")
                uploaded = None
                model_filename = None
                model_bytes = None
        elif model_filename and model_bytes:
            try:
                choices = supported_model_candidates(model_filename, model_bytes)
                model_entry = saved_upload.get("entry") if saved_upload.get("entry") in choices else choices[0]
                st.caption(f"Using the last robot imported in Robot Workspace: {model_filename}")
            except ValueError as exc:
                note(f"âš  Saved robot model is unavailable: {exc}")
                model_filename = None
                model_bytes = None
        virtual_motors = st.checkbox(
            "Add virtual motors to unactuated scalar joints", value=True,
            help="Generic simulation controls, not real motor specifications.",
        )
        st.caption("Fallback tool point for models without a MuJoCo site")
        offset_cols = st.columns(3)
        with offset_cols[0]:
            offset_x = st.number_input("Tool X (m)", -0.5, 0.5, 0.1, 0.01)
        with offset_cols[1]:
            offset_y = st.number_input("Tool Y (m)", -0.5, 0.5, 0.0, 0.01)
        with offset_cols[2]:
            offset_z = st.number_input("Tool Z (m)", -0.5, 0.5, 0.0, 0.01)
        auto_site_offset = (offset_x, offset_y, offset_z)
        samples = st.slider("Generated examples", 500, 5000, 1500, 250)
        epochs = st.slider("Training epochs", 50, 600, 250, 25)
        seed = st.number_input("Random seed", min_value=0, max_value=100000, value=17, step=1)

    try:
        obstacles = tuple(
            Obstacle(item["shape"], tuple(item["position"]), tuple(item["size"]))
            for item in st.session_state.get("robot_obstacles", [])
        )
        model, _ = build_robot_scene(
            model_filename=model_filename,
            model_content=model_bytes,
            model_entry=model_entry,
            auto_motorize=bool(model_filename and virtual_motors),
            auto_site_offset=auto_site_offset,
            obstacles=obstacles,
        )
        info = inspect_robot_model(model)
    except Exception as exc:  # model import libraries provide format-specific failures
        note(f"âš  Could not prepare this model: {exc}")
        return

    joint_ids = _direct_actuated_joints(model)
    site_names = info["sites"]
    if not joint_ids:
        note("This model needs at least one limited hinge or slide joint connected to a direct joint actuator.")
        return
    if not site_names:
        note("This model has no named site to use as the end effector. Add a site to its MJCF model, then reload it.")
        return

    model_label = model_filename or "Built-in three-joint arm"
    st.markdown(f"### {model_label}")
    st.caption(
        f"{info['body_count']} bodies Â· {info['joint_count']} joints Â· {model.nu} actuators Â· "
        f"{len(site_names)} end-effector sites"
    )
    selected_site = st.selectbox("End-effector site", site_names)
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}" for jid in joint_ids]
    selected_names = st.multiselect("Joints the model may command", joint_names, default=joint_names)
    selected_ids = [joint_ids[joint_names.index(name)] for name in selected_names]
    if not selected_ids:
        note("Select one or more joints before training.")
        return

    model_digest = hashlib.sha256(model_bytes).hexdigest() if model_bytes else "built-in-arm-v1"
    obstacle_key = tuple((item.shape, item.position, item.size) for item in obstacles)
    config_key = (model_label, model_digest, model_entry, selected_site, tuple(selected_names), auto_site_offset, obstacle_key, int(samples), int(epochs), int(seed))
    if st.session_state.get("ik_training_config") != config_key:
        st.session_state.pop("ik_training_result", None)
    if st.button("Generate MuJoCo data and train", type="primary", use_container_width=True):
        with st.spinner("Generating randomized joint/target examples and fitting the neural modelâ€¦"):
            try:
                st.session_state["ik_training_result"] = train_ik_policy(
                    model,
                    selected_site,
                    selected_ids,
                    samples=int(samples),
                    epochs=int(epochs),
                    seed=int(seed),
                )
                st.session_state["ik_training_config"] = config_key
            except Exception as exc:
                note(f"âš  Training could not complete: {exc}")

    result = st.session_state.get("ik_training_result")
    if result is None:
        st.info("Choose a robot and end effector, then train. Start with the built-in arm to verify the complete workflow.")
        return

    st.markdown("### Held-out evaluation")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_chip(f"{result.position_rmse_m:.3f} m", "position RMSE")
    with c2:
        metric_chip(f"{result.position_mae_m:.3f} m", "position MAE")
    with c3:
        metric_chip(f"{result.success_rate * 100:.1f}%", "within 8 cm")
    with c4:
        metric_chip(f"{result.joint_rmse:.3f}", "joint RMSE (model units)")
    with c5:
        metric_chip(f"{result.predicted_collision_rate * 100:.0f}%", "predicted-pose collision rate")
    st.caption(
        f"Trained on {result.train_samples:,} samples; evaluated on {result.test_samples:,} held-out samples. "
        f"Seed: {result.random_seed}. The evaluation tests generated targets within this model's sampled joint ranges."
    )

    loss_df = pd.DataFrame({"epoch": np.arange(1, len(result.loss_curve) + 1), "training_loss": result.loss_curve})
    st.line_chart(loss_df, x="epoch", y="training_loss", x_label="training epoch", y_label="loss")
    report = pd.DataFrame([{
        "robot_model": model_label,
        "end_effector_site": selected_site,
        "controlled_joints": ", ".join(result.joint_names),
        "train_samples": result.train_samples,
        "test_samples": result.test_samples,
        "seed": result.random_seed,
        "joint_rmse_model_units": result.joint_rmse,
        "joint_mae_model_units": result.joint_mae,
        "position_rmse_m": result.position_rmse_m,
        "position_mae_m": result.position_mae_m,
        "success_rate_within_8cm": result.success_rate,
    }])
    experiment_record = build_ik_experiment_record(
        robot_model=model_label,
        end_effector_site=selected_site,
        controlled_joints=result.joint_names,
        train_samples=result.train_samples,
        test_samples=result.test_samples,
        seed=result.random_seed,
        joint_rmse=result.joint_rmse,
        joint_mae=result.joint_mae,
        position_rmse_m=result.position_rmse_m,
        position_mae_m=result.position_mae_m,
        success_rate=result.success_rate,
        predicted_collision_rate=result.predicted_collision_rate,
        parameters={
            "epochs": int(epochs),
            "requested_samples": int(samples),
            "selected_joint_ids": [int(joint_id) for joint_id in selected_ids],
            "model_digest": model_digest,
            "model_entry": model_entry,
        },
    )
    download_cols = st.columns(3)
    with download_cols[0]:
        st.download_button("Download evaluation report (CSV)", report.to_csv(index=False), "robot_policy_evaluation.csv", "text/csv", use_container_width=True)
    with download_cols[1]:
        policy_buffer = BytesIO()
        joblib.dump({
            "policy": result.policy,
            "robot_model": model_label,
            "end_effector_site": selected_site,
            "controlled_joints": result.joint_names,
            "feature_names": result.feature_names,
            "joint_limits": list(zip(result.joint_lows, result.joint_highs)),
            "training_samples": result.train_samples,
            "evaluation_samples": result.test_samples,
            "random_seed": result.random_seed,
        }, policy_buffer)
        st.download_button(
            "Save trained policy (.joblib)", policy_buffer.getvalue(), "robot_ik_policy.joblib",
            "application/octet-stream", use_container_width=True,
        )

    with download_cols[2]:
        st.download_button(
            "Download experiment metadata (JSON)",
            experiment_record.to_json(),
            "robot_ik_experiment.json",
            "application/json",
            use_container_width=True,
        )

    st.markdown("### Closed-loop tracking evaluation")
    st.caption(
        "The learned joint targets are tracked by bounded motor PD with gravity compensation (or the robot's built-in position actuator). "
        "The expert target controller uses the known held-out joint goal. Both run in the current obstacle scene; collisions count as failures."
    )
    rollout_config_key = (config_key, obstacle_key)
    if st.button("Run learned vs expert tracking", use_container_width=True):
        with st.spinner("Running held-out closed-loop MuJoCo episodesâ€¦"):
            try:
                st.session_state["ik_rollout_rows"] = evaluate_ik_rollouts(
                    model, result, selected_site, selected_ids
                )
                st.session_state["ik_rollout_config"] = rollout_config_key
            except Exception as exc:
                note(f"âš  Rollout evaluation could not complete: {exc}")
    rollout_rows = st.session_state.get("ik_rollout_rows")
    if rollout_rows and st.session_state.get("ik_rollout_config") == rollout_config_key:
        rollout_df = pd.DataFrame(rollout_rows)
        learned_rows = rollout_df[rollout_df["controller"] == "learned IK"]
        expert_rows = rollout_df[rollout_df["controller"] == "expert target"]
        r1, r2, r3 = st.columns(3)
        with r1:
            metric_chip(f"{learned_rows['success'].mean() * 100:.0f}%", "learned closed-loop success")
        with r2:
            metric_chip(f"{expert_rows['success'].mean() * 100:.0f}%", "expert-target success")
        with r3:
            metric_chip(f"{learned_rows['obstacle_collision'].sum()}", "learned obstacle collisions")
        st.dataframe(rollout_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download rollout results (CSV)", rollout_df.to_csv(index=False),
            "robot_closed_loop_evaluation.csv", "text/csv",
        )

    data = mujoco.MjData(model)
    start = np.asarray(result.example_start)
    target = np.asarray(result.example_target)
    prediction = predict_joint_target(result, start, target)
    qadr = np.asarray([model.jnt_qposadr[jid] for jid in selected_ids], dtype=int)
    data.qpos[qadr] = prediction
    mujoco.mj_forward(model, data)
    st.markdown("### Example model output")
    st.image(render_robot_scene(model, data), caption="Predicted joint setpoints for one held-out target", use_container_width=True)
