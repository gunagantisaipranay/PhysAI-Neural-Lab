"""Robotics-focused home and project workflow."""

from __future__ import annotations

import streamlit as st

from ui_theme import banner, card, page_title, step


def render() -> None:
    page_title("PhysAI Robotics Lab", "A laptop-only workspace for robot simulation and model training")
    banner(
        "Load an open-source robot model, place it in a MuJoCo scene, generate task data, "
        "train a robot-specific model, and inspect held-out results. All robot motion here is simulated."
    )

    st.markdown("### Project workflow")
    columns = st.columns(4)
    stages = [
        ("01", "Load a robot", "Built-in arm or MJCF, URDF, MJZ, and ZIP asset bundle."),
        ("02", "Build a scene", "Set joint poses, targets, collision obstacles, and preview a clear tool route."),
        ("03", "Train a model", "Generate randomized MuJoCo examples and fit a supervised joint-target model."),
        ("04", "Evaluate", "Measure held-out end-effector error and download a report."),
    ]
    for col, (number, title, description) in zip(columns, stages):
        with col:
            step(number, title)
            st.caption(description)

    st.markdown("### Available workspaces")
    left, right = st.columns(2)
    with left:
        card(
            "Robot Workspace",
            "Scene setup",
            "View and pose the built-in arm or imported model. Add obstacles, plan a clearance-aware tool route, and inspect contacts.",
            "Ready",
        )
    with right:
        card(
            "Model Training",
            "Supervised inverse kinematics",
            "Generate task examples from robot geometry, train an MLP to predict joint setpoints, and evaluate on held-out targets.",
            "New milestone",
        )

    st.markdown("### What the model learns")
    st.latex(r"(q_{current},\; x_{target}, y_{target}, z_{target}) \longrightarrow q_{goal}")
    st.caption(
        "The first training model is a learned kinematic target mapper, not reinforcement learning. "
        "It currently supports robots with named end-effector sites and bounded hinge/slide joints connected to direct actuators. "
        "Obstacle avoidance, dynamics learning, and hardware transfer are later milestones."
    )
