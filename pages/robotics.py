"""Interactive MuJoCo cart-pole robotics lab."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from machine_learning.robotics_policy import train_lqr_policy
from physics.robotics import evaluate_robustness, simulate_cartpole
from ui_theme import metric_chip, note, page_title


def _plot(time, values, title: str, y_title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=time, y=values, mode="lines", line={"color": color, "width": 2}))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title=y_title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 20, "r": 20, "t": 45, "b": 35},
        height=320,
    )
    return fig


def render() -> None:
    page_title("Robotics Lab", "MuJoCo cart-pole simulation with linear-quadratic control")
    st.markdown(
        "Explore how a robot can balance an inverted pendulum by moving its cart. "
        "The baseline uses LQR feedback derived from the **MuJoCo** model; the "
        "learned MLP below imitates that baseline. It is a simulation experiment, "
        "not a hardware-validated robot controller."
    )
    st.latex(r"z = [x,\,\theta,\,\dot{x},\,\dot{\theta}]^T, \qquad u = \operatorname{clip}(-Kz,\,-F_{\max}, F_{\max})")

    with st.sidebar.expander("Robot and control settings", expanded=True):
        duration = st.slider("Run duration (s)", 2.0, 15.0, 8.0, 0.5)
        initial_angle = st.slider("Initial pole angle (degrees)", -25.0, 25.0, 12.0, 1.0)
        force_limit = st.slider("Maximum cart force (N)", 5.0, 25.0, 25.0, 1.0)
        cart_mass = st.slider("Cart mass (kg)", 0.2, 5.0, 1.0, 0.1)
        pole_mass = st.slider("Pole mass (kg)", 0.05, 1.5, 0.26, 0.01)
        pole_length = st.slider("Pole length (m)", 0.3, 1.5, 0.75, 0.05)
        controller = st.toggle("Enable balance controller", value=True)

    try:
        result = simulate_cartpole(
            duration,
            initial_angle,
            controller,
            force_limit,
            cart_mass_kg=cart_mass,
            pole_mass_kg=pole_mass,
            pole_length_m=pole_length,
        )
    except (RuntimeError, ValueError) as exc:
        note(f"⚠ {exc}")
        return

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_chip(f"{result['max_abs_angle_deg']:.2f}°", "maximum pole angle")
    with m2:
        metric_chip(f"{result['cart_position'][-1]:.3f} m", "final cart position")
    with m3:
        metric_chip("Balanced" if result["settled"] else "Unsettled", "final 0.5 s balance criterion")
    with m4:
        settle_label = f"{result['settling_time_s']:.2f} s" if result["settled"] else "—"
        metric_chip(settle_label, "time to settle")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_plot(result["time"], result["pole_angle_deg"], "Pole response", "Angle (degrees)", "#22d3ee"), use_container_width=True)
    with c2:
        st.plotly_chart(_plot(result["time"], result["cart_position"], "Cart motion", "Position (m)", "#f59e0b"), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(_plot(result["time"], result["pole_angular_velocity_deg_s"], "Pole angular velocity", "Angular velocity (degrees/s)", "#a78bfa"), use_container_width=True)
    with c4:
        st.plotly_chart(_plot(result["time"], result["force"], "Controller output", "Applied force (N)", "#34d399"), use_container_width=True)
    st.caption(
        "LQR gain vector K for [cart position, pole angle, cart speed, pole "
        "angular speed]: " + ", ".join(f"{gain:.3f}" for gain in result["lqr_gains"])
    )

    frame = pd.DataFrame({
        "time_s": result["time"],
        "cart_position_m": result["cart_position"],
        "pole_angle_deg": result["pole_angle_deg"],
        "pole_angular_velocity_deg_s": result["pole_angular_velocity_deg_s"],
        "applied_force_N": result["force"],
        "settling_time_s": result["settling_time_s"],
        "cart_mass_kg": result["cart_mass_kg"],
        "pole_mass_kg": result["pole_mass_kg"],
        "pole_length_m": result["pole_length_m"],
        "lqr_gain_x": result["lqr_gains"][0],
        "lqr_gain_theta": result["lqr_gains"][1],
        "lqr_gain_xdot": result["lqr_gains"][2],
        "lqr_gain_thetadot": result["lqr_gains"][3],
    })
    st.download_button("Download run data (CSV)", frame.to_csv(index=False), "cartpole_run.csv", "text/csv")

    st.markdown("### Controller robustness sweep")
    st.caption("Compare the model-derived controller across seven starting angles and three force limits. Each row records the robot settings used for that sweep.")
    if st.button("Run 21-case sweep"):
        with st.spinner("Simulating the angle and force-limit grid…"):
            st.session_state["robotics_sweep"] = evaluate_robustness(
                cart_mass_kg=cart_mass,
                pole_mass_kg=pole_mass,
                pole_length_m=pole_length,
            )
    if "robotics_sweep" in st.session_state:
        sweep = pd.DataFrame(st.session_state["robotics_sweep"])
        success_rate = float(sweep["settled"].mean()) * 100.0
        metric_chip(f"{success_rate:.0f}%", "settled runs in the saved grid")
        st.dataframe(sweep, use_container_width=True, hide_index=True)
        st.download_button(
            "Download sweep results (CSV)",
            sweep.to_csv(index=False),
            "cartpole_robustness_sweep.csv",
            "text/csv",
        )

    st.markdown("### Learned control policy")
    st.caption(
        "Train an MLP to imitate the current LQR controller, then compare both "
        "controllers on starting angles excluded from its random training set."
    )
    policy_key = (cart_mass, pole_mass, pole_length, force_limit)
    if st.button("Train MLP policy from LQR examples"):
        with st.spinner("Training the neural control policy…"):
            st.session_state["learned_robotics_policy"] = train_lqr_policy(
                result["lqr_gains"], max_force_n=force_limit
            )
            st.session_state["learned_robotics_policy_key"] = policy_key
            st.session_state.pop("learned_robotics_evaluation", None)

    learned_policy = st.session_state.get("learned_robotics_policy")
    learned_key = st.session_state.get("learned_robotics_policy_key")
    if learned_policy is not None and learned_key == policy_key:
        metric_chip(f"{len(learned_policy.loss_history)}", "MLP training epochs")
        st.caption(
            f"Training used {learned_policy.training_samples:,} synthetic LQR "
            f"state/action examples with random seed {learned_policy.random_seed}."
        )
        if st.button("Evaluate on held-out angles"):
            held_out_angles = (-25.0, -15.0, 15.0, 25.0)
            rows = []
            with st.spinner("Comparing the LQR and learned policy on held-out runs…"):
                for angle in held_out_angles:
                    lqr_run = simulate_cartpole(
                        duration=duration,
                        initial_angle_deg=angle,
                        force_limit=force_limit,
                        cart_mass_kg=cart_mass,
                        pole_mass_kg=pole_mass,
                        pole_length_m=pole_length,
                    )
                    mlp_run = simulate_cartpole(
                        duration=duration,
                        initial_angle_deg=angle,
                        force_limit=force_limit,
                        cart_mass_kg=cart_mass,
                        pole_mass_kg=pole_mass,
                        pole_length_m=pole_length,
                        policy=learned_policy,
                    )
                    rows.append({
                        "duration_s": duration,
                        "held_out_angle_deg": angle,
                        "force_limit_N": force_limit,
                        "cart_mass_kg": cart_mass,
                        "pole_mass_kg": pole_mass,
                        "pole_length_m": pole_length,
                        "policy_training_samples": learned_policy.training_samples,
                        "policy_training_seed": learned_policy.random_seed,
                        "policy_training_epochs": len(learned_policy.loss_history),
                        "lqr_settled": lqr_run["settled"],
                        "mlp_settled": mlp_run["settled"],
                        "lqr_peak_angle_deg": lqr_run["max_abs_angle_deg"],
                        "mlp_peak_angle_deg": mlp_run["max_abs_angle_deg"],
                        "lqr_settling_time_s": lqr_run["settling_time_s"],
                        "mlp_settling_time_s": mlp_run["settling_time_s"],
                        "lqr_max_cart_travel_m": max(abs(lqr_run["cart_position"])),
                        "mlp_max_cart_travel_m": max(abs(mlp_run["cart_position"])),
                    })
            st.session_state["learned_robotics_evaluation"] = rows
        evaluations = st.session_state.get("learned_robotics_evaluation")
        if evaluations:
            evaluation_df = pd.DataFrame(evaluations)
            st.dataframe(evaluation_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download policy comparison (CSV)",
                evaluation_df.to_csv(index=False),
                "cartpole_policy_comparison.csv",
                "text/csv",
            )
    elif learned_policy is not None:
        st.info("The robot settings changed. Retrain the learned policy for these settings before comparing it.")
    st.caption(
        "Balanced means the pole stays within ±5° and ±20°/s for the final "
        "0.5 s, with the cart inside its track limits. This is a simplified "
        "simulation; this LQR controller is not learned or validated "
        "for physical hardware."
    )
