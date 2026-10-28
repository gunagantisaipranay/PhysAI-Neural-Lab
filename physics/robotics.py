"""MuJoCo cart-pole simulation and a transparent balance controller."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are
from collections.abc import Callable

MAX_FORCE_N = 25.0
MAX_DURATION_S = 60.0
MAX_INITIAL_ANGLE_DEG = 45.0
BALANCE_ANGLE_LIMIT_DEG = 5.0
BALANCE_RATE_LIMIT_DEG_S = 20.0
LQR_STATE_COST = np.diag([1.0, 100.0, 1.0, 10.0])
LQR_CONTROL_COST = np.array([[1.0]])

MODEL_XML = """
<mujoco model="physai_cartpole">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="4 1 0.1" rgba="0.18 0.22 0.30 1"/>
    <body name="cart" pos="0 0 0.12">
      <joint name="cart_slide" type="slide" axis="1 0 0" limited="true" range="-2.5 2.5" damping="0.08"/>
      <geom type="box" size="0.18 0.12 0.08" mass="{cart_mass_kg}" rgba="0.13 0.68 0.84 1"/>
      <body name="pole" pos="0 0 0.08">
        <joint name="pole_hinge" type="hinge" axis="0 1 0" damping="0.01"/>
        <geom type="capsule" fromto="0 0 0 0 0 {pole_length_m}" size="0.035" mass="{pole_shaft_mass_kg}" rgba="0.96 0.62 0.16 1"/>
        <geom type="sphere" pos="0 0 {pole_length_m}" size="0.07" mass="{pole_tip_mass_kg}" rgba="0.96 0.62 0.16 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="cart_motor" joint="cart_slide" gear="1" ctrlrange="-25 25"/>
  </actuator>
</mujoco>
"""


def _linearized_lqr_gains(model, mujoco, cart_qpos: int, pole_qpos: int, cart_dof: int, pole_dof: int) -> np.ndarray:
    """Find local feedback gains from MuJoCo's equations around upright."""
    data = mujoco.MjData(model)

    def state_derivative(state: np.ndarray, force: float) -> np.ndarray:
        data.qpos[cart_qpos], data.qpos[pole_qpos] = state[:2]
        data.qvel[cart_dof], data.qvel[pole_dof] = state[2:]
        data.ctrl[0] = force
        mujoco.mj_forward(model, data)
        return np.asarray([state[2], state[3], data.qacc[cart_dof], data.qacc[pole_dof]])

    zero = np.zeros(4, dtype=float)
    epsilon = 1e-5
    basis = np.eye(4, dtype=float)
    A = np.column_stack([
        (state_derivative(zero + epsilon * basis[i], 0.0) - state_derivative(zero - epsilon * basis[i], 0.0))
        / (2.0 * epsilon)
        for i in range(4)
    ])
    force_epsilon = 1e-3
    B = (
        state_derivative(zero, force_epsilon) - state_derivative(zero, -force_epsilon)
    ).reshape(4, 1) / (2.0 * force_epsilon)
    riccati = solve_continuous_are(A, B, LQR_STATE_COST, LQR_CONTROL_COST)
    return np.linalg.solve(LQR_CONTROL_COST, B.T @ riccati).reshape(-1)


def simulate_cartpole(
    duration: float = 8.0,
    initial_angle_deg: float = 12.0,
    controller: bool = True,
    force_limit: float = 25.0,
    cart_mass_kg: float = 1.0,
    pole_mass_kg: float = 0.26,
    pole_length_m: float = 0.75,
    policy: Callable[[np.ndarray], float] | None = None,
) -> dict[str, np.ndarray | float | bool]:
    """Run the MuJoCo cart-pole and optionally balance it with state feedback.

    The feedback is a fixed linear-quadratic regulator (LQR) policy derived
    from the MuJoCo model linearized near the upright state. It is a classical
    controller, not a learned policy.
    """
    values = np.asarray([duration, initial_angle_deg, force_limit, cart_mass_kg, pole_mass_kg, pole_length_m], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Duration, initial angle, and force limit must be finite numbers.")
    if not 0.0 < duration <= MAX_DURATION_S:
        raise ValueError(f"Duration must be greater than 0 and at most {MAX_DURATION_S:g} seconds.")
    if abs(initial_angle_deg) > MAX_INITIAL_ANGLE_DEG:
        raise ValueError(f"Initial angle must be between ±{MAX_INITIAL_ANGLE_DEG:g} degrees.")
    if not 0.0 < force_limit <= MAX_FORCE_N:
        raise ValueError(f"Force limit must be greater than 0 and at most {MAX_FORCE_N:g} N.")
    if not 0.2 <= cart_mass_kg <= 5.0:
        raise ValueError("Cart mass must be between 0.2 and 5 kg.")
    if not 0.05 <= pole_mass_kg <= 1.5:
        raise ValueError("Pole mass must be between 0.05 and 1.5 kg.")
    if not 0.3 <= pole_length_m <= 1.5:
        raise ValueError("Pole length must be between 0.3 and 1.5 m.")
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo is required. Install the project requirements in the robotics environment.") from exc

    model_xml = MODEL_XML.format(
        cart_mass_kg=cart_mass_kg,
        pole_length_m=pole_length_m,
        pole_shaft_mass_kg=pole_mass_kg * 0.75,
        pole_tip_mass_kg=pole_mass_kg * 0.25,
    )
    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    pole_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pole_hinge")
    cart_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cart_slide")
    pole_qpos, pole_dof = model.jnt_qposadr[pole_joint], model.jnt_dofadr[pole_joint]
    cart_qpos, cart_dof = model.jnt_qposadr[cart_joint], model.jnt_dofadr[cart_joint]
    lqr_gains = _linearized_lqr_gains(model, mujoco, cart_qpos, pole_qpos, cart_dof, pole_dof)
    data.qpos[pole_qpos] = np.deg2rad(initial_angle_deg)

    dt = float(model.opt.timestep)
    steps = int(np.ceil(duration / dt))
    stride = max(1, steps // 1200)
    times, cart, cart_velocity, angle, angle_rate, force = [], [], [], [], [], []
    for i in range(steps):
        x, xdot = float(data.qpos[cart_qpos]), float(data.qvel[cart_dof])
        theta, theta_dot = float(data.qpos[pole_qpos]), float(data.qvel[pole_dof])
        # State order: x, theta, x_dot, theta_dot; LQR control is u = -K @ state.
        state = np.asarray([x, theta, xdot, theta_dot])
        if policy is not None:
            u = float(policy(state))
        elif controller:
            u = float(-lqr_gains @ state)
        else:
            u = 0.0
        u = float(np.clip(u, -force_limit, force_limit))
        data.ctrl[0] = u
        mujoco.mj_step(model, data)
        if i % stride == 0 or i == steps - 1:
            times.append(float(data.time))
            cart.append(float(data.qpos[cart_qpos]))
            cart_velocity.append(float(data.qvel[cart_dof]))
            angle.append(float(data.qpos[pole_qpos]))
            angle_rate.append(float(data.qvel[pole_dof]))
            force.append(u)

    angle_arr = np.rad2deg(np.asarray(angle))
    angle_rate_arr = np.rad2deg(np.asarray(angle_rate))
    window_samples = max(2, int(np.ceil(0.5 / (dt * stride))))
    settled_window = slice(max(0, len(angle_arr) - window_samples), None)
    settled = bool(
        np.isfinite(angle_arr).all()
        and np.isfinite(angle_rate_arr).all()
        and np.max(np.abs(angle_arr[settled_window])) <= BALANCE_ANGLE_LIMIT_DEG
        and np.max(np.abs(angle_rate_arr[settled_window])) <= BALANCE_RATE_LIMIT_DEG_S
        and abs(cart[-1]) < 2.4
    )
    in_balance_region = (
        (np.abs(angle_arr) <= BALANCE_ANGLE_LIMIT_DEG)
        & (np.abs(angle_rate_arr) <= BALANCE_RATE_LIMIT_DEG_S)
    )
    violations = np.flatnonzero(~in_balance_region)
    settle_index = int(violations[-1] + 1) if len(violations) else 0
    settling_time_s = (
        float(times[settle_index])
        if settled and settle_index < len(times)
        else float("nan")
    )
    return {
        "time": np.asarray(times),
        "cart_position": np.asarray(cart),
        "cart_velocity": np.asarray(cart_velocity),
        "pole_angle_deg": angle_arr,
        "pole_angular_velocity_deg_s": angle_rate_arr,
        "force": np.asarray(force),
        "lqr_gains": lqr_gains,
        "cart_mass_kg": float(cart_mass_kg),
        "pole_mass_kg": float(pole_mass_kg),
        "pole_length_m": float(pole_length_m),
        "max_abs_angle_deg": float(np.max(np.abs(angle_arr))),
        "final_angle_deg": float(angle_arr[-1]),
        "final_angular_velocity_deg_s": float(angle_rate_arr[-1]),
        "settling_time_s": settling_time_s,
        "settled": settled,
    }


def evaluate_robustness(
    initial_angles_deg: tuple[float, ...] = (-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0),
    force_limits: tuple[float, ...] = (5.0, 10.0, 25.0),
    duration: float = 8.0,
    cart_mass_kg: float = 1.0,
    pole_mass_kg: float = 0.26,
    pole_length_m: float = 0.75,
) -> list[dict[str, float | bool]]:
    """Evaluate a reproducible initial-angle/force-limit grid."""
    results: list[dict[str, float | bool]] = []
    for angle in initial_angles_deg:
        for force_limit in force_limits:
            run = simulate_cartpole(
                duration=duration,
                initial_angle_deg=angle,
                force_limit=force_limit,
                cart_mass_kg=cart_mass_kg,
                pole_mass_kg=pole_mass_kg,
                pole_length_m=pole_length_m,
            )
            results.append({
                "duration_s": float(duration),
                "initial_angle_deg": float(angle),
                "force_limit_N": float(force_limit),
                "cart_mass_kg": float(cart_mass_kg),
                "pole_mass_kg": float(pole_mass_kg),
                "pole_length_m": float(pole_length_m),
                "settled": bool(run["settled"]),
                "peak_angle_deg": float(run["max_abs_angle_deg"]),
                "settling_time_s": float(run["settling_time_s"]),
                "max_cart_travel_m": float(np.max(np.abs(run["cart_position"]))),
                "rms_force_N": float(np.sqrt(np.mean(np.square(run["force"])))),
                "final_angle_deg": float(run["final_angle_deg"]),
            })
    return results
