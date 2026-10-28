"""Acceptance checks for the MuJoCo cart-pole simulation."""

import numpy as np
import pytest

pytest.importorskip("mujoco")

from physics.robotics import evaluate_robustness, simulate_cartpole
from machine_learning.robotics_policy import train_lqr_policy


def test_default_controller_balances_and_returns_finite_aligned_signals():
    result = simulate_cartpole()
    keys = (
        "time",
        "cart_position",
        "pole_angle_deg",
        "pole_angular_velocity_deg_s",
        "force",
    )
    signals = [result[key] for key in keys]

    assert len({len(signal) for signal in signals}) == 1
    assert all(np.isfinite(signal).all() for signal in signals)
    assert np.all(np.diff(result["time"]) > 0)
    assert np.max(np.abs(result["force"])) <= 25.0
    assert result["settled"]
    assert np.isfinite(result["settling_time_s"])
    assert 0.0 <= result["settling_time_s"] <= 8.0


@pytest.mark.parametrize("initial_angle", [-12.0, 12.0])
def test_controller_recovers_from_both_sides(initial_angle):
    result = simulate_cartpole(initial_angle_deg=initial_angle)
    assert result["settled"]


def test_uncontrolled_pole_does_not_meet_balance_criterion():
    result = simulate_cartpole(controller=False)
    assert not result["settled"]
    assert np.isnan(result["settling_time_s"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration": 0.0},
        {"duration": float("nan")},
        {"duration": 61.0},
        {"initial_angle_deg": 46.0},
        {"force_limit": 0.0},
        {"force_limit": 26.0},
        {"cart_mass_kg": 0.1},
        {"pole_mass_kg": 2.0},
        {"pole_length_m": 1.6},
    ],
)
def test_invalid_simulation_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        simulate_cartpole(**kwargs)


def test_model_derived_controller_adapts_to_robot_parameters():
    default = simulate_cartpole()
    changed = simulate_cartpole(cart_mass_kg=2.0, pole_length_m=1.2)
    assert not np.allclose(default["lqr_gains"], changed["lqr_gains"])
    assert changed["settled"]


def test_robustness_sweep_records_each_case_and_settings():
    results = evaluate_robustness(
        initial_angles_deg=(-12.0, 12.0),
        force_limits=(10.0,),
        duration=4.0,
        cart_mass_kg=1.5,
    )
    assert len(results) == 2
    assert all(row["settled"] for row in results)
    assert all(row["cart_mass_kg"] == 1.5 for row in results)


def test_learned_policy_matches_lqr_on_held_out_start_angles():
    expert = simulate_cartpole(duration=0.1)
    policy = train_lqr_policy(expert["lqr_gains"], samples=2000, epochs=150)
    assert policy.loss_history[-1] < policy.loss_history[0]

    for angle in (-25.0, 25.0):
        reference = simulate_cartpole(initial_angle_deg=angle)
        learned = simulate_cartpole(initial_angle_deg=angle, policy=policy)
        assert reference["settled"]
        assert learned["settled"]
        assert np.max(np.abs(learned["force"])) <= 25.0
