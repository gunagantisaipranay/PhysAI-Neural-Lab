"""Tests for real-data damped oscillator utilities."""

import numpy as np

from physics.driven_oscillator import fit_spring_and_damping, metrics, simulate_damped_driven


def test_perfect_model_has_zero_error():
    time = np.linspace(0, 2, 30)
    measured = simulate_damped_driven(time, 1.0, 4.0, 0.5, 0.0, 1.0, 0.4, 0.0)
    result = metrics(measured, measured)
    assert result["rmse"] == 0.0
    assert result["mae"] == 0.0


def test_grid_search_recovers_in_grid_values():
    time = np.linspace(0, 2, 30)
    measured = simulate_damped_driven(time, 1.0, 4.0, 0.5, 0.0, 1.0, 0.4, 0.0)
    result = fit_spring_and_damping(time, measured, 1.0, 0.0, 1.0, 0.4, 0.0, 2.0, 6.0, 0.0, 1.0, 3)
    assert np.isclose(result["k"], 4.0)
    assert np.isclose(result["c"], 0.5)
