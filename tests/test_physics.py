"""Tests for the physics oscillator module."""

import numpy as np
import pytest

from physics.oscillator import (
    angular_frequency,
    analytical_solution,
    period,
    validate_parameters,
)
from physics.numerical_solver import rk4_solution


def test_angular_frequency_and_period():
    w = angular_frequency(1.0, 4.0)
    assert np.isclose(w, 2.0)
    T = period(1.0, 4.0)
    assert np.isclose(T, np.pi)


def test_initial_conditions():
    t, x, v = analytical_solution(1.0, 4.0, 0.5, 0.0, 1.0, 100)
    assert np.isclose(x[0], 0.5, atol=1e-10)
    assert np.isclose(v[0], 0.0, atol=1e-10)


def test_zero_displacement_pure_sine():
    # x0=0, v0=1 => x(t) = sin(w t)/w
    m, k = 1.0, 1.0
    w = angular_frequency(m, k)
    t, x, v = analytical_solution(m, k, 0.0, 1.0, 10.0, 200)
    assert np.allclose(x, np.sin(w * t) / w)
    assert np.allclose(v, np.cos(w * t))


def test_rk4_matches_analytical():
    m, k = 0.5, 8.0
    t, x_a, v_a = analytical_solution(m, k, 0.3, 0.1, 5.0, 500)
    _, x_n, v_n = rk4_solution(m, k, 0.3, 0.1, 5.0, 500)
    assert np.allclose(x_a, x_n, atol=1e-6)
    assert np.allclose(v_a, v_n, atol=1e-6)


@pytest.mark.parametrize("mass,k,dur,n", [
    (0.0, 1.0, 1.0, 10),
    (1.0, 0.0, 1.0, 10),
    (1.0, 1.0, 0.0, 10),
    (1.0, 1.0, 1.0, 1),
])
def test_invalid_parameters(mass, k, dur, n):
    assert len(validate_parameters(mass, k, 0.0, 0.0, dur, n)) > 0
    with pytest.raises(ValueError):
        analytical_solution(mass, k, 0.0, 0.0, dur, n)


def test_energy_conservation_periodicity():
    # Over one period the oscillator should return to its initial state.
    m, k = 1.0, 1.0
    T = period(m, k)
    t, x, v = analytical_solution(m, k, 0.2, 0.0, T, 1000)
    assert np.isclose(x[-1], 0.2, atol=1e-6)
    assert np.isclose(v[-1], 0.0, atol=1e-6)