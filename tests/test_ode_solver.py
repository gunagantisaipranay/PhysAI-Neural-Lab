"""Tests for the safe two-state RK4 equation solver."""

import numpy as np
import pytest

from physics.ode_solver import compile_expression, solve_two_state


def test_harmonic_oscillator_matches_cosine():
    t, x, v, acceleration = solve_two_state(
        "v", "-x", {}, 1.0, 0.0, 2.0, 1000
    )
    assert np.allclose(x, np.cos(t), atol=1e-8)
    assert np.allclose(v, -np.sin(t), atol=1e-8)
    assert np.allclose(acceleration, -x, atol=1e-8)


def test_rejects_unknown_symbols():
    with pytest.raises(ValueError, match="Unknown symbol"):
        compile_expression("unknown + x", set())


def test_rejects_unsafe_expression():
    with pytest.raises(ValueError):
        compile_expression("__import__('os').system('x')", set())
