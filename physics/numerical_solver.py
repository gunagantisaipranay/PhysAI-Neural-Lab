"""Numerical solver for the mass-spring oscillator.

The second-order ODE  m x'' + k x = 0  is rewritten as the first-order system

    dx/dt = v
    dv/dt = -(k / m) x

and integrated with a classical 4th-order Runge-Kutta scheme. This provides a
genuinely *numerical* solution that can be compared against the closed-form
analytical solution to verify both implementations.
"""

from __future__ import annotations

import numpy as np


def _derivs(state: np.ndarray, mass: float, k: float) -> np.ndarray:
    """Return the time derivative [dx/dt, dv/dt] of the state [x, v]."""
    x, v = state
    return np.array([v, -(k / mass) * x])


def rk4_solution(
    mass: float,
    spring_constant: float,
    x0: float,
    v0: float,
    duration: float,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the oscillator equations with 4th-order Runge-Kutta.

    Parameters mirror :func:`physics.oscillator.analytical_solution` and the
    return signature is identical: ``(t, x, v)``.
    """
    if mass <= 0 or spring_constant <= 0:
        raise ValueError("mass and spring_constant must be positive.")
    if duration <= 0:
        raise ValueError("duration must be positive.")
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2.")

    t = np.linspace(0.0, duration, num_samples)
    dt = t[1] - t[0]

    x = np.empty(num_samples)
    v = np.empty(num_samples)
    state = np.array([x0, v0], dtype=float)
    x[0], v[0] = state

    for i in range(1, num_samples):
        k1 = _derivs(state, mass, spring_constant)
        k2 = _derivs(state + 0.5 * dt * k1, mass, spring_constant)
        k3 = _derivs(state + 0.5 * dt * k2, mass, spring_constant)
        k4 = _derivs(state + dt * k3, mass, spring_constant)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x[i], v[i] = state

    return t, x, v