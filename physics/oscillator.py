"""Analytical solution of the undamped mass-spring oscillator.

The undamped mass-spring system obeys the second-order linear ODE

    m x''(t) + k x(t) = 0

whose analytical solution is

    x(t) = x0 * cos(w t) + (v0 / w) * sin(w t)

with w = sqrt(k / m), x0 the initial displacement and v0 the initial velocity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_parameters(
    mass: float,
    spring_constant: float,
    x0: float,
    v0: float,
    duration: float,
    num_samples: int,
) -> list[str]:
    """Validate physical and simulation parameters.

    Returns a list of human-readable error messages. An empty list means the
    parameters are valid.
    """
    errors: list[str] = []

    if mass <= 0:
        errors.append("Mass must be a positive number (kg).")
    if spring_constant <= 0:
        errors.append("Spring constant must be a positive number (N/m).")
    if duration <= 0:
        errors.append("Simulation duration must be a positive number (seconds).")
    if num_samples < 2:
        errors.append("Number of samples must be at least 2.")

    for name, value in (
        ("Mass", mass),
        ("Spring constant", spring_constant),
        ("Initial displacement", x0),
        ("Initial velocity", v0),
        ("Simulation duration", duration),
    ):
        if not np.isfinite(value):
            errors.append(f"{name} must be a finite number.")

    return errors


def angular_frequency(mass: float, spring_constant: float) -> float:
    """Return the angular frequency w = sqrt(k / m)."""
    if mass <= 0 or spring_constant <= 0:
        raise ValueError("mass and spring_constant must be positive.")
    return float(np.sqrt(spring_constant / mass))


def period(mass: float, spring_constant: float) -> float:
    """Return the oscillation period T = 2 pi / w."""
    return float(2.0 * np.pi / angular_frequency(mass, spring_constant))


def analytical_solution(
    mass: float,
    spring_constant: float,
    x0: float,
    v0: float,
    duration: float,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the analytical displacement and velocity over time.

    Parameters
    ----------
    mass : float
        Mass in kilograms (must be > 0).
    spring_constant : float
        Spring constant in N/m (must be > 0).
    x0 : float
        Initial displacement in metres.
    v0 : float
        Initial velocity in m/s.
    duration : float
        Total simulation duration in seconds (> 0).
    num_samples : int
        Number of time samples to generate (>= 2).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays ``(t, x, v)`` of length ``num_samples``.
    """
    errors = validate_parameters(mass, spring_constant, x0, v0, duration, num_samples)
    if errors:
        raise ValueError("; ".join(errors))

    t = np.linspace(0.0, duration, num_samples)
    w = angular_frequency(mass, spring_constant)
    x = x0 * np.cos(w * t) + (v0 / w) * np.sin(w * t)
    v = -x0 * w * np.sin(w * t) + v0 * np.cos(w * t)
    return t, x, v


def solution_dataframe(
    t: np.ndarray, x: np.ndarray, v: np.ndarray
) -> pd.DataFrame:
    """Package time, displacement and velocity into a tidy DataFrame."""
    return pd.DataFrame(
        {
            "time (s)": t,
            "displacement (m)": x,
            "velocity (m/s)": v,
        }
    )