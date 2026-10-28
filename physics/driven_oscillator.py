"""Simulation and lightweight parameter fitting for a damped driven oscillator."""

from __future__ import annotations

import numpy as np

from .ode_solver import solve_two_state


def simulate_damped_driven(
    times: np.ndarray,
    mass: float,
    spring_constant: float,
    damping: float,
    force_amplitude: float,
    force_frequency: float,
    x0: float,
    v0: float,
) -> np.ndarray:
    """Return displacement at ``times`` for m*x'' + c*x' + k*x = F0*sin(wd*t)."""
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Time data must contain at least two increasing values.")
    if mass <= 0 or spring_constant <= 0 or damping < 0:
        raise ValueError("Mass and spring constant must be positive; damping cannot be negative.")
    relative_time = times - times[0]
    calculation_samples = max(500, min(4000, len(times) * 2))
    grid, displacement, _, _ = solve_two_state(
        "v",
        "(force * sin(omega_d * t) - c * v - k * x) / m",
        {"m": mass, "k": spring_constant, "c": damping, "force": force_amplitude, "omega_d": force_frequency},
        x0,
        v0,
        float(relative_time[-1]),
        calculation_samples,
    )
    return np.interp(relative_time, grid, displacement)


def metrics(measured: np.ndarray, predicted: np.ndarray) -> dict[str, float | np.ndarray]:
    """Calculate residuals and familiar goodness-of-fit metrics."""
    error = predicted - measured
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    total = float(np.sum((measured - np.mean(measured)) ** 2))
    r_squared = float(1 - np.sum(error ** 2) / total) if total > 0 else float("nan")
    return {"error": error, "rmse": rmse, "mae": mae, "r_squared": r_squared}


def fit_spring_and_damping(
    times: np.ndarray,
    measured: np.ndarray,
    mass: float,
    force_amplitude: float,
    force_frequency: float,
    x0: float,
    v0: float,
    k_min: float,
    k_max: float,
    c_min: float,
    c_max: float,
    resolution: int,
) -> dict[str, float | np.ndarray]:
    """Fit k and c by a transparent bounded grid search using RMSE."""
    if k_min <= 0 or k_max < k_min or c_min < 0 or c_max < c_min:
        raise ValueError("Check the fit ranges: k must be positive and c cannot be negative.")
    if resolution < 2:
        raise ValueError("Fit resolution must be at least 2.")
    best: dict[str, float | np.ndarray] | None = None
    for spring_constant in np.linspace(k_min, k_max, resolution):
        for damping in np.linspace(c_min, c_max, resolution):
            predicted = simulate_damped_driven(times, mass, float(spring_constant), float(damping), force_amplitude, force_frequency, x0, v0)
            result = metrics(measured, predicted)
            if best is None or result["rmse"] < best["rmse"]:
                best = {"k": float(spring_constant), "c": float(damping), "predicted": predicted, **result}
    assert best is not None
    return best
