"""Generate training data for the neural network from the analytical solution."""

from __future__ import annotations

import numpy as np
from physics.oscillator import analytical_solution, validate_parameters


def generate_dataset(
    mass: float,
    spring_constant: float,
    x0: float,
    v0: float,
    duration: float,
    num_samples: int,
    test_fraction: float = 0.2,
    seed: int | None = 42,
) -> dict[str, np.ndarray]:
    """Create a train/test split of (t -> x) samples for the neural network.

    The network is trained to map the time ``t`` to the displacement ``x`` for a
    fixed set of physical parameters. Splitting is done randomly so that held-out
    test data is not contiguous (matching a typical regression setup).

    Returns a dict with keys: ``t_train, x_train, t_test, x_test``.
    """
    errors = validate_parameters(mass, spring_constant, x0, v0, duration, num_samples)
    if errors:
        raise ValueError("; ".join(errors))
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie in (0, 1).")

    t, x, _ = analytical_solution(
        mass, spring_constant, x0, v0, duration, num_samples
    )

    rng = np.random.default_rng(seed)
    n = len(t)
    indices = rng.permutation(n)
    n_test = max(1, int(round(test_fraction * n)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return {
        "t_train": t[train_idx],
        "x_train": x[train_idx],
        "t_test": t[test_idx],
        "x_test": x[test_idx],
        "t_all": t,
        "x_all": x,
    }