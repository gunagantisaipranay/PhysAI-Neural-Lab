"""Behavior-cloned neural controller for the simulated cart-pole robot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPRegressor


STATE_SCALES = np.asarray([2.5, 0.75, 8.0, 15.0], dtype=float)


@dataclass
class LearnedControlPolicy:
    """MLP policy whose target actions come from the model-derived LQR expert."""

    model: MLPRegressor
    max_force_n: float
    loss_history: list[float]
    training_samples: int
    random_seed: int

    def __call__(self, state: np.ndarray) -> float:
        state = np.asarray(state, dtype=float).reshape(1, 4)
        normalized = state / STATE_SCALES
        normalized_force = float(self.model.predict(normalized)[0])
        return float(np.clip(normalized_force, -1.0, 1.0) * self.max_force_n)


def train_lqr_policy(
    lqr_gains: np.ndarray,
    max_force_n: float = 25.0,
    samples: int = 6000,
    seed: int = 42,
    epochs: int = 250,
) -> LearnedControlPolicy:
    """Train an MLP to imitate the LQR expert over bounded cart-pole states.

    State columns are [cart position, pole angle, cart velocity, pole angular
    velocity]. Training states are deterministic for a fixed seed. Evaluation
    must still use held-out trajectories; matching the expert is not itself
    proof that the policy stabilizes the robot.
    """
    if samples < 100:
        raise ValueError("At least 100 policy training samples are required.")
    if epochs < 1:
        raise ValueError("Policy training epochs must be positive.")
    if not np.isfinite(max_force_n) or max_force_n <= 0.0:
        raise ValueError("Maximum force must be a finite positive number.")
    gains = np.asarray(lqr_gains, dtype=float).reshape(-1)
    if gains.size != 4 or not np.isfinite(gains).all():
        raise ValueError("The LQR expert must provide four finite state gains.")

    rng = np.random.default_rng(seed)
    uniform = rng.uniform(-1.0, 1.0, size=(samples, 4))
    local = np.clip(rng.normal(0.0, 0.30, size=(samples, 4)), -1.0, 1.0)
    normalized_states = np.where(rng.random((samples, 1)) < 0.7, local, uniform)
    states = normalized_states * STATE_SCALES
    expert_force = np.clip(-(states @ gains), -max_force_n, max_force_n)
    actions = expert_force / max_force_n

    model = MLPRegressor(
        hidden_layer_sizes=(32, 32),
        activation="tanh",
        solver="adam",
        learning_rate_init=0.005,
        max_iter=epochs,
        random_state=seed,
        tol=1e-5,
        n_iter_no_change=30,
        early_stopping=False,
    )
    model.fit(normalized_states, actions)
    return LearnedControlPolicy(
        model,
        float(max_force_n),
        list(model.loss_curve_),
        int(samples),
        int(seed),
    )
