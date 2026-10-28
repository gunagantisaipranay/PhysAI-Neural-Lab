"""Train the neural oscillator model."""

from __future__ import annotations

import numpy as np

from .model import NeuralOscillatorModel, TrainingResult


def train_model(
    model: NeuralOscillatorModel,
    t_train: np.ndarray,
    x_train: np.ndarray,
    epochs: int = 200,
    learning_rate: float = 0.01,
) -> TrainingResult:
    """Wrap ``NeuralOscillatorModel.train`` with explicit, validated arguments."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if len(t_train) < 2:
        raise ValueError("Need at least 2 training samples.")

    return model.train(
        t_train=t_train,
        x_train=x_train,
        epochs=epochs,
        learning_rate_init=learning_rate,
    )