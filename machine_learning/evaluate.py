"""Evaluate the trained neural network against reference values."""

from __future__ import annotations

import numpy as np

from .model import NeuralOscillatorModel


def evaluate_model(
    model: NeuralOscillatorModel,
    t: np.ndarray,
    x_reference: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Compute predictions and genuine error metrics against a reference.

    Returns a dict with keys:
        ``t`` -- the input times,
        ``x_reference`` -- the reference displacement,
        ``x_pred`` -- the model prediction,
        ``error`` -- signed prediction error (x_pred - x_reference),
        ``rmse`` -- root mean squared error,
        ``mae`` -- mean absolute error.
    """
    x_pred = model.predict(t)
    error = x_pred - x_reference
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    return {
        "t": t,
        "x_reference": x_reference,
        "x_pred": x_pred,
        "error": error,
        "rmse": rmse,
        "mae": mae,
    }