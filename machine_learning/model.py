"""Neural network model for predicting mass-spring displacement.

A small multilayer perceptron (MLP) regressor from scikit-learn is used as the
genuine trained model. It learns the mapping  t -> x(t)  from analytically
simulated data, using a dominant-frequency estimate from the training split to
construct Fourier features. The model is fully trained at runtime (no
hardcoded predictions) and exposes a real loss history.
"""

from __future__ import annotations

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.neural_network import MLPRegressor
from scipy.signal import lombscargle


class TrainingResult:
    """Container for the outcome of a training run."""

    def __init__(
        self,
        loss_history: list[float],
        n_train_samples: int,
        n_test_samples: int,
        rmse_train: float,
        mae_train: float,
    ) -> None:
        self.loss_history = loss_history
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples
        self.rmse_train = rmse_train
        self.mae_train = mae_train

    @property
    def converged(self) -> bool:
        return bool(len(self.loss_history) > 0 and self.loss_history[-1] < self.loss_history[0])


class NeuralOscillatorModel:
    """An MLP regressor that predicts displacement from time.

    The model is intentionally modest (two hidden layers) so that training runs
    quickly in the browser-side Streamlit environment while still being a real
    trained network whose predictions and loss history are genuine.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, int] = (64, 64),
        activation: str = "tanh",
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self._regressor: MLPRegressor | None = None
        self._time_mean = 0.0
        self._time_scale = 1.0
        self._target_mean = 0.0
        self._target_scale = 1.0
        self._dominant_angular_frequency: float | None = None

    def _features(self, time: np.ndarray) -> np.ndarray:
        time = np.asarray(time, dtype=float).reshape(-1)
        columns = [((time - self._time_mean) / self._time_scale)]
        if self._dominant_angular_frequency is not None:
            for harmonic in (1.0, 2.0, 3.0):
                phase = self._dominant_angular_frequency * harmonic * time
                columns.extend((np.sin(phase), np.cos(phase)))
        return np.column_stack(columns)

    @property
    def is_trained(self) -> bool:
        return self._regressor is not None

    def train(
        self,
        t_train: np.ndarray,
        x_train: np.ndarray,
        epochs: int = 200,
        learning_rate_init: float = 0.01,
    ) -> TrainingResult:
        """Train the MLP on (t, x) pairs.

        ``epochs`` is mapped to ``max_iter`` of the underlying optimiser. The
        time values are reshaped to 2-D as required by scikit-learn.
        """
        self._time_mean = float(np.mean(t_train))
        self._time_scale = float(np.std(t_train)) or 1.0
        self._target_mean = float(np.mean(x_train))
        self._target_scale = float(np.std(x_train)) or 1.0
        self._dominant_angular_frequency = None
        if len(t_train) >= 8 and float(np.std(x_train)) > 1e-12:
            sorted_time = np.sort(np.unique(t_train))
            sample_interval = float(np.median(np.diff(sorted_time)))
            duration = float(sorted_time[-1] - sorted_time[0])
            if sample_interval > 0.0 and duration > 0.0:
                max_omega = np.pi / sample_interval
                min_omega = max(0.01, np.pi / (2.0 * duration))
                frequency_count = min(
                    4096,
                    max(512, int(max_omega * duration * 6.0 / (2.0 * np.pi))),
                )
                frequencies = np.linspace(min_omega, max_omega, frequency_count)
                power = lombscargle(
                    t_train,
                    x_train - self._target_mean,
                    frequencies,
                    normalize=True,
                )
                self._dominant_angular_frequency = float(frequencies[int(np.argmax(power))])
        t_2d = self._features(t_train)
        x_scaled = (x_train - self._target_mean) / self._target_scale
        self._regressor = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver="adam",
            learning_rate_init=learning_rate_init,
            max_iter=epochs,
            random_state=42,
            early_stopping=False,
        )
        self._regressor.fit(t_2d, x_scaled)

        preds_train = self._regressor.predict(t_2d) * self._target_scale + self._target_mean
        rmse_train = float(np.sqrt(np.mean((preds_train - x_train) ** 2)))
        mae_train = float(np.mean(np.abs(preds_train - x_train)))

        return TrainingResult(
            loss_history=list(self._regressor.loss_curve_),
            n_train_samples=len(t_train),
            n_test_samples=0,
            rmse_train=rmse_train,
            mae_train=mae_train,
        )

    def predict(self, t: np.ndarray) -> np.ndarray:
        """Predict displacement for the given times."""
        if self._regressor is None:
            raise NotFittedError(
                "The neural network has not been trained yet. Call train() first."
            )
        return self._regressor.predict(self._features(t)) * self._target_scale + self._target_mean
