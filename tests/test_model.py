"""Tests for the neural-network model and evaluation."""

import numpy as np
import pytest

from machine_learning.dataset import generate_dataset
from machine_learning.model import NeuralOscillatorModel
from machine_learning.train import train_model
from machine_learning.evaluate import evaluate_model
from physics.oscillator import analytical_solution


@pytest.fixture
def dataset():
    return generate_dataset(1.0, 4.0, 0.3, 0.0, 10.0, 400, test_fraction=0.25)


def test_dataset_split_shapes(dataset):
    n_total = 400
    n_test = len(dataset["t_test"])
    n_train = len(dataset["t_train"])
    assert n_test + n_train == n_total
    assert n_test > 0 and n_train > 0
    assert len(dataset["x_train"]) == n_train
    assert len(dataset["x_test"]) == n_test


def test_train_and_predict(dataset):
    model = NeuralOscillatorModel()
    assert not model.is_trained
    result = train_model(model, dataset["t_train"], dataset["x_train"], epochs=50)
    assert model.is_trained
    assert len(result.loss_history) > 0
    preds = model.predict(dataset["t_train"])
    assert preds.shape == dataset["x_train"].shape


def test_predict_before_train_raises():
    model = NeuralOscillatorModel()
    with pytest.raises(Exception):
        model.predict(np.array([0.0, 1.0]))


def test_evaluate_metrics(dataset):
    model = NeuralOscillatorModel()
    train_model(model, dataset["t_train"], dataset["x_train"], epochs=80)
    m, k = 1.0, 4.0
    t_ref, x_ref, _ = analytical_solution(m, k, 0.3, 0.0, 10.0, 400)
    res = evaluate_model(model, t_ref, x_ref)
    assert res["rmse"] >= 0.0
    assert res["mae"] >= 0.0
    assert res["error"].shape == x_ref.shape
    # A trained network on smooth SHM data should fit reasonably well.
    assert res["rmse"] < 0.1


def test_train_invalid_args(dataset):
    model = NeuralOscillatorModel()
    with pytest.raises(ValueError):
        train_model(model, dataset["t_train"], dataset["x_train"], epochs=0)
    with pytest.raises(ValueError):
        train_model(model, dataset["t_train"], dataset["x_train"], learning_rate=-1)