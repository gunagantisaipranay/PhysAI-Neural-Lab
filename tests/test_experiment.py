
"""Tests for shared experiment metadata and metrics."""

import math

import pytest

import json

from machine_learning.experiment import ExperimentRecord


def test_record_stores_metadata_and_metrics():
    record = ExperimentRecord(
        name="robot-ik-baseline",
        algorithm="MLPRegressor",
        seed=17,
        parameters={"epochs": 100},
        dataset={"train_samples": 800, "test_samples": 200},
        metrics={"position_rmse_m": 0.025},
    )

    result = record.to_dict()
    assert result["name"] == "robot-ik-baseline"
    assert result["algorithm"] == "MLPRegressor"
    assert result["seed"] == 17
    assert result["parameters"]["epochs"] == 100
    assert result["dataset"]["test_samples"] == 200
    assert result["metrics"]["position_rmse_m"] == pytest.approx(0.025)
    assert result["schema_version"] == 1


def test_set_metric_accepts_finite_numeric_values():
    record = ExperimentRecord("test", "baseline", 0)
    record.set_metric("rmse", 0)
    record.set_metric("mae", 1.25)

    assert record.metrics == {"rmse": 0.0, "mae": 1.25}


@pytest.mark.parametrize("name", ["", "  ", None])
def test_invalid_experiment_name_rejected(name):
    with pytest.raises(ValueError):
        ExperimentRecord(name=name, algorithm="baseline", seed=0)


@pytest.mark.parametrize("seed", [-1, True, 1.5, "7"])
def test_invalid_seed_rejected(seed):
    with pytest.raises(ValueError):
        ExperimentRecord("test", "baseline", seed)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "1.0"])
def test_invalid_metric_value_rejected(value):
    record = ExperimentRecord("test", "baseline", 0)
    with pytest.raises(ValueError):
        record.set_metric("rmse", value)


def test_invalid_metric_name_rejected():
    record = ExperimentRecord("test", "baseline", 0)
    with pytest.raises(ValueError):
        record.set_metric("  ", 1.0)


def test_constructor_validates_initial_metrics():
    with pytest.raises(ValueError):
        ExperimentRecord("test", "baseline", 0, metrics={"rmse": math.nan})



def test_json_round_trip_preserves_record():
    original = ExperimentRecord(
        name="robot-ik-baseline",
        algorithm="MLPRegressor",
        seed=17,
        parameters={"epochs": 100, "learning_rate": 0.002},
        dataset={"train_samples": 800, "test_samples": 200},
        metrics={"position_rmse_m": 0.025},
    )

    serialized = original.to_json()
    decoded = json.loads(serialized)
    restored = ExperimentRecord.from_dict(decoded)

    assert restored.to_dict() == original.to_dict()


def test_json_export_rejects_non_json_metadata():
    record = ExperimentRecord(
        name="test",
        algorithm="baseline",
        seed=0,
        parameters={"unsupported": object()},
    )

    with pytest.raises(TypeError):
        record.to_json()


def test_from_dict_rejects_unexpected_fields():
    record = ExperimentRecord("test", "baseline", 0).to_dict()
    record["unexpected"] = True

    with pytest.raises(ValueError):
        ExperimentRecord.from_dict(record)
