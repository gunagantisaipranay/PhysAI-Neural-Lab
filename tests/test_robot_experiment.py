
"""Tests for robot IK experiment metadata."""

import pytest

from machine_learning.robot_experiment import build_ik_experiment_record


def test_ik_record_captures_configuration_and_metrics():
    record = build_ik_experiment_record(
        robot_model="three-joint-arm",
        end_effector_site="end_effector",
        controlled_joints=["shoulder", "elbow", "wrist"],
        train_samples=400,
        test_samples=100,
        seed=23,
        joint_rmse=0.12,
        joint_mae=0.08,
        position_rmse_m=0.03,
        position_mae_m=0.02,
        success_rate=0.85,
        predicted_collision_rate=0.04,
        parameters={"epochs": 100},
    )

    data = record.to_dict()
    assert data["algorithm"] == "MLPRegressor"
    assert data["seed"] == 23
    assert data["parameters"]["end_effector_site"] == "end_effector"
    assert data["parameters"]["epochs"] == 100
    assert data["parameters"]["controlled_joints"] == [
        "shoulder", "elbow", "wrist"
    ]
    assert data["dataset"]["train_samples"] == 400
    assert data["dataset"]["test_samples"] == 100
    assert data["metrics"]["position_rmse_m"] == pytest.approx(0.03)
    assert data["metrics"]["success_rate_within_8cm"] == pytest.approx(0.85)


def test_ik_record_rejects_empty_robot_model():
    with pytest.raises(ValueError, match="Robot model"):
        build_ik_experiment_record(
            robot_model="",
            end_effector_site="end_effector",
            controlled_joints=["shoulder"],
            train_samples=80,
            test_samples=20,
            seed=1,
            joint_rmse=0.1,
            joint_mae=0.1,
            position_rmse_m=0.1,
            position_mae_m=0.1,
            success_rate=0.5,
            predicted_collision_rate=0.0,
        )


def test_ik_record_rejects_empty_dataset_split():
    with pytest.raises(ValueError, match="sample counts"):
        build_ik_experiment_record(
            robot_model="arm",
            end_effector_site="end_effector",
            controlled_joints=["shoulder"],
            train_samples=0,
            test_samples=20,
            seed=1,
            joint_rmse=0.1,
            joint_mae=0.1,
            position_rmse_m=0.1,
            position_mae_m=0.1,
            success_rate=0.5,
            predicted_collision_rate=0.0,
        )
