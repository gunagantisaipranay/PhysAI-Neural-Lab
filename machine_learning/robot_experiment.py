
"""Build shared experiment records for robot inverse-kinematics evaluation."""

from __future__ import annotations

from machine_learning.experiment import ExperimentRecord


def build_ik_experiment_record(
    *,
    robot_model: str,
    end_effector_site: str,
    controlled_joints: list[str] | tuple[str, ...],
    train_samples: int,
    test_samples: int,
    seed: int,
    joint_rmse: float,
    joint_mae: float,
    position_rmse_m: float,
    position_mae_m: float,
    success_rate: float,
    predicted_collision_rate: float,
    parameters: dict | None = None,
) -> ExperimentRecord:
    """Create reproducibility metadata from an actual IK training result."""
    if not robot_model.strip():
        raise ValueError("Robot model identifier must not be empty.")
    if not end_effector_site.strip():
        raise ValueError("End-effector site must not be empty.")
    if not controlled_joints or any(
        not isinstance(name, str) or not name.strip()
        for name in controlled_joints
    ):
        raise ValueError("At least one valid controlled joint is required.")
    if train_samples <= 0 or test_samples <= 0:
        raise ValueError("Training and test sample counts must be positive.")

    return ExperimentRecord(
        name="robot-ik-evaluation",
        algorithm="MLPRegressor",
        seed=seed,
        parameters={
            "robot_model": robot_model,
            "end_effector_site": end_effector_site,
            "controlled_joints": list(controlled_joints),
            **(parameters or {}),
        },
        dataset={
            "type": "MuJoCo-generated inverse-kinematics examples",
            "train_samples": train_samples,
            "test_samples": test_samples,
        },
        metrics={
            "joint_rmse_model_units": joint_rmse,
            "joint_mae_model_units": joint_mae,
            "position_rmse_m": position_rmse_m,
            "position_mae_m": position_mae_m,
            "success_rate_within_8cm": success_rate,
            "predicted_collision_rate": predicted_collision_rate,
        },
    )
