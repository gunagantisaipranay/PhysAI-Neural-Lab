"""Train a supervised joint-target model from MuJoCo-generated arm samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class RobotIKTrainingResult:
    """Trained joint-target predictor and held-out task metrics."""

    policy: object
    feature_names: list[str]
    joint_names: list[str]
    joint_rmse: float
    joint_mae: float
    position_rmse_m: float
    position_mae_m: float
    success_rate: float
    predicted_collision_rate: float
    train_samples: int
    test_samples: int
    random_seed: int
    loss_curve: list[float]
    example_start: list[float]
    example_target: list[float]
    example_prediction: list[float]
    example_expert: list[float]
    joint_lows: list[float]
    joint_highs: list[float]
    heldout_tasks: list[dict[str, list[float]]]


def _candidate_joint_ids(model, mujoco) -> list[int]:
    actuator_joints = {
        int(model.actuator_trnid[index, 0])
        for index in range(model.nu)
        if int(model.actuator_trntype[index]) == int(mujoco.mjtTrn.mjTRN_JOINT)
        and int(model.actuator_trnid[index, 0]) >= 0
    }
    scalar = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
    return [joint_id for joint_id in actuator_joints if int(model.jnt_type[joint_id]) in scalar]


def train_ik_policy(
    model,
    site_name: str,
    joint_ids: list[int],
    *,
    samples: int = 1500,
    epochs: int = 250,
    seed: int = 17,
) -> RobotIKTrainingResult:
    """Train an MLP mapping current joints + target XYZ to joint setpoints.

    The synthetic teacher pairs nearby start and goal configurations and uses
    MuJoCo forward kinematics to label target points. A held-out set measures
    joint prediction and end-effector position errors.
    """
    import mujoco

    if samples < 200:
        raise ValueError("Use at least 200 generated samples for training.")
    if epochs < 1:
        raise ValueError("Training epochs must be positive.")
    if not joint_ids or len(set(joint_ids)) != len(joint_ids):
        raise ValueError("Select one or more unique actuated scalar joints.")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"End-effector site '{site_name}' does not exist in this robot.")
    valid = set(_candidate_joint_ids(model, mujoco))
    if not set(joint_ids).issubset(valid):
        raise ValueError("Training joints must be actuated hinge or slide joints.")

    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    qadr = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=int)
    lows, highs = [], []
    for joint_id in joint_ids:
        if not model.jnt_limited[joint_id]:
            raise ValueError("Selected joints need finite model limits before safe training can run.")
        low, high = model.jnt_range[joint_id]
        lows.append(float(low))
        highs.append(float(high))
    lows = np.asarray(lows)
    highs = np.asarray(highs)
    if np.any(highs - lows < 1e-3):
        raise ValueError("Selected joint ranges are too narrow for useful training.")

    # Reject placeholder/body-origin sites that would give a vacuous target score.
    probe = mujoco.MjData(model)
    mujoco.mj_resetData(model, probe)
    probe.qpos[qadr] = (lows + highs) / 2.0
    mujoco.mj_forward(model, probe)
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacSite(model, probe, jacp, jacr, site_id)
    dofs = [int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids]
    if np.linalg.norm(jacp[:, dofs]) < 1e-7:
        raise ValueError(
            "This end-effector site does not move with the selected joints. "
            "Choose a site farther along the tool link or add a tool-point site to the robot model."
        )

    features = np.empty((samples, len(joint_ids) + 3), dtype=float)
    labels = np.empty((samples, len(joint_ids)), dtype=float)
    positions = np.empty((samples, 3), dtype=float)
    for row in range(samples):
        for _attempt in range(100):
            start = rng.uniform(lows * 0.65 + highs * 0.35, lows * 0.35 + highs * 0.65)
            mujoco.mj_resetData(model, data)
            data.qpos[qadr] = start
            mujoco.mj_forward(model, data)
            if data.ncon:
                continue
            delta = rng.normal(0.0, (highs - lows) * 0.035)
            goal = np.clip(start + delta, lows, highs)
            data.qpos[qadr] = goal
            mujoco.mj_forward(model, data)
            if data.ncon:
                continue
            target = data.site_xpos[site_id].copy()
            features[row] = np.concatenate([start, target])
            # Learn a local joint correction so the same Cartesian target has fewer
            # ambiguous inverse-kinematics branches than an absolute joint label.
            labels[row] = goal - start
            positions[row] = target
            break
        else:
            raise ValueError(
                "Could not generate enough collision-free training poses. "
                "Move or remove obstacles, or reduce the selected joint range."
            )

    x_train, x_test, y_train, y_test, target_train, target_test = train_test_split(
        features, labels, positions, test_size=0.2, random_state=seed
    )
    policy = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation="tanh",
            solver="adam",
            learning_rate_init=0.002,
            max_iter=epochs,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=seed,
        ),
    )
    fit_labels = y_train.ravel() if len(joint_ids) == 1 else y_train
    policy.fit(x_train, fit_labels)
    predicted_delta = np.asarray(policy.predict(x_test), dtype=float).reshape(len(x_test), len(joint_ids))
    predicted = np.clip(x_test[:, : len(joint_ids)] + predicted_delta, lows, highs)
    expert_goal = np.clip(x_test[:, : len(joint_ids)] + y_test, lows, highs)
    joint_rmse = float(np.sqrt(mean_squared_error(expert_goal, predicted)))
    joint_mae = float(mean_absolute_error(expert_goal, predicted))

    position_errors = np.empty(len(x_test), dtype=float)
    prediction_collisions = np.zeros(len(x_test), dtype=bool)
    for row, predicted_goal in enumerate(predicted):
        mujoco.mj_resetData(model, data)
        data.qpos[qadr] = predicted_goal
        mujoco.mj_forward(model, data)
        position_errors[row] = np.linalg.norm(data.site_xpos[site_id] - target_test[row])
        prediction_collisions[row] = data.ncon > 0
    position_rmse = float(np.sqrt(np.mean(position_errors**2)))
    position_mae = float(np.mean(position_errors))
    return RobotIKTrainingResult(
        policy=policy,
        feature_names=[*(f"current_{name}" for name in [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or str(jid) for jid in joint_ids]), "target_x", "target_y", "target_z"],
        joint_names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}" for jid in joint_ids],
        joint_rmse=joint_rmse,
        joint_mae=joint_mae,
        position_rmse_m=position_rmse,
        position_mae_m=position_mae,
        success_rate=float(np.mean(position_errors < 0.08)),
        predicted_collision_rate=float(np.mean(prediction_collisions)),
        train_samples=len(x_train),
        test_samples=len(x_test),
        random_seed=seed,
        loss_curve=list(map(float, policy[-1].loss_curve_)),
        example_start=list(map(float, x_test[0, : len(joint_ids)])),
        example_target=list(map(float, target_test[0])),
        example_prediction=list(map(float, predicted[0])),
        example_expert=list(map(float, expert_goal[0])),
        joint_lows=list(map(float, lows)),
        joint_highs=list(map(float, highs)),
        heldout_tasks=[
            {
                "start": list(map(float, x_test[row, : len(joint_ids)])),
                "target": list(map(float, target_test[row])),
                "expert_goal": list(map(float, expert_goal[row])),
            }
            for row in range(min(8, len(x_test)))
        ],
    )


def predict_joint_target(result: RobotIKTrainingResult, current_q: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
    """Predict joint setpoints from current selected joints and target XYZ."""
    vector = np.concatenate([np.asarray(current_q, dtype=float), np.asarray(target_xyz, dtype=float)])
    if vector.shape != (len(result.joint_names) + 3,) or not np.isfinite(vector).all():
        raise ValueError("Policy input must contain finite current joint values and target XYZ.")
    delta = np.asarray(result.policy.predict(vector.reshape(1, -1))[0], dtype=float)
    return np.clip(
        np.asarray(current_q, dtype=float) + delta,
        np.asarray(result.joint_lows),
        np.asarray(result.joint_highs),
    )
