"""Tests for the standalone MuJoCo manipulation scene."""

import numpy as np
import pytest

pytest.importorskip("mujoco")
import mujoco

from physics.robot_manipulation import build_manipulation_scene
from physics.robot_scene import build_robot_scene, render_robot_scene


def _id(model, object_type, name):
    return mujoco.mj_name2id(model, object_type, name)


def test_manipulation_scene_compiles_and_runs_forward_kinematics():
    model, data, config = build_manipulation_scene()

    assert data is not None
    mujoco.mj_forward(model, data)
    assert config.end_effector_site == "end_effector"
    assert np.isfinite(data.site_xpos).all()


def test_original_arm_joints_and_actuators_are_preserved():
    model, _, config = build_manipulation_scene()

    expected = {
        "base_yaw": (mujoco.mjtJoint.mjJNT_HINGE, (-3.0, 3.0)),
        "shoulder": (mujoco.mjtJoint.mjJNT_HINGE, (-2.8, 2.8)),
        "elbow": (mujoco.mjtJoint.mjJNT_HINGE, (-2.8, 2.8)),
    }
    for name in config.arm_joints:
        joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert model.jnt_type[joint_id] == expected[name][0]
        np.testing.assert_allclose(model.jnt_range[joint_id], expected[name][1])
    for name in config.arm_actuators:
        assert _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) >= 0


def test_gripper_has_two_bounded_slide_joints_and_actuators():
    model, _, config = build_manipulation_scene()

    assert len(config.finger_joints) == 2
    assert len(config.finger_actuators) == 2
    for joint_name, actuator_name in zip(config.finger_joints, config.finger_actuators):
        joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        assert model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_SLIDE
        assert model.jnt_limited[joint_id]
        assert model.jnt_range[joint_id, 0] < model.jnt_range[joint_id, 1]
        assert model.actuator_ctrllimited[actuator_id]
        assert model.actuator_forcelimited[actuator_id]
        assert model.actuator_ctrlrange[actuator_id, 0] < model.actuator_ctrlrange[actuator_id, 1]
        assert model.actuator_forcerange[actuator_id, 0] < model.actuator_forcerange[actuator_id, 1]


def test_finger_pads_payload_and_destination_have_required_physics():
    model, _, config = build_manipulation_scene()

    for name in config.finger_pad_geoms:
        geom_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert model.geom_contype[geom_id] != 0
        assert model.geom_conaffinity[geom_id] != 0

    payload_joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, config.payload_joint)
    payload_body_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, config.payload_body)
    payload_geom_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, config.payload_geom)
    destination_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, config.destination_marker)

    assert model.jnt_type[payload_joint_id] == mujoco.mjtJoint.mjJNT_FREE
    assert np.isfinite(model.body_mass[payload_body_id])
    assert model.body_mass[payload_body_id] > 0.0
    assert model.geom_contype[payload_geom_id] != 0
    assert model.geom_conaffinity[payload_geom_id] != 0
    assert model.geom_contype[destination_id] == 0
    assert model.geom_conaffinity[destination_id] == 0


def test_manipulation_scene_renders_and_default_scene_regression_is_preserved():
    model, data, _ = build_manipulation_scene()
    frame = render_robot_scene(model, data, width=320, height=240)
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8
    assert frame.std() > 0

    default_model, _ = build_robot_scene()
    default_joint_names = [
        mujoco.mj_id2name(default_model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(default_model.njnt)
    ]
    assert default_model.njnt == 3
    assert default_joint_names == ["base_yaw", "shoulder", "elbow"]
