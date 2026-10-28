"""Checks for the 3D robot studio, model import, and obstacle scenes."""

from io import BytesIO
import zipfile

import numpy as np
import pytest

pytest.importorskip("mujoco")
import mujoco

from physics.robot_scene import (
    DEFAULT_ARM_XML,
    Obstacle,
    archive_model_candidates,
    build_robot_scene,
    inspect_robot_model,
    render_robot_scene,
)


URDF = b'''<?xml version="1.0"?>
<robot name="uploaded_arm">
  <link name="base"/>
  <link name="arm">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="arm"/>
    <origin xyz="0 0 0.2"/><axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="1.5" effort="2" velocity="2"/>
  </joint>
</robot>'''


def test_default_arm_compiles_with_obstacles_target_and_3d_renderer():
    model, data = build_robot_scene(
        obstacles=(Obstacle("box", (0.55, 0.3, 0.3), (0.1, 0.1, 0.2)),)
    )
    info = inspect_robot_model(model)
    assert info["joint_count"] == 3
    assert info["actuator_count"] == 3
    assert "base_yaw" in info["joints"]
    assert "end_effector" in info["sites"]
    assert info["geometry_count"] >= 6

    frame = render_robot_scene(model, data, width=320, height=240)
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8
    assert frame.std() > 0
    route_frame = render_robot_scene(
        model, data, width=320, height=240,
        path_waypoints=np.asarray([[0.4, 0.0, 0.5], [0.6, 0.2, 0.6], [0.8, 0.0, 0.7]]),
    )
    assert route_frame.shape == frame.shape


def test_urdf_model_upload_compiles_with_environment():
    model, _ = build_robot_scene(model_filename="arm.urdf", model_content=URDF)
    info = inspect_robot_model(model)
    assert info["model_name"] == "uploaded_arm"
    assert "shoulder" in info["joints"]
    assert info["actuator_count"] == 1
    assert info["actuators"] == ["physai_virtual_motor_shoulder"]
    assert "physai_end_effector_arm" in info["sites"]


def test_virtual_motors_can_be_disabled_for_an_uploaded_model():
    model, _ = build_robot_scene(
        model_filename="arm.urdf", model_content=URDF, auto_motorize=False
    )
    assert inspect_robot_model(model)["actuator_count"] == 0


def test_zip_bundle_model_entry_is_detected_and_loaded():
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("robot/arm.urdf", URDF)
        archive.writestr("robot/README.txt", "example robot")
    bundle = buffer.getvalue()
    assert archive_model_candidates("arm_bundle.zip", bundle) == ["robot/arm.urdf"]
    model, _ = build_robot_scene(
        model_filename="arm_bundle.zip",
        model_content=bundle,
        model_entry="robot/arm.urdf",
    )
    assert inspect_robot_model(model)["model_name"] == "uploaded_arm"
    assert inspect_robot_model(model)["actuator_count"] == 1


def test_mjz_asset_bundle_loads_from_virtual_filesystem():
    spec = mujoco.MjSpec.from_string(DEFAULT_ARM_XML)
    spec.compile()
    bundle = BytesIO()
    spec.to_zip(bundle)
    model, _ = build_robot_scene(model_filename="arm.mjz", model_content=bundle.getvalue())
    assert inspect_robot_model(model)["joint_count"] == 3


def test_unsafe_zip_paths_are_rejected():
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../escape.xml", "<mujoco/>")
    with pytest.raises(ValueError, match="unsafe"):
        archive_model_candidates("bad.zip", buffer.getvalue())


def test_invalid_obstacle_dimensions_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        build_robot_scene(obstacles=(Obstacle("sphere", (0, 0, 0.5), (0, 0, 0)),))
