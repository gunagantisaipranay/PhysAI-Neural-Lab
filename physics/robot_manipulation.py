"""Standalone MuJoCo manipulation scene with a parallel-jaw gripper.

The destination marker is visual-only and non-colliding in this milestone.
Placement and grasp execution are intentionally deferred to later milestones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import mujoco


MANIPULATION_SCENE_XML: Final[str] = """
<mujoco model="physai_manipulation_arm">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
  <visual>
    <global offwidth="1280" offheight="800"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.50 0.54 0.62" diffuse="0.78 0.80 0.86" specular="0.26 0.26 0.30"/>
  </visual>
  <default>
    <joint damping="0.08"/>
    <geom friction="0.8 0.1 0.02" solref="0.01 1" solimp="0.9 0.95 0.001"/>
  </default>
  <worldbody>
    <light pos="1.5 -2 3.2" dir="-0.3 0.35 -1" diffuse="0.95 0.94 0.9" specular="0.35 0.35 0.35"/>
    <light name="key_light" pos="2.5 -2.2 3.8" dir="-0.4 -0.35 -1.0" diffuse="1.0 0.95 0.88" specular="0.3 0.3 0.3"/>
    <light name="fill_light" pos="-2.2 -1.2 2.2" dir="0.3 0.25 -1.0" diffuse="0.42 0.52 0.72" specular="0.16 0.16 0.16"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.16 0.20 0.28 1" material="floor_grid"/>
    <body name="arm_base" pos="0 0 0.12">
      <geom name="base_visual" type="cylinder" size="0.12 0.12" mass="1.0" rgba="0.17 0.30 0.47 1" contype="0" conaffinity="0"/>
      <geom name="base_foot" type="box" pos="0 0 -0.05" size="0.20 0.20 0.04" mass="0" rgba="0.10 0.14 0.20 1" contype="0" conaffinity="0"/>
      <body name="base_yaw_link" pos="0 0 0.12">
        <joint name="base_yaw" type="hinge" axis="0 0 1" range="-3.0 3.0" damping="0.12" armature="0.03"/>
        <geom name="yaw_hub" type="cylinder" size="0.085 0.08" mass="0.35" rgba="0.24 0.36 0.55 1" contype="0" conaffinity="0"/>
        <body name="shoulder_link" pos="0 0 0.1">
          <joint name="shoulder" type="hinge" axis="0 1 0" range="-2.8 2.8" damping="0.08" armature="0.02"/>
          <geom name="shoulder_cap" type="sphere" pos="0 0 0" size="0.07" mass="0" rgba="0.16 0.30 0.47 1" contype="0" conaffinity="0"/>
          <geom name="upper_arm" type="capsule" fromto="0 0 0 0.55 0 0" size="0.05" mass="0.8" rgba="0.07 0.52 0.72 1"/>
          <body name="elbow_link" pos="0.55 0 0">
            <joint name="elbow" type="hinge" axis="0 1 0" range="-2.8 2.8" damping="0.06" armature="0.015"/>
            <geom name="forearm" type="capsule" fromto="0 0 0 0.45 0 0" size="0.042" mass="0.55" rgba="0.93 0.52 0.12 1"/>
            <geom name="wrist_flange" type="box" pos="0.45 0 0" size="0.05 0.05 0.05" mass="0" rgba="0.20 0.26 0.34 1" contype="0" conaffinity="0"/>
            <site name="end_effector" pos="0.45 0 0" size="0.025" rgba="0.25 0.95 0.55 1"/>
            <body name="gripper_base" pos="0.45 0 0">
              <geom name="gripper_mount" type="box" pos="0.05 0 0" size="0.06 0.09 0.045" mass="0.12" rgba="0.19 0.24 0.31 1"/>
              <site name="tool_center" pos="0.12 0 0" size="0.02" rgba="0.95 0.95 0.2 1"/>
              <body name="left_finger" pos="0.08 0.10 0">
                <joint name="left_finger_slide" type="slide" axis="0 1 0" range="-0.045 0.045" damping="0.12"/>
                <geom name="left_finger_cap" type="sphere" pos="0.04 0 0" size="0.03" mass="0" rgba="0.28 0.55 0.82 1" contype="0" conaffinity="0"/>
                <geom name="left_finger_link" type="box" pos="0.04 0 0" size="0.07 0.025 0.025" mass="0.06" rgba="0.28 0.55 0.82 1"/>
                <geom name="left_finger_pad" type="box" pos="0.105 0 0" size="0.018 0.028 0.035" mass="0.025" rgba="0.95 0.75 0.18 1" contype="1" conaffinity="1"/>
              </body>
              <body name="right_finger" pos="0.08 -0.10 0">
                <joint name="right_finger_slide" type="slide" axis="0 1 0" range="-0.045 0.045" damping="0.12"/>
                <geom name="right_finger_cap" type="sphere" pos="0.04 0 0" size="0.03" mass="0" rgba="0.28 0.55 0.82 1" contype="0" conaffinity="0"/>
                <geom name="right_finger_link" type="box" pos="0.04 0 0" size="0.07 0.025 0.025" mass="0.06" rgba="0.28 0.55 0.82 1"/>
                <geom name="right_finger_pad" type="box" pos="0.105 0 0" size="0.018 0.028 0.035" mass="0.025" rgba="0.95 0.75 0.18 1" contype="1" conaffinity="1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="payload" pos="0.65 0.42 0.08">
      <freejoint name="payload_free"/>
      <geom name="payload_geom" type="box" size="0.055 0.055 0.055" mass="0.12" rgba="0.92 0.25 0.28 1" contype="1" conaffinity="1"/>
    </body>
    <geom name="destination_marker" type="cylinder" pos="0.65 -0.42 0.012" size="0.13 0.012" rgba="0.2 0.9 0.45 0.95" contype="0" conaffinity="0"/>
  </worldbody>
  <contact>
    <exclude body1="shoulder_link" body2="elbow_link"/>
  </contact>
  <asset>
    <texture name="floor_grid_tex" type="2d" builtin="checker" width="512" height="512" rgb1="0.19 0.23 0.31" rgb2="0.12 0.15 0.22"/>
    <material name="floor_grid" texture="floor_grid_tex" texrepeat="8 8" reflectance="0.12"/>
  </asset>
  <actuator>
    <motor name="base_yaw_motor" joint="base_yaw" ctrlrange="-4.0 4.0" forcerange="-4.0 4.0"/>
    <motor name="shoulder_motor" joint="shoulder" ctrlrange="-10.0 10.0" forcerange="-10.0 10.0"/>
    <motor name="elbow_motor" joint="elbow" ctrlrange="-5.0 5.0" forcerange="-5.0 5.0"/>
    <motor name="left_finger_motor" joint="left_finger_slide" ctrlrange="-0.8 0.8" forcerange="-0.8 0.8"/>
    <motor name="right_finger_motor" joint="right_finger_slide" ctrlrange="-0.8 0.8" forcerange="-0.8 0.8"/>
  </actuator>
</mujoco>
"""


@dataclass(frozen=True)
class ManipulationSceneConfig:
    """Names of the physical components in the manipulation scene."""

    arm_joints: tuple[str, str, str] = ("base_yaw", "shoulder", "elbow")
    arm_actuators: tuple[str, str, str] = (
        "base_yaw_motor",
        "shoulder_motor",
        "elbow_motor",
    )
    finger_joints: tuple[str, str] = ("left_finger_slide", "right_finger_slide")
    finger_actuators: tuple[str, str] = ("left_finger_motor", "right_finger_motor")
    finger_pad_geoms: tuple[str, str] = ("left_finger_pad", "right_finger_pad")
    end_effector_site: str = "end_effector"
    tool_center_site: str = "tool_center"
    payload_body: str = "payload"
    payload_joint: str = "payload_free"
    payload_geom: str = "payload_geom"
    destination_marker: str = "destination_marker"


def _validate_scene(model: mujoco.MjModel, config: ManipulationSceneConfig) -> None:
    required = (
        (mujoco.mjtObj.mjOBJ_JOINT, (*config.arm_joints, *config.finger_joints)),
        (mujoco.mjtObj.mjOBJ_ACTUATOR, (*config.arm_actuators, *config.finger_actuators)),
        (mujoco.mjtObj.mjOBJ_SITE, (config.end_effector_site, config.tool_center_site)),
        (mujoco.mjtObj.mjOBJ_BODY, (config.payload_body,)),
        (mujoco.mjtObj.mjOBJ_GEOM, (*config.finger_pad_geoms, config.payload_geom, config.destination_marker)),
    )
    for object_type, names in required:
        missing = [name for name in names if mujoco.mj_name2id(model, object_type, name) < 0]
        if missing:
            raise ValueError(f"Manipulation scene is missing required MuJoCo components: {', '.join(missing)}")


def build_manipulation_scene() -> tuple[mujoco.MjModel, mujoco.MjData, ManipulationSceneConfig]:
    """Compile and initialize the standalone arm, gripper, payload, and marker scene."""
    config = ManipulationSceneConfig()
    try:
        model = mujoco.MjModel.from_xml_string(MANIPULATION_SCENE_XML)
        _validate_scene(model, config)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Could not build manipulation scene: {exc}") from exc
    return model, data, config
