"""MuJoCo robot-scene construction, safe model import, and inspection."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

import numpy as np


MAX_MODEL_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_ARCHIVE_UNPACKED_BYTES = 250 * 1024 * 1024
MAX_ARCHIVE_FILES = 5000
MODEL_SUFFIXES = {".xml", ".mjcf", ".urdf", ".mjz"}

DEFAULT_ARM_XML = """
<mujoco model="physai_3d_three_joint_arm">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
  <visual>
    <global offwidth="1280" offheight="800"/>
    <quality shadowsize="2048"/><headlight ambient="0.42 0.46 0.56" diffuse="0.72 0.76 0.84" specular="0.28 0.28 0.32"/>
  </visual>
  <worldbody>
    <light pos="1.5 -2 3.2" dir="-0.3 0.35 -1" diffuse="0.95 0.94 0.9" specular="0.35 0.35 0.35"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.16 0.20 0.28 1" material="floor_grid"/>
    <body name="arm_base" pos="0 0 0.12">
      <geom name="base_visual" type="cylinder" size="0.12 0.12" mass="1.0" rgba="0.24 0.36 0.54 1" contype="0" conaffinity="0"/>
      <body name="base_yaw_link" pos="0 0 0.12">
        <joint name="base_yaw" type="hinge" axis="0 0 1" range="-3.0 3.0" damping="0.12" armature="0.03"/>
        <geom name="yaw_hub" type="cylinder" size="0.085 0.08" mass="0.35" rgba="0.32 0.45 0.64 1" contype="0" conaffinity="0"/>
        <body name="shoulder_link" pos="0 0 0.1">
          <joint name="shoulder" type="hinge" axis="0 1 0" range="-2.8 2.8" damping="0.08" armature="0.02"/>
          <geom name="upper_arm" type="capsule" fromto="0 0 0 0.55 0 0" size="0.05" mass="0.8" rgba="0.13 0.68 0.84 1"/>
          <body name="elbow_link" pos="0.55 0 0">
            <joint name="elbow" type="hinge" axis="0 1 0" range="-2.8 2.8" damping="0.06" armature="0.015"/>
            <geom name="forearm" type="capsule" fromto="0 0 0 0.45 0 0" size="0.042" mass="0.55" rgba="0.96 0.62 0.16 1"/>
            <site name="end_effector" pos="0.45 0 0" size="0.025" rgba="0.25 0.95 0.55 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <contact><exclude body1="shoulder_link" body2="elbow_link"/></contact>
  <asset><texture name="floor_grid_tex" type="2d" builtin="checker" width="512" height="512" rgb1="0.19 0.23 0.31" rgb2="0.12 0.15 0.22"/><material name="floor_grid" texture="floor_grid_tex" texrepeat="8 8" reflectance="0.12"/></asset>
  <actuator>
    <motor name="base_yaw_motor" joint="base_yaw" ctrlrange="-4.0 4.0" forcerange="-4.0 4.0"/>
    <motor name="shoulder_motor" joint="shoulder" ctrlrange="-10.0 10.0" forcerange="-10.0 10.0"/>
    <motor name="elbow_motor" joint="elbow" ctrlrange="-5.0 5.0" forcerange="-5.0 5.0"/>
  </actuator>
</mujoco>
"""


@dataclass(frozen=True)
class Obstacle:
    """A static obstacle in world coordinates; sizes are half-extents/radii."""

    shape: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]


def _mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo is required to build a robot scene.") from exc
    return mujoco


def archive_model_candidates(filename: str, content: bytes) -> list[str]:
    """Return model file paths in a safe, bounded ZIP model bundle."""
    if len(content) > MAX_MODEL_UPLOAD_BYTES:
        raise ValueError("Model upload exceeds the 100 MB limit.")
    try:
        with zipfile.ZipFile(BytesIO(content)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ARCHIVE_FILES:
                raise ValueError("Model archive contains too many files.")
            unpacked = sum(entry.file_size for entry in entries)
            if unpacked > MAX_ARCHIVE_UNPACKED_BYTES:
                raise ValueError("Model archive expands beyond the 250 MB limit.")
            candidates = []
            for entry in entries:
                raw_name = entry.filename
                path = PurePosixPath(raw_name)
                mode = entry.external_attr >> 16
                if (
                    path.is_absolute()
                    or "\\" in raw_name
                    or any(":" in part for part in path.parts)
                    or any(part in {"", ".", ".."} for part in path.parts)
                    or (mode and stat.S_ISLNK(mode))
                ):
                    raise ValueError("Model archive contains an unsafe file path.")
                if entry.is_dir():
                    continue
                if entry.file_size > 100 * 1024 * 1024:
                    raise ValueError("A file in the model archive exceeds 100 MB.")
                if entry.file_size > 0 and entry.compress_size == 0:
                    raise ValueError("Model archive contains an invalid compressed file.")
                if entry.compress_size and entry.file_size / entry.compress_size > 1000:
                    raise ValueError("Model archive has an unsafe compression ratio.")
                if Path(path.name).suffix.lower() in MODEL_SUFFIXES:
                    candidates.append(raw_name)
    except zipfile.BadZipFile as exc:
        raise ValueError("The uploaded ZIP file is invalid or damaged.") from exc
    if not candidates:
        raise ValueError("No MJCF, URDF, or MJZ model file was found in the archive.")
    return sorted(candidates)


def supported_model_candidates(filename: str, content: bytes) -> list[str]:
    """List selectable model entries for either a single file or ZIP bundle."""
    if len(content) > MAX_MODEL_UPLOAD_BYTES:
        raise ValueError("Model upload exceeds the 100 MB limit.")
    if Path(filename).suffix.lower() == ".zip":
        return archive_model_candidates(filename, content)
    suffix = Path(filename).suffix.lower()
    if suffix == ".mjz":
        # MJZ is also a ZIP archive; validate its contents before MuJoCo reads it.
        archive_model_candidates(filename, content)
        return [Path(filename).name]
    if suffix not in MODEL_SUFFIXES:
        raise ValueError("Choose an MJCF (.xml/.mjcf), URDF (.urdf), or MJZ (.mjz) model.")
    return [Path(filename).name]


def _default_spec():
    mujoco = _mujoco()
    return mujoco.MjSpec.from_string(DEFAULT_ARM_XML)


def _add_missing_scalar_motors(spec, mujoco) -> list[str]:
    """Give unactuated hinge/slide joints bounded virtual motors for simulation."""
    joint_types = {
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    }
    actuated = {
        actuator.target
        for actuator in spec.actuators
        if int(actuator.trntype) == int(mujoco.mjtTrn.mjTRN_JOINT)
    }
    added = []
    for index, joint in enumerate(spec.joints):
        if int(joint.type) not in joint_types or not joint.name or joint.name in actuated:
            continue
        name = f"physai_virtual_motor_{joint.name}"
        max_force = 2.0 if int(joint.type) == int(mujoco.mjtJoint.mjJNT_HINGE) else 10.0
        spec.add_actuator(
            name=name,
            type=mujoco.mjtActuator.mjACTUATOR_MOTOR,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=joint.name,
            ctrllimited=mujoco.mjtLimited.mjLIMITED_TRUE,
            ctrlrange=[-max_force, max_force],
            forcelimited=mujoco.mjtLimited.mjLIMITED_TRUE,
            forcerange=[-max_force, max_force],
        )
        added.append(name)
    return added


def _load_uploaded_model(
    filename: str,
    content: bytes,
    model_entry: str | None,
    obstacles: tuple[Obstacle, ...],
    target_position: tuple[float, float, float],
    auto_motorize: bool,
    auto_site_offset: tuple[float, float, float],
):
    mujoco = _mujoco()
    if len(content) > MAX_MODEL_UPLOAD_BYTES:
        raise ValueError("Model upload exceeds the 100 MB limit.")
    suffix = Path(filename).suffix.lower()
    with tempfile.TemporaryDirectory(
        prefix=".physai_robot_model_", dir=Path.cwd()
    ) as temp_dir:
        root = Path(temp_dir)
        if suffix == ".zip":
            candidates = archive_model_candidates(filename, content)
            chosen = model_entry or (candidates[0] if len(candidates) == 1 else None)
            if chosen not in candidates:
                raise ValueError("Choose one model file from the uploaded ZIP bundle.")
            with zipfile.ZipFile(BytesIO(content)) as archive:
                for relative in archive.namelist():
                    item = PurePosixPath(relative)
                    if relative.endswith("/"):
                        (root / Path(*item.parts)).mkdir(parents=True, exist_ok=True)
                    else:
                        target = root / Path(*item.parts)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(relative) as source, target.open("wb") as destination:
                            shutil.copyfileobj(source, destination)
            model_path = root / Path(*PurePosixPath(chosen).parts)
        elif suffix == ".mjz":
            candidates = archive_model_candidates(filename, content)
            main_file = "model.xml" if "model.xml" in candidates else (
                candidates[0] if len(candidates) == 1 else None
            )
            if main_file is None:
                raise ValueError("This MJZ bundle has multiple possible model files; repackage it with one main model.xml.")
            with zipfile.ZipFile(BytesIO(content)) as archive, mujoco.MjVfs() as vfs:
                for relative in archive.namelist():
                    if not relative.endswith("/"):
                        vfs[relative] = archive.read(relative)
                spec = mujoco.MjSpec.from_file(main_file, vfs=vfs)
                if auto_motorize:
                    _add_missing_scalar_motors(spec, mujoco)
                _add_scene_elements(spec, mujoco, obstacles, target_position, auto_site_offset)
                return spec.compile(vfs=vfs)
        else:
            if suffix not in MODEL_SUFFIXES:
                raise ValueError("Choose an MJCF (.xml/.mjcf), URDF (.urdf), or MJZ (.mjz) model.")
            if suffix == ".mjz":
                archive_model_candidates(filename, content)
            model_path = root / Path(filename).name
            model_path.write_bytes(content)
        spec = mujoco.MjSpec.from_file(str(model_path))
        if auto_motorize:
            _add_missing_scalar_motors(spec, mujoco)
        _add_scene_elements(spec, mujoco, obstacles, target_position, auto_site_offset)
        # Compile while referenced mesh/texture assets still exist. Compiled
        # MuJoCo models own the asset data needed by subsequent renders.
        return spec.compile()


def _add_ground_if_needed(spec, mujoco) -> None:
    if any(int(geom.type) == int(mujoco.mjtGeom.mjGEOM_PLANE) for geom in spec.geoms):
        return
    spec.worldbody.add_geom(
        name="physai_ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2.5, 2.5, 0.1],
        rgba=[0.12, 0.17, 0.25, 1.0],
    )


def _add_environment_geom(spec, mujoco, obstacle: Obstacle, index: int) -> None:
    if obstacle.shape not in {"box", "sphere", "cylinder", "capsule"}:
        raise ValueError(f"Unsupported obstacle type: {obstacle.shape}")
    position = np.asarray(obstacle.position, dtype=float)
    size = np.asarray(obstacle.size, dtype=float)
    if position.shape != (3,) or size.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(size).all():
        raise ValueError("Obstacle position and size must contain three finite values each.")
    if np.any(size <= 0.0):
        raise ValueError("Obstacle dimensions must be positive.")
    shape_type = {
        "box": mujoco.mjtGeom.mjGEOM_BOX,
        "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
        "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
        "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    }[obstacle.shape]
    dimensions = size.tolist() if obstacle.shape == "box" else [float(size[0]), float(size[1]), 0.0]
    used_names = {geom.name for geom in spec.geoms}
    name = f"physai_obstacle_{index + 1}"
    suffix = 1
    while name in used_names:
        name = f"physai_obstacle_{index + 1}_{suffix}"
        suffix += 1
    spec.worldbody.add_geom(
        name=name,
        type=shape_type,
        pos=position.tolist(),
        size=dimensions,
        rgba=[0.92, 0.25, 0.28, 1.0],
        contype=1,
        conaffinity=1,
    )


def _add_target_marker(spec, mujoco, target_position: tuple[float, float, float]) -> None:
    target = np.asarray(target_position, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("Target position must contain three finite coordinates.")
    used_names = {geom.name for geom in spec.geoms}
    name = "physai_target_marker"
    suffix = 1
    while name in used_names:
        name = f"physai_target_marker_{suffix}"
        suffix += 1
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=target.tolist(),
        size=[0.055, 0.0, 0.0],
        rgba=[0.25, 0.95, 0.55, 0.85],
        contype=0,
        conaffinity=0,
    )


def _add_scene_elements(spec, mujoco, obstacles, target_position, auto_site_offset) -> None:
    _add_ground_if_needed(spec, mujoco)
    if not spec.sites:
        # URDF normally has links and joints but no MuJoCo sites. Add selectable
        # tool-point candidates at articulated terminal link origins.
        used_names = {body.name for body in spec.bodies}
        for body in spec.bodies:
            if body.name == "world" or body.bodies or not body.joints:
                continue
            site_name = f"physai_end_effector_{body.name or 'link'}"
            suffix = 1
            while site_name in used_names:
                site_name = f"physai_end_effector_{body.name or 'link'}_{suffix}"
                suffix += 1
            used_names.add(site_name)
            offset = np.asarray(auto_site_offset, dtype=float)
            if offset.shape != (3,) or not np.isfinite(offset).all():
                raise ValueError("Automatic tool-point offset must contain three finite coordinates.")
            body.add_site(
                name=site_name,
                pos=offset.tolist(),
                size=0.04,
                rgba=[0.25, 0.95, 0.55, 1.0],
            )
    for index, obstacle in enumerate(obstacles):
        _add_environment_geom(spec, mujoco, obstacle, index)
    _add_target_marker(spec, mujoco, target_position)


def build_robot_scene(
    model_filename: str | None = None,
    model_content: bytes | None = None,
    model_entry: str | None = None,
    obstacles: tuple[Obstacle, ...] = (),
    target_position: tuple[float, float, float] = (0.65, 0.0, 0.7),
    auto_motorize: bool = True,
    auto_site_offset: tuple[float, float, float] = (0.1, 0.0, 0.0),
):
    """Compile the built-in arm or an uploaded MJCF/URDF/MJZ model and scene."""
    mujoco = _mujoco()
    if (model_filename is None) != (model_content is None):
        raise ValueError("Model filename and content must be provided together.")
    if model_filename is None:
        spec = _default_spec()
        _add_scene_elements(spec, mujoco, obstacles, target_position, auto_site_offset)
        model = spec.compile()
    else:
        model = _load_uploaded_model(
            model_filename, model_content, model_entry, obstacles, target_position,
            auto_motorize, auto_site_offset,
        )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def inspect_robot_model(model) -> dict[str, object]:
    """Return names and counts used by the model-selection UI."""
    mujoco = _mujoco()

    def names(object_type, count: int) -> list[str]:
        return [
            mujoco.mj_id2name(model, object_type, i) or f"unnamed_{i}"
            for i in range(count)
        ]

    joint_names = names(mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    actuator_names = names(mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    site_names = names(mujoco.mjtObj.mjOBJ_SITE, model.nsite)
    return {
        "model_name": model.names[0:model.names.find(b"\0")].decode("utf-8", errors="replace") if b"\0" in model.names else "MuJoCo robot",
        "body_count": int(model.nbody),
        "joint_count": int(model.njnt),
        "actuator_count": int(model.nu),
        "geometry_count": int(model.ngeom),
        "joints": joint_names,
        "actuators": actuator_names,
        "sites": site_names,
        "joint_types": [int(value) for value in model.jnt_type],
        "joint_qpos_addresses": [int(value) for value in model.jnt_qposadr],
        "joint_ranges": [tuple(map(float, value)) for value in model.jnt_range],
        "joint_limited": [bool(value) for value in model.jnt_limited],
    }


def render_robot_scene(
    model,
    data,
    width: int = 1280,
    height: int = 760,
    path_waypoints: np.ndarray | None = None,
    camera: mujoco.MjvCamera | None = None,
) -> np.ndarray:
    """Render a 3D MuJoCo frame for display in the Streamlit workspace.

    Args:
        camera: Optional caller-configured ``MjvCamera``. When omitted, the
            default studio camera pose is used.
    """
    mujoco = _mujoco()
    with mujoco.Renderer(model, width=width, height=height) as renderer:
        if camera is None:
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.lookat[:] = [0.35, 0.0, 0.55]
            camera.distance = 2.5
            camera.azimuth = 135
            camera.elevation = -22
        renderer.update_scene(data, camera=camera)
        if path_waypoints is not None:
            points = np.asarray(path_waypoints, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                raise ValueError("Path overlay must contain finite 3D waypoints.")
            scene = renderer.scene
            color = np.asarray([0.08, 0.9, 0.95, 1.0], dtype=np.float32)
            identity = np.eye(3, dtype=float).ravel()
            for first, second in zip(points[:-1], points[1:]):
                if scene.ngeom >= scene.maxgeom:
                    break
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    np.zeros(3, dtype=float),
                    np.zeros(3, dtype=float),
                    identity,
                    color,
                )
                mujoco.mjv_connector(
                    geom,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    0.012,
                    first,
                    second,
                )
                scene.ngeom += 1
        return renderer.render()
