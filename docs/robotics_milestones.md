# PhysAI Robotics Lab milestones

This is a laptop-only MuJoCo and Streamlit robotics project. The app is now
organized around model import, simulation scenes, supervised training, and
evaluation. No physical robot is required.

## Milestone 1 — Robot scene workspace

**Status: Complete**

- Built-in three-joint arm, offscreen 3D renderer, pose controls, target marker,
  and collision-enabled box/sphere/cylinder/capsule obstacles.
- Import MuJoCo MJCF, URDF, MJZ, and ZIP bundles with relative assets.
- Add optional bounded virtual motors to imported unactuated hinge/slide joints.
- Improved camera, lighting, and textured ground.

## Milestone 2 — Robotics algorithms and control validation

**Status: Validated foundation complete**

- Verify MuJoCo forward kinematics for known arm poses and use MuJoCo sites as
  the canonical end-effector representation.
- Select a named end-effector site and bounded, actuated scalar joints.
- Add a selectable approximate tool-point site to models without any sites;
  adjust its local offset in the training controls.
- Generate randomized nearby start/goal examples using MuJoCo forward
  kinematics.
- Train a standardized scikit-learn MLP to predict local joint corrections
  from current joint positions and a Cartesian target.
- Evaluate joint errors and end-effector errors on a held-out split, visualize
  an example output, and download the trained policy and evaluation report.
- Reuse a model imported in Robot Workspace on the training page.
- Reject episode start and target configurations outside declared MuJoCo joint
  limits before simulation begins.
- Execute and measure multi-waypoint joint-space trajectories with bounded PD
  plus gravity compensation; report waypoint completion, Cartesian tracking
  error, final joint error, contacts, and collisions.
- Treat obstacle contact at the initial pose or during execution as a failed
  task, with automated validation for both cases.

**Measured check:** default arm training/evaluation has an automated test. On a
500-example, 100-epoch seed-23 run, the held-out end-effector RMSE was below
0.15 m and held-out success at 8 cm tolerance was recorded. Run the UI for the
current seed and settings to see its exact metrics.

**Current limits:** this is supervised local inverse kinematics, not RL and not
a hardware-ready torque controller. It only applies to tasks with named sites
and limited hinge/slide joints tied to direct joint actuators. The generated
examples use local joint corrections in a configured range. Multi-branch IK,
full-arm Cartesian path-to-joint conversion, and obstacle-aware training labels
are not handled.
The fallback tool point is only a starting estimate; inspect it in the scene and
adjust its offset or provide a model-specific MuJoCo site for accurate geometry.

## Milestone 3 — Shared training and experiment layer

**Status: Planned**

- Define common interfaces for datasets, experiment configurations, training
  algorithms, checkpoints, metrics, and model exports.
- Add reproducible regression baselines alongside the existing MLP inverse-
  kinematics model, with shared held-out evaluation and result exports.
- Compare conventional controllers, learned mappings, and later policies under
  identical robot, target, obstacle, and seed conditions.

**Current limit:** the path planner routes only the selected tool point. It does
not yet constrain the full arm's swept volume, and the learned mapping does not
condition its joint targets on the obstacle layout. Full-arm path following and
collision-aware learned policies remain future work.

## Milestone 4 — Dynamics and policy learning

**Status: Planned**

- Generate state/action trajectories and train a learned dynamics model or a
  policy using a documented algorithm and reproducible seeds.
- Keep training and held-out evaluation separate; report target error,
  collisions, completion rate, and control effort.

## Milestone 5 — Robustness and sim-to-real readiness

**Status: Planned**

- Randomize mass, inertia, friction, actuator strength, and observation noise
  within documented ranges.
- Evaluate parameter combinations excluded from training and preserve model
  provenance and experiment settings.
- Treat this as simulation evidence only; it does not establish physical safety
  or real-world transfer performance.

## Later option — Physical robot

**Status: Out of current scope**

Hardware support can be considered when a specific robot, sensors, and actuator
interfaces are selected. Current work is entirely simulated.
