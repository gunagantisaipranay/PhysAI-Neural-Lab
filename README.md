# PhysAI Robotics Lab

PhysAI Robotics Lab is a laptop-only robotics simulation and model-training workspace built with MuJoCo and Streamlit. It requires no physical robot.

## App workflow

1. **Home** explains the workflow and the capabilities of the current milestone.
2. **Robot Workspace** loads the built-in arm or an MJCF, URDF, MJZ, or ZIP model; it offers pose controls, a 3D scene, a target, collision-enabled obstacles, and a clearance-aware 3D tool-point route preview.
3. **Model Training** generates collision-free randomized MuJoCo examples, trains a supervised neural inverse-kinematics model, evaluates it on held-out targets, compares learned and expert targets in closed-loop simulator episodes, and exports the policy and reports. The last model imported in Robot Workspace can be reused here.
4. **Cart-Pole Benchmark** retains the project's verified classic control and learned imitation experiment as a robotics baseline.

### Current learned task

The first model learns a local joint correction from current joint values and a target point. Its output is a bounded joint setpoint. Evaluation reports held-out end-effector position error and the fraction within 8 cm.

Training requires limited hinge/slide joints connected to direct joint actuators. Models without a named end-effector site receive an adjustable approximate tool point at a terminal link; inspect that point and tune its local offset before trusting evaluation. Imported URDF models can receive virtual motors, whose generic limits are placeholders. Obstacle-aware path planning, reinforcement learning, learned dynamics, and hardware transfer are later milestones. Check `docs/robotics_milestones.md` for limitations and the staged plan.

### Run locally

```powershell
.\.venv-robotics312\Scripts\python.exe -m streamlit run app.py
```
