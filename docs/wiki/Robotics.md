# Robotics Stack

EchoLiner's robotics module encapsulates the manipulators, planners, and
controllers that coordinate modular workcells.  The code mirrors the structure
used in production by exposing light-weight Python implementations backed by the
same math as our embedded stack.

## Manipulator Modeling

* `DHLink` and `KinematicChain` define robot topology via standard Denavit–
  Hartenberg parameters.  `forward_kinematics` and `jacobian` provide the
  geometric backbone for both control and planning layers.
* `plan_minimum_jerk` and `TrajectorySmoother` produce time-parameterized joint
  trajectories that respect velocity boundary conditions—use them to seed
  computed torque or MPC controllers.

## Dynamics & Control

* `JointDynamics` aggregates link inertias, Coriolis forces, and gravity terms
  so that torque-level controllers can compensate model-based effects.
* `ComputedTorqueController` is our go-to for high-rate joint regulation on
  deterministic hardware.
* `OperationalSpaceController` extends the stack with task-space regulation and
  nullspace posture control.  The class wraps operational-space inertia
  (`lambda_matrix`), task-space PD gains, and adaptive nullspace damping so that
  the same controller can service research prototypes and production cells.

```python
from echoliner.robotics import (
    DHLink, JointDynamics, KinematicChain, OperationalSpaceController,
    RigidBodyParameters
)
import numpy as np

chain = KinematicChain([DHLink(a=0.45, alpha=0.0, d=0.0), DHLink(a=0.35, alpha=0.0, d=0.0)])
params = [
    RigidBodyParameters(mass=5.2, com=np.array([0.11, 0.0, 0.0]), inertia=np.eye(3)),
    RigidBodyParameters(mass=3.6, com=np.array([0.09, 0.0, 0.0]), inertia=np.eye(3)),
]
model = JointDynamics(chain, params)
controller = OperationalSpaceController(model, task_stiffness=np.ones(6) * 35.0, task_damping=np.ones(6) * 5.0)
current = np.zeros(chain.dof)
desired_pose = chain.forward_kinematics([0.2, -0.3])
desired_pose[1, 3] += 0.04
torque = controller.compute(current, np.zeros(chain.dof), desired_pose)
```

* Feed the resulting torque command into `simulate_dynamics` for software-in-the
  loop validation before touching the real cell.
* Tune `nullspace_gain` to balance end-effector accuracy against joint-limit
  avoidance.  Larger values prioritize posture control.

## Planning & Safety

* `RRTPlanner` and `WorkspaceObstacle` support coarse joint-space planning when
  workspace constraints are known.  For fine adjustments, reuse the operational
  space controller above with small task offsets.
* Integrate the `torque_norm` channel from the digital twin (see the analytics
  wiki page) to validate that proposed trajectories remain within actuator
  limits prior to deployment.

Return to the [wiki index](README.md) or continue with the
[Analytics & Digital Twins](Analytics.md) guidance.
