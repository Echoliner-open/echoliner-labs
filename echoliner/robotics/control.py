"""Robotic control policies for articulated manipulators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .dynamics import JointDynamics

__all__ = [
    "ComputedTorqueController",
    "ImpedanceController",
    "ModelPredictiveController",
    "OperationalSpaceController",
]


def _ensure_vector(vector: Sequence[float], dof: int) -> NDArray[np.float64]:
    array = np.asarray(vector, dtype=float)
    if array.shape != (dof,):
        raise ValueError("Expected vector of length {dof}".format(dof=dof))
    return array


@dataclass
class ComputedTorqueController:
    """Computed torque controller with gravity and Coriolis compensation."""

    model: JointDynamics
    kp: NDArray[np.float64]
    kd: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.kp.shape != (self.model.chain.dof,) or self.kd.shape != (self.model.chain.dof,):
            raise ValueError("Gain vectors must match dof")

    def __call__(
        self,
        joint_values: Sequence[float],
        joint_velocities: Sequence[float],
        target_positions: Sequence[float],
        target_velocities: Sequence[float] | None = None,
    ) -> NDArray[np.float64]:
        q = _ensure_vector(joint_values, self.model.chain.dof)
        dq = _ensure_vector(joint_velocities, self.model.chain.dof)
        qd = _ensure_vector(target_positions, self.model.chain.dof)
        if target_velocities is None:
            dqd = np.zeros_like(q)
        else:
            dqd = _ensure_vector(target_velocities, self.model.chain.dof)
        mass = self.model.mass_matrix(q)
        coriolis = self.model.coriolis(q, dq)
        gravity = self.model.gravity(q)
        error = qd - q
        derror = dqd - dq
        torque = mass @ (self.kp * error + self.kd * derror) + coriolis + gravity
        return torque


@dataclass
class ImpedanceController:
    """Cartesian impedance control with stiffness and damping gains."""

    stiffness: float
    damping: float

    def compute(
        self,
        pose_error: NDArray[np.float64],
        velocity_error: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self.stiffness * pose_error + self.damping * velocity_error


@dataclass
class ModelPredictiveController:
    """Linear MPC for joint regulation using finite horizon optimization."""

    model: JointDynamics
    horizon: int = 10
    control_limit: float = 50.0
    state_weight: float = 10.0
    control_weight: float = 1.0

    def _linearize(self, joint_values: NDArray[np.float64]) -> NDArray[np.float64]:
        mass = self.model.mass_matrix(joint_values)
        return np.linalg.inv(mass)

    def solve(
        self,
        joint_values: Sequence[float],
        joint_velocities: Sequence[float],
        target: Sequence[float],
    ) -> NDArray[np.float64]:
        q = _ensure_vector(joint_values, self.model.chain.dof)
        dq = _ensure_vector(joint_velocities, self.model.chain.dof)
        qd = _ensure_vector(target, self.model.chain.dof)
        acceleration_map = self._linearize(q)
        state = np.concatenate([q, dq])
        target_state = np.concatenate([qd, np.zeros_like(q)])
        control = np.zeros(self.model.chain.dof)
        for _ in range(self.horizon):
            error = state - target_state
            gradient = self.state_weight * error[: self.model.chain.dof]
            control -= acceleration_map @ gradient
            control = np.clip(control, -self.control_limit, self.control_limit)
            dq = dq + acceleration_map @ control
            q = q + dq
            state = np.concatenate([q, dq])
        return control


@dataclass
class OperationalSpaceController:
    """Operational-space control with nullspace posture regulation."""

    model: JointDynamics
    task_stiffness: NDArray[np.float64]
    task_damping: NDArray[np.float64]
    nullspace_gain: float = 0.2
    regularization: float = 1e-4

    def __post_init__(self) -> None:
        self._kp = self._as_matrix(self.task_stiffness)
        self._kd = self._as_matrix(self.task_damping)
        if self._kp.shape != (6, 6) or self._kd.shape != (6, 6):
            raise ValueError("task gains must be 6x6 or length-6 vectors")

    @staticmethod
    def _as_matrix(gain: NDArray[np.float64]) -> NDArray[np.float64]:
        gain_array = np.asarray(gain, dtype=float)
        if gain_array.shape == (6, 6):
            return gain_array
        if gain_array.shape == (6,):
            return np.diag(gain_array)
        raise ValueError("gain must be shaped (6,) or (6, 6)")

    def compute(
        self,
        joint_values: Sequence[float],
        joint_velocities: Sequence[float],
        target_pose: NDArray[np.float64],
        *,
        target_twist: Sequence[float] | None = None,
        target_acceleration: Sequence[float] | None = None,
    ) -> NDArray[np.float64]:
        if target_pose.shape != (4, 4):
            raise ValueError("target_pose must be 4x4")
        q = _ensure_vector(joint_values, self.model.chain.dof)
        dq = _ensure_vector(joint_velocities, self.model.chain.dof)
        desired_twist = (
            np.zeros(6)
            if target_twist is None
            else np.asarray(target_twist, dtype=float)
        )
        desired_acc = (
            np.zeros(6)
            if target_acceleration is None
            else np.asarray(target_acceleration, dtype=float)
        )
        if desired_twist.shape != (6,) or desired_acc.shape != (6,):
            raise ValueError("target_twist and target_acceleration must be length-6")

        pose = self.model.chain.forward_kinematics(q)
        rot_error = self.model.chain._so3_log(pose[:3, :3].T @ target_pose[:3, :3])
        pos_error = target_pose[:3, 3] - pose[:3, 3]
        task_error = np.concatenate([rot_error, pos_error])

        jacobian = self.model.chain.jacobian(q)
        mass = self.model.mass_matrix(q)
        mass_inv = np.linalg.pinv(mass)
        task_inertia = jacobian @ mass_inv @ jacobian.T
        task_inertia += self.regularization * np.eye(task_inertia.shape[0])
        lambda_matrix = np.linalg.pinv(task_inertia)

        task_velocity = jacobian @ dq
        velocity_error = desired_twist - task_velocity
        task_acc = desired_acc + self._kp @ task_error + self._kd @ velocity_error
        task_wrench = lambda_matrix @ task_acc
        torque_task = jacobian.T @ task_wrench

        nullspace = np.eye(self.model.chain.dof) - np.linalg.pinv(jacobian) @ jacobian
        posture_torque = -self.nullspace_gain * nullspace @ dq

        compensators = self.model.coriolis(q, dq) + self.model.gravity(q)
        return torque_task + posture_torque + compensators
