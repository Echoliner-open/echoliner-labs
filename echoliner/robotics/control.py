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
