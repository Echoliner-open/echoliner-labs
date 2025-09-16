"""Dynamics models and simulation primitives for manipulators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .kinematics import DHLink, KinematicChain

__all__ = [
    "JointDynamics",
    "RigidBodyParameters",
    "articulated_body_inertia",
    "coriolis_forces",
    "gravity_forces",
    "simulate_dynamics",
]


@dataclass
class RigidBodyParameters:
    """Inertial parameters for a single link."""

    mass: float
    com: NDArray[np.float64]
    inertia: NDArray[np.float64]


@dataclass
class JointDynamics:
    """Encapsulates the dynamic model for a manipulator."""

    chain: KinematicChain
    parameters: Sequence[RigidBodyParameters]
    gravity_vector: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 0.0, -9.81])
    )

    def __post_init__(self) -> None:
        if len(self.parameters) != self.chain.dof:
            raise ValueError("Inertial parameters must match chain dof")

    def mass_matrix(self, joint_values: Sequence[float]) -> NDArray[np.float64]:
        m = np.zeros((self.chain.dof, self.chain.dof))
        jacobians = []
        for idx, link in enumerate(self.chain._links):
            pose = self.chain.forward_kinematics(joint_values)
            rotation = pose[:3, :3]
            com = rotation @ self.parameters[idx].com
            jacobian = self.chain.jacobian(joint_values)
            jacobians.append(jacobian)
            mass = self.parameters[idx].mass
            inertia = rotation @ self.parameters[idx].inertia @ rotation.T
            m += jacobian[3:].T @ (mass * np.eye(3)) @ jacobian[3:]
            m += jacobian[:3].T @ inertia @ jacobian[:3]
        return m

    def coriolis(self, joint_values: Sequence[float], joint_velocities: Sequence[float]) -> NDArray[np.float64]:
        h = np.zeros(self.chain.dof)
        m = self.mass_matrix(joint_values)
        dq = np.asarray(joint_velocities)
        for k in range(self.chain.dof):
            for i in range(self.chain.dof):
                for j in range(self.chain.dof):
                    c = 0.5 * (
                        self._partial(m, k, i, j, joint_values)
                        + self._partial(m, k, j, i, joint_values)
                        - self._partial(m, i, j, k, joint_values)
                    )
                    h[k] += c * dq[i] * dq[j]
        return h

    def gravity(self, joint_values: Sequence[float]) -> NDArray[np.float64]:
        g = np.zeros(self.chain.dof)
        for idx, params in enumerate(self.parameters):
            pose = self.chain.forward_kinematics(joint_values)
            rotation = pose[:3, :3]
            com_world = rotation @ params.com
            torque = self.chain.jacobian(joint_values)[:3].T @ np.cross(com_world, params.mass * self.gravity_vector)
            g += torque
        return g

    def _partial(
        self,
        matrix: NDArray[np.float64],
        i: int,
        j: int,
        k: int,
        joint_values: Sequence[float],
        delta: float = 1e-4,
    ) -> float:
        perturbed = np.array(joint_values, dtype=float)
        perturbed[k] += delta
        m_plus = self.mass_matrix(perturbed)
        return float((m_plus[i, j] - matrix[i, j]) / delta)


def articulated_body_inertia(parameters: Sequence[RigidBodyParameters]) -> NDArray[np.float64]:
    return np.array([param.inertia for param in parameters])


def coriolis_forces(model: JointDynamics, joint_values: Sequence[float], joint_velocities: Sequence[float]) -> NDArray[np.float64]:
    return model.coriolis(joint_values, joint_velocities)


def gravity_forces(model: JointDynamics, joint_values: Sequence[float]) -> NDArray[np.float64]:
    return model.gravity(joint_values)


def simulate_dynamics(
    model: JointDynamics,
    joint_values: NDArray[np.float64],
    joint_velocities: NDArray[np.float64],
    torques: NDArray[np.float64],
    dt: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    m = model.mass_matrix(joint_values)
    c = model.coriolis(joint_values, joint_velocities)
    g = model.gravity(joint_values)
    acceleration = np.linalg.solve(m, torques - c - g)
    new_velocities = joint_velocities + acceleration * dt
    new_positions = joint_values + new_velocities * dt
    return new_positions, new_velocities
