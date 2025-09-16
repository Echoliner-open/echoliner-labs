"""Kinematic utilities powering EchoLiner's modular manipulators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = ["DHLink", "KinematicChain", "plan_minimum_jerk"]


@dataclass(frozen=True)
class DHLink:
    """Denavit–Hartenberg link specification for revolute joints."""

    a: float
    alpha: float
    d: float
    theta_offset: float = 0.0


class KinematicChain:
    """Manipulator model constructed from Denavit–Hartenberg parameters."""

    def __init__(self, links: Iterable[DHLink]):
        self._links = tuple(links)
        if not self._links:
            raise ValueError("A kinematic chain requires at least one link")

    @property
    def dof(self) -> int:
        return len(self._links)

    def _dh_transform(self, link: DHLink, theta: float) -> NDArray[np.float64]:
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(link.alpha)
        sa = np.sin(link.alpha)
        return np.array(
            [
                [ct, -st * ca, st * sa, link.a * ct],
                [st, ct * ca, -ct * sa, link.a * st],
                [0.0, sa, ca, link.d],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def forward_kinematics(self, joint_values: Sequence[float]) -> NDArray[np.float64]:
        if len(joint_values) != self.dof:
            raise ValueError("Joint vector dimension mismatch")
        transform = np.eye(4, dtype=float)
        for link, value in zip(self._links, joint_values):
            theta = link.theta_offset + value
            transform = transform @ self._dh_transform(link, theta)
        return transform

    def _so3_log(self, rotation: NDArray[np.float64]) -> NDArray[np.float64]:
        cos_theta = (np.trace(rotation) - 1.0) / 2.0
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        theta = np.arccos(cos_theta)
        if theta < 1e-9:
            return np.zeros(3, dtype=float)
        factor = theta / (2.0 * np.sin(theta))
        return factor * np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ],
            dtype=float,
        )

    def jacobian(self, joint_values: Sequence[float], delta: float = 1e-6) -> NDArray[np.float64]:
        base_pose = self.forward_kinematics(joint_values)
        base_rot = base_pose[:3, :3]
        base_pos = base_pose[:3, 3]
        jacobian = np.zeros((6, self.dof), dtype=float)
        for idx in range(self.dof):
            perturbed = np.array(joint_values, dtype=float)
            perturbed[idx] += delta
            pose = self.forward_kinematics(perturbed)
            rot_delta = base_rot.T @ pose[:3, :3]
            rot_vec = self._so3_log(rot_delta) / delta
            pos_vec = (pose[:3, 3] - base_pos) / delta
            jacobian[:3, idx] = rot_vec
            jacobian[3:, idx] = pos_vec
        return jacobian

    def _pose_error(
        self,
        current: NDArray[np.float64],
        target: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        rot_error = self._so3_log(current[:3, :3].T @ target[:3, :3])
        pos_error = target[:3, 3] - current[:3, 3]
        return np.concatenate([rot_error, pos_error])

    def inverse_kinematics(
        self,
        target_pose: NDArray[np.float64],
        initial_guess: Sequence[float],
        *,
        damping: float = 1e-3,
        max_iterations: int = 200,
        tolerance: float = 1e-6,
    ) -> NDArray[np.float64]:
        if target_pose.shape != (4, 4):
            raise ValueError("target_pose must be 4x4")
        joint_values = np.array(initial_guess, dtype=float)
        if joint_values.shape != (self.dof,):
            raise ValueError("Initial guess dimension mismatch")

        for _ in range(max_iterations):
            pose = self.forward_kinematics(joint_values)
            error = self._pose_error(pose, target_pose)
            if np.linalg.norm(error) < tolerance:
                return joint_values
            jacobian = self.jacobian(joint_values)
            jj_t = jacobian @ jacobian.T
            identity = np.eye(jj_t.shape[0]) * (damping**2)
            delta = jacobian.T @ np.linalg.solve(jj_t + identity, error)
            joint_values += delta
        raise RuntimeError("Inverse kinematics failed to converge")


def plan_minimum_jerk(
    start: NDArray[np.float64],
    goal: NDArray[np.float64],
    duration: float,
    samples: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Minimum-jerk trajectory connecting two joint configurations."""

    if duration <= 0:
        raise ValueError("Duration must be positive")
    if samples < 2:
        raise ValueError("At least two samples are required")
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if start.shape != goal.shape:
        raise ValueError("Start and goal must share the same dimensionality")

    time = np.linspace(0.0, duration, samples, dtype=float)
    tau = time / duration
    tau3 = tau**3
    tau4 = tau**4
    tau5 = tau**5
    blend = 10 * tau3 - 15 * tau4 + 6 * tau5
    positions = start + (goal - start) * blend[:, None]
    return time, positions
