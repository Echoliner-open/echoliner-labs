"""Sampling-based and trajectory optimization planners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .kinematics import KinematicChain

__all__ = ["RRTPlanner", "TrajectorySmoother", "WorkspaceObstacle"]


@dataclass
class WorkspaceObstacle:
    center: NDArray[np.float64]
    radius: float

    def contains(self, point: NDArray[np.float64]) -> bool:
        return float(np.linalg.norm(point - self.center)) <= self.radius


class RRTPlanner:
    """Rapidly-exploring random tree planner for joint space."""

    def __init__(
        self,
        chain: KinematicChain,
        bounds: Sequence[Tuple[float, float]],
        *,
        step_size: float = 0.1,
        goal_bias: float = 0.1,
        max_iterations: int = 1000,
    ):
        self.chain = chain
        self.bounds = bounds
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.max_iterations = max_iterations
        self._rng = np.random.default_rng()

    def _sample(self, goal: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._rng.random() < self.goal_bias:
            return goal
        sample = []
        for lower, upper in self.bounds:
            sample.append(self._rng.uniform(lower, upper))
        return np.asarray(sample)

    def _nearest(self, nodes: NDArray[np.float64], sample: NDArray[np.float64]) -> int:
        distances = np.linalg.norm(nodes - sample, axis=1)
        return int(np.argmin(distances))

    def _steer(self, source: NDArray[np.float64], target: NDArray[np.float64]) -> NDArray[np.float64]:
        direction = target - source
        norm = np.linalg.norm(direction)
        if norm <= self.step_size:
            return target
        return source + direction / norm * self.step_size

    def plan(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        collision_fn: Callable[[NDArray[np.float64]], bool],
    ) -> List[NDArray[np.float64]]:
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        nodes = [start]
        parents = [-1]
        for _ in range(self.max_iterations):
            sample = self._sample(goal)
            nearest_idx = self._nearest(np.array(nodes), sample)
            new = self._steer(nodes[nearest_idx], sample)
            if collision_fn(new):
                continue
            nodes.append(new)
            parents.append(nearest_idx)
            if np.linalg.norm(new - goal) < self.step_size:
                nodes.append(goal)
                parents.append(len(nodes) - 2)
                break
        path = []
        current = len(nodes) - 1
        while current != -1:
            path.append(nodes[current])
            current = parents[current]
        path.reverse()
        return path


class TrajectorySmoother:
    """Time-parameterized smoothing using cubic splines."""

    def __init__(self, chain: KinematicChain):
        self.chain = chain

    def smooth(self, waypoints: Sequence[Sequence[float]], duration: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        waypoints_arr = np.asarray(waypoints, dtype=float)
        segments = len(waypoints_arr) - 1
        time = np.linspace(0.0, duration, segments * 10)
        positions = np.zeros((time.size, self.chain.dof))
        for i in range(self.chain.dof):
            coefficients = self._cubic_spline(waypoints_arr[:, i])
            positions[:, i] = self._evaluate_spline(coefficients, segments, time / duration)
        return time, positions

    def _cubic_spline(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        n = len(values) - 1
        a = values[:-1]
        b = np.zeros(n)
        d = np.zeros(n)
        c = np.zeros(n + 1)
        alpha = np.zeros(n)
        for i in range(1, n):
            alpha[i] = 3 * (values[i + 1] - values[i]) - 3 * (values[i] - values[i - 1])
        l = np.ones(n + 1)
        mu = np.zeros(n + 1)
        z = np.zeros(n + 1)
        for i in range(1, n):
            l[i] = 4.0 - mu[i - 1]
            mu[i] = 1.0 / l[i]
            z[i] = (alpha[i] - z[i - 1]) / l[i]
        for j in range(n - 1, -1, -1):
            c[j] = z[j] - mu[j] * c[j + 1]
            b[j] = (values[j + 1] - values[j]) - (c[j + 1] + 2 * c[j]) / 3
            d[j] = (c[j + 1] - c[j]) / 3
        return np.stack([a, b, c[:-1], d], axis=1)

    def _evaluate_spline(self, coefficients: NDArray[np.float64], segments: int, tau: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.zeros_like(tau)
        segment_length = 1.0 / segments
        for i in range(segments):
            mask = (tau >= i * segment_length) & (tau <= (i + 1) * segment_length)
            local_tau = (tau[mask] - i * segment_length) / segment_length
            a, b, c, d = coefficients[i]
            values[mask] = a + b * local_tau + c * local_tau**2 + d * local_tau**3
        return values
