"""Structure-from-motion style reconstruction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from numpy.typing import NDArray

from .calibration import CameraExtrinsics, compose_projection_matrix, triangulate_points

__all__ = [
    "BundleAdjustmentProblem",
    "IterativeTriangulator",
    "TriangulationResult",
]


@dataclass
class TriangulationResult:
    """Result of a multi-view triangulation."""

    points: NDArray[np.float64]
    reprojection_error: float


class IterativeTriangulator:
    """Refine 3-D structure estimates via reweighted least squares."""

    def __init__(self, iterations: int = 5):
        self.iterations = iterations

    def solve(
        self,
        projections: Sequence[NDArray[np.float64]],
        observations: Sequence[NDArray[np.float64]],
    ) -> TriangulationResult:
        points = triangulate_points(projections, observations)
        for _ in range(self.iterations):
            weighted_projections = []
            weighted_observations = []
            for mat, obs in zip(projections, observations):
                weighted_projections.append(mat)
                weighted_observations.append(obs)
            points = triangulate_points(weighted_projections, weighted_observations)
            reprojection = []
            for mat, obs in zip(projections, observations):
                proj = mat @ np.hstack([points, np.ones((points.shape[0], 1))]).T
                proj = proj[:2] / proj[2:3]
                reprojection.append(np.linalg.norm(proj.T - obs, axis=1))
            residuals = np.concatenate(reprojection)
            median = np.median(residuals)
            scale = 1.4826 * np.median(np.abs(residuals - median))
            scale = max(scale, 1e-6)
            weights = 1.0 / np.clip(residuals / scale, 1e-6, None)
        error = float(np.mean(residuals))
        return TriangulationResult(points=points, reprojection_error=error)


@dataclass
class Observation:
    camera_index: int
    point_index: int
    measurement: NDArray[np.float64]


class BundleAdjustmentProblem:
    """Sparse bundle adjustment with Levenberg–Marquardt updates."""

    def __init__(self):
        self._intrinsics: List[NDArray[np.float64]] = []
        self._extrinsics: List[CameraExtrinsics] = []
        self._points: Dict[int, NDArray[np.float64]] = {}
        self._observations: List[Observation] = []
        self._damping = 1e-3

    def add_camera(
        self,
        intrinsics: NDArray[np.float64],
        extrinsics: CameraExtrinsics,
    ) -> int:
        self._intrinsics.append(intrinsics)
        self._extrinsics.append(extrinsics)
        return len(self._intrinsics) - 1

    def add_point(self, index: int, position: NDArray[np.float64]) -> None:
        self._points[index] = position.astype(float)

    def add_observation(self, camera_index: int, point_index: int, measurement: NDArray[np.float64]) -> None:
        self._observations.append(Observation(camera_index, point_index, measurement))

    def _project(self, camera_index: int, point: NDArray[np.float64]) -> NDArray[np.float64]:
        intr = self._intrinsics[camera_index]
        extr = self._extrinsics[camera_index]
        projection = compose_projection_matrix(intr, extr)
        homogeneous = np.hstack([point, [1.0]])
        projected = projection @ homogeneous
        return projected[:2] / projected[2]

    def residual_vector(self) -> NDArray[np.float64]:
        residuals = []
        for obs in self._observations:
            predicted = self._project(obs.camera_index, self._points[obs.point_index])
            residuals.append((predicted - obs.measurement).ravel())
        return np.concatenate(residuals)

    def solve(self, iterations: int = 10) -> float:
        for _ in range(iterations):
            residuals = self.residual_vector()
            jacobian_rows: List[NDArray[np.float64]] = []
            for obs in self._observations:
                cam_index = obs.camera_index
                point_index = obs.point_index
                intr = self._intrinsics[cam_index]
                extr = self._extrinsics[cam_index]
                rotation = extr.rotation
                translation = extr.translation
                point = self._points[point_index]
                cam_coords = rotation @ point + translation
                x, y, z = cam_coords
                fx, fy = intr[0, 0], intr[1, 1]
                jacobian = np.array(
                    [
                        [fx / z, 0.0, -fx * x / (z**2)],
                        [0.0, fy / z, -fy * y / (z**2)],
                    ]
                )
                jacobian_rows.append(jacobian)
            jacobian_matrix = np.vstack(jacobian_rows)
            hessian = jacobian_matrix.T @ jacobian_matrix
            step = np.linalg.solve(hessian + self._damping * np.eye(3), -jacobian_matrix.T @ residuals)
            for index in self._points:
                self._points[index] = self._points[index] + step
            if np.linalg.norm(step) < 1e-6:
                break
        final_residuals = self.residual_vector()
        return float(np.sqrt(np.mean(final_residuals**2)))
