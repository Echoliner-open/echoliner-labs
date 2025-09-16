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
    "VolumetricTSDF",
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


class VolumetricTSDF:
    """Incremental truncated signed distance field for dense fusion.

    The implementation focuses on practical research workflows where
    engineering teams wish to fuse depth imagery from multiple sensing rigs
    into a metrically consistent world volume.  The class tracks a TSDF grid,
    associated integration weights, and exposes helpers for extracting surface
    samples that can be streamed into downstream inspection or planning tools.
    """

    def __init__(
        self,
        bounds: NDArray[np.float64],
        resolution: tuple[int, int, int],
        *,
        truncation: float = 0.03,
        max_weight: float = 64.0,
    ) -> None:
        if bounds.shape != (2, 3):
            raise ValueError("bounds must be shaped (2, 3)")
        if any(axis <= 0 for axis in resolution):
            raise ValueError("resolution entries must be positive")
        if truncation <= 0.0:
            raise ValueError("truncation must be positive")
        self.bounds = bounds.astype(float)
        self.resolution = tuple(int(v) for v in resolution)
        self.truncation = truncation
        self.max_weight = max_weight
        self.tsdf = np.ones(self.resolution, dtype=float)
        self.weights = np.zeros(self.resolution, dtype=float)
        self._voxel_size = (self.bounds[1] - self.bounds[0]) / np.array(self.resolution)

    @property
    def voxel_size(self) -> NDArray[np.float64]:
        """Return the spacing between adjacent voxels along each axis."""

        return self._voxel_size

    def reset(self) -> None:
        """Reset TSDF values and integration weights."""

        self.tsdf.fill(1.0)
        self.weights.fill(0.0)

    def integrate(
        self,
        depth_map: NDArray[np.float64],
        intrinsics: NDArray[np.float64],
        extrinsics: CameraExtrinsics,
        *,
        depth_scale: float = 1.0,
        weight: float = 1.0,
    ) -> None:
        """Fuse a calibrated depth frame into the TSDF volume.

        Parameters
        ----------
        depth_map:
            Per-pixel depth values expressed in meters.
        intrinsics:
            3x3 calibration matrix describing the camera model.
        extrinsics:
            Pose of the camera in the world frame.
        depth_scale:
            Optional scalar applied to the input depths (useful when the raw
            sensor encodes millimetres or other units).
        weight:
            Confidence weight applied to the integration update.
        """

        if depth_map.ndim != 2:
            raise ValueError("depth_map must be two-dimensional")
        if intrinsics.shape != (3, 3):
            raise ValueError("intrinsics must be 3x3")
        if weight <= 0:
            raise ValueError("weight must be positive")

        height, width = depth_map.shape
        v_coords, u_coords = np.indices((height, width))
        z = depth_map.astype(float).reshape(-1) * depth_scale
        valid = z > 0.0
        if not np.any(valid):
            return

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        u = u_coords.reshape(-1)[valid]
        v = v_coords.reshape(-1)[valid]
        z = z[valid]
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        camera_points = np.stack([x, y, z], axis=1)

        rotation = extrinsics.rotation
        translation = extrinsics.translation
        world_points = (rotation.T @ (camera_points.T - translation[:, None])).T
        camera_center = -rotation.T @ translation

        grid_coords = (world_points - self.bounds[0]) / self._voxel_size
        indices = np.floor(grid_coords).astype(int)
        inside = np.all((indices >= 0) & (indices < np.array(self.resolution)), axis=1)
        if not np.any(inside):
            return

        world_points = world_points[inside]
        for point in world_points:
            ray = point - camera_center
            ray_norm = np.linalg.norm(ray)
            if ray_norm < 1e-9:
                continue
            ray_dir = ray / ray_norm
            min_dist = max(ray_norm - self.truncation, 0.0)
            max_dist = ray_norm + self.truncation
            samples = max(
                int(np.ceil((max_dist - min_dist) / float(np.min(self._voxel_size)))) + 1,
                3,
            )
            distances = np.linspace(min_dist, max_dist, samples)
            for dist in distances:
                sample = camera_center + ray_dir * dist
                grid_coord = (sample - self.bounds[0]) / self._voxel_size
                index = np.floor(grid_coord).astype(int)
                if np.any(index < 0) or np.any(index >= np.array(self.resolution)):
                    continue
                idx_tuple = tuple(index)
                sdf = ray_norm - dist
                sdf_normalized = float(np.clip(sdf / self.truncation, -1.0, 1.0))
                previous_weight = self.weights[idx_tuple]
                blended = (
                    previous_weight * self.tsdf[idx_tuple] + weight * sdf_normalized
                ) / (previous_weight + weight)
                self.tsdf[idx_tuple] = blended
                self.weights[idx_tuple] = min(previous_weight + weight, self.max_weight)
            point_index = np.floor((point - self.bounds[0]) / self._voxel_size).astype(int)
            if np.all(point_index >= 0) and np.all(point_index < np.array(self.resolution)):
                self.tsdf[tuple(point_index)] = 0.0

    def extract_point_cloud(
        self,
        *,
        iso_level: float = 0.0,
        min_weight: float = 1.0,
        band: float = 0.15,
    ) -> NDArray[np.float64]:
        """Return surface samples near the specified iso-surface."""

        if band <= 0.0:
            raise ValueError("band must be positive")
        mask = (self.weights >= min_weight) & (np.abs(self.tsdf - iso_level) <= band)
        if not np.any(mask):
            return np.zeros((0, 3), dtype=float)
        indices = np.argwhere(mask)
        points = self.bounds[0] + (indices + 0.5) * self._voxel_size
        return points.reshape(-1, 3)

    def occupancy_grid(self, threshold: float = 0.0) -> NDArray[np.bool_]:
        """Return a binary occupancy grid derived from the TSDF."""

        return self.tsdf <= threshold
