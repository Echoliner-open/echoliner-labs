"""Camera calibration and 3D reconstruction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CameraExtrinsics",
    "compose_projection_matrix",
    "project_points",
    "estimate_extrinsics_dlt",
    "triangulate_points",
    "reprojection_error",
]


@dataclass(frozen=True)
class CameraExtrinsics:
    """Rigid transform describing the pose of a camera in the world frame."""

    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.rotation.shape != (3, 3):
            raise ValueError("Rotation must be a 3x3 matrix")
        if self.translation.shape != (3,):
            raise ValueError("Translation must be a length-3 vector")

    def as_matrix(self) -> NDArray[np.float64]:
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix


def compose_projection_matrix(
    intrinsics: NDArray[np.float64], extrinsics: CameraExtrinsics
) -> NDArray[np.float64]:
    """Return the 3x4 projection matrix for a calibrated camera."""

    if intrinsics.shape != (3, 3):
        raise ValueError("Intrinsics must be a 3x3 matrix")
    extrinsic_matrix = extrinsics.as_matrix()[:3, :]
    return intrinsics @ extrinsic_matrix


def project_points(
    points_3d: NDArray[np.float64],
    intrinsics: NDArray[np.float64],
    extrinsics: CameraExtrinsics,
) -> NDArray[np.float64]:
    """Project world-space points into the image plane."""

    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError("points_3d must be shaped (N, 3)")

    projection = compose_projection_matrix(intrinsics, extrinsics)
    homogeneous = np.concatenate(
        [points_3d, np.ones((points_3d.shape[0], 1), dtype=float)], axis=1
    )
    image_h = homogeneous @ projection.T
    image = image_h[:, :2] / image_h[:, 2:3]
    return image


def estimate_extrinsics_dlt(
    world_points: NDArray[np.float64],
    image_points: NDArray[np.float64],
    intrinsics: NDArray[np.float64],
) -> CameraExtrinsics:
    """Estimate camera extrinsics using Direct Linear Transformation."""

    if world_points.shape[0] < 6:
        raise ValueError("At least six correspondences are required for DLT")
    if world_points.shape != (image_points.shape[0], 3):
        raise ValueError("Points must be shaped (N, 3) vs (N, 2)")

    world_h = np.concatenate(
        [world_points, np.ones((world_points.shape[0], 1), dtype=float)], axis=1
    )
    rows = []
    for point, img in zip(world_h, image_points):
        x, y, z, w = point
        u, v = img
        rows.append([0, 0, 0, 0, -x, -y, -z, -w, v * x, v * y, v * z, v * w])
        rows.append([x, y, z, w, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u * w])
    a = np.asarray(rows, dtype=float)
    _, _, vt = np.linalg.svd(a)
    p = vt[-1].reshape(3, 4)

    extrinsic_matrix = np.linalg.inv(intrinsics) @ p
    rotation_scaled = extrinsic_matrix[:, :3]
    translation_scaled = extrinsic_matrix[:, 3]
    u, s, vt_rot = np.linalg.svd(rotation_scaled)
    rotation = u @ vt_rot
    if np.linalg.det(rotation) < 0:
        rotation *= -1
        translation_scaled *= -1
    scale = np.mean(s)
    translation = translation_scaled / scale
    return CameraExtrinsics(rotation=rotation, translation=translation)


def triangulate_points(
    projection_matrices: Sequence[NDArray[np.float64]],
    image_points: Sequence[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Triangulate 3-D points from multiple calibrated camera observations."""

    if len(projection_matrices) != len(image_points):
        raise ValueError("Projection matrices and image points must align")
    if not projection_matrices:
        raise ValueError("At least one view is required")

    point_count = image_points[0].shape[0]
    result = np.zeros((point_count, 3), dtype=float)

    for idx in range(point_count):
        rows = []
        for p_mat, img in zip(projection_matrices, image_points):
            if p_mat.shape != (3, 4):
                raise ValueError("Projection matrices must be 3x4")
            u, v = img[idx]
            rows.append(u * p_mat[2, :] - p_mat[0, :])
            rows.append(v * p_mat[2, :] - p_mat[1, :])
        a = np.asarray(rows, dtype=float)
        _, _, vt = np.linalg.svd(a)
        point_h = vt[-1]
        point_h /= point_h[3]
        result[idx] = point_h[:3]
    return result


def reprojection_error(
    points_3d: NDArray[np.float64],
    observations: Sequence[NDArray[np.float64]],
    intrinsics: Sequence[NDArray[np.float64]],
    extrinsics: Sequence[CameraExtrinsics],
) -> float:
    """Return root-mean-square reprojection error across all views."""

    residuals = []
    for obs, k, ext in zip(observations, intrinsics, extrinsics):
        projected = project_points(points_3d, k, ext)
        residuals.append(np.linalg.norm(projected - obs, axis=1))
    concatenated = np.concatenate(residuals)
    return float(np.sqrt(np.mean(concatenated**2)))
