"""Geometric reasoning utilities for multi-view rigs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .calibration import CameraExtrinsics

__all__ = [
    "EssentialDecomposition",
    "decompose_essential_matrix",
    "essential_matrix",
    "fundamental_matrix",
    "normalize_image_points",
    "rectify_pair",
]


@dataclass(frozen=True)
class EssentialDecomposition:
    """Possible relative pose recovered from an essential matrix."""

    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]


def normalize_image_points(points: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return normalized points and the similarity transform."""

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be shaped (N, 2)")
    centroid = np.mean(points, axis=0)
    shifted = points - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))
    scale = np.sqrt(2) / mean_dist if mean_dist > 0 else 1.0
    transform = np.array([[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]])
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    normalized = (transform @ homogeneous.T).T
    return normalized[:, :2], transform


def essential_matrix(
    left_intrinsics: NDArray[np.float64],
    right_intrinsics: NDArray[np.float64],
    left_extrinsics: CameraExtrinsics,
    right_extrinsics: CameraExtrinsics,
) -> NDArray[np.float64]:
    """Compute the essential matrix for a calibrated camera pair."""

    rotation = right_extrinsics.rotation @ left_extrinsics.rotation.T
    translation = right_extrinsics.translation - rotation @ left_extrinsics.translation
    t_x = np.array([[0.0, -translation[2], translation[1]], [translation[2], 0.0, -translation[0]], [-translation[1], translation[0], 0.0]])
    essential = t_x @ rotation
    return essential


def fundamental_matrix(
    left_intrinsics: NDArray[np.float64],
    right_intrinsics: NDArray[np.float64],
    left_extrinsics: CameraExtrinsics,
    right_extrinsics: CameraExtrinsics,
) -> NDArray[np.float64]:
    """Compute the fundamental matrix given calibrated poses."""

    e = essential_matrix(left_intrinsics, right_intrinsics, left_extrinsics, right_extrinsics)
    f = np.linalg.inv(right_intrinsics).T @ e @ np.linalg.inv(left_intrinsics)
    return f


def decompose_essential_matrix(essential: NDArray[np.float64]) -> Sequence[EssentialDecomposition]:
    """Return the four candidate relative poses that satisfy the essential matrix."""

    u, _, vt = np.linalg.svd(essential)
    if np.linalg.det(u) < 0:
        u[:, -1] *= -1
    if np.linalg.det(vt) < 0:
        vt[-1, :] *= -1
    w = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotations = [u @ w @ vt, u @ w.T @ vt]
    translations = [u[:, 2], -u[:, 2]]
    candidates = [EssentialDecomposition(rotation=rot, translation=trans) for rot in rotations for trans in translations]
    return candidates


def rectify_pair(
    left_intrinsics: NDArray[np.float64],
    right_intrinsics: NDArray[np.float64],
    left_extrinsics: CameraExtrinsics,
    right_extrinsics: CameraExtrinsics,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return rectification homographies for a stereo pair."""

    rotation = right_extrinsics.rotation @ left_extrinsics.rotation.T
    translation = right_extrinsics.translation - rotation @ left_extrinsics.translation
    baseline = translation / np.linalg.norm(translation)
    new_x = baseline
    new_z = np.cross(new_x, left_extrinsics.rotation[:, 1])
    new_z /= np.linalg.norm(new_z)
    new_y = np.cross(new_z, new_x)
    rectify_rotation = np.vstack([new_x, new_y, new_z])
    h_left = left_intrinsics @ rectify_rotation @ np.linalg.inv(left_intrinsics)
    h_right = right_intrinsics @ rectify_rotation @ np.linalg.inv(right_intrinsics)
    return h_left, h_right
