"""Linear algebra helpers used across the stack.

The goal of this module is to provide numerically stable building blocks for
robotics kinematics, multi-view geometry, and probabilistic estimation.  The
functions here intentionally avoid dependencies beyond NumPy so that the
routines can run on constrained industrial controllers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DualQuaternion",
    "axis_angle_from_matrix",
    "axis_angle_to_matrix",
    "clamp_eigenvalues",
    "dual_quaternion_multiply",
    "dual_quaternion_to_matrix",
    "exp_so3",
    "interpolate_dual_quaternions",
    "is_rotation_matrix",
    "log_so3",
    "quaternion_conjugate",
    "quaternion_from_axis_angle",
    "quaternion_multiply",
    "quaternion_normalize",
    "quaternion_to_matrix",
    "skew_symmetric",
]


@dataclass(frozen=True)
class DualQuaternion:
    """Lightweight container for a dual quaternion.

    Attributes
    ----------
    real:
        The real quaternion component, representing rotation.  Expected to be a
        length-4 array `[w, x, y, z]` with a unit norm.
    dual:
        The dual quaternion component encoding translation.
    """

    real: NDArray[np.float64]
    dual: NDArray[np.float64]

    def as_matrix(self) -> NDArray[np.float64]:
        """Convert the dual quaternion into a 4x4 homogeneous transform matrix."""

        return dual_quaternion_to_matrix(self.real, self.dual)


_DEF_TOL = 1e-9


def quaternion_normalize(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a normalized copy of the quaternion.

    Parameters
    ----------
    q:
        Quaternion encoded as `[w, x, y, z]`.
    """

    norm = np.linalg.norm(q)
    if norm < _DEF_TOL:
        raise ValueError("Quaternion norm too small to normalize")
    return q / norm


def quaternion_conjugate(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the conjugate of a quaternion."""

    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quaternion_multiply(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Multiply two quaternions using Hamilton product."""

    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quaternion_to_matrix(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a quaternion to a 3x3 rotation matrix."""

    q_n = quaternion_normalize(q)
    w, x, y, z = q_n
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ],
        dtype=float,
    )


def axis_angle_to_matrix(axis: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    """Return a rotation matrix for an axis-angle description."""

    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,):
        raise ValueError("Axis must be a 3-vector")
    norm = np.linalg.norm(axis)
    if norm < _DEF_TOL:
        raise ValueError("Axis must have non-zero magnitude")
    x, y, z = axis / norm
    c = np.cos(angle)
    s = np.sin(angle)
    c1 = 1 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=float,
    )


def axis_angle_from_matrix(rotation: NDArray[np.float64]) -> Tuple[NDArray[np.float64], float]:
    """Extract the rotation axis and angle from a rotation matrix."""

    if not is_rotation_matrix(rotation):
        raise ValueError("Input must be a valid rotation matrix")
    angle = np.arccos(np.clip((np.trace(rotation) - 1) / 2.0, -1.0, 1.0))
    if abs(angle) < _DEF_TOL:
        return np.array([1.0, 0.0, 0.0], dtype=float), 0.0
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=float,
    )
    axis = axis / np.linalg.norm(axis)
    return axis, angle


def quaternion_from_axis_angle(axis: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    """Convert an axis-angle pair into a quaternion."""

    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,):
        raise ValueError("Axis must be a 3-vector")
    norm = np.linalg.norm(axis)
    if norm < _DEF_TOL:
        raise ValueError("Axis must have non-zero magnitude")
    axis = axis / norm
    half = angle / 2.0
    w = np.cos(half)
    xyz = axis * np.sin(half)
    return np.concatenate([[w], xyz])


def skew_symmetric(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the skew-symmetric matrix associated with the cross product."""

    x, y, z = np.asarray(vector, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def is_rotation_matrix(matrix: NDArray[np.float64]) -> bool:
    """Check whether a matrix is orthonormal with determinant +1."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3):
        return False
    should_be_identity = matrix.T @ matrix
    if not np.allclose(should_be_identity, np.eye(3), atol=1e-6):
        return False
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-6):
        return False
    return True


def exp_so3(omega: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the matrix exponential for so(3) using Rodrigues' formula."""

    theta = np.linalg.norm(omega)
    if theta < _DEF_TOL:
        return np.eye(3)
    axis = omega / theta
    k = skew_symmetric(axis)
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)


def log_so3(rotation: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the logarithm map from SO(3) to so(3)."""

    if not is_rotation_matrix(rotation):
        raise ValueError("Rotation must be orthonormal with determinant +1")
    angle = np.arccos(np.clip((np.trace(rotation) - 1) / 2.0, -1.0, 1.0))
    if angle < _DEF_TOL:
        return np.zeros(3)
    omega = (
        angle
        / (2 * np.sin(angle))
        * np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ],
            dtype=float,
        )
    )
    return omega


def clamp_eigenvalues(matrix: NDArray[np.float64], minimum: float = 1e-6) -> NDArray[np.float64]:
    """Symmetrically clamp eigenvalues to guarantee positive definiteness."""

    sym = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(sym)
    eigenvalues = np.clip(eigenvalues, minimum, None)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def dual_quaternion_multiply(
    a_real: NDArray[np.float64],
    a_dual: NDArray[np.float64],
    b_real: NDArray[np.float64],
    b_dual: NDArray[np.float64],
) -> DualQuaternion:
    """Multiply two dual quaternions."""

    real = quaternion_multiply(a_real, b_real)
    dual = quaternion_multiply(a_real, b_dual) + quaternion_multiply(a_dual, b_real)
    return DualQuaternion(real=real, dual=dual)


def dual_quaternion_to_matrix(real: NDArray[np.float64], dual: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a dual quaternion into a homogeneous transform matrix."""

    rotation = quaternion_to_matrix(real)
    translation = 2 * quaternion_multiply(dual, quaternion_conjugate(real))[1:]
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def interpolate_dual_quaternions(quaternions: Iterable[DualQuaternion], weights: Iterable[float]) -> DualQuaternion:
    """Blend a set of dual quaternions using linear blending with re-normalization."""

    quats = list(quaternions)
    weight_array = np.asarray(list(weights), dtype=float)
    if len(quats) != len(weight_array):
        raise ValueError("Weights must match the number of quaternions")
    if len(quats) == 0:
        raise ValueError("At least one quaternion required")
    weight_array /= weight_array.sum()
    real = np.zeros(4)
    dual = np.zeros(4)
    for dq, w in zip(quats, weight_array):
        real += w * dq.real
        dual += w * dq.dual
    real = quaternion_normalize(real)
    dual = dual - np.dot(real, dual) * real
    return DualQuaternion(real=real, dual=dual)
