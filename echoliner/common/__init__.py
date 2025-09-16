"""Common numerical utilities shared across EchoLiner subsystems."""

from .linalg import (
    axis_angle_from_matrix,
    axis_angle_to_matrix,
    clamp_eigenvalues,
    dual_quaternion_multiply,
    dual_quaternion_to_matrix,
    exp_so3,
    interpolate_dual_quaternions,
    is_rotation_matrix,
    log_so3,
    quaternion_conjugate,
    quaternion_from_axis_angle,
    quaternion_multiply,
    quaternion_normalize,
    quaternion_to_matrix,
    skew_symmetric,
)
from .sampling import gaussian_lhs, sobol_fill

__all__ = [
    "axis_angle_from_matrix",
    "axis_angle_to_matrix",
    "clamp_eigenvalues",
    "dual_quaternion_multiply",
    "dual_quaternion_to_matrix",
    "exp_so3",
    "gaussian_lhs",
    "interpolate_dual_quaternions",
    "is_rotation_matrix",
    "log_so3",
    "quaternion_conjugate",
    "quaternion_from_axis_angle",
    "quaternion_multiply",
    "quaternion_normalize",
    "quaternion_to_matrix",
    "skew_symmetric",
    "sobol_fill",
]
