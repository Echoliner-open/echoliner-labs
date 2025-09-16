"""Vision primitives for EchoLiner's AI-native automation stack."""

from .calibration import (
    CameraExtrinsics,
    compose_projection_matrix,
    estimate_extrinsics_dlt,
    project_points,
    reprojection_error,
    triangulate_points,
)
from .edge_detection import (
    canny_edge_map,
    gradient_magnitude_orientation,
    hysteresis_threshold,
    non_maximum_suppression,
)

__all__ = [
    "CameraExtrinsics",
    "compose_projection_matrix",
    "estimate_extrinsics_dlt",
    "project_points",
    "reprojection_error",
    "triangulate_points",
    "canny_edge_map",
    "gradient_magnitude_orientation",
    "hysteresis_threshold",
    "non_maximum_suppression",
]
