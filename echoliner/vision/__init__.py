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
from .fusion import AsynchronousFusionEngine, SensorStream, UnscentedKalmanFilter
from .geometry import (
    EssentialDecomposition,
    decompose_essential_matrix,
    essential_matrix,
    fundamental_matrix,
    normalize_image_points,
    rectify_pair,
)
from .reconstruction import BundleAdjustmentProblem, IterativeTriangulator
from .tracking import ConstantVelocityModel, JPDAF, KalmanTracker

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
    "essential_matrix",
    "fundamental_matrix",
    "decompose_essential_matrix",
    "normalize_image_points",
    "rectify_pair",
    "IterativeTriangulator",
    "BundleAdjustmentProblem",
    "ConstantVelocityModel",
    "KalmanTracker",
    "JPDAF",
    "AsynchronousFusionEngine",
    "UnscentedKalmanFilter",
    "SensorStream",
]
