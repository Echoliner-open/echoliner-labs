import numpy as np

from echoliner.vision import (
    CameraExtrinsics,
    canny_edge_map,
    compose_projection_matrix,
    estimate_extrinsics_dlt,
    project_points,
    triangulate_points,
)


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    c1 = 1 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ]
    )


def test_canny_edge_map_detects_vertical_interface() -> None:
    image = np.zeros((16, 16), dtype=float)
    image[:, 8:] = 1.0
    edges = canny_edge_map(image, low=0.05, high=0.1)
    assert edges.dtype == bool
    assert edges[:, 8].any() or edges[:, 7].any()


def test_estimate_extrinsics_dlt_reconstructs_pose() -> None:
    intrinsics = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    rotation = _rotation_matrix(np.array([0.2, -0.5, 0.7]), 0.35)
    translation = np.array([0.15, -0.08, 1.6])
    extrinsics = CameraExtrinsics(rotation=rotation, translation=translation)
    world_points = np.array(
        [
            [-0.2, -0.2, 0.0],
            [-0.2, 0.2, 0.0],
            [0.2, -0.2, 0.0],
            [0.2, 0.2, 0.0],
            [-0.2, -0.2, 0.4],
            [-0.2, 0.2, 0.4],
            [0.2, -0.2, 0.4],
            [0.2, 0.2, 0.4],
        ]
    )
    image_points = project_points(world_points, intrinsics, extrinsics)
    estimated = estimate_extrinsics_dlt(world_points, image_points, intrinsics)
    assert np.allclose(estimated.rotation, rotation, atol=1e-3)
    assert np.allclose(estimated.translation, translation, atol=1e-3)


def test_triangulate_points_matches_ground_truth() -> None:
    intrinsics = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    left_extr = CameraExtrinsics(rotation=np.eye(3), translation=np.zeros(3))
    right_extr = CameraExtrinsics(rotation=np.eye(3), translation=np.array([-0.2, 0.0, 0.0]))
    world_points = np.array([[0.0, 0.0, 2.0], [0.05, -0.04, 1.6]])
    observations_left = project_points(world_points, intrinsics, left_extr)
    observations_right = project_points(world_points, intrinsics, right_extr)
    projection_left = compose_projection_matrix(intrinsics, left_extr)
    projection_right = compose_projection_matrix(intrinsics, right_extr)
    triangulated = triangulate_points(
        [projection_left, projection_right],
        [observations_left, observations_right],
    )
    assert np.allclose(triangulated, world_points, atol=1e-3)
