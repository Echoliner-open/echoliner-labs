"""Advanced edge detection utilities for the EchoLiner vision stack.

The module implements Sobel gradients, non-maximum suppression, and a
light-weight Canny style hysteresis stage. These routines intentionally avoid
external dependencies so they can run on embedded edge devices that power
EchoLiner's modular robotics platforms.
"""

from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

__all__ = [
    "gradient_magnitude_orientation",
    "non_maximum_suppression",
    "hysteresis_threshold",
    "canny_edge_map",
]

_SOBEL_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
_SOBEL_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)


def _validate_image(image: NDArray[np.float64]) -> NDArray[np.float64]:
    if image.ndim != 2:
        raise ValueError("Image must be a 2-D grayscale array")
    return image.astype(float, copy=False)


def _convolve(image: NDArray[np.float64], kernel: NDArray[np.float64]) -> NDArray[np.float64]:
    padded = np.pad(image, ((1, 1), (1, 1)), mode="edge")
    windows = sliding_window_view(padded, kernel.shape)
    return np.tensordot(windows, kernel, axes=((2, 3), (0, 1)))


def gradient_magnitude_orientation(
    image: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return gradient magnitude and orientation for a grayscale image.

    Parameters
    ----------
    image:
        Two-dimensional grayscale image with intensities normalized to
        ``[0, 1]``. The function accepts any floating point dtype and performs
        computations in ``float64`` precision.

    Returns
    -------
    magnitude, orientation:
        Tuple containing the L2 gradient magnitude and the gradient orientation
        in radians within ``[-pi, pi)``.
    """

    image = _validate_image(image)
    gx = _convolve(image, _SOBEL_X)
    gy = _convolve(image, _SOBEL_Y)
    magnitude = np.hypot(gx, gy)
    orientation = np.arctan2(gy, gx)
    return magnitude, orientation


def non_maximum_suppression(
    magnitude: NDArray[np.float64],
    orientation: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Thin gradients using non-maximum suppression along local directions."""

    if magnitude.shape != orientation.shape:
        raise ValueError("Magnitude and orientation must share the same shape")

    suppressed = np.zeros_like(magnitude)
    rows, cols = magnitude.shape
    orientation_deg = (np.degrees(orientation) + 180.0) % 180.0

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle = orientation_deg[i, j]
            if (0 <= angle < 22.5) or (157.5 <= angle < 180):
                neighbors = (magnitude[i, j - 1], magnitude[i, j + 1])
            elif 22.5 <= angle < 67.5:
                neighbors = (magnitude[i - 1, j + 1], magnitude[i + 1, j - 1])
            elif 67.5 <= angle < 112.5:
                neighbors = (magnitude[i - 1, j], magnitude[i + 1, j])
            else:
                neighbors = (magnitude[i - 1, j - 1], magnitude[i + 1, j + 1])

            if magnitude[i, j] >= neighbors[0] and magnitude[i, j] >= neighbors[1]:
                suppressed[i, j] = magnitude[i, j]
    return suppressed


def hysteresis_threshold(
    magnitude: NDArray[np.float64],
    low: float,
    high: float,
) -> NDArray[np.bool_]:
    """Binary mask of edges after hysteresis thresholding."""

    if high <= low:
        raise ValueError("High threshold must be greater than low threshold")

    strong = magnitude >= high
    weak = (magnitude >= low) & ~strong
    edges = np.zeros_like(magnitude, dtype=bool)

    queue: deque[Tuple[int, int]] = deque(zip(*np.nonzero(strong)))
    while queue:
        r, c = queue.popleft()
        if edges[r, c]:
            continue
        edges[r, c] = True
        for nr in range(max(0, r - 1), min(magnitude.shape[0], r + 2)):
            for nc in range(max(0, c - 1), min(magnitude.shape[1], c + 2)):
                if weak[nr, nc] and not edges[nr, nc]:
                    queue.append((nr, nc))
    return edges


def canny_edge_map(
    image: NDArray[np.float64],
    low: float = 0.1,
    high: float = 0.3,
) -> NDArray[np.bool_]:
    """Produce a binarized edge map from the supplied grayscale image."""

    magnitude, orientation = gradient_magnitude_orientation(image)
    thinned = non_maximum_suppression(magnitude, orientation)
    return hysteresis_threshold(thinned, low=low, high=high)
