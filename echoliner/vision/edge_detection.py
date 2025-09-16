"""Simple image processing utilities for EchoLiner vision module.

This module currently implements a Sobel edge detector using NumPy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sobel_edges(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute edge magnitude using the Sobel operator.

    Parameters
    ----------
    image:
        Grayscale image represented as a 2-D NumPy array with dtype float.

    Returns
    -------
    edges:
        Edge magnitude of the image using the Sobel filter.
    """
    if image.ndim != 2:
        raise ValueError("Image must be a 2-D grayscale array")

    # Sobel kernels
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)

    # Convolve using padding
    padded = np.pad(image, ((1, 1), (1, 1)), mode="edge")
    gx = np.zeros_like(image)
    gy = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i : i + 3, j : j + 3]
            gx[i, j] = np.sum(region * kx)
            gy[i, j] = np.sum(region * ky)

    edges = np.hypot(gx, gy)
    return edges
