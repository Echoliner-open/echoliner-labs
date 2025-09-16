"""Low-discrepancy sampling schemes for design of experiments."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import NDArray

__all__ = ["gaussian_lhs", "sobol_fill"]


def gaussian_lhs(dimensions: int, samples: int, seed: int | None = None) -> NDArray[np.float64]:
    """Generate a Latin hypercube design transformed through a Gaussian quantile.

    Parameters
    ----------
    dimensions:
        Number of dimensions in the design space.
    samples:
        Number of samples to draw.
    seed:
        Optional random seed for reproducibility.
    """

    if dimensions <= 0 or samples <= 0:
        raise ValueError("dimensions and samples must be positive integers")
    rng = np.random.default_rng(seed)
    cut = np.linspace(0, 1, samples + 1)
    u = rng.uniform(low=cut[:-1], high=cut[1:], size=(dimensions, samples)).T
    for j in range(dimensions):
        rng.shuffle(u[:, j])
    gaussian = np.sqrt(2) * _erfinv(2 * u - 1)
    return gaussian


def _erfinv(values: np.ndarray) -> np.ndarray:
    a = 0.147
    clipped = np.clip(values, -0.999999, 0.999999)
    ln = np.log(1 - clipped**2)
    term = 2 / (np.pi * a) + ln / 2
    return np.sign(clipped) * np.sqrt(np.sqrt(term**2 - ln / a) - term)


def _gray_code(n: int) -> list[list[int]]:
    if n == 1:
        return [[0], [1]]
    prev = _gray_code(n - 1)
    mirrored = [code.copy() for code in reversed(prev)]
    return [[0] + code for code in prev] + [[1] + code for code in mirrored]


def _primitive_polynomials(dimension: int) -> list[int]:
    """Return the coefficients for primitive polynomials up to the given dimension."""

    # Precomputed primitive polynomials for Sobol sequence up to dimension 10.
    primitives = {
        1: [1],
        2: [1, 1],
        3: [1, 1],
        4: [1, 2],
        5: [1, 1],
        6: [1, 1],
        7: [1, 3],
        8: [1, 2],
        9: [1, 4],
        10: [1, 2],
    }
    coeffs = primitives.get(dimension)
    if coeffs is None:
        raise ValueError("Primitive polynomial not tabulated for dimension" f" {dimension}")
    return coeffs


def sobol_fill(dimension: int, samples: int, scramble: bool = True, seed: int | None = None) -> NDArray[np.float64]:
    """Generate a Sobol sequence and optionally apply Owen scrambling."""

    if dimension <= 0 or samples <= 0:
        raise ValueError("dimension and samples must be positive integers")
    if dimension > 10:
        raise ValueError("sobol_fill supports dimensions up to 10 in this implementation")

    rng = np.random.default_rng(seed)
    direction_numbers = np.zeros((dimension, 32), dtype=np.uint32)
    for dim in range(dimension):
        coeffs = _primitive_polynomials(dim + 1)
        degree = len(coeffs) - 1
        for i in range(1, degree + 1):
            direction_numbers[dim, i - 1] = 1 << (32 - i)
        for i in range(degree + 1, 32):
            value = direction_numbers[dim, i - degree - 1] << degree
            for j in range(1, degree + 1):
                value ^= coeffs[j] << (degree - j)
            direction_numbers[dim, i] = value

    sobol = np.zeros((samples, dimension))
    x = np.zeros(dimension, dtype=np.uint32)
    for i in range(samples):
        lsb = (i + 1) & -(i + 1)
        bit = int(np.log2(lsb))
        x ^= direction_numbers[:, bit]
        sobol[i] = x / 2**32

    if scramble:
        scramble_bits = rng.integers(0, 2**32, size=dimension, dtype=np.uint32)
        sobol ^= scramble_bits
        sobol = sobol / 2**32 + rng.random(size=sobol.shape) / 2**32
    return sobol.astype(float)
