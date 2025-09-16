"""Sequence alignment utilities for bilingual corpora."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = ["DynamicTimeWarping", "alignment_path", "soft_alignment"]


@dataclass
class DynamicTimeWarping:
    """Dynamic time warping for variable length embeddings."""

    distance_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float]

    def compute(self, source: NDArray[np.float64], target: NDArray[np.float64]) -> Tuple[float, List[Tuple[int, int]]]:
        n, m = source.shape[0], target.shape[0]
        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dist = self.distance_fn(source[i - 1], target[j - 1])
                cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
        path = alignment_path(cost)
        return cost[n, m], path


def alignment_path(cost_matrix: NDArray[np.float64]) -> List[Tuple[int, int]]:
    i, j = cost_matrix.shape[0] - 1, cost_matrix.shape[1] - 1
    path: List[Tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        steps = [cost_matrix[i - 1, j], cost_matrix[i, j - 1], cost_matrix[i - 1, j - 1]]
        step = int(np.argmin(steps))
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    path.reverse()
    return path


def soft_alignment(source: NDArray[np.float64], target: NDArray[np.float64], temperature: float = 0.1) -> NDArray[np.float64]:
    similarities = source @ target.T
    scaled = similarities / temperature
    exp = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    alignment = exp / exp.sum(axis=1, keepdims=True)
    return alignment
