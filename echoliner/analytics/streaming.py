"""Streaming analytics and telemetry pipelines."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from echoliner.common.sampling import gaussian_lhs

__all__ = [
    "ExponentialMovingStatistics",
    "KalmanSmoother",
    "StreamingAnomalyDetector",
    "generate_synthetic_streams",
]


@dataclass
class ExponentialMovingStatistics:
    """Maintain exponential moving averages and variances."""

    alpha: float
    mean: float = 0.0
    variance: float = 0.0
    initialized: bool = False

    def update(self, value: float) -> tuple[float, float]:
        if not self.initialized:
            self.mean = value
            self.variance = 0.0
            self.initialized = True
            return self.mean, self.variance
        delta = value - self.mean
        self.mean += self.alpha * delta
        self.variance = (1 - self.alpha) * (self.variance + self.alpha * delta**2)
        return self.mean, self.variance


class KalmanSmoother:
    """Scalar Kalman smoother for streaming signals."""

    def __init__(self, process_variance: float = 1e-2, measurement_variance: float = 1e-1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 0.0
        self.error = 1.0

    def update(self, measurement: float) -> float:
        prediction_error = self.error + self.process_variance
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate += kalman_gain * (measurement - self.estimate)
        self.error = (1 - kalman_gain) * prediction_error
        return self.estimate


@dataclass
class StreamingAnomalyDetector:
    """Detect anomalies in streaming telemetry with adaptive thresholds."""

    window: int
    threshold: float = 3.0

    def __post_init__(self) -> None:
        self._buffer: Deque[float] = deque(maxlen=self.window)
        self._stats = ExponentialMovingStatistics(alpha=0.1)

    def update(self, value: float) -> bool:
        self._buffer.append(value)
        mean, variance = self._stats.update(value)
        if len(self._buffer) < self.window:
            return False
        std = max(np.sqrt(variance), 1e-6)
        z = abs(value - mean) / std
        return z >= self.threshold

    def bootstrap_threshold(self, baseline: Sequence[float], false_positive_rate: float = 0.01) -> None:
        samples = np.sort(np.abs(np.asarray(baseline) - np.mean(baseline)))
        index = int((1 - false_positive_rate) * len(samples))
        self.threshold = samples[min(index, len(samples) - 1)]


def generate_synthetic_streams(
    probes: int,
    dimensions: int,
    duration: float,
    *,
    seed: int | None = None,
) -> Iterator[NDArray[np.float64]]:
    design = gaussian_lhs(dimensions, probes, seed=seed)
    time = np.linspace(0, duration, probes)
    for row, t in zip(design, time):
        yield np.sin(2 * np.pi * row * t)
