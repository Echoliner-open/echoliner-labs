"""Analytics primitives for EchoLiner's production intelligence layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

__all__ = [
    "ProductionRun",
    "moving_oee",
    "detect_anomalies_zscore",
    "exponential_smoothing_forecast",
    "mean_time_between_failures",
]


@dataclass(frozen=True)
class ProductionRun:
    """Snapshot of a production interval used for OEE calculations."""

    planned_minutes: float
    stop_minutes: Sequence[float]
    produced_units: int
    defective_units: int
    ideal_cycle_time: float

    def __post_init__(self) -> None:
        if self.planned_minutes <= 0:
            raise ValueError("Planned minutes must be positive")
        if any(s < 0 for s in self.stop_minutes):
            raise ValueError("Stop durations must be non-negative")
        if self.produced_units < 0 or self.defective_units < 0:
            raise ValueError("Unit counts must be non-negative")
        if self.defective_units > self.produced_units:
            raise ValueError("Defects cannot exceed produced units")
        if self.ideal_cycle_time <= 0:
            raise ValueError("Ideal cycle time must be positive")

    @property
    def runtime(self) -> float:
        runtime = self.planned_minutes - float(np.sum(self.stop_minutes))
        return max(runtime, 0.0)

    def availability(self) -> float:
        return self.runtime / self.planned_minutes

    def performance(self) -> float:
        if self.runtime == 0:
            return 0.0
        return min((self.ideal_cycle_time * self.produced_units) / self.runtime, 1.0)

    def quality(self) -> float:
        if self.produced_units == 0:
            return 0.0
        return (self.produced_units - self.defective_units) / self.produced_units

    def overall_equipment_effectiveness(self) -> float:
        return self.availability() * self.performance() * self.quality()


def moving_oee(runs: Sequence[ProductionRun], window: int) -> List[float]:
    """Rolling OEE average over the supplied production runs."""

    if window <= 0:
        raise ValueError("Window must be positive")
    if window > len(runs):
        raise ValueError("Window cannot exceed the number of runs")
    oee_values = [run.overall_equipment_effectiveness() for run in runs]
    result: List[float] = []
    for idx in range(window - 1, len(oee_values)):
        result.append(float(np.mean(oee_values[idx - window + 1 : idx + 1])))
    return result


def detect_anomalies_zscore(
    series: Sequence[float],
    *,
    threshold: float = 3.0,
    minimum_points: int = 5,
) -> List[int]:
    """Indices of points exceeding a z-score or robust MAD threshold."""

    if len(series) < minimum_points:
        return []
    values = np.asarray(series, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std > 0:
        zscores = (values - mean) / std
    else:
        zscores = np.zeros_like(values)

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 0:
        modified = 0.6745 * (values - median) / mad
    else:
        modified = np.zeros_like(values)

    mask = (np.abs(zscores) >= threshold) | (np.abs(modified) >= threshold)
    return [idx for idx, flag in enumerate(mask) if flag]


def exponential_smoothing_forecast(
    series: Sequence[float],
    *,
    alpha: float,
    horizon: int,
) -> np.ndarray:
    """Return a single-step exponential smoothing forecast over the horizon."""

    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in the interval (0, 1]")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not series:
        raise ValueError("series must not be empty")

    iterator = iter(series)
    forecast = float(next(iterator))
    for value in iterator:
        forecast = alpha * float(value) + (1 - alpha) * forecast
    return np.full(horizon, forecast, dtype=float)


def mean_time_between_failures(uptime_segments: Iterable[float]) -> float:
    """Mean time between failures given individual uptime segments."""

    segments = [seg for seg in uptime_segments if seg >= 0]
    if not segments:
        raise ValueError("At least one non-negative segment is required")
    return float(np.mean(segments))
