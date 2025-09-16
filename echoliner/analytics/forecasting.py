"""Forecasting utilities for predictive manufacturing analytics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = ["StateSpaceModel", "kalman_forecast", "seasonal_baseline"]


def seasonal_baseline(series: Sequence[float], period: int) -> NDArray[np.float64]:
    if period <= 0:
        raise ValueError("period must be positive")
    series_arr = np.asarray(series, dtype=float)
    usable = series_arr[: series_arr.size // period * period]
    if usable.size < period:
        raise ValueError("series must be longer than the period")
    mean = np.mean(usable.reshape(-1, period), axis=0)
    return mean


@dataclass
class StateSpaceModel:
    transition: NDArray[np.float64]
    control: NDArray[np.float64]
    observation: NDArray[np.float64]
    process_noise: NDArray[np.float64]
    observation_noise: NDArray[np.float64]


def kalman_forecast(
    model: StateSpaceModel,
    observations: Sequence[float],
    *,
    control_input: Sequence[float] | None = None,
) -> NDArray[np.float64]:
    n = len(observations)
    state_dim = model.transition.shape[0]
    states = np.zeros((n + 1, state_dim))
    covariances = np.zeros((n + 1, state_dim, state_dim))
    covariances[0] = np.eye(state_dim)
    control_sequence = (
        np.asarray(control_input, dtype=float) if control_input is not None else np.zeros(n)
    )
    for t in range(n):
        state_pred = model.transition @ states[t]
        if model.control.size > 0:
            state_pred += model.control @ np.array([control_sequence[t]])
        cov_pred = (
            model.transition @ covariances[t] @ model.transition.T + model.process_noise
        )
        innovation = observations[t] - model.observation @ state_pred
        innovation_cov = model.observation @ cov_pred @ model.observation.T + model.observation_noise
        gain = cov_pred @ model.observation.T / innovation_cov
        states[t + 1] = state_pred + gain * innovation
        covariances[t + 1] = cov_pred - gain[:, None] @ model.observation[None, :] @ cov_pred
    forecasts = model.observation @ states[1:].T
    return forecasts
