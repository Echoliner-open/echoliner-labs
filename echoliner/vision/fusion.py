"""Sensor fusion primitives for high-availability cell perception."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from echoliner.common.linalg import clamp_eigenvalues

__all__ = ["AsynchronousFusionEngine", "SensorStream", "UnscentedKalmanFilter"]


@dataclass
class SensorStream:
    """Container representing a sensor data stream."""

    name: str
    frequency_hz: float
    covariance: NDArray[np.float64]
    projection: Callable[[NDArray[np.float64]], NDArray[np.float64]]


class UnscentedKalmanFilter:
    """UKF implementation used for asynchronous sensor fusion."""

    def __init__(
        self,
        state_dim: int,
        measurement_dim: int,
        process_model: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        process_noise: NDArray[np.float64],
        *,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.process_model = process_model
        self.process_noise = process_noise
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.mean = np.zeros(state_dim)
        self.covariance = np.eye(state_dim)

    def _sigma_points(self, mean: NDArray[np.float64], covariance: NDArray[np.float64]) -> NDArray[np.float64]:
        lambda_ = self.alpha**2 * (self.state_dim + self.kappa) - self.state_dim
        sigma_count = 2 * self.state_dim + 1
        sigma = np.zeros((sigma_count, self.state_dim))
        chol = np.linalg.cholesky(covariance * (self.state_dim + lambda_))
        sigma[0] = mean
        for i in range(self.state_dim):
            sigma[i + 1] = mean + chol[i]
            sigma[self.state_dim + i + 1] = mean - chol[i]
        return sigma

    def _weights(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        lambda_ = self.alpha**2 * (self.state_dim + self.kappa) - self.state_dim
        sigma_count = 2 * self.state_dim + 1
        weights_mean = np.full(sigma_count, 1 / (2 * (self.state_dim + lambda_)))
        weights_cov = weights_mean.copy()
        weights_mean[0] = lambda_ / (self.state_dim + lambda_)
        weights_cov[0] = weights_mean[0] + (1 - self.alpha**2 + self.beta)
        return weights_mean, weights_cov

    def predict(self) -> None:
        sigma = self._sigma_points(self.mean, self.covariance)
        propagated = np.array([self.process_model(point) for point in sigma])
        weights_mean, weights_cov = self._weights()
        mean = np.sum(weights_mean[:, None] * propagated, axis=0)
        covariance = np.zeros_like(self.covariance)
        for weight, point in zip(weights_cov, propagated):
            delta = point - mean
            covariance += weight * np.outer(delta, delta)
        covariance += self.process_noise
        self.mean = mean
        self.covariance = clamp_eigenvalues(covariance, 1e-9)

    def update(
        self,
        measurement: NDArray[np.float64],
        projection: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        measurement_noise: NDArray[np.float64],
    ) -> None:
        sigma = self._sigma_points(self.mean, self.covariance)
        weights_mean, weights_cov = self._weights()
        predicted_measurements = np.array([projection(point) for point in sigma])
        measurement_mean = np.sum(weights_mean[:, None] * predicted_measurements, axis=0)
        measurement_cov = np.zeros((self.measurement_dim, self.measurement_dim))
        cross_cov = np.zeros((self.state_dim, self.measurement_dim))
        for weight, sigma_point, measurement_point in zip(weights_cov, sigma, predicted_measurements):
            delta_state = sigma_point - self.mean
            delta_measurement = measurement_point - measurement_mean
            measurement_cov += weight * np.outer(delta_measurement, delta_measurement)
            cross_cov += weight * np.outer(delta_state, delta_measurement)
        measurement_cov += measurement_noise
        measurement_cov = clamp_eigenvalues(measurement_cov, 1e-9)
        kalman_gain = cross_cov @ np.linalg.inv(measurement_cov)
        innovation = measurement - measurement_mean
        self.mean = self.mean + kalman_gain @ innovation
        self.covariance = self.covariance - kalman_gain @ measurement_cov @ kalman_gain.T
        self.covariance = clamp_eigenvalues(self.covariance, 1e-9)


class AsynchronousFusionEngine:
    """Fuse asynchronous sensor streams with a UKF backbone."""

    def __init__(
        self,
        filter_: UnscentedKalmanFilter,
        streams: Sequence[SensorStream],
    ):
        self.filter = filter_
        self.streams = list(streams)
        self._time = 0.0

    def step(self, dt: float) -> None:
        self._time += dt
        self.filter.predict()

    def ingest(self, name: str, measurement: NDArray[np.float64]) -> None:
        for stream in self.streams:
            if stream.name == name:
                self.filter.update(measurement, stream.projection, stream.covariance)
                break

    def state(self) -> NDArray[np.float64]:
        return self.filter.mean

    def covariance(self) -> NDArray[np.float64]:
        return self.filter.covariance
