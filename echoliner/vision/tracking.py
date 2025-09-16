"""Multi-object tracking utilities for production cells."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from echoliner.common.linalg import clamp_eigenvalues

__all__ = [
    "ConstantVelocityModel",
    "JPDAF",
    "KalmanHypothesis",
    "KalmanTracker",
]


@dataclass
class ConstantVelocityModel:
    """Planar constant-velocity motion model for conveyor tracking."""

    dt: float = 0.033
    process_variance: float = 1e-2
    measurement_variance: float = 4e-2

    def transition_matrix(self) -> NDArray[np.float64]:
        return np.array(
            [
                [1.0, 0.0, self.dt, 0.0],
                [0.0, 1.0, 0.0, self.dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def process_noise(self) -> NDArray[np.float64]:
        q = self.process_variance
        dt = self.dt
        return q * np.array(
            [
                [dt**4 / 4, 0.0, dt**3 / 2, 0.0],
                [0.0, dt**4 / 4, 0.0, dt**3 / 2],
                [dt**3 / 2, 0.0, dt**2, 0.0],
                [0.0, dt**3 / 2, 0.0, dt**2],
            ]
        )

    def measurement_matrix(self) -> NDArray[np.float64]:
        return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    def measurement_noise(self) -> NDArray[np.float64]:
        r = self.measurement_variance
        return r * np.eye(2)


@dataclass
class KalmanHypothesis:
    """Single Kalman filter state and covariance."""

    state: NDArray[np.float64]
    covariance: NDArray[np.float64]
    age: int = 0
    hits: int = 0
    miss_streak: int = 0


@dataclass
class KalmanTracker:
    """Multi-target Kalman tracker with track management."""

    model: ConstantVelocityModel
    gate_threshold: float = 9.21  # Chi-square 95% for 2 DOF
    max_age: int = 30
    min_hits: int = 3
    tracks: Dict[int, KalmanHypothesis] = field(default_factory=dict)
    _next_id: int = 1

    def predict(self) -> None:
        f = self.model.transition_matrix()
        q = self.model.process_noise()
        for track in self.tracks.values():
            track.state = f @ track.state
            track.covariance = f @ track.covariance @ f.T + q
            track.age += 1
            track.miss_streak += 1

    def gating_distance(self, track: KalmanHypothesis, measurement: NDArray[np.float64]) -> float:
        h = self.model.measurement_matrix()
        innovation = measurement - h @ track.state
        s = h @ track.covariance @ h.T + self.model.measurement_noise()
        s = clamp_eigenvalues(s, 1e-6)
        distance = float(innovation.T @ np.linalg.inv(s) @ innovation)
        return distance

    def update(self, measurements: Sequence[NDArray[np.float64]]) -> Dict[int, NDArray[np.float64]]:
        h = self.model.measurement_matrix()
        r = self.model.measurement_noise()
        associations: Dict[int, NDArray[np.float64]] = {}
        assigned_measurements: set[int] = set()
        for track_id, track in list(self.tracks.items()):
            best_index: Optional[int] = None
            best_distance = float("inf")
            for idx, measurement in enumerate(measurements):
                if idx in assigned_measurements:
                    continue
                distance = self.gating_distance(track, measurement)
                if distance < self.gate_threshold and distance < best_distance:
                    best_distance = distance
                    best_index = idx
            if best_index is not None:
                measurement = measurements[best_index]
                innovation = measurement - h @ track.state
                s = h @ track.covariance @ h.T + r
                k = track.covariance @ h.T @ np.linalg.inv(s)
                track.state = track.state + k @ innovation
                track.covariance = (np.eye(track.covariance.shape[0]) - k @ h) @ track.covariance
                track.hits += 1
                track.miss_streak = 0
                associations[track_id] = measurement
                assigned_measurements.add(best_index)
        for idx, measurement in enumerate(measurements):
            if idx not in assigned_measurements:
                self.start_track(measurement)
        self._prune_tracks()
        return associations

    def start_track(self, measurement: NDArray[np.float64]) -> int:
        f = self.model.transition_matrix()
        state = np.zeros(4)
        state[:2] = measurement
        covariance = np.eye(4)
        track_id = self._next_id
        self._next_id += 1
        self.tracks[track_id] = KalmanHypothesis(state=state, covariance=covariance)
        return track_id

    def _prune_tracks(self) -> None:
        to_delete = [track_id for track_id, track in self.tracks.items() if track.miss_streak > self.max_age]
        for track_id in to_delete:
            del self.tracks[track_id]

    def confirmed_tracks(self) -> Dict[int, KalmanHypothesis]:
        return {track_id: track for track_id, track in self.tracks.items() if track.hits >= self.min_hits}


class JPDAF:
    """Joint probabilistic data association filter wrapper."""

    def __init__(self, tracker: KalmanTracker):
        self.tracker = tracker

    def _association_probabilities(
        self, track: KalmanHypothesis, measurements: Sequence[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        scores = []
        for measurement in measurements:
            distance = self.tracker.gating_distance(track, measurement)
            scores.append(np.exp(-0.5 * distance))
        scores.append(0.05)  # clutter hypothesis
        total = sum(scores)
        return np.asarray(scores) / total

    def update(self, measurements: Sequence[NDArray[np.float64]]) -> Dict[int, NDArray[np.float64]]:
        self.tracker.predict()
        associations: Dict[int, NDArray[np.float64]] = {}
        associated_indices: set[int] = set()
        h = self.tracker.model.measurement_matrix()
        r = self.tracker.model.measurement_noise()
        for track_id, track in list(self.tracker.tracks.items()):
            probabilities = self._association_probabilities(track, measurements)
            expected_measurement = np.zeros(2)
            innovation_cov = np.zeros((2, 2))
            for idx, (prob, measurement) in enumerate(zip(probabilities[:-1], measurements)):
                expected_measurement += prob * measurement
                delta = measurement - h @ track.state
                innovation_cov += prob * np.outer(delta, delta)
                if prob > 0.25:
                    associated_indices.add(idx)
            innovation_cov += r
            innovation = expected_measurement - h @ track.state
            s = h @ track.covariance @ h.T + innovation_cov
            s = clamp_eigenvalues(s, 1e-6)
            k = track.covariance @ h.T @ np.linalg.inv(s)
            track.state = track.state + k @ innovation
            track.covariance = (np.eye(track.covariance.shape[0]) - k @ h) @ track.covariance
            track.hits += 1
            track.miss_streak = 0
            associations[track_id] = expected_measurement
        for idx, measurement in enumerate(measurements):
            if idx not in associated_indices:
                self.tracker.start_track(measurement)
        self.tracker._prune_tracks()
        return associations
