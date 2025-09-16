"""Analytics toolkit providing OEE and anomaly detection utilities."""

from .digital_twin import CellComponent, DigitalTwin
from .forecasting import StateSpaceModel, kalman_forecast, seasonal_baseline
from .metrics import (
    ProductionRun,
    detect_anomalies_zscore,
    exponential_smoothing_forecast,
    mean_time_between_failures,
    moving_oee,
)
from .streaming import (
    ExponentialMovingStatistics,
    KalmanSmoother,
    StreamingAnomalyDetector,
    generate_synthetic_streams,
)

__all__ = [
    "ProductionRun",
    "detect_anomalies_zscore",
    "exponential_smoothing_forecast",
    "mean_time_between_failures",
    "moving_oee",
    "ExponentialMovingStatistics",
    "KalmanSmoother",
    "StreamingAnomalyDetector",
    "generate_synthetic_streams",
    "CellComponent",
    "DigitalTwin",
    "StateSpaceModel",
    "kalman_forecast",
    "seasonal_baseline",
]
