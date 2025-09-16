"""Analytics toolkit providing OEE and anomaly detection utilities."""

from .metrics import (
    ProductionRun,
    detect_anomalies_zscore,
    exponential_smoothing_forecast,
    mean_time_between_failures,
    moving_oee,
)

__all__ = [
    "ProductionRun",
    "detect_anomalies_zscore",
    "exponential_smoothing_forecast",
    "mean_time_between_failures",
    "moving_oee",
]
