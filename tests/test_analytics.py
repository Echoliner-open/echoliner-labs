import numpy as np

from echoliner.analytics import (
    ProductionRun,
    detect_anomalies_zscore,
    exponential_smoothing_forecast,
    mean_time_between_failures,
    moving_oee,
)


def test_production_run_oee_components() -> None:
    run = ProductionRun(
        planned_minutes=480.0,
        stop_minutes=[30.0, 15.0],
        produced_units=900,
        defective_units=30,
        ideal_cycle_time=0.5,
    )
    availability = run.availability()
    performance = run.performance()
    quality = run.quality()
    oee = run.overall_equipment_effectiveness()
    assert np.isclose(run.runtime, 435.0)
    assert np.isclose(availability, 435.0 / 480.0)
    assert np.isclose(performance, 1.0)
    assert np.isclose(quality, (900 - 30) / 900)
    assert np.isclose(oee, availability * quality)


def test_moving_oee_window_average() -> None:
    runs = [
        ProductionRun(480, [30], 900, 30, 0.5),
        ProductionRun(480, [25], 880, 20, 0.5),
        ProductionRun(480, [20], 870, 25, 0.5),
    ]
    rolling = moving_oee(runs, window=2)
    expected = []
    for idx in range(1, len(runs)):
        expected.append(
            (runs[idx - 1].overall_equipment_effectiveness() + runs[idx].overall_equipment_effectiveness())
            / 2
        )
    np.testing.assert_allclose(rolling, expected)


def test_detect_anomalies_identifies_outlier() -> None:
    series = [100, 102, 101, 300, 99, 98, 102]
    indices = detect_anomalies_zscore(series, threshold=2.5, minimum_points=5)
    assert indices == [3]


def test_exponential_smoothing_forecast_returns_constant_vector() -> None:
    forecast = exponential_smoothing_forecast([10, 12, 18], alpha=0.6, horizon=3)
    np.testing.assert_allclose(forecast, np.array([15.28, 15.28, 15.28]))


def test_mtbf_computes_mean_segment_length() -> None:
    segments = [120.0, 95.0, 130.0]
    assert np.isclose(mean_time_between_failures(segments), np.mean(segments))
