import numpy as np

import numpy as np

from echoliner.analytics import (
    CellComponent,
    DigitalTwin,
    ProductionRun,
    StateSpaceModel,
    StreamingAnomalyDetector,
    detect_anomalies_zscore,
    exponential_smoothing_forecast,
    generate_synthetic_streams,
    kalman_forecast,
    mean_time_between_failures,
    moving_oee,
)
from echoliner.robotics import DHLink, KinematicChain


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


def test_streaming_detector_adapts_threshold() -> None:
    detector = StreamingAnomalyDetector(window=3, threshold=2.0)
    baseline = [100.0, 101.0, 100.5, 99.5, 100.2]
    detector.bootstrap_threshold(baseline, false_positive_rate=0.2)
    triggered = [bool(detector.update(value)) for value in baseline]
    assert sum(triggered) < len(baseline)
    assert detector.update(120.0)


def test_generate_synthetic_streams_shape() -> None:
    streams = list(generate_synthetic_streams(probes=4, dimensions=3, duration=1.0, seed=0))
    assert len(streams) == 4
    for row in streams:
        assert row.shape == (3,)


def test_kalman_forecast_tracks_process() -> None:
    model = StateSpaceModel(
        transition=np.array([[1.0]]),
        control=np.array([[0.0]]),
        observation=np.array([1.0]),
        process_noise=np.array([[0.01]]),
        observation_noise=np.array([[0.1]]),
    )
    observations = [1.0, 1.2, 1.1, 0.9, 1.0]
    forecasts = kalman_forecast(model, observations)
    assert forecasts.shape == (len(observations),)
    assert np.all(np.isfinite(forecasts))


def test_digital_twin_generates_demand_gap() -> None:
    twin = DigitalTwin(
        components=[CellComponent("assembly", 1.0, 0.0, 0.0)],
        manipulator=KinematicChain([DHLink(a=0.3, alpha=0.0, d=0.0)]),
        shift_minutes=10.0,
    )
    metrics = twin.run(demand=50.0, time_step=1.0)
    assert "demand_gap" in metrics
    assert metrics["demand_gap"].shape == metrics["throughput"].shape
