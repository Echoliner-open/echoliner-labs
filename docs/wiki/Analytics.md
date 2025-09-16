# Analytics & Digital Twins

Operations teams rely on the analytics module to benchmark equipment, detect
anomalies, and rehearse process changes in a safe environment.  The tooling is
split between classic KPI utilities and the stochastic digital twin.

## KPI & Forecasting Toolkit

* `ProductionRun` surfaces availability, performance, and quality metrics used
  to compute OEE dashboards.  Pair it with `moving_oee` when you need rolling
  averages for executive scorecards.
* `StreamingAnomalyDetector` applies adaptive z-score monitoring over streaming
  tags.  Calibrate thresholds via `bootstrap_threshold` using historical data
  before connecting to live PLC signals.
* `StateSpaceModel` and `kalman_forecast` implement Kalman-based forecasting for
  scrap rate, throughput, or energy consumption series.  The API accepts control
  inputs so that you can explore new shift schedules offline.

## Digital Twin Architecture

`DigitalTwin` aggregates component reliability models, manipulator motion, and
energy accounting to simulate an entire shift.  Each `CellComponent` encodes
cycle time, failure characteristics, quality yield, and energy per cycle, while
the manipulator model logs joint positions, velocities, and torque demand.

```python
from echoliner.analytics import CellComponent, DigitalTwin
from echoliner.robotics import DHLink, KinematicChain

components = [
    CellComponent("assembly", cycle_time=0.9, failure_rate=0.015, repair_time=4.0),
    CellComponent("inspection", cycle_time=1.2, failure_rate=0.008, repair_time=6.0,
                  quality_yield=0.998, energy_per_cycle=0.45),
]
chain = KinematicChain([DHLink(a=0.35, alpha=0.0, d=0.0)])
twin = DigitalTwin(components=components, manipulator=chain, shift_minutes=480.0)
metrics = twin.run(demand=420.0, time_step=1.0)
```

* `metrics["first_pass_yield"]` reports the evolving quality rate, while
  `metrics["condition_index"]` captures component health degradation.
* When you supply a `JointDynamics` instance via `DigitalTwin(..., dynamics=...)`,
  the twin also returns `torque_norm`, enabling actuator loading studies.
* Use `maintenance_windows` to reset component health mid-shift when modelling
  scheduled inspections.

## Monte Carlo Analysis

`run_monte_carlo` executes repeated simulations with fresh random seeds so that
engineering and operations leads can quantify variability before rolling out a
process change.

```python
summary = twin.run_monte_carlo(demand=420.0, trials=10, time_step=1.0)
throughput_mean = summary["throughput"]["mean"]
throughput_std = summary["throughput"]["std"]
```

* The returned dictionaries include `mean` and `std` arrays for every channel—
  throughput, downtime, quality output, energy usage, torque demand, and more.
* Use the statistics to derive service-level agreements or determine buffer
  sizes when introducing new product variants.

Return to the [wiki index](README.md) or continue with the
[Translation Layer](Translation.md).
