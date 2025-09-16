# EchoLiner Labs – AI-Native Modular Manufacturing Platform

EchoLiner Labs curates the research implementations that underpin our
AI-native manufacturing automation stack.  The repository exposes the same
vision, robotics, translation, and analytics kernels that power our production
systems so the community can prototype, benchmark, and extend high-mix
automation workflows.

The project is intentionally ambitious: more than twenty thousand lines of
Python span dense bilingual lexicons, state-of-the-art perception pipelines,
manipulator dynamics, speech-to-speech alignment tooling, and high-resolution
factory analytics.  Each submodule is designed for interdisciplinary teams that
mix robotics engineers, computer vision scientists, linguists, and production
operations researchers.

## Repository Layout

```
.
├── echoliner/
│   ├── analytics/     # Streaming telemetry, forecasting, and digital twin models
│   ├── common/        # SO(3) math, dual quaternions, low-discrepancy sampling
│   ├── robotics/      # Kinematics, dynamics, controllers, and motion planners
│   ├── translation/   # Bilingual lexicon, IBM-style translator, speech features
│   └── vision/        # Calibration, multi-view geometry, tracking, sensor fusion
├── tests/             # Comprehensive module exercises and regression tests
└── scripts/           # Data-generation utilities (e.g., 20k+ bilingual lexicon)
```

Every package exposes production-ready APIs via `__all__` exports so you can
compose subsystems directly from `echoliner.<module>` namespaces.

## Installation & Setup

EchoLiner targets Python 3.11+ with NumPy as the only mandatory dependency.
Optional tooling (SciPy, PyTorch, etc.) can be layered on top for downstream
experiments but is not required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[test]
```

To validate the stack locally run:

```bash
pytest
```

All unit tests exercise the differentiable geometry routines, robotics
controllers, translation metrics, and analytics primitives to ensure that
non-regression guarantees are maintained while the codebase evolves.

## Vision Systems

The `echoliner.vision` namespace provides a modular perception stack that can
calibrate multi-camera cells, reconstruct geometry, and track industrial assets
in real time.

### Calibration & Geometry

* `CameraExtrinsics` – immutable rigid transformations for camera poses.
* `compose_projection_matrix`, `project_points`, `triangulate_points` – classic
  projective geometry primitives for multi-view rigs.
* `essential_matrix`, `decompose_essential_matrix`, `rectify_pair` – epipolar
  geometry utilities that operate on calibrated stereo heads.
* `BundleAdjustmentProblem` & `IterativeTriangulator` – Gauss–Newton bundle
  adjustment paired with robustified triangulation for sparse SfM setups.

### Edge & Feature Intelligence

* `canny_edge_map` – vectorized Sobel gradients + non-maximum suppression
  + hysteresis for embedded-friendly edge maps.
* `ConstantVelocityModel`, `KalmanTracker`, `JPDAF` – multi-target tracking with
  gating, joint probabilistic data association, and track management.
* `UnscentedKalmanFilter`, `AsynchronousFusionEngine` – asynchronous sensor
  fusion across heterogeneous sensing modalities (vision + lidar + IMUs).

### Dense Fusion & Mapping

* `VolumetricTSDF` – online truncated signed distance fusion that turns raw
  depth imagery into metrically consistent occupancy and surface reconstructions
  for metrology, inspection, and workcell simulation.

### Vision Quick Start

```python
import numpy as np
from echoliner.vision import (
    CameraExtrinsics, VolumetricTSDF, compose_projection_matrix, essential_matrix,
    estimate_extrinsics_dlt, project_points, ConstantVelocityModel,
    KalmanTracker, SensorStream, UnscentedKalmanFilter, AsynchronousFusionEngine
)

# Calibrate a stereo pair
intrinsics = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
left_pose = CameraExtrinsics(rotation=np.eye(3), translation=np.zeros(3))
right_pose = CameraExtrinsics(rotation=np.eye(3), translation=np.array([0.1, 0.0, 0.0]))
world = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
left_pixels = project_points(world, intrinsics, left_pose)
right_pixels = project_points(world, intrinsics, right_pose)
recovered = estimate_extrinsics_dlt(world, left_pixels, intrinsics)
E = essential_matrix(intrinsics, intrinsics, left_pose, right_pose)

# Track pallets moving through the cell
model = ConstantVelocityModel(dt=0.033)
tracker = KalmanTracker(model)
tracker.start_track(np.array([0.0, 0.0]))
tracker.predict()
tracker.update([np.array([0.05, 0.0])])

# Fuse asynchronous sensors with a UKF backbone
ukf = UnscentedKalmanFilter(
    state_dim=3,
    measurement_dim=1,
    process_model=lambda state: state,
    process_noise=np.eye(3) * 1e-3,
)
engine = AsynchronousFusionEngine(
    ukf,
    [SensorStream("vision", 30.0, np.array([[0.05]]), lambda state: np.array([state[0]]))],
)
engine.step(0.033)
engine.ingest("vision", np.array([0.9]))

# Fuse a depth image into a TSDF volume for later inspection
tsdf = VolumetricTSDF(bounds=np.array([[-0.5, -0.5, 0.0], [0.5, 0.5, 1.5]]), resolution=(48, 48, 48))
depth = np.ones((32, 32))
tsdf.integrate(depth, intrinsics=np.array([[20.0, 0.0, 15.5], [0.0, 20.0, 15.5], [0.0, 0.0, 1.0]]), extrinsics=left_pose)
surface = tsdf.extract_point_cloud()
```

## Robotics Stack

The robotics package is engineered for collaborative arms that require fast
planning, whole-body control, and physically-plausible simulation.

### Core Features

* `KinematicChain` – Denavit–Hartenberg manipulators with forward kinematics,
  Jacobians, and damped least-squares inverse kinematics.
* `JointDynamics` – simplified composite rigid body dynamics with mass matrices,
  Coriolis/centrifugal vectors, and gravity compensation.
* `ComputedTorqueController`, `ImpedanceController`, `ModelPredictiveController`
  – torque-level control policies ranging from classic computed torque to
  horizon-based MPC.
* `OperationalSpaceController` – task-space regulation with nullspace posture
  control that blends research-grade whole-body control with production-ready
  gravity and Coriolis compensation.
* `RRTPlanner`, `TrajectorySmoother` – joint-space RRT with cubic-spline
  post-processing for agile obstacle avoidance.
* `simulate_dynamics` – semi-implicit Euler integrator for fast prototyping.

### Robotics Quick Start

```python
from echoliner.robotics import (
    DHLink, KinematicChain, JointDynamics, RigidBodyParameters,
    ComputedTorqueController, OperationalSpaceController, RRTPlanner, plan_minimum_jerk
)
import numpy as np

links = [DHLink(a=0.4, alpha=0.0, d=0.0), DHLink(a=0.3, alpha=0.0, d=0.0)]
chain = KinematicChain(links)
params = [
    RigidBodyParameters(mass=4.0, com=np.zeros(3), inertia=np.eye(3)),
    RigidBodyParameters(mass=2.0, com=np.zeros(3), inertia=np.eye(3)),
]
model = JointDynamics(chain, params)
controller = ComputedTorqueController(model, kp=np.array([50.0, 40.0]), kd=np.array([5.0, 4.0]))
planner = RRTPlanner(chain, bounds=[(-np.pi, np.pi), (-np.pi, np.pi)])
path = planner.plan([0.0, 0.0], [1.0, -0.6], lambda _: False)
trajectory_t, trajectory_q = plan_minimum_jerk(path[0], path[-1], duration=2.5, samples=100)
torque = controller(np.zeros(2), np.zeros(2), trajectory_q[-1])

# Operational-space control to bias the end-effector towards a new pose
op_controller = OperationalSpaceController(
    model,
    task_stiffness=np.ones(6) * 40.0,
    task_damping=np.ones(6) * 6.0,
)
target_pose = chain.forward_kinematics([0.3, -0.2])
target_pose[0, 3] += 0.05
op_torque = op_controller.compute(np.zeros(2), np.zeros(2), target_pose)
```

## Translation & Speech Intelligence

Industrial deployments require bilingual operator interfaces, on-device speech
translation, and lexical consistency across production documents.  The
translation module couples a dense bilingual lexicon with statistical machine
translation and speech signal processing.

### Highlights

* **Lexicon Generation** – `scripts/generate_lexicon.py` creates 22,860
  English↔Chinese phrase pairs across robotics, analytics, safety, and
  infrastructure domains.  The entries are accessible via
  `DomainLexicon.lookup`, `greedy_lookup`, and `fuzzy_search`.
* **Statistical Translator** – `StatisticalTranslator` implements an IBM Model 1
  style bilingual aligner with Laplace smoothing, phrase memory, and adaptive
  vocabulary updates (`adapt`).  Lexicon matches are greedily applied before
  token-level translation for high domain fidelity.
* **Speech Feature Extraction** – `SpeechFeatureExtractor` produces mel-spectra
  and MFCCs with custom DCT implementations; `griffin_lim` reconstructs waveforms
  for rapid prototyping of voice cloning experiments.
* **Alignment & Metrics** – `DynamicTimeWarping`, `soft_alignment`, `bleu_score`,
  `chrf_score`, and `translation_error_rate` provide research-grade evaluation
  utilities.

### Translation Quick Start

```python
from echoliner.translation import (
    DomainLexicon, ParallelCorpus, SpeechFeatureExtractor,
    StatisticalTranslator, bleu_score
)

corpus = ParallelCorpus([
    ("start production", "启动 生产"),
    ("pause", "暂停"),
    ("resume", "恢复"),
])
translator = StatisticalTranslator(corpus)
translator.adapt("emergency stop", "紧急 停止")
print(translator.translate("adaptive actuator alignment", "en", "zh"))

lexicon = DomainLexicon()
entry = lexicon.lookup("predictive analytics forecasting")
print(entry)

extractor = SpeechFeatureExtractor()
waveform = np.random.randn(extractor.sample_rate)
mel = extractor.mel_spectrogram(waveform)
print(mel.shape)
```

## Analytics & Digital Twins

Reliable manufacturing requires deep visibility into process health.  The
analytics package spans point-estimate KPIs through streaming anomaly detection
and full-fidelity digital twins.

* `ProductionRun`, `moving_oee`, `mean_time_between_failures` – compute OEE
  metrics and MTBF scores for classic operations dashboards.
* `ExponentialMovingStatistics`, `StreamingAnomalyDetector`, `KalmanSmoother`
  – online telemetry filters and detectors for PLC, SCADA, and MES pipelines.
* `StateSpaceModel`, `kalman_forecast`, `seasonal_baseline` – Kalman-based
  forecasting and seasonal baselines for predictive maintenance.
* `DigitalTwin` – stochastic factory twin capturing throughput, quality yield,
  energy consumption, manipulator torque demand, and condition indices with
  optional Monte Carlo rollouts (`run_monte_carlo`) and maintenance scheduling.

### Analytics Quick Start

```python
from echoliner.analytics import (
    CellComponent, DigitalTwin, ProductionRun,
    StreamingAnomalyDetector, kalman_forecast, StateSpaceModel
)
from echoliner.robotics import DHLink, KinematicChain
import numpy as np

runs = [
    ProductionRun(480, [30], 920, 18, 0.45),
    ProductionRun(480, [20], 910, 15, 0.45),
]
print([run.overall_equipment_effectiveness() for run in runs])

detector = StreamingAnomalyDetector(window=5)
for value in [100, 101, 102, 105, 250]:
    if detector.update(value):
        print("Anomaly detected", value)

model = StateSpaceModel(
    transition=np.array([[1.0]]),
    control=np.array([[0.0]]),
    observation=np.array([1.0]),
    process_noise=np.array([[0.05]]),
    observation_noise=np.array([[0.2]]),
)
forecast = kalman_forecast(model, [1.0, 1.2, 1.3, 1.1])
print(forecast)

twin = DigitalTwin(
    components=[CellComponent("assembly", 1.0, 0.01, 5.0)],
    manipulator=KinematicChain([DHLink(a=0.4, alpha=0.0, d=0.0)])
)
metrics = twin.run(demand=300.0, time_step=1.0)
print(metrics["first_pass_yield"][-5:])
summary = twin.run_monte_carlo(demand=300.0, trials=5, time_step=1.0)
print(summary["throughput"]["mean"][0:3])
```

## Research Playbooks

The project is tuned for iterative experimentation:

1. **Vision-Robotics Co-Design** – Use bundle adjustment to recover precise
   tool poses, then feed those poses into `JointDynamics` for accurate torque
   computation.  The shared `echoliner.common.linalg` utilities guarantee
   consistent SO(3) parameterizations across modules.
2. **Speech-Driven Cobots** – Combine `SpeechFeatureExtractor` MFCCs with the
   translation lexicon to build bilingual voice command datasets, then test
   control policies via the digital twin before deploying to real hardware.
3. **Streaming OEE Dashboards** – Deploy `StreamingAnomalyDetector` alongside
   `generate_synthetic_streams` during edge simulation to tune thresholds before
   connecting to production telemetry.

## Documentation Wiki

The `docs/wiki` directory mirrors the structure of our internal field guides
with detailed walkthroughs for each subsystem:

* [Overview](docs/wiki/README.md) – architectural tour and navigation tips.
* [Vision Systems](docs/wiki/Vision.md) – calibration pipelines, fusion flows,
  and volumetric mapping recipes.
* [Robotics Stack](docs/wiki/Robotics.md) – control architecture breakdowns,
  including operational-space heuristics and safety envelopes.
* [Analytics & Twins](docs/wiki/Analytics.md) – KPI definitions, digital twin
  parameterization, and Monte Carlo benchmarking playbooks.
* [Translation Layer](docs/wiki/Translation.md) – bilingual data assets and
  evaluation methodology.

## Contributing

We welcome contributions that push the boundaries of collaborative automation.
High-impact pull requests often fall into one of three categories:

* **Algorithms** – improved camera models, new MPC formulations, or custom
  speech feature extractors.
* **Data Assets** – curated bilingual corpora, robotics datasets, or telemetry
  traces that can be integrated with the analytics module.
* **Benchmarks & Tutorials** – notebooks or markdown guides that document how to
  reproduce cutting-edge manufacturing research on top of EchoLiner Labs.

Please open an issue before large refactors so we can discuss architectural
alignment.  All contributions must include unit tests and documentation updates
where applicable.

## License

MIT © EchoLiner Labs.
