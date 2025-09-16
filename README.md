# EchoLiner Labs – AI-Native Modular Manufacturing Toolkit

EchoLiner Labs publishes reference implementations of the vision, robotics,
linguistics, and analytics primitives that power our modular manufacturing
platform. The goal of this repository is to provide production-grade building
blocks that the community can study, extend, and deploy in advanced industrial
automation environments.

## Module Overview

### Vision
- **Calibration & Pose Reconstruction** – Direct Linear Transformation (DLT)
  utilities for estimating camera extrinsics, synthesizing projection matrices,
  and triangulating 3D keypoints across multi-view rigs.
- **Edge Intelligence** – Vectorized Sobel gradients, non-maximum suppression,
  and hysteresis-based Canny edge maps suitable for embedded inference.

### Robotics
- **Denavit–Hartenberg Manipulator Engine** – Arbitrary kinematic chains with
  forward kinematics, numerically robust Jacobians, and damped least-squares
  inverse kinematics.
- **Trajectory Generation** – Minimum-jerk interpolators for smooth joint-space
  motion profiles.

### Translation
- **Symmetric Statistical Translator** – IBM Model-1 style bilingual lexicon
  with Laplace smoothing, adaptive vocabulary updates, and alignment matrix
  introspection for cross-language operator interfaces.

### Analytics
- **OEE Instrumentation** – Rich production run data model with availability,
  performance, quality, and rolling OEE calculations.
- **Signal Intelligence** – Exponential smoothing forecasters, MTBF utilities,
  and z-score anomaly detection for telemetry streams.

## Quick Start

Install the project in editable mode together with the optional testing tools:

```bash
pip install -e .[test]
```

The snippet below demonstrates how the modules interoperate inside an automated
cell setup:

```python
import numpy as np

from echoliner.analytics import ProductionRun, moving_oee
from echoliner.robotics import DHLink, KinematicChain, plan_minimum_jerk
from echoliner.translation import ParallelCorpus, StatisticalTranslator
from echoliner.vision import (
    CameraExtrinsics,
    compose_projection_matrix,
    estimate_extrinsics_dlt,
    project_points,
)

# Vision: recover extrinsics from a calibration target and triangulate a point
intrinsics = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
world = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.1, 0.0]])
rotation = np.eye(3)
translation = np.array([0.0, 0.0, 0.6])
truth = CameraExtrinsics(rotation=rotation, translation=translation)
projections = project_points(world, intrinsics, truth)
estimated = estimate_extrinsics_dlt(world, projections, intrinsics)

# Robotics: plan a minimum-jerk trajectory for a 2-DOF collaborative arm
chain = KinematicChain([DHLink(a=0.4, alpha=0.0, d=0.0), DHLink(a=0.3, alpha=0.0, d=0.0)])
goal_pose = chain.forward_kinematics([0.5, -0.4])
trajectory_time, trajectory_positions = plan_minimum_jerk(
    start=np.zeros(2), goal=np.array([0.5, -0.4]), duration=2.0, samples=50
)

# Translation: bilingual operator prompts with adaptive vocabulary
corpus = ParallelCorpus([("start production", "启动 生产"), ("pause", "暂停")])
translator = StatisticalTranslator(corpus)
translator.adapt("emergency stop", "紧急 停止")
prompt = translator.translate("emergency stop", source_lang="en", target_lang="zh")

# Analytics: evaluate OEE across consecutive runs
runs = [
    ProductionRun(480.0, [25.0, 10.0], produced_units=920, defective_units=18, ideal_cycle_time=0.45),
    ProductionRun(480.0, [20.0, 12.0], produced_units=905, defective_units=15, ideal_cycle_time=0.45),
]
oee_trend = moving_oee(runs, window=2)
```

## Testing

Execute the automated verification suite with:

```bash
pytest
```

## Contributing

Contributions that push the boundaries of open industrial automation are
welcome. Please open an issue describing the enhancement or research question,
followed by a pull request with thorough tests and documentation.

## License

[MIT](LICENSE)
