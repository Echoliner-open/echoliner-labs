# Vision Systems

The vision stack provides calibrated sensing, geometry reasoning, and dense
mapping pipelines that can be composed for inspection cells, palletizers, and
adaptive fixturing.  The modules are designed to interoperate without external
frameworks so that they can be embedded on constrained edge appliances.

## Calibration & Geometry

* `CameraExtrinsics` and `compose_projection_matrix` implement the same
  rigid-body math we deploy in production calibration rigs.  Extrinsics follow
the world-to-camera convention, making it straightforward to align to
robotic tool frames.
* `estimate_extrinsics_dlt` is robust enough for coarse factory bring-up, while
  `BundleAdjustmentProblem` in `reconstruction.py` refines the solution with
  sparse Levenberg–Marquardt iterations.
* Use `triangulate_points` alongside the reprojection utilities when verifying
  multi-camera accuracy against ground-truth metrology targets.

## Tracking & Sensor Fusion

* `KalmanTracker` and `JPDAF` support per-part traceability through occlusions
  by combining a constant-velocity prior with probabilistic data association.
* `AsynchronousFusionEngine` fuses heterogeneous vision, lidar, and inertial
  feeds.  The class exposes hooks for latency compensation, making it easy to
  align sensors running at different rates on a production cell network.

## Dense Mapping with VolumetricTSDF

`VolumetricTSDF` turns raw depth imagery into a truncated signed distance field
that can be streamed into simulation and inspection workflows.

```python
from echoliner.vision import CameraExtrinsics, VolumetricTSDF
import numpy as np

intrinsics = np.array([[18.0, 0.0, 15.5], [0.0, 18.0, 15.5], [0.0, 0.0, 1.0]])
extrinsics = CameraExtrinsics(rotation=np.eye(3), translation=np.zeros(3))
depth = np.ones((32, 32))

volume = VolumetricTSDF(bounds=np.array([[-0.6, -0.6, 0.0], [0.6, 0.6, 1.6]]), resolution=(64, 64, 64))
volume.integrate(depth, intrinsics, extrinsics)
surface_points = volume.extract_point_cloud(min_weight=0.5)
```

* The TSDF stores both signed distance values and integration weights so that
you can merge frames from multiple sensors without fighting per-pixel noise.
* `occupancy_grid` converts the TSDF into a binary grid that can be consumed by
  path planners and collision monitors.
* Use `reset()` when switching between fixture variants—this keeps per-shift
  calibration isolated while sharing the same mapping parameters.

## Operational Guidance

1. Capture at least one `VolumetricTSDF` volume per workcell configuration and
   store it alongside the calibration pack.  The same volume can feed the
   analytics twin to simulate inspection coverage.
2. When deploying to embedded systems, precompute Jacobians with
   `KinematicChain.jacobian` to accelerate per-frame pose refinement during
   bundle adjustment.
3. Use the quick-start snippet in the main README as a smoke test after every
   camera swap.  The combination of calibration, tracking, and mapping steps
   mirrors the order of operations we run in production.

Return to the [wiki index](README.md) or jump to the
[Robotics Stack](Robotics.md).
