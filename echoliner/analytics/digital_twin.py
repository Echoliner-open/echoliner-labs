"""Digital twin simulator for modular manufacturing cells."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray

from echoliner.robotics.kinematics import KinematicChain

__all__ = ["CellComponent", "DigitalTwin"]


def _default_rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@dataclass
class CellComponent:
    name: str
    cycle_time: float
    failure_rate: float
    repair_time: float
    rng: np.random.Generator = field(default_factory=_default_rng)

    def simulate_step(self) -> Dict[str, float]:
        failure = self.rng.random() < self.failure_rate
        downtime = self.repair_time if failure else 0.0
        return {"production": 1.0 if not failure else 0.0, "downtime": downtime}


@dataclass
class DigitalTwin:
    components: List[CellComponent]
    manipulator: KinematicChain
    shift_minutes: float = 480.0
    rng: np.random.Generator = field(default_factory=_default_rng)

    def run(self, demand: float, *, time_step: float = 1.0) -> Dict[str, NDArray[np.float64]]:
        steps = int(self.shift_minutes / time_step)
        throughput = np.zeros(steps)
        downtime = np.zeros(steps)
        position_log = np.zeros((steps, self.manipulator.dof))
        joint_state = np.zeros(self.manipulator.dof)
        for step in range(steps):
            for component in self.components:
                result = component.simulate_step()
                throughput[step] += result["production"]
                downtime[step] += result["downtime"]
            joint_state = np.sin(0.1 * step) * np.ones(self.manipulator.dof)
            position_log[step] = joint_state
        metrics = {
            "throughput": throughput,
            "downtime": downtime,
            "joint_positions": position_log,
        }
        metrics["demand_gap"] = demand - np.cumsum(throughput)
        return metrics
