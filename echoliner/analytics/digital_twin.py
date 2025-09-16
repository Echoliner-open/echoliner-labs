"""Digital twin simulator for modular manufacturing cells."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
from numpy.typing import NDArray

from echoliner.robotics.dynamics import JointDynamics
from echoliner.robotics.kinematics import KinematicChain

__all__ = ["CellComponent", "DigitalTwin"]


def _default_rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@dataclass
class CellComponent:
    """Stochastic model of a production resource inside the twin."""

    name: str
    cycle_time: float
    failure_rate: float
    repair_time: float
    energy_per_cycle: float = 0.35
    quality_yield: float = 0.995
    degradation_rate: float = 1e-3
    failure_ramp: float = 4.0
    rng: np.random.Generator = field(default_factory=_default_rng)

    def __post_init__(self) -> None:
        if self.cycle_time <= 0.0:
            raise ValueError("cycle_time must be positive")
        if not (0.0 <= self.quality_yield <= 1.0):
            raise ValueError("quality_yield must be in [0, 1]")
        self.health = 1.0

    def reset(self, *, seed: int | None = None) -> None:
        """Reset health state and optionally reseed randomness."""

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.health = 1.0

    def simulate_step(self, time_step: float) -> Dict[str, float]:
        if time_step <= 0.0:
            raise ValueError("time_step must be positive")
        production_capacity = time_step / self.cycle_time
        adaptive_failure = self.failure_rate * (1.0 + self.failure_ramp * (1.0 - self.health))
        failure_probability = 1.0 - np.exp(-adaptive_failure * time_step)
        failure_probability = float(np.clip(failure_probability, 0.0, 1.0))
        failure = bool(self.rng.random() < failure_probability)
        if failure:
            downtime = self.repair_time
            production = 0.0
            quality = 0.0
            energy = 0.0
            self.health = 1.0
        else:
            downtime = 0.0
            production = production_capacity
            effective_yield = np.clip(self.quality_yield * self.health, 0.0, 1.0)
            quality = production * effective_yield
            energy = production * self.energy_per_cycle
            self.health = max(self.health - self.degradation_rate * time_step, 0.0)
        return {
            "production": production,
            "downtime": downtime,
            "quality": quality,
            "energy": energy,
            "health": self.health,
            "failure": float(failure),
        }


@dataclass
class DigitalTwin:
    """Factory-scale digital twin with energy, quality, and health metrics."""

    components: List[CellComponent]
    manipulator: KinematicChain
    dynamics: JointDynamics | None = None
    shift_minutes: float = 480.0
    rng: np.random.Generator = field(default_factory=_default_rng)

    def run(
        self,
        demand: float,
        *,
        time_step: float = 1.0,
        maintenance_windows: Sequence[float] | None = None,
        reset_components: bool = True,
    ) -> Dict[str, NDArray[np.float64]]:
        if time_step <= 0.0:
            raise ValueError("time_step must be positive")
        steps = int(np.ceil(self.shift_minutes / time_step))
        if reset_components:
            for component in self.components:
                component.reset()

        throughput = np.zeros(steps)
        downtime = np.zeros(steps)
        quality = np.zeros(steps)
        energy = np.zeros(steps)
        condition = np.zeros(steps)
        failures = np.zeros((steps, len(self.components)))
        health_trace = np.zeros((steps, len(self.components)))
        position_log = np.zeros((steps, self.manipulator.dof))
        velocity_log = np.zeros_like(position_log)
        torque_norm = np.zeros(steps)

        phases = np.linspace(0.0, np.pi, self.manipulator.dof)
        joint_state = np.zeros(self.manipulator.dof)
        maintenance_steps = set()
        if maintenance_windows is not None:
            maintenance_steps = {int(window / time_step) for window in maintenance_windows}

        for step in range(steps):
            if step in maintenance_steps:
                for component in self.components:
                    component.reset()

            for idx, component in enumerate(self.components):
                result = component.simulate_step(time_step)
                throughput[step] += result["production"]
                downtime[step] += result["downtime"]
                quality[step] += result["quality"]
                energy[step] += result["energy"]
                health_trace[step, idx] = result["health"]
                failures[step, idx] = result["failure"]
            condition[step] = float(np.mean(health_trace[step]))

            t = step * time_step
            target = 0.35 * np.sin(0.1 * t + phases)
            joint_velocity = (target - joint_state) / time_step
            joint_state = target
            position_log[step] = joint_state
            velocity_log[step] = joint_velocity
            if self.dynamics is not None:
                inertia = self.dynamics.mass_matrix(joint_state)
                torque = inertia @ joint_velocity
            else:
                torque = joint_velocity
            torque_norm[step] = float(np.linalg.norm(torque))

        demand_gap = np.maximum(demand - np.cumsum(throughput), 0.0)
        first_pass_yield = np.divide(
            quality,
            np.maximum(throughput, 1e-9),
            out=np.zeros_like(quality),
            where=throughput > 1e-9,
        )

        metrics: Dict[str, NDArray[np.float64]] = {
            "throughput": throughput,
            "downtime": downtime,
            "quality_output": quality,
            "energy_consumption": energy,
            "condition_index": condition,
            "joint_positions": position_log,
            "joint_velocities": velocity_log,
            "torque_norm": torque_norm,
            "demand_gap": demand_gap,
            "first_pass_yield": first_pass_yield,
            "failure_flags": failures,
            "component_health": health_trace,
        }
        return metrics

    def run_monte_carlo(
        self,
        demand: float,
        trials: int,
        *,
        time_step: float = 1.0,
        maintenance_windows: Sequence[float] | None = None,
    ) -> Dict[str, Dict[str, NDArray[np.float64]]]:
        if trials <= 0:
            raise ValueError("trials must be positive")
        aggregated: Dict[str, List[NDArray[np.float64]]] = {}
        for _ in range(trials):
            for component in self.components:
                component.reset(seed=int(self.rng.integers(0, 2**32 - 1)))
            metrics = self.run(
                demand,
                time_step=time_step,
                maintenance_windows=maintenance_windows,
                reset_components=False,
            )
            for key, value in metrics.items():
                aggregated.setdefault(key, []).append(np.asarray(value))
        summary: Dict[str, Dict[str, NDArray[np.float64]]] = {}
        for key, values in aggregated.items():
            stack = np.stack(values, axis=0)
            summary[key] = {
                "mean": np.mean(stack, axis=0),
                "std": np.std(stack, axis=0),
            }
        return summary
