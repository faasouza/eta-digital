from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eta_digital.scenarios import ScenarioBatch


@dataclass(frozen=True)
class QualityConstraints:
    turbidity_outputs: tuple[str, ...]
    ph_output: str
    maximum_turbidity_ntu: float
    minimum_ph: float
    maximum_ph: float
    minimum_probability: float = 0.95

    def compliance(self, scenarios: ScenarioBatch) -> tuple[float, np.ndarray]:
        indexes = {name: index for index, name in enumerate(scenarios.outputs)}
        turbidity_ok = np.ones(len(scenarios.values), dtype=bool)
        for output in self.turbidity_outputs:
            turbidity_ok &= scenarios.values[:, indexes[output]] <= self.maximum_turbidity_ntu
        ph = scenarios.values[:, indexes[self.ph_output]]
        ph_ok = (ph >= self.minimum_ph) & (ph <= self.maximum_ph)
        joint = turbidity_ok & ph_ok
        probability = float(np.sum(scenarios.weights[joint]))
        return probability, joint
