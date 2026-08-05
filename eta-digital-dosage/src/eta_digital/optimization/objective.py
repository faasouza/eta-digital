from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eta_digital.scenarios import ScenarioBatch


@dataclass(frozen=True)
class ObjectiveWeights:
    pac_cost: float = 1.0
    polymer_cost: float = 2.5
    turbidity_penalty: float = 20.0
    ph_penalty: float = 2.0
    change_penalty: float = 0.25
    ph_target: float = 7.0

    def evaluate(
        self,
        pac_mg_l: float,
        polymer_mg_l: float,
        previous_dosage: tuple[float, float],
        scenarios: ScenarioBatch,
    ) -> float:
        indexes = {name: index for index, name in enumerate(scenarios.outputs)}
        turbidity_columns = [
            index for name, index in indexes.items() if name.endswith("turbidity_ntu")
        ]
        ph_index = indexes["filtered_ph"]
        mean_turbidity = np.mean(scenarios.values[:, turbidity_columns], axis=1)
        quality = self.turbidity_penalty * float(
            np.average(mean_turbidity, weights=scenarios.weights)
        )
        ph_cost = self.ph_penalty * float(
            np.average(
                (scenarios.values[:, ph_index] - self.ph_target) ** 2,
                weights=scenarios.weights,
            )
        )
        chemical = self.pac_cost * pac_mg_l + self.polymer_cost * polymer_mg_l
        change = self.change_penalty * (
            (pac_mg_l - previous_dosage[0]) ** 2 + (polymer_mg_l - previous_dosage[1]) ** 2
        )
        return chemical + quality + ph_cost + change
