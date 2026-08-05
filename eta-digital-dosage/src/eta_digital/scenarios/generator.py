from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from eta_digital.experts import ContextualMixtureOfExperts


@dataclass(frozen=True)
class ScenarioBatch:
    values: np.ndarray
    weights: np.ndarray
    outputs: list[str]

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.values, columns=self.outputs)


class WeightedScenarioGenerator:
    def __init__(
        self,
        model: ContextualMixtureOfExperts,
        number_of_scenarios: int = 300,
        random_seed: int = 42,
    ):
        if number_of_scenarios < 20:
            raise ValueError("at least 20 scenarios are required")
        self.model = model
        self.number_of_scenarios = number_of_scenarios
        self.random_seed = random_seed

    def generate(self, state: pd.DataFrame) -> ScenarioBatch:
        if len(state) != 1:
            raise ValueError("scenario generation expects exactly one process state")
        mean, covariance, _ = self.model.predict_distribution(state)
        rng = np.random.default_rng(self.random_seed)
        values = rng.multivariate_normal(mean[0], covariance[0], size=self.number_of_scenarios)
        turbidity_indexes = [
            self.model.outputs.index(name)
            for name in self.model.outputs
            if name.endswith("turbidity_ntu")
        ]
        values[:, turbidity_indexes] = np.maximum(values[:, turbidity_indexes], 0.0)
        weights = np.full(self.number_of_scenarios, 1.0 / self.number_of_scenarios)
        return ScenarioBatch(values=values, weights=weights, outputs=self.model.outputs)
