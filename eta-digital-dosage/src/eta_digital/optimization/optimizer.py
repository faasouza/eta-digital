from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from eta_digital.scenarios import WeightedScenarioGenerator

from .constraints import QualityConstraints
from .objective import ObjectiveWeights


@dataclass(frozen=True)
class DosageBounds:
    pac_min: float
    pac_max: float
    polymer_min: float
    polymer_max: float
    pac_points: int = 29
    polymer_points: int = 20


@dataclass(frozen=True)
class OptimizationResult:
    pac_mg_l: float
    polymer_mg_l: float
    objective: float
    compliance_probability: float
    feasible: bool
    scenarios_evaluated: int


class ScenarioDosageOptimizer:
    def __init__(
        self,
        generator: WeightedScenarioGenerator,
        constraints: QualityConstraints,
        objective: ObjectiveWeights,
        bounds: DosageBounds,
    ):
        self.generator = generator
        self.constraints = constraints
        self.objective = objective
        self.bounds = bounds

    def solve(
        self, state: pd.DataFrame, previous_dosage: tuple[float, float]
    ) -> OptimizationResult:
        best: OptimizationResult | None = None
        least_violation: OptimizationResult | None = None
        count = 0
        for pac in np.linspace(self.bounds.pac_min, self.bounds.pac_max, self.bounds.pac_points):
            for polymer in np.linspace(
                self.bounds.polymer_min, self.bounds.polymer_max, self.bounds.polymer_points
            ):
                candidate = state.copy()
                candidate["pac_mg_l"] = pac
                candidate["polymer_mg_l"] = polymer
                scenarios = self.generator.generate(candidate)
                probability, _ = self.constraints.compliance(scenarios)
                score = self.objective.evaluate(pac, polymer, previous_dosage, scenarios)
                count += 1
                result = OptimizationResult(
                    pac_mg_l=float(pac),
                    polymer_mg_l=float(polymer),
                    objective=float(score),
                    compliance_probability=probability,
                    feasible=probability >= self.constraints.minimum_probability,
                    scenarios_evaluated=count,
                )
                if result.feasible and (best is None or result.objective < best.objective):
                    best = result
                if least_violation is None or (
                    result.compliance_probability,
                    -result.objective,
                ) > (
                    least_violation.compliance_probability,
                    -least_violation.objective,
                ):
                    least_violation = result
        chosen = best or least_violation
        if chosen is None:
            raise RuntimeError("optimizer evaluated no candidates")
        return OptimizationResult(
            pac_mg_l=chosen.pac_mg_l,
            polymer_mg_l=chosen.polymer_mg_l,
            objective=chosen.objective,
            compliance_probability=chosen.compliance_probability,
            feasible=chosen.feasible,
            scenarios_evaluated=count,
        )
