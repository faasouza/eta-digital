from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eta_digital.optimization import OptimizationResult

from .fallback import FallbackDosage


@dataclass(frozen=True)
class SupervisionResult:
    pac_mg_l: float
    polymer_mg_l: float
    status: str
    confidence: float
    reason: str


class FuzzySupervisor:
    def __init__(
        self,
        minimum_accept_confidence: float,
        minimum_limited_confidence: float,
        maximum_pac_change_mg_l: float,
        maximum_polymer_change_mg_l: float,
        fallback: FallbackDosage,
    ):
        self.minimum_accept_confidence = minimum_accept_confidence
        self.minimum_limited_confidence = minimum_limited_confidence
        self.maximum_pac_change_mg_l = maximum_pac_change_mg_l
        self.maximum_polymer_change_mg_l = maximum_polymer_change_mg_l
        self.fallback = fallback

    def evaluate(
        self,
        optimum: OptimizationResult,
        context_confidence: float,
        sensor_quality: float,
        previous_dosage: tuple[float, float],
    ) -> SupervisionResult:
        feasibility = min(1.0, optimum.compliance_probability / 0.95)
        confidence = float(
            np.clip(0.4 * context_confidence + 0.35 * sensor_quality + 0.25 * feasibility, 0, 1)
        )
        if not optimum.feasible or confidence < self.minimum_limited_confidence:
            return SupervisionResult(
                pac_mg_l=self.fallback.pac_mg_l,
                polymer_mg_l=self.fallback.polymer_mg_l,
                status="fallback",
                confidence=confidence,
                reason="low confidence or no chance-constrained feasible solution",
            )
        pac = float(
            np.clip(
                optimum.pac_mg_l,
                previous_dosage[0] - self.maximum_pac_change_mg_l,
                previous_dosage[0] + self.maximum_pac_change_mg_l,
            )
        )
        polymer = float(
            np.clip(
                optimum.polymer_mg_l,
                previous_dosage[1] - self.maximum_polymer_change_mg_l,
                previous_dosage[1] + self.maximum_polymer_change_mg_l,
            )
        )
        limited = pac != optimum.pac_mg_l or polymer != optimum.polymer_mg_l
        if confidence < self.minimum_accept_confidence or limited:
            return SupervisionResult(
                pac_mg_l=pac,
                polymer_mg_l=polymer,
                status="limited",
                confidence=confidence,
                reason="recommendation limited by confidence or dosage-rate constraints",
            )
        return SupervisionResult(
            pac_mg_l=pac,
            polymer_mg_l=polymer,
            status="accepted",
            confidence=confidence,
            reason="optimized recommendation accepted",
        )
