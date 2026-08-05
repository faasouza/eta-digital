from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from eta_digital.optimization import ScenarioDosageOptimizer
from eta_digital.supervision import FuzzySupervisor


class DosageDecisionPipeline:
    def __init__(self, model, optimizer: ScenarioDosageOptimizer, supervisor: FuzzySupervisor):
        self.model = model
        self.optimizer = optimizer
        self.supervisor = supervisor

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(frame)

    def recommend(
        self,
        state: pd.DataFrame,
        previous_dosage: tuple[float, float],
        sensor_quality: float = 1.0,
    ) -> dict:
        prediction = self.model.predict(state).iloc[0]
        optimum = self.optimizer.solve(state, previous_dosage)
        supervised = self.supervisor.evaluate(
            optimum=optimum,
            context_confidence=float(prediction["context_confidence"]),
            sensor_quality=sensor_quality,
            previous_dosage=previous_dosage,
        )
        return {
            "prediction": prediction.to_dict(),
            "optimization": asdict(optimum),
            "recommendation": asdict(supervised),
        }
