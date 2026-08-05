from dataclasses import dataclass
import numpy as np
from eta_digital.scenarios.generator import ScenarioSet

@dataclass(frozen=True)
class QualityConstraints:
    turbidity_max_ntu: float
    ph_min: float
    ph_max: float
    minimum_probability: float=0.95
    def compliance(self,scenarios:ScenarioSet)->tuple[float,np.ndarray]:
        names={name:i for i,name in enumerate(scenarios.output_names)}
        t=scenarios.outcomes[:,names["filtered_turbidity_ntu"]]; p=scenarios.outcomes[:,names["filtered_ph"]]
        valid=(t<=self.turbidity_max_ntu)&(p>=self.ph_min)&(p<=self.ph_max)
        return float(np.sum(scenarios.weights*valid)),valid
