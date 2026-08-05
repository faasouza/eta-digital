from dataclasses import dataclass
import numpy as np
from eta_digital.scenarios.generator import ScenarioSet

@dataclass(frozen=True)
class ObjectiveWeights:
    pac_cost:float=1.0; polymer_cost:float=2.5; turbidity_penalty:float=20.0; ph_penalty:float=2.0; change_penalty:float=0.25; ph_target:float=7.0
    def evaluate(self,pac:float,polymer:float,previous:tuple[float,float],scenarios:ScenarioSet)->float:
        names={name:i for i,name in enumerate(scenarios.output_names)}
        t=scenarios.outcomes[:,names["filtered_turbidity_ntu"]]; p=scenarios.outcomes[:,names["filtered_ph"]]
        expected_t=float(np.sum(scenarios.weights*np.maximum(t,0)))
        expected_ph=float(np.sum(scenarios.weights*(p-self.ph_target)**2))
        change=(pac-previous[0])**2+(polymer-previous[1])**2
        return self.pac_cost*pac+self.polymer_cost*polymer+self.turbidity_penalty*expected_t+self.ph_penalty*expected_ph+self.change_penalty*change
