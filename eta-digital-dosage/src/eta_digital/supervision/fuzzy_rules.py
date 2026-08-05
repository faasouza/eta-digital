from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from eta_digital.optimization.optimizer import OptimizationResult
from .fallback import FallbackPolicy

@dataclass(frozen=True)
class SupervisionConfig:
    minimum_accept_confidence:float=.75; minimum_limited_confidence:float=.50; maximum_pac_change_mg_l:float=2.; maximum_polymer_change_mg_l:float=.75
@dataclass
class SupervisionDecision:
    pac_mg_l:float; polymer_mg_l:float; status:str; confidence:float; reason:str

class FuzzySupervisor:
    def __init__(self,config:SupervisionConfig,fallback:FallbackPolicy): self.config=config; self.fallback=fallback
    def evaluate(self,result:OptimizationResult,model_confidence:float,sensor_quality:float,out_of_domain_score:float,previous_dosage:tuple[float,float])->SupervisionDecision:
        confidence=float(np.clip(0.45*model_confidence+0.35*sensor_quality+0.20*(1-out_of_domain_score),0,1))
        if not result.feasible or confidence<self.config.minimum_limited_confidence:
            return SupervisionDecision(self.fallback.pac_mg_l,self.fallback.polymer_mg_l,"fallback",confidence,"infeasible optimization or insufficient confidence")
        pac=float(np.clip(result.pac_mg_l,previous_dosage[0]-self.config.maximum_pac_change_mg_l,previous_dosage[0]+self.config.maximum_pac_change_mg_l))
        pol=float(np.clip(result.polymer_mg_l,previous_dosage[1]-self.config.maximum_polymer_change_mg_l,previous_dosage[1]+self.config.maximum_polymer_change_mg_l))
        limited=(pac!=result.pac_mg_l or pol!=result.polymer_mg_l or confidence<self.config.minimum_accept_confidence)
        return SupervisionDecision(pac,pol,"limited" if limited else "accepted",confidence,"rate/confidence limits applied" if limited else "validated recommendation")
