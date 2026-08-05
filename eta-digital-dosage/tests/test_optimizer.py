import pandas as pd
from eta_digital.scenarios.generator import WeightedScenarioGenerator
from eta_digital.optimization.constraints import QualityConstraints
from eta_digital.optimization.objective import ObjectiveWeights
from eta_digital.optimization.optimizer import DosageBounds,ScenarioDosageOptimizer

def test_optimizer_returns_bounded_result(predictor):
    opt=ScenarioDosageOptimizer(WeightedScenarioGenerator(predictor,30,random_state=1),QualityConstraints(.8,5.5,9.5,.8),ObjectiveWeights(),DosageBounds(5,20,1,6,4,4))
    state=pd.DataFrame([{"raw_turbidity_ntu":70,"raw_ph":7.2,"flow_m3_h":450,"temperature_c":27,"pac_mg_l":10,"polymer_mg_l":3}]); r=opt.solve(state,(10,3)); assert 5<=r.pac_mg_l<=20 and 1<=r.polymer_mg_l<=6; assert r.scenarios_evaluated==16
