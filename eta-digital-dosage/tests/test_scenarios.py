import pandas as pd
from eta_digital.scenarios.generator import WeightedScenarioGenerator

def test_scenarios(predictor):
    state=pd.DataFrame([{"raw_turbidity_ntu":70,"raw_ph":7.2,"flow_m3_h":450,"temperature_c":27,"pac_mg_l":12,"polymer_mg_l":3.5}])
    scenarios=WeightedScenarioGenerator(predictor,50,{"raw_turbidity_ntu":1.0}).generate(state); assert scenarios.outcomes.shape==(50,2); assert abs(scenarios.weights.sum()-1)<1e-12
