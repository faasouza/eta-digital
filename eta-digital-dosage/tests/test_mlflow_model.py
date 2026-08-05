import pandas as pd
from eta_digital.mlflow.pyfunc_model import EtaDigitalPredictionModel

def test_pyfunc_contains_prediction_not_optimization(predictor):
    x=pd.DataFrame([{"raw_turbidity_ntu":50,"raw_ph":7,"flow_m3_h":450,"temperature_c":27,"pac_mg_l":11,"polymer_mg_l":3.7}]); out=EtaDigitalPredictionModel(predictor).predict(None,x); assert "filtered_turbidity_ntu" in out and "context_confidence" in out; assert "recommended_pac_mg_l" not in out
