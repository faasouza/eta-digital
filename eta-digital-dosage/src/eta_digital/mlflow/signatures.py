import pandas as pd
from .pyfunc_model import EtaDigitalPredictionModel

def build_input_example():
    return pd.DataFrame([{"raw_turbidity_ntu":35.,"raw_ph":7.1,"flow_m3_h":450.,"temperature_c":27.,"pac_mg_l":10.95,"polymer_mg_l":3.70}])
def infer_prediction_signature(model:EtaDigitalPredictionModel):
    from mlflow.models import infer_signature
    x=build_input_example(); return infer_signature(x,model.predict(None,x))
