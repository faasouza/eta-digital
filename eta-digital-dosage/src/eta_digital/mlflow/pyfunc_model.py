from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from eta_digital.experts.mixture import ContextualMixtureOfExperts
from eta_digital.uncertainty.calibration import ConformalCalibrator
try:
    from mlflow.pyfunc import PythonModel as _PythonModelBase
except ImportError:
    class _PythonModelBase: pass

class EtaDigitalPredictionModel(_PythonModelBase):
    def __init__(self,predictor:ContextualMixtureOfExperts,calibrator:ConformalCalibrator|None=None): self.predictor=predictor; self.calibrator=calibrator
    def predict(self,context:Any,model_input,params=None)->pd.DataFrame:
        del context,params
        frame=model_input.copy() if isinstance(model_input,pd.DataFrame) else pd.DataFrame(model_input)
        d=self.predictor.predict_distribution(frame); cov=d.covariance if self.calibrator is None else self.calibrator.calibrate_covariance(d.covariance)
        std=np.sqrt(np.maximum(np.diagonal(cov,axis1=1,axis2=2),0)); out=d.mean.copy()
        for i,name in enumerate(self.predictor.output_features): out[f"{name}_std"]=std[:,i]
        out["context_confidence"]=d.context_confidence.to_numpy(float); out["dominant_context"]=d.dominant_context.to_numpy(str)
        for name in self.predictor.context_model.context_names: out[f"context_weight_{name}"]=d.context_weights[name].to_numpy(float)
        return out.reset_index(drop=True)
