from __future__ import annotations
import numpy as np
import pandas as pd
from eta_digital.experts.mixture import ContextualMixtureOfExperts
from .validation import validate_update_sample

class OnlineUpdater:
    def __init__(self,predictor:ContextualMixtureOfExperts,learning_rate:float=.02):
        if not 0<learning_rate<=1: raise ValueError("learning_rate must be in (0,1]")
        self.predictor=predictor; self.learning_rate=learning_rate
    def update(self,row:pd.Series)->bool:
        required=self.predictor.features+self.predictor.output_features
        validation=validate_update_sample(row,required)
        if not validation.valid: return False
        weights=self.predictor.context_model.weights(pd.DataFrame([row]))
        target=row[self.predictor.output_features].to_numpy(float)
        for name,expert in self.predictor.experts.items():
            expert.partial_update(row,target,self.learning_rate,float(weights.iloc[0][name]))
        return True
