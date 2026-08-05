from __future__ import annotations
import numpy as np
import pandas as pd
from eta_digital.contexts.context_model import ContextModel
from .base import PredictionDistribution
from .regression import WeightedLinearExpert

class ContextualMixtureOfExperts:
    def __init__(self, context_model:ContextModel, features:list[str], outputs:list[str], ridge_alpha:float=1.0, minimum_context_weight:float=1e-3):
        self.context_model=context_model; self.features=features; self.output_features=outputs
        self.minimum_context_weight=minimum_context_weight
        self.experts={name:WeightedLinearExpert(features,outputs,ridge_alpha) for name in context_model.context_names}
    def fit(self, frame:pd.DataFrame):
        weights=self.context_model.weights(frame)
        for name,expert in self.experts.items():
            w=np.maximum(weights[name].to_numpy(float),self.minimum_context_weight)
            expert.fit(frame,w)
        return self
    def predict_distribution(self, frame:pd.DataFrame)->PredictionDistribution:
        weights=self.context_model.weights(frame)
        preds=np.stack([self.experts[n].predict(frame) for n in self.context_model.context_names],axis=1)
        w=weights[self.context_model.context_names].to_numpy(float)
        mean=np.einsum('nk,nko->no',w,preds)
        n,o=mean.shape; cov=np.zeros((n,o,o))
        for j,name in enumerate(self.context_model.context_names):
            diff=preds[:,j,:]-mean
            cov += w[:,j,None,None]*(self.experts[name].residual_covariance_[None,:,:]+np.einsum('ni,nj->nij',diff,diff))
        return PredictionDistribution(pd.DataFrame(mean,columns=self.output_features,index=frame.index),cov,weights,weights.max(axis=1),weights.idxmax(axis=1))
    def predict(self, frame:pd.DataFrame)->pd.DataFrame: return self.predict_distribution(frame).mean
