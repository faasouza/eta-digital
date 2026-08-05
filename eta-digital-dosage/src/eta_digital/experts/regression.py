from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

class WeightedLinearExpert:
    def __init__(self, features:list[str], outputs:list[str], ridge_alpha:float=1.0):
        self.features=features; self.outputs=outputs; self.ridge_alpha=ridge_alpha
        self.model=Ridge(alpha=ridge_alpha); self.residual_covariance_=np.eye(len(outputs)); self.fitted_=False
    def fit(self, frame:pd.DataFrame, sample_weight:np.ndarray|None=None):
        X=frame[self.features].to_numpy(float); y=frame[self.outputs].to_numpy(float)
        if len(frame)<max(3,len(self.features)+1): raise ValueError("insufficient samples")
        self.model.fit(X,y,sample_weight=sample_weight)
        residual=y-self.model.predict(X)
        if sample_weight is None: cov=np.cov(residual,rowvar=False,ddof=1)
        else:
            w=np.asarray(sample_weight,float); w=np.maximum(w,0); w=w/w.sum()
            mean=np.sum(residual*w[:,None],axis=0); centered=residual-mean
            cov=(centered*w[:,None]).T@centered/max(1e-9,1-np.sum(w*w))
        self.residual_covariance_=np.atleast_2d(cov)+np.eye(len(self.outputs))*1e-9
        self.fitted_=True; return self
    def predict(self, frame:pd.DataFrame)->np.ndarray:
        if not self.fitted_: raise RuntimeError("expert is not fitted")
        return self.model.predict(frame[self.features].to_numpy(float))
    def partial_update(self, row:pd.Series, target:np.ndarray, learning_rate:float, context_weight:float):
        if not self.fitted_: raise RuntimeError("expert is not fitted")
        x=row[self.features].to_numpy(float); pred=self.model.predict(x.reshape(1,-1))[0]
        err=np.asarray(target,float)-pred; step=float(learning_rate)*float(context_weight)
        scale=float(np.dot(x,x)+1.0)
        self.model.coef_ += step*np.outer(err,x)/scale
        self.model.intercept_ += step*err
        self.residual_covariance_=(1-step)*self.residual_covariance_+step*np.outer(err,err)
