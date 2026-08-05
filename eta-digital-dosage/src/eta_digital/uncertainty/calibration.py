from __future__ import annotations
from statistics import NormalDist
import numpy as np

class ConformalCalibrator:
    def __init__(self, coverage:float=0.95):
        if not 0<coverage<1: raise ValueError("coverage must be between zero and one")
        self.coverage=coverage; self.scale_=None
    def fit(self,y_true:np.ndarray,y_pred:np.ndarray,std:np.ndarray):
        std=np.maximum(np.asarray(std,float),1e-9)
        scores=np.abs(np.asarray(y_true,float)-np.asarray(y_pred,float))/std
        self.scale_=np.quantile(scores,self.coverage,axis=0,method="higher")/NormalDist().inv_cdf((1+self.coverage)/2)
        return self
    def calibrate_covariance(self,covariance:np.ndarray)->np.ndarray:
        if self.scale_ is None: raise RuntimeError("calibrator is not fitted")
        scale=np.diag(np.asarray(self.scale_,float)); return np.einsum('ij,njk,kl->nil',scale,covariance,scale)
