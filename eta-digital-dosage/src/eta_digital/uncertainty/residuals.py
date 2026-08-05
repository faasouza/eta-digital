import numpy as np

def residual_covariance(y_true:np.ndarray,y_pred:np.ndarray)->np.ndarray:
    residual=np.asarray(y_true,float)-np.asarray(y_pred,float)
    if residual.ndim!=2 or len(residual)<2: raise ValueError("at least two residual samples are required")
    return np.atleast_2d(np.cov(residual,rowvar=False,ddof=1))+np.eye(residual.shape[1])*1e-9
