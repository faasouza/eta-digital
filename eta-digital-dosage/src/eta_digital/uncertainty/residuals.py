from __future__ import annotations

import numpy as np


def empirical_covariance(residuals: np.ndarray, minimum_variance: float = 1e-6) -> np.ndarray:
    values = np.asarray(residuals, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("residuals must contain at least two rows")
    covariance = np.cov(values, rowvar=False)
    return covariance + np.eye(values.shape[1]) * minimum_variance
