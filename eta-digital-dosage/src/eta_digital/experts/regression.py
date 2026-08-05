from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .base import PredictiveExpert


class MultiOutputLinearExpert(PredictiveExpert):
    def __init__(self, features: list[str], outputs: list[str], alpha: float = 1.0):
        self.features = features
        self.outputs = outputs
        self.model = Ridge(alpha=alpha)
        self.residual_covariance = np.eye(len(outputs), dtype=float)
        self.is_fitted = False

    def fit(self, frame: pd.DataFrame, targets: pd.DataFrame, sample_weight: np.ndarray) -> None:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.sum() <= 1e-8:
            weights = np.ones_like(weights)
        x = frame[self.features].to_numpy(dtype=float)
        y = targets[self.outputs].to_numpy(dtype=float)
        self.model.fit(x, y, sample_weight=weights)
        residuals = y - self.model.predict(x)
        normalized = weights / weights.sum()
        centered = residuals - np.average(residuals, axis=0, weights=weights)
        covariance = (centered * normalized[:, None]).T @ centered
        self.residual_covariance = covariance + np.eye(len(self.outputs)) * 1e-6
        self.is_fitted = True

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("expert has not been fitted")
        x = frame[self.features].to_numpy(dtype=float)
        mean = self.model.predict(x)
        covariance = np.repeat(self.residual_covariance[None, :, :], len(frame), axis=0)
        return mean, covariance

    def online_update(self, row: pd.Series, target: np.ndarray, learning_rate: float) -> None:
        if not self.is_fitted:
            raise RuntimeError("expert has not been fitted")
        x = row[self.features].to_numpy(dtype=float)
        prediction = self.model.predict(x.reshape(1, -1))[0]
        error = np.asarray(target, dtype=float) - prediction
        coefficients = np.asarray(self.model.coef_, dtype=float)
        intercept = np.asarray(self.model.intercept_, dtype=float)
        scale = 1.0 + float(x @ x)
        coefficients += learning_rate * np.outer(error, x) / scale
        intercept += learning_rate * error
        self.model.coef_ = coefficients
        self.model.intercept_ = intercept
