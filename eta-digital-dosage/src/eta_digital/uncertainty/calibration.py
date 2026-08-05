from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalScaleCalibrator:
    coverage: float = 0.95
    scale_: float = 1.0

    def fit(self, observed: np.ndarray, predicted: np.ndarray, std: np.ndarray) -> "ConformalScaleCalibrator":
        safe_std = np.maximum(np.asarray(std, dtype=float), 1e-8)
        scores = np.abs(np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float)) / safe_std
        self.scale_ = float(np.quantile(scores, self.coverage))
        return self

    def transform_covariance(self, covariance: np.ndarray) -> np.ndarray:
        return np.asarray(covariance, dtype=float) * self.scale_**2
