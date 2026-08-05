from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class PredictiveExpert(ABC):
    @abstractmethod
    def fit(self, frame: pd.DataFrame, targets: pd.DataFrame, sample_weight: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return mean [n, outputs] and covariance [n, outputs, outputs]."""
        raise NotImplementedError
