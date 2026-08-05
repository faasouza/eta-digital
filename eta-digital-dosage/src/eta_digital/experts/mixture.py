from __future__ import annotations

import numpy as np
import pandas as pd

from eta_digital.contexts import ContextModel

from .regression import MultiOutputLinearExpert


class ContextualMixtureOfExperts:
    def __init__(
        self,
        context_model: ContextModel,
        features: list[str],
        outputs: list[str],
        alpha: float = 1.0,
    ):
        self.context_model = context_model
        self.features = features
        self.outputs = outputs
        self.experts = {
            name: MultiOutputLinearExpert(features, outputs, alpha=alpha)
            for name in context_model.names
        }
        self.is_fitted = False

    def fit(self, frame: pd.DataFrame) -> "ContextualMixtureOfExperts":
        weights = self.context_model.weights(frame)
        targets = frame[self.outputs]
        for index, name in enumerate(self.context_model.names):
            self.experts[name].fit(frame, targets, weights[:, index])
        self.is_fitted = True
        return self

    def predict_distribution(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("mixture has not been fitted")
        weights = self.context_model.weights(frame)
        means = []
        covariances = []
        for name in self.context_model.names:
            mean, covariance = self.experts[name].predict(frame)
            means.append(mean)
            covariances.append(covariance)
        stacked_means = np.stack(means, axis=1)
        stacked_covariances = np.stack(covariances, axis=1)
        mixture_mean = np.einsum("nk,nko->no", weights, stacked_means)
        deltas = stacked_means - mixture_mean[:, None, :]
        within = np.einsum("nk,nkij->nij", weights, stacked_covariances)
        between = np.einsum("nk,nki,nkj->nij", weights, deltas, deltas)
        return mixture_mean, within + between, weights

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        mean, covariance, weights = self.predict_distribution(frame)
        result = pd.DataFrame(mean, columns=self.outputs, index=frame.index)
        for output_index, output in enumerate(self.outputs):
            result[f"{output}_std"] = np.sqrt(np.maximum(covariance[:, output_index, output_index], 0))
        result["dominant_context"] = [
            self.context_model.names[index] for index in np.argmax(weights, axis=1)
        ]
        result["context_confidence"] = weights.max(axis=1)
        return result
