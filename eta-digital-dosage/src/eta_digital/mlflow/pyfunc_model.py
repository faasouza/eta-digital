from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

try:
    import mlflow.pyfunc
except ImportError:  # pragma: no cover - permits core package use without MLflow installed
    mlflow = None


_BasePythonModel = mlflow.pyfunc.PythonModel if mlflow is not None else object


class EtaDigitalPredictionModel(_BasePythonModel):
    """Prediction-only MLflow model; optimization and supervision remain separate."""

    def __init__(self, model=None):
        self.model = model

    def load_context(self, context) -> None:
        if self.model is None:
            model_path = Path(context.artifacts["model_pickle"])
            with model_path.open("rb") as handle:
                self.model = pickle.load(handle)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("prediction model is not loaded")
        return self.model.predict(model_input)
