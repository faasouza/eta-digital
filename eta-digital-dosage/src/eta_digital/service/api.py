from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from eta_digital.data import ProcessState

app = FastAPI(title="ETA-DIGITAL Dosage API", version="0.2.0")


class RecommendationRequest(BaseModel):
    state: ProcessState
    previous_pac_mg_l: float = Field(ge=0)
    previous_polymer_mg_l: float = Field(ge=0)


@lru_cache(maxsize=1)
def load_prediction_model():
    import mlflow.pyfunc

    model_uri = os.getenv("ETA_MODEL_URI", "models:/eta-digital-dosage-predictor@champion")
    return mlflow.pyfunc.load_model(model_uri)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(state: ProcessState) -> dict:
    try:
        frame = pd.DataFrame([state.model_dump(exclude={"sensor_quality"})])
        result = load_prediction_model().predict(frame)
        return result.iloc[0].to_dict()
    except Exception as exc:  # pragma: no cover - operational boundary
        raise HTTPException(status_code=503, detail=str(exc)) from exc
