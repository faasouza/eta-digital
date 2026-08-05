from __future__ import annotations

from pydantic import BaseModel, Field


class ProcessState(BaseModel):
    raw_turbidity_ntu: float = Field(ge=0)
    raw_ph: float = Field(ge=0, le=14)
    pac_mg_l: float = Field(ge=0)
    polymer_mg_l: float = Field(ge=0)
    sensor_quality: float = Field(default=1.0, ge=0, le=1)


class PredictionRecord(BaseModel):
    filter_1_turbidity_ntu: float
    filter_2_turbidity_ntu: float
    filter_3_turbidity_ntu: float
    filtered_ph: float
    dominant_context: str
    context_confidence: float
