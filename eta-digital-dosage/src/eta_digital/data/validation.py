from __future__ import annotations

import pandas as pd

FEATURES = ["raw_turbidity_ntu", "raw_ph", "pac_mg_l", "polymer_mg_l"]
OUTPUTS = [
    "filter_1_turbidity_ntu",
    "filter_2_turbidity_ntu",
    "filter_3_turbidity_ntu",
    "filtered_ph",
]
REQUIRED_COLUMNS = ["timestamp", *FEATURES, *OUTPUTS]


def validate_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing training columns: {missing}")
    validated = frame.copy()
    validated["timestamp"] = pd.to_datetime(validated["timestamp"], utc=True, errors="raise")
    numeric = FEATURES + OUTPUTS
    validated[numeric] = validated[numeric].apply(pd.to_numeric, errors="raise")
    if validated[numeric].isna().any().any():
        raise ValueError("training frame contains missing numeric values")
    if (validated[["raw_turbidity_ntu", "pac_mg_l", "polymer_mg_l", *OUTPUTS[:3]]] < 0).any().any():
        raise ValueError("turbidity and dosage values must be nonnegative")
    if (
        not validated["raw_ph"].between(0, 14).all()
        or not validated["filtered_ph"].between(0, 14).all()
    ):
        raise ValueError("pH must be between 0 and 14")
    return validated.sort_values("timestamp").reset_index(drop=True)
