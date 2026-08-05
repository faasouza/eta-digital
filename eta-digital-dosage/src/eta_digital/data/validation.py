from __future__ import annotations
import pandas as pd

REQUIRED_COLUMNS=["raw_turbidity_ntu","raw_ph","flow_m3_h","temperature_c","pac_mg_l","polymer_mg_l","filtered_turbidity_ntu","filtered_ph"]
def validate_training_frame(frame:pd.DataFrame,minimum_rows:int=30)->pd.DataFrame:
    missing=sorted(set(REQUIRED_COLUMNS)-set(frame.columns))
    if missing: raise ValueError(f"missing training columns: {missing}")
    cleaned=frame.dropna(subset=REQUIRED_COLUMNS).copy()
    cleaned=cleaned[(cleaned.raw_turbidity_ntu>=0)&cleaned.raw_ph.between(0,14)&(cleaned.flow_m3_h>0)&(cleaned.pac_mg_l>=0)&(cleaned.polymer_mg_l>=0)&(cleaned.filtered_turbidity_ntu>=0)&cleaned.filtered_ph.between(0,14)]
    if len(cleaned)<minimum_rows: raise ValueError(f"only {len(cleaned)} valid rows; minimum is {minimum_rows}")
    return cleaned.reset_index(drop=True)
