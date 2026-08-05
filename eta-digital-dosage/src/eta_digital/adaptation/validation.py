from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class UpdateValidationResult:
    valid: bool
    reasons: tuple[str,...]

def validate_update_sample(row:pd.Series,required_columns:list[str])->UpdateValidationResult:
    reasons=[]
    for name in required_columns:
        if name not in row.index or pd.isna(row[name]): reasons.append(f"missing:{name}")
    if bool(row.get("backwash_active",False)): reasons.append("backwash_active")
    if bool(row.get("maintenance_active",False)): reasons.append("maintenance_active")
    if float(row.get("sensor_quality",1.0))<0.8: reasons.append("sensor_quality")
    if str(row.get("filter_state","filtration")).lower() not in {"filtration","normal"}: reasons.append("filter_state")
    return UpdateValidationResult(not reasons,tuple(reasons))
