from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class UpdateValidationResult:
    accepted: bool
    reason: str


def validate_online_update(row: pd.Series) -> UpdateValidationResult:
    if bool(row.get("backwash_active", False)):
        return UpdateValidationResult(False, "backwash active")
    if bool(row.get("maintenance_active", False)):
        return UpdateValidationResult(False, "maintenance active")
    if float(row.get("sensor_quality", 1.0)) < 0.8:
        return UpdateValidationResult(False, "sensor quality below threshold")
    return UpdateValidationResult(True, "validated observation")
