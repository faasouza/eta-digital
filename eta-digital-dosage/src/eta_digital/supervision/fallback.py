from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FallbackDosage:
    pac_mg_l: float
    polymer_mg_l: float
