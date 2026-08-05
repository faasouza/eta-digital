from dataclasses import dataclass
@dataclass(frozen=True)
class FallbackPolicy:
    pac_mg_l:float; polymer_mg_l:float
