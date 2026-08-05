from __future__ import annotations

import numpy as np
import pandas as pd

from eta_digital.experts import ContextualMixtureOfExperts

from .validation import validate_online_update


def update_active_experts(
    model: ContextualMixtureOfExperts,
    row: pd.Series,
    learning_rate: float = 0.02,
) -> bool:
    validation = validate_online_update(row)
    if not validation.accepted:
        return False
    frame = pd.DataFrame([row])
    weights = model.context_model.weights(frame)[0]
    target = row[model.outputs].to_numpy(dtype=float)
    for index, name in enumerate(model.context_model.names):
        model.experts[name].online_update(row, target, learning_rate * float(weights[index]))
    return True
