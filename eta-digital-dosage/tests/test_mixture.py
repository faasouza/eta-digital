import pandas as pd
import yaml

from generate_synthetic_data import generate_synthetic_dataset
from eta_digital.contexts import ContextModel
from eta_digital.data import OUTPUTS
from eta_digital.experts import ContextualMixtureOfExperts


def build_model():
    frame = generate_synthetic_dataset(240, 42)
    config = yaml.safe_load(open("configs/contexts.yaml"))
    context_model = ContextModel.from_config(config)
    model = ContextualMixtureOfExperts(
        context_model,
        ["raw_turbidity_ntu", "raw_ph", "pac_mg_l", "polymer_mg_l"],
        OUTPUTS,
    ).fit(frame.iloc[:200])
    return model, frame


def test_mixture_predicts_all_filters_and_ph():
    model, frame = build_model()
    result = model.predict(frame.iloc[200:205])
    for output in OUTPUTS:
        assert output in result
        assert f"{output}_std" in result
    assert result[OUTPUTS].shape == (5, 4)
    assert result[OUTPUTS].notna().all().all()
