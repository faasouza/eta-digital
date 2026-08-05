import numpy as np

from eta_digital.scenarios import WeightedScenarioGenerator
from test_mixture import build_model


def test_scenarios_have_four_outputs_and_normalized_weights():
    model, frame = build_model()
    generator = WeightedScenarioGenerator(model, number_of_scenarios=100, random_seed=1)
    batch = generator.generate(frame.iloc[[210]][model.features])
    assert batch.values.shape == (100, 4)
    assert np.isclose(batch.weights.sum(), 1.0)
    assert (batch.values[:, :3] >= 0).all()
