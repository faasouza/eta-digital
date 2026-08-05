import numpy as np
import pandas as pd

from eta_digital.contexts import ContextModel, TrapezoidalMembership, TriangularMembership


def test_memberships_are_bounded():
    x = np.array([-1, 0, 1, 2, 3, 4], dtype=float)
    for membership in [TriangularMembership(0, 2, 4), TrapezoidalMembership(0, 1, 3, 4)]:
        values = membership(x)
        assert np.all((values >= 0) & (values <= 1))


def test_context_weights_sum_to_one():
    config = {
        "contexts": {
            "low": {"raw_turbidity_ntu": {"type": "triangular", "parameters": [0, 0, 60]}},
            "high": {"raw_turbidity_ntu": {"type": "triangular", "parameters": [20, 100, 200]}},
        }
    }
    model = ContextModel.from_config(config)
    weights = model.weights(pd.DataFrame({"raw_turbidity_ntu": [10, 50, 150]}))
    assert np.allclose(weights.sum(axis=1), 1.0)
