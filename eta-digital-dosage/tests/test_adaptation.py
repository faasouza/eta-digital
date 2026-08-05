from eta_digital.adaptation import update_active_experts
from test_mixture import build_model


def test_online_update_accepts_valid_observation():
    model, frame = build_model()
    row = frame.iloc[205].copy()
    row["sensor_quality"] = 1.0
    assert update_active_experts(model, row, learning_rate=0.001)
