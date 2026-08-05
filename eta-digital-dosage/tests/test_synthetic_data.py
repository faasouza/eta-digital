import pandas as pd
import pytest

from generate_synthetic_data import generate_synthetic_dataset
from eta_digital.data import REQUIRED_COLUMNS, validate_training_frame


def test_synthetic_data_has_new_three_filter_schema():
    frame = generate_synthetic_dataset(points=360, seed=42)
    assert len(frame) == 360
    assert list(frame.columns) == REQUIRED_COLUMNS
    validate_training_frame(frame)
    assert not frame.isna().any().any()
    assert (frame[["filter_1_turbidity_ntu", "filter_2_turbidity_ntu", "filter_3_turbidity_ntu"]] >= 0).all().all()


def test_synthetic_data_is_reproducible():
    pd.testing.assert_frame_equal(
        generate_synthetic_dataset(200, 7), generate_synthetic_dataset(200, 7)
    )


def test_synthetic_data_rejects_invalid_size():
    with pytest.raises(ValueError):
        generate_synthetic_dataset(199, 42)
