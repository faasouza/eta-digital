from pathlib import Path
import importlib.util

import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_synthetic_data.py"
spec = importlib.util.spec_from_file_location("generate_synthetic_data", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_synthetic_dataset_schema_and_ranges():
    data = module.generate_dataset(n_points=240, seed=7)

    assert len(data) == 240
    assert list(data.columns) == [
        "timestamp",
        "raw_turbidity_ntu",
        "raw_ph",
        "pac_mg_l",
        "polymer_mg_l",
        "filter_1_turbidity_ntu",
        "filter_2_turbidity_ntu",
        "filter_3_turbidity_ntu",
        "filtered_ph",
    ]
    assert pd.api.types.is_datetime64_any_dtype(data["timestamp"])
    assert data.isna().sum().sum() == 0
    assert data["raw_turbidity_ntu"].between(2.0, 140.0).all()
    assert data["raw_ph"].between(6.3, 7.8).all()
    assert data["pac_mg_l"].between(5.0, 32.0).all()
    assert data["polymer_mg_l"].between(0.3, 4.5).all()
    assert data["filtered_ph"].between(5.8, 7.6).all()
    for column in [
        "filter_1_turbidity_ntu",
        "filter_2_turbidity_ntu",
        "filter_3_turbidity_ntu",
    ]:
        assert (data[column] >= 0.01).all()


def test_synthetic_dataset_is_reproducible():
    first = module.generate_dataset(n_points=200, seed=42)
    second = module.generate_dataset(n_points=200, seed=42)
    pd.testing.assert_frame_equal(first, second)
