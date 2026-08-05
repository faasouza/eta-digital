from pathlib import Path

from generate_synthetic_data import generate_synthetic_dataset
from train_and_validate import train_and_validate


def test_training_end_to_end(tmp_path: Path):
    data_path = tmp_path / "training.csv"
    generate_synthetic_dataset(240, 21).to_csv(data_path, index=False)
    metrics = train_and_validate(data_path, Path.cwd(), tmp_path / "artifacts")
    assert len(metrics) == 8
    assert (tmp_path / "artifacts/model.pkl").exists()
    assert (tmp_path / "artifacts/metrics.json").exists()
