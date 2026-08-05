from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from eta_digital.contexts import ContextModel
from eta_digital.data import OUTPUTS, validate_training_frame
from eta_digital.experts import ContextualMixtureOfExperts


def train_and_validate(data_path: Path, project_dir: Path, artifact_dir: Path) -> dict:
    frame = validate_training_frame(pd.read_csv(data_path))
    split = int(len(frame) * 0.8)
    train = frame.iloc[:split].copy()
    test = frame.iloc[split:].copy()
    with (project_dir / "configs/contexts.yaml").open() as handle:
        contexts_config = yaml.safe_load(handle)
    with (project_dir / "configs/model.yaml").open() as handle:
        model_config = yaml.safe_load(handle)
    context_model = ContextModel.from_config(
        contexts_config, minimum_weight=model_config["minimum_context_weight"]
    )
    model = ContextualMixtureOfExperts(
        context_model=context_model,
        features=model_config["features"],
        outputs=model_config["outputs"],
        alpha=model_config["ridge_alpha"],
    ).fit(train)
    predictions = model.predict(test)
    metrics = {}
    for output in OUTPUTS:
        metrics[f"{output}_rmse"] = float(
            root_mean_squared_error(test[output], predictions[output])
        )
        metrics[f"{output}_mae"] = float(mean_absolute_error(test[output], predictions[output]))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with (artifact_dir / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    predictions.to_csv(artifact_dir / "holdout_predictions.csv", index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    metrics = train_and_validate(args.data, args.project_dir, args.artifacts)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
