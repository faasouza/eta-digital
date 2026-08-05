from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_dataset(n_points: int = 360, seed: int = 42) -> pd.DataFrame:
    if not 200 <= n_points <= 500:
        raise ValueError("n_points must be between 200 and 500")

    rng = np.random.default_rng(seed)
    index = np.arange(n_points)
    timestamps = pd.date_range("2025-01-01", periods=n_points, freq="2h")

    raw_turbidity = np.zeros(n_points, dtype=float)
    raw_turbidity[0] = 18.0
    for i in range(1, n_points):
        baseline = 20.0 + 8.0 * np.sin(i / 35.0)
        raw_turbidity[i] = (
            0.88 * raw_turbidity[i - 1]
            + 0.12 * baseline
            + rng.normal(0.0, 2.2)
        )

    storm_events = [
        (int(0.13 * n_points), 45.0, 18.0),
        (int(0.37 * n_points), 70.0, 25.0),
        (int(0.68 * n_points), 35.0, 15.0),
        (int(0.86 * n_points), 60.0, 20.0),
    ]
    for center, amplitude, duration in storm_events:
        raw_turbidity += amplitude * np.exp(
            -0.5 * ((index - center) / (duration / 3.0)) ** 2
        )
    raw_turbidity = np.clip(raw_turbidity, 2.0, 140.0)

    raw_ph = (
        7.2
        + 0.18 * np.sin(index / 27.0)
        - 0.0025 * (raw_turbidity - 20.0)
        + rng.normal(0.0, 0.08, n_points)
    )
    raw_ph = np.clip(raw_ph, 6.3, 7.8)

    pac_mg_l = (
        7.5
        + 0.16 * raw_turbidity
        + 2.2 * np.maximum(0.0, 7.0 - raw_ph)
        + rng.normal(0.0, 1.0, n_points)
    )
    pac_mg_l = np.clip(pac_mg_l, 5.0, 32.0)

    polymer_mg_l = (
        0.7
        + 0.018 * raw_turbidity
        + 0.10 * np.maximum(0.0, pac_mg_l - 15.0)
        + rng.normal(0.0, 0.18, n_points)
    )
    polymer_mg_l = np.clip(polymer_mg_l, 0.3, 4.5)

    coagulation_effect = (
        0.22 * pac_mg_l
        + 0.45 * polymer_mg_l
        - 0.001 * (pac_mg_l - 20.0) ** 2
    )
    base_filtered_turbidity = raw_turbidity * np.exp(
        -np.clip(coagulation_effect, 0.5, 8.0)
    )

    cycle_length = 72
    filter_1_age = (index % cycle_length) / cycle_length
    filter_2_age = ((index + 20) % cycle_length) / cycle_length
    filter_3_age = ((index + 45) % cycle_length) / cycle_length

    filter_1 = np.clip(
        base_filtered_turbidity * (1.0 + 1.2 * filter_1_age**3)
        + rng.normal(0.0, 0.025 + 0.010 * base_filtered_turbidity),
        0.01,
        None,
    )
    filter_2 = np.clip(
        base_filtered_turbidity * 1.05 * (1.0 + 1.0 * filter_2_age**3)
        + rng.normal(0.0, 0.030 + 0.011 * base_filtered_turbidity),
        0.01,
        None,
    )
    filter_3 = np.clip(
        base_filtered_turbidity * 0.95 * (1.0 + 1.3 * filter_3_age**3)
        + rng.normal(0.0, 0.025 + 0.010 * base_filtered_turbidity),
        0.01,
        None,
    )

    filtered_ph = (
        raw_ph
        - 0.018 * pac_mg_l
        + 0.012 * polymer_mg_l
        + 0.16
        + rng.normal(0.0, 0.045, n_points)
    )
    filtered_ph = np.clip(filtered_ph, 5.8, 7.6)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "raw_turbidity_ntu": np.round(raw_turbidity, 3),
            "raw_ph": np.round(raw_ph, 3),
            "pac_mg_l": np.round(pac_mg_l, 3),
            "polymer_mg_l": np.round(polymer_mg_l, 3),
            "filter_1_turbidity_ntu": np.round(filter_1, 3),
            "filter_2_turbidity_ntu": np.round(filter_2, 3),
            "filter_3_turbidity_ntu": np.round(filter_3, 3),
            "filtered_ph": np.round(filtered_ph, 3),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=360)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../data/offline/processed/synthetic_eta_aquiraz_360.csv"),
    )
    args = parser.parse_args()

    dataset = generate_dataset(n_points=args.points, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(f"Created {len(dataset)} rows at {args.output}")


if __name__ == "__main__":
    main()
