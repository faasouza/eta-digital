from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_dataset(points: int = 360, seed: int = 42) -> pd.DataFrame:
    if not 200 <= points <= 500:
        raise ValueError("points must be between 200 and 500")
    rng = np.random.default_rng(seed)
    timestamp = pd.date_range("2025-01-01", periods=points, freq="30min", tz="UTC")
    event = np.zeros(points)
    for start in rng.choice(np.arange(20, points - 30), size=7, replace=False):
        width = int(rng.integers(8, 24))
        event[start : start + width] += np.linspace(0, rng.uniform(30, 120), width)
    baseline = 42 + 16 * np.sin(np.linspace(0, 8 * np.pi, points))
    raw_turbidity = np.clip(baseline + event + rng.normal(0, 6, points), 3, 260)
    raw_ph = np.clip(7.25 - 0.0018 * raw_turbidity + rng.normal(0, 0.11, points), 6.3, 8.1)
    pac = np.clip(5.5 + 0.085 * raw_turbidity + 1.4 * (7.1 - raw_ph) + rng.normal(0, 0.7, points), 4, 28)
    polymer = np.clip(0.65 + 0.014 * raw_turbidity + 0.03 * pac + rng.normal(0, 0.16, points), 0.3, 5.5)
    ph_efficiency = np.exp(-((raw_ph - 7.05) / 0.8) ** 2)
    dose_efficiency = 0.05 * pac + 0.18 * polymer
    common = np.clip(raw_turbidity * np.exp(-dose_efficiency * ph_efficiency) / 7.5, 0.04, None)
    fouling = 0.08 + 0.05 * np.sin(np.linspace(0, 14 * np.pi, points)) ** 2
    filter_1 = np.clip(common * (0.82 + fouling) + rng.normal(0, 0.055, points), 0.03, 3.5)
    filter_2 = np.clip(common * (0.88 + 0.8 * fouling) + rng.normal(0, 0.06, points), 0.03, 3.5)
    filter_3 = np.clip(common * (0.78 + 1.2 * fouling) + rng.normal(0, 0.05, points), 0.03, 3.5)
    filtered_ph = np.clip(raw_ph - 0.012 * pac + 0.004 * polymer + rng.normal(0, 0.035, points), 5.8, 8.5)
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "raw_turbidity_ntu": raw_turbidity,
            "raw_ph": raw_ph,
            "pac_mg_l": pac,
            "polymer_mg_l": polymer,
            "filter_1_turbidity_ntu": filter_1,
            "filter_2_turbidity_ntu": filter_2,
            "filter_3_turbidity_ntu": filter_3,
            "filtered_ph": filtered_ph,
        }
    ).round(5)


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
    frame = generate_synthetic_dataset(args.points, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"wrote {len(frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
