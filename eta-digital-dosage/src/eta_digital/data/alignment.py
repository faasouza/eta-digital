from __future__ import annotations

import pandas as pd


def align_outputs_by_delay(
    frame: pd.DataFrame,
    delay: str | pd.Timedelta,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    aligned = frame.copy()
    aligned[timestamp_column] = pd.to_datetime(aligned[timestamp_column], utc=True)
    output_columns = [
        "filter_1_turbidity_ntu",
        "filter_2_turbidity_ntu",
        "filter_3_turbidity_ntu",
        "filtered_ph",
    ]
    delayed = aligned[[timestamp_column, *output_columns]].copy()
    delayed[timestamp_column] = delayed[timestamp_column] - pd.Timedelta(delay)
    features = aligned.drop(columns=output_columns)
    return pd.merge_asof(
        features.sort_values(timestamp_column),
        delayed.sort_values(timestamp_column),
        on=timestamp_column,
        direction="nearest",
        tolerance=pd.Timedelta(delay) / 2,
    ).dropna(subset=output_columns)
