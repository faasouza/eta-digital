from __future__ import annotations

import pandas as pd


def prediction_input_example() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "raw_turbidity_ntu": 65.0,
                "raw_ph": 7.1,
                "pac_mg_l": 12.0,
                "polymer_mg_l": 2.5,
            }
        ]
    )
