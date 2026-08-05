# ETA-DIGITAL dosage module

This module separates prediction, uncertainty, optimization, supervision and model lifecycle logic.

## Model inputs

- `raw_turbidity_ntu`
- `raw_ph`
- `pac_mg_l`
- `polymer_mg_l`

## Model outputs

- `filter_1_turbidity_ntu`
- `filter_2_turbidity_ntu`
- `filter_3_turbidity_ntu`
- `filtered_ph`

The chance constraint is joint: all three filter-turbidity outputs and filtered pH must satisfy their limits in at least the configured fraction of weighted scenarios.

## Development

From the repository root:

```bash
make install
make tests
make lint
make black
make synthetic-data
make train
make e2e
```

## MLflow

```bash
make mlflow up
```

Run the notebooks in order to train, register and promote a model. The MLflow PyFunc contains prediction only. Scenario optimization and fuzzy supervision remain separate runtime components.
