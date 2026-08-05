# ETA-DIGITAL — Coagulant Dosage

Modular reference implementation for contextual prediction and robust recommendation of PAC and cationic-polymer dosage. Prediction, uncertainty, optimization, fuzzy supervision, adaptation, and MLflow lifecycle logic are separated.

## Scope

- Possibility-function context identification.
- Contextual mixture of multi-output linear experts.
- Predictive covariance and conformal calibration.
- Weighted Monte Carlo scenarios.
- Joint 95% chance-constrained dosage optimization.
- Fuzzy acceptance, limitation, and fallback.
- Controlled online expert updates.
- Prediction-only MLflow PyFunc packaging.
- FastAPI prediction and recommendation endpoints.

Hard interlocks, permissives, watchdogs, communications, and final command authority remain in DataBridge/PLC.

## Installation and tests

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

## Data

Offline source files are under `../data/offline/raw`. The attached `.xls` examples are retained unchanged. They are useful for ingestion/schema development, but numerical, time-aligned process records are required to train the model. See `../data/offline/README.md` and `../data/offline/schemas/training_schema.csv`.

## MLflow

```bash
cp .env.example .env
docker compose up -d
```

Run the notebooks in order:

1. `01_train_model.ipynb`
2. `02_register_mlflow.ipynb`
3. `03_promote_to_production.ipynb`

The MLflow model serves predictions and uncertainty for candidate dosage values. Scenario optimization and fuzzy supervision remain separate runtime components.
