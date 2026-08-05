# ETA-DIGITAL — Coagulant Dosage

Reference implementation of the ETA-DIGITAL dosage decision component for PAC and cationic polymer. The implementation separates process prediction from dosage optimization and operational supervision.

The predictive model is a contextual mixture of experts. Possibility functions identify overlapping raw-water operating contexts. Each contextual expert predicts filtered-water turbidity and pH for candidate PAC and polymer dosages and estimates predictive uncertainty. Scenario generation and 95% chance-constrained optimization run outside the MLflow model. Fuzzy supervision accepts, limits, or rejects the optimized recommendation.

## Repository scope

- Context identification with triangular, trapezoidal, and Gaussian possibility functions.
- Contextual multi-output adaptive linear experts.
- Mixture mean and covariance using within-expert and between-expert uncertainty.
- Conformal uncertainty calibration.
- Weighted Monte Carlo scenarios with sensor and process uncertainty.
- Grid-based scenario optimization for PAC and polymer.
- Joint 95% quality constraint for filtered turbidity and pH.
- Fuzzy supervision, rate limiting, and fallback.
- Controlled recursive model updates after validated observations.
- MLflow PyFunc packaging for the prediction model only.
- FastAPI endpoints for prediction and recommendation.

The PLC and DataBridge remain responsible for hard interlocks, permissives, watchdogs, communication failures, and final command authority.

## Layout

```text
eta-digital-dosage/
├── configs/
├── notebooks/
├── src/eta_digital/
└── tests/

../data/offline/
├── raw/
├── interim/
├── processed/
└── schemas/
```

The attached `.xls` files under `data/offline/raw` are source examples. They contain catalogue-like rows for available quality parameters and chemical products, not the time-stamped numerical values required to train the model. A training dataset must contain aligned raw-water conditions, applied dosages, and delayed filtered-water outcomes.

## Required training columns

```text
timestamp
raw_turbidity_ntu
raw_ph
flow_m3_h
temperature_c
pac_mg_l
polymer_mg_l
filtered_turbidity_ntu
filtered_ph
```

Optional operational columns include `filter_id`, `filter_state`, `sensor_quality`, and event flags for backwashing or maintenance.

## Installation

From `eta-digital-dosage`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

## Local MLflow

```bash
cp .env.example .env
docker compose up -d
```

The default tracking URI is `http://localhost:5000`. The three notebooks train the model, register a candidate version, and assign the `champion` alias after validation.

## Production separation

1. MLflow stores and serves the contextual prediction model.
2. The dosage service loads the approved model.
3. Scenario generation evaluates uncertainty for candidate dosages.
4. The optimizer selects PAC and polymer under the configured constraints.
5. Fuzzy supervision applies confidence rules and fallback.
6. DataBridge and the PLC enforce operational and safety constraints.
