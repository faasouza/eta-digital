# ETA-DIGITAL

ETA-DIGITAL is a modular platform for data-driven decision support in water treatment. The first module implements PAC and cationic-polymer dosage modelling. A filter-washing module can be added independently later.

## Dosage model

The implementation under [`eta-digital-dosage/`](eta-digital-dosage/) contains:

- context identification through possibility functions;
- a contextual mixture of multi-output experts;
- prediction of turbidity for Filters 1, 2 and 3 and one filtered-water pH value;
- predictive uncertainty and weighted scenarios;
- joint chance-constrained optimization at the configured probability;
- fuzzy supervision, rate limiting and fallback;
- controlled online adaptation;
- MLflow packaging, registration and model aliases.

## Local commands

```bash
make install
make tests
make lint
make black
make black-check
make synthetic-data
make train
make e2e
make mlflow up
make down
```

`make mlflow up` and `make mlflow` both start the local MLflow service through Docker Compose.

## CI policy

GitHub Actions does not run for ordinary pushes or pull requests. Validation runs only when a semantic version tag is pushed, for example:

```bash
git tag v1.2
git push origin v1.2
```

Accepted tag formats are `vMAJOR.MINOR` and `vMAJOR.MINOR.PATCH`.

## Data

A deterministic 200-point example is committed at:

```text
data/offline/processed/synthetic_eta_aquiraz_200.csv
```

`make synthetic-data` generates the default 360-point validation dataset at:

```text
data/offline/processed/synthetic_eta_aquiraz_360.csv
```

Both datasets contain raw-water turbidity and pH, PAC and polymer dosages, turbidity from three filters, and one filtered-water pH measurement. They are suitable for software validation only, not operational calibration.
