# ETA-DIGITAL

ETA-DIGITAL is a modular platform for data-driven support of water-treatment operations.

The first implemented component is the PAC and polymer dosage module:

- contextual mixture of experts;
- prediction of filtered-water pH and turbidity;
- uncertainty estimation and scenario generation;
- 95% chance-constrained dosage optimization;
- fuzzy supervision and fallback;
- controlled online adaptation;
- MLflow packaging, registration and production promotion.

The dosage implementation is located in [`eta-digital-dosage/`](eta-digital-dosage/).

Offline and synthetic datasets are located in [`data/offline/`](data/offline/). The repository is structured so that the filter-washing model can be added later as a separate module without mixing its logic with chemical-dosage control.

## Synthetic dataset

A deterministic generator creates 200 to 500 samples containing:

- raw-water turbidity;
- raw-water pH;
- PAC dosage;
- polymer dosage;
- filtered-water turbidity for Filters 1, 2 and 3;
- one filtered-water pH value.

Generate the default 360-point dataset locally:

```bash
cd eta-digital-dosage
python scripts/generate_synthetic_data.py \
  --points 360 \
  --seed 42 \
  --output ../data/offline/processed/synthetic_eta_aquiraz_360.csv
```

## Installation and local tests

```bash
cd eta-digital-dosage
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

Tests are not executed on GitHub. The version-gated local release validation runs only when a semantic version is explicitly supplied and matches `pyproject.toml`:

```bash
cd eta-digital-dosage
chmod +x scripts/release_local.sh
VERSION=0.1.0 ./scripts/release_local.sh
```

This command runs the complete local test suite and regenerates the 360-point synthetic dataset. A missing, invalid, or mismatched version stops the process before testing.

## Local MLflow

```bash
cd eta-digital-dosage
cp .env.example .env
docker compose up -d
```

See [`eta-digital-dosage/README.md`](eta-digital-dosage/README.md) for the model architecture, data requirements and deployment workflow.
