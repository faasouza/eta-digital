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

## Quick start

```bash
cd eta-digital-dosage
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

For local MLflow:

```bash
cp .env.example .env
docker compose up -d
```

See [`eta-digital-dosage/README.md`](eta-digital-dosage/README.md) for the model architecture, data requirements and deployment workflow.
