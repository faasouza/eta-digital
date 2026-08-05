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

The deterministic generator creates 200 to 500 samples containing raw-water turbidity and pH, PAC and polymer dosage, filtered-water turbidity for Filters 1, 2 and 3, and one filtered-water pH value.

```bash
make synthetic-data
```

## Local development

Install the project and development dependencies:

```bash
make install
```

Run tests locally:

```bash
make tests
```

Run Ruff lint checks:

```bash
make lint
```

Format Python code with Black:

```bash
make black
```

Run formatting checks, lint, and tests together:

```bash
make check
```

Start the local MLflow stack:

```bash
make mlflow up
```

Stop it with:

```bash
make down
```

## GitHub Actions

GitHub Actions does not run for ordinary commits or pull requests. The release-validation workflow runs only when a version tag is pushed.

Accepted tag formats are:

```text
v1.0
v1.2
v1.2.0
```

Example:

```bash
git tag v1.0
git push origin v1.0
```

The tagged workflow checks Black formatting, runs Ruff, and executes the pytest suite.

See [`eta-digital-dosage/README.md`](eta-digital-dosage/README.md) for the model architecture, data requirements and deployment workflow.
