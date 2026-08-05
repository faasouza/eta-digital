PROJECT_DIR := eta-digital-dosage
PROJECT_PYTHONPATH := $(CURDIR)/$(PROJECT_DIR)/src:$(CURDIR)/$(PROJECT_DIR)/scripts
PYTHON ?= python
PIP ?= $(PYTHON) -m pip
COMPOSE ?= docker compose
DATA_FILE := data/offline/processed/synthetic_eta_aquiraz_360.csv

.PHONY: help install tests lint black black-check check synthetic-data train e2e notebooks mlflow up down restart logs ps clean

help:
	@echo "Available targets: install tests lint black black-check check synthetic-data train e2e notebooks mlflow up down restart logs ps clean"

install:
	cd $(PROJECT_DIR) && $(PIP) install -e ".[dev]"

tests:
	cd $(PROJECT_DIR) && PYTHONPATH=$(PROJECT_PYTHONPATH) $(PYTHON) -m pytest

lint:
	cd $(PROJECT_DIR) && $(PYTHON) -m ruff check src tests scripts

black:
	cd $(PROJECT_DIR) && $(PYTHON) -m black src tests scripts

black-check:
	cd $(PROJECT_DIR) && $(PYTHON) -m black --check src tests scripts

synthetic-data:
	cd $(PROJECT_DIR) && PYTHONPATH=$(PROJECT_PYTHONPATH) $(PYTHON) scripts/generate_synthetic_data.py --points 360 --seed 42 --output ../$(DATA_FILE)

train: synthetic-data
	cd $(PROJECT_DIR) && PYTHONPATH=$(PROJECT_PYTHONPATH) $(PYTHON) scripts/train_and_validate.py --data ../$(DATA_FILE) --project-dir . --artifacts artifacts

e2e: train tests

notebooks: synthetic-data
	cd $(PROJECT_DIR) && PYTHONPATH=$(PROJECT_PYTHONPATH) $(PYTHON) scripts/run_notebooks.py

check: black-check lint tests synthetic-data train

mlflow: up

up:
	cd $(PROJECT_DIR) && $(COMPOSE) up -d

down:
	cd $(PROJECT_DIR) && $(COMPOSE) down

restart: down up

logs:
	cd $(PROJECT_DIR) && $(COMPOSE) logs -f mlflow

ps:
	cd $(PROJECT_DIR) && $(COMPOSE) ps

clean:
	rm -rf $(PROJECT_DIR)/artifacts $(PROJECT_DIR)/.pytest_cache $(PROJECT_DIR)/.ruff_cache $(PROJECT_DIR)/.mypy_cache
	find $(PROJECT_DIR) -type d -name __pycache__ -prune -exec rm -rf {} +
