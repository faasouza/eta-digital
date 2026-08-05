SHELL := /bin/bash

PROJECT_DIR := eta-digital-dosage
PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
COMPOSE := docker compose -f $(PROJECT_DIR)/docker-compose.yml

.PHONY: help install tests test lint black black-check format-check check mlflow mlflow-up up down restart logs ps clean synthetic-data

help:
	@echo "Available commands:"
	@echo "  make install       Install the package and development dependencies"
	@echo "  make tests         Run the local pytest suite"
	@echo "  make lint          Run Ruff lint checks"
	@echo "  make black         Format Python files with Black"
	@echo "  make black-check   Check Black formatting without modifying files"
	@echo "  make check         Run formatting check, lint, and tests"
	@echo "  make mlflow up     Start the local MLflow stack"
	@echo "  make down          Stop the local MLflow stack"
	@echo "  make logs          Follow MLflow container logs"
	@echo "  make synthetic-data Generate the default synthetic dataset"

install:
	cd $(PROJECT_DIR) && $(PIP) install -e ".[dev]"

tests test:
	cd $(PROJECT_DIR) && $(PYTHON) -m pytest

lint:
	cd $(PROJECT_DIR) && $(PYTHON) -m ruff check src tests scripts

black:
	cd $(PROJECT_DIR) && $(PYTHON) -m black src tests scripts

black-check format-check:
	cd $(PROJECT_DIR) && $(PYTHON) -m black --check src tests scripts

check: black-check lint tests

# `make mlflow up` is supported. The explicit `up` target performs the work;
# Make executes it only once even when requested both directly and as a prerequisite.
mlflow: up

mlflow-up: up

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

restart:
	$(COMPOSE) down
	$(COMPOSE) up -d

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

synthetic-data:
	cd $(PROJECT_DIR) && $(PYTHON) scripts/generate_synthetic_data.py

clean:
	find $(PROJECT_DIR) -type d -name __pycache__ -prune -exec rm -rf {} +
	find $(PROJECT_DIR) -type d -name .pytest_cache -prune -exec rm -rf {} +
	find $(PROJECT_DIR) -type d -name .ruff_cache -prune -exec rm -rf {} +
	find $(PROJECT_DIR) -type d -name .mypy_cache -prune -exec rm -rf {} +
	find $(PROJECT_DIR) -type f -name '*.py[co]' -delete
