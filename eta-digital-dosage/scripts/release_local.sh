#!/usr/bin/env bash
set -euo pipefail

VERSION="${VERSION:-${1:-}}"

if [[ -z "${VERSION}" ]]; then
  echo "VERSION is required. Example: VERSION=0.1.0 ./scripts/release_local.sh" >&2
  exit 2
fi

if [[ ! "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]]; then
  echo "Invalid VERSION: ${VERSION}. Use a semantic version such as 0.1.0." >&2
  exit 2
fi

PROJECT_VERSION="$(python - <<'PY'
from pathlib import Path
import tomllib
with Path('pyproject.toml').open('rb') as stream:
    print(tomllib.load(stream)['project']['version'])
PY
)"

if [[ "${PROJECT_VERSION}" != "${VERSION}" ]]; then
  echo "VERSION=${VERSION} does not match pyproject.toml version ${PROJECT_VERSION}." >&2
  exit 2
fi

python -m pytest
python scripts/generate_synthetic_data.py \
  --points 360 \
  --seed 42 \
  --output ../data/offline/processed/synthetic_eta_aquiraz_360.csv

echo "Local validation completed for version ${VERSION}."
echo "No GitHub-hosted tests were executed."
