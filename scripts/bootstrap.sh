#!/usr/bin/env bash
# One-time project setup. Run from the repo root.
set -euo pipefail

echo "=== Creating virtual environment ==="
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

echo "=== Installing dependencies ==="
pip install -e ".[dev]"

echo "=== Installing pre-commit hooks ==="
pre-commit install

echo "=== Pulling Greco1899 CSVs ==="
python -m ufc_predict.ingest.pull_greco

echo "=== Initialising database ==="
python -c "from ufc_predict.db.session import init_db; init_db()"

echo "=== Running tests ==="
pytest -q

echo ""
echo "Done. Next step: python -m ufc_predict.ingest.greco_loader data/raw/greco1899"
