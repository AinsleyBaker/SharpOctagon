"""CLI entry point: load feature matrix and run full training pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ufc_predict.models.train import run_full_training

log = logging.getLogger(__name__)
MATRIX_PATH = Path("data/feature_matrix.parquet")


def run() -> None:
    if not MATRIX_PATH.exists():
        raise FileNotFoundError(f"Feature matrix not found at {MATRIX_PATH}. Run build_matrix first.")

    df = pd.read_parquet(MATRIX_PATH)
    log.info("Loaded feature matrix: %s", df.shape)
    run_full_training(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
