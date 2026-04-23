"""
CLI entry point: compute conformal prediction quantiles from OOF predictions.
Run after train_runner.py, before predict.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ufc_predict.models.predict import calibrate_conformal, save_conformal_quantiles

log = logging.getLogger(__name__)
OOF_PATH = Path("models/oof_predictions.parquet")


def run(alpha: float = 0.10) -> None:
    if not OOF_PATH.exists():
        raise FileNotFoundError(f"OOF predictions not found at {OOF_PATH}. Run train_runner first.")

    oof = pd.read_parquet(OOF_PATH)
    y = oof["label"].values
    p = oof["pred_prob"].values

    quantiles = calibrate_conformal(y, p, alpha=alpha)
    save_conformal_quantiles(quantiles)
    log.info("Conformal quantiles saved.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
