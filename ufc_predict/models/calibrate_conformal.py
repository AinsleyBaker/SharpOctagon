"""
CLI entry point: compute conformal prediction quantiles from OOF predictions.
Run after train_runner.py, before predict.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ufc_predict.models.predict import (
    calibrate_conformal,
    calibrate_conformal_locally_weighted,
    calibrate_conformal_mondrian,
    save_conformal_quantiles,
)

log = logging.getLogger(__name__)
OOF_PATH = Path("models/oof_predictions.parquet")
MONDRIAN_PATH = Path("models/conformal_quantiles_mondrian.json")
LOCALLY_WEIGHTED_PATH = Path("models/conformal_quantiles_locally_weighted.json")


def run(alpha: float = 0.10) -> None:
    if not OOF_PATH.exists():
        raise FileNotFoundError(f"OOF predictions not found at {OOF_PATH}. Run train_runner first.")

    oof = pd.read_parquet(OOF_PATH)
    y = oof["label"].values
    p = oof["pred_prob"].values

    # Global halfwidth (kept for backward compatibility)
    quantiles = calibrate_conformal(y, p, alpha=alpha)
    save_conformal_quantiles(quantiles)
    log.info("Conformal quantiles saved.")

    # Mondrian (per weight class) — preferred when available at predict time.
    if "weight_class_clean" in oof.columns:
        groups = oof["weight_class_clean"].astype(str).fillna("Unknown").values
        m = calibrate_conformal_mondrian(y, p, groups, alpha=alpha)
        import json
        MONDRIAN_PATH.parent.mkdir(parents=True, exist_ok=True)
        MONDRIAN_PATH.write_text(json.dumps(m, indent=2), encoding="utf-8")
        log.info("Mondrian conformal saved (%d groups).", len(m["per_group"]))

    # Locally-weighted (Bernoulli-SD-normalized) conformal — preferred over
    # Mondrian when both are available because it gives per-prediction
    # halfwidths instead of per-group constants.
    import json
    lw = calibrate_conformal_locally_weighted(y, p, alpha=alpha)
    LOCALLY_WEIGHTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCALLY_WEIGHTED_PATH.write_text(json.dumps(lw, indent=2), encoding="utf-8")
    log.info("Locally-weighted conformal saved.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
