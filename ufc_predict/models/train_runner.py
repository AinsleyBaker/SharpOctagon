"""CLI entry point: load feature matrix and run full training pipeline.

After training, runs the full evaluation report on out-of-fold predictions —
critically including the closing-line (Vegas) benchmark when closing_odds_*
are available in the feature matrix.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from ufc_predict.eval.evaluate import full_report
from ufc_predict.models.train import MODEL_DIR, run_full_training

log = logging.getLogger(__name__)
MATRIX_PATH = Path("data/feature_matrix.parquet")


def run() -> None:
    if not MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {MATRIX_PATH}. Run build_matrix first."
        )

    df = pd.read_parquet(MATRIX_PATH)
    log.info("Loaded feature matrix: %s", df.shape)
    run_full_training(df)

    # Evaluation report — overall metrics, reliability, vs Vegas closing line,
    # Kelly ROI sim, per-year breakdown. Skipped silently if the OOF parquet
    # was not written (insufficient folds).
    oof_path = MODEL_DIR / "oof_predictions.parquet"
    if not oof_path.exists():
        log.warning("No OOF predictions written — skipping evaluation report")
        return
    oof_df = pd.read_parquet(oof_path)
    report = full_report(oof_df, output_dir=MODEL_DIR)
    (MODEL_DIR / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    log.info("Evaluation report saved to %s", MODEL_DIR / "evaluation_report.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
