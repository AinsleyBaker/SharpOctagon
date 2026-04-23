"""
Stage 14 — Model drift monitoring.

After each event, compares model log-loss on that event's fights
against a rolling baseline. Alerts if degradation exceeds threshold.

Run as part of the Monday weekly retrain workflow.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

log = logging.getLogger(__name__)

DRIFT_ALERT_THRESHOLD = 0.05      # alert if log-loss degrades more than this vs baseline
BASELINE_WINDOW_FIGHTS = 200      # rolling window for baseline log-loss
MONITOR_LOG = Path("data/monitor_log.jsonl")


def compute_recent_logloss(oof_path: Path, lookback_days: int = 90) -> dict | None:
    """Compute log-loss on fights in the last lookback_days from OOF predictions."""
    if not oof_path.exists():
        return None

    oof = pd.read_parquet(oof_path)
    cutoff = pd.Timestamp(date.today()) - pd.Timedelta(days=lookback_days)
    recent = oof[pd.to_datetime(oof["date"]) >= cutoff]

    if len(recent) < 10:
        log.info("Not enough recent fights (%d) for drift check", len(recent))
        return None

    y = recent["label"].values
    p = np.clip(recent["pred_prob"].values, 1e-7, 1 - 1e-7)
    ll = log_loss(y, p)

    # Baseline: older fights
    baseline = oof[pd.to_datetime(oof["date"]) < cutoff].tail(BASELINE_WINDOW_FIGHTS)
    if len(baseline) < 50:
        baseline_ll = None
    else:
        bl_y = baseline["label"].values
        bl_p = np.clip(baseline["pred_prob"].values, 1e-7, 1 - 1e-7)
        baseline_ll = log_loss(bl_y, bl_p)

    return {
        "date": str(date.today()),
        "recent_log_loss": ll,
        "baseline_log_loss": baseline_ll,
        "n_recent": len(recent),
        "n_baseline": len(baseline),
        "drift": (ll - baseline_ll) if baseline_ll else None,
        "alert": bool(baseline_ll and (ll - baseline_ll) > DRIFT_ALERT_THRESHOLD),
    }


def run(oof_path: Path = Path("models/oof_predictions.parquet")) -> dict | None:
    result = compute_recent_logloss(oof_path)
    if result is None:
        return None

    # Append to monitoring log
    MONITOR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(MONITOR_LOG, "a") as f:
        f.write(json.dumps(result) + "\n")

    if result["alert"]:
        log.warning(
            "DRIFT ALERT: recent log-loss %.4f vs baseline %.4f (delta=%.4f > threshold %.4f). "
            "Consider retraining with more recent data.",
            result["recent_log_loss"],
            result["baseline_log_loss"],
            result["drift"],
            DRIFT_ALERT_THRESHOLD,
        )
    else:
        log.info(
            "Drift check OK: recent=%.4f  baseline=%s  delta=%s",
            result["recent_log_loss"],
            f"{result['baseline_log_loss']:.4f}" if result["baseline_log_loss"] else "n/a",
            f"{result['drift']:.4f}" if result["drift"] else "n/a",
        )

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
