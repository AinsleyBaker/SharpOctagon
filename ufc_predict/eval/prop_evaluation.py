"""Prop model OOF evaluation — calibration without market prices.

We don't yet have historical prop closing-line odds (BFO scraper extension
is pending). But we can still evaluate whether the prop model's
probabilities are well-calibrated against actual fight outcomes:

  - Per-class log-loss (lower = better)
  - Per-class Brier (lower = better)
  - Reliability diagrams (predicted vs observed frequency by bin)
  - Top-prediction accuracy (fraction of fights where the highest-probability
    class was correct)
  - Base-rate comparison (does the model beat just predicting the mode?)

If calibration is bad, fix the model before pursuing market backtest.
If calibration is good, we have a strong prior that — IF prop markets are
less efficient than moneyline (well-documented in sports betting literature)
— our prop predictions can extract positive ROI even though our moneyline
cannot.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from ufc_predict.models.prop_models import METHOD_CLASSES, ROUND_CLASSES

log = logging.getLogger(__name__)
OOF_PATH = Path("models/prop_oof.parquet")
REPORT_PATH = Path("models/prop_evaluation.json")


def _per_class_metrics(probs: np.ndarray, y_true: np.ndarray, classes: list[str]) -> list[dict]:
    """Compute per-class binary metrics (one-vs-rest) on a probability matrix."""
    rows = []
    eps = 1e-7
    for j, cls in enumerate(classes):
        y_bin = (y_true == j).astype(int)
        p_j = np.clip(probs[:, j], eps, 1 - eps)
        try:
            ll = log_loss(y_bin, p_j, labels=[0, 1])
        except ValueError:
            ll = float("nan")
        rows.append({
            "class": cls,
            "log_loss": ll,
            "brier": brier_score_loss(y_bin, p_j),
            "base_rate": float(y_bin.mean()),
            "mean_pred": float(p_j.mean()),
            "n_positive": int(y_bin.sum()),
        })
    return rows


def _multiclass_log_loss(probs: np.ndarray, y_idx: np.ndarray) -> float:
    """Standard multi-class log-loss using full probability vectors."""
    eps = 1e-7
    p_chosen = np.clip(probs[np.arange(len(probs)), y_idx], eps, 1 - eps)
    return float(-np.mean(np.log(p_chosen)))


def _top_prediction_accuracy(probs: np.ndarray, y_idx: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == y_idx).mean())


def _empirical_base_rate_log_loss(y_idx: np.ndarray, n_classes: int) -> float:
    """The log-loss you'd get by predicting the empirical class distribution
    every time. Anything lower than this means the model is doing real work."""
    counts = np.bincount(y_idx, minlength=n_classes)
    base_probs = counts / counts.sum()
    fake_probs = np.tile(base_probs, (len(y_idx), 1))
    return _multiclass_log_loss(fake_probs, y_idx)


def evaluate(oof_path: Path = OOF_PATH) -> dict:
    if not oof_path.exists():
        raise FileNotFoundError(f"Prop OOF not found at {oof_path} — run prop_models.run_cv first")
    oof = pd.read_parquet(oof_path)
    log.info("Prop OOF loaded: %d rows", len(oof))

    # ------------------------------------------------------------------
    # Method model (6-class)
    # ------------------------------------------------------------------
    method_idx_map = {cls: i for i, cls in enumerate(METHOD_CLASSES)}
    valid_method = oof[oof["method_class_true"].isin(method_idx_map)].copy()
    if valid_method.empty:
        method_summary = {"skipped": "no valid method labels in OOF"}
    else:
        y_m = valid_method["method_class_true"].map(method_idx_map).values
        prob_cols = [f"prob_{c.lower()}" for c in METHOD_CLASSES]
        probs_m = valid_method[prob_cols].values
        # Renormalize defensively (should already sum to 1)
        probs_m = probs_m / probs_m.sum(axis=1, keepdims=True).clip(min=1e-9)

        method_summary = {
            "n": int(len(valid_method)),
            "multiclass_log_loss": _multiclass_log_loss(probs_m, y_m),
            "base_rate_log_loss": _empirical_base_rate_log_loss(y_m, len(METHOD_CLASSES)),
            "top_pred_accuracy": _top_prediction_accuracy(probs_m, y_m),
            "per_class": _per_class_metrics(probs_m, y_m, METHOD_CLASSES),
        }
        method_summary["lift_over_base_rate"] = (
            method_summary["base_rate_log_loss"] - method_summary["multiclass_log_loss"]
        )

    # ------------------------------------------------------------------
    # Round model (5-class, finish-conditional). Only finishes are scored;
    # decisions are excluded since the round model never sees them.
    # ------------------------------------------------------------------
    round_prob_cols = [f"prob_{c}" for c in ROUND_CLASSES]
    has_round = all(c in oof.columns for c in round_prob_cols)
    round_summary = {"skipped": "no round prob cols"}

    if has_round:
        finishes = oof[oof["round_class_true"].isin(ROUND_CLASSES)].copy()
        if not finishes.empty:
            round_idx_map = {c: i for i, c in enumerate(ROUND_CLASSES)}
            y_r = finishes["round_class_true"].map(round_idx_map).values
            probs_r = finishes[round_prob_cols].values
            probs_r = probs_r / probs_r.sum(axis=1, keepdims=True).clip(min=1e-9)

            round_summary = {
                "n_finishes": int(len(finishes)),
                "multiclass_log_loss": _multiclass_log_loss(probs_r, y_r),
                "base_rate_log_loss": _empirical_base_rate_log_loss(y_r, len(ROUND_CLASSES)),
                "top_pred_accuracy": _top_prediction_accuracy(probs_r, y_r),
                "per_class": _per_class_metrics(probs_r, y_r, ROUND_CLASSES),
            }
            round_summary["lift_over_base_rate"] = (
                round_summary["base_rate_log_loss"] - round_summary["multiclass_log_loss"]
            )

    # ------------------------------------------------------------------
    # Aggregate KO-vs-SUB-vs-DEC (3-class neutral, both fighters combined).
    # This is what method_neutral / "How will fight end" markets bet on.
    # ------------------------------------------------------------------
    if not valid_method.empty:
        prob_ko = valid_method["prob_a_ko_tko"].values + valid_method["prob_b_ko_tko"].values
        prob_sub = valid_method["prob_a_sub"].values + valid_method["prob_b_sub"].values
        prob_dec = valid_method["prob_a_dec"].values + valid_method["prob_b_dec"].values
        neutral_probs = np.stack([prob_ko, prob_sub, prob_dec], axis=1)
        neutral_probs = neutral_probs / neutral_probs.sum(axis=1, keepdims=True).clip(min=1e-9)

        # Map true labels: 0=KO, 1=SUB, 2=DEC
        def _neutral_idx(cls: str) -> int:
            if cls.endswith("_KO_TKO"): return 0
            if cls.endswith("_SUB"):    return 1
            return 2
        y_n = valid_method["method_class_true"].map(_neutral_idx).values
        method_summary["neutral_3class"] = {
            "log_loss": _multiclass_log_loss(neutral_probs, y_n),
            "base_rate_log_loss": _empirical_base_rate_log_loss(y_n, 3),
            "top_pred_accuracy": _top_prediction_accuracy(neutral_probs, y_n),
            "per_class": _per_class_metrics(neutral_probs, y_n, ["KO_TKO", "SUB", "DEC"]),
        }

    # Distance market: P(fight goes to decision) vs actual decision
    if not valid_method.empty:
        prob_dec_total = valid_method["prob_a_dec"].values + valid_method["prob_b_dec"].values
        actual_dec = valid_method["method_class_true"].str.endswith("_DEC").astype(int).values
        eps = 1e-7
        p_clip = np.clip(prob_dec_total, eps, 1 - eps)
        method_summary["distance_market"] = {
            "log_loss": float(log_loss(actual_dec, p_clip, labels=[0, 1])),
            "brier": float(brier_score_loss(actual_dec, prob_dec_total)),
            "base_rate": float(actual_dec.mean()),
            "mean_pred": float(prob_dec_total.mean()),
            "calibration_offset": float(actual_dec.mean() - prob_dec_total.mean()),
        }

    report = {"method": method_summary, "round": round_summary}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Pretty print
    print("\n=== METHOD MODEL (6-class) ===")
    if "skipped" not in method_summary:
        print(f"n={method_summary['n']}  multiclass LL={method_summary['multiclass_log_loss']:.4f}"
              f"  base-rate LL={method_summary['base_rate_log_loss']:.4f}"
              f"  lift={method_summary['lift_over_base_rate']:+.4f}")
        print(f"top-prediction accuracy: {100*method_summary['top_pred_accuracy']:.1f}%")
        print(f"\n  {'class':10s}  {'n+':>5s}  {'base':>6s}  {'pred':>6s}  {'LL':>7s}  {'Brier':>7s}")
        for r in method_summary["per_class"]:
            print(f"  {r['class']:10s}  {r['n_positive']:>5d}  "
                  f"{100*r['base_rate']:>5.1f}%  {100*r['mean_pred']:>5.1f}%  "
                  f"{r['log_loss']:>7.4f}  {r['brier']:>7.4f}")

        n3 = method_summary["neutral_3class"]
        print(f"\n  KO/SUB/DEC neutral (3-class):  LL={n3['log_loss']:.4f}  base={n3['base_rate_log_loss']:.4f}  top-acc={100*n3['top_pred_accuracy']:.1f}%")
        d = method_summary["distance_market"]
        print(f"  Distance (goes-to-decision):  LL={d['log_loss']:.4f}  base-rate={100*d['base_rate']:.1f}%  mean-pred={100*d['mean_pred']:.1f}%  offset={d['calibration_offset']:+.4f}")

    print("\n=== ROUND MODEL (5-class, finishes only) ===")
    if "skipped" not in round_summary:
        print(f"n={round_summary['n_finishes']}  multiclass LL={round_summary['multiclass_log_loss']:.4f}"
              f"  base-rate LL={round_summary['base_rate_log_loss']:.4f}"
              f"  lift={round_summary['lift_over_base_rate']:+.4f}")
        print(f"top-prediction accuracy: {100*round_summary['top_pred_accuracy']:.1f}%")
        print(f"\n  {'class':6s}  {'n+':>5s}  {'base':>6s}  {'pred':>6s}  {'LL':>7s}  {'Brier':>7s}")
        for r in round_summary["per_class"]:
            print(f"  {r['class']:6s}  {r['n_positive']:>5d}  "
                  f"{100*r['base_rate']:>5.1f}%  {100*r['mean_pred']:>5.1f}%  "
                  f"{r['log_loss']:>7.4f}  {r['brier']:>7.4f}")

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    evaluate()
