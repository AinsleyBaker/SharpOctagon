"""
Stage 9 — Evaluation harness.

Metrics: log-loss, Brier, AUC, reliability diagram.
Closing-line benchmark: compares our log-loss to Vegas implied-prob log-loss.
Kelly ROI simulation: historical bankroll growth at fractional Kelly sizing.

Closing odds are NEVER used as model features — only here in evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

log = logging.getLogger(__name__)

# Fraction of full Kelly to bet (quarter-Kelly is standard for risk management)
KELLY_FRACTION = 0.25
STARTING_BANKROLL = 1000.0


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute log-loss, Brier, AUC for a set of predictions."""
    # Clip to avoid log(0) — though calibrated probs should be bounded
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return {
        "log_loss": log_loss(y_true, y_pred),
        "brier":    brier_score_loss(y_true, y_pred),
        "auc":      roc_auc_score(y_true, y_pred),
        "n":        int(len(y_true)),
    }


def reliability_data(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Compute reliability diagram data (10 equal-width bins).
    Returns DataFrame with: bin_mid, mean_pred, frac_pos, count.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(y_pred, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    rows = []
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_mid":   (bins[i] + bins[i + 1]) / 2,
            "mean_pred": y_pred[mask].mean(),
            "frac_pos":  y_true[mask].mean(),
            "count":     mask.sum(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Closing-line benchmark
# ---------------------------------------------------------------------------

def american_odds_to_implied_prob(odds: float) -> float:
    """Convert American moneyline to decimal implied probability."""
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def remove_vig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Remove the bookmaker's vig (overround) to get fair probabilities."""
    total = prob_a + prob_b
    return prob_a / total, prob_b / total


def closing_line_log_loss(
    y_true: np.ndarray,
    closing_odds_a: np.ndarray,
    closing_odds_b: np.ndarray,
) -> float:
    """
    Compute Vegas closing line log-loss (the benchmark we're trying to beat).
    Uses vig-removed probabilities so it's a fair comparison.
    """
    probs = []
    for oa, ob in zip(closing_odds_a, closing_odds_b):
        if np.isnan(oa) or np.isnan(ob):
            probs.append(np.nan)
            continue
        pa = american_odds_to_implied_prob(oa)
        pb = american_odds_to_implied_prob(ob)
        pa_fair, _ = remove_vig(pa, pb)
        probs.append(pa_fair)

    probs = np.array(probs)
    valid = ~np.isnan(probs)
    if valid.sum() == 0:
        return np.nan
    return log_loss(y_true[valid], np.clip(probs[valid], 1e-7, 1 - 1e-7))


def benchmark_vs_closing_line(
    y_true: np.ndarray,
    y_model: np.ndarray,
    closing_odds_a: np.ndarray,
    closing_odds_b: np.ndarray,
) -> dict:
    """Compare model log-loss to Vegas closing-line log-loss."""
    model_ll = log_loss(y_true, np.clip(y_model, 1e-7, 1 - 1e-7))
    vegas_ll = closing_line_log_loss(y_true, closing_odds_a, closing_odds_b)

    result = {
        "model_log_loss": model_ll,
        "vegas_log_loss": vegas_ll,
        "delta": vegas_ll - model_ll,  # positive = model beats Vegas
        "model_beats_vegas": model_ll < vegas_ll,
    }
    log.info(
        "Log-loss: model=%.4f  vegas=%.4f  delta=%.4f  (%s)",
        model_ll, vegas_ll, result["delta"],
        "MODEL BEATS VEGAS" if result["model_beats_vegas"] else "vegas wins",
    )
    return result


# ---------------------------------------------------------------------------
# Kelly ROI simulation
# ---------------------------------------------------------------------------

def kelly_fraction(prob_win: float, decimal_odds: float, fraction: float = KELLY_FRACTION) -> float:
    """
    Compute fractional Kelly bet size as fraction of bankroll.
    prob_win: our model's estimated win probability for fighter A.
    decimal_odds: decimal odds for fighter A (e.g. 2.5 = +150 American).
    fraction: Kelly fraction (0.25 = quarter-Kelly).
    Returns bet size as fraction of bankroll (0 if negative EV).
    """
    b = decimal_odds - 1.0     # net profit per unit staked
    q = 1 - prob_win
    kelly = (b * prob_win - q) / b
    return max(0.0, kelly * fraction)


def american_to_decimal(american: float) -> float:
    if american >= 0:
        return american / 100.0 + 1.0
    else:
        return 100.0 / abs(american) + 1.0


def kelly_roi_simulation(
    y_true: np.ndarray,
    y_model: np.ndarray,
    closing_odds_a: np.ndarray,
    kelly_fraction: float = KELLY_FRACTION,
    starting_bankroll: float = STARTING_BANKROLL,
) -> pd.DataFrame:
    """
    Simulate bankroll growth using fractional Kelly bet sizing.

    For each fight where we have a model probability and closing odds:
      - If model_prob > implied_prob (we have edge), bet on A.
      - If model_prob < implied_prob, bet on B (same logic, inverted).
      - Bet size = Kelly fraction × bankroll.

    Returns a DataFrame with fight-by-fight bankroll history.
    """
    bankroll = starting_bankroll
    rows = []

    for i, (won, prob_a, odds_a) in enumerate(zip(y_true, y_model, closing_odds_a)):
        if np.isnan(odds_a):
            continue

        dec_odds_a = american_to_decimal(odds_a)
        implied_a = 1.0 / dec_odds_a

        # Decide which side to bet
        if prob_a > implied_a:
            bet_side = "A"
            our_prob = prob_a
            dec_odds = dec_odds_a
            outcome = int(won)
        else:
            bet_side = "B"
            our_prob = 1 - prob_a
            dec_odds_b = american_to_decimal(-odds_a)  # approximate B odds
            dec_odds = dec_odds_b
            outcome = 1 - int(won)

        frac = kelly_fraction_fn(our_prob, dec_odds, kelly_fraction)
        bet_amount = bankroll * frac

        pnl = bet_amount * (dec_odds - 1) if outcome == 1 else -bet_amount
        bankroll += pnl

        rows.append({
            "fight_idx": i,
            "bet_side": bet_side,
            "our_prob": our_prob,
            "implied_prob": implied_a if bet_side == "A" else (1 - implied_a),
            "edge": our_prob - (implied_a if bet_side == "A" else (1 - implied_a)),
            "kelly_frac": frac,
            "bet_amount": bet_amount,
            "pnl": pnl,
            "bankroll": bankroll,
            "outcome": outcome,
        })

    df = pd.DataFrame(rows)
    if len(df):
        total_roi = (bankroll - starting_bankroll) / starting_bankroll
        log.info(
            "Kelly ROI simulation: %d bets, final bankroll=%.2f, ROI=%.1f%%",
            len(df), bankroll, total_roi * 100,
        )
    return df


def kelly_fraction_fn(prob_win: float, decimal_odds: float, fraction: float = KELLY_FRACTION) -> float:
    b = decimal_odds - 1.0
    q = 1 - prob_win
    kelly = (b * prob_win - q) / b
    return max(0.0, kelly * fraction)


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------

def full_report(
    oof_df: pd.DataFrame,
    output_dir: Path = Path("models"),
) -> dict:
    """
    Run the complete evaluation suite on out-of-fold predictions.
    Expects oof_df to have: label, pred_prob, closing_odds_red, closing_odds_blue (optional).
    """
    y = oof_df["label"].values
    p = oof_df["pred_prob"].values

    metrics = compute_metrics(y, p)
    log.info("Overall — log_loss=%.4f  brier=%.4f  auc=%.4f  n=%d", **metrics)

    report = {"overall": metrics}

    # Reliability data
    rel = reliability_data(y, p)
    rel.to_csv(output_dir / "reliability.csv", index=False)

    # Closing-line benchmark (if odds available)
    if "closing_odds_red" in oof_df.columns:
        # Map closing odds back to fighter_a direction (corner randomization was applied)
        # In oof_df, fighter_a is whichever was assigned "A" after corner randomization.
        # closing_odds_red corresponds to the original red corner.
        # We need to track whether A was red or blue — store that during feature building.
        # For now: best-effort using stored columns if present.
        if "a_is_red" in oof_df.columns:
            odds_a = np.where(
                oof_df["a_is_red"],
                oof_df["closing_odds_red"].values,
                oof_df["closing_odds_blue"].values,
            )
        else:
            odds_a = oof_df["closing_odds_red"].values  # approximate

        bench = benchmark_vs_closing_line(y, p, odds_a, oof_df["closing_odds_blue"].values)
        report["vs_closing_line"] = bench

        kelly_df = kelly_roi_simulation(y, p, odds_a)
        kelly_df.to_csv(output_dir / "kelly_simulation.csv", index=False)

        if len(kelly_df):
            report["kelly_roi"] = {
                "n_bets": len(kelly_df),
                "final_bankroll": float(kelly_df["bankroll"].iloc[-1]),
                "roi_pct": float((kelly_df["bankroll"].iloc[-1] - STARTING_BANKROLL) / STARTING_BANKROLL * 100),
                "win_rate": float((kelly_df["outcome"] == 1).mean()),
            }

    # Per-year breakdown
    oof_df["year"] = pd.to_datetime(oof_df["date"]).dt.year
    yearly = []
    for yr, grp in oof_df.groupby("year"):
        m = compute_metrics(grp["label"].values, grp["pred_prob"].values)
        m["year"] = yr
        yearly.append(m)
    pd.DataFrame(yearly).to_csv(output_dir / "yearly_metrics.csv", index=False)
    report["yearly"] = yearly

    return report
