"""
Stage 11 — Live predictor.

For each upcoming bout in the DB:
  1. Build as-of features for both fighters (as of today)
  2. Attach Elo/Glicko-2 ratings
  3. Run ensemble prediction → mean prob + std
  4. Compute split-conformal prediction intervals
  5. Add Kelly fraction (with closing odds if available)
  6. Write to predictions table / CSV / JSON

Conformal calibration is computed offline on OOF predictions (calibrate_conformal()).
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ufc_predict.db.models import Fighter, UpcomingBout
from ufc_predict.eval.evaluate import kelly_fraction_fn, american_to_decimal
from ufc_predict.eval.bet_analysis import analyze_all_fights
from ufc_predict.features.aso_features import fighter_aso_stats, _fighter_age
from ufc_predict.models.prop_models import load_prop_artifacts, predict_props
from ufc_predict.models.train import (
    FEATURE_COLS,
    ensemble_predict,
    load_artifacts,
)

log = logging.getLogger(__name__)

PREDICTIONS_PATH = Path("data/predictions.json")
CONFORMAL_QUANTILES_PATH = Path("models/conformal_quantiles.json")


# ---------------------------------------------------------------------------
# Split-conformal prediction intervals
# ---------------------------------------------------------------------------

def calibrate_conformal(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.10,
) -> dict:
    """
    Compute split-conformal calibration on OOF predictions.
    alpha = 0.10 → 90% coverage intervals.

    Returns a dict with quantile values for lower and upper bounds.
    """
    # Nonconformity score: symmetric absolute residual on logit scale
    # Using pinball / quantile approach on raw probabilities
    residuals = np.abs(y_true - y_pred)
    n = len(residuals)

    # Quantile level adjusted for finite-sample coverage guarantee
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    q = float(np.quantile(residuals, level))

    result = {"alpha": alpha, "conformal_halfwidth": q, "n_calibration": n}
    log.info("Conformal halfwidth (%.0f%% coverage): %.4f", (1 - alpha) * 100, q)
    return result


def conformal_interval(
    y_pred: np.ndarray,
    halfwidth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply pre-computed conformal halfwidth to get prediction intervals."""
    lo = np.clip(y_pred - halfwidth, 0.0, 1.0)
    hi = np.clip(y_pred + halfwidth, 0.0, 1.0)
    return lo, hi


def load_conformal_quantiles() -> dict | None:
    if not CONFORMAL_QUANTILES_PATH.exists():
        return None
    with open(CONFORMAL_QUANTILES_PATH) as f:
        return json.load(f)


def save_conformal_quantiles(quantiles: dict) -> None:
    CONFORMAL_QUANTILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFORMAL_QUANTILES_PATH, "w") as f:
        json.dump(quantiles, f, indent=2)


# ---------------------------------------------------------------------------
# Feature building for upcoming bouts
# ---------------------------------------------------------------------------

def build_upcoming_features(session: Session) -> pd.DataFrame:
    """
    Build feature rows for all unresolved upcoming bouts.
    Uses today as the as-of date.
    """
    today = date.today()

    bouts = (
        session.query(UpcomingBout)
        .filter(
            UpcomingBout.event_date >= today,
            UpcomingBout.is_cancelled.is_(False),
            UpcomingBout.red_fighter_id.isnot(None),
            UpcomingBout.blue_fighter_id.isnot(None),
        )
        .all()
    )

    log.info("Building features for %d upcoming bouts", len(bouts))
    rows = []

    for bout in bouts:
        a_id = bout.red_fighter_id
        b_id = bout.blue_fighter_id

        a_feat = fighter_aso_stats(a_id, today, session, bout.weight_class)
        b_feat = fighter_aso_stats(b_id, today, session, bout.weight_class)

        a_age = _fighter_age(a_id, today, session)
        b_age = _fighter_age(b_id, today, session)

        a_name = session.get(Fighter, a_id).full_name if session.get(Fighter, a_id) else a_id
        b_name = session.get(Fighter, b_id).full_name if session.get(Fighter, b_id) else b_id

        row = {
            "upcoming_bout_id": bout.upcoming_bout_id,
            "event_date": bout.event_date,
            "event_name": bout.event_name,
            "fighter_a_id": a_id,
            "fighter_b_id": b_id,
            "fighter_a_name": a_name,
            "fighter_b_name": b_name,
            "weight_class": bout.weight_class,
            "is_title_bout": int(bool(bout.is_title_bout)),
            "is_five_round": int(bool(bout.is_five_round)),
            "a_n_fights": a_feat["n_fights"],
            "b_n_fights": b_feat["n_fights"],
            "a_short_notice": 0,
            "b_short_notice": 0,
            "a_missed_weight": 0,
            "b_missed_weight": 0,
            "a_age": a_age,
            "b_age": b_age,
        }

        diff_keys = [
            "win_rate", "finish_rate", "ko_rate", "sub_rate",
            "slpm", "sapm", "sig_acc", "td_per_min", "td_acc",
            "sub_per_min", "ctrl_ratio",
            "l3_win_rate", "l5_win_rate", "l3_finish_rate",
            "l3_kd", "l3_td_rate", "l3_slpm", "l5_slpm",
            "win_streak", "loss_streak", "fight_frequency_24m",
        ]
        for k in diff_keys:
            av = a_feat.get(k, np.nan)
            bv = b_feat.get(k, np.nan)
            row[f"diff_{k}"] = av - bv if not (pd.isna(av) or pd.isna(bv)) else np.nan

        if a_age and b_age:
            row["diff_age"] = a_age - b_age
        else:
            row["diff_age"] = np.nan

        row["a_win_streak"]  = int(a_feat.get("win_streak",  0) or 0)
        row["b_win_streak"]  = int(b_feat.get("win_streak",  0) or 0)
        row["a_loss_streak"] = int(a_feat.get("loss_streak", 0) or 0)
        row["b_loss_streak"] = int(b_feat.get("loss_streak", 0) or 0)
        row["a_l3_win_rate"] = a_feat.get("l3_win_rate")
        row["b_l3_win_rate"] = b_feat.get("l3_win_rate")

        # Absolute finish rates (see aso_features.py — used by prop model)
        for k in ("ko_rate", "sub_rate", "finish_rate", "sub_per_min", "td_per_min"):
            row[f"a_{k}"] = a_feat.get(k, np.nan)
            row[f"b_{k}"] = b_feat.get(k, np.nan)

        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Main prediction runner
# ---------------------------------------------------------------------------

def run_predictions(db_url: str | None = None) -> pd.DataFrame:
    """
    Full prediction pipeline for upcoming bouts.
    Returns a DataFrame ready for the serving layer.
    """
    from ufc_predict.db.session import get_session_factory
    from ufc_predict.features.ratings import attach_ratings

    factory = get_session_factory(db_url)

    # Load model artifacts
    try:
        model, iso, ensemble, feature_cols = load_artifacts()
    except FileNotFoundError:
        log.error("No trained model found. Run ufc-train first.")
        return pd.DataFrame()

    with factory() as session:
        upcoming_df = build_upcoming_features(session)

    if upcoming_df.empty:
        log.info("No upcoming bouts with resolved fighter IDs.")
        return pd.DataFrame()

    # Attach Elo/Glicko — we need to reconstruct ratings from historical data
    # For upcoming bouts, fetch the latest stored ratings (from training run)
    # For now, fill with NaN — ratings are attached during the full feature build
    upcoming_df["diff_elo"] = np.nan
    upcoming_df["diff_glicko"] = np.nan
    upcoming_df["glicko_rd_a"] = np.nan
    upcoming_df["glicko_rd_b"] = np.nan

    # Build feature matrix
    available = [c for c in feature_cols if c in upcoming_df.columns]
    X = upcoming_df[available].copy()

    # Ensemble prediction → mean ± std
    mean_prob, std_prob = ensemble_predict(ensemble, X)
    upcoming_df["prob_a_wins"] = mean_prob
    upcoming_df["prob_b_wins"] = 1 - mean_prob
    upcoming_df["uncertainty_std"] = std_prob

    # Conformal prediction intervals
    quantiles = load_conformal_quantiles()
    if quantiles:
        hw = quantiles["conformal_halfwidth"]
        lo, hi = conformal_interval(mean_prob, hw)
        coverage = int((1 - quantiles["alpha"]) * 100)
        upcoming_df[f"ci_{coverage}_lo"] = lo
        upcoming_df[f"ci_{coverage}_hi"] = hi
    else:
        log.warning("No conformal quantiles found — run calibrate_conformal() after training.")

    # Kelly fraction (if closing odds become available before event)
    # Placeholder: will be populated by the odds scraper
    upcoming_df["kelly_fraction"] = np.nan
    upcoming_df["has_edge"] = False

    # Format output
    output = upcoming_df[[
        "upcoming_bout_id", "event_date", "event_name",
        "fighter_a_name", "fighter_b_name", "weight_class",
        "is_title_bout", "is_five_round",
        "a_n_fights", "b_n_fights",
        "a_win_streak", "b_win_streak", "a_loss_streak", "b_loss_streak",
        "a_l3_win_rate", "b_l3_win_rate",
        "prob_a_wins", "prob_b_wins", "uncertainty_std",
        *(c for c in upcoming_df.columns if c.startswith("ci_")),
        "kelly_fraction", "has_edge",
    ]].copy()

    output = output.sort_values(["event_date", "is_title_bout"], ascending=[True, False])

    # -- Prop predictions (method + round) ---------------------------------
    # Pass the full upcoming_df so predict_props can use absolute finish-rate
    # features (a_ko_rate, b_ko_rate, etc.) that the prop model is trained on.
    prop_artifacts = load_prop_artifacts()
    props_list = predict_props(upcoming_df, mean_prob, prop_artifacts or {})

    # Map by upcoming_bout_id so the sort_values reindex doesn't break alignment
    bout_id_to_props = {
        row["upcoming_bout_id"]: props_list[i]
        for i, (_, row) in enumerate(upcoming_df.iterrows())
    }

    # -- SportsBet odds + EV analysis --------------------------------------
    predictions_list = _df_to_records(output, bout_id_to_props)
    try:
        from ufc_predict.ingest.sportsbet_scraper import (
            fetch_ufc_markets, load_markets, save_markets, match_odds_to_predictions,
        )
        # Prefer cached odds (committed to repo) so CI runners don't need
        # geo-access to sportsbet.com.au. Fall back to live fetch if no cache.
        sb_fights = load_markets()
        if not sb_fights:
            log.info("No cached SportsBet odds — attempting live fetch…")
            sb_fights = fetch_ufc_markets()
            if sb_fights:
                save_markets(sb_fights)

        if sb_fights:
            predictions_list = match_odds_to_predictions(sb_fights, predictions_list)
            log.info("SportsBet odds matched for %d/%d fights",
                     sum(1 for p in predictions_list if p.get("sportsbet_odds")),
                     len(predictions_list))
        else:
            log.warning("No SportsBet markets available — run sportsbet_scraper locally to cache odds")
    except Exception as exc:
        log.warning("SportsBet odds step failed (continuing without odds): %s", exc)

    # -- EV analysis -------------------------------------------------------
    predictions_list = analyze_all_fights(predictions_list)

    _save_predictions_list(predictions_list)
    log.info("Generated predictions for %d upcoming bouts", len(predictions_list))
    return output


def _df_to_records(df: pd.DataFrame, bout_id_to_props: dict | None = None) -> list[dict]:
    records = []
    for _, row in df.iterrows():
        r = row.to_dict()
        r["event_date"] = str(r["event_date"])
        bid = r.get("upcoming_bout_id", "")
        r["props"] = (bout_id_to_props or {}).get(bid) or {}
        records.append(r)
    return records


def _save_predictions_list(records: list[dict]) -> None:
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_PATH, "w") as f:
        json.dump(records, f, indent=2, default=str)
    log.info("Predictions written to %s", PREDICTIONS_PATH)


def _save_predictions(df: pd.DataFrame) -> None:
    _save_predictions_list(_df_to_records(df))


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    df = run_predictions()
    if not df.empty:
        print(df[["fighter_a_name", "fighter_b_name", "prob_a_wins", "prob_b_wins"]].to_string(index=False))
