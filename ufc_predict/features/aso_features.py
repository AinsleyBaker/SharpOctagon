"""
Stage 6 — As-of feature builder.

Computes fighter features using ONLY fights with date < target_date.
This is the primary anti-leakage guard: never use cumulative stats that
include the fight being predicted.

Key output: build_fight_feature_rows() → DataFrame where each row is one
(fight, fighter_A, fighter_B) with difference features ready for the model.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)

# Minimum fights before we trust rolling stats (use shrinkage below this)
MIN_FIGHTS_TRUST = 5
# Weight class means for shrinkage (populated lazily from DB)
_WC_MEANS: dict[str, dict[str, float]] = {}


# ---------------------------------------------------------------------------
# Core feature computation per (fighter, as-of date)
# ---------------------------------------------------------------------------

def fighter_aso_stats(
    fighter_id: str,
    as_of: date,
    session: Session,
    weight_class: str | None = None,
) -> dict:
    """
    Compute all per-fighter features as of `as_of` date.
    Uses only fights strictly before as_of.
    """
    sql = text("""
        SELECT
            f.fight_id,
            f.date,
            f.weight_class,
            f.winner_fighter_id,
            f.method,
            f.is_five_round,
            f.red_fighter_id,
            f.blue_fighter_id,
            s.knockdowns,
            s.sig_strikes_landed,   s.sig_strikes_attempted,
            s.total_strikes_landed, s.total_strikes_attempted,
            s.head_landed,          s.head_attempted,
            s.takedowns_landed,     s.takedowns_attempted,
            s.submission_attempts,
            s.control_time_sec,
            s.ground_landed,        s.ground_attempted,
            f.time_ended_sec,
            f.round_ended
        FROM fights f
        JOIN fight_stats_round s
          ON s.fight_id = f.fight_id AND s.fighter_id = :fid AND s.round = 0
        WHERE (f.red_fighter_id = :fid OR f.blue_fighter_id = :fid)
          AND f.date < :as_of
          AND f.method NOT IN ('NC', 'DQ', 'CANCELLED')
          AND f.method IS NOT NULL
        ORDER BY f.date ASC
    """)

    rows = session.execute(sql, {"fid": fighter_id, "as_of": as_of}).fetchall()

    if not rows:
        return _empty_features(fighter_id, as_of)

    df = pd.DataFrame(rows, columns=[
        "fight_id", "date", "weight_class", "winner_fighter_id",
        "method", "is_five_round", "red_fighter_id", "blue_fighter_id",
        "knockdowns", "sig_landed", "sig_attempted",
        "total_landed", "total_attempted",
        "head_landed", "head_attempted",
        "td_landed", "td_attempted",
        "sub_attempts", "ctrl_sec", "ground_landed", "ground_attempted",
        "time_ended_sec", "round_ended",
    ])

    df["date"] = pd.to_datetime(df["date"])
    df["won"] = (df["winner_fighter_id"] == fighter_id).astype(int)
    df["finish"] = df["method"].isin(["KO", "TKO", "SUB"]).astype(int)
    df["ko_tko"] = df["method"].isin(["KO", "TKO"]).astype(int)
    df["sub_win"] = ((df["method"] == "SUB") & (df["winner_fighter_id"] == fighter_id)).astype(int)

    # Total fight time in minutes (for per-minute rates)
    df["total_time_min"] = df["time_ended_sec"].fillna(0) / 60.0
    df["total_time_min"] = df["total_time_min"].clip(lower=0.5)  # avoid div/0

    n = len(df)
    last_fight_date = df["date"].iloc[-1].date()
    days_since_last = (as_of - last_fight_date).days

    # --- Career rates (per-minute) ---
    total_min = df["total_time_min"].sum()
    total_min = max(total_min, 0.5)

    slpm = df["sig_landed"].sum() / total_min
    sapm = df["sig_attempted"].sum() / total_min
    td_pm = df["td_landed"].sum() / total_min
    sub_pm = df["sub_attempts"].sum() / total_min
    ctrl_pm = df["ctrl_sec"].sum() / (total_min * 60)

    sig_acc = _safe_ratio(df["sig_landed"].sum(), df["sig_attempted"].sum())
    td_acc = _safe_ratio(df["td_landed"].sum(), df["td_attempted"].sum())

    # --- Rolling last-3 and last-5 form ---
    l3 = df.tail(3)
    l5 = df.tail(5)

    l3_win_rate = l3["won"].mean()
    l5_win_rate = l5["won"].mean()
    l3_finish_rate = l3["finish"].mean()
    l3_kd = l3["knockdowns"].sum()
    l3_td_rate = _safe_ratio(l3["td_landed"].sum(), l3["td_attempted"].sum())
    l3_slpm = l3["sig_landed"].sum() / max(l3["total_time_min"].sum(), 0.5)
    l5_slpm = l5["sig_landed"].sum() / max(l5["total_time_min"].sum(), 0.5)

    # Win streak / loss streak
    outcomes = df["won"].tolist()
    win_streak = _current_streak(outcomes, 1)
    loss_streak = _current_streak(outcomes, 0)

    features = {
        "fighter_id": fighter_id,
        "as_of": as_of,
        "n_fights": n,
        "wins": int(df["won"].sum()),
        "losses": int((df["won"] == 0).sum()),
        "win_rate": df["won"].mean(),
        "finish_rate": df["finish"].mean(),
        "ko_rate": df["ko_tko"].mean(),
        "sub_rate": df["sub_win"].mean(),
        "slpm": slpm,
        "sapm": sapm,
        "sig_acc": sig_acc,
        "td_per_min": td_pm,
        "td_acc": td_acc,
        "sub_per_min": sub_pm,
        "ctrl_ratio": ctrl_pm,
        "l3_win_rate": l3_win_rate,
        "l5_win_rate": l5_win_rate,
        "l3_finish_rate": l3_finish_rate,
        "l3_kd": l3_kd,
        "l3_td_rate": l3_td_rate,
        "l3_slpm": l3_slpm,
        "l5_slpm": l5_slpm,
        "win_streak": win_streak,
        "loss_streak": loss_streak,
        "days_since_last_fight": days_since_last,
        "fight_frequency_24m": _fight_freq(df, as_of, months=24),
    }

    # Apply Bayesian shrinkage for fighters with few fights
    if n < MIN_FIGHTS_TRUST and weight_class:
        features = _apply_shrinkage(features, weight_class, session)

    return features


def _safe_ratio(num: float, denom: float) -> float:
    return num / denom if denom > 0 else 0.0


def _current_streak(outcomes: list[int], target: int) -> int:
    streak = 0
    for o in reversed(outcomes):
        if o == target:
            streak += 1
        else:
            break
    return streak


def _fight_freq(df: pd.DataFrame, as_of: date, months: int) -> float:
    cutoff = pd.Timestamp(as_of) - pd.DateOffset(months=months)
    recent = df[df["date"] >= cutoff]
    return len(recent) / (months / 12)  # fights per year


def _empty_features(fighter_id: str, as_of: date) -> dict:
    """Return a zeroed feature dict for a fighter with no prior fights (debut)."""
    feat = {
        "fighter_id": fighter_id, "as_of": as_of, "n_fights": 0,
        "wins": 0, "losses": 0, "win_rate": np.nan, "finish_rate": np.nan,
        "ko_rate": np.nan, "sub_rate": np.nan,
        "slpm": np.nan, "sapm": np.nan, "sig_acc": np.nan,
        "td_per_min": np.nan, "td_acc": np.nan,
        "sub_per_min": np.nan, "ctrl_ratio": np.nan,
        "l3_win_rate": np.nan, "l5_win_rate": np.nan, "l3_finish_rate": np.nan,
        "l3_kd": 0, "l3_td_rate": np.nan, "l3_slpm": np.nan, "l5_slpm": np.nan,
        "win_streak": 0, "loss_streak": 0,
        "days_since_last_fight": np.nan, "fight_frequency_24m": 0.0,
    }
    return feat


def _apply_shrinkage(
    features: dict, weight_class: str, session: Session, alpha: float = 0.3
) -> dict:
    """
    Shrink rookie stats toward the weight-class mean.
    alpha controls how much weight the prior gets: higher = more shrinkage.
    alpha is scaled by (1 - n_fights / MIN_FIGHTS_TRUST) so it fades out.
    """
    global _WC_MEANS
    if weight_class not in _WC_MEANS:
        _WC_MEANS[weight_class] = _compute_wc_means(weight_class, session)

    means = _WC_MEANS.get(weight_class, {})
    n = features["n_fights"]
    w = alpha * (1 - n / MIN_FIGHTS_TRUST)

    rate_features = [
        "win_rate", "finish_rate", "ko_rate", "sub_rate",
        "slpm", "sapm", "sig_acc", "td_per_min", "td_acc",
        "sub_per_min", "ctrl_ratio",
    ]

    for key in rate_features:
        if key in means and not np.isnan(features.get(key, float("nan"))):
            features[key] = (1 - w) * features[key] + w * means[key]
        elif key in means:
            features[key] = means[key]  # full prior for NaN

    return features


def _compute_wc_means(weight_class: str, session: Session) -> dict[str, float]:
    """Compute weight-class mean stats from all fighters' career averages."""
    sql = text("""
        SELECT
            SUM(s.sig_strikes_landed) * 1.0 / NULLIF(SUM(f.time_ended_sec)/60.0, 0) AS slpm,
            SUM(s.sig_strikes_attempted) * 1.0 / NULLIF(SUM(f.time_ended_sec)/60.0, 0) AS sapm,
            AVG(CASE WHEN f.winner_fighter_id = s.fighter_id THEN 1.0 ELSE 0.0 END) AS win_rate,
            AVG(CASE WHEN f.method IN ('KO','TKO','SUB') THEN 1.0 ELSE 0.0 END) AS finish_rate,
            AVG(CASE WHEN f.method IN ('KO','TKO') THEN 1.0 ELSE 0.0 END) AS ko_rate,
            AVG(CASE WHEN f.method = 'SUB'
                      AND f.winner_fighter_id = s.fighter_id THEN 1.0 ELSE 0.0 END) AS sub_rate,
            SUM(s.sig_strikes_landed) * 1.0 / NULLIF(SUM(s.sig_strikes_attempted), 0) AS sig_acc,
            SUM(s.takedowns_landed) * 1.0 / NULLIF(SUM(f.time_ended_sec)/60.0, 0) AS td_per_min,
            SUM(s.takedowns_landed) * 1.0 / NULLIF(SUM(s.takedowns_attempted), 0) AS td_acc,
            SUM(s.submission_attempts) * 1.0 / NULLIF(SUM(f.time_ended_sec)/60.0, 0) AS sub_per_min,
            SUM(s.control_time_sec) * 1.0 / NULLIF(SUM(f.time_ended_sec), 0) AS ctrl_ratio
        FROM fights f
        JOIN fight_stats_round s ON s.fight_id = f.fight_id AND s.round = 0
        WHERE f.weight_class = :wc AND f.method IS NOT NULL
    """)
    row = session.execute(sql, {"wc": weight_class}).fetchone()
    if row is None:
        return {}
    return {
        "slpm":        row[0] or 0.0,
        "sapm":        row[1] or 0.0,
        "win_rate":    row[2] or 0.0,
        "finish_rate": row[3] or 0.0,
        "ko_rate":     row[4] or 0.0,
        "sub_rate":    row[5] or 0.0,
        "sig_acc":     row[6] or 0.0,
        "td_per_min":  row[7] or 0.0,
        "td_acc":      row[8] or 0.0,
        "sub_per_min": row[9] or 0.0,
        "ctrl_ratio":  row[10] or 0.0,
    }


# ---------------------------------------------------------------------------
# Build full training feature matrix
# ---------------------------------------------------------------------------

def build_fight_feature_rows(session: Session, since_year: int = 2001) -> pd.DataFrame:
    """
    For every fight since `since_year`, compute:
      - fighter_A and fighter_B as-of features (strictly before fight date)
      - difference features (A - B)
      - label: 1 if A won, 0 if B won
      - fight metadata

    Corner randomization applied here: with 50% probability, swap A/B.
    Returns one row per fight (after randomization, NOT doubled).
    For training use build_symmetric_rows() which doubles with label flip.
    """
    sql = text("""
        SELECT fight_id, date, red_fighter_id, blue_fighter_id,
               winner_fighter_id, weight_class, is_title_bout, is_five_round,
               red_is_short_notice, blue_is_short_notice,
               red_missed_weight, blue_missed_weight
        FROM fights
        WHERE date >= :since
          AND method NOT IN ('NC', 'DQ', 'CANCELLED')
          AND method IS NOT NULL
          AND winner_fighter_id IS NOT NULL
        ORDER BY date ASC
    """)

    fights = session.execute(sql, {"since": date(since_year, 1, 1)}).fetchall()
    log.info("Building feature rows for %d fights since %d", len(fights), since_year)

    rng = np.random.default_rng(seed=42)
    rows = []

    for i, fight in enumerate(fights):
        fight_date = fight.date if isinstance(fight.date, date) else date.fromisoformat(str(fight.date))

        # Corner randomization: 50/50 assignment of who is "A"
        swap = rng.random() < 0.5
        if swap:
            a_id, b_id = fight.blue_fighter_id, fight.red_fighter_id
            a_short_notice = fight.blue_is_short_notice
            b_short_notice = fight.red_is_short_notice
            a_missed_weight = fight.blue_missed_weight
            b_missed_weight = fight.red_missed_weight
        else:
            a_id, b_id = fight.red_fighter_id, fight.blue_fighter_id
            a_short_notice = fight.red_is_short_notice
            b_short_notice = fight.blue_is_short_notice
            a_missed_weight = fight.red_missed_weight
            b_missed_weight = fight.blue_missed_weight

        label = 1 if fight.winner_fighter_id == a_id else 0

        a_feat = fighter_aso_stats(a_id, fight_date, session, fight.weight_class)
        b_feat = fighter_aso_stats(b_id, fight_date, session, fight.weight_class)

        # Age at fight date (requires fighter DOB)
        a_age = _fighter_age(a_id, fight_date, session)
        b_age = _fighter_age(b_id, fight_date, session)

        row = {
            "fight_id": fight.fight_id,
            "date": fight_date,
            "fighter_a_id": a_id,
            "fighter_b_id": b_id,
            "weight_class": fight.weight_class,
            "is_title_bout": int(bool(fight.is_title_bout)),
            "is_five_round": int(bool(fight.is_five_round)),
            "label": label,
            # Raw features (used for debugging / interpretability)
            "a_n_fights": a_feat["n_fights"],
            "b_n_fights": b_feat["n_fights"],
            # Contextual
            "a_short_notice": int(bool(a_short_notice)),
            "b_short_notice": int(bool(b_short_notice)),
            "a_missed_weight": int(bool(a_missed_weight)),
            "b_missed_weight": int(bool(b_missed_weight)),
            "a_days_since_last": a_feat["days_since_last_fight"],
            "b_days_since_last": b_feat["days_since_last_fight"],
            "a_age": a_age,
            "b_age": b_age,
        }

        # Difference features (A - B) for all rate stats
        diff_keys = [
            "win_rate", "finish_rate", "ko_rate", "sub_rate",
            "slpm", "sapm", "sig_acc", "td_per_min", "td_acc",
            "sub_per_min", "ctrl_ratio",
            "l3_win_rate", "l5_win_rate", "l3_finish_rate",
            "l3_kd", "l3_td_rate", "l3_slpm", "l5_slpm",
            "win_streak", "loss_streak", "fight_frequency_24m",
        ]

        for k in diff_keys:
            a_val = a_feat.get(k, np.nan)
            b_val = b_feat.get(k, np.nan)
            row[f"diff_{k}"] = (
                a_val - b_val
                if not (np.isnan(a_val) or np.isnan(b_val))
                else np.nan
            )

        if a_age is not None and b_age is not None:
            row["diff_age"] = a_age - b_age
        else:
            row["diff_age"] = np.nan

        rows.append(row)

        if (i + 1) % 500 == 0:
            log.info("  … %d / %d fights processed", i + 1, len(fights))

    df = pd.DataFrame(rows)
    log.info("Feature matrix shape: %s", df.shape)
    return df


def symmetrize_rows(base: pd.DataFrame) -> pd.DataFrame:
    """
    Double a feature DataFrame by swapping A/B and flipping labels.
    Call this AFTER attach_ratings() so rating columns are included in the swap.
    """
    if base.empty or "label" not in base.columns:
        return base

    diff_cols = [c for c in base.columns if c.startswith("diff_")]

    # Explicit swap pairs — add rating columns when present
    _swap_pairs = [
        ("fighter_a_id",    "fighter_b_id"),
        ("a_n_fights",      "b_n_fights"),
        ("a_short_notice",  "b_short_notice"),
        ("a_missed_weight", "b_missed_weight"),
        ("a_days_since_last", "b_days_since_last"),
        ("a_age",           "b_age"),
        ("elo_a",           "elo_b"),
        ("glicko_a",        "glicko_b"),
        ("glicko_rd_a",     "glicko_rd_b"),
    ]
    swap_pairs = [(a, b) for a, b in _swap_pairs if a in base.columns and b in base.columns]

    mirrored = base.copy()
    mirrored["label"] = 1 - mirrored["label"]
    for col in diff_cols:
        mirrored[col] = -mirrored[col]
    for a_col, b_col in swap_pairs:
        mirrored[a_col], mirrored[b_col] = base[b_col].copy(), base[a_col].copy()

    result = pd.concat([base, mirrored], ignore_index=True)
    result = result.sort_values("date").reset_index(drop=True)
    return result


def build_symmetric_rows(session: Session, since_year: int = 2001) -> pd.DataFrame:
    """
    Returns base feature rows (without ratings).
    NOTE: build_matrix.py calls build_fight_feature_rows → attach_ratings → symmetrize_rows
    instead of this function, so that ratings are computed on N rows not 2N rows.
    This function is kept for any callers that don't need ratings columns.
    """
    base = build_fight_feature_rows(session, since_year)
    if base.empty or "label" not in base.columns:
        log.warning("build_symmetric_rows: no fights found — returning empty DataFrame")
        return base
    return symmetrize_rows(base)


def _fighter_age(fighter_id: str, fight_date: date, session: Session) -> float | None:
    sql = text("SELECT dob FROM fighters WHERE canonical_fighter_id = :fid")
    row = session.execute(sql, {"fid": fighter_id}).fetchone()
    if row is None or row[0] is None:
        return None
    dob = row[0] if isinstance(row[0], date) else date.fromisoformat(str(row[0]))
    return (fight_date - dob).days / 365.25
