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

# Empirical peak ages per weight class, derived from our 8.6k-fight data
# (data/age_curve_empirical.csv). Used to compute the post-peak decline
# feature. Note these are 2-7y EARLIER than the Tandfonline 2023 literature
# claims — the lit values are biological-peak; ours reflect win rate
# contaminated by UFC matchmaking selection (older fighters who are still
# competing face tougher opposition). For prediction we want the empirical
# value because the model is fitting our actual win-rate distribution.
EMPIRICAL_PEAK_AGE = {
    "heavyweight": 28,
    "light heavyweight": 24,
    "middleweight": 25,
    "welterweight": 25,
    "lightweight": 26,
    "featherweight": 26,
    "bantamweight": 24,
    "flyweight": 27,
    "women's strawweight": 24,
    "women's flyweight": 24,
    "women's bantamweight": 26,
}


def normalize_weight_class(s: str | None) -> str:
    """Normalize a raw weight class string to the canonical lowercase form
    used as the categorical feature value. 'UFC Lightweight Title Bout' →
    'lightweight'."""
    import re as _re
    s = str(s or "").lower()
    s = s.replace("ufc ", "").replace("title bout", "").replace(" bout", "")
    return _re.sub(r"\s+", " ", s).strip()


def post_peak_years(age: float | None, weight_class: str | None) -> float:
    """ReLU-style post-peak distance: 0 if at-or-below peak for the class,
    else (age - peak). NaN-safe — returns 0 when age or class is unknown."""
    if age is None or pd.isna(age):
        return 0.0
    wc = normalize_weight_class(weight_class)
    peak = EMPIRICAL_PEAK_AGE.get(wc)
    if peak is None:
        return 0.0
    return max(0.0, float(age) - peak)
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
    # LEFT JOIN to opponent stats — a small fraction of historical fights only
    # have one fighter recorded in fight_stats_round; we still want the self
    # row to count for n_fights / win_rate even when defensive stats are NaN.
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
            opp.takedowns_landed       AS opp_td_landed,
            opp.takedowns_attempted    AS opp_td_attempted,
            opp.sig_strikes_landed     AS opp_sig_landed,
            opp.sig_strikes_attempted  AS opp_sig_attempted,
            f.time_ended_sec,
            f.round_ended
        FROM fights f
        JOIN fight_stats_round s
          ON s.fight_id = f.fight_id AND s.fighter_id = :fid AND s.round = 0
        LEFT JOIN fight_stats_round opp
          ON opp.fight_id = f.fight_id AND opp.fighter_id != :fid AND opp.round = 0
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
        "opp_td_landed", "opp_td_attempted",
        "opp_sig_landed", "opp_sig_attempted",
        "time_ended_sec", "round_ended",
    ])

    df["date"] = pd.to_datetime(df["date"])
    df["won"] = (df["winner_fighter_id"] == fighter_id).astype(int)
    # Use substring matching — DB stores methods like "KO/TKO", "Submission", etc.
    # "KO" is also a substring of "TKO", so one contains() covers both.
    _m = df["method"].str.upper().fillna("")
    df["ko_tko"] = _m.str.contains("KO", na=False).astype(int)
    df["finish"] = (df["ko_tko"] | _m.str.contains("SUB", na=False)).astype(int)
    df["sub_win"] = (
        _m.str.contains("SUB", na=False) & (df["winner_fighter_id"] == fighter_id)
    ).astype(int)
    # Durability: did THIS fighter get finished?
    df["ko_loss"] = (df["ko_tko"] & (df["won"] == 0)).astype(int)
    df["sub_loss"] = (
        _m.str.contains("SUB", na=False) & (df["won"] == 0)
    ).astype(int)
    df["finish_loss"] = (df["ko_loss"] | df["sub_loss"]).astype(int)

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

    # Defensive rates — the missing piece for stylistic mismatches like
    # striker-vs-grappler. td_def captures e.g. Strickland's career-best
    # TDD that the offensive features alone could never see.
    opp_td_att_total = df["opp_td_attempted"].fillna(0).sum()
    opp_td_def = (
        1.0 - df["opp_td_landed"].fillna(0).sum() / opp_td_att_total
        if opp_td_att_total > 0 else float("nan")
    )
    opp_sig_att_total = df["opp_sig_attempted"].fillna(0).sum()
    sig_str_def = (
        1.0 - df["opp_sig_landed"].fillna(0).sum() / opp_sig_att_total
        if opp_sig_att_total > 0 else float("nan")
    )
    # Strikes truly absorbed per minute (opponent's LANDED on this fighter).
    # Not to be confused with the legacy `sapm` field which is actually the
    # fighter's own attempts per minute (= slpm / sig_acc).
    sig_abs_per_min = (
        df["opp_sig_landed"].fillna(0).sum() / total_min
        if total_min > 0 else float("nan")
    )

    # Durability — chin/grappling-defence proxy. Career rates of being
    # finished. never_finished is a hard 0/1 flag; rate features let the
    # model see "got KO'd 1/20 vs 5/20".
    ko_loss_rate = df["ko_loss"].mean() if n > 0 else float("nan")
    sub_loss_rate = df["sub_loss"].mean() if n > 0 else float("nan")
    finish_loss_rate = df["finish_loss"].mean() if n > 0 else float("nan")
    never_finished = int(df["finish_loss"].sum() == 0) if n > 0 else 0

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

    # --- EWMA recent form (halflife = 3 fights) ---
    # Smooth decay weights recent fights more without dropping the 4th-most-recent
    # like a fixed L3 window. Halflife 3 was chosen to roughly match L3 in
    # effective sample size while preserving older-fight signal.
    df_per_fight_slpm = df["sig_landed"] / df["total_time_min"]
    ewma_win_rate = _ewma(df["won"].values, halflife=3.0)
    ewma_finish_rate = _ewma(df["finish"].values, halflife=3.0)
    ewma_slpm = _ewma(df_per_fight_slpm.values, halflife=3.0)
    ewma_kd_per_fight = _ewma(df["knockdowns"].values, halflife=3.0)

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
        # EWMA form — exponentially decayed last-N (halflife 3 fights)
        "ewma_win_rate": ewma_win_rate,
        "ewma_finish_rate": ewma_finish_rate,
        "ewma_slpm": ewma_slpm,
        "ewma_kd_per_fight": ewma_kd_per_fight,
        # Defensive + durability (added to capture striker-vs-grappler
        # asymmetries the offence-only feature set was blind to)
        "td_def": opp_td_def,
        "sig_str_def": sig_str_def,
        "sig_abs_per_min": sig_abs_per_min,
        "ko_loss_rate": ko_loss_rate,
        "sub_loss_rate": sub_loss_rate,
        "finish_loss_rate": finish_loss_rate,
        "never_finished": never_finished,
    }

    # Apply Bayesian shrinkage for fighters with few fights
    if n < MIN_FIGHTS_TRUST and weight_class:
        features = _apply_shrinkage(features, weight_class, session)

    return features


def _safe_ratio(num: float, denom: float) -> float:
    return num / denom if denom > 0 else 0.0


def _ewma(values: np.ndarray, halflife: float = 3.0) -> float:
    """Recency-weighted mean of `values` (sorted oldest→newest, like the df).
    The most recent value gets weight 1.0; weight halves every `halflife`
    fights going back. NaN-tolerant — drops missing values from the mean.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    mask = ~np.isnan(arr)
    if not mask.any():
        return float("nan")
    arr = arr[mask]
    recency = (len(arr) - 1) - np.arange(len(arr))  # 0=newest, n-1=oldest
    weights = np.power(0.5, recency / halflife)
    return float((arr * weights).sum() / weights.sum())


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
        "ewma_win_rate": np.nan, "ewma_finish_rate": np.nan,
        "ewma_slpm": np.nan, "ewma_kd_per_fight": np.nan,
        "td_def": np.nan, "sig_str_def": np.nan, "sig_abs_per_min": np.nan,
        "ko_loss_rate": np.nan, "sub_loss_rate": np.nan,
        "finish_loss_rate": np.nan, "never_finished": 0,
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
               red_missed_weight, blue_missed_weight,
               closing_odds_red, closing_odds_blue
        FROM fights
        WHERE date >= :since
          AND method NOT IN ('NC', 'DQ', 'CANCELLED')
          AND method IS NOT NULL
          AND winner_fighter_id IS NOT NULL
        ORDER BY date ASC
    """)

    fights = session.execute(sql, {"since": date(since_year, 1, 1)}).fetchall()
    log.info("Building feature rows for %d fights since %d", len(fights), since_year)

    # Pre-load fighter physicals + stance once. Avoids N queries inside the loop
    # and gives us a stable dict keyed by canonical_fighter_id.
    fighter_meta_rows = session.execute(text(
        "SELECT canonical_fighter_id, stance, reach_cm, height_cm FROM fighters"
    )).fetchall()
    FIGHTER_META: dict[str, dict] = {
        r[0]: {"stance": r[1], "reach_cm": r[2], "height_cm": r[3]}
        for r in fighter_meta_rows
    }

    rng = np.random.default_rng(seed=42)
    rows = []

    for i, fight in enumerate(fights):
        fight_date = (
            fight.date if isinstance(fight.date, date) else date.fromisoformat(str(fight.date))
        )

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
            # Closing odds + corner tracking — used ONLY in evaluation (Vegas
            # benchmark, Kelly ROI). Never used as model features.
            "closing_odds_red":  fight.closing_odds_red,
            "closing_odds_blue": fight.closing_odds_blue,
            "a_is_red": int(not swap),
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
            # EWMA versions — recency-weighted (halflife 3 fights)
            "ewma_win_rate", "ewma_finish_rate", "ewma_slpm", "ewma_kd_per_fight",
            # Defensive + durability — added so the model can see e.g. high
            # TDD vs high TD output (striker-vs-grappler asymmetry) and
            # finish history (chin / submission defence).
            "td_def", "sig_str_def", "sig_abs_per_min",
            "ko_loss_rate", "sub_loss_rate", "finish_loss_rate",
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

        # Weight-class-aware age decline. Empirical peaks per class (NOT lit) —
        # see EMPIRICAL_PEAK_AGE. The linear class offset cancels in diff_age,
        # so the signal lives in this ReLU-style transform.
        row["weight_class_clean"] = normalize_weight_class(fight.weight_class)
        a_pp = post_peak_years(a_age, fight.weight_class)
        b_pp = post_peak_years(b_age, fight.weight_class)
        row["a_post_peak"] = a_pp
        row["b_post_peak"] = b_pp
        row["diff_post_peak"] = a_pp - b_pp

        # --- Physicals (reach, height) --------------------------------
        a_meta = FIGHTER_META.get(a_id) or {}
        b_meta = FIGHTER_META.get(b_id) or {}
        a_reach = a_meta.get("reach_cm")
        b_reach = b_meta.get("reach_cm")
        a_height = a_meta.get("height_cm")
        b_height = b_meta.get("height_cm")
        row["a_reach_cm"] = a_reach
        row["b_reach_cm"] = b_reach
        row["a_height_cm"] = a_height
        row["b_height_cm"] = b_height
        row["diff_reach_cm"] = (
            (a_reach - b_reach) if (a_reach is not None and b_reach is not None) else np.nan
        )
        row["diff_height_cm"] = (
            (a_height - b_height) if (a_height is not None and b_height is not None) else np.nan
        )

        # --- Stance interaction ---------------------------------------
        # Loffing 2017 + MMA replications: orthodox-vs-southpaw matchups have
        # a small but documented southpaw edge (~1-2pp). We encode the four
        # mismatch states so LGBM can pick up the asymmetric effect.
        a_stance = (a_meta.get("stance") or "").strip().lower()
        b_stance = (b_meta.get("stance") or "").strip().lower()
        # Treat "switch" as orthodox (the more common base for switch-stance)
        def _ortho(s): return s in {"orthodox", "switch", ""}
        def _south(s): return s == "southpaw"
        row["a_southpaw_vs_b_orthodox"] = int(_south(a_stance) and _ortho(b_stance))
        row["a_orthodox_vs_b_southpaw"] = int(_ortho(a_stance) and _south(b_stance))
        row["both_southpaw"]            = int(_south(a_stance) and _south(b_stance))

        # Absolute per-fighter finish rates — used by the prop model only.
        # Diff features cancel out when both fighters have similar styles;
        # absolute rates retain the signal that "both fighters are KO artists".
        for k in ("ko_rate", "sub_rate", "finish_rate", "sub_per_min", "td_per_min",
                  "slpm", "sapm", "sig_acc", "ctrl_ratio",
                  "td_def", "sig_str_def", "sig_abs_per_min",
                  "ko_loss_rate", "sub_loss_rate", "finish_loss_rate", "never_finished"):
            row[f"a_{k}"] = a_feat.get(k, np.nan)
            row[f"b_{k}"] = b_feat.get(k, np.nan)

        # Style-mismatch interaction features. Trees handle interactions
        # natively, but explicit cross-features give the gradient a stronger
        # signal on small samples — and they're the cleanest way to encode
        # "A is a striker, B can't take A down" → high A-by-KO probability.
        a_finish = a_feat.get("finish_rate")
        b_finish = b_feat.get("finish_rate")
        a_floss  = a_feat.get("finish_loss_rate")
        b_floss  = b_feat.get("finish_loss_rate")
        a_tddef  = a_feat.get("td_def")
        b_tddef  = b_feat.get("td_def")
        a_tdpm   = a_feat.get("td_per_min")
        b_tdpm   = b_feat.get("td_per_min")
        a_slpm_v = a_feat.get("slpm")
        b_slpm_v = b_feat.get("slpm")

        def _mul(x, y):
            if x is None or y is None or pd.isna(x) or pd.isna(y):
                return np.nan
            return float(x) * float(y)

        # Finish threat = my finish rate × opp's tendency to be finished
        a_finish_threat = _mul(a_finish, b_floss)
        b_finish_threat = _mul(b_finish, a_floss)
        row["a_finish_threat"] = a_finish_threat
        row["b_finish_threat"] = b_finish_threat
        row["diff_finish_threat"] = (
            (a_finish_threat - b_finish_threat)
            if not (pd.isna(a_finish_threat) or pd.isna(b_finish_threat))
            else np.nan
        )

        # Keep-it-standing edge = my striking volume × my TDD (so I land
        # strikes AND can stay on the feet). Higher value → I dictate the
        # range. Diff captures the asymmetric matchup.
        a_keep_standing = _mul(a_slpm_v, a_tddef)
        b_keep_standing = _mul(b_slpm_v, b_tddef)
        row["a_keep_standing"] = a_keep_standing
        row["b_keep_standing"] = b_keep_standing
        row["diff_keep_standing"] = (
            (a_keep_standing - b_keep_standing)
            if not (pd.isna(a_keep_standing) or pd.isna(b_keep_standing))
            else np.nan
        )

        # Wrestling pressure on me = opp's TD output gated by my TDD. If I
        # have high TDD, opp's high TD-per-min hurts me less. Diff measures
        # which fighter is more vulnerable to the other's wrestling.
        def _gated_td_pressure(opp_tdpm, my_tddef):
            if any(v is None or pd.isna(v) for v in (opp_tdpm, my_tddef)):
                return np.nan
            return float(opp_tdpm) * (1.0 - float(my_tddef))

        a_wrestled_pressure = _gated_td_pressure(b_tdpm, a_tddef)
        b_wrestled_pressure = _gated_td_pressure(a_tdpm, b_tddef)
        row["a_wrestled_pressure"] = a_wrestled_pressure
        row["b_wrestled_pressure"] = b_wrestled_pressure
        row["diff_wrestled_pressure"] = (
            (a_wrestled_pressure - b_wrestled_pressure)
            if not (pd.isna(a_wrestled_pressure) or pd.isna(b_wrestled_pressure))
            else np.nan
        )

        # ===== Offence × Opponent-Defence interactions ======================
        # Pair every offensive metric with the opponent's matching defensive
        # metric so the model sees "what A actually achieves against B" — not
        # "what A averages over a generic opponent". Trees can learn this
        # multiplicatively, but explicit cross features cut variance on small
        # samples and surface as ranked feature importance.
        a_sigdef = a_feat.get("sig_str_def")
        b_sigdef = b_feat.get("sig_str_def")
        a_kor    = a_feat.get("ko_rate")
        b_kor    = b_feat.get("ko_rate")
        a_subr   = a_feat.get("sub_rate")
        b_subr   = b_feat.get("sub_rate")
        a_kloss  = a_feat.get("ko_loss_rate")
        b_kloss  = b_feat.get("ko_loss_rate")
        a_sloss  = a_feat.get("sub_loss_rate")
        b_sloss  = b_feat.get("sub_loss_rate")
        a_sigacc = a_feat.get("sig_acc")
        b_sigacc = b_feat.get("sig_acc")

        def _gated(volume, opp_def_rate):
            """volume × (1 - opponent's defence rate). NaN-safe."""
            if any(v is None or pd.isna(v) for v in (volume, opp_def_rate)):
                return np.nan
            return float(volume) * (1.0 - float(opp_def_rate))

        # Effective sig-strike volume — A's per-min landed × B's miss rate.
        exp_a_strikes = _gated(a_slpm_v, b_sigdef)
        exp_b_strikes = _gated(b_slpm_v, a_sigdef)
        row["expected_a_strikes_landed"] = exp_a_strikes
        row["expected_b_strikes_landed"] = exp_b_strikes
        row["diff_expected_strikes_landed"] = (
            (exp_a_strikes - exp_b_strikes)
            if not (pd.isna(exp_a_strikes) or pd.isna(exp_b_strikes))
            else np.nan
        )

        # Effective sig-strike accuracy — A's accuracy gated by B's defence.
        # Captures "A is sniper-accurate but B is hard to hit".
        exp_a_sigacc = _gated(a_sigacc, b_sigdef)
        exp_b_sigacc = _gated(b_sigacc, a_sigdef)
        row["expected_a_sig_acc"] = exp_a_sigacc
        row["expected_b_sig_acc"] = exp_b_sigacc
        row["diff_expected_sig_acc"] = (
            (exp_a_sigacc - exp_b_sigacc)
            if not (pd.isna(exp_a_sigacc) or pd.isna(exp_b_sigacc))
            else np.nan
        )

        # Effective takedown volume — A's TD/min × (1 - B's TDD). The simple
        # `wrestled_pressure` above is the opposite-direction view (defence);
        # this is the offensive-direction view.
        exp_a_td = _gated(a_tdpm, b_tddef)
        exp_b_td = _gated(b_tdpm, a_tddef)
        row["expected_a_td_landed"] = exp_a_td
        row["expected_b_td_landed"] = exp_b_td
        row["diff_expected_td_landed"] = (
            (exp_a_td - exp_b_td)
            if not (pd.isna(exp_a_td) or pd.isna(exp_b_td))
            else np.nan
        )

        # Method-specific finish threats — A's KO/sub rate × B's KO/sub
        # vulnerability. Replaces `finish_threat` (which was finish-rate
        # blended) with finer-grained per-method signals the prop model
        # benefits from most.
        exp_a_ko = _mul(a_kor, b_kloss)
        exp_b_ko = _mul(b_kor, a_kloss)
        row["expected_a_ko_threat"] = exp_a_ko
        row["expected_b_ko_threat"] = exp_b_ko
        row["diff_expected_ko_threat"] = (
            (exp_a_ko - exp_b_ko)
            if not (pd.isna(exp_a_ko) or pd.isna(exp_b_ko))
            else np.nan
        )

        exp_a_sub = _mul(a_subr, b_sloss)
        exp_b_sub = _mul(b_subr, a_sloss)
        row["expected_a_sub_threat"] = exp_a_sub
        row["expected_b_sub_threat"] = exp_b_sub
        row["diff_expected_sub_threat"] = (
            (exp_a_sub - exp_b_sub)
            if not (pd.isna(exp_a_sub) or pd.isna(exp_b_sub))
            else np.nan
        )

        # Strikes A will take from B — opp's striking pace × my hit-ability.
        # A high value here is a chin/durability red flag for A.
        exp_a_taken = _gated(b_slpm_v, a_sigdef)
        exp_b_taken = _gated(a_slpm_v, b_sigdef)
        row["expected_a_strikes_taken"] = exp_a_taken
        row["expected_b_strikes_taken"] = exp_b_taken
        row["diff_expected_strikes_taken"] = (
            (exp_a_taken - exp_b_taken)
            if not (pd.isna(exp_a_taken) or pd.isna(exp_b_taken))
            else np.nan
        )

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
        # Absolute per-fighter rates (must swap with A/B, not negated like diffs)
        ("a_ko_rate",     "b_ko_rate"),
        ("a_sub_rate",    "b_sub_rate"),
        ("a_finish_rate", "b_finish_rate"),
        ("a_sub_per_min", "b_sub_per_min"),
        ("a_td_per_min",  "b_td_per_min"),
        ("a_slpm",        "b_slpm"),
        ("a_sapm",        "b_sapm"),
        ("a_sig_acc",     "b_sig_acc"),
        ("a_ctrl_ratio",  "b_ctrl_ratio"),
        # Defensive + durability (Week 7)
        ("a_td_def",          "b_td_def"),
        ("a_sig_str_def",     "b_sig_str_def"),
        ("a_sig_abs_per_min", "b_sig_abs_per_min"),
        ("a_ko_loss_rate",    "b_ko_loss_rate"),
        ("a_sub_loss_rate",   "b_sub_loss_rate"),
        ("a_finish_loss_rate","b_finish_loss_rate"),
        ("a_never_finished",  "b_never_finished"),
        # Style-mismatch interactions
        ("a_finish_threat",   "b_finish_threat"),
        ("a_keep_standing",   "b_keep_standing"),
        ("a_wrestled_pressure","b_wrestled_pressure"),
        # Offence × opp-defence cross features
        ("expected_a_strikes_landed", "expected_b_strikes_landed"),
        ("expected_a_sig_acc",        "expected_b_sig_acc"),
        ("expected_a_td_landed",      "expected_b_td_landed"),
        ("expected_a_ko_threat",      "expected_b_ko_threat"),
        ("expected_a_sub_threat",     "expected_b_sub_threat"),
        ("expected_a_strikes_taken",  "expected_b_strikes_taken"),
        # Post-peak (Week 2)
        ("a_post_peak", "b_post_peak"),
        # Physicals (week 3)
        ("a_reach_cm",  "b_reach_cm"),
        ("a_height_cm", "b_height_cm"),
        # Strength-of-schedule (Week 3, Phase 5) — last-5 opponent strength
        ("a_sos_avg_opp_elo",    "b_sos_avg_opp_elo"),
        ("a_sos_quality_wins",   "b_sos_quality_wins"),
        ("a_sos_quality_losses", "b_sos_quality_losses"),
        # Stance pairs are NOT a simple swap: a_southpaw_vs_b_orthodox in the
        # mirrored row should become a_orthodox_vs_b_southpaw (the opposite
        # asymmetric flag). both_southpaw is invariant under swap.
        ("a_southpaw_vs_b_orthodox", "a_orthodox_vs_b_southpaw"),
    ]
    swap_pairs = [(a, b) for a, b in _swap_pairs if a in base.columns and b in base.columns]

    mirrored = base.copy()
    mirrored["label"] = 1 - mirrored["label"]
    for col in diff_cols:
        mirrored[col] = -mirrored[col]
    for a_col, b_col in swap_pairs:
        mirrored[a_col], mirrored[b_col] = base[b_col].copy(), base[a_col].copy()
    # a_is_red flips when we mirror — the mirrored row's "A" is the other corner.
    if "a_is_red" in mirrored.columns:
        mirrored["a_is_red"] = 1 - mirrored["a_is_red"]

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
