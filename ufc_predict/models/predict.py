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
from ufc_predict.eval.bet_analysis import analyze_all_fights
from ufc_predict.features.aso_features import (
    _fighter_age,
    fighter_aso_stats,
    normalize_weight_class,
    post_peak_years,
)
from ufc_predict.models.prop_models import load_prop_artifacts, predict_props
from ufc_predict.models.train import (
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


def calibrate_conformal_mondrian(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    alpha: float = 0.10,
    min_group_size: int = 50,
) -> dict:
    """Per-group ("Mondrian") split-conformal calibration.

    Each group (e.g. weight class) gets its own halfwidth based on its local
    nonconformity distribution. Heavyweight (high variance, fewer fights)
    naturally gets a wider band; Lightweight (low variance, many fights) a
    tighter one. Same finite-sample coverage guarantee as global conformal,
    just stratified.

    Groups with fewer than `min_group_size` samples fall back to the global
    halfwidth — the per-group quantile estimate is too noisy below that.
    """
    residuals = np.abs(y_true - y_pred)
    n_total = len(residuals)
    level = min(np.ceil((n_total + 1) * (1 - alpha)) / n_total, 1.0)
    global_hw = float(np.quantile(residuals, level))

    per_group: dict[str, dict] = {}
    unique_groups = pd.unique(groups)
    for g in unique_groups:
        mask = (groups == g)
        n = int(mask.sum())
        if n < min_group_size:
            per_group[str(g)] = {"halfwidth": global_hw, "n": n, "fallback": True}
            continue
        glevel = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        gq = float(np.quantile(residuals[mask], glevel))
        per_group[str(g)] = {"halfwidth": gq, "n": n, "fallback": False}

    log.info(
        "Mondrian conformal: %d groups, global=%.4f, range=[%.4f, %.4f]",
        len(per_group), global_hw,
        min(v["halfwidth"] for v in per_group.values()),
        max(v["halfwidth"] for v in per_group.values()),
    )
    return {
        "alpha": alpha,
        "global_halfwidth": global_hw,
        "n_calibration": n_total,
        "per_group": per_group,
        "min_group_size": min_group_size,
    }


def conformal_interval(
    y_pred: np.ndarray,
    halfwidth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply pre-computed conformal halfwidth to get prediction intervals."""
    lo = np.clip(y_pred - halfwidth, 0.0, 1.0)
    hi = np.clip(y_pred + halfwidth, 0.0, 1.0)
    return lo, hi


def _bernoulli_sd(p: np.ndarray, eps: float = 1e-2) -> np.ndarray:
    """Local uncertainty proxy: sqrt(p(1-p)) — maximal at p=0.5, near zero at
    p=0 or 1. The eps floor avoids divide-by-zero for confident predictions
    that would otherwise get useless 0-width intervals."""
    return np.sqrt(np.clip(p * (1.0 - p), eps, None))


def calibrate_conformal_locally_weighted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.10,
) -> dict:
    """Locally-weighted (normalized) split-conformal calibration.

    Standardize residuals by the local Bernoulli SD: s_i = |y_i - p_i| / σ(p_i)
    where σ(p) = sqrt(p(1-p)). Compute the (1-α) quantile of s. At predict
    time, halfwidth(x) = q × σ(p(x)) — confident predictions get tighter bands,
    pick'em fights get wider ones, all under the same coverage guarantee.

    This addresses the Mondrian shortcoming: residuals are evenly distributed
    across weight classes, but they ARE concentrated near p=0.5. Locally-
    weighted conformal exploits that structure.
    """
    sigma = _bernoulli_sd(y_pred)
    scores = np.abs(y_true - y_pred) / sigma
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q = float(np.quantile(scores, level))

    # Diagnostic: estimate the average and range of resulting halfwidths
    sample_hws = q * sigma
    log.info(
        "Locally-weighted conformal: q=%.4f  halfwidths range=[%.4f, %.4f]  mean=%.4f",
        q, float(sample_hws.min()), float(sample_hws.max()), float(sample_hws.mean()),
    )
    return {
        "alpha": alpha,
        "scaled_quantile": q,
        "n_calibration": n,
        "kind": "locally_weighted",
    }


def locally_weighted_interval(
    y_pred: np.ndarray,
    quantiles: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply locally-weighted halfwidth: hw(x) = q × sqrt(p(x)(1-p(x)))."""
    q = float(quantiles.get("scaled_quantile", quantiles.get("conformal_halfwidth", 0.5)))
    sigma = _bernoulli_sd(y_pred)
    hw = q * sigma
    lo = np.clip(y_pred - hw, 0.0, 1.0)
    hi = np.clip(y_pred + hw, 0.0, 1.0)
    return lo, hi


def mondrian_interval(
    y_pred: np.ndarray,
    groups: np.ndarray,
    quantiles: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Mondrian halfwidths group-by-group; fall back to global for
    groups that weren't seen at calibration time."""
    per_group = quantiles.get("per_group", {})
    global_hw = quantiles.get("global_halfwidth", quantiles.get("conformal_halfwidth", 0.5))
    hws = np.array([
        per_group.get(str(g), {}).get("halfwidth", global_hw)
        for g in groups
    ])
    lo = np.clip(y_pred - hws, 0.0, 1.0)
    hi = np.clip(y_pred + hws, 0.0, 1.0)
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

        fa_obj = session.get(Fighter, a_id)
        fb_obj = session.get(Fighter, b_id)
        a_name = fa_obj.full_name if fa_obj else a_id
        b_name = fb_obj.full_name if fb_obj else b_id
        a_nat  = fa_obj.nationality if fa_obj else None
        b_nat  = fb_obj.nationality if fb_obj else None
        a_stance = fa_obj.stance if fa_obj else None
        b_stance = fb_obj.stance if fb_obj else None

        row = {
            "upcoming_bout_id": bout.upcoming_bout_id,
            "event_date": bout.event_date,
            "event_name": bout.event_name,
            "fighter_a_id": a_id,
            "fighter_b_id": b_id,
            "fighter_a_name": a_name,
            "fighter_b_name": b_name,
            "fighter_a_nationality": a_nat,
            "fighter_b_nationality": b_nat,
            "fighter_a_stance": a_stance,
            "fighter_b_stance": b_stance,
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
            "ewma_win_rate", "ewma_finish_rate", "ewma_slpm", "ewma_kd_per_fight",
            # Defensive + durability + style-mismatch (must mirror training)
            "td_def", "sig_str_def", "sig_abs_per_min",
            "ko_loss_rate", "sub_loss_rate", "finish_loss_rate",
        ]
        for k in diff_keys:
            av = a_feat.get(k, np.nan)
            bv = b_feat.get(k, np.nan)
            row[f"diff_{k}"] = av - bv if not (pd.isna(av) or pd.isna(bv)) else np.nan

        if a_age and b_age:
            row["diff_age"] = a_age - b_age
        else:
            row["diff_age"] = np.nan

        # Weight-class-aware age decline (must mirror training-time features)
        row["weight_class_clean"] = normalize_weight_class(bout.weight_class)
        a_pp = post_peak_years(a_age, bout.weight_class)
        b_pp = post_peak_years(b_age, bout.weight_class)
        row["diff_post_peak"] = a_pp - b_pp

        # --- Physicals + stance (Week 3) ---
        a_reach = fa_obj.reach_cm if fa_obj else None
        b_reach = fb_obj.reach_cm if fb_obj else None
        a_height = fa_obj.height_cm if fa_obj else None
        b_height = fb_obj.height_cm if fb_obj else None
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
        a_st = (a_stance or "").strip().lower()
        b_st = (b_stance or "").strip().lower()

        def _is_o(s):
            return s in {"orthodox", "switch", ""}

        def _is_s(s):
            return s == "southpaw"

        row["a_southpaw_vs_b_orthodox"] = int(_is_s(a_st) and _is_o(b_st))
        row["a_orthodox_vs_b_southpaw"] = int(_is_o(a_st) and _is_s(b_st))
        row["both_southpaw"]            = int(_is_s(a_st) and _is_s(b_st))

        row["a_win_streak"]  = int(a_feat.get("win_streak",  0) or 0)
        row["b_win_streak"]  = int(b_feat.get("win_streak",  0) or 0)
        row["a_loss_streak"] = int(a_feat.get("loss_streak", 0) or 0)
        row["b_loss_streak"] = int(b_feat.get("loss_streak", 0) or 0)
        row["a_l3_win_rate"] = a_feat.get("l3_win_rate")
        row["b_l3_win_rate"] = b_feat.get("l3_win_rate")

        # Absolute per-fighter rates (prop model features + dashboard display)
        for k in ("ko_rate", "sub_rate", "finish_rate", "sub_per_min", "td_per_min",
                  "slpm", "sapm", "sig_acc", "ctrl_ratio",
                  "td_def", "sig_str_def", "sig_abs_per_min",
                  "ko_loss_rate", "sub_loss_rate", "finish_loss_rate", "never_finished"):
            row[f"a_{k}"] = a_feat.get(k, np.nan)
            row[f"b_{k}"] = b_feat.get(k, np.nan)

        # Style-mismatch interactions (mirror build_fight_feature_rows)
        def _mul(x, y):
            if x is None or y is None or pd.isna(x) or pd.isna(y):
                return np.nan
            return float(x) * float(y)

        a_finish_threat = _mul(a_feat.get("finish_rate"), b_feat.get("finish_loss_rate"))
        b_finish_threat = _mul(b_feat.get("finish_rate"), a_feat.get("finish_loss_rate"))
        row["a_finish_threat"] = a_finish_threat
        row["b_finish_threat"] = b_finish_threat
        row["diff_finish_threat"] = (
            (a_finish_threat - b_finish_threat)
            if not (pd.isna(a_finish_threat) or pd.isna(b_finish_threat))
            else np.nan
        )

        a_keep_standing = _mul(a_feat.get("slpm"), a_feat.get("td_def"))
        b_keep_standing = _mul(b_feat.get("slpm"), b_feat.get("td_def"))
        row["a_keep_standing"] = a_keep_standing
        row["b_keep_standing"] = b_keep_standing
        row["diff_keep_standing"] = (
            (a_keep_standing - b_keep_standing)
            if not (pd.isna(a_keep_standing) or pd.isna(b_keep_standing))
            else np.nan
        )

        def _gated_td_pressure(opp_tdpm, my_tddef):
            if opp_tdpm is None or my_tddef is None:
                return np.nan
            if pd.isna(opp_tdpm) or pd.isna(my_tddef):
                return np.nan
            return float(opp_tdpm) * (1.0 - float(my_tddef))

        a_wrestled_pressure = _gated_td_pressure(b_feat.get("td_per_min"), a_feat.get("td_def"))
        b_wrestled_pressure = _gated_td_pressure(a_feat.get("td_per_min"), b_feat.get("td_def"))
        row["a_wrestled_pressure"] = a_wrestled_pressure
        row["b_wrestled_pressure"] = b_wrestled_pressure
        row["diff_wrestled_pressure"] = (
            (a_wrestled_pressure - b_wrestled_pressure)
            if not (pd.isna(a_wrestled_pressure) or pd.isna(b_wrestled_pressure))
            else np.nan
        )

        # Offence × opp-defence cross features (mirror build_fight_feature_rows).
        def _gated(volume, opp_def_rate):
            if any(v is None or pd.isna(v) for v in (volume, opp_def_rate)):
                return np.nan
            return float(volume) * (1.0 - float(opp_def_rate))

        def _diff(va, vb):
            if pd.isna(va) or pd.isna(vb):
                return np.nan
            return va - vb

        a_slpm_v = a_feat.get("slpm")
        b_slpm_v = b_feat.get("slpm")
        a_sigdef = a_feat.get("sig_str_def")
        b_sigdef = b_feat.get("sig_str_def")
        a_tdpm   = a_feat.get("td_per_min")
        b_tdpm   = b_feat.get("td_per_min")
        a_tddef  = a_feat.get("td_def")
        b_tddef  = b_feat.get("td_def")
        a_sigacc = a_feat.get("sig_acc")
        b_sigacc = b_feat.get("sig_acc")
        a_kor    = a_feat.get("ko_rate")
        b_kor    = b_feat.get("ko_rate")
        a_subr   = a_feat.get("sub_rate")
        b_subr   = b_feat.get("sub_rate")
        a_kloss  = a_feat.get("ko_loss_rate")
        b_kloss  = b_feat.get("ko_loss_rate")
        a_sloss  = a_feat.get("sub_loss_rate")
        b_sloss  = b_feat.get("sub_loss_rate")

        exp_a_strikes = _gated(a_slpm_v, b_sigdef)
        exp_b_strikes = _gated(b_slpm_v, a_sigdef)
        row["expected_a_strikes_landed"] = exp_a_strikes
        row["expected_b_strikes_landed"] = exp_b_strikes
        row["diff_expected_strikes_landed"] = _diff(exp_a_strikes, exp_b_strikes)

        exp_a_sigacc = _gated(a_sigacc, b_sigdef)
        exp_b_sigacc = _gated(b_sigacc, a_sigdef)
        row["expected_a_sig_acc"] = exp_a_sigacc
        row["expected_b_sig_acc"] = exp_b_sigacc
        row["diff_expected_sig_acc"] = _diff(exp_a_sigacc, exp_b_sigacc)

        exp_a_td = _gated(a_tdpm, b_tddef)
        exp_b_td = _gated(b_tdpm, a_tddef)
        row["expected_a_td_landed"] = exp_a_td
        row["expected_b_td_landed"] = exp_b_td
        row["diff_expected_td_landed"] = _diff(exp_a_td, exp_b_td)

        exp_a_ko = _mul(a_kor, b_kloss)
        exp_b_ko = _mul(b_kor, a_kloss)
        row["expected_a_ko_threat"] = exp_a_ko
        row["expected_b_ko_threat"] = exp_b_ko
        row["diff_expected_ko_threat"] = _diff(exp_a_ko, exp_b_ko)

        exp_a_sub = _mul(a_subr, b_sloss)
        exp_b_sub = _mul(b_subr, a_sloss)
        row["expected_a_sub_threat"] = exp_a_sub
        row["expected_b_sub_threat"] = exp_b_sub
        row["diff_expected_sub_threat"] = _diff(exp_a_sub, exp_b_sub)

        exp_a_taken = _gated(b_slpm_v, a_sigdef)
        exp_b_taken = _gated(a_slpm_v, b_sigdef)
        row["expected_a_strikes_taken"] = exp_a_taken
        row["expected_b_strikes_taken"] = exp_b_taken
        row["diff_expected_strikes_taken"] = _diff(exp_a_taken, exp_b_taken)

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

    # Attach Elo/Glicko from the snapshot persisted by build_matrix. RD is
    # inflated for days-since-last-fight so an inactive fighter's uncertainty
    # is reflected at predict time, mirroring the in-training inactivity rule.
    from ufc_predict.features.ratings import (
        load_latest_ratings,
        load_latest_sos,
        lookup_ratings,
        lookup_sos,
    )
    ratings_snapshot = load_latest_ratings()
    sos_snapshot = load_latest_sos()
    today = date.today()
    elo_a, elo_b, gl_a, gl_b, rd_a, rd_b = [], [], [], [], [], []
    sos_a_avg, sos_b_avg = [], []
    sos_a_qw, sos_b_qw = [], []
    sos_a_ql, sos_b_ql = [], []
    for _, row in upcoming_df.iterrows():
        ra = lookup_ratings(ratings_snapshot, row["fighter_a_id"], row.get("weight_class"), today)
        rb = lookup_ratings(ratings_snapshot, row["fighter_b_id"], row.get("weight_class"), today)
        elo_a.append(ra["elo"])
        elo_b.append(rb["elo"])
        gl_a.append(ra["glicko"])
        gl_b.append(rb["glicko"])
        rd_a.append(ra["glicko_rd"])
        rd_b.append(rb["glicko_rd"])
        sa = lookup_sos(sos_snapshot, row["fighter_a_id"])
        sb = lookup_sos(sos_snapshot, row["fighter_b_id"])
        sos_a_avg.append(sa["sos_avg_opp_elo"])
        sos_b_avg.append(sb["sos_avg_opp_elo"])
        sos_a_qw.append(sa["sos_quality_wins"])
        sos_b_qw.append(sb["sos_quality_wins"])
        sos_a_ql.append(sa["sos_quality_losses"])
        sos_b_ql.append(sb["sos_quality_losses"])
    upcoming_df["diff_elo"]    = np.array(elo_a) - np.array(elo_b)
    upcoming_df["diff_glicko"] = np.array(gl_a) - np.array(gl_b)
    upcoming_df["glicko_rd_a"] = rd_a
    upcoming_df["glicko_rd_b"] = rd_b
    upcoming_df["diff_sos_avg_opp_elo"]    = np.array(sos_a_avg) - np.array(sos_b_avg)
    upcoming_df["diff_sos_quality_wins"]   = np.array(sos_a_qw)  - np.array(sos_b_qw)
    upcoming_df["diff_sos_quality_losses"] = np.array(sos_a_ql)  - np.array(sos_b_ql)
    if not ratings_snapshot:
        log.warning(
            "No persisted fighter_ratings.json found — ratings defaulted to base. "
            "Run weekly_retrain (or `python -m ufc_predict.features.build_matrix`) to populate."
        )

    # Build feature matrix
    available = [c for c in feature_cols if c in upcoming_df.columns]
    X = upcoming_df[available].copy()

    # Restore categorical dtypes to match training. LightGBM's predict path
    # checks dtype equality for categorical features and errors on mismatch
    # ("train and valid dataset categorical_feature do not match"). Pull the
    # category levels from the booster's pandas_categorical metadata.
    try:
        booster_cats = model.booster_.pandas_categorical or []
        # booster_cats is a list aligned with the feature index of categoricals
        cat_feature_names = [c for c in available if c in {"weight_class_clean"}]
        for name, cats in zip(cat_feature_names, booster_cats):
            X[name] = pd.Categorical(X[name], categories=cats)
    except Exception as exc:
        log.warning("Could not align categorical dtypes (%s) — continuing", exc)

    # Ensemble prediction → mean ± std
    mean_prob, std_prob = ensemble_predict(ensemble, X)

    # Meta-blender (LGBM-stacked + Elo-only) — applied if the artifact exists
    # and was saved during training (only saved when it improved OOF LL).
    meta_path = Path("models/meta_blender.pkl")
    if meta_path.exists():
        try:
            with open(meta_path, "rb") as f:
                import pickle as _pkl
                meta = _pkl.load(f)
            elo_diff = upcoming_df["diff_elo"].astype(float).values
            elo_only = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
            X_meta = np.column_stack([
                np.clip(mean_prob, 1e-6, 1 - 1e-6),
                np.clip(elo_only, 1e-6, 1 - 1e-6),
            ])
            X_meta_logit = np.log(X_meta / (1 - X_meta))
            mean_prob = meta.predict_proba(X_meta_logit)[:, 1]
            log.info("Meta blender applied to %d predictions", len(mean_prob))
        except Exception as exc:
            log.warning("Meta blender load/apply failed: %s — using raw ensemble", exc)

    upcoming_df["prob_a_wins"] = mean_prob
    upcoming_df["prob_b_wins"] = 1 - mean_prob
    upcoming_df["uncertainty_std"] = std_prob

    # Conformal prediction intervals — preference order:
    #   1. Locally-weighted (per-prediction halfwidth via Bernoulli-SD scaling)
    #   2. Mondrian (per-weight-class halfwidths)
    #   3. Global split-conformal (single halfwidth for everyone)
    lw_path = Path("models/conformal_quantiles_locally_weighted.json")
    mondrian_path = Path("models/conformal_quantiles_mondrian.json")
    quantiles = load_conformal_quantiles()
    locally_weighted = None
    mondrian = None
    if lw_path.exists():
        try:
            locally_weighted = json.loads(lw_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            locally_weighted = None
    if mondrian_path.exists():
        try:
            mondrian = json.loads(mondrian_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            mondrian = None

    if locally_weighted:
        lo, hi = locally_weighted_interval(mean_prob, locally_weighted)
        coverage = int((1 - locally_weighted["alpha"]) * 100)
        upcoming_df[f"ci_{coverage}_lo"] = lo
        upcoming_df[f"ci_{coverage}_hi"] = hi
    elif mondrian and "weight_class_clean" in upcoming_df.columns:
        groups = upcoming_df["weight_class_clean"].astype(str).fillna("Unknown").values
        lo, hi = mondrian_interval(mean_prob, groups, mondrian)
        coverage = int((1 - mondrian["alpha"]) * 100)
        upcoming_df[f"ci_{coverage}_lo"] = lo
        upcoming_df[f"ci_{coverage}_hi"] = hi
    elif quantiles:
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
    _abs_stat_cols = [
        f"{side}_{stat}"
        for side in ("a", "b")
        for stat in ("ko_rate", "sub_rate", "finish_rate", "slpm", "sapm",
                     "sig_acc", "td_per_min", "sub_per_min", "ctrl_ratio",
                     "td_def", "sig_str_def", "sig_abs_per_min",
                     "ko_loss_rate", "sub_loss_rate", "finish_loss_rate",
                     "never_finished")
        if f"{side}_{stat}" in upcoming_df.columns
    ]
    output = upcoming_df[[
        "upcoming_bout_id", "event_date", "event_name",
        "fighter_a_name", "fighter_b_name", "weight_class",
        "fighter_a_nationality", "fighter_b_nationality",
        "fighter_a_stance", "fighter_b_stance",
        "is_title_bout", "is_five_round",
        "a_n_fights", "b_n_fights",
        "a_win_streak", "b_win_streak", "a_loss_streak", "b_loss_streak",
        "a_l3_win_rate", "b_l3_win_rate",
        *_abs_stat_cols,
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
            cache_age_hours,
            fetch_ufc_markets,
            load_markets,
            match_odds_to_predictions,
            save_markets,
        )
        # Prefer fresh data: live-fetch if cache is missing or older than the
        # threshold. Fall back to stale cache when live fetch fails (CI is
        # geo-blocked from sportsbet.com.au).
        CACHE_FRESH_HOURS = 36
        age = cache_age_hours()
        if age is not None and age < CACHE_FRESH_HOURS:
            sb_fights = load_markets()
            log.info("Using SportsBet cache (age %.1fh)", age)
        else:
            log.info("SportsBet cache stale or missing (age=%s) — live fetch…", age)
            sb_fights = fetch_ufc_markets()
            if sb_fights:
                save_markets(sb_fights)
            else:
                sb_fights = load_markets()
                if sb_fights:
                    log.warning("Live fetch failed; falling back to stale cache")

        if sb_fights:
            predictions_list = match_odds_to_predictions(sb_fights, predictions_list)
            log.info("SportsBet odds matched for %d/%d fights",
                     sum(1 for p in predictions_list if p.get("sportsbet_odds")),
                     len(predictions_list))
        else:
            log.warning(
                "No SportsBet markets available — run sportsbet_scraper locally to cache odds"
            )
    except Exception as exc:
        log.warning("SportsBet odds step failed (continuing without odds): %s", exc)

    # -- EV analysis -------------------------------------------------------
    predictions_list = analyze_all_fights(predictions_list)

    _save_predictions_list(predictions_list)

    # -- Snapshot for post-fight evaluation --------------------------------
    # Archive this prediction set so we can compare to actual results later.
    try:
        from ufc_predict.eval.track_predictions import snapshot_predictions
        snapshot_predictions(predictions_list)
    except Exception as exc:
        log.warning("Prediction snapshot failed (non-critical): %s", exc)

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
        print(
            df[["fighter_a_name", "fighter_b_name", "prob_a_wins", "prob_b_wins"]].to_string(
                index=False
            )
        )
