"""
Stage 8 — Training pipeline.

Implements:
  - Chronological walk-forward cross-validation (NEVER random splits)
  - LightGBM with binary log-loss objective
  - Isotonic calibration on a held-out temporal fold
  - Bootstrap ensemble (20 bags) for uncertainty estimates
  - Saves: model artifacts to models/ directory
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score

log = logging.getLogger(__name__)

MODEL_DIR = Path("models")
N_BOOTSTRAP = 20
CAL_YEARS = 1  # most recent N years held out from training, used only for isotonic calibration

# Half-life (years) for the time-decay sample weights. A fight from `halflife`
# years ago contributes half as much as today's. Determined empirically by a
# halflife sweep on stacked-OOF log-loss (2026-05-03): 6y was the optimum,
# beating no-decay by ~0.17pp. Halflives <4y hurt (starves the model of
# historical signal); halflives >10y are essentially no decay. The MMA meta
# evolves more slowly than the brief's a priori estimate of 3y.
TIME_DECAY_HALFLIFE_YEARS = 6.0

FEATURE_COLS = [
    # Difference features
    "diff_win_rate", "diff_finish_rate", "diff_ko_rate", "diff_sub_rate",
    "diff_slpm", "diff_sapm", "diff_sig_acc",
    "diff_td_per_min", "diff_td_acc", "diff_sub_per_min", "diff_ctrl_ratio",
    "diff_l3_win_rate", "diff_l5_win_rate", "diff_l3_finish_rate",
    "diff_l3_kd", "diff_l3_td_rate", "diff_l3_slpm", "diff_l5_slpm",
    "diff_win_streak", "diff_loss_streak", "diff_fight_frequency_24m",
    "diff_elo", "diff_glicko", "diff_age",
    # EWMA recent form (halflife 3 fights) — smooth alternative to L3/L5.
    # diff_ewma_kd_per_fight tested but had 0 gain in feature importance —
    # dropped. The other three are doing real work alongside l3/l5.
    "diff_ewma_win_rate", "diff_ewma_finish_rate", "diff_ewma_slpm",
    # Defensive + durability (Week 7) — fills the striker-vs-grappler blind
    # spot. td_def and sig_str_def measure how often opponents land on this
    # fighter; finish_loss_rate is a chin/grappling-defence proxy.
    # sig_abs_per_min is the proper "strikes absorbed" metric (the legacy
    # `sapm` is actually attempts; kept for back-compat but redundant).
    "diff_td_def", "diff_sig_str_def", "diff_sig_abs_per_min",
    "diff_ko_loss_rate", "diff_sub_loss_rate", "diff_finish_loss_rate",
    # Style-mismatch interactions — explicit cross-features so the gradient
    # on small samples isn't forced to discover the interaction unaided.
    "diff_finish_threat", "diff_keep_standing", "diff_wrestled_pressure",
    # Offence × opp-defence cross features — every offensive metric weighted
    # by the opponent's complementary defence. e.g. A's striking volume only
    # matters in proportion to (1 − B's sig-strike defence). These are the
    # features that let the model learn matchup-specific predictions rather
    # than averaging over a generic opponent.
    "diff_expected_strikes_landed", "diff_expected_sig_acc",
    "diff_expected_td_landed",
    "diff_expected_ko_threat", "diff_expected_sub_threat",
    "diff_expected_strikes_taken",
    # Physicals (Week 3) — UFC.com bio fills active-roster gaps
    "diff_reach_cm", "diff_height_cm",
    # Stance interaction (Loffing 2017): southpaw vs orthodox edge
    "a_southpaw_vs_b_orthodox", "a_orthodox_vs_b_southpaw", "both_southpaw",
    # Strength-of-schedule (Week 3) — quality-of-opposition adjusted form.
    # diff_sos_quality_wins = how much stronger were the opponents A beat
    # vs the opponents B beat (z-scored against league mean Elo). Captures
    # the difference between "5-fight win streak vs Top-10s" and "5-fight
    # win streak vs prelim journeymen". Largest single feature lever in
    # the original plan.
    "diff_sos_avg_opp_elo", "diff_sos_quality_wins", "diff_sos_quality_losses",
    # Weight-class-aware age decline. Empirical peaks per class were derived
    # from our 8.6k-fight data (NOT literature) — see data/age_curve_empirical.csv.
    # diff_post_peak = (a_age - peak[wc])+ - (b_age - peak[wc])+, capturing
    # asymmetric decline: only post-peak years count. Mathematically the
    # linear class-peak offset cancels out of diff_age, so the signal lives
    # in this ReLU-style transform.
    "diff_post_peak",
    # Weight class as categorical — lets LGBM split on class identity (HW
    # finishes more, women's classes have different age curves, etc.) and
    # discover interactions like age×class without us having to hand-engineer.
    "weight_class_clean",
    # NOTE: Explicit layoff diffs (diff_days_since_last, diff_log_days_since_last)
    # were tested and gave no measurable lift on top of diff_fight_frequency_24m
    # (3-seed mean delta ≈ +0.0008 LL — below nondeterminism noise floor).
    # Per-fighter contextual (not differences — asymmetric by design)
    "a_n_fights", "b_n_fights",
    "a_short_notice", "b_short_notice",
    "a_missed_weight", "b_missed_weight",
    "glicko_rd_a", "glicko_rd_b",
    # Fight metadata
    "is_title_bout", "is_five_round",
]

# Categorical columns — passed to LGBM as `categorical_feature=` so it learns
# native splits rather than treating them as ordinal floats.
CATEGORICAL_COLS = ["weight_class_clean"]


# Monotonic constraints — encode domain knowledge about which features are
# directionally tied to "fighter A wins". Regularizes the model away from
# spurious feature interactions that flip the sign on small data slices.
#
#   +1  : P(A wins) is non-decreasing in this feature
#   -1  : P(A wins) is non-increasing in this feature
#    0  : no constraint
#
# Only signs we are confident in are constrained; ambiguous features (e.g.
# finish_rate — a high-finish fighter could be a one-trick KO artist) get 0.
MONOTONE_BY_FEATURE: dict[str, int] = {
    # A's superior skills → A wins
    "diff_win_rate":               +1,
    "diff_slpm":                   +1,
    "diff_sapm":                   -1,   # A absorbs MORE = worse
    "diff_sig_acc":                +1,
    "diff_td_per_min":             +1,
    "diff_td_acc":                 +1,
    "diff_sub_per_min":            +1,
    "diff_ctrl_ratio":             +1,
    "diff_l3_win_rate":            +1,
    "diff_l5_win_rate":            +1,
    "diff_l3_kd":                  +1,
    "diff_l3_td_rate":             +1,
    "diff_l3_slpm":                +1,
    "diff_l5_slpm":                +1,
    "diff_win_streak":             +1,
    "diff_loss_streak":            -1,   # A on longer L-streak = worse
    "diff_elo":                    +1,
    "diff_glicko":                 +1,
    "diff_post_peak":              -1,   # A more years past peak = worse
    "diff_ewma_win_rate":          +1,   # recency-weighted form
    "diff_ewma_slpm":              +1,
    # Physicals — reach edge is a documented striking advantage; height ambiguous
    "diff_reach_cm":               +1,
    # Stance asymmetric flags — empirical southpaw edge vs orthodox
    "a_southpaw_vs_b_orthodox":    +1,
    "a_orthodox_vs_b_southpaw":    -1,
    # SOS — beating tougher opponents → stronger fighter
    "diff_sos_avg_opp_elo":        +1,
    "diff_sos_quality_wins":       +1,
    "diff_sos_quality_losses":     -1,   # losing to weak opponents = bad sign
    # Defensive + durability — A defends better / gets finished less → A wins
    "diff_td_def":                 +1,   # A stuffs more TDs = good for A
    "diff_sig_str_def":            +1,   # A absorbs less = good for A
    "diff_sig_abs_per_min":        -1,   # A absorbs more strikes/min = bad
    "diff_ko_loss_rate":           -1,   # A gets KO'd more = bad for A
    "diff_sub_loss_rate":          -1,   # A gets subbed more = bad for A
    "diff_finish_loss_rate":       -1,   # A gets finished more = bad for A
    # Style-mismatch interactions
    "diff_finish_threat":          +1,   # A more likely to finish B
    "diff_keep_standing":          +1,   # A dictates range better
    "diff_wrestled_pressure":      -1,   # A faces more wrestling threat = bad for A
    # Offence × opp-defence cross features
    "diff_expected_strikes_landed": +1,  # A lands more strikes against B's defence
    "diff_expected_sig_acc":        +1,  # A is more accurate against B's defence
    "diff_expected_td_landed":      +1,  # A's TD/min cuts through B's TDD better
    "diff_expected_ko_threat":      +1,  # A's KO rate × B's chin
    "diff_expected_sub_threat":     +1,  # A's sub rate × B's sub vulnerability
    "diff_expected_strikes_taken":  -1,  # A absorbs more from B than B from A
    # Per-side context flags
    "a_short_notice":              -1,
    "b_short_notice":              +1,
    "a_missed_weight":             -1,
    "b_missed_weight":             +1,
    # Ambiguous → leave at 0:
    #   diff_finish_rate, diff_ko_rate, diff_sub_rate (one-trick risk)
    #   diff_l3_finish_rate, diff_fight_frequency_24m (active ≠ sharp)
    #   diff_age (depends on weight class — captured via diff_post_peak)
    #   a/b_n_fights, glicko_rd_*, is_title_bout, is_five_round
    #   weight_class_clean (categorical — monotone N/A)
}


def monotone_constraints_for(feature_cols: list[str]) -> list[int]:
    """Return the LGBM `monotone_constraints` list aligned with `feature_cols`."""
    return [MONOTONE_BY_FEATURE.get(c, 0) for c in feature_cols]

# LGBM params tuned by Optuna (60-trial TPE, stacked-OOF-LL objective, W4).
# Best stacked LL: 0.6605 vs 0.6659 with prior manual params (-0.54pp).
# Pattern: shallower trees + slower learning + stronger L1 reg — the prior
# num_leaves=63 was overfitting on a 16k-row, ~50-feature dataset.
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 17,
    "max_depth": -1,
    "learning_rate": 0.0135,
    "n_estimators": 647,
    "min_child_samples": 41,
    "subsample": 0.644,
    "colsample_bytree": 0.959,
    "reg_alpha": 2.067,
    "reg_lambda": 1e-8,
    "random_state": 42,
    "verbose": -1,
}


@dataclass
class CVFold:
    train_end: int      # year (exclusive)
    val_start: int
    val_end: int        # year (exclusive)


def chronological_folds(start_year: int = 2001, eval_start: int = 2016) -> list[CVFold]:
    """
    Walk-forward folds: train on everything before val_start,
    validate on [val_start, val_start+1).
    Only generate folds for eval_start and later (earlier data is training-only).
    """
    current_year = date.today().year
    folds = []
    for val_start in range(eval_start, current_year):
        folds.append(CVFold(
            train_end=val_start,
            val_start=val_start,
            val_end=val_start + 1,
        ))
    return folds


def _split(df: pd.DataFrame, fold: CVFold):
    years = pd.to_datetime(df["date"]).dt.year
    train = df[years < fold.train_end]
    val = df[(years >= fold.val_start) & (years < fold.val_end)]
    return train, val


def _X_y(df: pd.DataFrame):
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available)
    if missing:
        log.debug("Missing feature cols (will be NaN): %s", missing)
    X = df[available].copy()
    # LGBM's internal dtype check rejects raw str columns even when they're
    # passed as categorical_feature; pyarrow/parquet rehydrates the column
    # as str rather than the in-memory category dtype, so we re-cast here.
    for cat in CATEGORICAL_COLS:
        if cat in X.columns and X[cat].dtype != "category":
            X[cat] = X[cat].astype("category")
    y = df["label"].values
    return X, y


def time_decay_weights(
    df: pd.DataFrame,
    halflife_years: float | None = None,
    reference: pd.Timestamp | None = None,
) -> np.ndarray:
    """Return weights w_i = 2^(-Δt_i / halflife), Δt = (reference - fight_date).
    LGBM normalizes weights internally so the choice of `reference` only
    affects absolute scale — relative weights between samples are identical
    for any reference date. Pass `halflife_years=None` to read the module
    constant at call time (so test sweeps can override it).
    """
    hl = halflife_years if halflife_years is not None else TIME_DECAY_HALFLIFE_YEARS
    if hl <= 0 or hl > 1e6:
        return np.ones(len(df))
    ref = reference or pd.Timestamp(date.today())
    dt_years = (ref - pd.to_datetime(df["date"])).dt.days.values / 365.25
    return np.power(0.5, np.maximum(dt_years, 0.0) / hl)


def train_lgbm(X_train, y_train, X_val=None, y_val=None,
               sample_weight=None) -> lgb.LGBMClassifier:
    params = {**LGBM_PARAMS,
              "monotone_constraints": monotone_constraints_for(list(X_train.columns))}
    model = lgb.LGBMClassifier(**params)
    eval_set = [(X_val, y_val)] if X_val is not None else None
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)] if eval_set else []
    cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]
    fit_kwargs = {"eval_set": eval_set, "callbacks": callbacks}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    if cat_cols:
        fit_kwargs["categorical_feature"] = cat_cols
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def fit_isotonic_calibration(model, X_cal, y_cal) -> IsotonicRegression:
    """Fit isotonic regression on raw model scores from a calibration fold."""
    raw_probs = model.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_probs, y_cal)
    return iso


def calibrated_predict(model, iso: IsotonicRegression, X) -> np.ndarray:
    raw = model.predict_proba(X)[:, 1]
    return iso.transform(raw)


def run_cv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run chronological walk-forward CV.
    Returns a DataFrame with out-of-fold predictions and metrics per fold.
    """
    folds = chronological_folds()
    oof_rows = []
    fold_metrics = []

    for fold in folds:
        train_df, val_df = _split(df, fold)
        if len(train_df) < 100 or len(val_df) < 10:
            continue

        # Hold out the most recent CAL_YEARS of training data as a calibration
        # slice the booster never sees. Fitting isotonic on data the booster
        # was trained on yields in-sample probabilities and inflates extreme
        # predictions, which is what the audit identified as the root cause
        # of mean CV log-loss > 1.0.
        train_years = pd.to_datetime(train_df["date"]).dt.year
        cal_mask = train_years >= (fold.train_end - CAL_YEARS)
        booster_df = train_df[~cal_mask]
        cal_df = train_df[cal_mask]
        if len(booster_df) < 100 or len(cal_df) < 50:
            log.warning(
                "Fold %d: insufficient booster/cal split (%d/%d) — skipping",
                fold.val_start, len(booster_df), len(cal_df),
            )
            continue

        X_booster, y_booster = _X_y(booster_df)
        X_cal, y_cal = _X_y(cal_df)
        X_val, y_val = _X_y(val_df)

        w_booster = time_decay_weights(booster_df)
        model = train_lgbm(X_booster, y_booster, X_val, y_val, sample_weight=w_booster)
        iso = fit_isotonic_calibration(model, X_cal, y_cal)
        raw_preds = model.predict_proba(X_val)[:, 1]
        preds = iso.transform(raw_preds)

        ll = log_loss(y_val, preds)
        auc = roc_auc_score(y_val, preds)
        fold_metrics.append(
            {"fold": fold.val_start, "log_loss": ll, "auc": auc, "n_val": len(val_df)}
        )
        log.info("Fold %d: log_loss=%.4f  auc=%.4f  n=%d", fold.val_start, ll, auc, len(val_df))

        val_with_preds = val_df.copy()
        val_with_preds["pred_prob"] = preds
        # Save raw (uncalibrated) booster output too — needed to (a) compare
        # per-fold vs stacked-OOF isotonic without leakage, and (b) fit the
        # production stacked isotonic on the entire OOF.
        val_with_preds["raw_pred_prob"] = raw_preds
        val_with_preds["fold"] = fold.val_start
        oof_rows.append(val_with_preds)

    oof_df = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame()
    metrics_df = pd.DataFrame(fold_metrics)

    if len(metrics_df):
        log.info(
            "Mean CV log_loss=%.4f  auc=%.4f",
            metrics_df["log_loss"].mean(),
            metrics_df["auc"].mean(),
        )

    return oof_df, metrics_df


def evaluate_stacked_calibration(oof_df: pd.DataFrame) -> pd.DataFrame:
    """Compare per-fold isotonic to leave-one-fold-out stacked isotonic.

    For each fold, fit a stacked isotonic on the OOF rows from the OTHER folds
    (no leakage), apply it to the current fold's raw predictions, and measure
    log-loss. Returns a per-fold comparison DataFrame with columns:
      fold, log_loss_per_fold, log_loss_stacked, delta, n_val

    A negative delta means stacked beats per-fold (lower log-loss).
    """
    if oof_df.empty or "raw_pred_prob" not in oof_df.columns:
        return pd.DataFrame()

    rows = []
    folds = sorted(oof_df["fold"].dropna().unique())
    for f in folds:
        is_f = oof_df["fold"] == f
        train_oof = oof_df[~is_f]
        if len(train_oof) < 100:
            continue
        stacked = IsotonicRegression(out_of_bounds="clip")
        stacked.fit(train_oof["raw_pred_prob"].values, train_oof["label"].values)

        fold_oof = oof_df[is_f]
        y = fold_oof["label"].values
        per_fold_pred = fold_oof["pred_prob"].values
        stacked_pred = stacked.transform(fold_oof["raw_pred_prob"].values)

        ll_pf = log_loss(y, np.clip(per_fold_pred, 1e-7, 1 - 1e-7))
        ll_st = log_loss(y, np.clip(stacked_pred, 1e-7, 1 - 1e-7))
        rows.append({
            "fold": int(f),
            "n_val": len(fold_oof),
            "log_loss_per_fold": ll_pf,
            "log_loss_stacked": ll_st,
            "delta": ll_st - ll_pf,
        })
    return pd.DataFrame(rows)


def fit_stacked_isotonic(oof_df: pd.DataFrame) -> IsotonicRegression:
    """Fit an isotonic calibrator on the union of all walk-forward OOF raw
    predictions. n_cal grows from one year (~1k) to the full eval window
    (~9.6k), giving the calibrator a much larger and temporally diverse fit.
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_df["raw_pred_prob"].values, oof_df["label"].values)
    log.info(
        "Fit stacked isotonic on %d OOF rows (vs ~%d for the old per-year slice)",
        len(oof_df), len(oof_df) // max(1, oof_df["fold"].nunique()),
    )
    return iso


def train_final_model(
    df: pd.DataFrame,
    stacked_iso: IsotonicRegression | None = None,
) -> tuple:
    """Train the final production model.

    If `stacked_iso` is provided, the booster is trained on ALL data (no cal
    holdout) and the stacked isotonic is used as the production calibrator —
    this gives the booster the most data and the calibrator the largest,
    most temporally diverse OOF sample.

    Falls back to the old behavior (booster on data older than CAL_YEARS, iso
    on the held-out year) when no stacked_iso is provided — useful if CV
    fails or for ablation studies.
    """
    if stacked_iso is not None:
        X_all, y_all = _X_y(df)
        w_all = time_decay_weights(df)
        log.info("Training final booster on ALL %d rows; using stacked OOF iso", len(df))
        model = train_lgbm(X_all, y_all, sample_weight=w_all)
        return model, stacked_iso

    cal_cutoff = pd.Timestamp(date.today()) - pd.DateOffset(years=CAL_YEARS)
    cal_mask = pd.to_datetime(df["date"]) >= cal_cutoff
    booster_df = df[~cal_mask]
    cal_df = df[cal_mask]

    X_b, y_b = _X_y(booster_df)
    X_cal, y_cal = _X_y(cal_df)
    w_b = time_decay_weights(booster_df)

    log.info(
        "Training final model on %d rows (held-out cal slice: %d rows, last %dy)",
        len(booster_df), len(cal_df), CAL_YEARS,
    )
    model = train_lgbm(X_b, y_b, sample_weight=w_b)
    iso = fit_isotonic_calibration(model, X_cal, y_cal)

    return model, iso


def train_bootstrap_ensemble(
    df: pd.DataFrame,
    n: int = N_BOOTSTRAP,
    stacked_iso: IsotonicRegression | None = None,
) -> list[tuple]:
    """Train N bagged models for uncertainty quantification.

    With `stacked_iso`, each bag is sampled with replacement from the FULL
    dataset and shares the stacked OOF isotonic — no calibration holdout
    needed because the iso was fit on data temporally outside each bag's
    expected influence (OOF by walk-forward construction).

    Without `stacked_iso`, falls back to the original holdout-slice behavior.
    """
    log.info("Training bootstrap ensemble (%d bags)…", n)
    ensemble = []
    rng = np.random.default_rng(seed=0)

    if stacked_iso is not None:
        booster_pool = df.reset_index(drop=True)
        weights_pool = time_decay_weights(booster_pool)
        for i in range(n):
            idx = rng.choice(len(booster_pool), size=len(booster_pool), replace=True)
            bag_df = booster_pool.iloc[idx]
            X_bag, y_bag = _X_y(bag_df)
            w_bag = weights_pool[idx]

            params = {**LGBM_PARAMS,
                      "random_state": i,
                      "monotone_constraints": monotone_constraints_for(list(X_bag.columns))}
            model = lgb.LGBMClassifier(**params)
            cat_cols = [c for c in CATEGORICAL_COLS if c in X_bag.columns]
            fit_kwargs = {"sample_weight": w_bag, "callbacks": [lgb.log_evaluation(0)]}
            if cat_cols:
                fit_kwargs["categorical_feature"] = cat_cols
            model.fit(X_bag, y_bag, **fit_kwargs)

            ensemble.append((model, stacked_iso))
            if (i + 1) % 5 == 0:
                log.info("  … %d/%d bags done", i + 1, n)
        return ensemble

    cal_cutoff = pd.Timestamp(date.today()) - pd.DateOffset(years=CAL_YEARS)
    cal_mask = pd.to_datetime(df["date"]) >= cal_cutoff
    booster_pool = df[~cal_mask].reset_index(drop=True)
    weights_pool = time_decay_weights(booster_pool)
    X_cal, y_cal = _X_y(df[cal_mask])

    if len(booster_pool) < 100 or len(X_cal) < 50:
        log.warning(
            "Bootstrap ensemble: insufficient booster/cal split (%d/%d)",
            len(booster_pool), len(X_cal),
        )

    for i in range(n):
        idx = rng.choice(len(booster_pool), size=len(booster_pool), replace=True)
        bag_df = booster_pool.iloc[idx]
        X_bag, y_bag = _X_y(bag_df)
        w_bag = weights_pool[idx]

        params = {**LGBM_PARAMS,
                  "random_state": i,
                  "monotone_constraints": monotone_constraints_for(list(X_bag.columns))}
        model = lgb.LGBMClassifier(**params)
        cat_cols = [c for c in CATEGORICAL_COLS if c in X_bag.columns]
        fit_kwargs = {"sample_weight": w_bag, "callbacks": [lgb.log_evaluation(0)]}
        if cat_cols:
            fit_kwargs["categorical_feature"] = cat_cols
        model.fit(X_bag, y_bag, **fit_kwargs)

        iso = fit_isotonic_calibration(model, X_cal, y_cal)

        ensemble.append((model, iso))
        if (i + 1) % 5 == 0:
            log.info("  … %d/%d bags done", i + 1, n)

    return ensemble


def ensemble_predict(ensemble: list[tuple], X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (mean_prob, std_prob) across the bootstrap ensemble.
    mean_prob is the point estimate; std_prob quantifies uncertainty.
    """
    preds = np.stack([
        calibrated_predict(model, iso, X)
        for model, iso in ensemble
    ])  # shape: (n_bags, n_samples)
    return preds.mean(axis=0), preds.std(axis=0)


def save_artifacts(
    model,
    iso: IsotonicRegression,
    ensemble: list[tuple],
    feature_cols: list[str],
    metrics: pd.DataFrame,
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_DIR / "lgbm_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODEL_DIR / "isotonic.pkl", "wb") as f:
        pickle.dump(iso, f)
    with open(MODEL_DIR / "ensemble.pkl", "wb") as f:
        pickle.dump(ensemble, f)
    with open(MODEL_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    metrics.to_csv(MODEL_DIR / "cv_metrics.csv", index=False)
    log.info("Artifacts saved to %s/", MODEL_DIR)


def load_artifacts() -> tuple:
    with open(MODEL_DIR / "lgbm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "isotonic.pkl", "rb") as f:
        iso = pickle.load(f)
    with open(MODEL_DIR / "ensemble.pkl", "rb") as f:
        ensemble = pickle.load(f)
    with open(MODEL_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)
    return model, iso, ensemble, feature_cols


def fit_meta_blender(oof_df: pd.DataFrame) -> tuple | None:
    """Fit a logistic-regression meta-learner that blends the LGBM stacked-iso
    prediction with an Elo-only baseline. Returns (sklearn_model, blend_log_loss)
    or None if Elo data is missing.

    The Elo-only baseline `1 / (1 + 10^((elo_b - elo_a) / 400))` captures pure
    rating consensus and is uncorrelated with LGBM's noise interactions. Even
    a small blend weight on Elo can shave log-loss because the failure modes
    differ. Inputs to the meta are bounded [0,1] so no scaling needed.
    """
    if not all(c in oof_df.columns for c in ("pred_prob", "elo_a", "elo_b", "label")):
        return None
    elo_diff = oof_df["elo_a"].astype(float) - oof_df["elo_b"].astype(float)
    elo_only = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

    X = np.column_stack([
        np.clip(oof_df["pred_prob"].values, 1e-6, 1 - 1e-6),
        np.clip(elo_only.values, 1e-6, 1 - 1e-6),
    ])
    # Logit transform so the LR sees additive log-odds inputs
    X_logit = np.log(X / (1 - X))
    y = oof_df["label"].values.astype(int)

    from sklearn.linear_model import LogisticRegression
    meta = LogisticRegression(C=10.0, max_iter=500)
    meta.fit(X_logit, y)
    blend_pred = meta.predict_proba(X_logit)[:, 1]
    ll = log_loss(y, np.clip(blend_pred, 1e-7, 1 - 1e-7))
    log.info(
        "Meta blender fit on %d OOF rows. Coefs (lgbm_logit, elo_logit) = (%.3f, %.3f), "
        "intercept = %.3f. Blend OOF LL = %.4f",
        len(y), meta.coef_[0][0], meta.coef_[0][1], meta.intercept_[0], ll,
    )
    return (meta, float(ll))


def run_full_training(df: pd.DataFrame) -> None:
    """End-to-end: CV → stacked-iso comparison → final model → ensemble → save."""
    oof_df, metrics_df = run_cv(df)

    # Compare per-fold isotonic vs leave-one-fold-out stacked isotonic on the
    # exact same booster outputs. Then choose the production calibrator.
    stacked_iso = None
    if not oof_df.empty and "raw_pred_prob" in oof_df.columns:
        cal_cmp = evaluate_stacked_calibration(oof_df)
        if not cal_cmp.empty:
            cal_cmp.to_csv(MODEL_DIR / "calibration_comparison.csv", index=False)
            mean_pf = cal_cmp["log_loss_per_fold"].mean()
            mean_st = cal_cmp["log_loss_stacked"].mean()
            log.info(
                "Calibration comparison: per-fold LL=%.4f  stacked LL=%.4f  delta=%+.4f",
                mean_pf, mean_st, mean_st - mean_pf,
            )
        stacked_iso = fit_stacked_isotonic(oof_df)

    model, iso = train_final_model(df, stacked_iso=stacked_iso)
    ensemble = train_bootstrap_ensemble(df, stacked_iso=stacked_iso)

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    save_artifacts(model, iso, ensemble, available_features, metrics_df)

    # Re-calibrate OOF predictions through the stacked isotonic before saving.
    # Downstream consumers (conformal calibration, evaluation report) need
    # OOF probabilities that match the production model's calibrator.
    if not oof_df.empty and stacked_iso is not None and "raw_pred_prob" in oof_df.columns:
        oof_df["pred_prob_per_fold"] = oof_df["pred_prob"].values
        oof_df["pred_prob"] = stacked_iso.transform(oof_df["raw_pred_prob"].values)

    # Fit a meta-blender (LGBM-stacked + Elo-only) and apply to OOF if it
    # measurably improves log-loss. Saved as models/meta_blender.pkl;
    # predict.py picks it up if present.
    meta_result = fit_meta_blender(oof_df) if not oof_df.empty else None
    if meta_result:
        meta, blend_ll = meta_result
        pre_ll = log_loss(
            oof_df["label"].values,
            np.clip(oof_df["pred_prob"].values, 1e-7, 1 - 1e-7),
        )
        if blend_ll < pre_ll:
            with open(MODEL_DIR / "meta_blender.pkl", "wb") as f:
                pickle.dump(meta, f)
            log.info("Meta blender saved (LL %.4f → %.4f, Δ %.4f)",
                     pre_ll, blend_ll, blend_ll - pre_ll)
            # Update OOF pred_prob to the blended value so conformal calibration
            # is fit on the same predictions production will produce.
            elo_diff = oof_df["elo_a"].astype(float) - oof_df["elo_b"].astype(float)
            elo_only = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
            X = np.column_stack([
                np.clip(oof_df["pred_prob"].values, 1e-6, 1 - 1e-6),
                np.clip(elo_only.values, 1e-6, 1 - 1e-6),
            ])
            X_logit = np.log(X / (1 - X))
            oof_df["pred_prob_pre_blend"] = oof_df["pred_prob"].values
            oof_df["pred_prob"] = meta.predict_proba(X_logit)[:, 1]
        else:
            log.info("Meta blender did not improve LL (%.4f vs %.4f) — discarding",
                     blend_ll, pre_ll)

    oof_df.to_parquet(MODEL_DIR / "oof_predictions.parquet", index=False)
    log.info("Training complete.")
