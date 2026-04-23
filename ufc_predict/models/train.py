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
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score

log = logging.getLogger(__name__)

MODEL_DIR = Path("models")
N_BOOTSTRAP = 20

FEATURE_COLS = [
    # Difference features
    "diff_win_rate", "diff_finish_rate", "diff_ko_rate", "diff_sub_rate",
    "diff_slpm", "diff_sapm", "diff_sig_acc",
    "diff_td_per_min", "diff_td_acc", "diff_sub_per_min", "diff_ctrl_ratio",
    "diff_l3_win_rate", "diff_l5_win_rate", "diff_l3_finish_rate",
    "diff_l3_kd", "diff_l3_td_rate", "diff_l3_slpm", "diff_l5_slpm",
    "diff_win_streak", "diff_loss_streak", "diff_fight_frequency_24m",
    "diff_elo", "diff_glicko", "diff_age",
    # Per-fighter contextual (not differences — asymmetric by design)
    "a_n_fights", "b_n_fights",
    "a_short_notice", "b_short_notice",
    "a_missed_weight", "b_missed_weight",
    "glicko_rd_a", "glicko_rd_b",
    # Fight metadata
    "is_title_bout", "is_five_round",
]

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "n_estimators": 400,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
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
    y = df["label"].values
    return X, y


def train_lgbm(X_train, y_train, X_val=None, y_val=None) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    eval_set = [(X_val, y_val)] if X_val is not None else None
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)] if eval_set else []
    model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)
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

        X_train, y_train = _X_y(train_df)
        X_val, y_val = _X_y(val_df)

        model = train_lgbm(X_train, y_train, X_val, y_val)

        # Use last 2 years of training data as calibration fold
        cal_years = pd.to_datetime(train_df["date"]).dt.year
        cal_mask = cal_years >= (fold.train_end - 2)
        X_cal, y_cal = _X_y(train_df[cal_mask])

        iso = fit_isotonic_calibration(model, X_cal, y_cal)
        preds = calibrated_predict(model, iso, X_val)

        ll = log_loss(y_val, preds)
        auc = roc_auc_score(y_val, preds)
        fold_metrics.append({"fold": fold.val_start, "log_loss": ll, "auc": auc, "n_val": len(val_df)})
        log.info("Fold %d: log_loss=%.4f  auc=%.4f  n=%d", fold.val_start, ll, auc, len(val_df))

        val_with_preds = val_df.copy()
        val_with_preds["pred_prob"] = preds
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


def train_final_model(df: pd.DataFrame) -> tuple:
    """
    Train the final production model on ALL data.
    Uses the most recent 2 years as calibration fold.
    Returns (model, iso_calibrator, feature_cols_used).
    """
    X, y = _X_y(df)

    # Calibration fold: last 2 years before today
    cal_cutoff = pd.Timestamp(date.today()) - pd.DateOffset(years=2)
    cal_mask = pd.to_datetime(df["date"]) >= cal_cutoff
    X_cal, y_cal = _X_y(df[cal_mask])

    log.info("Training final model on %d rows (calibration on %d)", len(df), cal_mask.sum())
    model = train_lgbm(X, y)
    iso = fit_isotonic_calibration(model, X_cal, y_cal)

    return model, iso


def train_bootstrap_ensemble(df: pd.DataFrame, n: int = N_BOOTSTRAP) -> list[tuple]:
    """
    Train N bagged models for uncertainty quantification.
    Each bag samples fights WITH replacement.
    Returns list of (model, iso) tuples.
    """
    log.info("Training bootstrap ensemble (%d bags)…", n)
    ensemble = []
    rng = np.random.default_rng(seed=0)

    all_indices = np.arange(len(df))
    cal_cutoff = pd.Timestamp(date.today()) - pd.DateOffset(years=2)
    cal_mask = pd.to_datetime(df["date"]) >= cal_cutoff

    for i in range(n):
        idx = rng.choice(len(df), size=len(df), replace=True)
        bag_df = df.iloc[idx]
        X_bag, y_bag = _X_y(bag_df)

        params = {**LGBM_PARAMS, "random_state": i}
        model = lgb.LGBMClassifier(**params)
        model.fit(X_bag, y_bag, callbacks=[lgb.log_evaluation(0)])

        # Calibrate on out-of-bag samples (unseen by this bag's model).
        # Fall back to global recent data if OOB set is too small.
        oob_mask = np.ones(len(df), dtype=bool)
        oob_mask[np.unique(idx)] = False
        oob_df = df.iloc[oob_mask]
        if len(oob_df) >= 50:
            X_cal, y_cal = _X_y(oob_df)
        else:
            X_cal, y_cal = _X_y(df[cal_mask])
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


def run_full_training(df: pd.DataFrame) -> None:
    """End-to-end: CV → final model → ensemble → save."""
    oof_df, metrics_df = run_cv(df)
    model, iso = train_final_model(df)
    ensemble = train_bootstrap_ensemble(df)

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    save_artifacts(model, iso, ensemble, available_features, metrics_df)

    oof_df.to_parquet(MODEL_DIR / "oof_predictions.parquet", index=False)
    log.info("Training complete.")
