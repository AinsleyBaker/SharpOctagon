"""Optuna hyperparameter sweep for the LightGBM booster.

Optimization target: leave-one-fold-out stacked-OOF log-loss. We deliberately
do NOT optimize per-fold log-loss because empirically (Week 3 audit) it's
dominated by per-fold-iso variance on small calibration slices and is
unreliable as an objective. Stacked LL aggregates over ~9.6k samples and is
the metric the production calibrator is fit on.

Run:
  python -m ufc_predict.models.tune --trials 150
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import optuna
import pandas as pd

from ufc_predict.models import train as T

log = logging.getLogger(__name__)
TUNE_PATH = Path("models/best_lgbm_params.json")


def _objective(trial: optuna.trial.Trial, df: pd.DataFrame) -> float:
    # Search space — tighter ranges around the current defaults so we don't
    # explode unfathomably. Each value is a sensible LGBM range.
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbose": -1,
        "random_state": 42,
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
    }

    # Patch the module's LGBM_PARAMS so run_cv picks them up via train_lgbm.
    saved = T.LGBM_PARAMS
    T.LGBM_PARAMS = params
    try:
        oof, _ = T.run_cv(df)
        if oof.empty or "raw_pred_prob" not in oof.columns:
            return float("inf")
        cmp = T.evaluate_stacked_calibration(oof)
        if cmp.empty:
            return float("inf")
        return float(cmp["log_loss_stacked"].mean())
    finally:
        T.LGBM_PARAMS = saved


def run(trials: int = 150, seed: int = 0) -> dict:
    df = pd.read_parquet("data/feature_matrix.parquet")
    if "weight_class_clean" in df.columns:
        df["weight_class_clean"] = df["weight_class_clean"].astype("category")

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=20)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(lambda t: _objective(t, df), n_trials=trials, show_progress_bar=False)

    log.info("Best stacked LL: %.4f", study.best_value)
    log.info("Best params: %s", study.best_params)

    out = {
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
    TUNE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TUNE_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Saved best params to %s", TUNE_PATH)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(trials=args.trials, seed=args.seed)
