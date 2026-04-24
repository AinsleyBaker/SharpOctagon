"""
Prop bet models — method of victory and round prediction.

Trains using the same feature matrix as the main win model, adding
fight outcome labels (method, round) from the DB.

Models trained:
  1. method_model — 6-class: A_KO_TKO / A_SUB / A_DEC / B_KO_TKO / B_SUB / B_DEC
  2. round_model  — 5-class: R1 / R2 / R3 / R4 / R5
                   (for decisions: class = max rounds of the fight)

Both use LightGBM + per-class isotonic calibration on a chronological holdout.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MODELS_DIR = Path("models")
FEATURE_MATRIX_PATH = Path("data/feature_matrix.parquet")

METHOD_CLASSES = ["A_KO_TKO", "A_SUB", "A_DEC", "B_KO_TKO", "B_SUB", "B_DEC"]
ROUND_CLASSES  = ["R1", "R2", "R3", "R4", "R5"]

# Absolute per-fighter rates added to the prop feature set.
# These capture "both fighters are KO artists" — a signal that cancels out in
# diff features but is critical for method/round prediction.
PROP_EXTRA_COLS = [
    "a_ko_rate",    "b_ko_rate",
    "a_sub_rate",   "b_sub_rate",
    "a_finish_rate","b_finish_rate",
    "a_sub_per_min","b_sub_per_min",
    "a_td_per_min", "b_td_per_min",
    # Strike pace — high combined slpm → more KO likelihood, longer fights have more volume
    "a_slpm",       "b_slpm",
    "a_sapm",       "b_sapm",
    "a_sig_acc",    "b_sig_acc",
    "a_ctrl_ratio", "b_ctrl_ratio",
]

_CAL_SPLIT = 0.80  # 80/20 chronological split for calibration


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------

def _method_class(label: int, method: str) -> str:
    """
    Convert (label, raw_method) → one of METHOD_CLASSES.
    label=1 means fighter A (post-symmetrization) won.
    """
    m = str(method).upper().strip()
    if "KO" in m or "TKO" in m:
        base = "KO_TKO"
    elif "SUB" in m:
        base = "SUB"
    else:
        base = "DEC"
    prefix = "A" if label == 1 else "B"
    return f"{prefix}_{base}"


def _round_class(round_ended, is_five_round: int) -> str:
    """
    Round the fight ended by FINISH (KO/TKO/Sub).
    Returns "DEC" for decisions/draws — these rows are excluded from round
    model training so the model learns P(round | finish), not P(round | any result).
    """
    if round_ended is None or (isinstance(round_ended, float) and np.isnan(round_ended)):
        return "DEC"
    r = int(round_ended)
    r = max(1, min(5, r))
    return f"R{r}"


def _load_labeled_matrix(db_url: str | None = None) -> pd.DataFrame:
    """
    Load the feature matrix and attach method + round labels from DB.
    The feature matrix already has fight_id and label columns.
    """
    from ufc_predict.db.session import get_session_factory
    from sqlalchemy import text

    if not FEATURE_MATRIX_PATH.exists():
        raise FileNotFoundError(f"Feature matrix not found at {FEATURE_MATRIX_PATH}")

    fm = pd.read_parquet(FEATURE_MATRIX_PATH)

    if "fight_id" not in fm.columns:
        raise ValueError("Feature matrix missing 'fight_id' column — rebuild the feature matrix")

    factory = get_session_factory(db_url)
    with factory() as session:
        sql = text("""
            SELECT fight_id, method, round_ended
            FROM fights
            WHERE method IS NOT NULL
        """)
        rows = session.execute(sql).fetchall()

    outcomes = pd.DataFrame(rows, columns=["fight_id", "method", "round_ended"])

    df = fm.merge(outcomes, on="fight_id", how="inner")

    # is_five_round comes from the feature matrix (it is a training feature).
    # If somehow absent, default to 0 (3-round fight assumption).
    if "is_five_round" not in df.columns:
        df["is_five_round"] = 0
    df["is_five_round"] = df["is_five_round"].fillna(0).astype(int)

    df["method_class"] = df.apply(
        lambda r: _method_class(int(r["label"]), r["method"]), axis=1
    )
    df["round_class"] = df.apply(
        lambda r: _round_class(r["round_ended"], int(r["is_five_round"])), axis=1
    )

    log.info(
        "Labeled matrix: %d rows  method dist: %s",
        len(df),
        df["method_class"].value_counts().to_dict(),
    )
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_prop_models(feature_cols: list[str], db_url: str | None = None) -> dict:
    """
    Train method and round models on the existing feature matrix.
    Returns a dict suitable for save_prop_artifacts().
    """
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    df = _load_labeled_matrix(db_url)
    # Extend with absolute per-fighter rates — these are in the feature matrix
    # after the aso_features.py change but were not in the win-model feature_cols.
    prop_cols = list(dict.fromkeys(feature_cols + PROP_EXTRA_COLS))
    available = [c for c in prop_cols if c in df.columns]
    if not available:
        raise ValueError("No matching feature columns found in the feature matrix")

    # Sort chronologically for the calibration split
    df = df.sort_values("date").reset_index(drop=True)
    split = int(len(df) * _CAL_SPLIT)

    X_tr = df[available].iloc[:split]
    X_val = df[available].iloc[split:]

    artifacts: dict = {}

    # -----------------------------------------------------------------------
    # Method model
    # -----------------------------------------------------------------------
    method_enc = {cls: i for i, cls in enumerate(METHOD_CLASSES)}
    y_method = df["method_class"].map(method_enc).fillna(2).astype(int)  # fallback → A_DEC
    y_tr_m = y_method.iloc[:split].values
    y_val_m = y_method.iloc[split:].values

    method_params = {
        "objective": "multiclass",
        "num_class": len(METHOD_CLASSES),
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbose": -1,
    }
    method_model = lgb.LGBMClassifier(**method_params)
    method_model.fit(
        X_tr, y_tr_m,
        eval_set=[(X_val.values, y_val_m)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )

    # Calibrate each class independently with isotonic regression
    val_probs_m = method_model.predict_proba(X_val)
    method_isos: list[IsotonicRegression] = []
    for cls_idx in range(len(METHOD_CLASSES)):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(val_probs_m[:, cls_idx], (y_val_m == cls_idx).astype(float))
        method_isos.append(iso)

    artifacts["method_model"] = method_model
    artifacts["method_isos"] = method_isos
    artifacts["method_classes"] = METHOD_CLASSES

    log.info("Method model trained (%d trees)", method_model.best_iteration_ or method_model.n_estimators)

    # -----------------------------------------------------------------------
    # Round model — trained on FINISHES ONLY (decisions excluded).
    #
    # The model learns P(round X | finish occurred).  At inference time we
    # multiply by prob_finish to get the absolute P(fight ends in round X by
    # finish).  This prevents decisions from inflating any single round bucket
    # (historically the last round of 3- and 5-round bouts).
    # -----------------------------------------------------------------------
    df_finish = (
        df[df["round_class"] != "DEC"]
        .sort_values("date")
        .reset_index(drop=True)
    )
    if len(df_finish) < 50:
        log.warning("Very few finish rows (%d) — skipping round model", len(df_finish))
        artifacts["round_model"] = None
        artifacts["round_isos"] = []
        artifacts["round_classes"] = ROUND_CLASSES
        artifacts["feature_cols"] = available
        return artifacts

    split_f = int(len(df_finish) * _CAL_SPLIT)
    X_tr_f  = df_finish[available].iloc[:split_f]
    X_val_f = df_finish[available].iloc[split_f:]

    round_enc = {cls: i for i, cls in enumerate(ROUND_CLASSES)}
    y_round_f = df_finish["round_class"].map(round_enc).fillna(0).astype(int)
    y_tr_r    = y_round_f.iloc[:split_f].values
    y_val_r   = y_round_f.iloc[split_f:].values

    round_params = {**method_params, "num_class": len(ROUND_CLASSES)}
    round_model = lgb.LGBMClassifier(**round_params)
    round_model.fit(
        X_tr_f, y_tr_r,
        eval_set=[(X_val_f.values, y_val_r)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )

    val_probs_r = round_model.predict_proba(X_val_f)
    round_isos: list[IsotonicRegression] = []
    for cls_idx in range(len(ROUND_CLASSES)):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(val_probs_r[:, cls_idx], (y_val_r == cls_idx).astype(float))
        round_isos.append(iso)

    artifacts["round_model"] = round_model
    artifacts["round_isos"] = round_isos
    artifacts["round_classes"] = ROUND_CLASSES
    artifacts["feature_cols"] = available

    log.info("Round model trained (%d trees)", round_model.best_iteration_ or round_model.n_estimators)
    return artifacts


def save_prop_artifacts(artifacts: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "prop_models.pkl"
    with open(path, "wb") as f:
        pickle.dump(artifacts, f)
    log.info("Prop artifacts saved to %s", path)


def load_prop_artifacts() -> dict | None:
    path = MODELS_DIR / "prop_models.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _apply_isos(raw_probs: np.ndarray, isos: list) -> np.ndarray:
    """Apply per-class isotonic calibration and renormalize."""
    cal = np.array([float(iso.predict([raw_probs[i]])[0]) for i, iso in enumerate(isos)])
    total = cal.sum()
    if total > 0:
        cal /= total
    return cal


def predict_props(
    X: pd.DataFrame,
    prob_a_wins: np.ndarray,
    artifacts: dict,
) -> list[dict]:
    """
    Predict method + round probabilities for upcoming bouts.

    X: feature DataFrame (same features as win model)
    prob_a_wins: main model's P(A wins) per row
    artifacts: loaded from load_prop_artifacts()

    Returns list of prop dicts, one per row in X.
    """
    feature_cols = artifacts.get("feature_cols", [])
    available = [c for c in feature_cols if c in X.columns]
    X_feat = X[available].copy() if available else X.copy()

    method_model = artifacts.get("method_model")
    method_isos  = artifacts.get("method_isos", [])
    round_model  = artifacts.get("round_model")
    round_isos   = artifacts.get("round_isos", [])

    results: list[dict] = []

    for i in range(len(X_feat)):
        row = X_feat.iloc[[i]]
        prob_a = float(prob_a_wins[i])
        prop: dict = {}

        # ---- method probabilities ----------------------------------------
        if method_model:
            raw_m = method_model.predict_proba(row)[0]
            cal_m = _apply_isos(raw_m, method_isos) if method_isos else raw_m

            idx = {cls: j for j, cls in enumerate(METHOD_CLASSES)}

            prop["prob_a_wins_ko_tko"] = round(float(cal_m[idx["A_KO_TKO"]]), 4)
            prop["prob_a_wins_sub"]    = round(float(cal_m[idx["A_SUB"]]),    4)
            prop["prob_a_wins_dec"]    = round(float(cal_m[idx["A_DEC"]]),    4)
            prop["prob_b_wins_ko_tko"] = round(float(cal_m[idx["B_KO_TKO"]]), 4)
            prop["prob_b_wins_sub"]    = round(float(cal_m[idx["B_SUB"]]),    4)
            prop["prob_b_wins_dec"]    = round(float(cal_m[idx["B_DEC"]]),    4)

            prop["prob_finish"]  = round(
                float(cal_m[idx["A_KO_TKO"]] + cal_m[idx["A_SUB"]] +
                      cal_m[idx["B_KO_TKO"]] + cal_m[idx["B_SUB"]]), 4
            )
            prop["prob_decision"] = round(
                float(cal_m[idx["A_DEC"]] + cal_m[idx["B_DEC"]]), 4
            )
        else:
            # Fallback heuristic: use absolute KO/sub rates when available,
            # falling back to diff features. The average of both fighters'
            # rates captures "two strikers → high KO probability" correctly.
            def _col(name: str) -> float:
                v = row.get(name)
                if v is None:
                    return float("nan")
                val = v.iloc[0] if hasattr(v, "iloc") else float(v)
                return float(val) if val == val else float("nan")  # nan check

            a_ko  = _col("a_ko_rate");  b_ko  = _col("b_ko_rate")
            a_sub = _col("a_sub_rate"); b_sub = _col("b_sub_rate")

            if a_ko == a_ko and b_ko == b_ko:
                base_ko  = (a_ko + b_ko) / 2
                base_sub = (a_sub + b_sub) / 2 if (a_sub == a_sub and b_sub == b_sub) else 0.12
            else:
                dko  = _col("diff_ko_rate")  if _col("diff_ko_rate")  == _col("diff_ko_rate")  else 0.0
                dsub = _col("diff_sub_rate") if _col("diff_sub_rate") == _col("diff_sub_rate") else 0.0
                base_ko  = 0.28 + np.clip(dko * 0.5, -0.12, 0.12)
                base_sub = 0.12 + np.clip(dsub * 0.5, -0.06, 0.06)

            base_dec = max(0.01, 1.0 - base_ko - base_sub)
            total = base_ko + base_sub + base_dec

            prop["prob_a_wins_ko_tko"] = round(base_ko  / total * prob_a, 4)
            prop["prob_a_wins_sub"]    = round(base_sub / total * prob_a, 4)
            prop["prob_a_wins_dec"]    = round(base_dec / total * prob_a, 4)
            prop["prob_b_wins_ko_tko"] = round(base_ko  / total * (1 - prob_a), 4)
            prop["prob_b_wins_sub"]    = round(base_sub / total * (1 - prob_a), 4)
            prop["prob_b_wins_dec"]    = round(base_dec / total * (1 - prob_a), 4)
            prop["prob_finish"]  = round(base_ko / total + base_sub / total, 4)
            prop["prob_decision"] = round(base_dec / total, 4)

        # ---- round probabilities -----------------------------------------
        # Round model outputs P(round X | finish).  Multiply by prob_finish
        # to get absolute P(fight ends in round X by finish).
        # Decisions are NOT in this distribution — use prob_decision separately.
        prob_finish_val = float(prop.get("prob_finish", 1.0))
        if round_model:
            raw_r = round_model.predict_proba(row)[0]
            cal_r = _apply_isos(raw_r, round_isos) if round_isos else raw_r

            prob_rounds: dict[str, float] = {}
            for j, cls in enumerate(ROUND_CLASSES):
                prob_rounds[cls] = round(float(cal_r[j]) * prob_finish_val, 4)
            prop["prob_rounds"] = prob_rounds
        else:
            # Flat conditional prior (P(Ri | finish) ≈ equal weight each round)
            # scaled to absolute probability
            base = prob_finish_val / 5
            prop["prob_rounds"] = {cls: round(base, 4) for cls in ROUND_CLASSES}

        results.append(prop)

    return results


def run_training(db_url: str | None = None) -> None:
    """CLI entry point: train prop models and save artifacts."""
    import json
    from ufc_predict.models.train import FEATURE_COLS

    feature_cols_path = MODELS_DIR / "feature_cols.json"
    if feature_cols_path.exists():
        with open(feature_cols_path) as f:
            feature_cols = json.load(f)
        log.info("Using %d feature cols from models/feature_cols.json", len(feature_cols))
    else:
        feature_cols = FEATURE_COLS
        log.warning("feature_cols.json not found, using default FEATURE_COLS")

    artifacts = train_prop_models(feature_cols, db_url)
    save_prop_artifacts(artifacts)
    log.info("Prop model training complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_training()
