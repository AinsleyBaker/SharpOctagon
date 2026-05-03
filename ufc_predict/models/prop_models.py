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

# Three-way chronological split: booster / early-stop / calibration.
# Previously this was 80/20 with the val slice doing double duty as early-stop
# AND isotonic-fit data — which made the calibrator see in-sample probabilities.
# The cal slice below is held out from BOTH training and early-stopping so the
# isotonic fit is unbiased.
_TRAIN_END   = 0.70
_EARLYSTOP_END = 0.85   # → cal slice = last 15% of chronological rows


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


_DEC_KEYWORDS = ("DEC", "DRAW", "SPLIT", "MAJORITY", "UNANIM", "POINTS")


def _round_class(round_ended, method: str) -> str:
    """
    Round of fight-ending FINISH only.  Returns 'DEC' for decisions/draws.

    Crucially: checks the METHOD string directly, not just whether round_ended
    is None.  Some data sources record round_ended=3 for decisions (the last
    round), which would otherwise mis-label those fights as R3 finishes and
    inflate the round model's R3 probability.
    """
    m = str(method).upper().strip()
    if any(kw in m for kw in _DEC_KEYWORDS):
        return "DEC"
    if round_ended is None or (isinstance(round_ended, float) and np.isnan(round_ended)):
        return "DEC"
    r = max(1, min(5, int(round_ended)))
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
        lambda r: _round_class(r["round_ended"], r["method"]), axis=1
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

    # Sort chronologically for the three-way split
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    s_train = int(n * _TRAIN_END)
    s_es    = int(n * _EARLYSTOP_END)

    X_tr  = df[available].iloc[:s_train]
    X_es  = df[available].iloc[s_train:s_es]
    X_cal = df[available].iloc[s_es:]

    artifacts: dict = {}

    # -----------------------------------------------------------------------
    # Method model
    # -----------------------------------------------------------------------
    method_enc = {cls: i for i, cls in enumerate(METHOD_CLASSES)}
    y_method = df["method_class"].map(method_enc).fillna(2).astype(int)  # fallback → A_DEC
    y_tr_m  = y_method.iloc[:s_train].values
    y_es_m  = y_method.iloc[s_train:s_es].values
    y_cal_m = y_method.iloc[s_es:].values

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
    # Pass DataFrames (not .values) so categorical dtype survives. categorical_feature
    # tells LGBM which columns to treat as native categorical.
    cat_feat = [c for c in ("weight_class_clean",) if c in X_tr.columns]
    method_model.fit(
        X_tr, y_tr_m,
        eval_set=[(X_es, y_es_m)],
        categorical_feature=cat_feat or "auto",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )

    # Calibrate each class on the held-out cal slice (booster never saw it).
    cal_probs_m = method_model.predict_proba(X_cal)
    method_isos: list[IsotonicRegression] = []
    for cls_idx in range(len(METHOD_CLASSES)):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(cal_probs_m[:, cls_idx], (y_cal_m == cls_idx).astype(float))
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
    # Keep only genuine finishes: filter by both round_class AND method_class
    # so that any decision with a numeric round_ended cannot slip through.
    df_finish = (
        df[(df["round_class"] != "DEC") & (~df["method_class"].str.endswith("_DEC"))]
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

    nf = len(df_finish)
    s_train_f = int(nf * _TRAIN_END)
    s_es_f    = int(nf * _EARLYSTOP_END)
    X_tr_f  = df_finish[available].iloc[:s_train_f]
    X_es_f  = df_finish[available].iloc[s_train_f:s_es_f]
    X_cal_f = df_finish[available].iloc[s_es_f:]

    round_enc = {cls: i for i, cls in enumerate(ROUND_CLASSES)}
    y_round_f = df_finish["round_class"].map(round_enc).fillna(0).astype(int)
    y_tr_r  = y_round_f.iloc[:s_train_f].values
    y_es_r  = y_round_f.iloc[s_train_f:s_es_f].values
    y_cal_r = y_round_f.iloc[s_es_f:].values

    round_params = {**method_params, "num_class": len(ROUND_CLASSES)}
    round_model = lgb.LGBMClassifier(**round_params)
    cat_feat_r = [c for c in ("weight_class_clean",) if c in X_tr_f.columns]
    round_model.fit(
        X_tr_f, y_tr_r,
        eval_set=[(X_es_f, y_es_r)],
        categorical_feature=cat_feat_r or "auto",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )

    cal_probs_r = round_model.predict_proba(X_cal_f)
    round_isos: list[IsotonicRegression] = []
    for cls_idx in range(len(ROUND_CLASSES)):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(cal_probs_r[:, cls_idx], (y_cal_r == cls_idx).astype(float))
        round_isos.append(iso)

    artifacts["round_model"] = round_model
    artifacts["round_isos"] = round_isos
    artifacts["round_classes"] = ROUND_CLASSES
    artifacts["feature_cols"] = available
    # Version flag — consumers check this to know round probs are finish-conditional.
    # v1 (absent) = old model trained on all fights (decisions mixed in, do not use).
    # v2 = trained on finishes only; output × prob_finish = P(finish in Rx).
    artifacts["prop_schema_v"] = 2

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

    # Align categorical dtypes with training (LGBM predict_proba checks this)
    method_model = artifacts.get("method_model")
    if method_model is not None:
        try:
            booster_cats = method_model.booster_.pandas_categorical or []
            cat_names = [c for c in available if c in {"weight_class_clean"}]
            for name, cats in zip(cat_names, booster_cats):
                X_feat[name] = pd.Categorical(X_feat[name], categories=cats)
        except Exception:
            pass

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

            # The method model produces 6 probs that sum to 1, but A's three
            # don't generally sum to the main model's P(A wins). Rescale each
            # side so:
            #   prob_a_wins_ko_tko + prob_a_wins_sub + prob_a_wins_dec == prob_a
            #   prob_b_wins_ko_tko + prob_b_wins_sub + prob_b_wins_dec == 1 - prob_a
            # while preserving the method model's WITHIN-side method ratios.
            a_raw = (float(cal_m[idx["A_KO_TKO"]]) + float(cal_m[idx["A_SUB"]])
                     + float(cal_m[idx["A_DEC"]]))
            b_raw = (float(cal_m[idx["B_KO_TKO"]]) + float(cal_m[idx["B_SUB"]])
                     + float(cal_m[idx["B_DEC"]]))
            a_scale = (prob_a / a_raw) if a_raw > 1e-9 else 0.0
            b_scale = ((1.0 - prob_a) / b_raw) if b_raw > 1e-9 else 0.0

            prop["prob_a_wins_ko_tko"] = round(float(cal_m[idx["A_KO_TKO"]]) * a_scale, 4)
            prop["prob_a_wins_sub"]    = round(float(cal_m[idx["A_SUB"]])    * a_scale, 4)
            prop["prob_a_wins_dec"]    = round(float(cal_m[idx["A_DEC"]])    * a_scale, 4)
            prop["prob_b_wins_ko_tko"] = round(float(cal_m[idx["B_KO_TKO"]]) * b_scale, 4)
            prop["prob_b_wins_sub"]    = round(float(cal_m[idx["B_SUB"]])    * b_scale, 4)
            prop["prob_b_wins_dec"]    = round(float(cal_m[idx["B_DEC"]])    * b_scale, 4)

            prop["prob_finish"]  = round(
                prop["prob_a_wins_ko_tko"] + prop["prob_a_wins_sub"]
                + prop["prob_b_wins_ko_tko"] + prop["prob_b_wins_sub"], 4
            )
            prop["prob_decision"] = round(
                prop["prob_a_wins_dec"] + prop["prob_b_wins_dec"], 4
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
        # Round model v2 outputs P(round X | finish); we scale by prob_finish
        # to get absolute P(fight ends in round X by finish).
        # Old models (v1, no prop_schema_v key) mixed decisions into R3/R5 —
        # those are silently ignored and replaced with the empirical prior.
        prob_finish_val = float(prop.get("prob_finish", 1.0))
        schema_v = artifacts.get("prop_schema_v", 1)

        is_five_r = False
        try:
            v = row["is_five_round"]
            is_five_r = bool(int(v.iloc[0]) if hasattr(v, "iloc") else int(v))
        except Exception:
            pass

        if round_model and schema_v >= 2:
            raw_r = round_model.predict_proba(row)[0]
            cal_r = _apply_isos(raw_r, round_isos) if round_isos else raw_r
            prob_rounds: dict[str, float] = {}
            for j, cls in enumerate(ROUND_CLASSES):
                prob_rounds[cls] = round(float(cal_r[j]) * prob_finish_val, 4)
            prop["prob_rounds"] = prob_rounds
        else:
            # Empirical UFC finish-round prior (approximate historical distribution).
            # Scaled to sum to prob_finish so all downstream maths stays consistent.
            if is_five_r:
                cond = {"R1": 0.26, "R2": 0.20, "R3": 0.20, "R4": 0.18, "R5": 0.16}
            else:
                cond = {"R1": 0.38, "R2": 0.28, "R3": 0.34, "R4": 0.00, "R5": 0.00}
            prop["prob_rounds"] = {
                cls: round(cond.get(cls, 0) * prob_finish_val, 4)
                for cls in ROUND_CLASSES
            }

        results.append(prop)

    return results


def run_cv(feature_cols: list[str], db_url: str | None = None,
           start_year: int = 2018) -> pd.DataFrame:
    """Walk-forward CV for the prop model. For each year ≥ start_year, train
    on all earlier data and predict on that year. Returns a DataFrame with
    fight_id, date, true labels, and per-class probabilities for both method
    (6 classes) and round (5 classes, finish-conditional).

    The CV gives us OOF prop predictions across thousands of historical
    fights — the foundation for prop ROI backtesting and prop calibration
    evaluation. Without this, prop probabilities only exist for upcoming
    bouts and we have no way to validate them.
    """
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    df = _load_labeled_matrix(db_url)
    prop_cols = list(dict.fromkeys(feature_cols + PROP_EXTRA_COLS))
    available = [c for c in prop_cols if c in df.columns]
    if not available:
        raise ValueError("No matching feature columns found in the feature matrix")
    df = df.sort_values("date").reset_index(drop=True)

    method_enc = {cls: i for i, cls in enumerate(METHOD_CLASSES)}
    df["_method_idx"] = df["method_class"].map(method_enc).fillna(2).astype(int)
    df["_year"] = pd.to_datetime(df["date"]).dt.year

    method_params = {
        "objective": "multiclass", "num_class": len(METHOD_CLASSES),
        "num_leaves": 31, "learning_rate": 0.05, "n_estimators": 500,
        "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 0.1, "random_state": 42, "verbose": -1,
    }

    cat_feat = [c for c in ("weight_class_clean",) if c in df.columns]
    last_year = int(df["_year"].max())
    oof_rows: list[dict] = []

    def _apply_per_class_iso(raw: np.ndarray, isos: list) -> np.ndarray:
        """Apply per-class isotonic + renormalize so rows sum to 1."""
        cal = np.zeros_like(raw)
        for j, iso in enumerate(isos):
            cal[:, j] = iso.transform(raw[:, j])
        s = cal.sum(axis=1, keepdims=True).clip(min=1e-9)
        return cal / s

    for val_year in range(start_year, last_year + 1):
        train = df[df["_year"] < val_year]
        val   = df[df["_year"] == val_year]
        if len(train) < 200 or len(val) < 10:
            continue

        # Within-train chronological cal slice (last 15% of train) — used to
        # fit per-class isotonic. Booster trains on the remaining 85% so iso
        # sees out-of-sample probabilities, matching production behavior.
        train_sorted = train.sort_values("date").reset_index(drop=True)
        s_b = int(len(train_sorted) * 0.85)
        booster_df = train_sorted.iloc[:s_b]
        cal_df = train_sorted.iloc[s_b:]

        # --- Method model (6-class) with per-class iso calibration ---
        m = lgb.LGBMClassifier(**method_params)
        m.fit(booster_df[available], booster_df["_method_idx"].values,
              categorical_feature=cat_feat or "auto",
              callbacks=[lgb.log_evaluation(period=-1)])
        # Calibrate on cal slice
        cal_raw_m = m.predict_proba(cal_df[available])
        method_isos = []
        for j in range(len(METHOD_CLASSES)):
            iso_j = IsotonicRegression(out_of_bounds="clip")
            iso_j.fit(cal_raw_m[:, j], (cal_df["_method_idx"].values == j).astype(float))
            method_isos.append(iso_j)
        method_raw = m.predict_proba(val[available])
        method_probs = _apply_per_class_iso(method_raw, method_isos)

        # --- Round model (5-class, finish-conditional) with iso calibration ---
        finish_train = booster_df[~booster_df["method_class"].str.endswith("_DEC")]
        finish_cal   = cal_df[~cal_df["method_class"].str.endswith("_DEC")]
        round_probs = None
        if len(finish_train) >= 100 and len(finish_cal) >= 30:
            round_enc = {c: i for i, c in enumerate(ROUND_CLASSES)}
            y_r_train = finish_train["round_class"].map(round_enc).fillna(0).astype(int)
            y_r_cal   = finish_cal["round_class"].map(round_enc).fillna(0).astype(int)
            r_params = {**method_params, "num_class": len(ROUND_CLASSES)}
            r = lgb.LGBMClassifier(**r_params)
            r.fit(finish_train[available], y_r_train.values,
                  categorical_feature=cat_feat or "auto",
                  callbacks=[lgb.log_evaluation(period=-1)])
            cal_raw_r = r.predict_proba(finish_cal[available])
            round_isos = []
            for j in range(len(ROUND_CLASSES)):
                iso_j = IsotonicRegression(out_of_bounds="clip")
                iso_j.fit(cal_raw_r[:, j], (y_r_cal.values == j).astype(float))
                round_isos.append(iso_j)
            round_raw = r.predict_proba(val[available])
            round_probs = _apply_per_class_iso(round_raw, round_isos)

        for i, (_, row) in enumerate(val.iterrows()):
            rec = {
                "fight_id": row.get("fight_id"),
                "date": row.get("date"),
                "fold_year": val_year,
                "method_class_true": row.get("method_class"),
                "round_class_true": row.get("round_class"),
                "is_five_round": int(row.get("is_five_round", 0) or 0),
            }
            for j, cls in enumerate(METHOD_CLASSES):
                rec[f"prob_{cls.lower()}"] = float(method_probs[i, j])
            if round_probs is not None:
                for j, cls in enumerate(ROUND_CLASSES):
                    rec[f"prob_{cls}"] = float(round_probs[i, j])
            oof_rows.append(rec)

        log.info("prop CV fold %d: train=%d val=%d", val_year, len(train), len(val))

    out = pd.DataFrame(oof_rows)
    out_path = MODELS_DIR / "prop_oof.parquet"
    out.to_parquet(out_path, index=False)
    log.info("Prop OOF saved: %s  shape=%s", out_path, out.shape)
    return out


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
