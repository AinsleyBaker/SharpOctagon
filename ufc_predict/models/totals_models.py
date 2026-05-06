"""Totals quantile regression — sig strikes, takedowns, knockdowns.

For each of the seven totals targets, we fit five LightGBM quantile
regressors (q ∈ {0.1, 0.25, 0.5, 0.75, 0.9}) on the same chronological
70/15/15 split used by the prop models. At inference time the five quantile
predictions form an empirical CDF; ``prob_over(line, quantiles)`` linearly
interpolates this CDF to compute P(over X.5).

The targets are:
    total_sig_strikes_combined   — A_landed + B_landed (round=0)
    total_sig_strikes_a, _b      — per-fighter
    total_takedowns_combined
    total_takedowns_a, _b
    total_knockdowns_combined

Labels come from ``fight_stats_round`` aggregated at round=0 (the fight
total). They are NEVER used as features (audit C.1 enforces this).
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sqlalchemy import text

from ufc_predict.db.session import get_session_factory
from ufc_predict.models.prop_models import _EARLYSTOP_END, _TRAIN_END, PROP_EXTRA_COLS

log = logging.getLogger(__name__)

MODELS_DIR = Path("models")
ARTIFACTS_PATH = MODELS_DIR / "totals_models.pkl"
OOF_PATH = MODELS_DIR / "totals_oof.parquet"
FEATURE_MATRIX_PATH = Path("data/feature_matrix.parquet")

QUANTILES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)

# (target_key, sql_template) — the SQL fragment selects per-fighter
# round=0 stats; the loader pivots and sums for combined targets.
TOTAL_TARGETS: list[str] = [
    "total_sig_strikes_combined",
    "total_sig_strikes_a",
    "total_sig_strikes_b",
    "total_takedowns_combined",
    "total_takedowns_a",
    "total_takedowns_b",
    "total_knockdowns_combined",
]

# LGBM hyperparams — slightly slower learning than the win model since
# quantile loss is harder to early-stop cleanly.
_LGBM_TOTALS_PARAMS = {
    "objective": "quantile",
    "num_leaves": 31,
    "learning_rate": 0.04,
    "n_estimators": 600,
    "min_child_samples": 30,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbose": -1,
}


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def _load_totals_labels(db_url: str | None = None) -> pd.DataFrame:
    """Pull per-fight totals from ``fight_stats_round`` (round=0) and join
    with ``fights`` to get red/blue identity. Pivot per-fighter sums into
    ``_a`` / ``_b`` columns (matching the symmetrisation in the feature
    matrix is done by the caller through ``a_is_red``).
    """
    factory = get_session_factory(db_url)
    sql = text(
        """
        SELECT s.fight_id,
               s.fighter_id,
               f.red_fighter_id,
               s.sig_strikes_landed,
               s.takedowns_landed,
               s.knockdowns
        FROM fight_stats_round s
        JOIN fights f ON f.fight_id = s.fight_id
        WHERE s.round = 0
        """
    )
    with factory() as session:
        df = pd.read_sql(sql, session.bind)
    if df.empty:
        return pd.DataFrame()
    df["is_red"] = (df["fighter_id"] == df["red_fighter_id"]).astype(int)
    # Pivot to red/blue per fight, then sum for combined.
    red = df[df["is_red"] == 1].rename(
        columns={
            "sig_strikes_landed": "ssl_red",
            "takedowns_landed": "td_red",
            "knockdowns": "kd_red",
        }
    )[["fight_id", "ssl_red", "td_red", "kd_red"]]
    blue = df[df["is_red"] == 0].rename(
        columns={
            "sig_strikes_landed": "ssl_blue",
            "takedowns_landed": "td_blue",
            "knockdowns": "kd_blue",
        }
    )[["fight_id", "ssl_blue", "td_blue", "kd_blue"]]
    out = red.merge(blue, on="fight_id", how="inner")
    out["total_sig_strikes_combined"] = out["ssl_red"] + out["ssl_blue"]
    out["total_takedowns_combined"] = out["td_red"] + out["td_blue"]
    out["total_knockdowns_combined"] = out["kd_red"] + out["kd_blue"]
    # Per-fighter columns will be re-keyed to A/B after merging with the
    # feature matrix (which carries ``a_is_red``).
    return out


def _load_features_with_labels(db_url: str | None = None) -> pd.DataFrame:
    if not FEATURE_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"feature matrix not found at {FEATURE_MATRIX_PATH}; run build_matrix first"
        )
    fm = pd.read_parquet(FEATURE_MATRIX_PATH)
    labels = _load_totals_labels(db_url)
    if labels.empty:
        raise RuntimeError("no totals labels loaded — check fight_stats_round availability")
    df = fm.merge(labels, on="fight_id", how="inner")
    # Re-key red/blue per-fighter totals to A/B based on a_is_red so they align
    # with the feature matrix's coordinate frame. After this, ``_a`` is the
    # per-fighter total for the row's fighter_a_id, regardless of corner.
    a_is_red = df["a_is_red"].astype(bool).values
    df["total_sig_strikes_a"] = np.where(a_is_red, df["ssl_red"], df["ssl_blue"])
    df["total_sig_strikes_b"] = np.where(a_is_red, df["ssl_blue"], df["ssl_red"])
    df["total_takedowns_a"] = np.where(a_is_red, df["td_red"], df["td_blue"])
    df["total_takedowns_b"] = np.where(a_is_red, df["td_blue"], df["td_red"])
    df = df.drop(columns=["ssl_red", "ssl_blue", "td_red", "td_blue", "kd_red", "kd_blue"])
    return df


# ---------------------------------------------------------------------------
# Quantile-monotonisation utilities
# ---------------------------------------------------------------------------

def _row_monotonise(matrix: np.ndarray) -> np.ndarray:
    """Force columns of ``matrix`` to be non-decreasing left-to-right per row.

    Pool-adjacent-violators isn't needed — we just walk left-to-right and
    take the running max. This preserves the median (q50) when the only
    violations are at the tails, which is the common case.
    """
    out = matrix.astype(float, copy=True)
    for j in range(1, out.shape[1]):
        out[:, j] = np.maximum(out[:, j], out[:, j - 1])
    return out


def _check_quantile_monotonicity(matrix: np.ndarray) -> float:
    """Fraction of rows that are already non-decreasing."""
    if matrix.size == 0:
        return 1.0
    diffs = np.diff(matrix, axis=1)
    ok = (diffs >= -1e-9).all(axis=1)
    return float(ok.mean())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class _SplitFrames:
    train: pd.DataFrame
    early_stop: pd.DataFrame
    cal: pd.DataFrame
    full: pd.DataFrame


def _chronological_split(df: pd.DataFrame) -> _SplitFrames:
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    s_train = int(n * _TRAIN_END)
    s_es = int(n * _EARLYSTOP_END)
    return _SplitFrames(
        train=df.iloc[:s_train],
        early_stop=df.iloc[s_train:s_es],
        cal=df.iloc[s_es:],
        full=df,
    )


def _pinball_loss(y: np.ndarray, q_pred: np.ndarray, alpha: float) -> float:
    err = y - q_pred
    return float(np.mean(np.maximum(alpha * err, (alpha - 1) * err)))


def _prepare_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    if "weight_class_clean" in X.columns and X["weight_class_clean"].dtype != "category":
        X["weight_class_clean"] = X["weight_class_clean"].astype("category")
    return X


def _train_one_target(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
) -> dict:
    splits = _chronological_split(df)
    X_tr = _prepare_features(splits.train, feature_cols)
    y_tr = splits.train[target].astype(float).values
    X_es = _prepare_features(splits.early_stop, feature_cols)
    y_es = splits.early_stop[target].astype(float).values
    X_cal = _prepare_features(splits.cal, feature_cols)
    y_cal = splits.cal[target].astype(float).values

    cat_feat = [c for c in ("weight_class_clean",) if c in X_tr.columns]

    models: dict[float, lgb.LGBMRegressor] = {}
    pinball: dict[str, float] = {}

    for alpha in QUANTILES:
        params = {**_LGBM_TOTALS_PARAMS, "alpha": alpha}
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_es, y_es)],
            categorical_feature=cat_feat or "auto",
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        models[alpha] = model
        cal_pred = model.predict(X_cal)
        pinball[f"q{int(alpha*100):02d}"] = _pinball_loss(y_cal, cal_pred, alpha)

    # Monotonicity report on the calibration slice (post-monotonisation).
    cal_matrix = np.column_stack([models[a].predict(X_cal) for a in QUANTILES])
    pre_ratio = _check_quantile_monotonicity(cal_matrix)
    cal_matrix = _row_monotonise(cal_matrix)
    post_ratio = _check_quantile_monotonicity(cal_matrix)

    log.info(
        "totals[%s]: pinball %s  monotone %.4f -> %.4f",
        target,
        {k: round(v, 3) for k, v in pinball.items()},
        pre_ratio,
        post_ratio,
    )
    return {
        "models": models,
        "pinball": pinball,
        "monotone_pre": pre_ratio,
        "monotone_post": post_ratio,
        "feature_cols": list(_prepare_features(df, feature_cols).columns),
    }


def train_totals_models(
    feature_cols: list[str],
    db_url: str | None = None,
) -> dict:
    """Train quantile models for every totals target. Returns an artifact
    dict suitable for ``save_totals_artifacts()``.
    """
    df = _load_features_with_labels(db_url)
    log.info("totals training: %d rows after label join", len(df))

    cols = list(dict.fromkeys(feature_cols + PROP_EXTRA_COLS))
    artifacts: dict = {
        "quantiles": list(QUANTILES),
        "targets": list(TOTAL_TARGETS),
        "feature_cols": [c for c in cols if c in df.columns],
        "by_target": {},
        "schema_v": 1,
    }
    for target in TOTAL_TARGETS:
        if target not in df.columns:
            log.warning("totals target %s missing from joined frame — skipping", target)
            continue
        artifacts["by_target"][target] = _train_one_target(df, cols, target)
    return artifacts


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_totals(X: pd.DataFrame, artifacts: dict) -> dict[str, dict]:
    """Predict the five quantiles + a mean estimate for every totals target.

    Returns a dict keyed by target name. Each value is itself a dict with
    keys ``q10, q25, q50, q75, q90, mean_estimate`` and arrays sized to len(X).
    """
    feature_cols = artifacts.get("feature_cols", [])
    by_target = artifacts.get("by_target", {})
    out: dict[str, dict] = {}
    if not by_target:
        return out
    X_feat = _prepare_features(X, feature_cols)

    for target, payload in by_target.items():
        models = payload["models"]
        # Align categorical levels (parquet rehydrates as str)
        m_first = next(iter(models.values()))
        try:
            booster_cats = m_first.booster_.pandas_categorical or []
            cat_names = [c for c in X_feat.columns if c in {"weight_class_clean"}]
            for name, cats in zip(cat_names, booster_cats):
                X_feat[name] = pd.Categorical(X_feat[name], categories=cats)
        except Exception:
            pass

        matrix = np.column_stack([models[a].predict(X_feat) for a in QUANTILES])
        matrix = _row_monotonise(matrix)
        out[target] = {
            f"q{int(a*100):02d}": matrix[:, i]
            for i, a in enumerate(QUANTILES)
        }
        # Mean estimate from the empirical CDF: trapezoidal integration of x dF.
        # With only 5 quantiles this is an approximation, but it's a better
        # single-number summary than the median for skewed distributions.
        out[target]["mean_estimate"] = matrix.mean(axis=1)
    return out


def prob_over(line: float, quantiles: dict) -> float:
    """Probability the realised total exceeds ``line``, given the predictive
    quantiles ``{q10, q25, q50, q75, q90}`` for one row.

    The five quantiles are treated as samples from the predictive CDF.
    Linearly interpolate to find F(line); P(over) = 1 - F(line). Clipped
    to [0.005, 0.995] to keep Kelly sizing finite.
    """
    if not quantiles:
        return 0.5
    qs = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
    vals = np.array([
        float(quantiles.get("q10", 0.0)),
        float(quantiles.get("q25", 0.0)),
        float(quantiles.get("q50", 0.0)),
        float(quantiles.get("q75", 0.0)),
        float(quantiles.get("q90", 0.0)),
    ])
    # Enforce monotonicity (defensive — values may come in unmonotonised).
    for j in range(1, len(vals)):
        vals[j] = max(vals[j], vals[j - 1])

    line = float(line)
    if line <= vals[0]:
        # Below the 10th percentile — extrapolate sub-linearly downward.
        slope = (vals[1] - vals[0]) if vals[1] > vals[0] else 1.0
        f = max(0.0, qs[0] - (vals[0] - line) / max(slope, 1e-6) * (qs[1] - qs[0]))
    elif line >= vals[-1]:
        slope = (vals[-1] - vals[-2]) if vals[-1] > vals[-2] else 1.0
        f = min(1.0, qs[-1] + (line - vals[-1]) / max(slope, 1e-6) * (qs[-1] - qs[-2]))
    else:
        # Interior: find the bracketing quantile pair and linearly interpolate.
        idx = int(np.searchsorted(vals, line, side="right") - 1)
        idx = max(0, min(idx, len(vals) - 2))
        x0, x1 = vals[idx], vals[idx + 1]
        y0, y1 = qs[idx], qs[idx + 1]
        if x1 - x0 < 1e-9:
            f = float(y1)
        else:
            f = float(y0 + (line - x0) / (x1 - x0) * (y1 - y0))

    p_over = 1.0 - f
    return float(np.clip(p_over, 0.005, 0.995))


# ---------------------------------------------------------------------------
# Walk-forward CV / OOF
# ---------------------------------------------------------------------------

def run_cv(
    feature_cols: list[str],
    db_url: str | None = None,
    start_year: int = 2018,
) -> pd.DataFrame:
    """Year-by-year walk-forward CV. For each year ≥ start_year, train on
    everything earlier and predict the year's quantiles. Saves to
    ``models/totals_oof.parquet``.
    """
    df = _load_features_with_labels(db_url)
    df = df.sort_values("date").reset_index(drop=True)
    df["_year"] = pd.to_datetime(df["date"]).dt.year
    cols = list(dict.fromkeys(feature_cols + PROP_EXTRA_COLS))

    last_year = int(df["_year"].max())
    rows: list[dict] = []
    for year in range(start_year, last_year + 1):
        train = df[df["_year"] < year]
        val = df[df["_year"] == year]
        if len(train) < 200 or len(val) < 10:
            continue
        # Reuse the chronological split inside the train slice for per-target
        # early stopping; calibration slice is unused at CV time.
        train_sorted = train.sort_values("date").reset_index(drop=True)
        s_b = int(len(train_sorted) * 0.85)
        booster = train_sorted.iloc[:s_b]
        es = train_sorted.iloc[s_b:]

        X_b = _prepare_features(booster, cols)
        X_es = _prepare_features(es, cols)
        X_v = _prepare_features(val, cols)
        cat_feat = [c for c in ("weight_class_clean",) if c in X_b.columns]

        per_target_preds: dict[str, np.ndarray] = {}
        for target in TOTAL_TARGETS:
            if target not in df.columns:
                continue
            y_b = booster[target].astype(float).values
            y_es = es[target].astype(float).values
            quantile_preds = []
            for alpha in QUANTILES:
                params = {**_LGBM_TOTALS_PARAMS, "alpha": alpha}
                m = lgb.LGBMRegressor(**params)
                m.fit(
                    X_b, y_b,
                    eval_set=[(X_es, y_es)],
                    categorical_feature=cat_feat or "auto",
                    callbacks=[
                        lgb.early_stopping(50, verbose=False),
                        lgb.log_evaluation(period=-1),
                    ],
                )
                quantile_preds.append(m.predict(X_v))
            matrix = np.column_stack(quantile_preds)
            matrix = _row_monotonise(matrix)
            per_target_preds[target] = matrix

        for i, (_, vrow) in enumerate(val.iterrows()):
            rec: dict = {
                "fight_id": vrow.get("fight_id"),
                "date": vrow.get("date"),
                "fold_year": year,
                "is_five_round": int(vrow.get("is_five_round", 0) or 0),
            }
            for target, matrix in per_target_preds.items():
                for j, alpha in enumerate(QUANTILES):
                    rec[f"{target}__q{int(alpha*100):02d}"] = float(matrix[i, j])
                rec[f"{target}__actual"] = float(vrow[target]) if target in vrow else np.nan
            rows.append(rec)
        log.info("totals CV fold %d: train=%d val=%d", year, len(train), len(val))

    out = pd.DataFrame(rows)
    if not out.empty:
        OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(OOF_PATH, index=False)
        log.info("totals OOF saved: %s  shape=%s", OOF_PATH, out.shape)
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_totals_artifacts(artifacts: dict, path: Path = ARTIFACTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(artifacts, f)
    # Also dump a sidecar JSON summarising pinball losses for easy inspection.
    summary = {
        "quantiles": list(QUANTILES),
        "targets": list(artifacts.get("by_target", {}).keys()),
        "pinball_by_target": {
            t: payload.get("pinball", {})
            for t, payload in artifacts.get("by_target", {}).items()
        },
        "monotone_post": {
            t: payload.get("monotone_post", None)
            for t, payload in artifacts.get("by_target", {}).items()
        },
    }
    (MODELS_DIR / "totals_models_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log.info("totals artifacts saved to %s", path)


def load_totals_artifacts(path: Path = ARTIFACTS_PATH) -> dict | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_training(db_url: str | None = None, run_cv_too: bool = True) -> None:
    feature_cols_path = MODELS_DIR / "feature_cols.json"
    if feature_cols_path.exists():
        with open(feature_cols_path) as f:
            feature_cols = json.load(f)
    else:
        from ufc_predict.models.train import FEATURE_COLS
        feature_cols = list(FEATURE_COLS)

    artifacts = train_totals_models(feature_cols, db_url)
    save_totals_artifacts(artifacts)

    if run_cv_too:
        try:
            run_cv(feature_cols, db_url)
        except Exception as exc:  # pragma: no cover
            log.warning("totals OOF CV failed (non-blocking): %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_training()
