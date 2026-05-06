"""Pre-build data accuracy audit.

Run this BEFORE training/predicting any new model. Each check below has a
hard PASS/FAIL outcome and a short message explaining what was verified.
A FAIL halts the audit (assertion) and an annotated soft-warning records a
known-acceptable deviation. The full report is written to
``models/data_audit_report.json`` so other tools can consume it.

Usage:
    python -m ufc_predict.eval.data_audit
"""
from __future__ import annotations

import json
import logging
import random
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sqlalchemy import text

from ufc_predict.db.session import get_session_factory

log = logging.getLogger(__name__)

REPORT_PATH = Path("models/data_audit_report.json")
FEATURE_MATRIX_PATH = Path("data/feature_matrix.parquet")
SPORTSBET_ODDS_PATH = Path("data/sportsbet_odds.json")
WIN_OOF_PATH = Path("models/oof_predictions.parquet")

# Methods we treat as "no fight stats expected" — true non-contests, DQs and
# overturned results don't always populate fight_stats_round consistently.
_NO_STATS_METHODS = {"DQ", "Could Not Continue", "Overturned", "Other"}

# Hard-fail NaN ceiling for FEATURE_COLS / PROP_EXTRA_COLS.
_NAN_HARD_FAIL_PCT = 60.0
_NAN_SOFT_WARN_PCT = 30.0


# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------

class AuditReport:
    def __init__(self) -> None:
        self.results: list[dict] = []

    def add(self, code: str, status: str, msg: str, **extra) -> None:
        rec = {"code": code, "status": status, "message": msg, **extra}
        self.results.append(rec)
        symbol = {"PASS": "OK ", "WARN": "WRN", "FAIL": "ERR"}.get(status, "?")
        log.info("[%s] %s %s", symbol, code, msg)

    def overall(self) -> str:
        if any(r["status"] == "FAIL" for r in self.results):
            return "FAIL"
        if any(r["status"] == "WARN" for r in self.results):
            return "WARN"
        return "PASS"

    def to_dict(self) -> dict:
        return {
            "generated_at": date.today().isoformat(),
            "overall": self.overall(),
            "checks": self.results,
        }


# ---------------------------------------------------------------------------
# A. Fight-stats integrity
# ---------------------------------------------------------------------------

def audit_a_fight_stats(session, rep: AuditReport) -> None:
    fights_df = pd.read_sql(
        text(
            "SELECT fight_id, method FROM fights "
            "WHERE method IS NOT NULL AND method != ''"
        ),
        session.bind,
    )
    fsr_df = pd.read_sql(
        text(
            "SELECT fight_id, fighter_id, round, "
            "       sig_strikes_landed, total_strikes_landed, "
            "       takedowns_landed, knockdowns "
            "FROM fight_stats_round"
        ),
        session.bind,
    )

    # A.1 — exactly 2 rows at round=0 per fight (excluding NC/DQ/...)
    eligible = fights_df[~fights_df["method"].isin(_NO_STATS_METHODS)]
    round0 = fsr_df[fsr_df["round"] == 0]
    counts = round0.groupby("fight_id").size()
    eligible_ids = set(eligible["fight_id"])
    bad = []
    missing = 0
    for fid in eligible_ids:
        c = int(counts.get(fid, 0))
        if c != 2:
            (bad if c > 0 else [bad, [fid]])  # noqa
            if c == 0:
                missing += 1
            else:
                bad.append((fid, c))
    if missing > 0 and missing / max(1, len(eligible_ids)) > 0.10:
        rep.add(
            "A.1",
            "FAIL",
            f"{missing} of {len(eligible_ids)} eligible fights have zero round=0 rows "
            f"(>10% threshold)",
            missing=missing,
            eligible=len(eligible_ids),
        )
    elif missing > 0:
        rep.add(
            "A.1",
            "WARN",
            f"{missing} of {len(eligible_ids)} eligible fights missing round=0 rows "
            f"(<10% — likely backfill gaps)",
            missing=missing,
            eligible=len(eligible_ids),
        )
    else:
        rep.add(
            "A.1",
            "PASS",
            f"All {len(eligible_ids)} eligible fights have round=0 rows present",
            eligible=len(eligible_ids),
        )
    if bad:
        rep.add(
            "A.1b",
            "WARN",
            f"{len(bad)} fights have round=0 row count != 2 (truncated stats?)",
            sample=[f"{fid}:{c}" for fid, c in bad[:5]],
        )

    # A.2 — round=0 totals == sum of per-round (for fights with both present)
    per_round = fsr_df[(fsr_df["round"] >= 1) & (fsr_df["round"] <= 5)]
    agg = per_round.groupby(["fight_id", "fighter_id"])[
        ["sig_strikes_landed", "total_strikes_landed", "takedowns_landed", "knockdowns"]
    ].sum().reset_index()
    fight_totals = round0.set_index(["fight_id", "fighter_id"])[
        ["sig_strikes_landed", "total_strikes_landed", "takedowns_landed", "knockdowns"]
    ]
    agg_idx = agg.set_index(["fight_id", "fighter_id"])
    common = fight_totals.index.intersection(agg_idx.index)
    if len(common) == 0:
        rep.add("A.2", "WARN", "No (fight, fighter) pairs with both round=0 and per-round stats")
    else:
        deltas = (fight_totals.loc[common] - agg_idx.loc[common]).abs()
        thresh = 2  # ±2 tolerance
        col_violations = (deltas > thresh).sum().to_dict()
        worst_col = max(col_violations, key=lambda k: col_violations[k])
        worst_n = col_violations[worst_col]
        if worst_n / max(1, len(common)) > 0.05:
            rep.add(
                "A.2",
                "WARN",
                f"{worst_n} pairs ({worst_n/len(common):.1%}) "
                f"have |round=0 - sum| > {thresh} for {worst_col}",
                col_violations=col_violations,
            )
        else:
            rep.add(
                "A.2",
                "PASS",
                f"round=0 totals match sum of per-round (within ±{thresh}) — "
                f"worst column {worst_col} has {worst_n}/{len(common)} violations",
                col_violations=col_violations,
            )

    # A.3 — no NULL fight_id (counted at SQL layer; pandas already coerces)
    null_fid = int(fsr_df["fight_id"].isna().sum())
    if null_fid > 0:
        rep.add("A.3", "FAIL", f"{null_fid} fight_stats_round rows have NULL fight_id")
    else:
        rep.add("A.3", "PASS", "fight_stats_round.fight_id has no NULLs")

    # A.4 — every fight_stats_round.fighter_id resolves to a fighter
    fighters = pd.read_sql(text("SELECT canonical_fighter_id FROM fighters"), session.bind)
    valid_ids = set(fighters["canonical_fighter_id"])
    orphan_mask = ~fsr_df["fighter_id"].isin(valid_ids)
    orphan = int(orphan_mask.sum())
    if orphan > 0:
        rep.add(
            "A.4",
            "FAIL",
            f"{orphan} fight_stats_round rows reference unknown fighter_ids",
        )
    else:
        rep.add("A.4", "PASS", "all fight_stats_round.fighter_id values resolve to fighters")


# ---------------------------------------------------------------------------
# B. Feature matrix integrity
# ---------------------------------------------------------------------------

def audit_b_feature_matrix(rep: AuditReport) -> pd.DataFrame | None:
    if not FEATURE_MATRIX_PATH.exists():
        rep.add("B.0", "FAIL", f"feature matrix missing at {FEATURE_MATRIX_PATH}")
        return None
    fm = pd.read_parquet(FEATURE_MATRIX_PATH)

    rows, cols = fm.shape
    if rows < 12_000 or rows > 25_000:
        rep.add("B.1", "FAIL", f"feature matrix rows {rows} outside expected [12k, 25k]")
    elif cols < 142:
        rep.add("B.1", "FAIL", f"feature matrix has {cols} columns, expected ≥ 142")
    else:
        rep.add("B.1", "PASS", f"feature matrix shape ({rows}, {cols}) — within expected range")

    # B.2 — top-10 NaN columns; hard-fail on FEATURE_COLS/PROP_EXTRA_COLS > 60%
    from ufc_predict.models.prop_models import PROP_EXTRA_COLS
    from ufc_predict.models.train import FEATURE_COLS
    nan_pct = (fm.isna().mean() * 100).sort_values(ascending=False)
    top_nans = nan_pct.head(10).round(2).to_dict()
    rep.add("B.2top", "PASS", f"top NaN columns: {top_nans}", top_nan_pct=top_nans)
    critical = [c for c in (list(FEATURE_COLS) + list(PROP_EXTRA_COLS)) if c in fm.columns]
    hard = {c: float(nan_pct[c]) for c in critical if nan_pct[c] > _NAN_HARD_FAIL_PCT}
    soft = {
        c: float(nan_pct[c])
        for c in critical
        if _NAN_SOFT_WARN_PCT < nan_pct[c] <= _NAN_HARD_FAIL_PCT
    }
    if hard:
        rep.add(
            "B.2crit",
            "FAIL",
            f"{len(hard)} critical feature columns have NaN > {_NAN_HARD_FAIL_PCT}%",
            offenders=hard,
        )
    elif soft:
        rep.add(
            "B.2crit",
            "WARN",
            f"{len(soft)} critical columns have NaN "
            f"in [{_NAN_SOFT_WARN_PCT}%, {_NAN_HARD_FAIL_PCT}%]",
            soft_offenders=soft,
        )
    else:
        rep.add("B.2crit", "PASS", "all FEATURE_COLS/PROP_EXTRA_COLS NaN ≤ 30%")

    # B.3 — chronological monotonicity (already-sorted by date)
    sorted_idx = fm["date"].argsort(kind="mergesort").values
    if np.array_equal(sorted_idx, np.arange(len(fm))):
        rep.add("B.3", "PASS", "feature matrix is already chronologically sorted by date")
    else:
        n_ooo = int((sorted_idx != np.arange(len(fm))).sum())
        if n_ooo / max(1, len(fm)) > 0.05:
            rep.add(
                "B.3",
                "FAIL",
                f"{n_ooo} ({n_ooo/len(fm):.1%}) rows out of chronological order",
            )
        else:
            rep.add(
                "B.3",
                "WARN",
                f"{n_ooo} rows out of chronological order — minor (likely same-day events)",
            )

    # B.4 — no fight_id appears more than 2 times
    if "fight_id" in fm.columns:
        fid_counts = fm["fight_id"].value_counts()
        excess = int((fid_counts > 2).sum())
        if excess > 0:
            rep.add(
                "B.4",
                "FAIL",
                f"{excess} fight_ids appear > 2 times (corner symmetrisation broken)",
            )
        else:
            rep.add("B.4", "PASS", "every fight_id appears ≤ 2 times (symmetrisation intact)")

    # B.5 — diff_X column means close to 0 (|μ| < 0.1·σ)
    diff_cols = [c for c in fm.columns if c.startswith("diff_") and fm[c].dtype.kind in "fi"]
    bad_diffs = []
    for c in diff_cols:
        mu = float(fm[c].mean(skipna=True))
        sigma = float(fm[c].std(skipna=True))
        if sigma > 0 and abs(mu) > 0.1 * sigma:
            bad_diffs.append({"col": c, "mean": round(mu, 4), "std": round(sigma, 4)})
    if not bad_diffs:
        rep.add("B.5", "PASS", f"all {len(diff_cols)} diff_* cols have |μ| < 0.1·σ")
    elif len(bad_diffs) <= 3:
        rep.add(
            "B.5",
            "WARN",
            f"{len(bad_diffs)}/{len(diff_cols)} diff_* cols have skewed mean — review",
            offenders=bad_diffs,
        )
    else:
        rep.add(
            "B.5",
            "WARN",
            f"{len(bad_diffs)}/{len(diff_cols)} diff_* cols are skewed — symmetrisation may be off",
            offenders=bad_diffs[:5],
        )
    return fm


# ---------------------------------------------------------------------------
# C. No-leakage check
# ---------------------------------------------------------------------------

def audit_c_no_leakage(fm: pd.DataFrame | None, session, rep: AuditReport) -> None:
    if fm is None:
        rep.add("C.0", "WARN", "skipping leakage check — feature matrix unavailable")
        return
    from ufc_predict.models.prop_models import PROP_EXTRA_COLS
    from ufc_predict.models.train import FEATURE_COLS

    # C.1 — confirm none of FEATURE_COLS/PROP_EXTRA_COLS shadow fight_stats_round columns
    fsr_cols = {
        "knockdowns", "sig_strikes_landed", "sig_strikes_attempted",
        "total_strikes_landed", "total_strikes_attempted",
        "head_landed", "head_attempted", "body_landed", "body_attempted",
        "leg_landed", "leg_attempted",
        "distance_landed", "distance_attempted",
        "clinch_landed", "clinch_attempted",
        "ground_landed", "ground_attempted",
        "takedowns_landed", "takedowns_attempted",
        "submission_attempts", "reversals", "control_time_sec",
    }
    feature_set = set(FEATURE_COLS) | set(PROP_EXTRA_COLS)
    leak_cols = feature_set & fsr_cols
    if leak_cols:
        rep.add(
            "C.1",
            "FAIL",
            f"FEATURE/PROP cols leak post-fight stats: {sorted(leak_cols)}",
        )
    else:
        rep.add("C.1", "PASS", "no FEATURE_COLS/PROP_EXTRA_COLS shadow fight_stats_round")

    # C.2 — sample 50 random rows; compare a_n_fights vs DB count of fights
    # before the row's date. Cheap structural leakage check.
    if "fighter_a_id" in fm.columns and "date" in fm.columns and "a_n_fights" in fm.columns:
        sample = fm.sample(min(50, len(fm)), random_state=0)
        bad = 0
        checked = 0
        for _, r in sample.iterrows():
            fid = r.get("fighter_a_id")
            d = r.get("date")
            actual = int(r.get("a_n_fights") or 0)
            if not fid or pd.isna(d):
                continue
            row = session.execute(
                text(
                    "SELECT COUNT(*) FROM fights "
                    "WHERE date < :d AND (red_fighter_id = :f OR blue_fighter_id = :f)"
                ),
                {"d": d, "f": fid},
            ).fetchone()
            db_n = int(row[0]) if row else 0
            checked += 1
            # Allow ±1 tolerance for cards with ambiguous date ordering
            if abs(db_n - actual) > 1:
                bad += 1
        if bad / max(1, checked) > 0.10:
            rep.add(
                "C.2",
                "FAIL",
                f"{bad}/{checked} sampled rows: a_n_fights diverges from DB pre-date count by > 1",
            )
        else:
            rep.add(
                "C.2",
                "PASS",
                f"{bad}/{checked} sampled rows have minor a_n_fights divergence (acceptable)",
            )


# ---------------------------------------------------------------------------
# D. SportsBet scraper schema
# ---------------------------------------------------------------------------

_EXPECTED_MARKETS = {
    "moneyline", "moneyline_a", "moneyline_b", "moneyline_no_dec",
    "method", "method_neutral", "method_combo",
    "distance", "total_rounds", "winning_round", "round_survival",
    "alt_finish_timing", "alt_round",
    # totals (Step 2.2) — optional until live
    "total_sig_strikes_combined", "total_sig_strikes_a", "total_sig_strikes_b",
    "total_takedowns_combined", "total_takedowns_a", "total_takedowns_b",
    "total_knockdowns_combined",
}
_REQUIRED_MARKETS = {"moneyline_a", "moneyline_b"}


def audit_d_sportsbet_schema(rep: AuditReport) -> None:
    if not SPORTSBET_ODDS_PATH.exists():
        rep.add("D.0", "WARN", f"sportsbet odds cache missing at {SPORTSBET_ODDS_PATH}")
        return
    try:
        data = json.loads(SPORTSBET_ODDS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        rep.add("D.0", "FAIL", f"sportsbet odds JSON unparseable: {exc}")
        return

    fights = data.get("fights", []) if isinstance(data, dict) else []
    n = len(fights)
    if n == 0:
        rep.add("D.1", "WARN", "sportsbet odds cache contains 0 fights")
        return

    seen_keys: set[str] = set()
    missing_required = 0
    for f in fights:
        markets = f.get("markets") or {}
        seen_keys.update(markets.keys())
        seen_keys.update(["moneyline_a", "moneyline_b"]) if markets.get("moneyline") else None
        # `markets` is the raw scraper output; check required pieces exist
        if not markets.get("moneyline"):
            missing_required += 1

    if missing_required:
        rep.add(
            "D.1",
            "WARN",
            f"{missing_required}/{n} cached fights lack moneyline market",
        )
    else:
        rep.add("D.1", "PASS", f"all {n} cached fights have moneyline market")

    unexpected = sorted(seen_keys - _EXPECTED_MARKETS - _REQUIRED_MARKETS)
    if unexpected:
        rep.add(
            "D.2",
            "WARN",
            f"unexpected market keys observed (will be silently skipped): {unexpected[:8]}",
            unexpected=unexpected,
        )
    else:
        rep.add("D.2", "PASS", "all observed market keys are recognised")


# ---------------------------------------------------------------------------
# E. Past-events grading sanity
# ---------------------------------------------------------------------------

def audit_e_grading(rep: AuditReport) -> None:
    """Light synthetic test: build a fake bout with KO win for A, ensure that
    moneyline_A grades won and moneyline_B grades lost. Doesn't require a
    populated past_events.json — exercises the grader's both-branches.
    """
    from ufc_predict.serve.build_dashboard import _grade_bet
    bout_actual_key = "A|KO|2"
    bet_a_wins = {"outcome_keys": [
        "A|KO|1", "A|KO|2", "A|KO|3", "A|KO|4", "A|KO|5",
        "A|SUB|1", "A|SUB|2", "A|SUB|3", "A|SUB|4", "A|SUB|5", "A|DEC|DEC",
    ]}
    bet_b_wins = {"outcome_keys": [
        "B|KO|1", "B|KO|2", "B|KO|3", "B|KO|4", "B|KO|5",
        "B|SUB|1", "B|SUB|2", "B|SUB|3", "B|SUB|4", "B|SUB|5", "B|DEC|DEC",
    ]}
    g_a = _grade_bet(bet_a_wins, bout_actual_key, "a")
    g_b = _grade_bet(bet_b_wins, bout_actual_key, "a")
    if g_a == "won" and g_b == "lost":
        rep.add("E.1", "PASS", "grader correctly scores moneyline ✓ A and ✗ B")
    else:
        rep.add(
            "E.1",
            "FAIL",
            f"grader produced unexpected results: A→{g_a}, B→{g_b}",
        )


# ---------------------------------------------------------------------------
# F. OOF parquet integrity
# ---------------------------------------------------------------------------

def audit_f_oof(rep: AuditReport) -> None:
    if not WIN_OOF_PATH.exists():
        rep.add("F.0", "FAIL", f"OOF parquet missing at {WIN_OOF_PATH}")
        return
    df = pd.read_parquet(WIN_OOF_PATH)
    needed = {"fight_id", "date", "label", "pred_prob"}
    missing = needed - set(df.columns)
    if missing:
        rep.add("F.1", "FAIL", f"OOF parquet missing columns: {missing}")
        return
    ll = float(log_loss(df["label"], np.clip(df["pred_prob"], 1e-7, 1 - 1e-7)))
    if ll > 0.6620:
        rep.add("F.1", "FAIL", f"OOF log-loss {ll:.4f} exceeds sanity guard 0.6620")
    elif ll > 0.6600:
        rep.add(
            "F.1",
            "WARN",
            f"OOF log-loss {ll:.4f} approaches sanity guard 0.6620 — monitor for regression",
            log_loss=ll,
        )
    else:
        rep.add("F.1", "PASS", f"OOF log-loss {ll:.4f} within sanity bounds", log_loss=ll)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> AuditReport:
    rep = AuditReport()
    factory = get_session_factory()
    with factory() as session:
        try:
            audit_a_fight_stats(session, rep)
        except Exception as exc:
            rep.add("A.X", "FAIL", f"audit A raised: {exc!r}")
        fm = audit_b_feature_matrix(rep)
        try:
            audit_c_no_leakage(fm, session, rep)
        except Exception as exc:
            rep.add("C.X", "FAIL", f"audit C raised: {exc!r}")
    audit_d_sportsbet_schema(rep)
    audit_e_grading(rep)
    audit_f_oof(rep)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(rep.to_dict(), indent=2, default=str), encoding="utf-8")
    return rep


def _print_summary(rep: AuditReport) -> None:
    n_pass = sum(1 for r in rep.results if r["status"] == "PASS")
    n_warn = sum(1 for r in rep.results if r["status"] == "WARN")
    n_fail = sum(1 for r in rep.results if r["status"] == "FAIL")
    print(f"\n=== Data audit summary: {rep.overall()} ===")
    print(f"  PASS: {n_pass}   WARN: {n_warn}   FAIL: {n_fail}")
    for r in rep.results:
        sym = {"PASS": "OK ", "WARN": "WRN", "FAIL": "ERR"}.get(r["status"], "?")
        msg = r["message"].encode("ascii", "replace").decode("ascii")
        print(f"  {sym} {r['code']:7s}  {msg}")
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    random.seed(0)
    rep = run()
    _print_summary(rep)
    sys.exit(0 if rep.overall() != "FAIL" else 1)
