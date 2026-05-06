"""Totals quantile-model edge backtest.

Joins ``models/totals_oof.parquet`` with historical totals odds to compute
per-bucket ROI for over/under bets on sig-strike, takedown, and knockdown
totals. Produces ``models/totals_edge_backtest.json`` in the same schema as
``models/prop_edge_backtest.json`` so ``bet_analysis._load_backtests()``
can read it and apply the empirical-ROI gate.

Important: at the time of writing, the BFO prop scraper covers
``total_rounds_X.X`` lines but not sig-strikes / takedowns / knockdowns
totals. When no historical odds are available for a market class, the
report emits an empty bucket list for that class with
``market_backtested: false`` recorded at the report level.

Usage:
    python -m ufc_predict.eval.totals_edge_backtest
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from ufc_predict.eval.evaluate import (
    american_odds_to_implied_prob,
    american_to_decimal,
    remove_vig,
)
from ufc_predict.models.totals_models import TOTAL_TARGETS, prob_over

log = logging.getLogger(__name__)

OOF_PATH = Path("models/totals_oof.parquet")
REPORT_PATH = Path("models/totals_edge_backtest.json")
DETAIL_PATH = Path("models/totals_edge_backtest_bets.csv")


@dataclass
class TotalBet:
    fight_id: str
    market_class: str
    line: float
    side: str            # "over" / "under"
    our_prob: float
    fair_implied: float | None
    edge: float | None
    decimal_odds: float
    outcome: int         # 1=won, 0=lost
    pnl_per_unit: float


# ---------------------------------------------------------------------------
# DB → odds mapping. Each row is keyed by (fight_id, prop_type, side).
# We only use rows whose prop_type maps to a totals market we model.
# ---------------------------------------------------------------------------

# prop_type prefix patterns that we accept as totals odds for backtesting
def _market_class_for_prop_type(prop_type: str) -> str | None:
    pt = prop_type.lower()
    if pt.startswith("total_sig_strikes_"):
        return "total_sig_strikes_combined"
    if pt.startswith("a_sig_strikes_") or pt.startswith("r_sig_strikes_"):
        return "total_sig_strikes_a"
    if pt.startswith("b_sig_strikes_"):
        return "total_sig_strikes_b"
    if pt.startswith("total_takedowns_"):
        return "total_takedowns_combined"
    if pt.startswith("a_takedowns_") or pt.startswith("r_takedowns_"):
        return "total_takedowns_a"
    if pt.startswith("b_takedowns_"):
        return "total_takedowns_b"
    if pt.startswith("total_knockdowns_"):
        return "total_knockdowns_combined"
    return None


def _parse_line(prop_type: str) -> float | None:
    """Extract the line from a prop_type string of the form ``..._X.X``."""
    parts = prop_type.split("_")
    try:
        return float(parts[-1])
    except (ValueError, IndexError):
        return None


def _build_quantiles(row: pd.Series, target: str) -> dict | None:
    keys = [f"{target}__q{int(a*100):02d}" for a in (0.10, 0.25, 0.50, 0.75, 0.90)]
    if any(k not in row or pd.isna(row[k]) for k in keys):
        return None
    return {
        "q10": float(row[keys[0]]),
        "q25": float(row[keys[1]]),
        "q50": float(row[keys[2]]),
        "q75": float(row[keys[3]]),
        "q90": float(row[keys[4]]),
    }


# ---------------------------------------------------------------------------
# Aggregation helpers (shared shape with prop_edge_backtest)
# ---------------------------------------------------------------------------

_BUCKETS = [
    (-1.0, 0.0), (0.0, 0.02), (0.02, 0.05), (0.05, 0.10),
    (0.10, 0.20), (0.20, 1.0),
]


def _aggregate_buckets(bets: list[TotalBet]) -> list[dict]:
    """Return per-(market_class, edge_bucket) ROI rows. Min n=30, ROI>5% to flag."""
    rows: list[dict] = []
    market_classes = sorted({b.market_class for b in bets})
    for mc in market_classes:
        mc_bets = [b for b in bets if b.market_class == mc]
        for lo, hi in _BUCKETS:
            sub = [b for b in mc_bets if (b.edge or 0) >= lo and (b.edge or 0) < hi]
            n = len(sub)
            if n == 0:
                continue
            total_pnl = sum(b.pnl_per_unit for b in sub)
            wins = sum(1 for b in sub if b.outcome == 1)
            rows.append({
                "market_class": mc,
                "edge_lo": lo, "edge_hi": hi,
                "n": n, "win_rate": wins / n,
                "flat_roi_pct": 100.0 * total_pnl / n,
                "avg_odds": float(np.mean([b.decimal_odds for b in sub])),
            })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(report_path: Path = REPORT_PATH) -> dict:
    if not OOF_PATH.exists():
        raise FileNotFoundError(
            f"totals OOF parquet missing at {OOF_PATH} — run "
            "ufc-train totals first"
        )
    oof = pd.read_parquet(OOF_PATH)
    log.info("totals OOF: %d rows", len(oof))

    from ufc_predict.db.session import get_session_factory
    factory = get_session_factory()
    with factory() as session:
        rows = session.execute(text(
            "SELECT fight_id, prop_type, side, american_odds FROM fight_prop_odds"
        )).fetchall()
    odds_df = pd.DataFrame(rows, columns=["fight_id", "prop_type", "side", "american_odds"])
    log.info("loaded %d prop closing-line rows from DB", len(odds_df))

    # Filter to rows whose prop_type maps to one of our totals targets.
    odds_df["market_class"] = odds_df["prop_type"].map(_market_class_for_prop_type)
    odds_df = odds_df.dropna(subset=["market_class"])
    odds_df["line"] = odds_df["prop_type"].map(_parse_line)
    odds_df = odds_df.dropna(subset=["line"])
    log.info("retained %d odds rows after totals filter", len(odds_df))

    markets_backtested = {t: False for t in TOTAL_TARGETS}
    if len(odds_df) > 0:
        for mc in odds_df["market_class"].unique():
            markets_backtested[str(mc)] = True

    bets: list[TotalBet] = []
    if not odds_df.empty:
        oof_lookup = oof.set_index("fight_id").to_dict("index")
        # Index odds by (fight_id, market_class, line) for over/under pairing
        grouped = odds_df.groupby(["fight_id", "market_class", "line", "side"]).first()
        idx_keys = list({(fid, mc, ln) for (fid, mc, ln, _s) in grouped.index})

        for fid, mc, line in idx_keys:
            oof_row = oof_lookup.get(fid)
            if oof_row is None:
                continue
            quantiles = _build_quantiles(pd.Series(oof_row), mc)
            actual_col = f"{mc}__actual"
            actual = oof_row.get(actual_col)
            if quantiles is None or actual is None or pd.isna(actual):
                continue

            sides_present = grouped.loc[(fid, mc, line)]
            over_am = (
                sides_present.loc["over", "american_odds"]
                if "over" in sides_present.index else None
            )
            under_am = (
                sides_present.loc["under", "american_odds"]
                if "under" in sides_present.index else None
            )
            if over_am is None and under_am is None:
                continue

            # Devig if we have both sides; otherwise use raw implied as fair
            fair_over = fair_under = None
            if over_am is not None and under_am is not None:
                io = american_odds_to_implied_prob(float(over_am))
                iu = american_odds_to_implied_prob(float(under_am))
                fo, fu = remove_vig(io, iu)
                fair_over, fair_under = fo, fu

            p_over = prob_over(line, quantiles)
            p_under = 1.0 - p_over
            actual_under = float(actual) < line  # exact equality counts as under per convention

            if over_am is not None:
                dec = american_to_decimal(float(over_am))
                fair = (
                    fair_over if fair_over is not None
                    else american_odds_to_implied_prob(float(over_am))
                )
                outcome = int(not actual_under)
                pnl = (dec - 1.0) if outcome else -1.0
                bets.append(TotalBet(
                    fight_id=fid, market_class=mc, line=float(line),
                    side="over", our_prob=p_over, fair_implied=fair,
                    edge=p_over - fair, decimal_odds=dec,
                    outcome=outcome, pnl_per_unit=pnl,
                ))
            if under_am is not None:
                dec = american_to_decimal(float(under_am))
                fair = (
                    fair_under if fair_under is not None
                    else american_odds_to_implied_prob(float(under_am))
                )
                outcome = int(actual_under)
                pnl = (dec - 1.0) if outcome else -1.0
                bets.append(TotalBet(
                    fight_id=fid, market_class=mc, line=float(line),
                    side="under", our_prob=p_under, fair_implied=fair,
                    edge=p_under - fair, decimal_odds=dec,
                    outcome=outcome, pnl_per_unit=pnl,
                ))

    log.info("built %d totals bets across %d market classes",
             len(bets), len({b.market_class for b in bets}))

    bucket_rows = _aggregate_buckets(bets)

    # Detail CSV (always written, even if empty, so CI artefacts are stable)
    DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if bets:
        pd.DataFrame([b.__dict__ for b in bets]).to_csv(DETAIL_PATH, index=False)
    else:
        DETAIL_PATH.write_text("", encoding="utf-8")

    note = (
        "BFO does not currently scrape sig-strike / takedown / knockdown totals odds. "
        "Markets without historical data fall through to the default EV-gate in "
        "bet_analysis (positive EV + edge >= MIN_EDGE)."
        if not bets
        else "Backtested against fight_prop_odds rows that map to totals targets."
    )
    report = {
        "n_bets_total": len(bets),
        "market_classes": sorted({b.market_class for b in bets}),
        "markets_backtested": markets_backtested,
        "buckets": bucket_rows,
        "note": note,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n=== Totals edge-bucket ROI ===")
    print(report["note"])
    if bucket_rows:
        print(
            f"{'market':32s}  {'edge':>14s}  {'n':>5s}  {'win%':>6s}  "
            f"{'avg odds':>9s}  {'flat ROI%':>10s}"
        )
        for r in bucket_rows:
            print(
                f"{r['market_class']:32s}  [{r['edge_lo']:+.2f},{r['edge_hi']:+.2f})  "
                f"{r['n']:>5d}  {100*r['win_rate']:>5.1f}%  {r['avg_odds']:>9.2f}  "
                f"{r['flat_roi_pct']:>+9.2f}%"
            )
    else:
        print("(no bets — report flags every totals market as not yet backtested)")
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
