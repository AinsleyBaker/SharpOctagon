"""Edge-bucket backtest on OOF predictions vs Vegas closing line.

The naive-Kelly simulation in evaluate.py bets every fight where our
calibrated probability disagrees with the vig-removed implied probability.
Empirically (Week 4 audit) that yields ROI ~ -100%: we systematically bet
underdogs when our model is less confident than Vegas, and underdogs lose
more often than they win. Vegas wins by ~5.7pp log-loss on the 1792-row
overlap sample.

This module bins those same OOF bets by *edge size* (our_prob − vig_removed
implied_prob) and computes ROI per bucket. The output answers: "is there a
minimum edge above which our model historically returns positive ROI?"

Usage:
    python -m ufc_predict.eval.edge_backtest
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_predict.eval.evaluate import (
    KELLY_FRACTION, STARTING_BANKROLL, american_odds_to_implied_prob,
    american_to_decimal, kelly_fraction, remove_vig,
)

log = logging.getLogger(__name__)
OOF_PATH = Path("models/oof_predictions.parquet")
REPORT_PATH = Path("models/edge_backtest.json")
DETAIL_PATH = Path("models/edge_backtest_bets.csv")


@dataclass
class Bet:
    fight_id: str
    date: str
    side: str            # "A" or "B"
    our_prob: float
    fair_implied: float  # vig-removed implied prob
    edge: float
    decimal_odds: float
    outcome: int         # 1=won, 0=lost
    pnl_per_unit: float  # PnL per $1 staked (decimal_odds-1 if won, -1 if lost)


def _build_bets(oof: pd.DataFrame) -> list[Bet]:
    """For every OOF row with both closing odds, compute the bet we'd have
    placed on whichever side has positive edge (model > implied)."""
    bets: list[Bet] = []
    a_is_red = oof.get("a_is_red")
    for i, row in oof.iterrows():
        odds_red = row.get("closing_odds_red")
        odds_blue = row.get("closing_odds_blue")
        if pd.isna(odds_red) or pd.isna(odds_blue):
            continue

        # Map red/blue → A/B according to corner randomization
        if a_is_red is not None and bool(row.get("a_is_red", True)):
            odds_a, odds_b = float(odds_red), float(odds_blue)
        else:
            odds_a, odds_b = float(odds_blue), float(odds_red)

        impl_a = american_odds_to_implied_prob(odds_a)
        impl_b = american_odds_to_implied_prob(odds_b)
        fair_a, fair_b = remove_vig(impl_a, impl_b)

        prob_a = float(row["pred_prob"])
        prob_b = 1.0 - prob_a
        won_a = int(row["label"])

        edge_a = prob_a - fair_a
        edge_b = prob_b - fair_b
        # Bet whichever side has the larger edge (only one can be > 0)
        if edge_a >= edge_b:
            side = "A"
            our_p, fair_p, dec, outcome = prob_a, fair_a, american_to_decimal(odds_a), won_a
        else:
            side = "B"
            our_p, fair_p, dec, outcome = prob_b, fair_b, american_to_decimal(odds_b), 1 - won_a

        edge = our_p - fair_p
        pnl = (dec - 1.0) if outcome == 1 else -1.0
        bets.append(Bet(
            fight_id=str(row.get("fight_id", i)),
            date=str(row.get("date", "")),
            side=side,
            our_prob=our_p,
            fair_implied=fair_p,
            edge=edge,
            decimal_odds=dec,
            outcome=outcome,
            pnl_per_unit=pnl,
        ))
    return bets


def _bucket_stats(bets: list[Bet], buckets: list[tuple[float, float]]) -> list[dict]:
    """Aggregate ROI/win-rate per edge bucket. Each bucket is [lo, hi)."""
    rows = []
    for lo, hi in buckets:
        sample = [b for b in bets if lo <= b.edge < hi]
        n = len(sample)
        if n == 0:
            rows.append({"edge_lo": lo, "edge_hi": hi, "n": 0,
                         "roi_pct": np.nan, "win_rate": np.nan,
                         "avg_odds": np.nan, "avg_edge": np.nan})
            continue
        # Flat-stake ROI
        total_pnl = sum(b.pnl_per_unit for b in sample)
        wins = sum(1 for b in sample if b.outcome == 1)
        rows.append({
            "edge_lo": lo, "edge_hi": hi, "n": n,
            "roi_pct": 100.0 * total_pnl / n,
            "win_rate": wins / n,
            "avg_odds": float(np.mean([b.decimal_odds for b in sample])),
            "avg_edge": float(np.mean([b.edge for b in sample])),
        })
    return rows


def _kelly_curve(bets: list[Bet], min_edges: list[float]) -> list[dict]:
    """For each minimum-edge threshold, simulate quarter-Kelly bankroll and
    report ROI. Threshold of 0 = bet whenever model has any positive edge."""
    rows = []
    for thresh in min_edges:
        bankroll = STARTING_BANKROLL
        n_bets = 0
        wins = 0
        for b in bets:
            if b.edge < thresh:
                continue
            n_bets += 1
            frac = kelly_fraction(b.our_prob, b.decimal_odds, KELLY_FRACTION)
            stake = bankroll * frac
            bankroll += stake * b.pnl_per_unit
            if b.outcome == 1:
                wins += 1
        rows.append({
            "min_edge": thresh,
            "n_bets": n_bets,
            "final_bankroll": float(bankroll),
            "roi_pct": 100.0 * (bankroll - STARTING_BANKROLL) / STARTING_BANKROLL,
            "win_rate": (wins / n_bets) if n_bets else float("nan"),
        })
    return rows


def run(report_path: Path = REPORT_PATH) -> dict:
    if not OOF_PATH.exists():
        raise FileNotFoundError(f"OOF predictions not found at {OOF_PATH}")
    oof = pd.read_parquet(OOF_PATH)
    needed = {"closing_odds_red", "closing_odds_blue", "pred_prob", "label"}
    if not needed.issubset(oof.columns):
        raise RuntimeError(f"OOF missing required columns: {needed - set(oof.columns)}")

    bets = _build_bets(oof)
    log.info("edge_backtest: %d bets across %d OOF rows", len(bets), len(oof))

    buckets = [
        (-1.0, 0.00),   # negative edge — sanity check
        (0.00, 0.02),   # razor-thin
        (0.02, 0.05),   # marginal
        (0.05, 0.10),   # solid
        (0.10, 0.20),   # large
        (0.20, 1.0),    # extreme — usually parsing artifacts
    ]
    bucket_rows = _bucket_stats(bets, buckets)

    kelly_thresholds = [0.0, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
    kelly_rows = _kelly_curve(bets, kelly_thresholds)

    # Save details for inspection
    pd.DataFrame([b.__dict__ for b in bets]).to_csv(DETAIL_PATH, index=False)

    report = {
        "n_bets_total": len(bets),
        "buckets": bucket_rows,
        "kelly_thresholds": kelly_rows,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Pretty-print summary
    print("\n=== Edge bucket ROI (flat $1 stake per bet) ===")
    print(f"{'edge bucket':>14}  {'n':>5}  {'win%':>6}  {'avg odds':>8}  {'flat ROI%':>9}")
    for r in bucket_rows:
        if r["n"] == 0:
            continue
        print(f"  [{r['edge_lo']:+.2f}, {r['edge_hi']:+.2f})  {r['n']:5d}  "
              f"{100*r['win_rate']:5.1f}%  {r['avg_odds']:>8.2f}  {r['roi_pct']:+8.2f}%")
    print("\n=== Kelly curve by min-edge threshold (¼ Kelly) ===")
    print(f"{'min edge':>9}  {'n_bets':>7}  {'win%':>6}  {'final $':>12}  {'ROI%':>9}")
    for r in kelly_rows:
        wr = r["win_rate"]
        wr_str = f"{100*wr:5.1f}%" if isinstance(wr, float) and not np.isnan(wr) else "    -"
        print(f"  {100*r['min_edge']:>+5.0f}pp  {r['n_bets']:>7d}  {wr_str}  "
              f"{r['final_bankroll']:>12.2f}  {r['roi_pct']:+8.2f}%")

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
