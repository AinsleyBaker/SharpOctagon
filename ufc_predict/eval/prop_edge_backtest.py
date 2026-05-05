"""Prop edge-bucket backtest — analogous to edge_backtest.py but per market.

Joins the OOF prop predictions (prop_oof.parquet) with historical prop
closing odds (fight_prop_odds table). For each prop bet:
  - Compute model's predicted probability for that specific (prop_type, side)
  - Compute fair (vig-removed) implied probability from yes/no pair odds
  - Compute edge = model_prob − fair_implied
  - Compute outcome from method_class_true / round_class_true
  - Compute PnL per unit stake (decimal_odds − 1 if won, −1 if lost)

Aggregate ROI per (prop_market_class, edge_bucket) to identify subsets
where our model has positive expected ROI vs the prop closing line.

Run after the BFO prop scraper has populated fight_prop_odds.
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
    KELLY_FRACTION,
    STARTING_BANKROLL,
    american_odds_to_implied_prob,
    american_to_decimal,
    kelly_fraction,
    remove_vig,
)

log = logging.getLogger(__name__)
PROP_OOF_PATH = Path("models/prop_oof.parquet")
WIN_OOF_PATH = Path("models/oof_predictions.parquet")
REPORT_PATH = Path("models/prop_edge_backtest.json")
DETAIL_PATH = Path("models/prop_edge_backtest_bets.csv")


@dataclass
class PropBet:
    fight_id: str
    prop_type: str
    side: str
    market_class: str
    our_prob: float
    fair_implied: float | None
    edge: float | None
    decimal_odds: float
    outcome: int
    pnl_per_unit: float


# -------------------------------------------------------------------
# Mapping prop_type → (model_prob_fn, outcome_fn)
# -------------------------------------------------------------------
# Each entry takes a row dict (joined OOF + ground truth) and a `side`
# ('yes'/'no'/'over'/'under'/...) and returns a probability or outcome.

_R_METHODS = {"KO_TKO", "SUB", "DEC", "UD", "SMD"}


def _model_prob(prop_type: str, side: str, row: dict) -> float | None:
    """Return our model's predicted probability for this prop+side."""
    pt = prop_type
    s = side

    # ---- Distance ----
    if pt == "distance":
        p_dec = float(row.get("prob_a_dec", 0)) + float(row.get("prob_b_dec", 0))
        return p_dec if s == "yes" else (1 - p_dec)

    # ---- Total rounds (over/under N.5) ----
    if pt.startswith("total_rounds_"):
        try:
            line = float(pt.split("_")[-1])
        except ValueError:
            return None
        threshold = int(line + 0.5)  # 2.5 → 2 (over = past R2 = R3 onward)
        prob_rounds = {f"R{i}": float(row.get(f"prob_R{i}", 0)) for i in range(1, 6)}
        prob_dec = float(row.get("prob_a_dec", 0)) + float(row.get("prob_b_dec", 0))
        # under = finish in rounds 1..threshold; over = anything else (later finish or decision)
        p_under = sum(v for k, v in prob_rounds.items() if int(k[1:]) <= threshold)
        p_over = max(0.0, 1.0 - p_under)
        return p_over if s == "over" else p_under

    # ---- Starts round N (yes if fight reaches R(N), else no) ----
    if pt.startswith("starts_round_"):
        try:
            n = int(pt.split("_")[-1])
        except ValueError:
            return None
        prob_rounds = {i: float(row.get(f"prob_R{i}", 0)) for i in range(1, 6)}
        # P(fight finishes before R(N)) = sum P(R_i) for i<N
        p_finishes_early = sum(v for i, v in prob_rounds.items() if i < n)
        # P(reaches R(N)) = 1 - P(finished earlier).
        # Decisions reach all rounds, so they count as "starts"
        p_starts = max(0.0, 1.0 - p_finishes_early)
        return p_starts if s == "yes" else (1 - p_starts)

    # ---- Ends round N (neutral) ----
    if pt.startswith("ends_round_"):
        try:
            n = int(pt.split("_")[-1])
        except ValueError:
            return None
        p_rn = float(row.get(f"prob_R{n}", 0))
        return p_rn if s == "yes" else (1 - p_rn)

    # ---- Method per fighter (r_method_KO_TKO, b_method_SUB, etc.) ----
    if "_method_" in pt:
        side_letter = pt[0]   # 'r' or 'b'
        method_key = pt.split("_method_")[1]
        # Skip UD/SMD splits — model doesn't predict these granularly
        if method_key not in {"KO_TKO", "SUB", "DEC"}:
            return None
        # Resolve r/b → a/b via row's a_is_red
        a_is_red = bool(row.get("a_is_red", True))
        target = "a" if (side_letter == "r" and a_is_red) or (side_letter == "b" and not a_is_red) else "b"
        col = f"prob_{target}_{method_key.lower()}"
        p = float(row.get(col, 0))
        return p if s == "yes" else (1 - p)

    # ---- Inside distance per fighter ----
    if pt.endswith("_inside_distance"):
        side_letter = pt[0]
        a_is_red = bool(row.get("a_is_red", True))
        target = "a" if (side_letter == "r" and a_is_red) or (side_letter == "b" and not a_is_red) else "b"
        # P(target wins by finish) = ko + sub
        p = float(row.get(f"prob_{target}_ko_tko", 0)) + float(row.get(f"prob_{target}_sub", 0))
        return p if s == "yes" else (1 - p)

    # ---- Wins in round N per fighter ----
    if "_wins_round_" in pt:
        side_letter = pt[0]
        try:
            n = int(pt.split("_")[-1])
        except ValueError:
            return None
        a_is_red = bool(row.get("a_is_red", True))
        target = "a" if (side_letter == "r" and a_is_red) or (side_letter == "b" and not a_is_red) else "b"
        # P(target wins in round n) = P(target wins) × P(R_n | finish) × P(finish)
        # ≈ P(target_KO_TKO + target_SUB) × P(round_n | finish)
        target_finish = (
            float(row.get(f"prob_{target}_ko_tko", 0))
            + float(row.get(f"prob_{target}_sub", 0))
        )
        prob_finish = (
            float(row.get("prob_a_ko_tko", 0)) + float(row.get("prob_a_sub", 0))
            + float(row.get("prob_b_ko_tko", 0)) + float(row.get("prob_b_sub", 0))
        )
        if prob_finish < 1e-6:
            return None
        p_round = float(row.get(f"prob_R{n}", 0))  # P(R_n | finish)
        # target_finish is unconditional P(target wins by finish); round prob
        # is P(R_n | finish). Multiply for joint, since round model trained on
        # finishes only.
        p_joint = target_finish * p_round
        return p_joint if s == "yes" else (1 - p_joint)

    return None


def _outcome(prop_type: str, side: str, row: dict) -> int | None:
    """1 if the bet would have won, 0 if lost, None if not gradable."""
    method_true = str(row.get("method_class_true", ""))
    round_true = str(row.get("round_class_true", ""))
    a_is_red = bool(row.get("a_is_red", True))

    if not method_true:
        return None

    # ---- Distance ----
    if prop_type == "distance":
        is_dec = method_true.endswith("_DEC")
        return int(is_dec if side == "yes" else not is_dec)

    # ---- Total rounds ----
    if prop_type.startswith("total_rounds_"):
        try:
            line = float(prop_type.split("_")[-1])
        except ValueError:
            return None
        threshold = int(line + 0.5)
        # If decision: fight reached its full schedule → always "over" any line ≤ max-rounds-1.5
        if method_true.endswith("_DEC"):
            actual_under = False
        else:
            try:
                rn = int(round_true[1:])
            except (ValueError, IndexError):
                return None
            actual_under = rn <= threshold
        return int(actual_under if side == "under" else not actual_under)

    # ---- Starts round N ----
    if prop_type.startswith("starts_round_"):
        try:
            n = int(prop_type.split("_")[-1])
        except ValueError:
            return None
        if method_true.endswith("_DEC"):
            reached = True
        else:
            try:
                rn = int(round_true[1:])
            except (ValueError, IndexError):
                return None
            reached = rn >= n
        return int(reached if side == "yes" else not reached)

    # ---- Ends round N ----
    if prop_type.startswith("ends_round_"):
        try:
            n = int(prop_type.split("_")[-1])
        except ValueError:
            return None
        if method_true.endswith("_DEC"):
            ends_in_n = False  # decisions don't "end" in a round
        else:
            try:
                rn = int(round_true[1:])
            except (ValueError, IndexError):
                return None
            ends_in_n = (rn == n)
        return int(ends_in_n if side == "yes" else not ends_in_n)

    # ---- Method per fighter ----
    if "_method_" in prop_type and not prop_type.endswith("_distance"):
        side_letter = prop_type[0]
        method_key = prop_type.split("_method_")[1]
        if method_key not in {"KO_TKO", "SUB", "DEC"}:
            return None
        # Resolve r/b → A/B
        target_ab = "A" if (side_letter == "r" and a_is_red) or (side_letter == "b" and not a_is_red) else "B"
        actual = (method_true == f"{target_ab}_{method_key}")
        return int(actual if side == "yes" else not actual)

    # ---- Inside distance per fighter ----
    if prop_type.endswith("_inside_distance"):
        side_letter = prop_type[0]
        target_ab = "A" if (side_letter == "r" and a_is_red) or (side_letter == "b" and not a_is_red) else "B"
        actual = (method_true == f"{target_ab}_KO_TKO" or method_true == f"{target_ab}_SUB")
        return int(actual if side == "yes" else not actual)

    # ---- Wins in round N per fighter ----
    if "_wins_round_" in prop_type:
        side_letter = prop_type[0]
        try:
            n = int(prop_type.split("_")[-1])
        except ValueError:
            return None
        target_ab = "A" if (side_letter == "r" and a_is_red) or (side_letter == "b" and not a_is_red) else "B"
        # Bet wins if target_ab finished the fight in round n (decisions don't count)
        try:
            rn = int(round_true[1:]) if round_true.startswith("R") else None
        except (ValueError, IndexError):
            rn = None
        target_won_by_finish = method_true in {f"{target_ab}_KO_TKO", f"{target_ab}_SUB"}
        actual = bool(rn == n and target_won_by_finish)
        return int(actual if side == "yes" else not actual)

    return None


def _market_class(prop_type: str) -> str:
    """Group prop_types into market classes for aggregation."""
    if prop_type == "distance":
        return "distance"
    if prop_type.startswith("total_rounds_"):
        return "total_rounds"
    if prop_type.startswith("starts_round_"):
        return "starts_round"
    if prop_type.startswith("ends_round_"):
        return "ends_round"
    if "_method_" in prop_type:
        return "method"
    if prop_type.endswith("_inside_distance"):
        return "inside_distance"
    if "_wins_round_" in prop_type:
        return "wins_round"
    return "other"


# -------------------------------------------------------------------
# Main backtest
# -------------------------------------------------------------------

def _build_oof_index(prop_oof: pd.DataFrame, win_oof: pd.DataFrame) -> pd.DataFrame:
    """Join prop OOF with win OOF on fight_id, picking one row per fight (the
    a_is_red=True copy when both halves are present). Adds the `a_is_red`
    column that the prop predictions don't have."""
    # Win OOF has 2 rows per fight (symmetrized). Pick the half where
    # a_is_red=True so the A=red interpretation is canonical.
    w = win_oof[win_oof["a_is_red"] == 1][["fight_id", "a_is_red"]].drop_duplicates("fight_id")
    p = prop_oof.drop_duplicates("fight_id")  # take any one half — probs are A-coordinate
    return p.merge(w, on="fight_id", how="inner")


def run(report_path: Path = REPORT_PATH) -> dict:
    if not PROP_OOF_PATH.exists() or not WIN_OOF_PATH.exists():
        raise FileNotFoundError("OOF parquets missing — run prop_models.run_cv and train_runner")

    prop_oof = pd.read_parquet(PROP_OOF_PATH)
    win_oof = pd.read_parquet(WIN_OOF_PATH)
    log.info("prop_oof: %d rows, win_oof: %d rows", len(prop_oof), len(win_oof))

    joined = _build_oof_index(prop_oof, win_oof)
    log.info("joined OOF: %d unique fights", len(joined))

    # Pull prop closing odds
    from ufc_predict.db.session import get_session_factory
    factory = get_session_factory()
    with factory() as session:
        odds_rows = session.execute(text("""
            SELECT fight_id, prop_type, side, american_odds
            FROM fight_prop_odds
        """)).fetchall()
    if not odds_rows:
        raise RuntimeError("No rows in fight_prop_odds — run backfill_props_all first")
    odds_df = pd.DataFrame(odds_rows, columns=["fight_id", "prop_type", "side", "american_odds"])
    log.info("loaded %d prop closing-line rows from DB", len(odds_df))

    # Index odds by (fight_id, prop_type) for vig removal across yes/no pairs
    odds_lookup: dict[tuple[str, str], dict[str, float]] = {}
    for fid, pt, side, am in odds_rows:
        odds_lookup.setdefault((fid, pt), {})[side] = float(am)

    # Build the bet list
    oof_lookup = joined.set_index("fight_id").to_dict("index")
    bets: list[PropBet] = []
    skipped_no_oof = 0
    skipped_no_outcome = 0
    skipped_no_pair = 0

    for (fid, pt), sides in odds_lookup.items():
        oof_row = oof_lookup.get(fid)
        if not oof_row:
            skipped_no_oof += 1
            continue
        # Need a yes/no pair for vig removal — most prop markets come in pairs
        pair_keys = list(sides.keys())
        if len(pair_keys) < 2:
            # Single-sided prop — fall back to using just implied prob without devig
            for side, am in sides.items():
                model_p = _model_prob(pt, side, oof_row)
                outcome = _outcome(pt, side, oof_row)
                if model_p is None or outcome is None:
                    skipped_no_outcome += 1
                    continue
                impl = american_odds_to_implied_prob(am)
                edge = model_p - impl  # not vig-removed
                dec = american_to_decimal(am)
                pnl = (dec - 1.0) if outcome else -1.0
                bets.append(PropBet(
                    fight_id=fid, prop_type=pt, side=side,
                    market_class=_market_class(pt), our_prob=model_p,
                    fair_implied=None, edge=edge, decimal_odds=dec,
                    outcome=outcome, pnl_per_unit=pnl,
                ))
            continue

        # Devig the pair
        side1, side2 = pair_keys[0], pair_keys[1]
        i1 = american_odds_to_implied_prob(sides[side1])
        i2 = american_odds_to_implied_prob(sides[side2])
        f1, f2 = remove_vig(i1, i2)
        fair = {side1: f1, side2: f2}

        for side, am in sides.items():
            model_p = _model_prob(pt, side, oof_row)
            outcome = _outcome(pt, side, oof_row)
            if model_p is None or outcome is None:
                skipped_no_outcome += 1
                continue
            edge = model_p - fair[side]
            dec = american_to_decimal(am)
            pnl = (dec - 1.0) if outcome else -1.0
            bets.append(PropBet(
                fight_id=fid, prop_type=pt, side=side,
                market_class=_market_class(pt), our_prob=model_p,
                fair_implied=fair[side], edge=edge, decimal_odds=dec,
                outcome=outcome, pnl_per_unit=pnl,
            ))

    log.info(
        "Built %d prop bets. skipped: no_oof=%d  no_outcome=%d  no_pair=%d",
        len(bets), skipped_no_oof, skipped_no_outcome, skipped_no_pair,
    )

    # Aggregate per (market_class, edge_bucket)
    buckets = [(-1.0, 0.0), (0.0, 0.02), (0.02, 0.05), (0.05, 0.10),
               (0.10, 0.20), (0.20, 1.0)]
    market_classes = sorted({b.market_class for b in bets})

    agg_rows = []
    for mc in market_classes:
        mc_bets = [b for b in bets if b.market_class == mc]
        for lo, hi in buckets:
            sub = [b for b in mc_bets if (b.edge or 0) >= lo and (b.edge or 0) < hi]
            n = len(sub)
            if n == 0:
                continue
            wins = sum(1 for b in sub if b.outcome == 1)
            total_pnl = sum(b.pnl_per_unit for b in sub)
            agg_rows.append({
                "market_class": mc,
                "edge_lo": lo, "edge_hi": hi,
                "n": n, "win_rate": wins / n,
                "flat_roi_pct": 100.0 * total_pnl / n,
                "avg_odds": float(np.mean([b.decimal_odds for b in sub])),
            })

    # Kelly curve per market_class at varying min-edge thresholds.
    # Cap stake at MAX_STAKE_FRAC of bankroll per bet — uncapped Kelly with
    # positive-EV bets compounds to fantasy numbers ($1T+ on a small sample),
    # which is technically correct but useless for sizing decisions. The cap
    # also matches real-world bankroll discipline.
    MAX_STAKE_FRAC = 0.05
    kelly_thresh = [0.0, 0.02, 0.05, 0.10]
    kelly_rows = []
    for mc in market_classes:
        mc_bets = [b for b in bets if b.market_class == mc]
        for thresh in kelly_thresh:
            bankroll = STARTING_BANKROLL
            n_bets = 0
            for b in mc_bets:
                if (b.edge or 0) < thresh:
                    continue
                if b.decimal_odds <= 1.0:
                    continue
                n_bets += 1
                frac = min(
                    kelly_fraction(b.our_prob, b.decimal_odds, KELLY_FRACTION),
                    MAX_STAKE_FRAC,
                )
                stake = bankroll * frac
                bankroll += stake * b.pnl_per_unit
            kelly_rows.append({
                "market_class": mc,
                "min_edge": thresh,
                "n_bets": n_bets,
                "final_bankroll": float(bankroll),
                "roi_pct": 100.0 * (bankroll - STARTING_BANKROLL) / STARTING_BANKROLL,
            })

    # Save details + report
    pd.DataFrame([b.__dict__ for b in bets]).to_csv(DETAIL_PATH, index=False)
    report = {
        "n_bets_total": len(bets),
        "market_classes": market_classes,
        "buckets": agg_rows,
        "kelly": kelly_rows,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Pretty print
    print("\n=== Prop edge-bucket ROI per market class ===")
    print(f"{'market':18s}  {'edge bucket':>15s}  {'n':>5s}  {'win%':>6s}  {'avg_odds':>9s}  {'flat ROI%':>10s}")
    for r in agg_rows:
        print(f"{r['market_class']:18s}  [{r['edge_lo']:+.2f},{r['edge_hi']:+.2f})  "
              f"{r['n']:>5d}  {100*r['win_rate']:>5.1f}%  {r['avg_odds']:>9.2f}  {r['flat_roi_pct']:>+9.2f}%")

    print("\n=== Kelly curve per market class ===")
    print(f"{'market':18s}  {'min edge':>9s}  {'n_bets':>7s}  {'final $':>12s}  {'ROI%':>9s}")
    for r in kelly_rows:
        print(f"{r['market_class']:18s}  {100*r['min_edge']:>+5.0f}pp   {r['n_bets']:>7d}  "
              f"{r['final_bankroll']:>12.2f}  {r['roi_pct']:>+9.2f}%")

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
