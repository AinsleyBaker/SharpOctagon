"""Tests for build_portfolio() — bankroll allocation logic."""

import pytest

from ufc_predict.eval.bet_analysis import (
    PORTFOLIO_MAX_BETS_PER_FIGHT,
    PORTFOLIO_MAX_PER_BET_PCT,
    PORTFOLIO_MAX_PER_FIGHT_PCT,
    PORTFOLIO_MAX_TOTAL_PCT,
    PORTFOLIO_TOP_N,
    build_portfolio,
)


def _bet(fight, our_prob, odds, kelly_pct, **extra):
    return {
        "fight":     fight,
        "our_prob":  our_prob,
        "sb_odds":   odds,
        "kelly_pct": kelly_pct,
        "description": extra.get("description", f"{fight} bet"),
        "event":     extra.get("event", "Test Event"),
        "bet_type":  extra.get("bet_type", "method"),
    }


def test_empty_input_returns_zero_summary():
    p = build_portfolio([], strategy="kelly")
    assert p["bets"] == []
    assert p["summary"]["n_bets"] == 0
    assert p["summary"]["total_stake_pct"] == 0


def test_kelly_caps_per_bet():
    # A single bet whose raw kelly_pct exceeds max_per_bet should be clipped
    bets = [_bet("A vs B", 0.7, 2.5, kelly_pct=12.0)]
    p = build_portfolio(bets, strategy="kelly")
    assert len(p["bets"]) == 1
    assert p["bets"][0]["stake_pct"] == PORTFOLIO_MAX_PER_BET_PCT


def test_kelly_caps_per_fight():
    # Three correlated legs on the same bout, each at max_per_bet.
    # Without the per-fight cap, sum would be 15% — exactly at the limit.
    # Push one extra bet on the same fight: total should scale to 15%.
    bets = [_bet("A vs B", 0.7, 2.5, kelly_pct=PORTFOLIO_MAX_PER_BET_PCT) for _ in range(4)]
    p = build_portfolio(bets, strategy="kelly")
    fight_total = sum(b["stake_pct"] for b in p["bets"])
    assert abs(fight_total - PORTFOLIO_MAX_PER_FIGHT_PCT) < 0.01


def test_kelly_caps_total_across_fights():
    # 20 separate fights × 5% each = 100% raw. Should scale down to 50%.
    bets = [_bet(f"Fight {i}", 0.6, 2.0, kelly_pct=8.0) for i in range(20)]
    p = build_portfolio(bets, strategy="kelly")
    total = sum(b["stake_pct"] for b in p["bets"])
    assert abs(total - PORTFOLIO_MAX_TOTAL_PCT) < 0.5


def test_flat_distributes_equally():
    bets = [_bet(f"Fight {i}", 0.55, 2.0, kelly_pct=3.0) for i in range(5)]
    p = build_portfolio(bets, strategy="flat")
    stakes = [b["stake_pct"] for b in p["bets"]]
    # 50% / 5 = 10%, but per-bet cap is 5%
    for s in stakes:
        assert s <= PORTFOLIO_MAX_PER_BET_PCT + 1e-6
    # All equal
    assert max(stakes) - min(stakes) < 0.01


def test_flat_drops_tiny_stakes():
    # 200 fights → each gets 50/200 = 0.25%, below min — all dropped
    bets = [_bet(f"Fight {i}", 0.55, 2.0, kelly_pct=2.0) for i in range(200)]
    p = build_portfolio(bets, strategy="flat")
    assert p["bets"] == []
    assert p["summary"]["total_stake_pct"] == 0


def test_summary_sums_match_bets():
    bets = [
        _bet("A vs B", 0.6, 2.2, kelly_pct=3.5),
        _bet("C vs D", 0.55, 2.5, kelly_pct=2.8),
    ]
    p = build_portfolio(bets, strategy="kelly")
    s = p["summary"]
    assert abs(s["total_stake_pct"] - sum(b["stake_pct"] for b in p["bets"])) < 0.05
    assert s["worst_case_pct"] == -s["total_stake_pct"]
    assert s["n_bets"] == len(p["bets"])
    assert s["n_fights"] == 2


def test_caps_at_max_bets_per_fight():
    # 8 legs on the same fight, all with overlapping outcome_keys (e.g.
    # different "A wins by KO in round N" lines that all share A|KO|*).
    # Should drop down to PORTFOLIO_MAX_BETS_PER_FIGHT after the conflict
    # filter — even before the per-fight stake scaling kicks in.
    bets = []
    for r in range(1, 9):
        b = _bet("A vs B", 0.55, 2.5, kelly_pct=2.0)
        b["outcome_keys"] = [f"A|KO|{r}"]  # all share method=KO; mutually exclusive on round
        bets.append(b)
    # Outcome_keys are pairwise disjoint, so the filter keeps only the first.
    p = build_portfolio(bets, strategy="kelly")
    assert len(p["bets"]) == 1


def test_drops_contradictory_bets_on_same_fight():
    # The user's example: betting "Khamzat KO R1" alongside "fight goes
    # to decision" — these can never both win. The lower-stake leg gets
    # dropped by the conflict filter.
    ko_r1 = _bet("Khamzat vs X", 0.4, 2.6, kelly_pct=4.0)
    ko_r1["outcome_keys"] = ["A|KO|1"]
    decision = _bet("Khamzat vs X", 0.35, 2.9, kelly_pct=2.0)
    decision["outcome_keys"] = ["A|DEC|DEC", "B|DEC|DEC"]
    p = build_portfolio([ko_r1, decision], strategy="kelly")
    descs = [b["description"] for b in p["bets"]]
    # Higher-stake leg (KO R1) is kept; the disjoint decision bet is dropped.
    assert len(p["bets"]) == 1
    assert "Khamzat vs X bet" in descs[0]


def test_compatible_bets_on_same_fight_both_kept():
    # "A wins by KO" (any round) and "A wins by KO in R2" share A|KO|2 —
    # they CAN both win, so both should survive the conflict filter.
    a_ko_any = _bet("A vs B", 0.4, 2.5, kelly_pct=3.0)
    a_ko_any["outcome_keys"] = [f"A|KO|{r}" for r in range(1, 6)]
    a_ko_r2 = _bet("A vs B", 0.15, 5.0, kelly_pct=2.0)
    a_ko_r2["outcome_keys"] = ["A|KO|2"]
    p = build_portfolio([a_ko_any, a_ko_r2], strategy="kelly")
    assert len(p["bets"]) == 2


def test_max_bets_per_fight_constant():
    # Sanity: the constant matches user's "5 bets per fight" requirement.
    assert PORTFOLIO_MAX_BETS_PER_FIGHT == 5


def test_bets_sorted_by_stake_descending():
    bets = [
        _bet("Small", 0.55, 2.1, kelly_pct=1.2),
        _bet("Big",   0.65, 2.0, kelly_pct=4.0),
        _bet("Mid",   0.58, 2.2, kelly_pct=2.5),
    ]
    p = build_portfolio(bets, strategy="kelly")
    stakes = [b["stake_pct"] for b in p["bets"]]
    assert stakes == sorted(stakes, reverse=True)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        build_portfolio([_bet("A vs B", 0.6, 2.0, kelly_pct=2.0)], strategy="martingale")


# ---------------------------------------------------------------------------
# Concentrated strategy
# ---------------------------------------------------------------------------

def test_concentrated_keeps_top_n_only():
    # 20 +EV bets, varying Kelly. Concentrated should keep only the top N.
    bets = [
        _bet(f"Fight {i}", 0.55 + 0.005 * i, 2.0, kelly_pct=0.5 + 0.2 * i)
        for i in range(20)
    ]
    p = build_portfolio(bets, strategy="concentrated")
    # Output bets should be at most top_n (some may drop below min_bet_pct).
    assert len(p["bets"]) <= PORTFOLIO_TOP_N
    # The bets that remain should be the highest-Kelly ones from the input.
    kept_kelly = sorted([b["kelly_pct"] for b in p["bets"]], reverse=True)
    all_kelly = sorted([b["kelly_pct"] for b in bets], reverse=True)
    assert kept_kelly == all_kelly[:len(kept_kelly)]


def test_concentrated_yields_larger_per_bet_than_kelly():
    # Same input — concentrated should give meaningfully higher avg stake.
    bets = [
        _bet(f"Fight {i}", 0.55 + 0.003 * i, 2.0, kelly_pct=1.0 + 0.15 * i)
        for i in range(30)
    ]
    pk = build_portfolio(bets, strategy="kelly")
    pc = build_portfolio(bets, strategy="concentrated")
    avg_kelly = sum(b["stake_pct"] for b in pk["bets"]) / max(len(pk["bets"]), 1)
    avg_conc  = sum(b["stake_pct"] for b in pc["bets"]) / max(len(pc["bets"]), 1)
    assert avg_conc > avg_kelly


# ---------------------------------------------------------------------------
# Within-fight conflict modelling
# ---------------------------------------------------------------------------

def _bet_with_keys(fight, our_prob, odds, kelly_pct, outcome_keys, **extra):
    b = _bet(fight, our_prob, odds, kelly_pct, **extra)
    b["outcome_keys"] = list(outcome_keys)
    return b


def test_best_case_overlapping_bets_both_win():
    # Moneyline A (covers any A win) + Method A KO (covers A|KO|*).
    # On scenario A|KO|2 BOTH win — best case sums their payouts.
    ml_keys = {f"A|{m}|{r}" for m in ("KO", "SUB") for r in (1, 2, 3, 4, 5)} | {"A|DEC|DEC"}
    method_keys = {f"A|KO|{r}" for r in (1, 2, 3, 4, 5)}
    bets = [
        _bet_with_keys("A vs B", 0.6, 2.0, kelly_pct=4.0, outcome_keys=ml_keys),
        _bet_with_keys("A vs B", 0.3, 4.0, kelly_pct=3.0, outcome_keys=method_keys),
    ]
    p = build_portfolio(bets, strategy="kelly")
    # In A|KO|1 both win: 4*1 + 3*3 = 4 + 9 = 13. That's the best scenario.
    assert abs(p["summary"]["best_case_pct"] - 13.0) < 0.05


def test_best_case_independent_fights_sum():
    # Bets on two different fights are independent — best case is the sum
    # of each fight's best case.
    bets = [
        _bet_with_keys("A vs B", 0.3, 4.0, kelly_pct=3.0,
                       outcome_keys={f"A|KO|{r}" for r in (1, 2, 3, 4, 5)}),
        _bet_with_keys("C vs D", 0.4, 3.0, kelly_pct=3.0,
                       outcome_keys={f"A|SUB|{r}" for r in (1, 2, 3, 4, 5)}),
    ]
    p = build_portfolio(bets, strategy="kelly")
    # Fight 1: best case = 3 * (4-1) = 9 (no other bet on this fight)
    # Fight 2: best case = 3 * (3-1) = 6
    # Total: 15
    assert abs(p["summary"]["best_case_pct"] - 15.0) < 0.05


def test_best_case_legacy_bets_without_keys_falls_back():
    # Bets with no outcome_keys still work — fall back to optimistic sum.
    bets = [_bet("A vs B", 0.6, 2.0, kelly_pct=4.0)]  # no outcome_keys field
    p = build_portfolio(bets, strategy="kelly")
    # 4% stake × payout_mult=1 = 4%
    assert abs(p["summary"]["best_case_pct"] - 4.0) < 0.05
