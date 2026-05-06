"""Tests for the totals model, method×round joint, round-survival pairs,
match insights, and the dashboard grader for totals bets.

Run with::

    pytest -q tests/test_round_totals_insights.py
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ufc_predict.eval import insights as insights_mod
from ufc_predict.eval.bet_analysis import (
    analyze_all_fights,
    analyze_fight_bets,
    compute_method_round_joint,
)
from ufc_predict.models.totals_models import (
    QUANTILES,
    _row_monotonise,
    prob_over,
)
from ufc_predict.serve.build_dashboard import _grade_bet

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_prediction.json"


@pytest.fixture
def sample_prediction():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# A. Unit tests
# ---------------------------------------------------------------------------

def test_prob_over_monotonic_in_line():
    q = {"q10": 80, "q25": 110, "q50": 140, "q75": 175, "q90": 215}
    p_low = prob_over(50, q)
    p_med = prob_over(140, q)
    p_high = prob_over(300, q)
    assert p_low > p_med > p_high


def test_prob_over_extreme_lines():
    q = {"q10": 80, "q25": 110, "q50": 140, "q75": 175, "q90": 215}
    assert prob_over(0, q) >= 0.95
    assert prob_over(10_000, q) <= 0.05


def test_predict_totals_quantile_monotonicity():
    # The row-monotonise pass turns any 5-tuple into a non-decreasing one.
    raw = np.array([[120, 110, 130, 140, 200], [50, 60, 70, 65, 80]], dtype=float)
    mono = _row_monotonise(raw)
    diffs = np.diff(mono, axis=1)
    assert (diffs >= -1e-9).all(), "monotonisation failed"
    # Median preserved when only tails violate
    assert mono[0, 2] == raw[0, 2]


def test_method_round_joint_normalises():
    props = {
        "prob_a_wins_ko_tko": 0.20, "prob_a_wins_sub": 0.10, "prob_a_wins_dec": 0.30,
        "prob_b_wins_ko_tko": 0.10, "prob_b_wins_sub": 0.05, "prob_b_wins_dec": 0.25,
        "prob_finish": 0.45, "prob_decision": 0.55,
        "prob_rounds": {"R1": 0.18, "R2": 0.13, "R3": 0.14, "R4": 0.0, "R5": 0.0},
    }
    joint = compute_method_round_joint(props, is_five_round=False)
    s_ako = sum(v for k, v in joint.items() if k.startswith("A_KO_"))
    s_asub = sum(v for k, v in joint.items() if k.startswith("A_SUB_"))
    s_bko = sum(v for k, v in joint.items() if k.startswith("B_KO_"))
    s_bsub = sum(v for k, v in joint.items() if k.startswith("B_SUB_"))
    assert s_ako == pytest.approx(props["prob_a_wins_ko_tko"], abs=1e-6)
    assert s_asub == pytest.approx(props["prob_a_wins_sub"], abs=1e-6)
    assert s_bko == pytest.approx(props["prob_b_wins_ko_tko"], abs=1e-6)
    assert s_bsub == pytest.approx(props["prob_b_wins_sub"], abs=1e-6)


def test_grader_totals():
    bet_over = {"outcome_keys": ["TOTAL|total_sig_strikes_combined|over|165.5"]}
    bet_under = {"outcome_keys": ["TOTAL|total_sig_strikes_combined|under|165.5"]}
    actual = {"total_sig_strikes_combined": 180.0}
    assert _grade_bet(bet_over, None, "a", actual) == "won"
    assert _grade_bet(bet_under, None, "a", actual) == "lost"
    actual_low = {"total_sig_strikes_combined": 100.0}
    assert _grade_bet(bet_over, None, "a", actual_low) == "lost"
    assert _grade_bet(bet_under, None, "a", actual_low) == "won"


def test_grader_side_only_fallback():
    """The pre-existing side-only fallback must keep working when method
    is unknown. moneyline-style bet for A is graded ✓ when A wins, ✗ for B."""
    bet_a = {"outcome_keys": [
        "A|KO|1", "A|KO|2", "A|KO|3", "A|KO|4", "A|KO|5",
        "A|SUB|1", "A|SUB|2", "A|SUB|3", "A|SUB|4", "A|SUB|5", "A|DEC|DEC",
    ]}
    bet_b = {"outcome_keys": [
        "B|KO|1", "B|KO|2", "B|KO|3", "B|DEC|DEC",
    ]}
    assert _grade_bet(bet_a, None, "a") == "won"
    assert _grade_bet(bet_b, None, "a") == "lost"


def test_insights_shape(sample_prediction):
    ins = insights_mod.generate_insights(sample_prediction)
    assert isinstance(ins, dict)
    assert "top_factors" in ins and 1 <= len(ins["top_factors"]) <= 3
    assert "stat_bars" in ins and 4 <= len(ins["stat_bars"]) <= 6
    assert "confidence_drivers" in ins and 2 <= len(ins["confidence_drivers"]) <= 3
    for f in ins["top_factors"]:
        assert {"label", "summary", "magnitude", "direction"}.issubset(f.keys())
    for b in ins["stat_bars"]:
        assert {"label", "a_value", "b_value", "advantage_pct", "advantage_side"}.issubset(b.keys())


def test_insights_no_nan_leak(sample_prediction):
    """No top_factor summary or driver string should contain a stand-alone
    'nan'/'NaN'/'None' token."""
    leak_re = re.compile(r"\b(nan|NaN|None)\b")
    ins = insights_mod.generate_insights(sample_prediction)
    for f in ins["top_factors"]:
        assert not leak_re.search(f["summary"]), f"leak in {f['summary']!r}"
        assert not leak_re.search(f["label"]), f"leak in label {f['label']!r}"
    for d in ins["confidence_drivers"]:
        assert not leak_re.search(d), f"leak in driver {d!r}"


def test_outcome_keys_coverage(sample_prediction):
    """Every emitted bet must carry a non-empty outcome_keys list (totals
    use the TOTAL| sentinel)."""
    bets = analyze_fight_bets(sample_prediction)
    assert bets, "fixture should produce at least one bet"
    for bet in bets:
        keys = bet.get("outcome_keys")
        assert keys, f"bet has empty outcome_keys: {bet.get('description')}"


def test_outcome_keys_totals_use_sentinel(sample_prediction):
    bets = analyze_fight_bets(sample_prediction)
    totals_bets = [b for b in bets if b["bet_type"].startswith("total_")]
    assert totals_bets, "fixture should produce totals bets"
    for bet in totals_bets:
        for key in bet["outcome_keys"]:
            assert key.startswith("TOTAL|"), f"non-sentinel totals key: {key}"


def test_scraper_parses_method_round_markets():
    from ufc_predict.ingest.sportsbet_scraper import _parse_markets

    raw = [
        {"name": "Method & Round Combo (5 Rounds)", "selections": [
            {"name": "Ilia Topuria KO/TKO & Round 1", "price": {"winPrice": 3.9}},
            {"name": "Ilia Topuria Submission & Round 3", "price": {"winPrice": 26.0}},
            {"name": "Justin Gaethje KO/TKO & Round 5", "price": {"winPrice": 46.0}},
        ]},
        {"name": "Alt. Method & Round Combo (5 Rounds)", "selections": [
            {"name": "KO/TKO & Round 1", "price": {"winPrice": 3.5}},
            {"name": "Submission & Round 5", "price": {"winPrice": 41.0}},
        ]},
        {"name": "KO/TKO Round Combos (5 Rounds)", "selections": [
            {"name": "Ilia Topuria to win by KO/TKO in Rounds 1 or 2", "price": {"winPrice": 2.35}},
            {"name": "Ilia Topuria to win by KO/TKO in Rounds 1,2 or 3",
             "price": {"winPrice": 1.86}},
        ]},
        {"name": "Submission Round Combos (5 Rounds)", "selections": [
            {"name": "Ilia Topuria to win by Submission in Rounds 2 or 3",
             "price": {"winPrice": 11.5}},
        ]},
    ]
    parsed = _parse_markets(raw, "Ilia Topuria", "Justin Gaethje")
    assert len(parsed["method_round_fighter"]) == 3
    assert len(parsed["method_round_neutral"]) == 2
    assert len(parsed["method_round_ranges"]) == 3
    # Side resolution
    fighters = parsed["method_round_fighter"]
    sides = {(f["side"], f["method"], f["round"]) for f in fighters}
    assert ("A", "KO", 1) in sides
    assert ("A", "SUB", 3) in sides
    assert ("B", "KO", 5) in sides
    # Range parsing
    ranges = {(r["side"], r["method"], tuple(r["rounds"])) for r in parsed["method_round_ranges"]}
    assert ("A", "KO", (1, 2)) in ranges
    assert ("A", "KO", (1, 2, 3)) in ranges
    assert ("A", "SUB", (2, 3)) in ranges


def test_method_round_neutral_market_class_route():
    """method_round_neutral bets must route to the 'ends_round' market class
    so the empirical-ROI gate uses the correct backtest data."""
    from ufc_predict.eval.bet_analysis import _BET_TYPE_TO_MARKET
    assert _BET_TYPE_TO_MARKET["method_round_neutral"] == "ends_round"
    assert _BET_TYPE_TO_MARKET["method_round"] == "wins_round"
    assert _BET_TYPE_TO_MARKET["method_round_range"] == "wins_round"


def test_method_round_priced_emission_outcome_keys():
    """A priced fighter-attributed method×round bet must produce outcome_keys
    that match the equivalent winning_round bet — guaranteeing dedup collapses
    duplicates correctly."""
    from ufc_predict.eval.bet_analysis import analyze_fight_bets

    pred = {
        "fighter_a_name": "Topuria", "fighter_b_name": "Gaethje",
        "prob_a_wins": 0.7, "prob_b_wins": 0.3, "is_five_round": True,
        "props": {
            "prob_a_wins_ko_tko": 0.30, "prob_a_wins_sub": 0.05, "prob_a_wins_dec": 0.35,
            "prob_b_wins_ko_tko": 0.10, "prob_b_wins_sub": 0.02, "prob_b_wins_dec": 0.18,
            "prob_finish": 0.47, "prob_decision": 0.53,
            "prob_rounds": {"R1": 0.16, "R2": 0.12, "R3": 0.10, "R4": 0.05, "R5": 0.04},
        },
        "sportsbet_odds": {
            "moneyline_a": 1.45, "moneyline_b": 2.85,
            "method": {}, "method_neutral": {}, "method_combo": {},
            "distance": {}, "total_rounds": {}, "winning_round": {},
            "round_survival": {}, "alt_finish_timing": {}, "alt_round": {},
            "method_round_fighter": [
                {"side": "A", "method": "KO", "round": 2, "odds": 5.4},
            ],
            "method_round_neutral": [{"method": "KO", "round": 2, "odds": 4.8}],
            "method_round_ranges":  [
                {"side": "A", "method": "KO", "rounds": [1, 2], "odds": 2.35},
            ],
            "total_sig_strikes_combined": [], "total_sig_strikes_a": [],
            "total_sig_strikes_b": [], "total_takedowns_combined": [],
            "total_takedowns_a": [], "total_takedowns_b": [],
            "total_knockdowns_combined": [],
        },
    }
    bets = analyze_fight_bets(pred)
    by_type = {b["bet_type"]: b for b in bets if b["bet_type"].startswith("method_round")}
    # Single-round fighter-attributed bet → exactly the atomic key for that leg
    assert "method_round" in by_type
    assert set(by_type["method_round"]["outcome_keys"]) == {"A|KO|2"}
    # Multi-round range covers every leg in the range
    assert "method_round_range" in by_type
    assert set(by_type["method_round_range"]["outcome_keys"]) == {"A|KO|1", "A|KO|2"}
    # Neutral covers both fighters in the round
    assert "method_round_neutral" in by_type
    assert set(by_type["method_round_neutral"]["outcome_keys"]) == {"A|KO|2", "B|KO|2"}


def test_method_round_zero_finish_safe():
    """compute_method_round_joint must not divide by zero when prob_finish=0."""
    props = {
        "prob_a_wins_ko_tko": 0, "prob_a_wins_sub": 0, "prob_a_wins_dec": 0.5,
        "prob_b_wins_ko_tko": 0, "prob_b_wins_sub": 0, "prob_b_wins_dec": 0.5,
        "prob_finish": 0.0, "prob_decision": 1.0,
        "prob_rounds": {"R1": 0, "R2": 0, "R3": 0, "R4": 0, "R5": 0},
    }
    out = compute_method_round_joint(props, is_five_round=False)
    assert out == {} or all(v == 0 for v in out.values())


# ---------------------------------------------------------------------------
# B. Integration tests
# ---------------------------------------------------------------------------

def test_data_audit_passes():
    """Run the data_audit module and assert no FAIL outcomes. WARN is OK."""
    from ufc_predict.eval.data_audit import run as run_audit
    rep = run_audit()
    fails = [r for r in rep.results if r["status"] == "FAIL"]
    assert not fails, f"audit FAILs: {fails}"


def test_predict_then_dashboard(sample_prediction):
    """Pipe a single prediction through analyze_all_fights → attach_insights →
    confirm the full bet/insight surface is populated, and that grading
    works for both totals and method bets."""
    pred = copy.deepcopy(sample_prediction)
    preds = analyze_all_fights([pred])
    insights_mod.attach_insights(preds)
    out = preds[0]

    assert "bet_analysis" in out and out["bet_analysis"]
    assert "insights" in out
    assert "top_factors" in out["insights"]
    assert any(b["bet_type"].startswith("total_") for b in out["bet_analysis"])

    # Grading: a fight that ends with combined sig strikes 200 (over the 165.5 line)
    # should grade the over bet as 'won' and the under bet as 'lost'.
    actual_totals = {
        "total_sig_strikes_combined": 200.0,
        "total_takedowns_combined": 0.5,
        "total_knockdowns_combined": 0.0,
    }
    over_bets = [
        b for b in out["bet_analysis"]
        if any("|over|" in k for k in (b.get("outcome_keys") or []))
        and any("total_sig_strikes_combined" in k for k in (b.get("outcome_keys") or []))
    ]
    under_bets = [
        b for b in out["bet_analysis"]
        if any("|under|" in k for k in (b.get("outcome_keys") or []))
        and any("total_sig_strikes_combined" in k for k in (b.get("outcome_keys") or []))
    ]
    assert over_bets and under_bets
    assert _grade_bet(over_bets[0], None, "a", actual_totals) == "won"
    assert _grade_bet(under_bets[0], None, "a", actual_totals) == "lost"


def test_full_pipeline_smoke(tmp_path):
    """Build a tiny synthetic feature matrix, train one totals quantile
    model on it, predict, ensure no exceptions and shapes are right."""
    import lightgbm as lgb

    from ufc_predict.models.totals_models import _LGBM_TOTALS_PARAMS

    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
    })
    y = (5 * X["f1"] + 2 * X["f2"] + rng.normal(size=n) * 1.5).values

    qs = []
    for alpha in QUANTILES:
        m = lgb.LGBMRegressor(**{**_LGBM_TOTALS_PARAMS, "alpha": alpha,
                                 "n_estimators": 60})
        m.fit(X, y, callbacks=[lgb.log_evaluation(period=-1)])
        qs.append(m.predict(X))
    matrix = _row_monotonise(np.column_stack(qs))
    diffs = np.diff(matrix, axis=1)
    assert (diffs >= -1e-9).all()
    # Sanity: prob_over should produce values in (0, 1) for a real row
    sample = {f"q{int(a*100):02d}": matrix[0, i] for i, a in enumerate(QUANTILES)}
    p = prob_over(float(matrix[0, 2]), sample)
    assert 0 < p < 1
