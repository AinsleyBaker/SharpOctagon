"""
Expected Value analysis for all available SportsBet markets.

For each fight prediction, computes:
  - EV%       = (our_prob × decimal_odds - 1) × 100
  - Kelly%    = max(0, edge / (odds-1)) × quarter_kelly_fraction × 100
  - Edge      = our_prob - implied_prob
  - is_value  = EV > 0 and edge >= MIN_EDGE_THRESHOLD

Supports moneyline, method of victory, go-the-distance, total rounds, winning round.

Usage:
    from ufc_predict.eval.bet_analysis import analyze_fight_bets
    bets = analyze_fight_bets(prediction_dict)
"""

from __future__ import annotations

import logging
import re

from rapidfuzz import fuzz

log = logging.getLogger(__name__)

KELLY_FRACTION   = 0.25   # quarter-Kelly
MIN_EDGE         = 0.03   # minimum 3% edge to flag as value
MIN_ODDS         = 1.05   # skip near-certainty markets (likely bad parse)
MAX_PROB_SANITY  = 0.995  # clamp our probability here

# Empirical edge → flat-ROI maps from backtest reports. Loaded lazily.
# Two sources:
#   models/edge_backtest.json       — moneyline OOF backtest (1792 bets).
#                                     ALL buckets currently negative — Vegas wins.
#   models/prop_edge_backtest.json  — prop OOF backtest (~20k bets across markets).
#                                     Several markets show positive ROI in mid-edge buckets:
#                                     method, starts_round, inside_distance, wins_round.
_BACKTEST_ML: list[tuple[float, float, float, int]] | None = None
_BACKTEST_PROPS: dict[str, list[tuple[float, float, float, int]]] | None = None

# Bet-type → backtest market_class mapping
_BET_TYPE_TO_MARKET: dict[str, str] = {
    "moneyline":           "moneyline",
    "method":              "method",
    "method_neutral":      "method",      # KO/SUB/DEC neutral — same model bucket
    "method_combo":        "method",
    "distance":            "distance",
    "total_rounds":        "total_rounds",
    "round_survival":      "starts_round",
    "winning_round":       "wins_round",  # primary mapping; ends_round also overlaps
    "alt_finish_timing":   "ends_round",
    "alt_round":           "wins_round",
}

# Threshold for marking a bucket "historically profitable" — guard against
# small-sample noise. n=30 minimum and ROI > 5% as the value gate.
_MIN_BUCKET_N = 30
_MIN_BUCKET_ROI_PCT = 5.0


def _load_backtests() -> tuple[list, dict]:
    """Lazy-load both moneyline and prop backtest results."""
    global _BACKTEST_ML, _BACKTEST_PROPS
    if _BACKTEST_ML is not None and _BACKTEST_PROPS is not None:
        return _BACKTEST_ML, _BACKTEST_PROPS
    import json
    from pathlib import Path

    # Moneyline
    ml_path = Path("models/edge_backtest.json")
    _BACKTEST_ML = []
    if ml_path.exists():
        try:
            data = json.loads(ml_path.read_text(encoding="utf-8"))
            _BACKTEST_ML = [
                (float(b["edge_lo"]), float(b["edge_hi"]),
                 float(b.get("roi_pct", 0)), int(b.get("n", 0)))
                for b in data.get("buckets", []) if int(b.get("n", 0)) > 0
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            _BACKTEST_ML = []

    # Props (per market_class)
    props_path = Path("models/prop_edge_backtest.json")
    _BACKTEST_PROPS = {}
    if props_path.exists():
        try:
            data = json.loads(props_path.read_text(encoding="utf-8"))
            for b in data.get("buckets", []):
                if int(b.get("n", 0)) <= 0:
                    continue
                mc = b.get("market_class", "")
                _BACKTEST_PROPS.setdefault(mc, []).append((
                    float(b["edge_lo"]), float(b["edge_hi"]),
                    float(b.get("flat_roi_pct", 0)), int(b.get("n", 0)),
                ))
        except (json.JSONDecodeError, KeyError, TypeError):
            _BACKTEST_PROPS = {}

    return _BACKTEST_ML, _BACKTEST_PROPS


def _historical_roi_for_edge(edge: float, bet_type: str) -> tuple[float | None, int, bool]:
    """Look up backtested flat ROI% for the bet's edge bucket.

    Returns (roi_pct, n_samples, market_backtested). market_backtested=True
    means we have empirical data for this bet type's market class.
    """
    ml, props = _load_backtests()
    market = _BET_TYPE_TO_MARKET.get(bet_type)
    if market == "moneyline":
        for lo, hi, roi, n in ml:
            if lo <= edge < hi:
                return roi, n, True
        return None, 0, True  # market backtested but bucket has no data
    if market and market in props:
        for lo, hi, roi, n in props[market]:
            if lo <= edge < hi:
                return roi, n, True
        return None, 0, True
    return None, 0, False  # market not backtested


# ---------------------------------------------------------------------------
# Core formulas
# ---------------------------------------------------------------------------

def expected_value(prob: float, decimal_odds: float) -> float:
    """EV as a fraction: 0.07 = +7%. Negative means expected loss."""
    return prob * decimal_odds - 1.0


def kelly(prob: float, decimal_odds: float, fraction: float = KELLY_FRACTION) -> float:
    """Fractional Kelly bet size as fraction of bankroll. Returns 0 if negative EV."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    full_kelly = (b * prob - (1 - prob)) / b
    return max(0.0, full_kelly * fraction)


def implied_prob(decimal_odds: float) -> float:
    if decimal_odds <= 0:
        return 1.0
    return 1.0 / decimal_odds


def _bet_row(
    bet_type: str,
    description: str,
    our_prob: float,
    decimal_odds: float,
    outcome_keys: list[str] | None = None,
) -> dict:
    if decimal_odds < MIN_ODDS or our_prob <= 0:
        return {}
    our_prob = min(our_prob, MAX_PROB_SANITY)
    impl    = implied_prob(decimal_odds)
    edge    = our_prob - impl
    ev      = expected_value(our_prob, decimal_odds)
    kel     = kelly(our_prob, decimal_odds)

    # Per-market historical ROI for this edge bucket. Both moneyline and
    # prop markets are now backtested (Week 6: prop_edge_backtest covers
    # ~20k prop bets across method/distance/starts_round/wins_round/etc).
    hist_roi, hist_n, market_backtested = _historical_roi_for_edge(edge, bet_type)

    # Empirical value gate: only flag is_value=True if the bet's bucket has
    # *backtested* positive ROI > _MIN_BUCKET_ROI_PCT and n_bets ≥ _MIN_BUCKET_N.
    # If the market is backtested but the bucket lacks data, default to False
    # (we have evidence in adjacent buckets — silence is informative).
    # If the market isn't backtested at all, fall back to the edge threshold.
    if market_backtested:
        is_value = (
            hist_roi is not None
            and hist_roi > _MIN_BUCKET_ROI_PCT
            and hist_n >= _MIN_BUCKET_N
            and ev > 0
            and edge >= MIN_EDGE
        )
    else:
        is_value = (ev > 0 and edge >= MIN_EDGE)

    return {
        "bet_type":     bet_type,
        "description":  description,
        "our_prob":     round(our_prob, 4),
        "sb_odds":      round(decimal_odds, 2),
        "implied_prob": round(impl, 4),
        "edge":         round(edge, 4),
        "ev_pct":       round(ev * 100, 2),
        "kelly_pct":    round(kel * 100, 2),
        "is_value":     is_value,
        # Empirical bucket ROI from prop_edge_backtest.json. None means either
        # the bucket has no historical samples or this market isn't backtested.
        "historical_roi_pct": (round(hist_roi, 2) if hist_roi is not None else None),
        "historical_n_bets":  hist_n,
        "market_backtested":  market_backtested,
        # Atomic outcome keys ("A|KO|2", "B|DEC|DEC", …) the bet pays on.
        # Used by build_portfolio() to detect within-fight conflicts so two
        # bets that can't both win don't double-count in best-case.
        "outcome_keys": list(outcome_keys) if outcome_keys else [],
    }


# ---------------------------------------------------------------------------
# Atomic outcome keys: "{winner}|{method}|{round}" where
#   winner ∈ {A, B}; method ∈ {KO, SUB, DEC}; round ∈ {1..5} for finishes,
#   "DEC" for decisions. Draws are ignored (separate market, no fighter
#   attribution). Used to detect within-fight conflicts in best-case calc.
# ---------------------------------------------------------------------------

_ROUNDS = (1, 2, 3, 4, 5)


def _canon_method(m: str) -> str:
    """Normalise a method string to {KO, SUB, DEC}. KO/TKO collapse to KO."""
    m = (m or "").upper()
    if "KO" in m or m == "TKO":
        return "KO"
    if "SUB" in m:
        return "SUB"
    return "DEC"


def _wk_method(side: str, method: str, rounds=None) -> set[str]:
    """Outcome keys for `side` winning by `method` in any of `rounds`.
    DEC ignores rounds (single key); finishes default to all rounds."""
    method = _canon_method(method)
    if method == "DEC":
        return {f"{side}|DEC|DEC"}
    rs = tuple(rounds) if rounds is not None else _ROUNDS
    return {f"{side}|{method}|{r}" for r in rs}


def _wk_winner_all(side: str) -> set[str]:
    """All winning outcomes for `side` (any method, any round)."""
    return _wk_method(side, "KO") | _wk_method(side, "SUB") | _wk_method(side, "DEC")


def _wk_neutral_method(method: str, rounds=None) -> set[str]:
    """Either fighter wins by the given method (in given rounds)."""
    return _wk_method("A", method, rounds) | _wk_method("B", method, rounds)


def _wk_dec_any() -> set[str]:
    """Fight goes the distance (either fighter wins by decision)."""
    return _wk_neutral_method("DEC")


def _wk_finish_any(rounds=None) -> set[str]:
    """Any KO or SUB by either fighter, optionally restricted to `rounds`."""
    return _wk_neutral_method("KO", rounds) | _wk_neutral_method("SUB", rounds)


# ---------------------------------------------------------------------------
# Method market parsing
# ---------------------------------------------------------------------------

_METHOD_RE = re.compile(
    r"(ko|tko|ko/tko|submission|sub|decision|points|dec)",
    re.IGNORECASE,
)


def _classify_method_selection(name: str) -> str | None:
    """
    Map a SportsBet selection name to KO_TKO | SUB | DEC, or None to skip.
    Returns None for Draw entries — they have no fighter attribution and
    DEC would be a wrong label.
    """
    lname = name.lower()
    if lname.strip() in ("draw", "draw."):
        return None
    if "draw" == lname.split()[0]:  # "Draw" as standalone word at start
        return None
    if "ko/tko" in lname or "tko" in lname or " ko" in lname:
        return "KO_TKO"
    if "sub" in lname:
        return "SUB"
    return "DEC"  # "decision", "points", "judges", etc.


def _fighter_side(selection_name: str, name_a: str, name_b: str) -> str | None:
    """
    Determine whether a method selection refers to fighter A or B.
    Returns 'A', 'B', or None if can't determine.
    """
    score_a = max(
        fuzz.token_set_ratio(name_a, selection_name),
        fuzz.partial_ratio(name_a.split()[-1] if name_a else "", selection_name),
    )
    score_b = max(
        fuzz.token_set_ratio(name_b, selection_name),
        fuzz.partial_ratio(name_b.split()[-1] if name_b else "", selection_name),
    )
    if score_a >= 65 and score_a > score_b:
        return "A"
    if score_b >= 65 and score_b > score_a:
        return "B"
    return None


# ---------------------------------------------------------------------------
# Total rounds market parsing
# ---------------------------------------------------------------------------

def _parse_total_rounds_line(
    selection_name: str, is_five_round: bool = False
) -> tuple[str, float] | None:
    """
    Parse 'Over 2.5 Rounds' → ('over', 2.5).
    Bare 'Over'/'Under' (SportsBet omits the line): infer from fight type.
      3-round fights → 2.5  (over = fight reaches R3)
      5-round fights → 2.5  (standard UFC title-fight line)
    """
    m = re.search(r"(over|under)\s+(\d+\.?\d*)", selection_name, re.IGNORECASE)
    if m:
        return m.group(1).lower(), float(m.group(2))
    lname = selection_name.lower().strip()
    if lname in ("over", "under"):
        return lname, 2.5
    return None


def _prob_over_under(
    prob_rounds: dict[str, float], direction: str, line: float, prob_dec: float = 0.0
) -> float:
    """
    Compute P(over/under X.5 rounds).

    prob_rounds values are P(finish in round X) — they sum to prob_finish, NOT 1.
    Decisions are NOT in prob_rounds; pass prob_dec separately.

    "Under X.5" = fight ends (by finish) in round 1..X
    "Over X.5"  = fight lasts past X rounds
                = 1 - P(finish in rounds 1..X)
                which naturally includes decisions (they survive all rounds).
    """
    threshold = int(line + 0.5)  # e.g. 2.5 → 2
    under_prob = sum(v for k, v in prob_rounds.items() if int(k[1:]) <= threshold)
    if direction == "under":
        return under_prob
    return max(0.0, 1.0 - under_prob)  # includes decisions implicitly


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_fight_bets(prediction: dict) -> list[dict]:
    """
    Compute EV + Kelly for every available SportsBet market for one fight.
    Returns list of bet dicts sorted by EV descending, value bets first.
    """
    sb = prediction.get("sportsbet_odds") or {}
    props = prediction.get("props") or {}
    name_a = prediction.get("fighter_a_name") or "Fighter A"
    name_b = prediction.get("fighter_b_name") or "Fighter B"
    prob_a = float(prediction.get("prob_a_wins") or 0.5)
    prob_b = float(prediction.get("prob_b_wins") or (1 - prob_a))
    is_five_round = bool(prediction.get("is_five_round"))

    prob_rounds = props.get("prob_rounds") or {}
    prob_dec    = float(props.get("prob_decision", 0))
    prob_finish = float(props.get("prob_finish", 1 - prob_dec))

    # Neutral method totals (used by method_neutral + method_combo)
    prob_ko_any = (
        float(props.get("prob_a_wins_ko_tko", 0)) + float(props.get("prob_b_wins_ko_tko", 0))
    )
    prob_sub_any = (
        float(props.get("prob_a_wins_sub", 0)) + float(props.get("prob_b_wins_sub", 0))
    )

    bets: list[dict] = []

    def _add(bet_type, label, our_p, odds, outcome_keys=None):
        b = _bet_row(bet_type, label, our_p, odds, outcome_keys=outcome_keys)
        if b:
            bets.append(b)

    # ---- Moneyline --------------------------------------------------------
    for name, prob, key, side in [
        (name_a, prob_a, "moneyline_a", "A"),
        (name_b, prob_b, "moneyline_b", "B"),
    ]:
        odds = sb.get(key)
        if odds and odds > MIN_ODDS:
            _add("moneyline", f"{name} wins", prob, odds, _wk_winner_all(side))

    # ---- Method of victory (fighter-attributed) ---------------------------
    for sel_name, odds in (sb.get("method") or {}).items():
        if odds < MIN_ODDS or not props:
            continue
        method = _classify_method_selection(sel_name)
        if method is None:
            continue  # Draw or unrecognised — skip
        side = _fighter_side(sel_name, name_a, name_b)
        if side == "A":
            our_p = props.get(f"prob_a_wins_{method.lower()}", 0)
            label = f"{name_a} wins by {method.replace('_', '/')}"
        elif side == "B":
            our_p = props.get(f"prob_b_wins_{method.lower()}", 0)
            label = f"{name_b} wins by {method.replace('_', '/')}"
        else:
            continue
        _add("method", label, our_p, odds, _wk_method(side, method))

    # ---- Method neutral ("How fight will End") ----------------------------
    for sel_name, odds in (sb.get("method_neutral") or {}).items():
        if odds < MIN_ODDS or not props:
            continue
        lname = sel_name.lower()
        if "ko" in lname or "tko" in lname:
            _add("method_neutral", "Fight ends by KO/TKO", prob_ko_any, odds,
                 _wk_neutral_method("KO"))
        elif "sub" in lname:
            _add("method_neutral", "Fight ends by Submission", prob_sub_any, odds,
                 _wk_neutral_method("SUB"))
        elif any(w in lname for w in ("point", "dec", "judge")):
            _add("method_neutral", "Fight goes to Decision", prob_dec, odds,
                 _wk_dec_any())

    # ---- Double chance (method combo per fighter) -------------------------
    for sel_name, odds in (sb.get("method_combo") or {}).items():
        if odds < MIN_ODDS or not props:
            continue
        if sel_name.lower().strip() == "draw":
            continue
        side = _fighter_side(sel_name, name_a, name_b)
        if side == "A":
            ko, sub, dec = (float(props.get("prob_a_wins_ko_tko", 0)),
                            float(props.get("prob_a_wins_sub", 0)),
                            float(props.get("prob_a_wins_dec", 0)))
            pfx = name_a
        elif side == "B":
            ko, sub, dec = (float(props.get("prob_b_wins_ko_tko", 0)),
                            float(props.get("prob_b_wins_sub", 0)),
                            float(props.get("prob_b_wins_dec", 0)))
            pfx = name_b
        else:
            continue
        lname = sel_name.lower()
        if "ko" in lname and "sub" in lname:
            _add("method_combo", f"{pfx} wins by KO or Sub", ko + sub, odds,
                 _wk_method(side, "KO") | _wk_method(side, "SUB"))
        elif "ko" in lname and any(w in lname for w in ("point", "dec")):
            _add("method_combo", f"{pfx} wins by KO or Decision", ko + dec, odds,
                 _wk_method(side, "KO") | _wk_method(side, "DEC"))
        elif "sub" in lname and any(w in lname for w in ("point", "dec")):
            _add("method_combo", f"{pfx} wins by Sub or Decision", sub + dec, odds,
                 _wk_method(side, "SUB") | _wk_method(side, "DEC"))

    # ---- Go the distance --------------------------------------------------
    for sel_name, odds in (sb.get("distance") or {}).items():
        if odds < MIN_ODDS or not props:
            continue
        lname = sel_name.lower()
        if "yes" in lname or "go" in lname:
            _add("distance", "Fight goes to decision", prob_dec, odds, _wk_dec_any())
        elif "no" in lname or "not" in lname:
            _add("distance", "Fight ends before decision", prob_finish, odds, _wk_finish_any())

    # ---- Total rounds (over/under line) ----------------------------------
    for sel_name, odds in (sb.get("total_rounds") or {}).items():
        if odds < MIN_ODDS or not prob_rounds:
            continue
        parsed = _parse_total_rounds_line(sel_name, is_five_round)
        if parsed is None:
            continue
        direction, line = parsed
        our_p = _prob_over_under(prob_rounds, direction, line, prob_dec)
        label = f"{'Over' if direction == 'over' else 'Under'} {line} rounds"
        # "Over X.5" → finishes in (X+1)..5 OR decision; "Under X.5" → finishes in 1..X.
        threshold = int(line + 0.5)  # 2.5 → 2
        if direction == "over":
            keys = _wk_finish_any(rounds=range(threshold + 1, 6)) | _wk_dec_any()
        else:
            keys = _wk_finish_any(rounds=range(1, threshold + 1))
        _add("total_rounds", label, our_p, odds, keys)

    # ---- Round survival ("Fight To Start Round X") -----------------------
    for sel_name, odds in (sb.get("round_survival") or {}).items():
        if odds < MIN_ODDS or not prob_rounds:
            continue
        # Key format: "Start R2 Yes" / "Start R2 No"
        m = re.search(r"Start R(\d)", sel_name)
        if not m:
            continue
        rnum = int(m.group(1))
        # P(fight starts round X) = 1 - P(finish in any earlier round)
        # Decisions always survive to their max round, so they're included in "Yes".
        p_early_finish = sum(prob_rounds.get(f"R{i}", 0) for i in range(1, rnum))
        p_survives = max(0.0, 1.0 - p_early_finish)
        if "yes" in sel_name.lower():
            # Reaches round N = finishes in N..5 OR goes to decision
            keys = _wk_finish_any(rounds=range(rnum, 6)) | _wk_dec_any()
            _add("round_survival", f"Fight reaches Round {rnum}", p_survives, odds, keys)
        elif "no" in sel_name.lower():
            keys = _wk_finish_any(rounds=range(1, rnum))
            _add("round_survival", f"Finish before Round {rnum}", p_early_finish, odds, keys)

    # ---- Alt finish timing ("Round 1 or 2" / "Round 3 or Decision") -----
    for sel_name, odds in (sb.get("alt_finish_timing") or {}).items():
        if odds < MIN_ODDS or not prob_rounds:
            continue
        lname = sel_name.lower()
        if "1 or 2" in lname:
            # Only finishes end here — no decisions
            our_p = prob_rounds.get("R1", 0) + prob_rounds.get("R2", 0)
            _add("alt_finish_timing", "Ends in Round 1 or 2", our_p, odds,
                 _wk_finish_any(rounds=(1, 2)))
        elif "3" in lname or "dec" in lname:
            # Finishes in R3+ PLUS all decisions (they definitely go to R3)
            our_p = sum(prob_rounds.get(f"R{i}", 0) for i in range(3, 6)) + prob_dec
            _add("alt_finish_timing", "Ends in Round 3+ / Decision", our_p, odds,
                 _wk_finish_any(rounds=(3, 4, 5)) | _wk_dec_any())

    # ---- Alt round betting (fighter + timing combo) ----------------------
    for sel_name, odds in (sb.get("alt_round") or {}).items():
        if odds < MIN_ODDS or not prob_rounds:
            continue
        side = _fighter_side(sel_name, name_a, name_b)
        if side == "A":
            winner_p, pfx = prob_a, name_a
        elif side == "B":
            winner_p, pfx = prob_b, name_b
        else:
            continue
        lname = sel_name.lower()
        if "1 or 2" in lname:
            timing_p = prob_rounds.get("R1", 0) + prob_rounds.get("R2", 0)
            _add("alt_round", f"{pfx} wins in R1 or R2",
                 winner_p * timing_p, odds,
                 _wk_method(side, "KO", (1, 2)) | _wk_method(side, "SUB", (1, 2)))
        elif "3" in lname or "dec" in lname:
            timing_p = sum(prob_rounds.get(f"R{i}", 0) for i in range(3, 6))
            _add("alt_round", f"{pfx} wins in R3 or by Decision",
                 winner_p * timing_p, odds,
                 _wk_method(side, "KO", (3, 4, 5))
                 | _wk_method(side, "SUB", (3, 4, 5))
                 | _wk_method(side, "DEC"))

    # ---- Winning round ---------------------------------------------------
    # Covers three selection types from SportsBet:
    #   "Fighter A to Win in Round N"   → fighter-specific finish
    #   "Round N"                       → neutral finish in that round
    #   "Fighter A to Win by Decision"  → fighter wins by decision
    # Also present but explicitly skipped:
    #   "X Minute of Round N"           → too granular, model can't predict
    #   "Draw"                          → no fighter attribution
    for sel_name, odds in (sb.get("winning_round") or {}).items():
        if odds < MIN_ODDS:
            continue
        lname = sel_name.lower()

        # Skip sub-minute entries ("Winning Round & Minute" market bleeds in)
        if "minute" in lname:
            continue
        # Skip draws
        if lname.strip() == "draw":
            continue

        # "Fighter A to Win by Decision" / "to Win by Points"
        if ("decision" in lname or "points" in lname) and "round" not in lname:
            if not props:
                continue
            side = _fighter_side(sel_name, name_a, name_b)
            if side == "A":
                _add("winning_round", f"{name_a} wins by Decision",
                     float(props.get("prob_a_wins_dec", 0)), odds, _wk_method("A", "DEC"))
            elif side == "B":
                _add("winning_round", f"{name_b} wins by Decision",
                     float(props.get("prob_b_wins_dec", 0)), odds, _wk_method("B", "DEC"))
            continue

        # "Fighter A Round X or by Decision" (alt_round entries that bled into winning_round)
        if ("or by decision" in lname or "or by dec" in lname) and prob_rounds:
            side = _fighter_side(sel_name, name_a, name_b)
            m_rn = re.search(r"round\s+(\d)", sel_name, re.IGNORECASE)
            rnum = int(m_rn.group(1)) if m_rn else None
            if side and rnum and props:
                winner_p = prob_a if side == "A" else prob_b
                pfx      = name_a if side == "A" else name_b
                timing_p = sum(prob_rounds.get(f"R{i}", 0) for i in range(rnum, 6))
                our_p    = winner_p * timing_p + (prob_a if side == "A" else prob_b) * prob_dec
                keys = (
                    _wk_method(side, "KO", range(rnum, 6))
                    | _wk_method(side, "SUB", range(rnum, 6))
                    | _wk_method(side, "DEC")
                )
                _add("winning_round", f"{pfx} wins in R{rnum}+ or by Decision",
                     our_p, odds, keys)
            continue

        # Standard: "Fighter A to Win in Round N" or neutral "Round N"
        m = re.search(r"round\s+(\d)", sel_name, re.IGNORECASE)
        if not m or not prob_rounds:
            continue
        rnum = int(m.group(1))
        rkey = f"R{rnum}"
        p_finish_rn = prob_rounds.get(rkey, 0)

        side = _fighter_side(sel_name, name_a, name_b)
        if side == "A":
            cond  = p_finish_rn / prob_finish if prob_finish > 0.01 else 0
            our_p = prob_a * cond
            _add("winning_round", f"{name_a} wins by finish in Round {rnum}",
                 our_p, odds,
                 _wk_method("A", "KO", (rnum,)) | _wk_method("A", "SUB", (rnum,)))
        elif side == "B":
            cond  = p_finish_rn / prob_finish if prob_finish > 0.01 else 0
            our_p = prob_b * cond
            _add("winning_round", f"{name_b} wins by finish in Round {rnum}",
                 our_p, odds,
                 _wk_method("B", "KO", (rnum,)) | _wk_method("B", "SUB", (rnum,)))
        else:
            _add("winning_round", f"Fight ends by finish in Round {rnum}",
                 p_finish_rn, odds, _wk_finish_any(rounds=(rnum,)))

    # Deduplicate by outcome identity, not description. The same atomic outcome
    # (e.g. "fight goes to decision" → {A|DEC|DEC, B|DEC|DEC}) appears in
    # multiple SportsBet markets at slightly different prices — distance.Yes
    # vs method_neutral.Points, or method.X-by-DEC vs winning_round.X-by-Decision.
    # Bets without outcome_keys (legacy markets) fall back to description so we
    # don't accidentally collapse unrelated legacy bets.
    #
    # Within a duplicate group: keep the best-priced (highest-EV) version, but
    # inherit the value flag and backtest provenance from any source that *was*
    # value-flagged. The is_value gate uses per-market historical ROI; if the
    # backtested market says "this outcome class is profitable" we trust that
    # signal even when the user is grabbing a higher price from a non-backtested
    # market for the identical outcome.
    groups: dict = {}
    for bet in bets:
        keys = bet.get("outcome_keys") or []
        key = frozenset(keys) if keys else ("_legacy", bet.get("description", ""))
        groups.setdefault(key, []).append(bet)

    unique_bets: list[dict] = []
    for group in groups.values():
        best = max(group, key=lambda b: b.get("ev_pct", 0))
        valued = next((b for b in group if b.get("is_value")), None)
        if valued and not best.get("is_value"):
            best = {
                **best,
                "is_value": True,
                "historical_roi_pct": valued.get("historical_roi_pct"),
                "historical_n_bets":  valued.get("historical_n_bets"),
                "market_backtested":  valued.get("market_backtested"),
            }
        unique_bets.append(best)

    # Sort: value bets first, then by EV descending
    unique_bets.sort(key=lambda x: (-int(x.get("is_value", False)), -x.get("ev_pct", 0)))
    return unique_bets


def analyze_all_fights(predictions: list[dict]) -> list[dict]:
    """Add bet_analysis list to every prediction dict. Returns predictions."""
    for pred in predictions:
        pred["bet_analysis"] = analyze_fight_bets(pred)
    return predictions


def top_value_bets(predictions: list[dict], n: int = 20) -> list[dict]:
    """
    Return the top-N value bets across all fights, sorted by EV descending.
    Each item includes the fight context.
    """
    all_bets = []
    for pred in predictions:
        for bet in pred.get("bet_analysis") or []:
            if bet.get("is_value"):
                all_bets.append({
                    **bet,
                    "fight": f"{pred.get('fighter_a_name')} vs {pred.get('fighter_b_name')}",
                    "event": pred.get("event_name"),
                    "event_date": pred.get("event_date"),
                })
    all_bets.sort(key=lambda x: -x.get("ev_pct", 0))
    return all_bets[:n]


# ---------------------------------------------------------------------------
# Portfolio allocation: Kelly vs flat-stake, with correlation/risk caps
# ---------------------------------------------------------------------------

# A single bet on a single fight max
PORTFOLIO_MAX_PER_BET_PCT   = 5.0
# Sum of all stakes on the same bout (handles correlated legs — e.g. method
# KO + winning round 1 of the same fighter overlap heavily).
PORTFOLIO_MAX_PER_FIGHT_PCT = 15.0
# Sum across the whole portfolio. 50% keeps a sane reserve even if the entire
# card fails — half-Kelly already assumes our probs are calibrated, so this
# is a belt-and-braces guard against model error.
PORTFOLIO_MAX_TOTAL_PCT     = 50.0
# Drop bets that round to less than this — saves the user from $0.30 stakes
# on a $100 bankroll that aren't actually placeable.
PORTFOLIO_MIN_BET_PCT       = 1.0
# How many bets the "concentrated" strategy keeps. Picked to give 5–10
# actionable stakes ($3–5 each on a $100 bankroll) instead of 30+ tiny ones.
PORTFOLIO_TOP_N             = 8
# Max legs per fight after conflict filtering. Without this cap a single
# bout can hog 5+ correlated lines (KO method + winning round + inside
# distance + …) which clutters the recommendation list.
PORTFOLIO_MAX_BETS_PER_FIGHT = 5


def _filter_fight_bets(
    fight_bets: list[tuple[int, dict, float]],
    max_count: int,
) -> set[int]:
    """Pick at most `max_count` bets per fight that can all win in some
    common outcome. Returns the set of indices to keep.

    Bets are ranked by stake desc; greedily kept while compatible with the
    intersection of already-kept outcome_keys. A bet with empty outcome_keys
    (legacy / moneyline-only) is treated as unconstrained — kept without
    shrinking the compat set.
    """
    sorted_bets = sorted(fight_bets, key=lambda t: -t[2])
    kept: set[int] = set()
    compat: set[str] | None = None
    for idx, bet, _stake in sorted_bets:
        if len(kept) >= max_count:
            break
        keys = set(bet.get("outcome_keys") or [])
        if not keys:
            kept.add(idx)
            continue
        if compat is None:
            kept.add(idx)
            compat = keys
        elif compat & keys:
            kept.add(idx)
            compat = compat & keys
        # else: incompatible with already-kept legs — drop
    return kept


def _fight_best_case_pct(fight_bets: list[dict]) -> float:
    """Best achievable net pct for one bout's bets, accounting for the fact
    that mutually-exclusive bets (e.g. Method KO + Method SUB on the same
    fighter) cannot both win.

    For each atomic outcome the bet set covers, sum the wins minus the losses
    in that outcome, then take the max. If a bet has no outcome_keys (legacy
    or unsupported market), conservatively treat it as winning in every
    scenario the other bets cover — this collapses to the old "sum all
    payouts" behavior, which is the right upper bound when we don't know
    which scenarios it actually pays on.
    """
    if not fight_bets:
        return 0.0
    universe: set[str] = set()
    for b in fight_bets:
        keys = b.get("outcome_keys")
        if keys:
            universe.update(keys)
    if not universe:
        # No outcome data anywhere — fall back to the optimistic upper bound.
        return sum(b["stake_pct"] * b["payout_mult"] for b in fight_bets)

    best = -float("inf")
    for omega in universe:
        net = 0.0
        for b in fight_bets:
            keys = b.get("outcome_keys") or []
            wins = (omega in keys) if keys else True   # legacy bets: always win
            if wins:
                net += b["stake_pct"] * b["payout_mult"]
            else:
                net -= b["stake_pct"]
        if net > best:
            best = net
    return best


def build_portfolio(
    bets: list[dict],
    strategy: str = "kelly",
    max_per_bet_pct: float = PORTFOLIO_MAX_PER_BET_PCT,
    max_per_fight_pct: float = PORTFOLIO_MAX_PER_FIGHT_PCT,
    max_total_pct: float = PORTFOLIO_MAX_TOTAL_PCT,
    min_bet_pct: float = PORTFOLIO_MIN_BET_PCT,
    top_n: int = PORTFOLIO_TOP_N,
    max_bets_per_fight: int = PORTFOLIO_MAX_BETS_PER_FIGHT,
) -> dict:
    """Allocate bankroll across value bets.

    Strategies:
      - "kelly":        fractional-Kelly stakes (each bet's pre-computed
                        kelly_pct, already 1/4 Kelly), then per-bet/fight/total
                        caps applied.
      - "flat":         equal stakes across all qualifying bets, same caps.
      - "concentrated": top-N Kelly-ranked bets only, each at its full Kelly
                        stake. Yields fewer, larger, more actionable stakes —
                        designed for small bankrolls where 30 tiny bets aren't
                        practical at most sportsbooks.

    Each bet must contain: 'fight' (correlation key), 'our_prob', 'sb_odds',
    'kelly_pct'. Other descriptive fields pass through unchanged.

    Returns:
      {
        "strategy": str,
        "bets":     [bet + stake_pct sorted desc by stake],
        "summary":  {total_stake_pct, expected_pnl_pct, best_case_pct,
                     worst_case_pct, n_bets, n_fights},
      }
    """
    empty = {
        "strategy": strategy,
        "bets":     [],
        "summary": {
            "total_stake_pct":  0.0,
            "expected_pnl_pct": 0.0,
            "best_case_pct":    0.0,
            "worst_case_pct":   0.0,
            "n_bets":           0,
            "n_fights":         0,
        },
    }
    if not bets:
        return empty

    # 1) Initial stake per bet
    if strategy == "kelly":
        stakes = [float(b.get("kelly_pct") or 0) for b in bets]
    elif strategy == "flat":
        equal = max_total_pct / len(bets)
        stakes = [equal] * len(bets)
    elif strategy == "concentrated":
        # Rank bets by raw Kelly stake; keep the top N, zero the rest.
        # The downstream caps then apply only to this concentrated subset.
        kelly_pcts = [float(b.get("kelly_pct") or 0) for b in bets]
        order = sorted(range(len(bets)), key=lambda i: -kelly_pcts[i])
        keep = set(order[:top_n])
        stakes = [kelly_pcts[i] if i in keep else 0.0 for i in range(len(bets))]
    else:
        raise ValueError(f"Unknown portfolio strategy: {strategy}")

    # 2) Per-bet cap
    stakes = [min(s, max_per_bet_pct) for s in stakes]

    # 2b) Per-fight conflict filter: keep at most `max_bets_per_fight` legs
    # whose outcome_keys can all win in some common outcome. Drops e.g.
    # "KO round 1" alongside "fight goes to decision" on the same bout.
    fight_keys_pre = [b.get("fight") or f"_unk_{i}" for i, b in enumerate(bets)]
    by_fight_idx: dict[str, list[tuple[int, dict, float]]] = {}
    for i, (b, s) in enumerate(zip(bets, stakes)):
        if s <= 0:
            continue
        by_fight_idx.setdefault(fight_keys_pre[i], []).append((i, b, s))
    keep_idx: set[int] = set()
    for fight_legs in by_fight_idx.values():
        keep_idx |= _filter_fight_bets(fight_legs, max_bets_per_fight)
    stakes = [s if i in keep_idx or s == 0 else 0.0 for i, s in enumerate(stakes)]

    # 3) Per-fight scaling — correlated legs on the same bout share a budget.
    fight_keys = [b.get("fight") or f"_unk_{i}" for i, b in enumerate(bets)]
    fight_totals: dict[str, float] = {}
    for fk, s in zip(fight_keys, stakes):
        fight_totals[fk] = fight_totals.get(fk, 0.0) + s
    for fk, total in fight_totals.items():
        if total > max_per_fight_pct and total > 0:
            scale = max_per_fight_pct / total
            stakes = [
                (s * scale if fight_keys[i] == fk else s)
                for i, s in enumerate(stakes)
            ]

    # 4) Total cap
    grand_total = sum(stakes)
    if grand_total > max_total_pct and grand_total > 0:
        scale = max_total_pct / grand_total
        stakes = [s * scale for s in stakes]

    # 5) Drop tiny stakes & build output
    out_bets: list[dict] = []
    for bet, stake in zip(bets, stakes):
        if stake < min_bet_pct:
            continue
        b = dict(bet)
        b["stake_pct"]   = round(stake, 2)
        b["ev_per_unit"] = float(bet.get("our_prob", 0)) * float(bet.get("sb_odds", 1)) - 1
        b["payout_mult"] = float(bet.get("sb_odds", 1)) - 1   # net multiple if win
        out_bets.append(b)

    if not out_bets:
        return empty

    out_bets.sort(key=lambda b: -b["stake_pct"])

    total_pct    = sum(b["stake_pct"] for b in out_bets)
    expected_pnl = sum(b["stake_pct"] * b["ev_per_unit"] for b in out_bets)
    # Best case = sum across fights of the per-fight max-payoff scenario.
    # Mutually-exclusive bets on the same bout (e.g. Method KO + Method SUB
    # for the same fighter) can never both hit — the per-fight calc enumerates
    # atomic outcomes (winner × method × round) and picks the one that
    # maximises net pct. Across fights we sum because outcomes are independent.
    by_fight: dict[str, list[dict]] = {}
    for b in out_bets:
        by_fight.setdefault(b.get("fight") or "_", []).append(b)
    best_case  = sum(_fight_best_case_pct(fb) for fb in by_fight.values())
    worst_case = -total_pct
    n_fights   = len(by_fight)

    return {
        "strategy": strategy,
        "bets":     out_bets,
        "summary": {
            "total_stake_pct":  round(total_pct, 1),
            "expected_pnl_pct": round(expected_pnl, 2),
            "best_case_pct":    round(best_case, 1),
            "worst_case_pct":   round(worst_case, 1),
            "n_bets":           len(out_bets),
            "n_fights":         n_fights,
        },
    }
