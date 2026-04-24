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
) -> dict:
    if decimal_odds < MIN_ODDS or our_prob <= 0:
        return {}
    our_prob = min(our_prob, MAX_PROB_SANITY)
    impl    = implied_prob(decimal_odds)
    edge    = our_prob - impl
    ev      = expected_value(our_prob, decimal_odds)
    kel     = kelly(our_prob, decimal_odds)
    return {
        "bet_type":    bet_type,
        "description": description,
        "our_prob":    round(our_prob, 4),
        "sb_odds":     round(decimal_odds, 2),
        "implied_prob": round(impl, 4),
        "edge":        round(edge, 4),
        "ev_pct":      round(ev * 100, 2),
        "kelly_pct":   round(kel * 100, 2),
        "is_value":    (ev > 0 and edge >= MIN_EDGE),
    }


# ---------------------------------------------------------------------------
# Method market parsing
# ---------------------------------------------------------------------------

_METHOD_RE = re.compile(
    r"(ko|tko|ko/tko|submission|sub|decision|points|dec)",
    re.IGNORECASE,
)


def _classify_method_selection(name: str) -> str:
    """Map a SportsBet selection name to one of: KO_TKO | SUB | DEC."""
    lname = name.lower()
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


def _prob_over_under(prob_rounds: dict[str, float], direction: str, line: float) -> float:
    """
    Compute P(over/under X.5) from a round-probability distribution.
    prob_rounds: {"R1": ..., "R2": ..., "R3": ..., "R4": ..., "R5": ...}
    """
    threshold = int(line + 0.5)  # e.g. 2.5 → 2, so "over 2.5" means round >= 3
    total_over  = sum(v for k, v in prob_rounds.items() if int(k[1:]) > threshold)
    total_under = sum(v for k, v in prob_rounds.items() if int(k[1:]) <= threshold)
    if direction == "over":
        return total_over
    return total_under


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
    prob_ko_any  = float(props.get("prob_a_wins_ko_tko", 0)) + float(props.get("prob_b_wins_ko_tko", 0))
    prob_sub_any = float(props.get("prob_a_wins_sub", 0))    + float(props.get("prob_b_wins_sub", 0))

    bets: list[dict] = []

    def _add(bet_type, label, our_p, odds):
        b = _bet_row(bet_type, label, our_p, odds)
        if b:
            bets.append(b)

    # ---- Moneyline --------------------------------------------------------
    for name, prob, key in [(name_a, prob_a, "moneyline_a"), (name_b, prob_b, "moneyline_b")]:
        odds = sb.get(key)
        if odds and odds > MIN_ODDS:
            _add("moneyline", f"{name} wins", prob, odds)

    # ---- Method of victory (fighter-attributed) ---------------------------
    for sel_name, odds in (sb.get("method") or {}).items():
        if odds < MIN_ODDS or not props:
            continue
        side   = _fighter_side(sel_name, name_a, name_b)
        method = _classify_method_selection(sel_name)
        if side == "A":
            our_p = props.get(f"prob_a_wins_{method.lower()}", 0)
            label = f"{name_a} wins by {method.replace('_', '/')}"
        elif side == "B":
            our_p = props.get(f"prob_b_wins_{method.lower()}", 0)
            label = f"{name_b} wins by {method.replace('_', '/')}"
        else:
            continue
        _add("method", label, our_p, odds)

    # ---- Method neutral ("How fight will End") ----------------------------
    for sel_name, odds in (sb.get("method_neutral") or {}).items():
        if odds < MIN_ODDS or not props:
            continue
        lname = sel_name.lower()
        if "ko" in lname or "tko" in lname:
            _add("method_neutral", "Fight ends by KO/TKO", prob_ko_any, odds)
        elif "sub" in lname:
            _add("method_neutral", "Fight ends by Submission", prob_sub_any, odds)
        elif any(w in lname for w in ("point", "dec", "judge")):
            _add("method_neutral", "Fight goes to Decision", prob_dec, odds)

    # ---- Double chance (method combo per fighter) -------------------------
    for sel_name, odds in (sb.get("method_combo") or {}).items():
        if odds < MIN_ODDS or not props:
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
            _add("method_combo", f"{pfx} wins by KO or Sub", ko + sub, odds)
        elif "ko" in lname and any(w in lname for w in ("point", "dec")):
            _add("method_combo", f"{pfx} wins by KO or Decision", ko + dec, odds)
        elif "sub" in lname and any(w in lname for w in ("point", "dec")):
            _add("method_combo", f"{pfx} wins by Sub or Decision", sub + dec, odds)

    # ---- Go the distance --------------------------------------------------
    for sel_name, odds in (sb.get("distance") or {}).items():
        if odds < MIN_ODDS or not props:
            continue
        lname = sel_name.lower()
        if "yes" in lname or "go" in lname:
            _add("distance", "Fight goes to decision", prob_dec, odds)
        elif "no" in lname or "not" in lname:
            _add("distance", "Fight ends before decision", prob_finish, odds)

    # ---- Total rounds (over/under line) ----------------------------------
    for sel_name, odds in (sb.get("total_rounds") or {}).items():
        if odds < MIN_ODDS or not prob_rounds:
            continue
        parsed = _parse_total_rounds_line(sel_name, is_five_round)
        if parsed is None:
            continue
        direction, line = parsed
        our_p = _prob_over_under(prob_rounds, direction, line)
        label = f"{'Over' if direction == 'over' else 'Under'} {line} rounds"
        _add("total_rounds", label, our_p, odds)

    # ---- Round survival ("Fight To Start Round X") -----------------------
    for sel_name, odds in (sb.get("round_survival") or {}).items():
        if odds < MIN_ODDS or not prob_rounds:
            continue
        # Key format: "Start R2 Yes" / "Start R2 No"
        m = re.search(r"Start R(\d)", sel_name)
        if not m:
            continue
        rnum = int(m.group(1))
        # P(fight starts round X) = 1 - sum(P(end in R1..R(X-1)))
        p_survives = 1.0 - sum(
            prob_rounds.get(f"R{i}", 0) for i in range(1, rnum)
        )
        if "yes" in sel_name.lower():
            _add("round_survival", f"Fight reaches Round {rnum}", p_survives, odds)
        elif "no" in sel_name.lower():
            _add("round_survival", f"Fight ends before Round {rnum}", 1 - p_survives, odds)

    # ---- Alt finish timing ("Round 1 or 2" / "Round 3 or Decision") -----
    for sel_name, odds in (sb.get("alt_finish_timing") or {}).items():
        if odds < MIN_ODDS or not prob_rounds:
            continue
        lname = sel_name.lower()
        if "1 or 2" in lname:
            our_p = prob_rounds.get("R1", 0) + prob_rounds.get("R2", 0)
            _add("alt_finish_timing", "Ends in Round 1 or 2", our_p, odds)
        elif "3" in lname:
            our_p = sum(prob_rounds.get(f"R{i}", 0) for i in range(3, 6))
            _add("alt_finish_timing", "Ends in Round 3+ / Decision", our_p, odds)

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
                 winner_p * timing_p, odds)
        elif "3" in lname or "dec" in lname:
            timing_p = sum(prob_rounds.get(f"R{i}", 0) for i in range(3, 6))
            _add("alt_round", f"{pfx} wins in R3 or by Decision",
                 winner_p * timing_p, odds)

    # ---- Winning round (fighter-attributed) ------------------------------
    for sel_name, odds in (sb.get("winning_round") or {}).items():
        if odds < MIN_ODDS or not prob_rounds:
            continue
        m = re.search(r"round\s+(\d)", sel_name, re.IGNORECASE)
        if not m:
            continue
        rnum = int(m.group(1))
        rkey = f"R{rnum}"
        # Scale by finish prob — round bets pay only on finishes
        our_p = prob_rounds.get(rkey, 0) * prob_finish
        _add("winning_round", f"Fight ends in Round {rnum}", our_p, odds)

    # Sort: value bets first, then by EV descending
    bets.sort(key=lambda x: (-int(x.get("is_value", False)), -x.get("ev_pct", 0)))
    return bets


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
