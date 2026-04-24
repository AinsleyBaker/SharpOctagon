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

def _parse_total_rounds_line(selection_name: str) -> tuple[str, float] | None:
    """
    Parse 'Over 2.5 Rounds' → ('over', 2.5), 'Under 2.5 Rounds' → ('under', 2.5).
    Returns None if unrecognised.
    """
    m = re.search(r"(over|under)\s+(\d+\.?\d*)", selection_name, re.IGNORECASE)
    if m:
        direction = m.group(1).lower()
        line = float(m.group(2))
        return direction, line
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

    prediction must contain:
      - fighter_a_name, fighter_b_name
      - prob_a_wins, prob_b_wins
      - props (from prop_models.predict_props)
      - sportsbet_odds (from sportsbet_scraper.match_odds_to_predictions)

    Returns list of bet dicts sorted by EV descending.
    """
    sb = prediction.get("sportsbet_odds") or {}
    props = prediction.get("props") or {}
    name_a = prediction.get("fighter_a_name") or "Fighter A"
    name_b = prediction.get("fighter_b_name") or "Fighter B"
    prob_a = float(prediction.get("prob_a_wins") or 0.5)
    prob_b = float(prediction.get("prob_b_wins") or (1 - prob_a))

    bets: list[dict] = []

    # ---- Moneyline --------------------------------------------------------
    odds_a = sb.get("moneyline_a")
    odds_b = sb.get("moneyline_b")
    if odds_a and odds_a > MIN_ODDS:
        b = _bet_row("moneyline", f"{name_a} wins", prob_a, odds_a)
        if b:
            bets.append(b)
    if odds_b and odds_b > MIN_ODDS:
        b = _bet_row("moneyline", f"{name_b} wins", prob_b, odds_b)
        if b:
            bets.append(b)

    # ---- Method of victory -----------------------------------------------
    method_market = sb.get("method") or {}
    if method_market and props:
        for sel_name, odds in method_market.items():
            if odds < MIN_ODDS:
                continue
            side   = _fighter_side(sel_name, name_a, name_b)
            method = _classify_method_selection(sel_name)

            if side == "A":
                if method == "KO_TKO":
                    our_p = props.get("prob_a_wins_ko_tko", 0)
                elif method == "SUB":
                    our_p = props.get("prob_a_wins_sub", 0)
                else:
                    our_p = props.get("prob_a_wins_dec", 0)
                label = f"{name_a} wins by {method.replace('_', '/')}"
            elif side == "B":
                if method == "KO_TKO":
                    our_p = props.get("prob_b_wins_ko_tko", 0)
                elif method == "SUB":
                    our_p = props.get("prob_b_wins_sub", 0)
                else:
                    our_p = props.get("prob_b_wins_dec", 0)
                label = f"{name_b} wins by {method.replace('_', '/')}"
            else:
                continue

            b = _bet_row("method", label, our_p, odds)
            if b:
                bets.append(b)

    # ---- Go the distance --------------------------------------------------
    distance_market = sb.get("distance") or {}
    prob_dec = props.get("prob_decision", 0)
    if distance_market and prob_dec > 0:
        for sel_name, odds in distance_market.items():
            if odds < MIN_ODDS:
                continue
            lname = sel_name.lower()
            if "yes" in lname or "go" in lname:
                b = _bet_row("distance", "Fight goes to decision", prob_dec, odds)
            elif "no" in lname or "not" in lname:
                b = _bet_row("distance", "Fight ends before decision", 1.0 - prob_dec, odds)
            else:
                continue
            if b:
                bets.append(b)

    # ---- Total rounds (over/under) ----------------------------------------
    total_rounds_market = sb.get("total_rounds") or {}
    prob_rounds = props.get("prob_rounds") or {}
    if total_rounds_market and prob_rounds:
        for sel_name, odds in total_rounds_market.items():
            if odds < MIN_ODDS:
                continue
            parsed = _parse_total_rounds_line(sel_name)
            if parsed is None:
                continue
            direction, line = parsed
            our_p = _prob_over_under(prob_rounds, direction, line)
            label = f"{'Over' if direction == 'over' else 'Under'} {line} rounds"
            b = _bet_row("total_rounds", label, our_p, odds)
            if b:
                bets.append(b)

    # ---- Winning round ----------------------------------------------------
    winning_round_market = sb.get("winning_round") or {}
    if winning_round_market and prob_rounds:
        for sel_name, odds in winning_round_market.items():
            if odds < MIN_ODDS:
                continue
            m = re.search(r"round\s+(\d)", sel_name, re.IGNORECASE)
            if not m:
                continue
            rnum = int(m.group(1))
            rkey = f"R{rnum}"
            our_p = prob_rounds.get(rkey, 0)
            # Winning round bets are typically for finishes only — scale by finish prob
            finish_prob = props.get("prob_finish", 1.0)
            our_p_finish = our_p * finish_prob
            b = _bet_row("winning_round", f"Fight ends in Round {rnum}", our_p_finish, odds)
            if b:
                bets.append(b)

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
