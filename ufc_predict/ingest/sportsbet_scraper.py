"""
SportsBet Australia — MMA/UFC odds scraper (no account required).

Uses SportsBet's public JSON API that powers their website.
Collects all available markets per fight: moneyline, method of victory,
round betting, total rounds, go-the-distance.

NOTE: Scrape responsibly. 2-second delays between requests are enforced.
For personal research/analysis only.

If API endpoints stop working, inspect network requests on:
  https://www.sportsbet.com.au/betting/mixed-martial-arts
and update _API / endpoint paths below.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from rapidfuzz import fuzz

log = logging.getLogger(__name__)

_BASE = "https://www.sportsbet.com.au"
ODDS_CACHE_PATH = Path("data/sportsbet_odds.json")

# API discovered from JS bundle analysis — path changed ~2024-Q4
_API = f"{_BASE}/apigw/sportsbook-sports/Sportsbook/Sports"
_COMPETITION_ID = 3703  # "UFC Matches" competition on SportsBet AU

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-AU,en;q=0.9",
    "Referer": f"{_BASE}/betting/ufc-mma",
}
_DELAY_S = 2.0
_MATCH_THRESHOLD = 72


def _get(url: str) -> Any | None:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        if "json" not in resp.headers.get("content-type", ""):
            return None
        return resp.json()
    except (requests.RequestException, ValueError) as exc:
        log.warning("SportsBet request failed %s — %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# API fetch layer
# ---------------------------------------------------------------------------

def _fetch_competition_events() -> list[dict]:
    """Return all fight events in the UFC Matches competition."""
    time.sleep(_DELAY_S)
    url = f"{_API}/Competitions/{_COMPETITION_ID}?displayType=default"
    data = _get(url)
    if not isinstance(data, dict):
        log.warning(
            "Could not fetch SportsBet UFC events. "
            "If the API has changed, re-inspect network requests on "
            "sportsbet.com.au/betting/ufc-mma and update _API / _COMPETITION_ID."
        )
        return []
    events = data.get("events", [])
    log.info("Fetched %d UFC events from SportsBet competition %s", len(events), _COMPETITION_ID)
    return events


def _fetch_markets(event_id: int) -> list[dict]:
    time.sleep(_DELAY_S)
    url = f"{_API}/Events/{event_id}/Markets"
    data = _get(url)
    return data if isinstance(data, list) else []


# ---------------------------------------------------------------------------
# Market parsing
# ---------------------------------------------------------------------------

# Maps substring of market name → canonical type
_MARKET_MAP: dict[str, str] = {
    # Moneyline
    "match betting":                "moneyline",
    "fight winner":                 "moneyline",
    "fight outcome":                "moneyline",
    "match winner":                 "moneyline",
    "head to head":                 "moneyline",
    "decision no bet":              "moneyline_no_dec",
    # Method — fighter-attributed
    "method of victory":            "method",
    "method of winning":            "method",
    "fight method":                 "method",
    "winning method":               "method",
    # Method — neutral (no fighter attribution)
    "how fight will end":           "method_neutral",
    "how will the fight end":       "method_neutral",
    "how will fight end":           "method_neutral",
    # Method combo (double chance)
    "double chance":                "method_combo",
    # Distance
    "will the fight go the distance": "distance",
    "to go the distance":           "distance",
    "fight to go distance":         "distance",
    "go the distance":              "distance",
    "goes the distance":            "distance",
    # Total rounds over/under
    "total rounds":                 "total_rounds",
    "over/under":                   "total_rounds",
    # Winning round — fighter-attributed (more specific keys must come first)
    "winning round & minute":       "other",   # too granular — exclude
    "gone in 60 seconds":           "other",   # exclude
    "round betting":                "winning_round",
    "winning round":                "winning_round",
    "what round will fight end":    "winning_round",
    "round of victory":             "winning_round",
    "round winner":                 "winning_round",
    "fight to end":                 "winning_round",
    # Round survival (will the fight START round X?)
    "fight to start round":         "round_survival",
    # Alt grouped finish timing (Round 1 or 2 / Round 3 or decision)
    "alt. when will the fight end": "alt_finish_timing",
    "alt. when will fight end":     "alt_finish_timing",
    "alt. round betting":           "alt_round",
}


def _market_type(name: str) -> str:
    lname = name.lower()
    # Totals markets (sig strikes / takedowns / knockdowns) parse via a
    # dedicated path — _annotate_totals_markets — and must NOT be bucketed
    # under total_rounds / over_under here. Without this guard, the
    # "Significant Strikes Over/Under" market name matches "over/under" and
    # leaks into total_rounds, breaking the totals quantile bet emission.
    if any(kw in lname for kw in (
        "significant strike", "sig strike", "sig. strike", "total strikes",
        "total strike", "takedown", "knockdown",
    )):
        return "other"
    for key, mtype in _MARKET_MAP.items():
        if key in lname:
            return mtype
    return "other"


def _extract_price(sel: dict) -> float | None:
    # New API: sel["price"]["winPrice"]
    for key in ("price", "displayPrice", "winPrice", "odds", "win"):
        v = sel.get(key)
        if isinstance(v, dict):
            for inner in ("winPrice", "win", "decimal", "displayPrice"):
                if isinstance(v.get(inner), (int, float)) and v[inner] > 1.0:
                    return float(v[inner])
        elif isinstance(v, (int, float)) and v > 1.0:
            return float(v)
    return None


def _parse_selections(raw: list) -> dict[str, float]:
    out: dict[str, float] = {}
    for sel in raw:
        name = (
            sel.get("name") or sel.get("selectionName") or sel.get("label") or ""
        ).strip()
        p = _extract_price(sel)
        if name and p is not None:
            out[name] = p
    return out


def _annotate_round_survival(raw_markets: list[dict], out: dict) -> None:
    """
    'Fight To Start Round 2' / 'Fight To Start Round 3' markets don't encode
    the round number in the selection name — only in the market name itself.
    Store them as 'round_survival' with selection keys like 'Start R2 Yes'.
    """
    for m in raw_markets:
        mname = m.get("name") or ""
        if "fight to start round" not in mname.lower():
            continue
        rnum_match = re.search(r"round\s+(\d)", mname, re.IGNORECASE)
        if not rnum_match:
            continue
        rnum = rnum_match.group(1)
        sels_raw = m.get("selections") or []
        for sel in sels_raw:
            sname = (sel.get("name") or "").strip()
            price = _extract_price(sel)
            if price and sname:
                key = f"Start R{rnum} {sname}"
                if "round_survival" not in out:
                    out["round_survival"] = {}
                out["round_survival"][key] = price


_TOTALS_LINE_RE = re.compile(r"(over|under)\s+(\d+\.?\d*)", re.IGNORECASE)


def _classify_totals_market(mname: str, fa_raw: str, fb_raw: str) -> str | None:
    """Identify which totals market a SportsBet market name represents.

    Returns one of: total_sig_strikes_combined, total_sig_strikes_a,
    total_sig_strikes_b, total_takedowns_combined, total_takedowns_a,
    total_takedowns_b, total_knockdowns_combined — or None if the market
    isn't a recognised totals market.
    """
    n = mname.lower()
    has_strike = ("significant strike" in n or "sig strike" in n or "sig. strike" in n)
    has_total_strike = "total strike" in n
    is_takedown = "takedown" in n
    is_knockdown = "knockdown" in n
    if not (has_strike or has_total_strike or is_takedown or is_knockdown):
        return None
    if "over/under" not in n and "total " not in n and "totals " not in n:
        # Some markets just say "Knockdowns Over Under" — still match
        if "over" not in n and "under" not in n:
            return None

    fa = (fa_raw or "").lower()
    fb = (fb_raw or "").lower()
    fa_last = fa.split()[-1] if fa else ""
    fb_last = fb.split()[-1] if fb else ""
    is_fa = bool(fa_last and fa_last in n)
    is_fb = bool(fb_last and fb_last in n)
    # If both names match, neither is a per-fighter market — fall back to combined
    if is_fa and is_fb:
        is_fa = is_fb = False
    side = "a" if is_fa else ("b" if is_fb else None)

    if has_strike or has_total_strike:
        if side == "a":
            return "total_sig_strikes_a"
        if side == "b":
            return "total_sig_strikes_b"
        return "total_sig_strikes_combined"
    if is_takedown:
        if side == "a":
            return "total_takedowns_a"
        if side == "b":
            return "total_takedowns_b"
        return "total_takedowns_combined"
    if is_knockdown:
        # Per-fighter knockdowns are rare; we collapse to combined since
        # the model's _a/_b knockdown targets aren't separately exposed.
        return "total_knockdowns_combined"
    return None


_METHOD_ROUND_FIGHTER_RE = re.compile(
    r"^(?P<who>.+?)\s+(?P<method>KO/TKO|Submission|Sub)\s*&\s*Round\s+(?P<rnd>\d)\s*$",
    re.IGNORECASE,
)
_METHOD_ROUND_NEUTRAL_RE = re.compile(
    r"^(?P<method>KO/TKO|Submission|Sub)\s*&\s*Round\s+(?P<rnd>\d)\s*$",
    re.IGNORECASE,
)
_ROUND_COMBO_RANGE_RE = re.compile(
    r"^(?P<who>.+?)\s+to\s+win\s+by\s+(?P<method>KO/TKO|Submission|Sub)"
    r"\s+in\s+Rounds?\s+(?P<rounds>[\d,\s]+(?:\s+or\s+\d+)?)\s*$",
    re.IGNORECASE,
)


def _canon_method_token(token: str) -> str:
    t = token.upper().replace(" ", "")
    if "KO" in t or t == "TKO":
        return "KO"
    if "SUB" in t:
        return "SUB"
    return ""


def _parse_round_range(spec: str) -> tuple[int, ...]:
    """Parse SportsBet's round-range syntax. Examples:
        "1 or 2"           -> (1, 2)
        "1,2 or 3"         -> (1, 2, 3)
        "2,3 or 4"         -> (2, 3, 4)
    """
    cleaned = spec.replace(",", " ").lower().replace(" or ", " ")
    rounds: list[int] = []
    for tok in cleaned.split():
        try:
            r = int(tok)
            if 1 <= r <= 5:
                rounds.append(r)
        except ValueError:
            continue
    return tuple(sorted(set(rounds)))


def _which_fighter(who: str, fa_raw: str, fb_raw: str) -> str | None:
    """Map a 'who' token from a SportsBet selection to A/B based on the
    raw fighter names captured at scrape time. Returns None when neither
    name is a confident match.
    """
    fa = (fa_raw or "").lower()
    fb = (fb_raw or "").lower()
    fa_last = fa.split()[-1] if fa else ""
    fb_last = fb.split()[-1] if fb else ""
    w = who.lower()
    score_a = max(
        fuzz.token_set_ratio(fa, w),
        fuzz.partial_ratio(fa_last, w) if fa_last else 0,
    )
    score_b = max(
        fuzz.token_set_ratio(fb, w),
        fuzz.partial_ratio(fb_last, w) if fb_last else 0,
    )
    if score_a >= 70 and score_a > score_b:
        return "A"
    if score_b >= 70 and score_b > score_a:
        return "B"
    return None


def _annotate_method_round_markets(
    raw_markets: list[dict],
    out: dict,
    fa_raw: str,
    fb_raw: str,
) -> None:
    """Parse the four method×round markets SportsBet posts on big cards:

        - "Method & Round Combo (N Rounds)"        — fighter + method + single round
        - "Alt. Method & Round Combo (N Rounds)"   — neutral method + single round
        - "KO/TKO Round Combos (N Rounds)"         — fighter, multi-round range
        - "Submission Round Combos (N Rounds)"     — fighter, multi-round range

    Output keys:
        out["method_round_fighter"]  = list of dicts:
            {"side": "A"|"B", "method": "KO"|"SUB", "round": int, "odds": float}
        out["method_round_neutral"]  = list of dicts:
            {"method": "KO"|"SUB", "round": int, "odds": float}
        out["method_round_ranges"]   = list of dicts:
            {"side": "A"|"B", "method": "KO"|"SUB", "rounds": tuple[int],
             "odds": float, "label": str}
    """
    for m in raw_markets:
        mname = (m.get("name") or m.get("marketName") or "").lower()
        is_fighter_combo = (
            "method & round combo" in mname and "alt." not in mname
        )
        is_neutral_combo = "alt. method & round combo" in mname
        is_ko_range = "ko/tko round combos" in mname
        is_sub_range = "submission round combos" in mname
        if not (is_fighter_combo or is_neutral_combo or is_ko_range or is_sub_range):
            continue

        sels = m.get("selections") or m.get("outcomes") or []
        for sel in sels:
            sname = (sel.get("name") or "").strip()
            price = _extract_price(sel)
            if not sname or price is None:
                continue

            if is_fighter_combo:
                match = _METHOD_ROUND_FIGHTER_RE.match(sname)
                if not match:
                    continue
                method = _canon_method_token(match.group("method"))
                if not method:
                    continue
                side = _which_fighter(match.group("who"), fa_raw, fb_raw)
                if side is None:
                    continue
                out.setdefault("method_round_fighter", []).append({
                    "side": side,
                    "method": method,
                    "round": int(match.group("rnd")),
                    "odds": float(price),
                    "raw_name": sname,
                })

            elif is_neutral_combo:
                match = _METHOD_ROUND_NEUTRAL_RE.match(sname)
                if not match:
                    continue
                method = _canon_method_token(match.group("method"))
                if not method:
                    continue
                out.setdefault("method_round_neutral", []).append({
                    "method": method,
                    "round": int(match.group("rnd")),
                    "odds": float(price),
                    "raw_name": sname,
                })

            elif is_ko_range or is_sub_range:
                match = _ROUND_COMBO_RANGE_RE.match(sname)
                if not match:
                    continue
                method = _canon_method_token(match.group("method"))
                if not method:
                    continue
                side = _which_fighter(match.group("who"), fa_raw, fb_raw)
                if side is None:
                    continue
                rounds = _parse_round_range(match.group("rounds"))
                if not rounds:
                    continue
                out.setdefault("method_round_ranges", []).append({
                    "side": side,
                    "method": method,
                    "rounds": rounds,
                    "odds": float(price),
                    "label": sname,
                })


def _annotate_totals_markets(
    raw_markets: list[dict],
    out: dict,
    fa_raw: str,
    fb_raw: str,
) -> None:
    """Parse totals markets (sig strikes / takedowns / knockdowns) into a
    canonical structure SportsBet doesn't directly expose.

    Output schema for each canonical key:
        out[key] = [
            {"line": 165.5, "over_odds": 1.91, "under_odds": 1.83},
            ...
        ]

    Multiple lines per market (alt totals) are preserved as separate entries.
    Raw decimal odds are kept un-de-vigged so downstream code can decide
    whether to remove the overround.
    """
    for m in raw_markets:
        mname = m.get("name") or m.get("marketName") or ""
        canonical = _classify_totals_market(mname, fa_raw, fb_raw)
        if canonical is None:
            continue
        sels_raw = m.get("selections") or m.get("outcomes") or m.get("runners") or []
        # Group selections by parsed line, capturing over/under prices.
        by_line: dict[float, dict] = {}
        for sel in sels_raw:
            sname = (sel.get("name") or sel.get("selectionName") or "").strip()
            price = _extract_price(sel)
            if not sname or price is None:
                continue
            match = _TOTALS_LINE_RE.search(sname)
            if not match:
                # Some markets put the line in the market name and just label
                # selections "Over" / "Under". Try to recover from market name.
                line_match = re.search(r"(\d+\.?\d*)", mname)
                if not line_match:
                    continue
                line = float(line_match.group(1))
                direction = "over" if "over" in sname.lower() else (
                    "under" if "under" in sname.lower() else None
                )
            else:
                direction = match.group(1).lower()
                line = float(match.group(2))
            if direction not in ("over", "under"):
                continue
            entry = by_line.setdefault(line, {"line": line})
            entry[f"{direction}_odds"] = float(price)

        if not by_line:
            continue
        # Only emit lines with both sides priced — single-sided totals are
        # almost always parser glitches. Allow single-sided as a fallback if
        # nothing has both.
        complete = [v for v in by_line.values() if "over_odds" in v and "under_odds" in v]
        partial = [v for v in by_line.values() if v not in complete]
        keep = complete or partial
        out.setdefault(canonical, []).extend(keep)


def _parse_markets(
    raw_markets: list[dict],
    fa_raw: str = "",
    fb_raw: str = "",
) -> dict[str, dict[str, float]]:
    out: dict[str, dict] = {}
    for m in raw_markets:
        mname = m.get("name") or m.get("marketName") or m.get("label") or ""
        mtype = _market_type(mname)
        if mtype == "other":
            continue
        sels_raw = m.get("selections") or m.get("outcomes") or m.get("runners") or []
        sels = _parse_selections(sels_raw)
        if sels:
            if mtype not in out:
                out[mtype] = {}
            out[mtype].update(sels)
    # Round-survival markets need special handling (round number is in market name)
    _annotate_round_survival(raw_markets, out)
    # Method×round joint markets — fighter-attributed, neutral, and multi-round
    # ranges. These pair with our compute_method_round_joint() probabilities
    # for the cleanest cross-fight EV comparison on big cards.
    _annotate_method_round_markets(raw_markets, out, fa_raw, fb_raw)
    # Totals markets (sig strikes / takedowns / knockdowns) — listed under
    # idiosyncratic market names that don't fit _MARKET_MAP cleanly.
    _annotate_totals_markets(raw_markets, out, fa_raw, fb_raw)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_ufc_markets() -> list[dict]:
    """
    Fetch all upcoming UFC fight markets from SportsBet Australia.

    Returns list of fight dicts with keys:
      sportsbet_event_id, fight_name, fighter_a_raw, fighter_b_raw,
      start_time, markets (moneyline / method / distance / total_rounds / winning_round)
    """
    events = _fetch_competition_events()
    if not events:
        return []

    all_fights: list[dict] = []

    for event in events:
        eid  = event.get("id")
        fa   = (event.get("participant1") or "").strip()
        fb   = (event.get("participant2") or "").strip()
        if not eid or not fa or not fb:
            continue

        raw_markets = _fetch_markets(eid)
        markets = _parse_markets(raw_markets, fa, fb)

        if not markets.get("moneyline"):
            log.debug("No moneyline for event %s '%s v %s', skipping", eid, fa, fb)
            continue

        fname = event.get("name") or f"{fa} v {fb}"
        all_fights.append({
            "sportsbet_event_id": str(eid),
            "fight_name": fname,
            "fighter_a_raw": fa,
            "fighter_b_raw": fb,
            "start_time": event.get("startTime", ""),
            "markets": markets,
        })
        log.info("  ✓ %s  [%s]", fname, ", ".join(sorted(markets)))

    log.info("SportsBet scrape complete: %d fights with markets", len(all_fights))
    return all_fights


# ---------------------------------------------------------------------------
# Matching to predictions
# ---------------------------------------------------------------------------

def _name_score(a: str, b: str) -> int:
    if not a or not b:
        return 0
    return max(
        fuzz.token_set_ratio(a, b),
        fuzz.partial_ratio(a.split()[-1], b),
        fuzz.partial_ratio(a, b.split()[-1]),
    )


def _find_odds_by_name(market: dict[str, float], target: str) -> float | None:
    if not market or not target:
        return None
    best = max(market, key=lambda n: _name_score(target, n), default=None)
    if best and _name_score(target, best) >= 60:
        return market[best]
    return None


def match_odds_to_predictions(
    sb_fights: list[dict],
    predictions: list[dict],
) -> list[dict]:
    """
    Attach SportsBet market data to each prediction by fuzzy fighter name matching.
    Adds a 'sportsbet_odds' key to each prediction dict. Modifies in-place.
    """
    for pred in predictions:
        pa = (pred.get("fighter_a_name") or "").strip()
        pb = (pred.get("fighter_b_name") or "").strip()

        best_sb: dict | None = None
        best_score = 0
        best_reversed = False

        for sb in sb_fights:
            sba = sb.get("fighter_a_raw") or ""
            sbb = sb.get("fighter_b_raw") or ""

            s_normal  = min(_name_score(pa, sba), _name_score(pb, sbb))
            s_flipped = min(_name_score(pa, sbb), _name_score(pb, sba))
            score = max(s_normal, s_flipped)

            if score > best_score:
                best_score = score
                best_sb = sb
                best_reversed = s_flipped > s_normal

        if best_sb and best_score >= _MATCH_THRESHOLD:
            markets = best_sb.get("markets", {})
            moneyline = markets.get("moneyline", {})

            fa_key = best_sb["fighter_b_raw"] if best_reversed else best_sb["fighter_a_raw"]
            fb_key = best_sb["fighter_a_raw"] if best_reversed else best_sb["fighter_b_raw"]

            # Per-fighter totals are scraped under raw fa/fb keys; if the
            # match was reversed, swap the _a/_b assignments accordingly.
            tssa, tssb = "total_sig_strikes_a", "total_sig_strikes_b"
            ttda, ttdb = "total_takedowns_a", "total_takedowns_b"
            if best_reversed:
                tot_ss_a = markets.get(tssb, [])
                tot_ss_b = markets.get(tssa, [])
                tot_td_a = markets.get(ttdb, [])
                tot_td_b = markets.get(ttda, [])
            else:
                tot_ss_a = markets.get(tssa, [])
                tot_ss_b = markets.get(tssb, [])
                tot_td_a = markets.get(ttda, [])
                tot_td_b = markets.get(ttdb, [])

            # Method×round market entries carry their A/B side based on the
            # raw scraper-time fighter assignment. When the prediction maps
            # to SportsBet's red/blue in reverse, every "A" → "B" and vice
            # versa so the prediction-side bet emission stays consistent.
            def _flip_side_entries(entries, side_key="side"):
                if not best_reversed:
                    return entries
                flipped = []
                for e in entries:
                    e2 = dict(e)
                    if e2.get(side_key) == "A":
                        e2[side_key] = "B"
                    elif e2.get(side_key) == "B":
                        e2[side_key] = "A"
                    flipped.append(e2)
                return flipped

            mr_fighter = _flip_side_entries(markets.get("method_round_fighter", []))
            mr_neutral = list(markets.get("method_round_neutral", []))
            mr_ranges = _flip_side_entries(markets.get("method_round_ranges", []))

            pred["sportsbet_odds"] = {
                "source": "sportsbet.com.au",
                "fight_name_raw": best_sb.get("fight_name"),
                "start_time": best_sb.get("start_time"),
                "match_score": best_score,
                "moneyline_a": _find_odds_by_name(moneyline, fa_key or ""),
                "moneyline_b": _find_odds_by_name(moneyline, fb_key or ""),
                "method":            markets.get("method", {}),
                "method_neutral":    markets.get("method_neutral", {}),
                "method_combo":      markets.get("method_combo", {}),
                "distance":          markets.get("distance", {}),
                "total_rounds":      markets.get("total_rounds", {}),
                "winning_round":     markets.get("winning_round", {}),
                "round_survival":    markets.get("round_survival", {}),
                "alt_finish_timing": markets.get("alt_finish_timing", {}),
                "alt_round":         markets.get("alt_round", {}),
                # Totals — list[{line, over_odds, under_odds}], possibly empty
                "total_sig_strikes_combined": markets.get("total_sig_strikes_combined", []),
                "total_sig_strikes_a":        tot_ss_a,
                "total_sig_strikes_b":        tot_ss_b,
                "total_takedowns_combined":   markets.get("total_takedowns_combined", []),
                "total_takedowns_a":          tot_td_a,
                "total_takedowns_b":          tot_td_b,
                "total_knockdowns_combined":  markets.get("total_knockdowns_combined", []),
                # Method×round joint markets (only present on big cards)
                "method_round_fighter": mr_fighter,
                "method_round_neutral": mr_neutral,
                "method_round_ranges":  mr_ranges,
            }
            log.info(
                "Matched '%s vs %s' ↔ '%s' (score=%d)",
                pa, pb, best_sb.get("fight_name"), best_score,
            )
        else:
            pred["sportsbet_odds"] = None

    return predictions


def save_markets(fights: list[dict], path: Path = ODDS_CACHE_PATH) -> None:
    """Persist scraped markets to a JSON cache so CI can read them without geo-access."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "fights": fights,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("SportsBet odds cached to %s (%d fights)", path, len(fights))


def cache_age_hours(path: Path = ODDS_CACHE_PATH) -> float | None:
    """Age of the SportsBet cache in hours, or None if missing/unreadable."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(data["fetched_at"])
        return (datetime.now(UTC) - fetched_at).total_seconds() / 3600
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def load_markets(path: Path = ODDS_CACHE_PATH) -> list[dict] | None:
    """Load previously cached markets. Returns None if no cache exists."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        fights = data.get("fights", [])
        fetched_at = data.get("fetched_at", "unknown")
        log.info("Loaded %d fights from SportsBet cache (fetched %s)", len(fights), fetched_at)
        return fights
    except (json.JSONDecodeError, KeyError) as exc:
        log.warning("Could not load SportsBet cache: %s", exc)
        return None


def run() -> list[dict]:
    logging.basicConfig(level=logging.INFO)
    fights = fetch_ufc_markets()
    if fights:
        save_markets(fights)
    else:
        log.warning("No markets fetched — cache not updated")
    return fights


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
