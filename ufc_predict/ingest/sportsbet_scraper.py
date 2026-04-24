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
from datetime import datetime, timezone
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
    # Winning round — fighter-attributed
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


def _parse_markets(raw_markets: list[dict]) -> dict[str, dict[str, float]]:
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
        markets = _parse_markets(raw_markets)

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

            pred["sportsbet_odds"] = {
                "source": "sportsbet.com.au",
                "fight_name_raw": best_sb.get("fight_name"),
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
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "fights": fights,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("SportsBet odds cached to %s (%d fights)", path, len(fights))


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
