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

import logging
import time
from typing import Any

import requests
from rapidfuzz import fuzz

log = logging.getLogger(__name__)

_BASE = "https://www.sportsbet.com.au"
_API = f"{_BASE}/apigw/sportsbook-sports"
_SPORT_SLUG = "mixed-martial-arts"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-AU,en;q=0.9",
    "Referer": f"{_BASE}/betting/{_SPORT_SLUG}",
    "Origin": _BASE,
    "X-Requested-With": "XMLHttpRequest",
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


def _to_list(data: Any, *keys: str) -> list:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in keys:
            v = data.get(k)
            if isinstance(v, list):
                return v
    return []


# ---------------------------------------------------------------------------
# API fetch layer (tries multiple known endpoint patterns)
# ---------------------------------------------------------------------------

def _fetch_competitions() -> list[dict]:
    time.sleep(_DELAY_S)
    for path in [
        f"{_API}/Sport/{_SPORT_SLUG}/competitions",
        f"{_API}/Thoroughbred/sport/{_SPORT_SLUG}/competitions",
        f"{_API}/sport/{_SPORT_SLUG}/competitions",
    ]:
        data = _get(path)
        if data is not None:
            result = _to_list(data, "competitions", "data", "result")
            if result:
                log.info("Fetched %d competitions via %s", len(result), path)
                return result
    log.warning(
        "Could not fetch SportsBet MMA competitions. "
        "The API endpoint may have changed — check network requests on sportsbet.com.au/betting/mixed-martial-arts"
    )
    return []


def _fetch_events(competition_id: str) -> list[dict]:
    time.sleep(_DELAY_S)
    for path in [
        f"{_API}/Competition/{competition_id}/events",
        f"{_API}/Thoroughbred/competitions/{competition_id}/events",
    ]:
        data = _get(path)
        if data is not None:
            result = _to_list(data, "events", "data", "result")
            if result:
                return result
    return []


def _fetch_children(event_id: str) -> list[dict]:
    """Individual fight events nested under a UFC card (parent) event."""
    time.sleep(_DELAY_S)
    for path in [
        f"{_API}/Event/{event_id}/children",
        f"{_API}/Thoroughbred/events/{event_id}/children",
    ]:
        data = _get(path)
        if data is not None:
            result = _to_list(data, "events", "children", "data", "result")
            if result:
                return result
    return []


def _fetch_markets(event_id: str) -> list[dict]:
    time.sleep(_DELAY_S)
    for path in [
        f"{_API}/Event/{event_id}/markets",
        f"{_API}/Thoroughbred/events/{event_id}/markets",
    ]:
        data = _get(path)
        if data is not None:
            result = _to_list(data, "markets", "data", "result")
            if result:
                return result
    return []


# ---------------------------------------------------------------------------
# Market parsing
# ---------------------------------------------------------------------------

# Maps substring of market name → canonical type
_MARKET_MAP: dict[str, str] = {
    "fight winner":          "moneyline",
    "fight outcome":         "moneyline",
    "match winner":          "moneyline",
    "head to head":          "moneyline",
    "method of victory":     "method",
    "method of winning":     "method",
    "fight method":          "method",
    "winning method":        "method",
    "to go the distance":    "distance",
    "fight to go distance":  "distance",
    "go the distance":       "distance",
    "total rounds":          "total_rounds",
    "over/under":            "total_rounds",
    "round betting":         "winning_round",
    "winning round":         "winning_round",
    "round of victory":      "winning_round",
    "round winner":          "winning_round",
    "fight to end":          "winning_round",
}


def _market_type(name: str) -> str:
    lname = name.lower()
    for key, mtype in _MARKET_MAP.items():
        if key in lname:
            return mtype
    return "other"


def _extract_price(sel: dict) -> float | None:
    for key in ("displayPrice", "price", "winPrice", "odds", "win"):
        v = sel.get(key)
        if isinstance(v, dict):
            for inner in ("win", "decimal", "displayPrice"):
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
    return out


def _vs_split(name: str) -> tuple[str, str]:
    for sep in (" vs ", " v ", " VS ", " V "):
        if sep in name:
            a, b = name.split(sep, 1)
            return a.strip(), b.strip()
    return name.strip(), ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_ufc_markets() -> list[dict]:
    """
    Fetch all upcoming UFC fight markets from SportsBet Australia.

    Returns list of fight dicts:
    [
        {
            "sportsbet_event_id": "...",
            "fight_name": "Jones vs Miocic",
            "fighter_a_raw": "Jones",
            "fighter_b_raw": "Miocic",
            "start_time": "2026-04-25T08:00:00.000Z",
            "markets": {
                "moneyline": {"Jones": 1.45, "Miocic": 2.80},
                "method": {"Jones to win by KO/TKO": 2.50, ...},
                "distance": {"Yes": 2.10, "No": 1.70},
                "total_rounds": {"Over 2.5 Rounds": 1.85, "Under 2.5 Rounds": 1.90},
                "winning_round": {"Round 1": 5.00, "Round 2": 6.50, ...},
            }
        }
    ]
    """
    competitions = _fetch_competitions()
    if not competitions:
        return []

    all_fights: list[dict] = []

    for comp in competitions:
        cid = str(comp.get("id") or comp.get("competitionId") or "")
        cname = comp.get("name") or ""
        if not cid:
            continue

        events = _fetch_events(cid)
        log.info("Competition '%s': %d events", cname, len(events))

        for event in events:
            eid = str(event.get("id") or event.get("eventId") or "")
            ename = event.get("name") or event.get("eventName") or ""
            estart = event.get("startTime") or event.get("start_time") or ""

            if not eid:
                continue

            # UFC cards are parent events; individual fights are children
            children = _fetch_children(eid)
            fight_events = [c for c in children if _vs_split(c.get("name") or "")[1]]

            if not fight_events:
                # Some layouts embed fights as the event itself
                if _vs_split(ename)[1]:
                    fight_events = [event]
                else:
                    continue

            for fe in fight_events:
                fid = str(fe.get("id") or fe.get("eventId") or "")
                fname = fe.get("name") or fe.get("eventName") or ename
                fstart = fe.get("startTime") or fe.get("start_time") or estart

                fa, fb = _vs_split(fname)
                if not fa or not fb:
                    continue

                raw_markets = _fetch_markets(fid)
                markets = _parse_markets(raw_markets)

                if not markets.get("moneyline"):
                    log.debug("No moneyline found for '%s', skipping", fname)
                    continue

                all_fights.append({
                    "sportsbet_event_id": fid,
                    "fight_name": fname,
                    "fighter_a_raw": fa,
                    "fighter_b_raw": fb,
                    "start_time": fstart,
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
                "method":         markets.get("method", {}),
                "distance":       markets.get("distance", {}),
                "total_rounds":   markets.get("total_rounds", {}),
                "winning_round":  markets.get("winning_round", {}),
            }
            log.info(
                "Matched '%s vs %s' ↔ '%s' (score=%d)",
                pa, pb, best_sb.get("fight_name"), best_score,
            )
        else:
            pred["sportsbet_odds"] = None

    return predictions


def run() -> list[dict]:
    logging.basicConfig(level=logging.INFO)
    return fetch_ufc_markets()


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
