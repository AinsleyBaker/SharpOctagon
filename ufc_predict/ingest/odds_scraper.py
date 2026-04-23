"""
Closing odds scraper — BestFightOdds.com.

Scrapes American moneyline closing odds and stores them in fights.closing_odds_*
EVALUATION AND KELLY SIZING ONLY. These values are NEVER fed into model features.

Also updates upcoming_bouts with pre-fight odds for Kelly fraction computation.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date

import requests
from bs4 import BeautifulSoup
from sqlalchemy import text
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)

_BASE = "https://www.bestfightodds.com"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.bestfightodds.com/",
}
_DELAY_S = 2.0


def _get(url: str) -> BeautifulSoup | None:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "lxml")
    except requests.RequestException as exc:
        log.warning("BestFightOdds fetch failed: %s — %s", url, exc)
        return None


def _parse_american(text: str) -> float | None:
    """Parse '+150' / '-200' → float. Returns None on failure."""
    s = str(text).strip().replace("−", "-")  # handle unicode minus
    m = re.match(r"^([+-]?\d+)$", s)
    if m:
        return float(m.group(1))
    return None


def scrape_event_odds(event_slug: str) -> list[dict]:
    """
    Scrape odds for a single event page.
    event_slug: e.g. 'ufc-300-pereira-vs-hill'
    Returns list of {red_name, blue_name, closing_odds_red, closing_odds_blue}
    """
    url = f"{_BASE}/events/{event_slug}"
    soup = _get(url)
    if soup is None:
        return []

    bouts = []
    # BestFightOdds uses a table structure with fighter names and odds columns
    for row in soup.select("tr.table-row"):
        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        fighter_cell = cells[0].get_text(strip=True)
        # Closing odds are in the last column (rightmost bookmaker or "close" column)
        closing_cells = [c for c in cells if "close" in c.get("class", []) or
                         c.get("data-type") == "close"]

        if not closing_cells:
            # Fall back: last numeric cell
            for c in reversed(cells[1:]):
                val = _parse_american(c.get_text(strip=True))
                if val is not None:
                    closing_cells = [c]
                    break

        if not closing_cells:
            continue

        odds = _parse_american(closing_cells[0].get_text(strip=True))
        if odds is not None and fighter_cell:
            bouts.append({"name": fighter_cell, "odds": odds})

    # Pair up rows (alternating red/blue)
    results = []
    for i in range(0, len(bouts) - 1, 2):
        results.append({
            "red_name_raw":    bouts[i]["name"],
            "blue_name_raw":   bouts[i + 1]["name"],
            "closing_odds_red":  bouts[i]["odds"],
            "closing_odds_blue": bouts[i + 1]["odds"],
        })

    log.info("Scraped %d bouts from BestFightOdds event: %s", len(results), event_slug)
    return results


def fetch_upcoming_odds() -> list[dict]:
    """Scrape the BestFightOdds homepage for upcoming event odds."""
    soup = _get(_BASE)
    if soup is None:
        return []

    all_bouts = []
    for row in soup.select("tr.table-row"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        name = cells[0].get_text(strip=True)
        # Get best available odds (first bookmaker column)
        for c in cells[1:]:
            val = _parse_american(c.get_text(strip=True))
            if val is not None:
                all_bouts.append({"name": name, "odds": val})
                break

    results = []
    for i in range(0, len(all_bouts) - 1, 2):
        results.append({
            "red_name_raw":   all_bouts[i]["name"],
            "blue_name_raw":  all_bouts[i + 1]["name"],
            "closing_odds_red":  all_bouts[i]["odds"],
            "closing_odds_blue": all_bouts[i + 1]["odds"],
        })

    return results


def update_fight_odds(bouts: list[dict], session: Session) -> int:
    """
    Match scraped odds to Fight rows by fighter name and update closing_odds_*.
    Returns number of rows updated.
    """
    updated = 0
    for bout in bouts:
        # Try to find matching fight by fuzzy name match within 90 days
        # Using a simple approach: find fights where red/blue names approximately match
        sql = text("""
            SELECT f.fight_id, fr.full_name as red_name, fb.full_name as blue_name
            FROM fights f
            JOIN fighters fr ON fr.canonical_fighter_id = f.red_fighter_id
            JOIN fighters fb ON fb.canonical_fighter_id = f.blue_fighter_id
            WHERE f.closing_odds_red IS NULL
            ORDER BY f.date DESC
            LIMIT 200
        """)
        rows = session.execute(sql).fetchall()

        red_raw = (bout.get("red_name_raw") or "").lower().strip()
        blue_raw = (bout.get("blue_name_raw") or "").lower().strip()

        for row in rows:
            rn = (row[1] or "").lower().strip()
            bn = (row[2] or "").lower().strip()
            # Simple last-name match
            if (red_raw.split()[-1] in rn or rn.split()[-1] in red_raw) and \
               (blue_raw.split()[-1] in bn or bn.split()[-1] in blue_raw):
                session.execute(
                    text("""UPDATE fights
                            SET closing_odds_red=:ro, closing_odds_blue=:bo
                            WHERE fight_id=:fid"""),
                    {"ro": bout["closing_odds_red"], "bo": bout["closing_odds_blue"],
                     "fid": row[0]},
                )
                updated += 1
                break

    session.commit()
    return updated


def update_upcoming_kelly(upcoming_odds: list[dict], session: Session) -> None:
    """
    For each upcoming bout with odds, compute and store Kelly fraction.
    Requires predictions to already exist in data/predictions.json.
    """
    import json
    from pathlib import Path
    from ufc_predict.eval.evaluate import american_to_decimal, kelly_fraction_fn

    preds_path = Path("data/predictions.json")
    if not preds_path.exists():
        return

    with open(preds_path) as f:
        predictions = json.load(f)

    updated_preds = []
    for pred in predictions:
        matched_odds = None
        for odds_row in upcoming_odds:
            rn = (odds_row.get("red_name_raw") or "").lower()
            an = (pred.get("fighter_a_name") or "").lower()
            if any(p in rn for p in an.split()[-1:]):
                matched_odds = odds_row
                break

        if matched_odds:
            dec_odds = american_to_decimal(matched_odds["closing_odds_red"])
            prob_a = pred.get("prob_a_wins", 0.5) or 0.5
            frac = kelly_fraction_fn(prob_a, dec_odds)
            implied = 1.0 / dec_odds
            pred["kelly_fraction"] = round(frac, 4)
            pred["has_edge"] = prob_a > implied
            pred["implied_prob_a"] = round(implied, 4)

        updated_preds.append(pred)

    with open(preds_path, "w") as f:
        json.dump(updated_preds, f, indent=2, default=str)
    log.info("Updated Kelly fractions in predictions.json")


def run(db_url: str | None = None) -> None:
    from ufc_predict.db.session import get_session_factory
    factory = get_session_factory(db_url)

    log.info("Fetching upcoming odds from BestFightOdds…")
    upcoming = fetch_upcoming_odds()
    time.sleep(_DELAY_S)

    with factory() as session:
        n = update_fight_odds(upcoming, session)
        update_upcoming_kelly(upcoming, session)

    log.info("Odds update complete — %d historical fights updated", n)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
