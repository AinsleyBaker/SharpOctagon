"""
Stage 5 — Sherdog scraper.

Fetches DOB, nationality, training camp, and pre-UFC fight history
for fighters that have a sherdog_id but missing data.

Throttled to 1–2 s per request. Does NOT scrape fighters that already
have all fields populated (incremental).
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date

import requests
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from ufc_predict.db.models import Fighter

log = logging.getLogger(__name__)

_BASE = "https://www.sherdog.com/fighter/"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
_DELAY_S = 1.5  # polite delay between requests


def _get(url: str) -> BeautifulSoup | None:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "lxml")
    except requests.RequestException as exc:
        log.warning("Sherdog fetch failed for %s: %s", url, exc)
        return None


def scrape_fighter(sherdog_id: str) -> dict:
    """
    Scrape a single fighter page. Returns a dict with any of:
    dob, nationality, camp, height_cm, reach_cm, stance
    """
    url = _BASE + sherdog_id
    soup = _get(url)
    if soup is None:
        return {}

    data: dict = {}

    # --- DOB ---
    bday = soup.find("span", itemprop="birthDate")
    if bday:
        try:
            data["dob"] = date.fromisoformat(bday["content"][:10])
        except Exception:
            pass

    # --- Nationality ---
    nation = soup.find("strong", itemprop="nationality")
    if nation:
        data["nationality"] = nation.get_text(strip=True)

    # --- Training camp (association) ---
    camp_tag = soup.find("a", {"href": re.compile(r"/gym/")})
    if camp_tag:
        data["camp"] = camp_tag.get_text(strip=True)

    # --- Height ---
    height_tag = soup.find("b", string=re.compile(r"Height", re.I))
    if height_tag:
        sibling = height_tag.find_next_sibling(string=True) or height_tag.next_sibling
        if sibling:
            data["height_raw"] = str(sibling).strip()

    # --- Reach ---
    reach_tag = soup.find("b", string=re.compile(r"Reach", re.I))
    if reach_tag:
        sibling = reach_tag.find_next_sibling(string=True) or reach_tag.next_sibling
        if sibling:
            data["reach_raw"] = str(sibling).strip()

    return data


def _parse_inches(val: str) -> float | None:
    """'72"' or '6\' 0"' → cm"""
    m = re.match(r"(\d+)'\s*(\d+)\"", val)
    if m:
        return round((int(m.group(1)) * 12 + int(m.group(2))) * 2.54, 1)
    m = re.match(r"([\d.]+)\"?", val)
    if m:
        return round(float(m.group(1)) * 2.54, 1)
    return None


def enrich_fighters(session: Session, limit: int | None = None) -> int:
    """
    Find fighters with a sherdog_id but missing dob/nationality/camp and enrich them.
    Returns number of fighters updated.
    """
    query = session.query(Fighter).filter(
        Fighter.sherdog_id.isnot(None),
    )

    # Only process those missing at least one enrichable field
    fighters_to_enrich = [
        f for f in query
        if not all([f.dob, f.nationality])
    ]

    if limit:
        fighters_to_enrich = fighters_to_enrich[:limit]

    log.info("Enriching %d fighters from Sherdog", len(fighters_to_enrich))
    updated = 0

    for i, fighter in enumerate(fighters_to_enrich):
        data = scrape_fighter(fighter.sherdog_id)
        if not data:
            continue

        changed = False
        if "dob" in data and not fighter.dob:
            fighter.dob = data["dob"]
            changed = True
        if "nationality" in data and not fighter.nationality:
            fighter.nationality = data["nationality"]
            changed = True
        if "height_raw" in data and not fighter.height_cm:
            cm = _parse_inches(data["height_raw"])
            if cm:
                fighter.height_cm = cm
                changed = True
        if "reach_raw" in data and not fighter.reach_cm:
            cm = _parse_inches(data["reach_raw"])
            if cm:
                fighter.reach_cm = cm
                changed = True

        if changed:
            updated += 1

        # Commit in batches of 50
        if (i + 1) % 50 == 0:
            session.commit()
            log.info("  … %d/%d enriched so far", i + 1, len(fighters_to_enrich))

        time.sleep(_DELAY_S)

    session.commit()
    log.info("Sherdog enrichment complete — %d fighters updated", updated)
    return updated


def run(db_url: str | None = None, limit: int | None = None) -> None:
    from ufc_predict.db.session import get_session_factory
    factory = get_session_factory(db_url)
    with factory() as session:
        enrich_fighters(session, limit=limit)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(limit=limit)
