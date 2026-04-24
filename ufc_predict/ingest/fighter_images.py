"""
Fetch and cache UFC fighter profile images from ufc.com.

Scrapes each upcoming fighter's athlete page to extract their CDN image URL,
saves to data/fighter_images.json. Included in the workflow after upcoming_poller.

Usage:
    python -m ufc_predict.ingest.fighter_images
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

CACHE_PATH   = Path("data/fighter_images.json")
_CDN_BASE    = "https://dmxg5wxfqgde4.cloudfront.net/styles/athlete_bio_full_body/s3/"
_UFC_BASE    = "https://www.ufc.com/athlete"
_HEADERS     = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-AU,en;q=0.9",
}
_DELAY_S = 1.5


def _name_to_slug(name: str) -> str:
    import unicodedata
    # Normalise accents and build slug
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^a-zA-Z0-9\s]", "", name).strip().lower()
    return re.sub(r"\s+", "-", name)


def fetch_image_url(name: str) -> str | None:
    """Return the UFC CDN full-body image URL for a fighter, or None."""
    slug = _name_to_slug(name)
    url  = f"{_UFC_BASE}/{slug}"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=12)
        if r.status_code != 200:
            return None
        # Extract the CDN path from page source
        m = re.search(
            r"full[_\-]body/s3/(\d{4}-\d{2}/[A-Z0-9_\-]+\.(?:png|jpg|webp))",
            r.text,
            re.IGNORECASE,
        )
        if m:
            return _CDN_BASE + m.group(1)
    except requests.RequestException as exc:
        log.debug("Image fetch failed for '%s': %s", name, exc)
    return None


def load_cache() -> dict[str, str]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def save_cache(cache: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def refresh_for_upcoming(db_url: str | None = None) -> dict[str, str]:
    """
    Fetch image URLs for all fighters in upcoming bouts.
    Returns the updated cache dict.
    """
    from ufc_predict.db.session import get_session_factory
    from ufc_predict.db.models import Fighter, UpcomingBout
    from datetime import date

    cache = load_cache()
    factory = get_session_factory(db_url)

    with factory() as session:
        bouts = (
            session.query(UpcomingBout)
            .filter(
                UpcomingBout.event_date >= date.today(),
                UpcomingBout.is_cancelled.is_(False),
                UpcomingBout.red_fighter_id.isnot(None),
                UpcomingBout.blue_fighter_id.isnot(None),
            )
            .all()
        )
        fighter_ids = set()
        for b in bouts:
            fighter_ids.add(b.red_fighter_id)
            fighter_ids.add(b.blue_fighter_id)

        names: list[str] = []
        for fid in fighter_ids:
            f = session.get(Fighter, fid)
            if f:
                names.append(f.full_name)

    new_fetches = 0
    for name in names:
        if name in cache:
            continue
        log.info("Fetching image for %s…", name)
        url = fetch_image_url(name)
        cache[name] = url or ""
        new_fetches += 1
        if url:
            log.info("  ✓ %s", url)
        else:
            log.info("  ✗ no image found")
        time.sleep(_DELAY_S)

    save_cache(cache)
    log.info("Fighter images: %d total, %d newly fetched", len(cache), new_fetches)
    return cache


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    refresh_for_upcoming()
