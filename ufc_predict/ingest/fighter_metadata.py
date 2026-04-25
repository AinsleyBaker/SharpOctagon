"""
Fetch fighter metadata (country, fighting style, image URL) from ufc.com
for every fighter that appears in upcoming predictions OR the schedule.

Saves to data/fighter_metadata.json so the dashboard can render flags,
official UFC fighting styles, and headshots without re-scraping each build.

Usage:
    python -m ufc_predict.ingest.fighter_metadata
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from pathlib import Path

import requests

from ufc_predict.ingest.fighter_images import (
    download_image, fetch_image_url, fetch_wikipedia_image, _name_to_slug,
)

log = logging.getLogger(__name__)

METADATA_PATH = Path("data/fighter_metadata.json")
PREDICTIONS_PATH = Path("data/predictions.json")
SCHEDULE_PATH    = Path("data/upcoming_schedule.json")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-AU,en;q=0.9",
    "Referer": "https://www.ufc.com/",
}
_DELAY_S = 1.0
_UFC_BASE = "https://www.ufc.com/athlete"


def _parse_bio_fields(html: str) -> dict[str, str]:
    """Extract all <Label, Text> pairs from the c-bio__* blocks."""
    labels = re.findall(
        r'class="[^"]*c-bio__label[^"]*"[^>]*>\s*([^<]+?)\s*</', html
    )
    texts = re.findall(
        r'class="[^"]*c-bio__text[^"]*"[^>]*>\s*([^<]+?)\s*</', html
    )
    out: dict[str, str] = {}
    for lbl, txt in zip(labels, texts):
        out[lbl.strip()] = txt.strip()
    return out


def _extract_country(hometown: str) -> str | None:
    """
    Hometown is shown as 'City, ST' (US), 'City, Country', or just 'Country'.
    Returns the country, defaulting to 'United States' for US state codes.
    """
    if not hometown:
        return None
    parts = [p.strip() for p in hometown.split(",")]
    if len(parts) == 1:
        return parts[0]
    last = parts[-1]
    # 2-letter state codes (US/Canada) — assume US unless explicit
    if len(last) == 2 and last.isupper():
        return "United States"
    return last


def fetch_metadata(name: str) -> dict | None:
    """Return {country, fighting_style, image_url, hometown} or None."""
    slug = _name_to_slug(name)
    url  = f"{_UFC_BASE}/{slug}"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=12)
        if r.status_code != 200:
            return None
        bio = _parse_bio_fields(r.text)
    except requests.RequestException:
        return None

    hometown = bio.get("Hometown", "")
    country  = _extract_country(hometown)
    style    = bio.get("Fighting style", "")
    age      = bio.get("Age", "")

    # Reuse fighter_images.fetch_image_url logic for the URL with itok
    img_url = fetch_image_url(name)
    return {
        "country":   country,
        "hometown":  hometown,
        "style":     style,
        "age":       age,
        "image_url": img_url,
    }


def load_metadata() -> dict[str, dict]:
    if not METADATA_PATH.exists():
        return {}
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_metadata(meta: dict[str, dict]) -> None:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def collect_all_fighter_names() -> set[str]:
    """Combine fighter names from predictions.json AND upcoming_schedule.json."""
    names: set[str] = set()
    if PREDICTIONS_PATH.exists():
        for p in json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8")):
            for k in ("fighter_a_name", "fighter_b_name"):
                n = (p.get(k) or "").strip()
                if n:
                    names.add(n)
    if SCHEDULE_PATH.exists():
        for ev in json.loads(SCHEDULE_PATH.read_text(encoding="utf-8")):
            for b in ev.get("bouts", []):
                for k in ("fighter_a", "fighter_b"):
                    n = (b.get(k) or "").strip()
                    if n:
                        names.add(n)
    return names


def refresh(force: bool = False) -> dict[str, dict]:
    """
    Fetch metadata + images for every upcoming-bout fighter.
    Skips fighters already in the cache unless force=True.
    Also downloads images to docs/fighter-images/.
    """
    names = collect_all_fighter_names()
    cache = load_metadata()
    log.info("Refreshing metadata for %d fighters (cached: %d)", len(names), len(cache))

    new_count = 0
    for name in sorted(names):
        if not force and name in cache and cache[name].get("country"):
            # Verify image still exists locally — if not, re-fetch
            img = cache[name].get("image_url", "")
            if not img.startswith("fighter-images/") or (Path("docs") / img).exists():
                continue

        log.info("Fetching %s…", name)
        meta = fetch_metadata(name) or {}

        # Image: download to docs/fighter-images/, fall back to Wikipedia
        img_url = meta.get("image_url")
        if not img_url:
            img_url = fetch_wikipedia_image(name)
        local_img = ""
        if img_url:
            local = download_image(name, img_url)
            local_img = local if local else img_url

        cache[name] = {
            "country":   meta.get("country") or "",
            "hometown":  meta.get("hometown") or "",
            "style":     meta.get("style") or "",
            "age":       meta.get("age") or "",
            "image_url": local_img,
        }
        new_count += 1
        log.info(
            "  ✓ %s | %s | %s",
            cache[name]["country"] or "?",
            cache[name]["style"] or "?",
            "image" if local_img else "no image",
        )
        time.sleep(_DELAY_S)

    save_metadata(cache)
    log.info("Metadata: %d total, %d processed", len(cache), new_count)
    return cache


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    refresh()
