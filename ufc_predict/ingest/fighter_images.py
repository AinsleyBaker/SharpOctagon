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
IMAGES_DIR   = Path("docs/fighter-images")  # served by GitHub Pages
_CDN_BASE    = "https://dmxg5wxfqgde4.cloudfront.net/styles/athlete_bio_full_body/s3/"
_UFC_BASE    = "https://www.ufc.com/athlete"
_HEADERS     = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-AU,en;q=0.9",
    "Referer": "https://www.ufc.com/",
}
_DELAY_S = 1.0


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
        # Try full-body first, fall back to headshot
        for pattern in [
            r"full[_\-]body/s3/(\d{4}-\d{2}/[A-Z0-9_\-]+\.(?:png|jpg|webp))",
            r"athlete[_\-]bio[_\-]headshot/s3/(\d{4}-\d{2}/[A-Z0-9_\-]+\.(?:png|jpg|webp))",
            r"/styles/[a-z_]+/s3/(\d{4}-\d{2}/[A-Z0-9_\-]+\.(?:png|jpg|webp))",
        ]:
            m = re.search(pattern, r.text, re.IGNORECASE)
            if m:
                if "full_body" in pattern:
                    return _CDN_BASE + m.group(1)
                # Other styles: use the matched style path
                style_match = re.search(r"(/styles/[a-z_]+/s3/" + re.escape(m.group(1)) + ")", r.text)
                if style_match:
                    return "https://dmxg5wxfqgde4.cloudfront.net" + style_match.group(1)
    except requests.RequestException as exc:
        log.debug("Image fetch failed for '%s': %s", name, exc)
    return None


def download_image(name: str, url: str) -> str | None:
    """
    Download a fighter image to docs/fighter-images/ and return the relative path.
    Files are served by GitHub Pages, bypassing UFC CDN hotlink restrictions.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    slug = _name_to_slug(name)
    ext  = url.rsplit(".", 1)[-1].lower() if "." in url else "png"
    if ext not in ("png", "jpg", "jpeg", "webp"):
        ext = "png"
    target = IMAGES_DIR / f"{slug}.{ext}"
    if target.exists() and target.stat().st_size > 1000:
        return f"fighter-images/{slug}.{ext}"  # already cached
    try:
        r = requests.get(url, headers=_HEADERS, timeout=20)
        if r.status_code == 200 and len(r.content) > 1000:
            target.write_bytes(r.content)
            return f"fighter-images/{slug}.{ext}"
    except requests.RequestException as exc:
        log.debug("Image download failed for '%s': %s", name, exc)
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

    new_fetches, new_downloads = 0, 0
    for name in names:
        # Skip if we already have a local image cached and the file still exists
        cached = cache.get(name, "")
        if cached.startswith("fighter-images/") and (Path("docs") / cached).exists():
            continue

        log.info("Processing %s…", name)
        url = fetch_image_url(name)
        if url:
            local = download_image(name, url)
            if local:
                cache[name] = local
                new_downloads += 1
                log.info("  ✓ downloaded → %s", local)
            else:
                cache[name] = url  # fall back to remote URL (may not load due to hotlink)
                log.info("  ⚠ download failed, kept remote URL")
        else:
            cache[name] = ""
            log.info("  ✗ no image found")
        new_fetches += 1
        time.sleep(_DELAY_S)

    save_cache(cache)
    log.info("Fighter images: %d total, %d processed, %d downloaded locally",
             len(cache), new_fetches, new_downloads)
    return cache


def refresh_from_predictions(predictions_path: Path = Path("data/predictions.json")) -> dict[str, str]:
    """
    Fallback for environments without DB access — read fighter names from
    predictions.json and download images for each.
    """
    if not predictions_path.exists():
        log.warning("No predictions found at %s", predictions_path)
        return load_cache()

    preds = json.loads(predictions_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for p in preds:
        for k in ("fighter_a_name", "fighter_b_name"):
            n = p.get(k, "").strip()
            if n:
                names.add(n)

    cache = load_cache()
    new_fetches = 0
    for name in sorted(names):
        cached = cache.get(name, "")
        if cached.startswith("fighter-images/") and (Path("docs") / cached).exists():
            continue
        log.info("Processing %s…", name)
        url = fetch_image_url(name)
        if url:
            local = download_image(name, url)
            if local:
                cache[name] = local
                log.info("  ✓ downloaded → %s", local)
            else:
                cache[name] = url
        else:
            cache[name] = ""
        new_fetches += 1
        time.sleep(_DELAY_S)

    save_cache(cache)
    log.info("Fighter images: %d total, %d processed", len(cache), new_fetches)
    return cache


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Try DB-based first; always also try predictions.json for fighters
    # that may be in predictions but not yet polled into the upcoming_bouts table.
    try:
        cache = refresh_for_upcoming()
    except Exception as exc:
        log.warning("DB-based fetch failed (%s)", exc)
        cache = load_cache()

    # Always also try predictions.json — covers fighters present in
    # predictions but missing from the upcoming_bouts table.
    try:
        refresh_from_predictions()
    except Exception as exc:
        log.warning("predictions.json fallback failed: %s", exc)
