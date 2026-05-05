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
OVERRIDES_PATH   = Path("data/fighter_overrides.json")


def _load_overrides() -> dict:
    """Read manual overrides for fighters whose UFC.com profile uses a
    different name (slug_aliases) or whose bio is missing a parseable
    field (country). Returns the raw JSON dict, or empty if the file
    is missing/invalid.
    """
    if not OVERRIDES_PATH.exists():
        return {}
    try:
        return json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

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
    """Extract <Label, Text> pairs from the c-bio__field blocks.

    Each block is `<div class="c-bio__field"><div class="c-bio__label">L</div>
    <div class="c-bio__text">T</div></div>`. For some labels (Age) the text
    is nested inside `<div class="field field__item">VALUE</div>`. The earlier
    flat zip(labels, texts) misaligned in those cases. BeautifulSoup handles
    the nesting cleanly.
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    out: dict[str, str] = {}
    for field in soup.select("div.c-bio__field"):
        label_el = field.select_one(".c-bio__label")
        text_el = field.select_one(".c-bio__text")
        if not label_el or not text_el:
            continue
        # Prefer nested .field__item (Age) over the outer text node
        nested = text_el.select_one(".field__item")
        value = (nested or text_el).get_text(strip=True)
        out[label_el.get_text(strip=True)] = value
    return out


def _inches_to_cm(text: str) -> float | None:
    """UFC.com renders height/reach as inches (e.g. '77.00'). Convert to cm."""
    try:
        v = float(text)
    except (TypeError, ValueError):
        return None
    return round(v * 2.54, 1) if v > 0 else None


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


def _parse_record_from_html(html: str) -> str:
    """
    UFC athlete pages display the fighter record as '25-5-0' inside a
    <p class="hero-profile__division-body"> tag. Match it directly.
    """
    m = re.search(
        r'hero-profile__division-body[^>]*>\s*(\d{1,2}-\d{1,2}-\d{1,2})',
        html,
    )
    if m:
        return m.group(1)
    # Fallback: any standalone X-Y-Z pattern in a tag
    m = re.search(r'>\s*(\d{1,2}-\d{1,2}-\d{1,2})\s*(?:\([^)]*\))?\s*<', html)
    if m:
        return m.group(1)
    return ""


def fetch_metadata(name: str) -> dict | None:
    """Return {country, style, image_url, hometown, age, record} or None.

    Resolution order:
      1. Try the requested name's slug.
      2. If that 404s/redirects to search and the name has a slug_alias
         in fighter_overrides.json (e.g. "Tommy Gantt" → "Thomas Gantt"),
         retry with the alias.
      3. If a country override exists for this fighter, fill it in even
         when the UFC.com bio omits a Hometown field (Khamzat Chimaev
         is a notable example).
    """
    import re
    overrides = _load_overrides()
    slug_aliases = overrides.get("slug_aliases", {}) or {}
    country_overrides = overrides.get("country", {}) or {}

    candidates = [name]
    alias = slug_aliases.get(name)
    if alias:
        candidates.append(alias)

    bio: dict[str, str] = {}
    record = ""
    matched_name: str | None = None
    img_url_arg_name = name  # which name to query for the image asset
    for cand in candidates:
        slug = _name_to_slug(cand)
        url  = f"{_UFC_BASE}/{slug}"
        try:
            r = requests.get(url, headers=_HEADERS, timeout=12)
        except requests.RequestException:
            continue
        if r.status_code != 200:
            continue
        # Confirm the page is actually for this fighter — otherwise we
        # capture another fighter's bio (or the search-results placeholder
        # for debut fighters) and pollute the dataset.
        title_m = re.search(r"<title>([^<]+)</title>", r.text)
        raw_title = (title_m.group(1) if title_m else "").lower()
        title_text = re.sub(r"[^a-z0-9]+", " ", raw_title)
        last_name_raw = (cand or "").strip().split()[-1].lower() if cand.strip() else ""
        last_name = re.sub(r"[^a-z0-9]+", " ", last_name_raw).strip()
        if not last_name or last_name not in title_text:
            continue
        bio = _parse_bio_fields(r.text)
        record = _parse_record_from_html(r.text)
        matched_name = cand
        img_url_arg_name = cand
        break

    if matched_name is None:
        # No UFC.com profile found even after alias retry. Still allow a
        # country override to surface (some fighters have only the country
        # override set — their bio data may come from elsewhere later).
        if country_overrides.get(name):
            return {
                "country":   country_overrides[name],
                "hometown":  "",
                "style":     "",
                "age":       "",
                "record":    "",
                "image_url": None,
            }
        return None

    hometown = bio.get("Hometown", "")
    country  = _extract_country(hometown) or country_overrides.get(name) or country_overrides.get(matched_name)
    style    = bio.get("Fighting style", "")
    age      = bio.get("Age", "")

    img_url = fetch_image_url(img_url_arg_name)
    return {
        "country":   country,
        "hometown":  hometown,
        "style":     style,
        "age":       age,
        "record":    record,
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


def _fallback_record_from_db(name: str, session) -> tuple[str, dict]:
    """
    If UFC.com didn't expose a record, compute it from our local fight DB.
    Returns (record_str, stats_dict).
    """
    from ufc_predict.db.models import Fighter
    from ufc_predict.features.aso_features import fighter_aso_stats
    from datetime import date as _date

    fighter = session.query(Fighter).filter(Fighter.full_name == name).first()
    if not fighter:
        return "", {}

    stats = fighter_aso_stats(fighter.canonical_fighter_id, _date.today(), session)
    wins   = int(stats.get("wins", 0) or 0)
    losses = int(stats.get("losses", 0) or 0)
    record = f"{wins}-{losses}-0" if (wins + losses) > 0 else ""
    return record, {
        "wins":   wins,
        "losses": losses,
        "stance": fighter.stance or "",
        "nationality_db": fighter.nationality or "",
    }


def _full_stats_from_db(name: str, session) -> dict:
    """
    Pull every field used by the dashboard stats panel for this fighter.
    Lets preview events (no predictions yet) still show meaningful data.
    """
    from ufc_predict.db.models import Fighter
    from ufc_predict.features.aso_features import fighter_aso_stats
    from datetime import date as _date
    import math

    fighter = session.query(Fighter).filter(Fighter.full_name == name).first()
    if not fighter:
        return {}
    s = fighter_aso_stats(fighter.canonical_fighter_id, _date.today(), session)

    def _f(v):
        # JSON-safe: NaN → None
        if v is None: return None
        try:
            f = float(v)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    return {
        "n_fights":     int(s.get("n_fights", 0) or 0),
        "win_streak":   int(s.get("win_streak", 0) or 0),
        "loss_streak":  int(s.get("loss_streak", 0) or 0),
        "l3_win_rate":  _f(s.get("l3_win_rate")),
        "finish_rate":  _f(s.get("finish_rate")),
        "ko_rate":      _f(s.get("ko_rate")),
        "sub_rate":     _f(s.get("sub_rate")),
        "slpm":         _f(s.get("slpm")),
        "sapm":         _f(s.get("sapm")),
        "td_per_min":   _f(s.get("td_per_min")),
        "stance":       fighter.stance or "",
    }


def refresh(force: bool = False) -> dict[str, dict]:
    """
    Fetch metadata + images for every upcoming-bout fighter.
    Skips fighters already in the cache unless force=True.
    """
    from ufc_predict.db.session import get_session_factory

    names = collect_all_fighter_names()
    cache = load_metadata()
    log.info("Refreshing metadata for %d fighters (cached: %d)", len(names), len(cache))

    factory = None
    try:
        factory = get_session_factory()
    except Exception as exc:
        log.warning("DB unavailable for fallback record lookup: %s", exc)

    new_count = 0
    session = factory() if factory else None
    try:
        for name in sorted(names):
            if not force and name in cache and cache[name].get("country") and cache[name].get("record"):
                img = cache[name].get("image_url", "")
                if not img.startswith("fighter-images/") or (Path("docs") / img).exists():
                    continue

            log.info("Fetching %s…", name)
            meta = fetch_metadata(name) or {}

            img_url = meta.get("image_url")
            if not img_url:
                img_url = fetch_wikipedia_image(name)
            local_img = ""
            if img_url:
                local = download_image(name, img_url)
                local_img = local if local else img_url

            record = meta.get("record", "")
            db_extra: dict = {}
            if not record and session:
                record, db_extra = _fallback_record_from_db(name, session)

            # Always pull full DB stats too (used by preview events that have
            # no predictions yet — gives them a meaningful expandable panel).
            db_stats = _full_stats_from_db(name, session) if session else {}

            cache[name] = {
                "country":   meta.get("country") or db_extra.get("nationality_db", ""),
                "hometown":  meta.get("hometown") or "",
                "style":     meta.get("style") or "",
                "age":       meta.get("age") or "",
                "record":    record,
                "image_url": local_img,
                "stats":     db_stats,
            }
            new_count += 1
            log.info(
                "  ✓ %s | %s | %s | %s",
                cache[name]["country"] or "?",
                cache[name]["style"]   or "?",
                cache[name]["record"]  or "?",
                "image" if local_img else "no image",
            )
            time.sleep(_DELAY_S)
    finally:
        if session:
            session.close()

    save_metadata(cache)
    log.info("Metadata: %d total, %d processed", len(cache), new_count)
    return cache


def enrich_physicals(
    session=None,
    force: bool = False,
    active_since: str | None = None,
    limit: int | None = None,
) -> dict:
    """Fill `fighters.reach_cm` / `height_cm` from UFC.com athlete pages.

    Iterates fighters with reach_cm OR height_cm missing, fetches the bio
    block, and persists. UFC.com renders both as inches (e.g. "78.00") which
    we convert to cm.

    Args:
      force: overwrite existing non-null values too (e.g. correcting bad
             upstream data like reach=height for some Greco rows).
      active_since: ISO date — only enrich fighters with at least one fight
             on or after this date. Use to prioritise active roster (high
             value for bet predictions).
      limit: cap how many fighters to attempt this run.
    """
    from sqlalchemy import text
    from ufc_predict.db.models import Fighter
    from ufc_predict.db.session import get_session_factory

    own_session = False
    if session is None:
        factory = get_session_factory()
        session = factory()
        own_session = True

    q = session.query(Fighter)
    if not force:
        q = q.filter((Fighter.reach_cm.is_(None)) | (Fighter.height_cm.is_(None)))

    if active_since:
        active_ids = {row[0] for row in session.execute(text("""
            SELECT red_fighter_id FROM fights WHERE date >= :d
            UNION
            SELECT blue_fighter_id FROM fights WHERE date >= :d
        """), {"d": active_since}).fetchall()}
        q = q.filter(Fighter.canonical_fighter_id.in_(active_ids))

    fighters = q.all()
    if limit:
        fighters = fighters[:limit]

    stats = {"n_attempted": 0, "n_updated": 0, "n_no_match": 0, "n_no_data": 0}
    log.info("enrich_physicals: %d fighters with missing reach/height", len(fighters))

    try:
        for fighter in fighters:
            stats["n_attempted"] += 1
            slug = _name_to_slug(fighter.full_name)
            url = f"{_UFC_BASE}/{slug}"
            try:
                r = requests.get(url, headers=_HEADERS, timeout=12)
            except requests.RequestException:
                stats["n_no_match"] += 1
                continue
            if r.status_code != 200:
                stats["n_no_match"] += 1
                continue

            # Sanity: title must contain the fighter's last name (avoid scraping
            # a different person's bio for fighters that share a slug stem).
            title_m = re.search(r"<title>([^<]+)</title>", r.text)
            title = (title_m.group(1) if title_m else "").lower()
            last = (fighter.full_name or "").strip().split()[-1].lower()
            if not last or last not in title:
                stats["n_no_match"] += 1
                continue

            bio = _parse_bio_fields(r.text)
            h_cm = _inches_to_cm(bio.get("Height", ""))
            r_cm = _inches_to_cm(bio.get("Reach", ""))
            if h_cm is None and r_cm is None:
                stats["n_no_data"] += 1
                continue

            changed = False
            if r_cm is not None and (force or fighter.reach_cm is None):
                fighter.reach_cm = r_cm
                changed = True
            if h_cm is not None and (force or fighter.height_cm is None):
                fighter.height_cm = h_cm
                changed = True
            if changed:
                stats["n_updated"] += 1
                if stats["n_updated"] % 25 == 0:
                    session.commit()
                    log.info("  ✓ %d updated…", stats["n_updated"])
            time.sleep(_DELAY_S)

        session.commit()
    finally:
        if own_session:
            session.close()

    log.info("enrich_physicals: %s", stats)
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "physicals":
        enrich_physicals(force="--force" in sys.argv)
    else:
        refresh()
