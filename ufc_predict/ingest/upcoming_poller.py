"""
Stage 10 — Upcoming card poller.

Sources (in priority order):
  1. UFC.com HTML scrape  — official, moderate lead time
  2. ESPN hidden API      — reliable, no auth, good cross-check

Writes results to upcoming_bouts table.
Refresh schedule: daily normally, 6-hourly in fight week, T-2h on fight day.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from ufc_predict.db.models import Fighter, UpcomingBout

log = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html",
}

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"
UFC_EVENTS_URL  = "https://www.ufc.com/events"
SCHEDULE_PATH   = Path("data/upcoming_schedule.json")
LOOKAHEAD_DAYS  = 90
# Also pull recently-finished events so the dashboard can persist their
# results into past_events.json before they disappear from ESPN's window.
LOOKBACK_DAYS   = 14


# ---------------------------------------------------------------------------
# ESPN API
# ---------------------------------------------------------------------------

def fetch_espn_upcoming() -> list[dict]:
    """
    Fetch UFC events from ESPN's hidden API spanning the last LOOKBACK_DAYS
    through the next LOOKAHEAD_DAYS. Returning recent past events as well as
    upcoming ones lets build_dashboard persist their results into
    past_events.json before they fall out of ESPN's window — without this
    pre-poll, an event that just finished is dropped from the scoreboard
    after a couple of days and the Past Events panel shows "result pending"
    until the next Greco CSV ingest catches up (sometimes ~24h later).
    """
    today = date.today()
    start = today - timedelta(days=LOOKBACK_DAYS)
    end   = today + timedelta(days=LOOKAHEAD_DAYS)
    date_range = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"
    url = f"{ESPN_SCOREBOARD}?dates={date_range}&limit=100"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("ESPN scoreboard fetch failed: %s", exc)
        return []

    bouts = []
    for event in data.get("events", []):
        event_name = event.get("name", "")
        event_date_str = event.get("date", "")
        try:
            event_date = datetime.fromisoformat(event_date_str.rstrip("Z")).date()
        except Exception:
            continue

        for comp in event.get("competitions", []):
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue

            red = competitors[0].get("athlete", {}).get("displayName", "")
            blue = competitors[1].get("athlete", {}).get("displayName", "")
            status = comp.get("status", {}).get("type", {}).get("name", "")

            if status == "STATUS_CANCELED":
                continue  # cancelled fights drop entirely

            # ESPN result fields populated only when STATUS_FINAL.
            winner_name = ""
            for side in competitors:
                if side.get("winner"):
                    winner_name = side.get("athlete", {}).get("displayName", "")
                    break
            # Method / round live in different shapes per ESPN sport. Try the
            # common ones; absent values fall through as empty strings.
            status_obj = comp.get("status", {}) or {}
            # ESPN's `type.description` is just "Final"/"Scheduled"/etc.
            # The actual finish method (KO/TKO, Submission, Decision) usually
            # lives in `type.detail` or `type.shortDetail`. Prefer those, and
            # only fall back to `description` if both are missing.
            type_obj = status_obj.get("type") or {}
            method = (
                type_obj.get("detail")
                or type_obj.get("shortDetail")
                or status_obj.get("description")
                or type_obj.get("description")
                or ""
            )
            # Strip generic prefix like "Final - " that ESPN sometimes prepends.
            if method.lower().startswith("final"):
                _rest = method[5:].lstrip(" -:")
                if _rest:
                    method = _rest
            round_ended = status_obj.get("period") or 0

            # ESPN puts the weight class in `competition.type.abbreviation`
            # ("Flyweight", "Heavyweight", "Welterweight", …).  The previous
            # implementation read `notes[0].headline` which the API no longer
            # populates, so weight_class came back empty for every bout.
            weight_class = (comp.get("type") or {}).get("abbreviation", "") or ""
            # ESPN doesn't expose a clean title-bout flag — fall back to the
            # event name (cards usually advertise titles in the headline).
            ev_name_l = (event_name or "").lower()
            is_title = "title" in ev_name_l or "championship" in ev_name_l
            # 5-round bouts are signalled by format.regulation.periods on the
            # competition (3 = standard prelim/main, 5 = main event / title).
            # ESPN often omits this field, in which case the main-event
            # post-pass below will still flag the headline bout as 5-round.
            periods = (
                ((comp.get("format") or {}).get("regulation") or {}).get("periods")
            )
            is_five_round = periods == 5 or is_title

            # Per-bout start time. ESPN often returns the same value as the
            # event's date, but for cards with staggered prelims it's per-fight.
            start_time_iso = comp.get("date") or event.get("date") or ""

            bouts.append({
                "event_name": event_name,
                "event_date": event_date,
                "start_time_iso": start_time_iso,
                "red_name_raw": red,
                "blue_name_raw": blue,
                "weight_class": weight_class,
                "is_title_bout": is_title,
                "is_five_round": is_five_round,
                "source": "espn",
                "is_confirmed": True,
                # Live status — surfaced through to the schedule JSON so the
                # dashboard can render per-bout LIVE/COMPLETED/SCHEDULED state.
                "espn_status":      status,           # STATUS_SCHEDULED / IN_PROGRESS / FINAL
                "espn_winner_name": winner_name,
                "espn_method":      method,
                "espn_round":       int(round_ended) if round_ended else 0,
            })

    # Main-event 5-round backfill: every UFC card since UFC Fight Night 50
    # (Sept 2014) has had a 5-round main event, but ESPN's
    # format.regulation.periods field is unreliable. Identify the headline
    # bout per event by latest start_time_iso (or last in array if all start
    # times tie) and flag it as 5-round if not already.
    by_event: dict[tuple, list[int]] = {}
    for idx, b in enumerate(bouts):
        key = (str(b.get("event_date")), b.get("event_name") or "")
        by_event.setdefault(key, []).append(idx)
    for idx_list in by_event.values():
        # Sort indices by start_time_iso desc, breaking ties on original
        # ESPN order (later index = later position on card per ESPN).
        idx_list.sort(
            key=lambda i: (bouts[i].get("start_time_iso") or "", i),
            reverse=True,
        )
        main_idx = idx_list[0]
        bouts[main_idx]["is_five_round"] = True

    log.info("ESPN: %d upcoming bouts fetched", len(bouts))
    return bouts


# ---------------------------------------------------------------------------
# UFC.com scrape
# ---------------------------------------------------------------------------

def fetch_ufc_upcoming() -> list[dict]:
    """Scrape upcoming events from UFC.com events page."""
    try:
        resp = requests.get(UFC_EVENTS_URL, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as exc:
        log.warning("UFC.com scrape failed: %s", exc)
        return []

    bouts = []
    # UFC.com lists upcoming events in cards with fight details
    # Structure varies; we extract event names and dates at minimum
    # Full fight listing requires drilling into individual event pages
    for card in soup.select("[class*='l-listing__group--bordered']"):
        date_el = card.select_one("[class*='c-card-event--result__date']")
        name_el = card.select_one("[class*='c-card-event--result__headline']")
        if not date_el or not name_el:
            continue

        try:
            event_date_str = date_el.get("data-main-card-timestamp") or date_el.get_text(strip=True)
            event_date = datetime.fromisoformat(event_date_str[:10]).date()
        except Exception:
            continue

        event_name = name_el.get_text(strip=True)
        # Minimal: store the event without individual fights (ESPN covers fights)
        bouts.append({
            "event_name": event_name,
            "event_date": event_date,
            "red_name_raw": None,
            "blue_name_raw": None,
            "weight_class": None,
            "is_title_bout": False,
            "is_five_round": False,
            "source": "ufc.com",
            "is_confirmed": True,
        })

    log.info("UFC.com: %d upcoming events scraped", len(bouts))
    return bouts


# ---------------------------------------------------------------------------
# DB upsert
# ---------------------------------------------------------------------------

def _bout_id(event_date, red_name, blue_name) -> str:
    key = f"{event_date}|{(red_name or '').lower()}|{(blue_name or '').lower()}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _resolve_fighter(
    name: str | None,
    session: Session,
    weight_class: str | None = None,
    fighter_cache: list | None = None,
) -> str | None:
    """
    Attempt to match raw fighter name to a canonical_fighter_id.

    Strategy: exact (case-insensitive) → unaccented exact → fuzzy token-set
    ratio ≥ 88, gated by weight_class when provided. Diacritics-stripping
    fixes the "José Aldo" ≠ "Jose Aldo" failure mode; punctuation-stripping
    fixes "Waldo Cortes-Acosta" ≠ "Waldo Cortes Acosta" — the hyphen fuses
    two tokens, dropping token_set_ratio from 100 to ~68 and missing the
    threshold.
    """
    if not name:
        return None
    raw = name.strip()
    if not raw:
        return None

    fighter = session.query(Fighter).filter(
        Fighter.full_name.ilike(raw)
    ).first()
    if fighter:
        return fighter.canonical_fighter_id

    import re
    import unicodedata
    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()
        # Punctuation → space, then collapse runs of whitespace, so
        # "cortes-acosta" tokenises identically to "cortes acosta".
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    target = _norm(raw)

    # Cache the candidate set so we don't re-query the DB per bout.
    if fighter_cache is None or not fighter_cache:
        all_fighters = session.query(Fighter).all()
        if fighter_cache is not None:
            fighter_cache.extend(all_fighters)
    else:
        all_fighters = fighter_cache

    for f in all_fighters:
        if f.full_name and _norm(f.full_name) == target:
            return f.canonical_fighter_id

    # Fuzzy fallback. Higher threshold than the 70 used elsewhere because
    # this runs at ingest time, where a wrong match silently corrupts the
    # whole upcoming card. Weight-class gate when known.
    from rapidfuzz import fuzz
    best_id, best_score = None, 0
    for f in all_fighters:
        if not f.full_name:
            continue
        if weight_class and f.primary_weight_class and f.primary_weight_class != weight_class:
            continue
        score = fuzz.token_set_ratio(target, _norm(f.full_name))
        if score > best_score:
            best_id, best_score = f.canonical_fighter_id, score
    if best_score >= 88:
        return best_id
    return None


def upsert_bouts(bouts: list[dict], session: Session) -> int:
    now = datetime.now(UTC)
    upserted = 0
    seen_ids: set[str] = set()
    fighter_cache: list = []  # populated lazily by the first _resolve_fighter call

    for bout in bouts:
        if not bout.get("red_name_raw") and not bout.get("blue_name_raw"):
            continue  # event-only stubs from UFC.com

        red_raw = (bout.get("red_name_raw") or "").lower().strip()
        blue_raw = (bout.get("blue_name_raw") or "").lower().strip()
        # Skip TBA placeholder fights — they all hash to the same ID
        _tba = ("tba", "tbd", "opponent tba", "")
        if red_raw in _tba or blue_raw in _tba:
            continue

        bout_id = _bout_id(
            bout["event_date"],
            bout.get("red_name_raw"),
            bout.get("blue_name_raw"),
        )

        # Within this run, never insert the same bout_id twice
        if bout_id in seen_ids:
            continue
        seen_ids.add(bout_id)

        existing = session.get(UpcomingBout, bout_id)
        if existing:
            existing.last_updated_at = now
            existing.is_cancelled = bout.get("is_cancelled", False)
            # Refresh round/title flags if the source learned new info
            # (e.g. main-event detection added a 5-round flag that wasn't
            # set when the bout was first inserted).
            if bout.get("is_five_round") and not existing.is_five_round:
                existing.is_five_round = True
            if bout.get("is_title_bout") and not existing.is_title_bout:
                existing.is_title_bout = True
            continue

        wc = bout.get("weight_class") or None
        red_id  = _resolve_fighter(bout.get("red_name_raw"),  session, wc, fighter_cache)
        blue_id = _resolve_fighter(bout.get("blue_name_raw"), session, wc, fighter_cache)

        session.add(UpcomingBout(
            upcoming_bout_id=bout_id,
            event_date=bout["event_date"],
            event_name=bout.get("event_name"),
            red_fighter_id=red_id,
            blue_fighter_id=blue_id,
            red_name_raw=bout.get("red_name_raw"),
            blue_name_raw=bout.get("blue_name_raw"),
            weight_class=bout.get("weight_class"),
            is_title_bout=bout.get("is_title_bout", False),
            is_five_round=bout.get("is_five_round", False),
            source=bout.get("source"),
            first_seen_at=now,
            last_updated_at=now,
            is_confirmed=bout.get("is_confirmed", True),
            is_cancelled=False,
        ))
        upserted += 1

    session.commit()
    log.info("Upserted %d new upcoming bouts", upserted)
    return upserted


def export_schedule(bouts: list[dict]) -> None:
    """
    Write a flat JSON of upcoming events for the dashboard carousel.
    This runs even when predictions can't be generated (no model PKL),
    so the user can browse future cards.
    """
    by_event: dict[tuple, dict] = {}
    for bout in bouts:
        if not bout.get("event_name") or not bout.get("event_date"):
            continue
        key = (str(bout["event_date"]), bout["event_name"])
        ev = by_event.setdefault(key, {
            "event_name": bout["event_name"],
            "event_date": str(bout["event_date"]),
            "bouts": [],
        })
        if bout.get("red_name_raw") or bout.get("blue_name_raw"):
            ev["bouts"].append({
                "fighter_a": bout.get("red_name_raw", ""),
                "fighter_b": bout.get("blue_name_raw", ""),
                "weight_class": bout.get("weight_class", ""),
                "is_title_bout": bout.get("is_title_bout", False),
                "is_five_round": bout.get("is_five_round", False),
                "start_time_iso": bout.get("start_time_iso", ""),
                # Live state from ESPN (only meaningful on/near fight day).
                "espn_status":      bout.get("espn_status", ""),
                "espn_winner_name": bout.get("espn_winner_name", ""),
                "espn_method":      bout.get("espn_method", ""),
                "espn_round":       bout.get("espn_round", 0),
            })

    schedule = sorted(by_event.values(), key=lambda e: e["event_date"])
    SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEDULE_PATH.write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    log.info("Schedule exported: %d events to %s", len(schedule), SCHEDULE_PATH)


def poll(db_url: str | None = None) -> int:
    espn_bouts = fetch_espn_upcoming()
    time.sleep(1)
    ufc_bouts = fetch_ufc_upcoming()

    all_bouts = espn_bouts + ufc_bouts

    # Always export the schedule first — the dashboard only needs this JSON,
    # and we want the live workflow to keep working even when the DB is
    # missing or unreachable (fresh runner, broken file, schema drift).
    export_schedule(all_bouts)

    try:
        from ufc_predict.db.session import get_session_factory
        factory = get_session_factory(db_url)
        with factory() as session:
            return upsert_bouts(all_bouts, session)
    except Exception as e:  # pragma: no cover — defensive guard
        log.warning(
            "Skipping DB upsert (%s: %s). Schedule JSON was still written.",
            type(e).__name__, e,
        )
        return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n = poll()
    print(f"Polled {n} new bouts.")
