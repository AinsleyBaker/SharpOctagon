"""
Closing odds scraper — BestFightOdds.com.

Past-event pages render the closing-line moneyline table as the *second* tbody
on the page (the first tbody is just labels). Inside it, each fight is two
consecutive `<tr>` rows (no class — `<tr class="pr">` are prop rows we skip).
Each row contains the fighter name plus a `<td class="but-sg" data-li="[B,F,M]">`
cell per bookmaker (B=bookmaker_id, F=1 or 2 fighter index, M=matchup_id).

We take the **median across populated bookmaker cells** as the closing line —
this de-noises booker-specific bias without losing the consensus signal.

Closing odds are NEVER fed into model features. Stored on `fights.closing_odds_*`
for evaluation against Vegas (and on `upcoming_bouts.*` for Kelly sizing).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from statistics import median
from urllib.parse import quote_plus

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
    "Referer": _BASE + "/",
}
_DELAY_S = 2.0
_AMERICAN_RE = re.compile(r"^[+-]?\d+$")
_DATA_LI_RE = re.compile(r"^\[(\d+),(\d+),(\d+)\]$")
# Prop cell data-li format: [bookmaker, fighter_idx, matchup_id, prop_type_id, prop_value]
_DATA_LI_PROP_RE = re.compile(r"^\[(\d+),(\d+),(\d+)(?:,(\d+))?(?:,(\d+))?\]$")


@dataclass
class ScrapedBout:
    matchup_id: int
    fighter1_name: str
    fighter1_bfo_id: str | None
    fighter1_odds: float | None      # median American closing line
    fighter1_n_books: int            # how many bookmakers contributed
    fighter2_name: str
    fighter2_bfo_id: str | None
    fighter2_odds: float | None
    fighter2_n_books: int
    event_date: date | None


@dataclass
class ScrapedProp:
    """One prop bet scraped from a BFO event page.

    `prop_type` is a canonical key like 'distance_yes', 'rounds_over_2.5',
    'f1_method_KO_TKO', 'f1_round_3'. `side` is 'yes' / 'no' / 'over' /
    'under' / specific outcome. `odds` is the American median across
    bookmakers (same approach as moneyline).
    """
    matchup_id: int
    prop_type: str
    side: str
    odds: float
    n_books: int
    raw_label: str


def _get(url: str) -> BeautifulSoup | None:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "lxml")
    except requests.RequestException as exc:
        log.warning("BestFightOdds fetch failed: %s — %s", url, exc)
        return None


def _parse_american(s: str) -> float | None:
    """Parse '+150' / '-200' / '−150' (unicode minus) → float."""
    s = (s or "").strip().replace("−", "-").replace("–", "-")
    return float(s) if _AMERICAN_RE.match(s) else None


def _extract_bfo_id(href: str) -> str | None:
    """`/fighters/Alex-Pereira-10463` → `10463`."""
    m = re.search(r"/fighters/.+-(\d+)/?$", href or "")
    return m.group(1) if m else None


def _moneyline_tbody(soup: BeautifulSoup):
    """The second <tbody> on a BFO event page is the moneyline odds table.
    The first contains label/prop rows only.
    """
    tbodies = soup.find_all("tbody")
    if len(tbodies) < 2:
        return None
    return tbodies[1]


def _parse_event_date(soup: BeautifulSoup) -> date | None:
    """Pull the event date from the JSON-LD `startDate` field (preferred —
    machine-readable). Falls back to prose date parsing in title/meta tags."""
    # 1) JSON-LD structured data — most reliable
    for script in soup.find_all("script", type="application/ld+json"):
        m = re.search(r'"startDate"\s*:\s*"(\d{4}-\d{2}-\d{2})', script.string or "")
        if m:
            try:
                return date.fromisoformat(m.group(1))
            except ValueError:
                pass

    # 2) Fall back to prose like "April 14, 2024" in title or meta description
    for tag in soup.find_all(["title", "meta", "h1", "h2"]):
        txt = tag.get("content", "") if tag.name == "meta" else tag.get_text(" ", strip=True)
        m = re.search(
            r"\b(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+(\d{1,2}),\s+(\d{4})\b",
            txt,
        )
        if m:
            try:
                return datetime.strptime(m.group(0), "%B %d, %Y").date()
            except ValueError:
                continue
    return None


def _parse_fighter_row(tr) -> dict | None:
    """Extract fighter identity + per-bookmaker prices from one moneyline TR."""
    a = tr.find("a", href=re.compile(r"^/fighters/"))
    if a is None:
        return None
    name = a.get_text(strip=True)
    bfo_id = _extract_bfo_id(a.get("href", ""))

    prices: list[float] = []
    fighter_idx: int | None = None
    matchup_id: int | None = None
    for td in tr.find_all("td", class_="but-sg"):
        m = _DATA_LI_RE.match(td.get("data-li", ""))
        if not m:
            continue
        _bookie, fidx, mu = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # The first numeric span in the cell holds the price
        span = td.find("span", id=re.compile(r"^oID"))
        if span is None:
            continue
        price = _parse_american(span.get_text(strip=True))
        if price is None:
            continue
        prices.append(price)
        if fighter_idx is None:
            fighter_idx, matchup_id = fidx, mu

    return {
        "name": name,
        "bfo_id": bfo_id,
        "fighter_idx": fighter_idx,
        "matchup_id": matchup_id,
        "prices": prices,
    }


def scrape_event(slug: str) -> list[ScrapedBout]:
    """Scrape closing moneyline odds for one past or upcoming event.

    `slug` may be the full path (`/events/ufc-300-3205`) or just the slug suffix.
    """
    if not slug.startswith("/"):
        slug = "/events/" + slug.lstrip("/")
    url = _BASE + slug
    soup = _get(url)
    if soup is None:
        return []

    tbody = _moneyline_tbody(soup)
    if tbody is None:
        log.warning("BFO: moneyline tbody not found at %s", url)
        return []

    event_date = _parse_event_date(soup)

    # Fighter rows are <tr> with no class. Prop rows are <tr class="pr"> — skip.
    fighter_rows = [
        tr for tr in tbody.find_all("tr", recursive=False)
        if not tr.get("class")
    ]
    # Group by matchup_id, preserving order so fighter_idx==1 row comes first
    by_mu: dict[int, list[dict]] = {}
    for tr in fighter_rows:
        parsed = _parse_fighter_row(tr)
        if not parsed or parsed["matchup_id"] is None:
            continue
        by_mu.setdefault(parsed["matchup_id"], []).append(parsed)

    bouts: list[ScrapedBout] = []
    for mu_id, rows in by_mu.items():
        if len(rows) != 2:
            log.debug("BFO: matchup %s has %d rows, skipping", mu_id, len(rows))
            continue
        rows.sort(key=lambda r: r["fighter_idx"])  # 1 then 2
        f1, f2 = rows[0], rows[1]
        bouts.append(ScrapedBout(
            matchup_id=mu_id,
            fighter1_name=f1["name"],
            fighter1_bfo_id=f1["bfo_id"],
            fighter1_odds=median(f1["prices"]) if f1["prices"] else None,
            fighter1_n_books=len(f1["prices"]),
            fighter2_name=f2["name"],
            fighter2_bfo_id=f2["bfo_id"],
            fighter2_odds=median(f2["prices"]) if f2["prices"] else None,
            fighter2_n_books=len(f2["prices"]),
            event_date=event_date,
        ))

    log.info("BFO: scraped %d bouts from %s (date=%s)", len(bouts), slug, event_date)
    return bouts


# ---------------------------------------------------------------------------
# Prop bet scraping
# ---------------------------------------------------------------------------

# Markets we predict and care to backtest. Each handler returns
# (prop_type, side) given (label_text, last_named_prop, fighter1_lastname,
# fighter2_lastname). Returns None to skip.
_TOTAL_RE = re.compile(r"^(Over|Under)\s+([\d½]+)\s+rounds?$", re.I)
_STARTS_RE = re.compile(r"^Fight\s+(starts|won't start)\s+round\s+(\d)$", re.I)
_ENDS_RE = re.compile(r"^Fight\s+(ends|doesn't end)\s+in\s+round\s+(\d)$", re.I)
_METHOD_RE = re.compile(
    r"^(.+?)\s+wins\s+by\s+"
    r"(TKO/KO|submission|decision|unanimous decision|split/majority decision)$",
    re.I,
)
_INSIDE_RE = re.compile(
    r"^(?:(.+?)\s+wins\s+inside\s+distance|Not\s+(.+?)\s+inside\s+distance)$",
    re.I,
)
_DEC_NEG_RE = re.compile(r"^Not\s+(.+?)\s+by\s+decision$", re.I)
_ROUND_RE = re.compile(r"^(.+?)\s+wins\s+in\s+round\s+(\d)$", re.I)


def _label_to_half(label: str) -> str:
    """Normalize '1½' / '2½' / etc. — labels use the unicode fraction."""
    return label.replace("½", ".5")


def _which_fighter(name_in_label: str, f1_last: str, f2_last: str) -> str | None:
    """Match a name fragment in a label to fighter 1 or fighter 2 by last name."""
    if not name_in_label:
        return None
    n = name_in_label.lower().strip()
    if f1_last and f1_last.lower() in n:
        return "f1"
    if f2_last and f2_last.lower() in n:
        return "f2"
    return None


def _classify_prop(label: str, prev_prop: tuple | None,
                   f1_last: str, f2_last: str) -> tuple[str, str] | None:
    """Map a prop row label → (prop_type, side). Returns None for markets we
    don't currently predict (UD vs SMD split, takedown counts, etc.).

    `prev_prop` is the (prop_type, side) tuple from the previous parsed row,
    used to interpret 'Any other result' as the negation of the prior prop.
    """
    s = label.strip()
    if not s:
        return None

    # ---- Distance market ----
    if s.lower() == "fight goes to decision":
        return ("distance", "yes")
    if s.lower() == "fight doesn't go to decision":
        return ("distance", "no")

    # ---- Total rounds ----
    m = _TOTAL_RE.match(s)
    if m:
        direction = m.group(1).lower()
        line = _label_to_half(m.group(2))
        return (f"total_rounds_{line}", "over" if direction == "over" else "under")

    # ---- Starts round N ----
    m = _STARTS_RE.match(s)
    if m:
        direction, n = m.group(1).lower(), m.group(2)
        return (f"starts_round_{n}", "yes" if "won't" not in direction else "no")

    # ---- Ends in round N ----
    m = _ENDS_RE.match(s)
    if m:
        direction, n = m.group(1).lower(), m.group(2)
        return (f"ends_round_{n}", "yes" if "doesn't" not in direction else "no")

    # ---- Method per fighter ----
    m = _METHOD_RE.match(s)
    if m:
        name, method = m.group(1).strip(), m.group(2).lower()
        side = _which_fighter(name, f1_last, f2_last)
        if not side:
            return None
        method_key = {
            "tko/ko": "KO_TKO",
            "submission": "SUB",
            "decision": "DEC",
            "unanimous decision": "UD",
            "split/majority decision": "SMD",
        }.get(method)
        if not method_key:
            return None
        return (f"{side}_method_{method_key}", "yes")

    # ---- Inside distance ----
    m = _INSIDE_RE.match(s)
    if m:
        name = m.group(1) or m.group(2)
        is_negation = bool(m.group(2))
        side = _which_fighter(name, f1_last, f2_last)
        if not side:
            return None
        return (f"{side}_inside_distance", "no" if is_negation else "yes")

    # ---- Wins in round N ----
    m = _ROUND_RE.match(s)
    if m:
        name, n = m.group(1).strip(), m.group(2)
        side = _which_fighter(name, f1_last, f2_last)
        if not side:
            return None
        return (f"{side}_wins_round_{n}", "yes")

    # ---- "Any other result" — negate the previous prop ----
    if s.lower() == "any other result" and prev_prop:
        ptype, _ = prev_prop
        return (ptype, "no")

    # ---- Skipped patterns: UD/SMD splits, more strikes/TDs, draw, etc. ----
    return None


def scrape_event_props(slug: str) -> tuple[list[ScrapedBout], list[ScrapedProp]]:
    """Scrape both moneyline AND prop closing odds for a past event.

    Returns (bouts, props). Bouts is the same as `scrape_event(slug)`.
    Props is the parsed prop rows, keyed by matchup_id and classified into
    canonical prop_type / side strings — ready to persist or backtest.
    """
    bouts = scrape_event(slug)
    if not bouts:
        return [], []

    bouts_by_mu = {b.matchup_id: b for b in bouts}

    # Re-fetch and find tr.pr rows
    if not slug.startswith("/"):
        slug = "/events/" + slug.lstrip("/")
    soup = _get(_BASE + slug)
    if soup is None:
        return bouts, []

    tbody = _moneyline_tbody(soup)
    if tbody is None:
        return bouts, []

    pr_rows = tbody.find_all("tr", class_="pr", recursive=False)

    # Aggregate prices by (matchup, prop_type, side) — BFO often emits
    # multiple rows for the same logical prop (different sub-market IDs)
    # so we collect all prices across the duplicates and median once at end.
    accum: dict[tuple[int, str, str], dict] = {}
    prev_by_mu: dict[int, tuple[str, str]] = {}

    for tr in pr_rows:
        # Find matchup_id via any but-sgp cell's data-li
        matchup_id = None
        prices: list[float] = []
        for td in tr.find_all("td", class_="but-sgp"):
            m = _DATA_LI_PROP_RE.match(td.get("data-li", ""))
            if not m:
                continue
            mu = int(m.group(3))
            if matchup_id is None:
                matchup_id = mu
            span = td.find("span", id=re.compile(r"^oID"))
            if span is None:
                continue
            price = _parse_american(span.get_text(strip=True))
            if price is not None:
                prices.append(price)

        if matchup_id is None or matchup_id not in bouts_by_mu:
            continue
        if not prices:
            continue

        bout = bouts_by_mu[matchup_id]
        f1_last = (bout.fighter1_name or "").split()[-1] if bout.fighter1_name else ""
        f2_last = (bout.fighter2_name or "").split()[-1] if bout.fighter2_name else ""

        th = tr.find("th")
        if not th:
            continue
        label = th.get_text(strip=True).replace(" ", " ").replace("&#189;", "½")
        # Normalize unicode half-fraction
        label = label.replace("½", "½")

        classified = _classify_prop(label, prev_by_mu.get(matchup_id), f1_last, f2_last)
        if classified is None:
            continue
        prop_type, side = classified

        key = (matchup_id, prop_type, side)
        bucket = accum.setdefault(key, {"prices": [], "raw_label": label})
        bucket["prices"].extend(prices)
        prev_by_mu[matchup_id] = classified

    props: list[ScrapedProp] = []
    for (mu, ptype, side), bucket in accum.items():
        if not bucket["prices"]:
            continue
        props.append(ScrapedProp(
            matchup_id=mu,
            prop_type=ptype,
            side=side,
            odds=median(bucket["prices"]),
            n_books=len(bucket["prices"]),
            raw_label=bucket["raw_label"],
        ))

    log.info("BFO: scraped %d prop rows (after dedup) from %s", len(props), slug)
    return bouts, props


def search_event_slug(
    query: str,
    *,
    ufc_event_number: str | None = None,
    near_date: date | None = None,
    date_window_days: int = 7,
) -> str | None:
    """Find an event page slug via the BFO site search.

    BFO search returns many UFC events for any "UFC" query, ordered by
    relevance which is unreliable for numbered cards. We filter results:

      1. If `ufc_event_number` is given (e.g. "280"), require slugs matching
         `^ufc-{N}-\\d+$` — the strict numbered-card pattern.
      2. Else require any slug starting with `ufc-`.
      3. If `near_date` is given, fetch each candidate's page and pick the
         one whose JSON-LD startDate is closest within `date_window_days`.
    """
    url = f"{_BASE}/search?query={quote_plus(query)}"
    soup = _get(url)
    if soup is None:
        return None

    if ufc_event_number:
        # Two slug formats observed:
        #   newer: /events/ufc-300-3205
        #   older: /events/ufc-200-tate-vs-nunes-1102 (tagline between number and id)
        slug_re = re.compile(
            rf"^/events/ufc-{re.escape(str(ufc_event_number))}(?:-[a-z0-9\-]+)?-\d+$"
        )
    else:
        slug_re = re.compile(r"^/events/ufc-[a-z0-9\-]+$")

    candidates: list[str] = []
    for a in soup.find_all("a", href=slug_re):
        href = a.get("href", "")
        if href and href not in candidates:
            candidates.append(href)
    if not candidates:
        return None
    if near_date is None:
        return candidates[0]

    # When `near_date` is given, taglines like "Sterling vs. Zalal" can collide
    # with old events. Require a date match within the window — do NOT fall
    # back to the first candidate, which would silently bind the wrong event.
    # This applies even when len(candidates) == 1: a single match with the
    # wrong date is still wrong.
    near = near_date if isinstance(near_date, date) else date.fromisoformat(str(near_date))
    best, best_gap = None, None
    for href in candidates[:8]:
        time.sleep(_DELAY_S)
        ev_soup = _get(_BASE + href)
        if ev_soup is None:
            continue
        d = _parse_event_date(ev_soup)
        if d is None:
            continue
        gap = abs((d - near).days)
        if gap <= date_window_days and (best_gap is None or gap < best_gap):
            best, best_gap = href, gap
    return best


# ---------------------------------------------------------------------------
# DB matching + persistence
# ---------------------------------------------------------------------------

_NAME_NORM_RE = re.compile(r"[^a-z0-9]+")


def _normalize_name(name: str) -> str:
    return _NAME_NORM_RE.sub(" ", (name or "").lower()).strip()


def _name_key(name: str) -> tuple[str, str]:
    """(last_token, first_initial) — robust to extra middle names.
    'Jesus Santos Aguilar' and 'Jesus Aguilar' both → ('aguilar', 'j').
    """
    toks = _normalize_name(name).split()
    if not toks:
        return ("", "")
    return (toks[-1], toks[0][:1] if toks[0] else "")


def update_fight_odds_for_event(
    bouts: list[ScrapedBout],
    event_date,
    session: Session,
    date_window_days: int = 3,
) -> int:
    if not isinstance(event_date, date):
        event_date = date.fromisoformat(str(event_date))
    """Match scraped bouts to fights within ±N days of `event_date` by fighter
    name and update closing_odds_red/blue. Returns rows updated.

    A bout matches a fight when:
      - both scraped fighter names normalize-equal to the fight's red/blue
        canonical names (in either red/blue assignment), AND
      - the fight's date is within `date_window_days` of `event_date`.
    """
    # Pull candidate fights once
    sql = text("""
        SELECT f.fight_id, f.date, f.red_fighter_id, f.blue_fighter_id,
               fr.full_name, fb.full_name,
               f.closing_odds_red, f.closing_odds_blue
        FROM fights f
        JOIN fighters fr ON fr.canonical_fighter_id = f.red_fighter_id
        JOIN fighters fb ON fb.canonical_fighter_id = f.blue_fighter_id
        WHERE ABS(julianday(f.date) - julianday(:d)) <= :win
    """)
    rows = session.execute(sql, {"d": event_date, "win": date_window_days}).fetchall()

    # Key by (lastname, first_initial) pair — handles middle-name variants
    by_pair: dict[frozenset, tuple] = {}
    for r in rows:
        key = frozenset({_name_key(r[4]), _name_key(r[5])})
        by_pair[key] = r

    updated = 0
    for b in bouts:
        if b.fighter1_odds is None or b.fighter2_odds is None:
            continue
        key = frozenset({_name_key(b.fighter1_name), _name_key(b.fighter2_name)})
        row = by_pair.get(key)
        if row is None:
            log.debug("BFO: no fight match for %s vs %s near %s",
                      b.fighter1_name, b.fighter2_name, event_date)
            continue
        # Decide which side is red vs blue
        if _name_key(row[4]) == _name_key(b.fighter1_name):
            ro, bo = b.fighter1_odds, b.fighter2_odds
        else:
            ro, bo = b.fighter2_odds, b.fighter1_odds
        session.execute(
            text("""UPDATE fights SET closing_odds_red=:ro, closing_odds_blue=:bo
                    WHERE fight_id=:fid"""),
            {"ro": ro, "bo": bo, "fid": row[0]},
        )
        updated += 1

    session.commit()
    return updated


_NUMBERED_CARD_RE = re.compile(r"^UFC\s+(\d{2,3})\b", re.I)
_TAGLINE_RE = re.compile(r":\s*(.+)$")


def update_props_for_event(
    bouts: list[ScrapedBout],
    props: list[ScrapedProp],
    event_date,
    session: Session,
    date_window_days: int = 3,
) -> int:
    """Match scraped bouts (by fighter pairs near event_date) to fights in
    our DB, then persist each prop into fight_prop_odds keyed by the matched
    fight_id. Idempotent — uses INSERT OR REPLACE on the unique constraint.
    """
    from datetime import datetime as _dt
    if not isinstance(event_date, date):
        event_date = date.fromisoformat(str(event_date))

    rows = session.execute(text("""
        SELECT f.fight_id, f.date, fr.full_name, fb.full_name
        FROM fights f
        JOIN fighters fr ON fr.canonical_fighter_id = f.red_fighter_id
        JOIN fighters fb ON fb.canonical_fighter_id = f.blue_fighter_id
        WHERE ABS(julianday(f.date) - julianday(:d)) <= :win
    """), {"d": event_date, "win": date_window_days}).fetchall()

    # Index by fighter pair AND store red/blue last names so we can canonicalise
    # BFO's f1/f2 ordering to our DB red/blue ordering at persist time.
    by_pair: dict[frozenset, tuple[str, str, str]] = {}
    for r in rows:
        key = frozenset({_name_key(r[2]), _name_key(r[3])})
        # (fight_id, red_lastname, blue_lastname) — keyed by name_key so we can
        # match BFO's last-name to the correct corner.
        by_pair[key] = (r[0], _name_key(r[2])[0], _name_key(r[3])[0])

    # For each matchup, determine f1↔red mapping
    mu_info: dict[int, tuple[str, bool]] = {}  # matchup_id → (fight_id, f1_is_red)
    for b in bouts:
        key = frozenset({_name_key(b.fighter1_name), _name_key(b.fighter2_name)})
        info = by_pair.get(key)
        if not info:
            continue
        fight_id, red_last, _blue_last = info
        f1_last = _name_key(b.fighter1_name)[0]
        f1_is_red = (f1_last == red_last)
        mu_info[b.matchup_id] = (fight_id, f1_is_red)

    if not mu_info:
        return 0

    def _canonicalise(prop_type: str, f1_is_red: bool) -> str:
        """Rewrite f1_ / f2_ prefixes to canonical r_ / b_ based on which BFO
        fighter sits in the red corner. Non-fighter-attributed props (distance,
        total_rounds, starts_round_N, ends_round_N) pass through unchanged."""
        if prop_type.startswith("f1_"):
            return ("r_" if f1_is_red else "b_") + prop_type[3:]
        if prop_type.startswith("f2_"):
            return ("b_" if f1_is_red else "r_") + prop_type[3:]
        return prop_type

    inserted = 0
    now = _dt.utcnow()
    for p in props:
        info = mu_info.get(p.matchup_id)
        if not info:
            continue
        fight_id, f1_is_red = info
        canon_type = _canonicalise(p.prop_type, f1_is_red)
        session.execute(text("""
            INSERT OR REPLACE INTO fight_prop_odds
                (fight_id, prop_type, side, american_odds, n_books, raw_label, scraped_at)
            VALUES (:fid, :pt, :sd, :odds, :n, :lab, :ts)
        """), {
            "fid": fight_id, "pt": canon_type, "sd": p.side,
            "odds": p.odds, "n": p.n_books, "lab": p.raw_label, "ts": now,
        })
        inserted += 1
    session.commit()
    return inserted


def backfill_props_event(event_id: str, session: Session) -> tuple[int, str]:
    """Look up an event in our DB, find its BFO slug, scrape both moneyline
    and props, and persist props to fight_prop_odds.
    Returns (n_props_inserted, status).
    """
    row = session.execute(
        text("SELECT date, ufc_event_number FROM events WHERE event_id=:e"),
        {"e": event_id},
    ).first()
    if row is None:
        return 0, "event_not_in_db"
    event_date, ev_name = row[0], row[1] or ""

    m = _NUMBERED_CARD_RE.match(ev_name)
    if m:
        slug = search_event_slug(
            f"UFC {m.group(1)}", ufc_event_number=m.group(1),
            near_date=event_date,
        )
    else:
        tag_m = _TAGLINE_RE.search(ev_name)
        query = tag_m.group(1).strip() if tag_m else ev_name
        slug = search_event_slug(query, near_date=event_date)
    if not slug:
        return 0, "no_slug"
    time.sleep(_DELAY_S)
    bouts, props = scrape_event_props(slug)
    if not props:
        return 0, "no_props"
    n = update_props_for_event(bouts, props, event_date, session)
    return n, "ok"


def backfill_props_all(
    session: Session,
    limit: int | None = None,
    earliest: date | None = None,
    latest: date | None = None,
) -> dict:
    """Walk events that already have moneyline closing odds and scrape their
    props. Same date bounds as the moneyline backfill. Run after the
    moneyline backfill (which populates the slug-mapping side effect).
    """
    from datetime import timedelta
    today = date.today()
    earliest = earliest or date(2021, 1, 1)
    latest = latest or (today - timedelta(days=30))

    rows = session.execute(text("""
        SELECT DISTINCT e.event_id, e.date, e.ufc_event_number
        FROM events e
        JOIN fights f ON f.event_id = e.event_id
        WHERE f.closing_odds_red IS NOT NULL
        ORDER BY e.date ASC
    """)).fetchall()
    rows = [
        r for r in rows
        if earliest <= date.fromisoformat(str(r[1])) <= latest
    ]
    if limit:
        rows = rows[:limit]

    stats = {"ok": 0, "no_slug": 0, "no_props": 0, "event_not_in_db": 0,
             "total_props_inserted": 0, "events_attempted": 0}
    for ev_id, ev_date, ufc_num in rows:
        n, status = backfill_props_event(ev_id, session)
        stats[status] += 1
        stats["total_props_inserted"] += n
        stats["events_attempted"] += 1
        log.info("event %s (%s, %s): %s — %d props", ev_id, ev_date, ufc_num, status, n)
        time.sleep(_DELAY_S)
    return stats


def backfill_event(event_id: str, session: Session) -> tuple[int, str]:
    """Look up an event in our DB, find its BFO slug, scrape, and persist.
    Returns (rows_updated, status) where status is one of:
      'ok' | 'no_slug' | 'no_bouts' | 'event_not_in_db'
    """
    row = session.execute(
        text("SELECT date, ufc_event_number, location FROM events WHERE event_id=:e"),
        {"e": event_id},
    ).first()
    if row is None:
        return 0, "event_not_in_db"
    event_date, ev_name, _loc = row[0], row[1] or "", row[2]

    # `ufc_event_number` actually holds the full event title like
    # "UFC 327: Prochazka vs. Ulberg" or "UFC Fight Night: Sterling vs. Zalal".
    # Numbered cards: extract the integer; Fight Nights: search by tagline.
    m = _NUMBERED_CARD_RE.match(ev_name)
    if m:
        slug = search_event_slug(
            f"UFC {m.group(1)}",
            ufc_event_number=m.group(1),
            near_date=event_date,
        )
    else:
        # Search by the headliner tagline ("Sterling vs. Zalal")
        tag_m = _TAGLINE_RE.search(ev_name)
        query = tag_m.group(1).strip() if tag_m else ev_name
        slug = search_event_slug(query, near_date=event_date)
    if not slug:
        return 0, "no_slug"
    time.sleep(_DELAY_S)
    bouts = scrape_event(slug)
    if not bouts:
        return 0, "no_bouts"
    n = update_fight_odds_for_event(bouts, event_date, session)
    return n, "ok"


def backfill_all(
    session: Session,
    limit: int | None = None,
    earliest: date | None = None,
    latest: date | None = None,
) -> dict:
    """Walk every event in our DB lacking closing-odds coverage, attempt a BFO
    backfill, and return summary stats.

    Bounds work to [`earliest`, `latest`]. Defaults: earliest = 2021-01-01
    (BFO's archived odds coverage begins around then) and latest = today − 30
    days (BFO archives appear to lag the actual event by weeks). Events are
    processed oldest-first so the high-hit-rate range completes early.
    """
    from datetime import timedelta
    today = date.today()
    earliest = earliest or date(2021, 1, 1)
    latest = latest or (today - timedelta(days=30))
    rows = session.execute(text("""
        SELECT e.event_id, e.date, e.ufc_event_number,
               COUNT(*) AS n_fights,
               SUM(CASE WHEN f.closing_odds_red IS NULL THEN 1 ELSE 0 END) AS n_missing
        FROM events e
        JOIN fights f ON f.event_id = e.event_id
        GROUP BY e.event_id
        HAVING n_missing > 0
        ORDER BY e.date ASC
    """)).fetchall()
    rows = [
        r for r in rows
        if earliest <= date.fromisoformat(str(r[1])) <= latest
    ]

    if limit:
        rows = rows[:limit]

    stats = {"ok": 0, "no_slug": 0, "no_bouts": 0, "event_not_in_db": 0,
             "total_rows_updated": 0, "events_attempted": 0}
    for ev_id, ev_date, ufc_num, _nf, _nm in rows:
        n, status = backfill_event(ev_id, session)
        stats[status] += 1
        stats["total_rows_updated"] += n
        stats["events_attempted"] += 1
        log.info("event %s (%s, UFC %s): %s — %d rows", ev_id, ev_date, ufc_num, status, n)
        time.sleep(_DELAY_S)
    return stats


def update_upcoming_kelly(bouts: list[ScrapedBout], session: Session) -> None:
    """Compute Kelly fractions for bouts in `data/predictions.json` using best
    median closing-line odds.
    """
    import json
    from pathlib import Path

    from ufc_predict.eval.evaluate import american_to_decimal, kelly_fraction_fn

    preds_path = Path("data/predictions.json")
    if not preds_path.exists():
        return
    with open(preds_path) as f:
        preds = json.load(f)

    name_to_odds: dict[str, float] = {}
    for b in bouts:
        if b.fighter1_odds is not None:
            name_to_odds[_normalize_name(b.fighter1_name)] = b.fighter1_odds
        if b.fighter2_odds is not None:
            name_to_odds[_normalize_name(b.fighter2_name)] = b.fighter2_odds

    out = []
    for p in preds:
        a_norm = _normalize_name(p.get("fighter_a_name", ""))
        odds = name_to_odds.get(a_norm)
        if odds is not None:
            dec = american_to_decimal(odds)
            prob_a = p.get("prob_a_wins") or 0.5
            implied = 1.0 / dec
            p["kelly_fraction"] = round(kelly_fraction_fn(prob_a, dec), 4)
            p["has_edge"] = prob_a > implied
            p["implied_prob_a"] = round(implied, 4)
        out.append(p)

    with open(preds_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info("Updated Kelly fractions in predictions.json")


def run(db_url: str | None = None) -> None:
    """Default entrypoint: scrape upcoming events, update Kelly on predictions."""
    from ufc_predict.db.session import get_session_factory
    factory = get_session_factory(db_url)

    soup = _get(_BASE + "/")
    if soup is None:
        return
    upcoming_slugs: list[str] = []
    for a in soup.find_all("a", href=re.compile(r"^/events/ufc-")):
        href = a.get("href")
        if href and href not in upcoming_slugs:
            upcoming_slugs.append(href)

    all_bouts: list[ScrapedBout] = []
    for slug in upcoming_slugs[:5]:
        all_bouts.extend(scrape_event(slug))
        time.sleep(_DELAY_S)

    with factory() as session:
        update_upcoming_kelly(all_bouts, session)
    log.info("Odds run complete — %d bouts scraped from %d upcoming events",
             len(all_bouts), len(upcoming_slugs))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
