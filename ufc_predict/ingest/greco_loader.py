"""
Load Greco1899/scrape_ufc_stats CSVs into the canonical DB.

Actual Greco1899 CSV schema (confirmed by inspection):

  ufc_event_details.csv  : EVENT, URL, DATE, LOCATION
  ufc_fighter_details.csv: FIRST, LAST, NICKNAME, URL
  ufc_fighter_tott.csv   : FIGHTER, HEIGHT, WEIGHT, REACH, STANCE, DOB, URL
  ufc_fight_details.csv  : EVENT, BOUT, URL
  ufc_fight_results.csv  : EVENT, BOUT, OUTCOME, WEIGHTCLASS, METHOD, ROUND,
                           TIME, TIME FORMAT, REFEREE, DETAILS, URL
  ufc_fight_stats.csv    : EVENT, BOUT, ROUND, FIGHTER, KD, SIG.STR.,
                           SIG.STR. %, TOTAL STR., TD, TD %, SUB.ATT, REV.,
                           CTRL, HEAD, BODY, LEG, DISTANCE, CLINCH, GROUND

BOUT strings look like "Fighter A vs. Fighter B".
OUTCOME strings look like "W/L" (first won) or "L/W" (second won).
ROUND strings look like "Round 1", "Round 2", etc.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from ufc_predict.db.models import Event, Fight, Fighter, FightStatsRound
from ufc_predict.db.session import get_session_factory

log = logging.getLogger(__name__)

_CSV = {
    "event_details":  "ufc_event_details.csv",
    "fighter_det":    "ufc_fighter_details.csv",
    "fighter_tott":   "ufc_fighter_tott.csv",
    "fight_details":  "ufc_fight_details.csv",
    "fight_results":  "ufc_fight_results.csv",
    "fight_stats":    "ufc_fight_stats.csv",
}

_TIME_RE = re.compile(r"^(\d+):(\d{2})$")


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------

def _id(url) -> str | None:
    """Extract the hex ID from a UFCStats URL."""
    if pd.isna(url):
        return None
    s = str(url).rstrip("/").split("/")[-1].strip()
    return s if s else None


def _norm(name) -> str:
    """Lowercase + collapse whitespace — used for name-based joins."""
    return " ".join(str(name).lower().split())


def _parse_ctrl(val) -> int:
    """'2:34' → 154 seconds.  Missing/bad → 0."""
    if pd.isna(val):
        return 0
    m = _TIME_RE.match(str(val).strip())
    return int(m.group(1)) * 60 + int(m.group(2)) if m else 0


def _landed_attempted(val) -> tuple[int, int]:
    """'11 of 38' → (11, 38).  Bad input → (0, 0)."""
    if pd.isna(val):
        return 0, 0
    parts = str(val).split(" of ")
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    return 0, 0


def _safe_int(val) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _parse_round(val) -> int | None:
    """'Round 1' → 1.  Any non-numeric → None (skip row)."""
    if pd.isna(val):
        return None
    m = re.search(r"\d+", str(val))
    return int(m.group()) if m else None


def _parse_height_cm(val) -> float | None:
    if pd.isna(val) or str(val).strip() in ("--", ""):
        return None
    m = re.match(r"(\d+)'\s*(\d+)", str(val).strip())
    if m:
        return round((int(m.group(1)) * 12 + int(m.group(2))) * 2.54, 1)
    return None


def _parse_reach_cm(val) -> float | None:
    if pd.isna(val) or str(val).strip() in ("--", ""):
        return None
    try:
        return round(float(re.sub(r"[\"']", "", str(val).strip())) * 2.54, 1)
    except ValueError:
        return None


def _parse_dob(val) -> date | None:
    if pd.isna(val) or str(val).strip() in ("--", ""):
        return None
    try:
        return pd.to_datetime(val).date()
    except Exception:
        return None


def _parse_date(val) -> date | None:
    if pd.isna(val):
        return None
    try:
        return pd.to_datetime(val).date()
    except Exception:
        return None


def _normalise_method(raw: str) -> str | None:
    if not raw or pd.isna(raw):
        return None
    r = str(raw).strip().upper()
    if r.startswith("KO") or r.startswith("TKO"):
        return "KO/TKO"
    if r.startswith("SUB"):
        return "SUB"
    if "DEC" in r or "DECISION" in r:
        if "U" in r:
            return "Decision (U)"
        if "S" in r:
            return "Decision (S)"
        if "M" in r:
            return "Decision (M)"
        return "Decision"
    if r in ("DQ", "NC", "CANCELLED", "--"):
        return r
    return raw.strip()


# ---------------------------------------------------------------------------
# Main load function
# ---------------------------------------------------------------------------

def load_all(csv_dir: str | Path, db_url: str | None = None) -> None:
    csv_dir = Path(csv_dir)
    factory = get_session_factory(db_url)

    with factory() as session:
        fighter_id_map = _load_fighters(csv_dir, session)
        session.commit()
        log.info("Fighters committed: %d canonical IDs", len(fighter_id_map))

        event_id_map = _load_events(csv_dir, session)
        session.commit()
        log.info("Events committed: %d", len(event_id_map))

        fight_id_map = _load_fights(csv_dir, session, fighter_id_map, event_id_map)
        session.commit()
        log.info("Fights committed: %d", len(fight_id_map))

        _load_fight_stats(csv_dir, session, fighter_id_map, fight_id_map)
        session.commit()
        log.info("Fight stats committed.")

    log.info("Greco1899 CSV load complete.")


# ---------------------------------------------------------------------------
# Step 1 — fighters
# ---------------------------------------------------------------------------

def _load_fighters(csv_dir: Path, session: Session) -> dict[str, str]:
    """
    Returns name_norm → canonical_fighter_id mapping (used for fight joins).
    Loads from fighter_details (name + URL) joined with fighter_tott (physical stats).
    """
    det_path  = csv_dir / _CSV["fighter_det"]
    tott_path = csv_dir / _CSV["fighter_tott"]

    if not det_path.exists():
        log.error("Missing %s", det_path)
        return {}

    det = pd.read_csv(det_path, dtype=str)
    log.info("fighter_details rows: %d  cols: %s", len(det), list(det.columns))

    tott = pd.read_csv(tott_path, dtype=str) if tott_path.exists() else pd.DataFrame()
    if not tott.empty:
        log.info("fighter_tott rows: %d  cols: %s", len(tott), list(tott.columns))
        # Index tott by URL for O(1) lookup
        tott_by_url = {
            _id(row["URL"]): row.to_dict()
            for _, row in tott.iterrows()
            if not pd.isna(row.get("URL"))
        }
    else:
        tott_by_url = {}

    existing_ids = {f.ufcstats_id for f in session.query(Fighter.ufcstats_id)}
    name_map: dict[str, str] = {}

    for _, row in det.iterrows():
        url_id = _id(row.get("URL"))
        if not url_id:
            continue

        canonical_id = "ufcs_" + url_id
        first = str(row.get("FIRST", "")).strip()
        last  = str(row.get("LAST",  "")).strip()
        full_name = f"{first} {last}".strip()
        if not full_name:
            continue

        name_map[_norm(full_name)] = canonical_id

        if url_id in existing_ids:
            continue

        # Physical stats from tott
        tott_row = tott_by_url.get(url_id, {})
        nickname = row.get("NICKNAME")

        session.add(Fighter(
            canonical_fighter_id=canonical_id,
            ufcstats_id=url_id,
            full_name=full_name,
            nickname=None if pd.isna(nickname) else str(nickname).strip(),
            dob=_parse_dob(tott_row.get("DOB")) if tott_row else None,
            stance=str(tott_row.get("STANCE", "")).strip() or None if tott_row else None,
            height_cm=_parse_height_cm(tott_row.get("HEIGHT")) if tott_row else None,
            reach_cm=_parse_reach_cm(tott_row.get("REACH")) if tott_row else None,
            name_variants=[],
        ))

    log.info("Fighter name map size: %d", len(name_map))
    return name_map


# ---------------------------------------------------------------------------
# Step 2 — events
# ---------------------------------------------------------------------------

def _load_events(csv_dir: Path, session: Session) -> dict[str, str]:
    """Returns event_name → event_id map."""
    path = csv_dir / _CSV["event_details"]
    if not path.exists():
        log.error("Missing %s", path)
        return {}

    df = pd.read_csv(path, dtype=str)
    log.info("event_details rows: %d  cols: %s", len(df), list(df.columns))

    existing = {e.event_id for e in session.query(Event.event_id)}
    event_name_map: dict[str, str] = {}

    for _, row in df.iterrows():
        event_id = _id(row.get("URL"))
        if not event_id:
            continue

        event_name = str(row.get("EVENT", "")).strip()
        event_name_map[event_name] = event_id

        if event_id in existing:
            continue

        event_date = _parse_date(row.get("DATE"))
        if not event_date:
            continue

        location = str(row.get("LOCATION", "")).strip()
        country = location.split(",")[-1].strip() if location else None

        session.add(Event(
            event_id=event_id,
            ufc_event_number=event_name,
            date=event_date,
            location=location,
            country=country,
        ))

    return event_name_map


# ---------------------------------------------------------------------------
# Step 3 — fights
# ---------------------------------------------------------------------------

def _load_fights(
    csv_dir: Path,
    session: Session,
    fighter_id_map: dict[str, str],
    event_name_map: dict[str, str],
) -> dict[str, str]:
    """
    Returns fight_url_id → fight_id map (same value — kept for consistency).
    BOUT format: "Fighter A vs. Fighter B"
    OUTCOME format: "W/L" (first won) or "L/W" (second won)
    """
    results_path = csv_dir / _CSV["fight_results"]
    if not results_path.exists():
        log.error("Missing %s", results_path)
        return {}

    df = pd.read_csv(results_path, dtype=str)
    log.info("fight_results rows: %d  cols: %s", len(df), list(df.columns))

    existing = {f.fight_id for f in session.query(Fight.fight_id)}
    fight_id_map: dict[str, str] = {}

    # Build event_name → event_date lookup
    event_dates: dict[str, date] = {}
    for e in session.query(Event):
        event_dates[e.ufc_event_number] = e.date

    skipped = 0
    for _, row in df.iterrows():
        fight_id = _id(row.get("URL"))
        if not fight_id:
            continue

        fight_id_map[fight_id] = fight_id

        if fight_id in existing:
            continue

        # Parse BOUT string
        bout = str(row.get("BOUT", ""))
        if " vs. " in bout:
            parts = bout.split(" vs. ", 1)
        elif " vs " in bout:
            parts = bout.split(" vs ", 1)
        else:
            skipped += 1
            continue

        name_a, name_b = _norm(parts[0]), _norm(parts[1])
        id_a = fighter_id_map.get(name_a)
        id_b = fighter_id_map.get(name_b)

        if not id_a or not id_b:
            log.debug("Unmapped fighters in bout '%s': a=%s b=%s", bout, id_a, id_b)
            skipped += 1
            continue

        # Parse OUTCOME
        outcome = str(row.get("OUTCOME", "")).strip().upper()
        if outcome == "W/L":
            winner_id = id_a
        elif outcome == "L/W":
            winner_id = id_b
        else:
            winner_id = None  # draw / NC / upcoming

        method = _normalise_method(row.get("METHOD"))

        # Skip NC / no-outcome / upcoming bouts for training data
        # (they go into upcoming_bouts, not fights)
        if method in ("NC", "CANCELLED") or (not method and not winner_id):
            skipped += 1
            continue

        event_name = str(row.get("EVENT", "")).strip()
        event_id   = event_name_map.get(event_name)
        fight_date = event_dates.get(event_name)
        if not event_id or not fight_date:
            skipped += 1
            continue

        time_sec  = _parse_ctrl(row.get("TIME"))
        round_num = _safe_int(row.get("ROUND"))
        fmt       = str(row.get("TIME FORMAT", "")).strip()
        is_five   = fmt.startswith("5")

        weight_class = str(row.get("WEIGHTCLASS", "")).strip()
        title_bout   = "title" in weight_class.lower() or "championship" in weight_class.lower()

        session.add(Fight(
            fight_id=fight_id,
            event_id=event_id,
            date=fight_date,
            red_fighter_id=id_a,
            blue_fighter_id=id_b,
            weight_class=weight_class,
            is_title_bout=title_bout,
            is_five_round=is_five,
            winner_fighter_id=winner_id,
            method=method,
            round_ended=round_num,
            time_ended_sec=time_sec,
            referee=str(row.get("REFEREE", "")).strip() or None,
        ))

    log.info("Fights loaded: %d  skipped: %d", len(fight_id_map) - skipped, skipped)
    return fight_id_map


# ---------------------------------------------------------------------------
# Step 4 — fight stats
# ---------------------------------------------------------------------------

def _load_fight_stats(
    csv_dir: Path,
    session: Session,
    fighter_id_map: dict[str, str],
    fight_id_map: dict[str, str],
) -> None:
    stats_path    = csv_dir / _CSV["fight_stats"]
    details_path  = csv_dir / _CSV["fight_details"]

    if not stats_path.exists():
        log.error("Missing %s", stats_path)
        return

    # Build (event_name, bout_string) → fight_id lookup from fight_details
    bout_to_fight: dict[tuple[str, str], str] = {}
    if details_path.exists():
        details_df = pd.read_csv(details_path, dtype=str)
        for _, row in details_df.iterrows():
            fid = _id(row.get("URL"))
            if fid:
                bout_to_fight[
                    (str(row.get("EVENT", "")).strip(), str(row.get("BOUT", "")).strip())
                ] = fid

    df = pd.read_csv(stats_path, dtype=str)
    log.info("fight_stats rows: %d  cols: %s", len(df), list(df.columns))

    # Collect per-round rows; aggregate to totals (round=0) per (fight, fighter)
    # Structure: {(fight_id, fighter_id): {round: stats_dict}}
    from collections import defaultdict
    data: dict[tuple, dict[int, dict]] = defaultdict(dict)

    existing_keys = {
        (s.fight_id, s.fighter_id, s.round)
        for s in session.query(
            FightStatsRound.fight_id,
            FightStatsRound.fighter_id,
            FightStatsRound.round,
        )
    }

    skipped = 0
    for _, row in df.iterrows():
        event = str(row.get("EVENT", "")).strip()
        bout  = str(row.get("BOUT",  "")).strip()
        fight_id = bout_to_fight.get((event, bout))
        if not fight_id or fight_id not in fight_id_map:
            skipped += 1
            continue

        fighter_norm = _norm(row.get("FIGHTER", ""))
        fighter_id   = fighter_id_map.get(fighter_norm)
        if not fighter_id:
            skipped += 1
            continue

        round_num = _parse_round(row.get("ROUND"))
        if round_num is None:
            skipped += 1
            continue

        sig_l,   sig_a   = _landed_attempted(row.get("SIG.STR."))
        total_l, total_a = _landed_attempted(row.get("TOTAL STR."))
        td_l,    td_a    = _landed_attempted(row.get("TD"))
        head_l,  head_a  = _landed_attempted(row.get("HEAD"))
        body_l,  body_a  = _landed_attempted(row.get("BODY"))
        leg_l,   leg_a   = _landed_attempted(row.get("LEG"))
        dist_l,  dist_a  = _landed_attempted(row.get("DISTANCE"))
        clinch_l,clinch_a= _landed_attempted(row.get("CLINCH"))
        gnd_l,   gnd_a   = _landed_attempted(row.get("GROUND"))

        stats = {
            "kd":        _safe_int(row.get("KD")),
            "sig_l":     sig_l,   "sig_a":    sig_a,
            "total_l":   total_l, "total_a":  total_a,
            "td_l":      td_l,    "td_a":     td_a,
            "head_l":    head_l,  "head_a":   head_a,
            "body_l":    body_l,  "body_a":   body_a,
            "leg_l":     leg_l,   "leg_a":    leg_a,
            "dist_l":    dist_l,  "dist_a":   dist_a,
            "clinch_l":  clinch_l,"clinch_a": clinch_a,
            "gnd_l":     gnd_l,   "gnd_a":    gnd_a,
            "sub_att":   _safe_int(row.get("SUB.ATT")),
            "rev":       _safe_int(row.get("REV.")),
            "ctrl":      _parse_ctrl(row.get("CTRL")),
        }

        key = (fight_id, fighter_id)
        data[key][round_num] = stats

    # Write per-round rows + compute totals (round=0)
    added = 0
    for (fight_id, fighter_id), rounds in data.items():
        for round_num, stats in rounds.items():
            if (fight_id, fighter_id, round_num) not in existing_keys:
                session.add(_make_stats_row(fight_id, fighter_id, round_num, stats))
                added += 1

        # Aggregate to totals row (round=0)
        if (fight_id, fighter_id, 0) not in existing_keys:
            totals = _aggregate_rounds(rounds)
            session.add(_make_stats_row(fight_id, fighter_id, 0, totals))
            added += 1

    log.info("Fight stat rows added: %d  skipped: %d", added, skipped)


def _make_stats_row(fight_id, fighter_id, round_num, s) -> FightStatsRound:
    return FightStatsRound(
        fight_id=fight_id,
        fighter_id=fighter_id,
        round=round_num,
        knockdowns=s["kd"],
        sig_strikes_landed=s["sig_l"],
        sig_strikes_attempted=s["sig_a"],
        total_strikes_landed=s["total_l"],
        total_strikes_attempted=s["total_a"],
        head_landed=s["head_l"],
        head_attempted=s["head_a"],
        body_landed=s["body_l"],
        body_attempted=s["body_a"],
        leg_landed=s["leg_l"],
        leg_attempted=s["leg_a"],
        distance_landed=s["dist_l"],
        distance_attempted=s["dist_a"],
        clinch_landed=s["clinch_l"],
        clinch_attempted=s["clinch_a"],
        ground_landed=s["gnd_l"],
        ground_attempted=s["gnd_a"],
        takedowns_landed=s["td_l"],
        takedowns_attempted=s["td_a"],
        submission_attempts=s["sub_att"],
        reversals=s["rev"],
        control_time_sec=s["ctrl"],
    )


def _aggregate_rounds(rounds: dict[int, dict]) -> dict:
    """Sum all per-round stats to produce a fight-total dict."""
    keys = [
        "kd", "sig_l", "sig_a", "total_l", "total_a",
        "td_l", "td_a", "head_l", "head_a", "body_l", "body_a",
        "leg_l", "leg_a", "dist_l", "dist_a", "clinch_l", "clinch_a",
        "gnd_l", "gnd_a", "sub_att", "rev", "ctrl",
    ]
    totals = {k: 0 for k in keys}
    for stats in rounds.values():
        for k in keys:
            totals[k] = totals[k] + stats.get(k, 0)
    return totals


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    csv_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/greco1899"
    load_all(csv_dir)
