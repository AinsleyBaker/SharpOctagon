"""
Load Greco1899/scrape_ufc_stats CSVs into the canonical DB.

Usage:
    python -m ufc_predict.ingest.greco_loader --csv-dir ./data/raw/greco1899
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from ufc_predict.db.models import Event, Fight, Fighter, FightStatsRound
from ufc_predict.db.session import get_session_factory

log = logging.getLogger(__name__)

# Greco1899 CSV filenames
_CSV_FILES = {
    "events":       "ufc_events.csv",
    "results":      "ufc_fight_results.csv",
    "fight_stats":  "ufc_fight_stats.csv",
    "fighter_tott": "ufc_fighter_tott.csv",
    "fighter_det":  "ufc_fighter_details.csv",
    "fight_det":    "ufc_fight_details.csv",
}

_TIME_RE = re.compile(r"^(\d+):(\d{2})$")


def _parse_ctrl(val) -> int | None:
    """Convert mm:ss control time string to seconds. Returns None for NaN/missing."""
    if pd.isna(val):
        return None
    m = _TIME_RE.match(str(val).strip())
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


def _safe_int(val) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _parse_pct(val) -> float | None:
    """'75%' → 0.75"""
    if pd.isna(val):
        return None
    s = str(val).strip().rstrip("%")
    try:
        return float(s) / 100
    except ValueError:
        return None


def _parse_landed_attempted(val) -> tuple[int | None, int | None]:
    """'12 of 34' → (12, 34)"""
    if pd.isna(val):
        return None, None
    parts = str(val).split(" of ")
    if len(parts) != 2:
        return None, None
    return _safe_int(parts[0]), _safe_int(parts[1])


def load_all(csv_dir: str | Path, db_url: str | None = None) -> None:
    csv_dir = Path(csv_dir)
    factory = get_session_factory(db_url)

    with factory() as session:
        _load_fighters(csv_dir, session)
        session.commit()
        _load_events(csv_dir, session)
        session.commit()
        _load_fights(csv_dir, session)
        session.commit()
        _load_fight_stats(csv_dir, session)
        session.commit()

    log.info("Greco1899 CSV load complete.")


def _load_fighters(csv_dir: Path, session: Session) -> None:
    path = csv_dir / _CSV_FILES["fighter_det"]
    if not path.exists():
        log.warning("Fighter details CSV not found: %s", path)
        return

    df = pd.read_csv(path)
    log.info("Loading %d fighters", len(df))

    existing = {f.ufcstats_id for f in session.query(Fighter.ufcstats_id)}

    for _, row in df.iterrows():
        ufcstats_id = str(row.get("fighter_url", "")).split("/")[-1]
        if not ufcstats_id or ufcstats_id in existing:
            continue

        canonical_id = "ufcs_" + ufcstats_id

        dob_raw = row.get("fighter_dob")
        dob = None
        if not pd.isna(dob_raw):
            try:
                dob = pd.to_datetime(dob_raw).date()
            except Exception:
                pass

        fighter = Fighter(
            canonical_fighter_id=canonical_id,
            ufcstats_id=ufcstats_id,
            full_name=str(row.get("fighter_f_name", "")).strip()
                      + " "
                      + str(row.get("fighter_l_name", "")).strip(),
            nickname=row.get("fighter_nickname") if not pd.isna(row.get("fighter_nickname", float("nan"))) else None,
            dob=dob,
            stance=row.get("fighter_stance") if not pd.isna(row.get("fighter_stance", float("nan"))) else None,
            height_cm=_inches_to_cm(row.get("fighter_height")),
            reach_cm=_inches_to_cm(row.get("fighter_reach")),
            name_variants=[],
        )
        session.add(fighter)

    log.info("Fighters upserted.")


def _inches_to_cm(val) -> float | None:
    """'72\"' or '6\' 0\"' → cm"""
    if pd.isna(val):
        return None
    s = str(val).strip()
    # feet' inches" format
    m = re.match(r"(\d+)'\s*(\d+)\"", s)
    if m:
        return round((int(m.group(1)) * 12 + int(m.group(2))) * 2.54, 1)
    # plain inches
    m = re.match(r"([\d.]+)\"?", s)
    if m:
        return round(float(m.group(1)) * 2.54, 1)
    return None


def _load_events(csv_dir: Path, session: Session) -> None:
    path = csv_dir / _CSV_FILES["events"]
    if not path.exists():
        log.warning("Events CSV not found: %s", path)
        return

    df = pd.read_csv(path)
    log.info("Loading %d events", len(df))

    existing = {e.event_id for e in session.query(Event.event_id)}

    for _, row in df.iterrows():
        event_id = str(row.get("event_url", "")).split("/")[-1]
        if not event_id or event_id in existing:
            continue

        try:
            event_date = pd.to_datetime(row["event_date"]).date()
        except Exception:
            continue

        session.add(Event(
            event_id=event_id,
            ufc_event_number=row.get("event_name"),
            date=event_date,
            location=row.get("event_location"),
            country=_parse_country(row.get("event_location")),
        ))


def _parse_country(location) -> str | None:
    if pd.isna(location):
        return None
    parts = str(location).split(",")
    return parts[-1].strip() if parts else None


def _load_fights(csv_dir: Path, session: Session) -> None:
    results_path = csv_dir / _CSV_FILES["results"]
    if not results_path.exists():
        log.warning("Fight results CSV not found: %s", results_path)
        return

    df = pd.read_csv(results_path)
    log.info("Loading %d fight results", len(df))

    # Build ufcstats_id → canonical_id lookup
    id_map = {
        f.ufcstats_id: f.canonical_fighter_id
        for f in session.query(Fighter)
    }

    existing = {f.fight_id for f in session.query(Fight.fight_id)}

    for _, row in df.iterrows():
        fight_id = str(row.get("fight_url", "")).split("/")[-1]
        if not fight_id or fight_id in existing:
            continue

        event_id = str(row.get("event_url", "")).split("/")[-1]

        red_ufcs = str(row.get("fighter_1_url", "")).split("/")[-1]
        blue_ufcs = str(row.get("fighter_2_url", "")).split("/")[-1]
        red_id = id_map.get(red_ufcs)
        blue_id = id_map.get(blue_ufcs)
        if not red_id or not blue_id:
            log.debug("Skipping fight %s — unmapped fighters", fight_id)
            continue

        winner_ufcs = str(row.get("winner_url", "")).split("/")[-1]
        winner_id = id_map.get(winner_ufcs)

        method_raw = str(row.get("method", "")).strip().upper()
        method = _normalise_method(method_raw)

        # Skip cancelled/null-outcome bouts
        if not method or method == "CANCELLED":
            continue

        try:
            fight_date = pd.to_datetime(row["event_date"]).date()
        except Exception:
            continue

        time_str = row.get("time_format", "")
        is_five_round = "5" in str(time_str)[:3]

        time_sec = None
        t = row.get("time")
        if not pd.isna(t):
            time_sec = _parse_ctrl(t)

        session.add(Fight(
            fight_id=fight_id,
            event_id=event_id,
            date=fight_date,
            red_fighter_id=red_id,
            blue_fighter_id=blue_id,
            weight_class=row.get("weight_class"),
            is_title_bout=bool(row.get("title_bout", False)),
            is_five_round=is_five_round,
            winner_fighter_id=winner_id,
            method=method,
            round_ended=_safe_int(row.get("round")),
            time_ended_sec=time_sec,
            referee=row.get("referee"),
        ))


def _normalise_method(raw: str) -> str | None:
    if not raw or raw == "NAN":
        return None
    if raw.startswith("KO") or raw.startswith("TKO"):
        return raw[:3]
    if raw.startswith("SUB"):
        return "SUB"
    if raw.startswith("DEC"):
        kind = raw.split()[-1] if " " in raw else ""
        return f"Decision ({kind})" if kind else "Decision"
    if raw in ("DQ", "NC", "CANCELLED"):
        return raw
    return raw


def _load_fight_stats(csv_dir: Path, session: Session) -> None:
    path = csv_dir / _CSV_FILES["fight_stats"]
    if not path.exists():
        log.warning("Fight stats CSV not found: %s", path)
        return

    df = pd.read_csv(path)
    log.info("Loading %d fight-stat rows", len(df))

    id_map = {
        f.ufcstats_id: f.canonical_fighter_id
        for f in session.query(Fighter)
    }

    existing_fights = {f.fight_id for f in session.query(Fight.fight_id)}

    for _, row in df.iterrows():
        fight_id = str(row.get("fight_url", "")).split("/")[-1]
        if fight_id not in existing_fights:
            continue

        fighter_ufcs = str(row.get("fighter_url", "")).split("/")[-1]
        fighter_id = id_map.get(fighter_ufcs)
        if not fighter_id:
            continue

        round_num = _safe_int(row.get("round")) or 0

        # Check for duplicate
        exists = session.query(FightStatsRound).filter_by(
            fight_id=fight_id, fighter_id=fighter_id, round=round_num
        ).first()
        if exists:
            continue

        kd = _safe_int(row.get("knockdowns")) or 0

        sig_l, sig_a = _parse_landed_attempted(row.get("sig_str"))
        total_l, total_a = _parse_landed_attempted(row.get("total_str"))
        head_l, head_a = _parse_landed_attempted(row.get("head"))
        body_l, body_a = _parse_landed_attempted(row.get("body"))
        leg_l, leg_a = _parse_landed_attempted(row.get("leg"))
        dist_l, dist_a = _parse_landed_attempted(row.get("distance"))
        clinch_l, clinch_a = _parse_landed_attempted(row.get("clinch"))
        ground_l, ground_a = _parse_landed_attempted(row.get("ground"))
        td_l, td_a = _parse_landed_attempted(row.get("td"))

        session.add(FightStatsRound(
            fight_id=fight_id,
            fighter_id=fighter_id,
            round=round_num,
            knockdowns=kd,
            sig_strikes_landed=sig_l or 0,
            sig_strikes_attempted=sig_a or 0,
            total_strikes_landed=total_l or 0,
            total_strikes_attempted=total_a or 0,
            head_landed=head_l or 0,
            head_attempted=head_a or 0,
            body_landed=body_l or 0,
            body_attempted=body_a or 0,
            leg_landed=leg_l or 0,
            leg_attempted=leg_a or 0,
            distance_landed=dist_l or 0,
            distance_attempted=dist_a or 0,
            clinch_landed=clinch_l or 0,
            clinch_attempted=clinch_a or 0,
            ground_landed=ground_l or 0,
            ground_attempted=ground_a or 0,
            takedowns_landed=td_l or 0,
            takedowns_attempted=td_a or 0,
            submission_attempts=_safe_int(row.get("sub_att")) or 0,
            reversals=_safe_int(row.get("rev")) or 0,
            control_time_sec=_parse_ctrl(row.get("ctrl")) or 0,
        ))


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    csv_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/greco1899"
    load_all(csv_dir)
