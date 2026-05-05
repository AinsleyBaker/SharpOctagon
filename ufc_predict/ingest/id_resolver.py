"""
Stage 4b — Fuzzy fighter ID resolution.

After the Wikidata bootstrap, many fighters still lack a Sherdog ID.
This module fuzzy-matches UFCStats fighters to Sherdog fighter pages using:
  - rapidfuzz token_sort_ratio >= SCORE_THRESHOLD (default 90)
  - DOB gate: |dob_diff| <= 1 day  ← prevents "two Michael Johnsons" problem

Output: populates fighters.sherdog_id for matched rows.
Unmatched fighters are written to data/review/unmatched_fighters.csv for manual review.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from rapidfuzz import fuzz, process
from sqlalchemy.orm import Session

from ufc_predict.db.models import Fighter

log = logging.getLogger(__name__)

SCORE_THRESHOLD = 90
DOB_TOLERANCE_DAYS = 1


@dataclass
class SherdogStub:
    """Minimal Sherdog fighter record needed for matching."""
    sherdog_id: str     # e.g. "Conor-McGregor-29688"
    name: str
    dob: date | None


def match_fighters_to_sherdog(
    sherdog_stubs: list[SherdogStub],
    session: Session,
    review_path: Path | None = None,
) -> dict[str, str]:
    """
    Match UFC fighters (loaded into DB) to Sherdog stubs.

    Returns a dict of {canonical_fighter_id: sherdog_id} for confident matches.
    Writes unmatched fighters to review_path CSV.
    """
    # Only consider fighters without a Sherdog ID already
    fighters = session.query(Fighter).filter(Fighter.sherdog_id.is_(None)).all()
    log.info("Attempting to match %d fighters to Sherdog", len(fighters))

    # Build Sherdog lookup by name → list[SherdogStub] for DOB gate
    sherdog_by_name: dict[str, list[SherdogStub]] = {}
    for stub in sherdog_stubs:
        sherdog_by_name.setdefault(stub.name.lower(), []).append(stub)

    # rapidfuzz needs a list of names to search against
    sherdog_names = [s.name for s in sherdog_stubs]

    matched: dict[str, str] = {}
    unmatched: list[Fighter] = []

    for fighter in fighters:
        if not fighter.full_name:
            unmatched.append(fighter)
            continue

        # rapidfuzz top-1 match
        result = process.extractOne(
            fighter.full_name,
            sherdog_names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=SCORE_THRESHOLD,
        )

        if result is None:
            unmatched.append(fighter)
            continue

        best_name, score, idx = result
        candidates = sherdog_by_name.get(best_name.lower(), [sherdog_stubs[idx]])

        # DOB gate: filter candidates by date of birth proximity
        if fighter.dob:
            dob_gated = [
                c for c in candidates
                if c.dob and abs((c.dob - fighter.dob).days) <= DOB_TOLERANCE_DAYS
            ]
            if not dob_gated:
                log.debug(
                    "DOB mismatch: %s (UFC dob=%s) vs Sherdog candidates %s",
                    fighter.full_name, fighter.dob,
                    [(c.name, c.dob) for c in candidates],
                )
                unmatched.append(fighter)
                continue
            if len(dob_gated) > 1:
                log.warning("Multiple DOB-gated matches for %s — skipping", fighter.full_name)
                unmatched.append(fighter)
                continue
            stub = dob_gated[0]
        else:
            # No DOB in UFC data — only accept a single exact name match
            if len(candidates) != 1:
                unmatched.append(fighter)
                continue
            stub = candidates[0]

        matched[fighter.canonical_fighter_id] = stub.sherdog_id
        fighter.sherdog_id = stub.sherdog_id

    log.info("Matched %d / %d fighters (%.0f%%)",
             len(matched), len(fighters),
             100 * len(matched) / max(len(fighters), 1))
    log.info("%d fighters require manual review", len(unmatched))

    if unmatched and review_path:
        _write_review_csv(unmatched, review_path)

    return matched


def apply_matches(matched: dict[str, str], session: Session) -> None:
    """Persist sherdog_id updates to DB."""
    for canonical_id, sherdog_id in matched.items():
        f = session.get(Fighter, canonical_id)
        if f:
            f.sherdog_id = sherdog_id
    session.commit()


def load_manual_corrections(csv_path: Path, session: Session) -> int:
    """
    Apply manually-corrected ID mappings from a CSV with columns:
    canonical_fighter_id, sherdog_id, wikidata_qid (optional), tapology_id (optional)
    """
    if not csv_path.exists():
        return 0

    updated = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fighter = session.get(Fighter, row["canonical_fighter_id"])
            if not fighter:
                log.warning(
                    "Manual correction: unknown canonical_id %s",
                    row["canonical_fighter_id"],
                )
                continue
            for field in ("sherdog_id", "wikidata_qid", "tapology_id"):
                val = row.get(field, "").strip()
                if val:
                    setattr(fighter, field, val)
                    updated += 1

    session.commit()
    log.info("Applied %d manual ID corrections", updated)
    return updated


def _write_review_csv(fighters: list[Fighter], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["canonical_fighter_id", "full_name", "dob", "ufcstats_id",
                          "sherdog_id", "wikidata_qid", "notes"])
        for fighter in fighters:
            writer.writerow([
                fighter.canonical_fighter_id,
                fighter.full_name,
                fighter.dob or "",
                fighter.ufcstats_id or "",
                fighter.sherdog_id or "",
                fighter.wikidata_qid or "",
                "",
            ])
    log.info("Wrote %d unmatched fighters to %s", len(fighters), path)
