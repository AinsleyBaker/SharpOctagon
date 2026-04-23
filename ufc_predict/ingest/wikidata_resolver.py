"""
Stage 4a — Wikidata SPARQL bootstrap.

Queries Wikidata for all MMA fighters that have a UFCStats ID, Sherdog ID,
or Tapology ID, and writes the cross-reference mapping into the fighters table.

Run once to bootstrap, then incrementally as new fighters debut.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from SPARQLWrapper import JSON, SPARQLWrapper
from sqlalchemy.orm import Session

from ufc_predict.db.models import Fighter

log = logging.getLogger(__name__)

_WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

_SPARQL_QUERY = """
SELECT DISTINCT
  ?fighter
  ?fighterLabel
  ?dob
  ?sherdog_id
  ?tapology_id
WHERE {
  ?fighter wdt:P31 wd:Q5 .            # instance of human
  ?fighter wdt:P641 wd:Q114466 .      # sport = mixed martial arts
  OPTIONAL { ?fighter wdt:P2818 ?sherdog_id . }   # Sherdog fighter ID
  OPTIONAL { ?fighter wdt:P9728 ?tapology_id . }  # Tapology fighter ID
  OPTIONAL { ?fighter wdt:P569  ?dob . }          # date of birth
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en" .
  }
}
"""


@dataclass
class WikidataFighter:
    qid: str
    name: str
    dob: str | None
    sherdog_id: str | None
    tapology_id: str | None


def fetch_wikidata_fighters(max_retries: int = 3) -> list[WikidataFighter]:
    sparql = SPARQLWrapper(_WIKIDATA_ENDPOINT)
    sparql.addCustomHttpHeader("User-Agent", "UFCPredictBot/0.1 (research; contact via github)")
    sparql.setQuery(_SPARQL_QUERY)
    sparql.setReturnFormat(JSON)

    for attempt in range(1, max_retries + 1):
        try:
            log.info("Querying Wikidata SPARQL (attempt %d)…", attempt)
            results = sparql.query().convert()
            break
        except Exception as exc:
            log.warning("SPARQL attempt %d failed: %s", attempt, exc)
            if attempt == max_retries:
                raise
            time.sleep(5 * attempt)

    fighters: list[WikidataFighter] = []
    for row in results["results"]["bindings"]:
        qid = row["fighter"]["value"].split("/")[-1]
        name = row.get("fighterLabel", {}).get("value", "")
        dob_raw = row.get("dob", {}).get("value")
        dob = dob_raw[:10] if dob_raw else None  # ISO date prefix
        sherdog = row.get("sherdog_id", {}).get("value")
        tapology = row.get("tapology_id", {}).get("value")
        fighters.append(WikidataFighter(qid=qid, name=name, dob=dob, sherdog_id=sherdog, tapology_id=tapology))

    log.info("Fetched %d fighters from Wikidata", len(fighters))
    return fighters


def apply_wikidata_mappings(fighters: list[WikidataFighter], session: Session) -> int:
    """
    Attempt to match each Wikidata record to an existing Fighter row by:
      1. Sherdog ID exact match (if already stored)
      2. Name similarity (fallback — low confidence, logged for manual review)

    Returns number of rows updated.
    """
    updated = 0

    # Index existing fighters by name (lowercased) for cheap lookup
    all_fighters = session.query(Fighter).all()
    by_sherdog: dict[str, Fighter] = {f.sherdog_id: f for f in all_fighters if f.sherdog_id}
    by_name: dict[str, list[Fighter]] = {}
    for f in all_fighters:
        key = f.full_name.lower().strip()
        by_name.setdefault(key, []).append(f)

    needs_review: list[tuple[str, str]] = []

    for wd in fighters:
        candidate: Fighter | None = None

        # Priority 1: exact Sherdog ID match
        if wd.sherdog_id and wd.sherdog_id in by_sherdog:
            candidate = by_sherdog[wd.sherdog_id]

        # Priority 2: exact name match (case-insensitive)
        if candidate is None:
            matches = by_name.get(wd.name.lower().strip(), [])
            if len(matches) == 1:
                candidate = matches[0]
            elif len(matches) > 1:
                needs_review.append((wd.qid, wd.name))
                continue

        if candidate is None:
            continue

        changed = False
        if wd.qid and not candidate.wikidata_qid:
            candidate.wikidata_qid = wd.qid
            changed = True
        if wd.sherdog_id and not candidate.sherdog_id:
            candidate.sherdog_id = wd.sherdog_id
            changed = True
        if wd.tapology_id and not candidate.tapology_id:
            candidate.tapology_id = wd.tapology_id
            changed = True
        if changed:
            updated += 1

    if needs_review:
        log.warning(
            "%d Wikidata fighters had ambiguous name matches — manual review needed:\n%s",
            len(needs_review),
            "\n".join(f"  {qid}: {name}" for qid, name in needs_review[:20]),
        )

    return updated


def run(db_url: str | None = None) -> None:
    from ufc_predict.db.session import get_session_factory
    factory = get_session_factory(db_url)

    wd_fighters = fetch_wikidata_fighters()
    with factory() as session:
        n = apply_wikidata_mappings(wd_fighters, session)
        session.commit()
    log.info("Wikidata bootstrap complete — %d fighter rows updated.", n)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
