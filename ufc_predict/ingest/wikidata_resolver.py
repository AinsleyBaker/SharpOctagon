"""
Stage 4a — Wikidata SPARQL bootstrap.

Queries Wikidata for all MMA fighters that have a UFCStats ID, Sherdog ID,
or Tapology ID, and writes the cross-reference mapping into the fighters table.

Run once to bootstrap, then incrementally as new fighters debut.
"""

from __future__ import annotations

import logging
import re as _re
import time
from dataclasses import dataclass
from datetime import date

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
  ?height_m
  ?reach_m
  ?sherdog_id
  ?tapology_id
WHERE {
  ?fighter wdt:P31 wd:Q5 .            # instance of human
  ?fighter wdt:P641 wd:Q114466 .      # sport = mixed martial arts
  OPTIONAL { ?fighter wdt:P2818 ?sherdog_id . }   # Sherdog fighter ID
  OPTIONAL { ?fighter wdt:P9728 ?tapology_id . }  # Tapology fighter ID
  OPTIONAL { ?fighter wdt:P569  ?dob . }          # date of birth
  OPTIONAL { ?fighter wdt:P2048 ?height_m . }     # height (metres)
  OPTIONAL { ?fighter wdt:P2240 ?reach_m . }      # reach (metres)
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
    height_cm: float | None
    reach_cm: float | None
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

    def _to_cm(v: str | None) -> float | None:
        # Wikidata height/reach values come back as strings in metres (e.g. "1.83").
        if not v:
            return None
        try:
            return round(float(v) * 100.0, 1)
        except ValueError:
            return None

    fighters: list[WikidataFighter] = []
    for row in results["results"]["bindings"]:
        qid = row["fighter"]["value"].split("/")[-1]
        name = row.get("fighterLabel", {}).get("value", "")
        dob_raw = row.get("dob", {}).get("value")
        dob = dob_raw[:10] if dob_raw else None  # ISO date prefix
        height_cm = _to_cm(row.get("height_m", {}).get("value"))
        reach_cm = _to_cm(row.get("reach_m", {}).get("value"))
        sherdog = row.get("sherdog_id", {}).get("value")
        tapology = row.get("tapology_id", {}).get("value")
        fighters.append(WikidataFighter(
            qid=qid, name=name, dob=dob, height_cm=height_cm, reach_cm=reach_cm,
            sherdog_id=sherdog, tapology_id=tapology,
        ))

    log.info("Fetched %d fighters from Wikidata", len(fighters))
    return fighters


_NAME_NORM_RE = _re.compile(r"[^a-z0-9]+")


def _norm_name(s: str) -> str:
    return _NAME_NORM_RE.sub(" ", (s or "").lower()).strip()


def _name_key(name: str) -> tuple[str, str]:
    """(last_token, first_initial) — handles middle-name variants."""
    toks = _norm_name(name).split()
    if not toks:
        return ("", "")
    return (toks[-1], toks[0][:1] if toks[0] else "")


def apply_wikidata_mappings(fighters: list[WikidataFighter], session: Session) -> int:
    """
    Attempt to match each Wikidata record to an existing Fighter row by:
      1. Sherdog ID exact match (if already stored)
      2. Exact normalized full-name match
      3. Last-name + first-initial match, disambiguated by DOB when available

    Returns number of rows updated.
    """
    updated = 0

    all_fighters = session.query(Fighter).all()
    by_sherdog: dict[str, Fighter] = {f.sherdog_id: f for f in all_fighters if f.sherdog_id}

    by_full: dict[str, list[Fighter]] = {}
    by_lf: dict[tuple[str, str], list[Fighter]] = {}
    for f in all_fighters:
        by_full.setdefault(_norm_name(f.full_name), []).append(f)
        by_lf.setdefault(_name_key(f.full_name), []).append(f)

    needs_review: list[tuple[str, str]] = []

    for wd in fighters:
        candidate: Fighter | None = None

        # Priority 1: exact Sherdog ID match
        if wd.sherdog_id and wd.sherdog_id in by_sherdog:
            candidate = by_sherdog[wd.sherdog_id]

        # Priority 2: exact normalized full-name match
        if candidate is None:
            matches = by_full.get(_norm_name(wd.name), [])
            if len(matches) == 1:
                candidate = matches[0]

        # Priority 3: last-name + first-initial, disambiguated by DOB
        if candidate is None:
            matches = by_lf.get(_name_key(wd.name), [])
            if len(matches) == 1:
                candidate = matches[0]
            elif len(matches) > 1 and wd.dob:
                try:
                    wd_dob = date.fromisoformat(wd.dob)
                except ValueError:
                    wd_dob = None
                dob_matches = [f for f in matches if f.dob == wd_dob] if wd_dob else []
                if len(dob_matches) == 1:
                    candidate = dob_matches[0]
                else:
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
        if wd.dob and not candidate.dob:
            try:
                candidate.dob = date.fromisoformat(wd.dob)
                changed = True
            except ValueError:
                pass
        if wd.height_cm and not candidate.height_cm:
            candidate.height_cm = wd.height_cm
            changed = True
        if wd.reach_cm and not candidate.reach_cm:
            candidate.reach_cm = wd.reach_cm
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
