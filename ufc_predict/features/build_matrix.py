"""CLI entry point: build full feature matrix from DB and save as parquet."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ufc_predict.db.session import get_session_factory
from ufc_predict.features.aso_features import build_fight_feature_rows, symmetrize_rows
from ufc_predict.features.ratings import attach_ratings, save_latest_ratings

log = logging.getLogger(__name__)
OUTPUT_PATH = Path("data/feature_matrix.parquet")


def run(db_url: str | None = None, since_year: int = 2001) -> None:
    factory = get_session_factory(db_url)
    with factory() as session:
        log.info("Building feature rows (since %d)…", since_year)
        df = build_fight_feature_rows(session, since_year=since_year)

    if df.empty:
        raise RuntimeError(
            "Feature matrix is empty — no fights were loaded from the DB. "
            "Check that greco_loader ran successfully and the DB has fight rows."
        )

    # Ratings must be computed on the N-row base df (chronological order matters).
    # Symmetrize AFTER rating attachment so rating columns are swapped correctly.
    # Capture the final post-fight states so we can persist the latest rating
    # per (fighter, weight_class) for the live predict step to consume.
    df, elo_states, glicko_states = attach_ratings(df, return_states=True)
    save_latest_ratings(elo_states, glicko_states)
    df = symmetrize_rows(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    log.info("Feature matrix saved: %s  shape=%s", OUTPUT_PATH, df.shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
