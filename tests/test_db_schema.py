"""Verify all ORM tables create without errors against an in-memory SQLite DB."""

import pytest
from sqlalchemy import create_engine, inspect

from ufc_predict.db.models import Base


@pytest.fixture
def engine():
    e = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(e)
    return e


def test_all_tables_created(engine):
    expected = {
        "fighters",
        "fighter_snapshots",
        "events",
        "fights",
        "fight_stats_round",
        "upcoming_bouts",
    }
    actual = set(inspect(engine).get_table_names())
    assert expected == actual


def test_fighter_insert(engine):
    from datetime import date

    from sqlalchemy.orm import Session

    from ufc_predict.db.models import Fighter

    with Session(engine) as s:
        s.add(Fighter(
            canonical_fighter_id="test_001",
            ufcstats_id="abc123",
            full_name="Test Fighter",
            dob=date(1990, 1, 1),
        ))
        s.commit()
        f = s.get(Fighter, "test_001")
        assert f.full_name == "Test Fighter"
