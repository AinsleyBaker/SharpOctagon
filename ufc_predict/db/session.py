"""Database engine and session factory."""

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

_DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "ufc.db"

def get_engine(db_url: str | None = None):
    url = db_url or os.environ.get("UFC_DB_URL") or f"sqlite:///{_DEFAULT_DB}"
    _DEFAULT_DB.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, echo=False)


def get_session_factory(db_url: str | None = None):
    return sessionmaker(bind=get_engine(db_url), autoflush=False, autocommit=False)


def init_db(db_url: str | None = None):
    """Create all tables (dev convenience — production uses Alembic migrations)."""
    from ufc_predict.db.models import Base
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)
    return engine
