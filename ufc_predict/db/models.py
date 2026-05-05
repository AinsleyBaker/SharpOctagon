"""SQLAlchemy ORM models — one class per canonical table."""

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Fighter(Base):
    __tablename__ = "fighters"

    canonical_fighter_id = Column(String, primary_key=True)
    ufcstats_id = Column(String, unique=True, index=True)
    sherdog_id = Column(String, unique=True, index=True)
    tapology_id = Column(String, unique=True)
    wikidata_qid = Column(String, unique=True)
    full_name = Column(String, nullable=False)
    nickname = Column(String)
    name_variants = Column(JSON)  # list[str]
    dob = Column(Date)
    nationality = Column(String)
    stance = Column(String)  # current; history in snapshots
    height_cm = Column(Float)
    reach_cm = Column(Float)
    primary_weight_class = Column(String)

    snapshots = relationship("FighterSnapshot", back_populates="fighter")


class FighterSnapshot(Base):
    """Fighter attributes at a specific point in time (handles weight-class / stance changes)."""

    __tablename__ = "fighter_snapshots"
    __table_args__ = (UniqueConstraint("canonical_fighter_id", "as_of_date"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    canonical_fighter_id = Column(
        String, ForeignKey("fighters.canonical_fighter_id"), nullable=False
    )
    as_of_date = Column(Date, nullable=False)
    stance = Column(String)
    weight_class = Column(String)
    camp = Column(String)
    is_active = Column(Boolean)

    fighter = relationship("Fighter", back_populates="snapshots")


class Event(Base):
    __tablename__ = "events"

    event_id = Column(String, primary_key=True)  # UFCStats 16-hex
    ufc_event_number = Column(String)
    date = Column(Date, nullable=False, index=True)
    location = Column(String)
    country = Column(String)
    altitude_m = Column(Float)

    fights = relationship("Fight", back_populates="event")


class Fight(Base):
    __tablename__ = "fights"

    fight_id = Column(String, primary_key=True)  # UFCStats 16-hex
    event_id = Column(String, ForeignKey("events.event_id"), nullable=False)
    date = Column(Date, nullable=False, index=True)  # denormalised for as-of queries
    red_fighter_id = Column(String, ForeignKey("fighters.canonical_fighter_id"), nullable=False)
    blue_fighter_id = Column(String, ForeignKey("fighters.canonical_fighter_id"), nullable=False)
    weight_class = Column(String)
    is_title_bout = Column(Boolean, default=False)
    is_five_round = Column(Boolean, default=False)
    # NULL = draw/NC
    winner_fighter_id = Column(String, ForeignKey("fighters.canonical_fighter_id"))
    method = Column(String)  # KO / TKO / SUB / Decision / DQ / NC
    method_detail = Column(String)
    round_ended = Column(Integer)
    time_ended_sec = Column(Integer)
    referee = Column(String)
    bonus_awards = Column(JSON)  # {"POTN": [...], "FOTN": [...]}
    red_is_short_notice = Column(Boolean, default=False)
    blue_is_short_notice = Column(Boolean, default=False)
    red_missed_weight = Column(Boolean, default=False)
    blue_missed_weight = Column(Boolean, default=False)
    # Closing odds stored for EVALUATION ONLY — never used as model features
    closing_odds_red = Column(Float)
    closing_odds_blue = Column(Float)

    event = relationship("Event", back_populates="fights")
    round_stats = relationship("FightStatsRound", back_populates="fight")


class FightStatsRound(Base):
    """Per-fighter, per-round striking and grappling stats. round=0 means fight total."""

    __tablename__ = "fight_stats_round"
    __table_args__ = (UniqueConstraint("fight_id", "fighter_id", "round"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    fight_id = Column(String, ForeignKey("fights.fight_id"), nullable=False)
    fighter_id = Column(String, ForeignKey("fighters.canonical_fighter_id"), nullable=False)
    round = Column(Integer, nullable=False)  # 0=total, 1..N=per-round

    knockdowns = Column(Integer, default=0)
    sig_strikes_landed = Column(Integer, default=0)
    sig_strikes_attempted = Column(Integer, default=0)
    total_strikes_landed = Column(Integer, default=0)
    total_strikes_attempted = Column(Integer, default=0)

    head_landed = Column(Integer, default=0)
    head_attempted = Column(Integer, default=0)
    body_landed = Column(Integer, default=0)
    body_attempted = Column(Integer, default=0)
    leg_landed = Column(Integer, default=0)
    leg_attempted = Column(Integer, default=0)

    distance_landed = Column(Integer, default=0)
    distance_attempted = Column(Integer, default=0)
    clinch_landed = Column(Integer, default=0)
    clinch_attempted = Column(Integer, default=0)
    ground_landed = Column(Integer, default=0)
    ground_attempted = Column(Integer, default=0)

    takedowns_landed = Column(Integer, default=0)
    takedowns_attempted = Column(Integer, default=0)
    submission_attempts = Column(Integer, default=0)
    reversals = Column(Integer, default=0)
    control_time_sec = Column(Integer, default=0)  # stored as seconds, NOT mm:ss

    fight = relationship("Fight", back_populates="round_stats")


class FightPropOdds(Base):
    """Closing odds for prop markets (totals, distance, method, round-of-finish).

    One row per (fight, prop_type, side). Populated by the BFO prop scraper
    and used by the prop ROI backtest to identify which prop markets our
    model can profitably bet against.

    `prop_type` examples: 'distance', 'total_rounds_2.5', 'f1_method_KO_TKO',
    'f1_wins_round_3', 'starts_round_2'. `side` is 'yes' / 'no' / 'over' /
    'under'. Together they uniquely identify a prop bet.
    """

    __tablename__ = "fight_prop_odds"
    __table_args__ = (UniqueConstraint("fight_id", "prop_type", "side"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    fight_id = Column(String, ForeignKey("fights.fight_id"), nullable=False, index=True)
    prop_type = Column(String, nullable=False)
    side = Column(String, nullable=False)
    american_odds = Column(Float, nullable=False)
    n_books = Column(Integer, default=1)
    raw_label = Column(String)
    scraped_at = Column(DateTime)


class UpcomingBout(Base):
    """Scheduled but not-yet-fought bouts. Written by the live-data pipeline."""

    __tablename__ = "upcoming_bouts"

    upcoming_bout_id = Column(String, primary_key=True)  # hash(event_id + fighter_ids)
    event_date = Column(Date, nullable=False, index=True)
    event_name = Column(String)
    red_fighter_id = Column(String, ForeignKey("fighters.canonical_fighter_id"))
    blue_fighter_id = Column(String, ForeignKey("fighters.canonical_fighter_id"))
    red_name_raw = Column(String)   # pre-canonicalisation spelling
    blue_name_raw = Column(String)
    weight_class = Column(String)
    is_title_bout = Column(Boolean, default=False)
    is_five_round = Column(Boolean, default=False)
    source = Column(String)  # 'ufc.com' | 'espn' | 'tapology'
    first_seen_at = Column(DateTime)
    last_updated_at = Column(DateTime)
    is_confirmed = Column(Boolean, default=True)
    is_cancelled = Column(Boolean, default=False)
