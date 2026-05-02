"""
Stage 7 — Elo and Glicko-2 rating systems.

Ratings are computed per weight class, chronologically.
Both systems produce:
  - A rating value (Elo: ~1500 base; Glicko: ~1500 base)
  - An uncertainty / RD term (Elo: implicit; Glicko: explicit RD)

The RD from Glicko-2 is itself a feature — a high RD means we're uncertain
about the fighter's current form (inactive, few fights).

CRITICAL: ratings are computed as-of-fight (pre-fight), never post-fight.
We update ratings AFTER storing the pre-fight value used as a feature.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import DefaultDict
from collections import defaultdict

import pandas as pd

log = logging.getLogger(__name__)

LATEST_RATINGS_PATH = Path("models/fighter_ratings.json")

# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------

ELO_BASE = 1500.0
ELO_K = 32.0
ELO_K_DEBUT = 48.0  # larger K for first 5 fights


@dataclass
class EloState:
    rating: float = ELO_BASE
    n_fights: int = 0


def _elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def compute_elo(
    fights_df: pd.DataFrame,
    return_states: bool = False,
):
    """
    Compute Elo ratings incrementally in chronological order per weight class.

    fights_df must have columns:
      fight_id, date, weight_class, fighter_a_id, fighter_b_id, label (1=A won)

    Returns the same DataFrame with added columns:
      elo_a, elo_b, elo_diff (a - b)  [all PRE-FIGHT values]

    When return_states=True, also returns the final post-fight states dict
    keyed by (fighter_id, weight_class) — used by build_matrix to persist
    latest ratings for inference.
    """
    df = fights_df.sort_values("date").copy()
    states: DefaultDict[tuple, EloState] = defaultdict(EloState)

    elo_a_list, elo_b_list = [], []

    for _, row in df.iterrows():
        wc = row["weight_class"] or "Unknown"
        key_a = (row["fighter_a_id"], wc)
        key_b = (row["fighter_b_id"], wc)

        sa = states[key_a]
        sb = states[key_b]

        # Store pre-fight ratings as features
        elo_a_list.append(sa.rating)
        elo_b_list.append(sb.rating)

        # Update ratings post-fight
        ea = _elo_expected(sa.rating, sb.rating)
        label = float(row["label"])

        k_a = ELO_K_DEBUT if sa.n_fights < 5 else ELO_K
        k_b = ELO_K_DEBUT if sb.n_fights < 5 else ELO_K

        sa.rating += k_a * (label - ea)
        sb.rating += k_b * ((1 - label) - (1 - ea))
        sa.n_fights += 1
        sb.n_fights += 1

    df["elo_a"] = elo_a_list
    df["elo_b"] = elo_b_list
    df["diff_elo"] = df["elo_a"] - df["elo_b"]
    if return_states:
        return df, states
    return df


# ---------------------------------------------------------------------------
# Glicko-2
# ---------------------------------------------------------------------------

GLICKO_MU_BASE = 0.0          # internal scale (maps to 1500 for display)
GLICKO_PHI_BASE = 350 / 173.7177  # ≈ 2.015 (RD in Glicko-2 units)
GLICKO_SIGMA_BASE = 0.06
GLICKO_TAU = 0.5               # system volatility constraint
GLICKO_Q = math.log(10) / 400
GLICKO_DISPLAY_SCALE = 173.7177
GLICKO_DISPLAY_BASE = 1500


@dataclass
class GlickoState:
    mu: float = GLICKO_MU_BASE
    phi: float = GLICKO_PHI_BASE
    sigma: float = GLICKO_SIGMA_BASE
    n_fights: int = 0
    last_fight_date: date | None = None

    @property
    def display_rating(self) -> float:
        return self.mu * GLICKO_DISPLAY_SCALE + GLICKO_DISPLAY_BASE

    @property
    def display_rd(self) -> float:
        return self.phi * GLICKO_DISPLAY_SCALE


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1 + 3 * phi**2 / math.pi**2)


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    arg = max(-500.0, min(500.0, -_g(phi_j) * (mu - mu_j)))
    return 1.0 / (1 + math.exp(arg))


def _update_glicko2(
    player: GlickoState,
    opponents: list[tuple[GlickoState, float]],
    days_inactive: int = 0,
) -> GlickoState:
    """Single Glicko-2 rating period update for one fighter."""
    # Inflate RD for inactivity (c = 34.6 Glicko-2 units / year ≈ standard)
    if days_inactive > 0:
        c = 34.6 / GLICKO_DISPLAY_SCALE
        phi_star = math.sqrt(player.phi**2 + c**2 * (days_inactive / 365.0))
    else:
        phi_star = player.phi

    if not opponents:
        # No fights this period — only inflate RD
        return GlickoState(mu=player.mu, phi=phi_star, sigma=player.sigma,
                           n_fights=player.n_fights, last_fight_date=player.last_fight_date)

    # Step 3: compute v (cap to prevent overflow in downstream squaring)
    v_inv = sum(
        _g(opp.phi)**2 * _E(player.mu, opp.mu, opp.phi) * (1 - _E(player.mu, opp.mu, opp.phi))
        for opp, _ in opponents
    )
    v = min(1.0 / v_inv if v_inv > 1e-10 else 1e10, 1e10)

    # Step 4: delta
    delta = v * sum(
        _g(opp.phi) * (score - _E(player.mu, opp.mu, opp.phi))
        for opp, score in opponents
    )

    # Step 5: new sigma (Illinois algorithm)
    a = math.log(player.sigma**2)
    eps = 1e-6

    def f(x: float) -> float:
        ex = math.exp(min(x, 350.0))
        d2 = delta**2
        phi2 = phi_star**2
        return (
            ex * (d2 - phi2 - v - ex) / (2 * (phi2 + v + ex)**2)
            - (x - a) / GLICKO_TAU**2
        )

    A = a
    if delta**2 > phi_star**2 + v:
        B = math.log(delta**2 - phi_star**2 - v)
    else:
        k = 1
        while f(a - k * GLICKO_TAU) < 0:
            k += 1
        B = a - k * GLICKO_TAU

    fa, fb = f(A), f(B)
    for _ in range(100):
        C = A + (A - B) * fa / (fb - fa)
        fc = f(C)
        if abs(C - B) < eps:
            break
        if fc * fb < 0:
            A, fa = B, fb
        else:
            fa /= 2
        B, fb = C, fc

    new_sigma = math.exp(A / 2)

    # Step 6–7: new phi, mu
    # Step 6 uses player.phi (original pre-period RD), not phi_star (inactivity-inflated)
    phi_star2 = math.sqrt(player.phi**2 + new_sigma**2)
    new_phi = 1.0 / math.sqrt(1.0 / phi_star2**2 + 1.0 / v)
    new_mu = player.mu + new_phi**2 * sum(
        _g(opp.phi) * (score - _E(player.mu, opp.mu, opp.phi))
        for opp, score in opponents
    )

    return GlickoState(mu=new_mu, phi=new_phi, sigma=new_sigma,
                       n_fights=player.n_fights + len(opponents))


def compute_glicko2(fights_df: pd.DataFrame, return_states: bool = False):
    """
    Compute Glicko-2 ratings chronologically per weight class.
    Returns df with added columns:
      glicko_a, glicko_b, glicko_rd_a, glicko_rd_b, diff_glicko
    All values are PRE-FIGHT.

    When return_states=True, also returns the final post-fight states dict.
    """
    df = fights_df.sort_values("date").copy()
    states: DefaultDict[tuple, GlickoState] = defaultdict(GlickoState)

    g_a, g_b, rd_a, rd_b = [], [], [], []

    for _, row in df.iterrows():
        wc = row["weight_class"] or "Unknown"
        key_a = (row["fighter_a_id"], wc)
        key_b = (row["fighter_b_id"], wc)

        sa = states[key_a]
        sb = states[key_b]

        # Store pre-fight ratings
        g_a.append(sa.display_rating)
        g_b.append(sb.display_rating)
        rd_a.append(sa.display_rd)
        rd_b.append(sb.display_rd)

        label = float(row["label"])
        fight_dt = row["date"]
        if not isinstance(fight_dt, date):
            try:
                fight_dt = date.fromisoformat(str(fight_dt)[:10])
            except ValueError:
                fight_dt = None

        # Update both fighters (each treated as a single rating period)
        new_sa = _update_glicko2(sa, [(sb, label)])
        new_sb = _update_glicko2(sb, [(sa, 1 - label)])
        new_sa.last_fight_date = fight_dt
        new_sb.last_fight_date = fight_dt

        states[key_a] = new_sa
        states[key_b] = new_sb

    df["glicko_a"] = g_a
    df["glicko_b"] = g_b
    df["glicko_rd_a"] = rd_a
    df["glicko_rd_b"] = rd_b
    df["diff_glicko"] = df["glicko_a"] - df["glicko_b"]
    if return_states:
        return df, states
    return df


def attach_ratings(
    fights_df: pd.DataFrame, return_states: bool = False
):
    """Convenience: attach both Elo and Glicko-2 columns to a feature DataFrame.

    When return_states=True, also returns (elo_states, glicko_states) for
    persistence via save_latest_ratings().
    """
    if return_states:
        df, elo_states = compute_elo(fights_df, return_states=True)
        df, glicko_states = compute_glicko2(df, return_states=True)
        return df, elo_states, glicko_states
    df = compute_elo(fights_df)
    df = compute_glicko2(df)
    return df


# ---------------------------------------------------------------------------
# Persistence (post-training snapshot used at inference)
# ---------------------------------------------------------------------------

def save_latest_ratings(
    elo_states: dict,
    glicko_states: dict,
    path: Path = LATEST_RATINGS_PATH,
) -> None:
    """
    Persist the final per-(fighter, weight_class) Elo + Glicko state to JSON
    so the predict step can look them up instead of receiving NaN.

    Key format: "<canonical_fighter_id>|<weight_class>".
    """
    out: dict[str, dict] = {}
    keys = set(elo_states) | set(glicko_states)
    for key in keys:
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        fighter_id, wc = key
        if not fighter_id:
            continue
        e = elo_states.get(key)
        g = glicko_states.get(key)
        out[f"{fighter_id}|{wc or 'Unknown'}"] = {
            "elo": float(e.rating) if e else ELO_BASE,
            "elo_n": int(e.n_fights) if e else 0,
            "glicko": float(g.display_rating) if g else GLICKO_DISPLAY_BASE,
            "glicko_rd": float(g.display_rd) if g else 350.0,
            "glicko_n": int(g.n_fights) if g else 0,
            "last_fight_date": (
                g.last_fight_date.isoformat()
                if g and g.last_fight_date else None
            ),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out), encoding="utf-8")
    log.info("Saved latest ratings for %d (fighter, weight_class) keys to %s", len(out), path)


def load_latest_ratings(path: Path = LATEST_RATINGS_PATH) -> dict:
    """Load persisted ratings; empty dict if file missing/unreadable."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("fighter_ratings.json unreadable — using base ratings")
        return {}


def lookup_ratings(
    ratings: dict,
    fighter_id: str,
    weight_class: str | None,
    as_of: date | None = None,
) -> dict:
    """
    Look up an individual fighter's stored ratings, falling back to base values
    when unknown. Inflates Glicko RD for inactivity between last_fight_date and
    `as_of`, mirroring the in-training inactivity rule.
    """
    base = {
        "elo": ELO_BASE,
        "glicko": GLICKO_DISPLAY_BASE,
        "glicko_rd": 350.0,  # max RD for an unknown fighter
        "n_fights": 0,
    }
    if not fighter_id:
        return base
    key = f"{fighter_id}|{weight_class or 'Unknown'}"
    rec = ratings.get(key)
    if not rec:
        return base
    rd = float(rec.get("glicko_rd", 350.0))
    last_str = rec.get("last_fight_date")
    if as_of and last_str:
        try:
            last = date.fromisoformat(last_str)
            days = max(0, (as_of - last).days)
            if days > 0:
                phi = rd / GLICKO_DISPLAY_SCALE
                c = 34.6 / GLICKO_DISPLAY_SCALE
                phi_inflated = math.sqrt(phi**2 + c**2 * (days / 365.0))
                rd = min(350.0, phi_inflated * GLICKO_DISPLAY_SCALE)
        except ValueError:
            pass
    return {
        "elo": float(rec.get("elo", ELO_BASE)),
        "glicko": float(rec.get("glicko", GLICKO_DISPLAY_BASE)),
        "glicko_rd": rd,
        "n_fights": int(rec.get("glicko_n", 0)),
    }
