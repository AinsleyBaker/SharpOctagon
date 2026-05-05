"""Tests for Elo and Glicko-2 rating computations."""

import pandas as pd

from ufc_predict.features.ratings import compute_elo, compute_glicko2


def _make_fights():
    return pd.DataFrame([
        {"fight_id": "f1", "date": "2020-01-01", "weight_class": "Lightweight",
         "fighter_a_id": "A", "fighter_b_id": "B", "label": 1},
        {"fight_id": "f2", "date": "2020-03-01", "weight_class": "Lightweight",
         "fighter_a_id": "B", "fighter_b_id": "C", "label": 0},
        {"fight_id": "f3", "date": "2020-06-01", "weight_class": "Lightweight",
         "fighter_a_id": "A", "fighter_b_id": "C", "label": 1},
    ])


def test_elo_adds_columns():
    df = compute_elo(_make_fights())
    assert "elo_a" in df.columns
    assert "elo_b" in df.columns
    assert "diff_elo" in df.columns


def test_elo_winner_gains_rating():
    df = compute_elo(_make_fights())
    # After A beats B in f1, A's elo should have risen vs baseline 1500
    # (We can check that elo_a for fight f3 > elo_a for fight f1 since A won both)
    f1 = df[df["fight_id"] == "f1"].iloc[0]
    f3 = df[df["fight_id"] == "f3"].iloc[0]
    assert f3["elo_a"] > f1["elo_a"]


def test_elo_diff_sign():
    df = compute_elo(_make_fights())
    # Fight 1: A beats B. Pre-fight both at 1500 → diff = 0
    f1 = df[df["fight_id"] == "f1"].iloc[0]
    assert abs(f1["diff_elo"]) < 1.0  # both start at 1500


def test_glicko_adds_columns():
    df = compute_glicko2(_make_fights())
    for col in ["glicko_a", "glicko_b", "glicko_rd_a", "glicko_rd_b", "diff_glicko"]:
        assert col in df.columns


def test_glicko_rd_is_positive():
    df = compute_glicko2(_make_fights())
    assert (df["glicko_rd_a"] > 0).all()
    assert (df["glicko_rd_b"] > 0).all()
