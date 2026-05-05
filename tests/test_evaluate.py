"""Tests for the evaluation harness — metrics, Kelly, conformal."""

import numpy as np

from ufc_predict.eval.evaluate import (
    american_odds_to_implied_prob,
    compute_metrics,
    kelly_fraction_fn,
    reliability_data,
    remove_vig,
)
from ufc_predict.models.predict import calibrate_conformal, conformal_interval


def test_implied_prob_plus_odds():
    # +150 → 100/(150+100) = 0.4
    assert abs(american_odds_to_implied_prob(150) - 0.4) < 1e-6


def test_implied_prob_minus_odds():
    # -200 → 200/(200+100) = 0.667
    assert abs(american_odds_to_implied_prob(-200) - 2/3) < 1e-4


def test_remove_vig_sums_to_one():
    pa, pb = remove_vig(0.55, 0.50)
    assert abs(pa + pb - 1.0) < 1e-10


def test_kelly_positive_ev():
    # +150 odds (decimal 2.5), we estimate 50% win → EV positive
    frac = kelly_fraction_fn(0.50, 2.5, fraction=1.0)
    assert frac > 0


def test_kelly_negative_ev():
    # -300 odds (decimal 1.333), we estimate 50% win → negative EV → 0 bet
    frac = kelly_fraction_fn(0.50, 1.333, fraction=1.0)
    assert frac == 0.0


def test_compute_metrics_perfect():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.99, 0.99, 0.01, 0.01])
    m = compute_metrics(y, p)
    assert m["auc"] == 1.0
    assert m["log_loss"] < 0.05
    assert m["brier"] < 0.01


def test_reliability_data_shape():
    y = np.array([0, 1] * 50)
    p = np.linspace(0.05, 0.95, 100)
    rel = reliability_data(y, p, n_bins=10)
    assert len(rel) <= 10
    assert "frac_pos" in rel.columns


def test_conformal_interval_bounds():
    preds = np.array([0.3, 0.5, 0.8])
    lo, hi = conformal_interval(preds, halfwidth=0.1)
    assert (lo >= 0).all()
    assert (hi <= 1).all()
    assert (hi >= lo).all()


def test_calibrate_conformal_returns_dict():
    y = np.array([0, 1] * 100)
    p = np.clip(y + np.random.default_rng(0).normal(0, 0.2, 200), 0.01, 0.99)
    result = calibrate_conformal(y, p, alpha=0.10)
    assert "conformal_halfwidth" in result
    assert 0 < result["conformal_halfwidth"] < 1
