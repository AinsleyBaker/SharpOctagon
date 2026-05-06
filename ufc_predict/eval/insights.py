"""Deterministic, rule-based pre-fight insights.

Renders three blocks per upcoming bout:

    1. ``top_factors``        — the matchup factors carrying the most weight
                                 in our model's prediction, ranked by
                                 ``|feature_importance × diff_value|``
    2. ``stat_bars``           — per-side absolute stats with a percent
                                 advantage gauge for the better fighter
    3. ``confidence_drivers``  — short rule-based "what makes us confident"
                                 sentences fired from a curated rule list

NOT LLM-generated. The output is a pure function of the prediction's
feature vector + the model's gain-based feature importances. No randomness,
no per-bout text generation. Same inputs → same insights.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

log = logging.getLogger(__name__)

FEATURE_IMPORTANCES_PATH = Path("models/feature_importances.json")


# ---------------------------------------------------------------------------
# Factor labels: feature column → (display label, format hint, summary template)
# ---------------------------------------------------------------------------
# template tokens:
#   {a}, {b}    — fighter A / B raw value (formatted via fmt)
#   {a_name}, {b_name} — fighter names
#   {leader}, {laggard} — A or B name based on direction
#   {edge}      — formatted absolute difference

FACTOR_LABELS: dict[str, dict] = {
    "diff_elo": {
        "label": "Elo rating",
        "abs_a": "elo_a",
        "abs_b": "elo_b",
        "fmt": ".0f",
        "units": "Elo",
        "good_when": "high",
        "summary": "{leader} carries a {edge} Elo edge over {laggard}",
    },
    "diff_glicko": {
        "label": "Glicko-2 rating",
        "abs_a": "glicko_a",
        "abs_b": "glicko_b",
        "fmt": ".0f",
        "units": "Glicko",
        "good_when": "high",
        "summary": "{leader}'s Glicko-2 outpaces {laggard} by {edge}",
    },
    "diff_age": {
        "label": "Age",
        "abs_a": "a_age",
        "abs_b": "b_age",
        "fmt": ".0f",
        "units": "yrs",
        "good_when": "low",
        "summary": "{leader} is {edge} years younger than {laggard}",
    },
    "diff_post_peak": {
        "label": "Years past peak (class-aware)",
        "fmt": ".1f",
        "units": "yrs past peak",
        "good_when": "low",
        "summary": "{leader} is {edge} fewer years past their weight-class peak",
    },
    "diff_reach_cm": {
        "label": "Reach",
        "abs_a": "a_reach_cm",
        "abs_b": "b_reach_cm",
        "fmt": ".0f",
        "units": "cm",
        "good_when": "high",
        "summary": "{leader} has a {edge} cm reach advantage",
    },
    "diff_height_cm": {
        "label": "Height",
        "abs_a": "a_height_cm",
        "abs_b": "b_height_cm",
        "fmt": ".0f",
        "units": "cm",
        "good_when": "high",
        "summary": "{leader} is {edge} cm taller",
    },
    "diff_slpm": {
        "label": "Striking volume",
        "abs_a": "a_slpm",
        "abs_b": "b_slpm",
        "fmt": ".1f",
        "units": "sig/min",
        "good_when": "high",
        "summary": "{leader} lands {edge} more sig strikes per minute",
    },
    "diff_sig_acc": {
        "label": "Striking accuracy",
        "abs_a": "a_sig_acc",
        "abs_b": "b_sig_acc",
        "fmt": ".0%",
        "units": "accuracy",
        "good_when": "high",
        "summary": "{leader} is {edge} more accurate on the feet",
    },
    "diff_sig_str_def": {
        "label": "Striking defence",
        "abs_a": "a_sig_str_def",
        "abs_b": "b_sig_str_def",
        "fmt": ".0%",
        "units": "TDD",
        "good_when": "high",
        "summary": "{leader} avoids {edge} more incoming strikes",
    },
    "diff_sig_abs_per_min": {
        "label": "Strikes absorbed",
        "abs_a": "a_sig_abs_per_min",
        "abs_b": "b_sig_abs_per_min",
        "fmt": ".1f",
        "units": "abs/min",
        "good_when": "low",
        "summary": "{leader} eats {edge} fewer strikes per minute",
    },
    "diff_td_per_min": {
        "label": "Takedown volume",
        "abs_a": "a_td_per_min",
        "abs_b": "b_td_per_min",
        "fmt": ".2f",
        "units": "TD/min",
        "good_when": "high",
        "summary": "{leader} hits {edge} more takedowns per minute",
    },
    "diff_td_acc": {
        "label": "Takedown accuracy",
        "fmt": ".0%",
        "units": "TD acc",
        "good_when": "high",
        "summary": "{leader} converts {edge} more of their takedown attempts",
    },
    "diff_td_def": {
        "label": "Takedown defence",
        "abs_a": "a_td_def",
        "abs_b": "b_td_def",
        "fmt": ".0%",
        "units": "TDD",
        "good_when": "high",
        "summary": "{leader} stuffs {edge} more incoming takedowns",
    },
    "diff_sub_per_min": {
        "label": "Sub-attempt volume",
        "abs_a": "a_sub_per_min",
        "abs_b": "b_sub_per_min",
        "fmt": ".2f",
        "units": "subs/min",
        "good_when": "high",
        "summary": "{leader} hunts subs more aggressively (+{edge}/min)",
    },
    "diff_ctrl_ratio": {
        "label": "Control time",
        "abs_a": "a_ctrl_ratio",
        "abs_b": "b_ctrl_ratio",
        "fmt": ".0%",
        "units": "ctrl",
        "good_when": "high",
        "summary": "{leader} controls {edge} more cage time",
    },
    "diff_ko_rate": {
        "label": "KO rate",
        "abs_a": "a_ko_rate",
        "abs_b": "b_ko_rate",
        "fmt": ".0%",
        "units": "KO%",
        "good_when": "high",
        "summary": "{leader} finishes by KO {edge} more often",
    },
    "diff_sub_rate": {
        "label": "Submission rate",
        "abs_a": "a_sub_rate",
        "abs_b": "b_sub_rate",
        "fmt": ".0%",
        "units": "Sub%",
        "good_when": "high",
        "summary": "{leader} taps opponents {edge} more often",
    },
    "diff_finish_rate": {
        "label": "Finish rate",
        "abs_a": "a_finish_rate",
        "abs_b": "b_finish_rate",
        "fmt": ".0%",
        "units": "fin%",
        "good_when": "high",
        "summary": "{leader} finishes {edge} more of their wins",
    },
    "diff_ko_loss_rate": {
        "label": "KO durability",
        "abs_a": "a_ko_loss_rate",
        "abs_b": "b_ko_loss_rate",
        "fmt": ".0%",
        "units": "KO loss%",
        "good_when": "low",
        "summary": "{leader} has been KO'd {edge} less often",
    },
    "diff_sub_loss_rate": {
        "label": "Sub defence",
        "abs_a": "a_sub_loss_rate",
        "abs_b": "b_sub_loss_rate",
        "fmt": ".0%",
        "units": "Sub loss%",
        "good_when": "low",
        "summary": "{leader} has been submitted {edge} less often",
    },
    "diff_finish_loss_rate": {
        "label": "Durability",
        "abs_a": "a_finish_loss_rate",
        "abs_b": "b_finish_loss_rate",
        "fmt": ".0%",
        "units": "fin loss%",
        "good_when": "low",
        "summary": "{leader} has been finished {edge} less often",
    },
    "diff_l3_win_rate": {
        "label": "Last-3 form",
        "abs_a": "a_l3_win_rate",
        "abs_b": "b_l3_win_rate",
        "fmt": ".0%",
        "units": "L3 win%",
        "good_when": "high",
        "summary": "{leader} carries the hotter recent form ({edge} better L3)",
    },
    "diff_win_streak": {
        "label": "Win streak",
        "abs_a": "a_win_streak",
        "abs_b": "b_win_streak",
        "fmt": ".0f",
        "units": "wins",
        "good_when": "high",
        "summary": "{leader} rides a {edge}-fight longer win streak",
    },
    "diff_loss_streak": {
        "label": "Loss streak",
        "fmt": ".0f",
        "units": "losses",
        "good_when": "low",
        "summary": "{leader} comes in on a {edge}-fight cleaner skid",
    },
    "diff_sos_avg_opp_elo": {
        "label": "Strength of schedule (avg)",
        "fmt": ".0f",
        "units": "Elo",
        "good_when": "high",
        "summary": "{leader} has fought {edge} Elo-points stronger competition",
    },
    "diff_sos_quality_wins": {
        "label": "Quality wins",
        "fmt": ".1f",
        "units": "z",
        "good_when": "high",
        "summary": "{leader} has the deeper résumé of wins over Top-tier opposition",
    },
    "diff_expected_strikes_landed": {
        "label": "Striking matchup edge",
        "fmt": ".1f",
        "units": "sig/min",
        "good_when": "high",
        "summary": "{leader}'s striking attacks {laggard}'s defensive weakness",
    },
    "diff_expected_td_landed": {
        "label": "Wrestling matchup edge",
        "fmt": ".2f",
        "units": "TD/min",
        "good_when": "high",
        "summary": "{leader}'s wrestling attacks {laggard}'s takedown defence",
    },
    "diff_expected_ko_threat": {
        "label": "KO threat × chin",
        "fmt": ".2f",
        "units": "KO threat",
        "good_when": "high",
        "summary": "{leader}'s KO power lines up with {laggard}'s chin history",
    },
    "diff_expected_sub_threat": {
        "label": "Submission threat × scramble",
        "fmt": ".2f",
        "units": "sub threat",
        "good_when": "high",
        "summary": "{leader}'s sub game targets {laggard}'s defensive weak spots",
    },
    "diff_finish_threat": {
        "label": "Finish vs durability mismatch",
        "fmt": ".2f",
        "good_when": "high",
        "summary": "{leader} brings finishing pressure {laggard} struggles to absorb",
    },
    "diff_keep_standing": {
        "label": "Range control",
        "fmt": ".1f",
        "good_when": "high",
        "summary": "{leader} dictates where the fight happens (striking + TDD)",
    },
    "diff_wrestled_pressure": {
        "label": "Wrestling pressure faced",
        "fmt": ".2f",
        "good_when": "low",
        "summary": "{leader} faces less wrestling pressure than {laggard}",
    },
}


# ---------------------------------------------------------------------------
# Stat bar definitions: per-row absolute stats for both fighters.
# ---------------------------------------------------------------------------

STAT_BARS: list[dict] = [
    {"label": "Striking volume", "a": "a_slpm", "b": "b_slpm",
     "units": "sig/min", "fmt": ".1f", "higher_better": True},
    {"label": "Striking accuracy", "a": "a_sig_acc", "b": "b_sig_acc",
     "units": "acc", "fmt": ".0%", "higher_better": True},
    {"label": "Striking defence", "a": "a_sig_str_def", "b": "b_sig_str_def",
     "units": "def", "fmt": ".0%", "higher_better": True},
    {"label": "Takedown volume", "a": "a_td_per_min", "b": "b_td_per_min",
     "units": "TD/min", "fmt": ".2f", "higher_better": True},
    {"label": "Takedown defence", "a": "a_td_def", "b": "b_td_def",
     "units": "TDD", "fmt": ".0%", "higher_better": True},
    {"label": "Finish rate", "a": "a_finish_rate", "b": "b_finish_rate",
     "units": "fin%", "fmt": ".0%", "higher_better": True},
    {"label": "Durability (lower = better)", "a": "a_finish_loss_rate", "b": "b_finish_loss_rate",
     "units": "fin loss%", "fmt": ".0%", "higher_better": False},
]


# ---------------------------------------------------------------------------
# Confidence-driver rules. Each rule is a (predicate, template) pair where
# predicate(pred) returns a strength score (>0 fires, sorted desc) or 0.
# ---------------------------------------------------------------------------

def _f(pred: dict, key: str, default: float | None = None) -> float | None:
    v = pred.get(key)
    if v is None:
        return default
    try:
        v = float(v)
        if v != v:
            return default
        return v
    except (TypeError, ValueError):
        return default


def _name(pred: dict, side: str) -> str:
    raw = pred.get(f"fighter_{side}_name") or ("Fighter " + side.upper())
    # Some upstream rows have "nan FIRSTNAME" or "None LASTNAME" because the
    # DB join lost a part of the name. Strip those tokens so they don't
    # bleed into rendered insights as the literal word "nan".
    parts = [t for t in str(raw).split() if t.lower() not in ("nan", "none")]
    return " ".join(parts) if parts else ("Fighter " + side.upper())


_CONFIDENCE_RULES: list = []


def _rule(score_fn, template_fn):
    _CONFIDENCE_RULES.append((score_fn, template_fn))


def _pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v*100:.0f}%"


_rule(
    lambda p: max(0.0, (1.0 if (_f(p, "a_never_finished") == 1) else 0.0)
                  + (_f(p, "a_n_fights", 0) / 25 if _f(p, "a_n_fights", 0) >= 5 else 0)),
    lambda p: f"{_name(p, 'a')} has never been finished in {int(_f(p, 'a_n_fights') or 0)} fights",
)
_rule(
    lambda p: max(0.0, (1.0 if (_f(p, "b_never_finished") == 1) else 0.0)
                  + (_f(p, "b_n_fights", 0) / 25 if _f(p, "b_n_fights", 0) >= 5 else 0)),
    lambda p: f"{_name(p, 'b')} has never been finished in {int(_f(p, 'b_n_fights') or 0)} fights",
)


def _wrestling_mismatch(p):
    a_td = _f(p, "a_td_per_min", 0) or 0
    b_tdd = _f(p, "b_td_def", 0) or 0
    if a_td > 1.5 and b_tdd is not None and b_tdd < 0.6:
        return (a_td - 1.0) + (0.6 - b_tdd)
    b_td = _f(p, "b_td_per_min", 0) or 0
    a_tdd = _f(p, "a_td_def", 0) or 0
    if b_td > 1.5 and a_tdd is not None and a_tdd < 0.6:
        return (b_td - 1.0) + (0.6 - a_tdd)
    return 0.0


def _wrestling_template(p):
    a_td = _f(p, "a_td_per_min", 0) or 0
    a_tdd = _f(p, "a_td_def", 0) or 0
    b_td = _f(p, "b_td_per_min", 0) or 0
    b_tdd = _f(p, "b_td_def", 0) or 0
    if a_td > b_td:
        return (
            f"{_name(p, 'a')}'s {a_td:.1f} TD/min vs {_name(p, 'b')}'s "
            f"{_pct(b_tdd)} TDD points to a clear wrestling mismatch"
        )
    return (
        f"{_name(p, 'b')}'s {b_td:.1f} TD/min vs {_name(p, 'a')}'s "
        f"{_pct(a_tdd)} TDD points to a clear wrestling mismatch"
    )


_rule(_wrestling_mismatch, _wrestling_template)


def _striking_volume_gap(p):
    a = _f(p, "a_slpm", 0) or 0
    b = _f(p, "b_slpm", 0) or 0
    return abs(a - b) if max(a, b) >= 4.0 else 0.0


def _striking_volume_template(p):
    a = _f(p, "a_slpm", 0) or 0
    b = _f(p, "b_slpm", 0) or 0
    if a >= b:
        return (
            f"{_name(p, 'a')} out-volumes opponents at "
            f"{a:.1f} sig/min vs {_name(p, 'b')}'s {b:.1f}"
        )
    return (
        f"{_name(p, 'b')} out-volumes opponents at "
        f"{b:.1f} sig/min vs {_name(p, 'a')}'s {a:.1f}"
    )


_rule(_striking_volume_gap, _striking_volume_template)


def _ko_threat_chin(p):
    a_ko = _f(p, "a_ko_rate", 0) or 0
    b_kloss = _f(p, "b_ko_loss_rate", 0) or 0
    if a_ko >= 0.4 and b_kloss is not None and b_kloss >= 0.2:
        return a_ko + b_kloss
    b_ko = _f(p, "b_ko_rate", 0) or 0
    a_kloss = _f(p, "a_ko_loss_rate", 0) or 0
    if b_ko >= 0.4 and a_kloss is not None and a_kloss >= 0.2:
        return b_ko + a_kloss
    return 0.0


def _ko_threat_template(p):
    a_ko = _f(p, "a_ko_rate", 0) or 0
    a_kloss = _f(p, "a_ko_loss_rate", 0) or 0
    b_ko = _f(p, "b_ko_rate", 0) or 0
    b_kloss = _f(p, "b_ko_loss_rate", 0) or 0
    if a_ko + b_kloss > b_ko + a_kloss:
        return (
            f"{_name(p, 'a')} KOs {_pct(a_ko)} of opponents — "
            f"{_name(p, 'b')} has been KO'd in {_pct(b_kloss)} of losses"
        )
    return (
        f"{_name(p, 'b')} KOs {_pct(b_ko)} of opponents — "
        f"{_name(p, 'a')} has been KO'd in {_pct(a_kloss)} of losses"
    )


_rule(_ko_threat_chin, _ko_threat_template)


def _experience_gap(p):
    a_n = _f(p, "a_n_fights", 0) or 0
    b_n = _f(p, "b_n_fights", 0) or 0
    diff = abs(a_n - b_n)
    if diff >= 5 and min(a_n, b_n) <= 5:
        return diff / 10.0
    return 0.0


def _experience_template(p):
    a_n = int(_f(p, "a_n_fights", 0) or 0)
    b_n = int(_f(p, "b_n_fights", 0) or 0)
    if a_n > b_n:
        return f"{_name(p, 'a')} has a {a_n - b_n}-fight UFC experience edge ({a_n} vs {b_n})"
    return f"{_name(p, 'b')} has a {b_n - a_n}-fight UFC experience edge ({b_n} vs {a_n})"


_rule(_experience_gap, _experience_template)


def _reach_gap(p):
    a = _f(p, "a_reach_cm")
    b = _f(p, "b_reach_cm")
    if a is None or b is None:
        return 0.0
    return max(0.0, abs(a - b) - 5)


def _reach_template(p):
    a = _f(p, "a_reach_cm") or 0
    b = _f(p, "b_reach_cm") or 0
    if a > b:
        return f"{_name(p, 'a')} owns a {a - b:.0f} cm reach advantage at {a:.0f} cm"
    return f"{_name(p, 'b')} owns a {b - a:.0f} cm reach advantage at {b:.0f} cm"


_rule(_reach_gap, _reach_template)


def _striking_def_gap(p):
    a = _f(p, "a_sig_str_def")
    b = _f(p, "b_sig_str_def")
    if a is None or b is None:
        return 0.0
    return abs(a - b) if max(a or 0, b or 0) >= 0.6 else 0.0


def _striking_def_template(p):
    a = _f(p, "a_sig_str_def") or 0
    b = _f(p, "b_sig_str_def") or 0
    if a > b:
        return f"{_name(p, 'a')} carries the better striking defence ({_pct(a)} vs {_pct(b)})"
    return f"{_name(p, 'b')} carries the better striking defence ({_pct(b)} vs {_pct(a)})"


_rule(_striking_def_gap, _striking_def_template)


def _streak_gap(p):
    a_w = int(_f(p, "a_win_streak", 0) or 0)
    b_w = int(_f(p, "b_win_streak", 0) or 0)
    return abs(a_w - b_w) if max(a_w, b_w) >= 3 else 0.0


def _streak_template(p):
    a_w = int(_f(p, "a_win_streak", 0) or 0)
    b_w = int(_f(p, "b_win_streak", 0) or 0)
    if a_w > b_w:
        return f"{_name(p, 'a')} rides a {a_w}-fight win streak ({_name(p, 'b')}: {b_w})"
    return f"{_name(p, 'b')} rides a {b_w}-fight win streak ({_name(p, 'a')}: {a_w})"


_rule(_streak_gap, _streak_template)


def _age_gap(p):
    a = _f(p, "a_age")
    b = _f(p, "b_age")
    if a is None or b is None:
        return 0.0
    diff = abs(a - b)
    return max(0.0, diff - 4)  # >4 yrs to fire


def _age_template(p):
    a = _f(p, "a_age") or 0
    b = _f(p, "b_age") or 0
    if a < b:
        return f"{_name(p, 'a')} is {b - a:.0f} years younger ({a:.0f} vs {b:.0f})"
    return f"{_name(p, 'b')} is {a - b:.0f} years younger ({b:.0f} vs {a:.0f})"


_rule(_age_gap, _age_template)


def _sub_threat_gap(p):
    a = _f(p, "a_sub_per_min", 0) or 0
    b_sloss = _f(p, "b_sub_loss_rate", 0) or 0
    bs = _f(p, "b_sub_per_min", 0) or 0
    a_sloss = _f(p, "a_sub_loss_rate", 0) or 0
    s1 = (a - 0.3) + (b_sloss - 0.15) if a > 0.3 and b_sloss > 0.15 else 0
    s2 = (bs - 0.3) + (a_sloss - 0.15) if bs > 0.3 and a_sloss > 0.15 else 0
    return max(s1, s2)


def _sub_threat_template(p):
    a = _f(p, "a_sub_per_min", 0) or 0
    b = _f(p, "b_sub_per_min", 0) or 0
    a_sloss = _f(p, "a_sub_loss_rate", 0) or 0
    b_sloss = _f(p, "b_sub_loss_rate", 0) or 0
    if a > b:
        return (
            f"{_name(p, 'a')}'s {a:.2f} subs/min vs {_name(p, 'b')}'s "
            f"{_pct(b_sloss)} sub-loss rate flags a real grappling threat"
        )
    return (
        f"{_name(p, 'b')}'s {b:.2f} subs/min vs {_name(p, 'a')}'s "
        f"{_pct(a_sloss)} sub-loss rate flags a real grappling threat"
    )


_rule(_sub_threat_gap, _sub_threat_template)


def _confidence_low(p):
    """High uncertainty isn't a confidence driver — it's an anti-driver. We
    surface it explicitly so the panel doesn't oversell pick'em fights."""
    std = _f(p, "uncertainty_std")
    if std is None:
        return 0.0
    return std if std >= 0.12 else 0.0


def _confidence_low_template(p):
    return (
        "Model uncertainty is elevated — both fighters' profiles cluster "
        "around the same outcome bands"
    )


_rule(_confidence_low, _confidence_low_template)


def _short_notice_template(p, side):
    return f"{_name(p, side)} stepped in on short notice — full camp asymmetry"


def _short_notice_score(p):
    if int(_f(p, "a_short_notice", 0) or 0) == 1:
        return 1.0
    if int(_f(p, "b_short_notice", 0) or 0) == 1:
        return 1.0
    return 0.0


def _short_notice_text(p):
    if int(_f(p, "a_short_notice", 0) or 0) == 1:
        return _short_notice_template(p, "a")
    return _short_notice_template(p, "b")


_rule(_short_notice_score, _short_notice_text)


# ---------------------------------------------------------------------------
# Loaders / formatters
# ---------------------------------------------------------------------------

_IMPORTANCES_CACHE: dict[str, float] | None = None


def load_feature_importances() -> dict[str, float]:
    global _IMPORTANCES_CACHE
    if _IMPORTANCES_CACHE is not None:
        return _IMPORTANCES_CACHE
    if not FEATURE_IMPORTANCES_PATH.exists():
        log.warning(
            "feature_importances.json missing — top_factors fall back to a flat ranking"
        )
        _IMPORTANCES_CACHE = {}
        return _IMPORTANCES_CACHE
    try:
        _IMPORTANCES_CACHE = json.loads(
            FEATURE_IMPORTANCES_PATH.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError:
        _IMPORTANCES_CACHE = {}
    return _IMPORTANCES_CACHE


def _format(value: float, fmt: str) -> str:
    try:
        return format(value, fmt)
    except (TypeError, ValueError):
        return str(value)


def _direction_for_diff(diff_value: float, good_when: str) -> str:
    if diff_value > 0:
        return "a" if good_when == "high" else "b"
    if diff_value < 0:
        return "b" if good_when == "high" else "a"
    return "tie"


def _factor_summary(
    pred: dict, feature: str, meta: dict, diff_value: float
) -> tuple[str, str]:
    """Build the (label, summary) pair for a top factor. Returns ("", "")
    if any required value is missing/NaN — the caller drops these silently.
    """
    if diff_value is None or (isinstance(diff_value, float) and math.isnan(diff_value)):
        return "", ""
    direction = _direction_for_diff(diff_value, meta.get("good_when", "high"))
    if direction == "tie":
        return "", ""
    leader = _name(pred, direction)
    laggard = _name(pred, "b" if direction == "a" else "a")
    fmt = meta.get("fmt", ".2f")
    edge_val = abs(diff_value)
    template = meta.get("summary", "{leader} has the edge")
    summary = template.format(
        leader=leader,
        laggard=laggard,
        a_name=_name(pred, "a"),
        b_name=_name(pred, "b"),
        edge=_format(edge_val, fmt),
    )
    return meta.get("label", feature), summary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _diff_from(prediction: dict, feature: str, meta: dict) -> float | None:
    """Resolve a diff_* feature value, deriving from a/b absolutes when the
    serialised prediction doesn't expose the diff column directly.
    """
    direct = _f(prediction, feature)
    if direct is not None:
        return direct
    abs_a_key = meta.get("abs_a")
    abs_b_key = meta.get("abs_b")
    if not abs_a_key or not abs_b_key:
        # Fall back to the canonical "a_X" / "b_X" pair derived from the
        # diff key — covers diff_slpm → a_slpm/b_slpm etc.
        if feature.startswith("diff_"):
            stem = feature[len("diff_"):]
            abs_a_key = f"a_{stem}"
            abs_b_key = f"b_{stem}"
        else:
            return None
    a_val = _f(prediction, abs_a_key)
    b_val = _f(prediction, abs_b_key)
    if a_val is None or b_val is None:
        return None
    return a_val - b_val


def generate_insights(prediction: dict) -> dict:
    """Build the insights block for a single upcoming bout's prediction."""
    importances = load_feature_importances()

    # ---- top_factors ------------------------------------------------------
    scored: list[tuple[float, str, dict, float]] = []
    for feature, meta in FACTOR_LABELS.items():
        diff_val = _diff_from(prediction, feature, meta)
        if diff_val is None:
            continue
        # Use absolute diff × importance for ranking. A flat default of 1.0
        # avoids a zeroed score when importances aren't loaded.
        weight = importances.get(feature, 1.0) if importances else 1.0
        score = abs(diff_val) * weight
        scored.append((score, feature, meta, diff_val))
    scored.sort(key=lambda t: -t[0])

    top_factors: list[dict] = []
    for score, feature, meta, diff_val in scored:
        if len(top_factors) >= 3:
            break
        label, summary = _factor_summary(prediction, feature, meta, diff_val)
        if not label:
            continue
        # Reject only the literal string "None" or NaN-coded numeric output.
        # Lower-case "nan" can appear inside legitimate words (e.g. an Asian
        # fighter named "Banana") — we already sanitise upstream names.
        if "None" in summary or " nan " in f" {summary} ":
            continue
        # Keep magnitude on a 0..1ish scale by dividing by the top score
        magnitude = score / max(scored[0][0], 1e-9) if scored else 0.0
        direction = _direction_for_diff(diff_val, meta.get("good_when", "high"))
        top_factors.append({
            "label": label,
            "summary": summary,
            "magnitude": round(float(magnitude), 3),
            "direction": direction,  # 'a' or 'b'
        })

    # Fallback: if we couldn't compute any diff (super-thin per-fighter
    # absolute data), surface the model's overall lean as a single factor so
    # the panel renders something rather than nothing.
    if not top_factors:
        prob_a = _f(prediction, "prob_a_wins")
        if prob_a is not None and abs(prob_a - 0.5) > 0.02:
            leader = "a" if prob_a > 0.5 else "b"
            top_factors.append({
                "label": "Model lean",
                "summary": (
                    f"{_name(prediction, leader)} is favoured at "
                    f"{max(prob_a, 1-prob_a)*100:.0f}% — fight-history features sparse"
                ),
                "magnitude": float(round(abs(prob_a - 0.5) * 2, 3)),
                "direction": leader,
            })

    # ---- stat_bars --------------------------------------------------------
    stat_bars: list[dict] = []
    for bar in STAT_BARS:
        a_val = _f(prediction, bar["a"])
        b_val = _f(prediction, bar["b"])
        if a_val is None or b_val is None:
            continue
        higher_better = bar.get("higher_better", True)
        # Compute percentage edge of the better side over the worse.
        # Use the larger of the two to avoid divide-by-tiny.
        denom = max(abs(a_val), abs(b_val), 1e-6)
        if higher_better:
            adv_side = "a" if a_val >= b_val else "b"
            adv_pct = abs(a_val - b_val) / denom * 100.0
        else:
            adv_side = "a" if a_val <= b_val else "b"
            adv_pct = abs(a_val - b_val) / denom * 100.0
        adv_pct = max(0.0, min(99.0, adv_pct))
        stat_bars.append({
            "label": bar["label"],
            "a_value": round(float(a_val), 4),
            "b_value": round(float(b_val), 4),
            "a_display": _format(a_val, bar.get("fmt", ".2f")),
            "b_display": _format(b_val, bar.get("fmt", ".2f")),
            "advantage_pct": round(float(adv_pct), 1),
            "advantage_side": adv_side,
            "units": bar.get("units", ""),
        })
    # Cap at 6 bars (the brief allows 4–6)
    stat_bars = stat_bars[:6]

    # ---- confidence_drivers ----------------------------------------------
    rule_scores: list[tuple[float, str]] = []
    for score_fn, template_fn in _CONFIDENCE_RULES:
        try:
            score = float(score_fn(prediction) or 0.0)
        except Exception:
            score = 0.0
        if score <= 0:
            continue
        try:
            text = str(template_fn(prediction) or "")
        except Exception:
            continue
        if not text or "None" in text or " nan " in f" {text} ":
            continue
        rule_scores.append((score, text))
    rule_scores.sort(key=lambda t: -t[0])
    drivers = [t[1] for t in rule_scores[:3]]
    # Always emit at least 2 — pad with a generic but factual line if needed.
    if len(drivers) < 2:
        std = _f(prediction, "uncertainty_std")
        if std is not None:
            drivers.append(f"Model uncertainty: ±{std*100:.0f}pp on the win probability")
        prob_a = _f(prediction, "prob_a_wins")
        if prob_a is not None:
            drivers.append(
                f"Model favours {_name(prediction, 'a')} at "
                f"{prob_a*100:.0f}% / {_name(prediction, 'b')} at {(1-prob_a)*100:.0f}%"
            )
    # Keep at most 3
    drivers = [d for d in drivers if d][:3]

    return {
        "top_factors": top_factors,
        "stat_bars": stat_bars,
        "confidence_drivers": drivers,
    }


def attach_insights(predictions: list[dict]) -> list[dict]:
    """Add ``insights`` dict to every prediction. Returns the input list."""
    for p in predictions:
        try:
            p["insights"] = generate_insights(p)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("insights generation failed for one bout (%s)", exc)
            p["insights"] = {"top_factors": [], "stat_bars": [], "confidence_drivers": []}
    return predictions
