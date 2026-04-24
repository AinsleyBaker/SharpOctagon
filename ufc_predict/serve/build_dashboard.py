"""
Stage 12 — Static HTML dashboard generator.

Reads data/predictions.json and renders a fight-card dashboard to docs/index.html.
Designed to be committed to GitHub Pages (branch deploy from /docs) or any static host.
Regenerated after every prediction run.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

log = logging.getLogger(__name__)

PREDICTIONS_PATH = Path("data/predictions.json")
TEMPLATES_DIR    = Path(__file__).parent / "templates"
OUTPUT_DIR       = Path("docs")


def load_predictions() -> list[dict]:
    if not PREDICTIONS_PATH.exists():
        return []
    with open(PREDICTIONS_PATH) as f:
        return json.load(f)


def _group_by_event(predictions: list[dict]) -> list[dict]:
    events: dict[str, dict] = {}
    for p in predictions:
        key = f"{p.get('event_date')}|{p.get('event_name', '')}"
        if key not in events:
            events[key] = {
                "event_name": p.get("event_name", "Unknown Event"),
                "event_date": p.get("event_date"),
                "bouts": [],
            }
        events[key]["bouts"].append(p)
    return sorted(events.values(), key=lambda e: e["event_date"] or "")


def _format_prob(p) -> str:
    if p is None or (isinstance(p, float) and p != p):  # NaN check
        return "—"
    return f"{float(p)*100:.1f}%"


def _format_ci(lo, hi) -> str:
    if lo is None or hi is None:
        return ""
    return f"{float(lo)*100:.0f}–{float(hi)*100:.0f}%"


def _confidence_tier(std) -> str:
    if std is None or (isinstance(std, float) and std != std):
        return "—"
    s = float(std)
    if s < 0.07:
        return "High"
    if s < 0.12:
        return "Medium"
    return "Low"


def _data_quality(a_n, b_n) -> str:
    n = min(int(a_n or 0), int(b_n or 0))
    if n >= 10:
        return "good"
    if n >= 5:
        return "limited"
    return "sparse"


def _streak_display(win_s, loss_s) -> str:
    win_s  = int(win_s  or 0)
    loss_s = int(loss_s or 0)
    if win_s > 0:
        return f"{win_s}W"
    if loss_s > 0:
        return f"{loss_s}L"
    return "—"


def _kelly_display(frac) -> str:
    if frac is None or (isinstance(frac, float) and frac != frac) or frac == 0:
        return "—"
    return f"{float(frac)*100:.1f}% of bankroll"


def _enrich_props(bout: dict) -> None:
    """Add formatted prop probability strings to a bout dict."""
    props = bout.get("props") or {}
    if not props:
        bout["has_props"] = False
        return

    bout["has_props"] = True
    fa = bout.get("fighter_a_name", "A")
    fb = bout.get("fighter_b_name", "B")

    def pct(key: str) -> str:
        v = props.get(key)
        return f"{float(v)*100:.1f}%" if v is not None else "—"

    bout["prop_rows"] = [
        {"label": f"{fa} wins by KO/TKO",  "prob": pct("prob_a_wins_ko_tko")},
        {"label": f"{fa} wins by Submission", "prob": pct("prob_a_wins_sub")},
        {"label": f"{fa} wins by Decision",  "prob": pct("prob_a_wins_dec")},
        {"label": f"{fb} wins by KO/TKO",  "prob": pct("prob_b_wins_ko_tko")},
        {"label": f"{fb} wins by Submission", "prob": pct("prob_b_wins_sub")},
        {"label": f"{fb} wins by Decision",  "prob": pct("prob_b_wins_dec")},
        {"label": "Goes to Decision",       "prob": pct("prob_decision")},
    ]

    prob_rounds = props.get("prob_rounds") or {}
    bout["round_rows"] = [
        {"label": f"Ends in Round {i}", "prob": f"{float(prob_rounds.get(f'R{i}', 0))*100:.1f}%"}
        for i in range(1, 6)
        if prob_rounds.get(f"R{i}", 0) > 0
    ]


def _enrich_bet_analysis(bout: dict) -> None:
    """Format bet_analysis list for display in the template."""
    bets = bout.get("bet_analysis") or []
    formatted = []
    for bet in bets:
        ev = bet.get("ev_pct", 0)
        kelly_pct = bet.get("kelly_pct", 0)
        formatted.append({
            "bet_type":    bet.get("bet_type", ""),
            "description": bet.get("description", ""),
            "our_prob":    f"{float(bet.get('our_prob', 0))*100:.1f}%",
            "sb_odds":     f"{float(bet.get('sb_odds', 0)):.2f}",
            "implied_prob": f"{float(bet.get('implied_prob', 0))*100:.1f}%",
            "edge":        f"{float(bet.get('edge', 0))*100:+.1f}%",
            "ev_pct":      f"{ev:+.1f}%",
            "kelly":       f"{kelly_pct:.1f}% bank" if kelly_pct > 0 else "—",
            "is_value":    bet.get("is_value", False),
            "ev_raw":      ev,
        })
    bout["bet_rows"] = formatted
    bout["has_bets"] = bool(formatted)
    bout["value_bet_count"] = sum(1 for b in formatted if b["is_value"])


def build(output_dir: Path = OUTPUT_DIR) -> None:
    predictions = load_predictions()
    if not predictions:
        log.warning("No predictions found at %s", PREDICTIONS_PATH)

    events = _group_by_event(predictions)

    # Enrich each bout for display
    for event in events:
        for bout in event["bouts"]:
            bout["prob_a_pct"] = _format_prob(bout.get("prob_a_wins"))
            bout["prob_b_pct"] = _format_prob(bout.get("prob_b_wins"))

            lo_key = next((k for k in bout if k.startswith("ci_") and k.endswith("_lo")), None)
            hi_key = next((k for k in bout if k.startswith("ci_") and k.endswith("_hi")), None)
            bout["ci_display"] = _format_ci(
                bout.get(lo_key), bout.get(hi_key)
            ) if lo_key and hi_key else ""

            bout["kelly_display"] = _kelly_display(bout.get("kelly_fraction"))
            bout["std_pct"] = _format_prob(bout.get("uncertainty_std"))

            # Determine favourite
            pa = bout.get("prob_a_wins") or 0.5
            if abs(pa - 0.5) < 0.03:
                bout["favourite"] = "toss-up"
            elif pa > 0.5:
                bout["favourite"] = "a"
            else:
                bout["favourite"] = "b"

            # Confidence tier and data quality
            bout["confidence_tier"] = _confidence_tier(bout.get("uncertainty_std"))
            bout["data_quality"] = _data_quality(
                bout.get("a_n_fights"), bout.get("b_n_fights")
            )
            bout["a_streak"] = _streak_display(
                bout.get("a_win_streak"), bout.get("a_loss_streak")
            )
            bout["b_streak"] = _streak_display(
                bout.get("b_win_streak"), bout.get("b_loss_streak")
            )
            def _l3(v):
                if v is None or (isinstance(v, float) and v != v):
                    return "—"
                return f"{float(v)*100:.0f}%"
            bout["a_l3"] = _l3(bout.get("a_l3_win_rate"))
            bout["b_l3"] = _l3(bout.get("b_l3_win_rate"))
            bout["a_n_fights"] = int(bout.get("a_n_fights") or 0)
            bout["b_n_fights"] = int(bout.get("b_n_fights") or 0)

            # Prop probabilities
            _enrich_props(bout)
            # EV / Kelly bet analysis
            _enrich_bet_analysis(bout)

            # SportsBet moneyline odds for display
            sb = bout.get("sportsbet_odds") or {}
            bout["sb_odds_a"] = f"{sb['moneyline_a']:.2f}" if sb.get("moneyline_a") else None
            bout["sb_odds_b"] = f"{sb['moneyline_b']:.2f}" if sb.get("moneyline_b") else None

    from ufc_predict.eval.bet_analysis import top_value_bets
    top_bets = top_value_bets(predictions, n=30)

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)
    template = env.get_template("dashboard.html")

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = template.render(
        events=events,
        top_bets=top_bets,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        n_bouts=len(predictions),
    )

    out_path = output_dir / "index.html"
    out_path.write_text(rendered, encoding="utf-8")
    log.info("Dashboard written to %s (%d bouts, %d events)", out_path, len(predictions), len(events))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build()
