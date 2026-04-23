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


def _kelly_display(frac) -> str:
    if frac is None or (isinstance(frac, float) and frac != frac) or frac == 0:
        return "—"
    return f"{float(frac)*100:.1f}% of bankroll"


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

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)
    template = env.get_template("dashboard.html")

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = template.render(
        events=events,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        n_bouts=len(predictions),
    )

    out_path = output_dir / "index.html"
    out_path.write_text(rendered, encoding="utf-8")
    log.info("Dashboard written to %s (%d bouts, %d events)", out_path, len(predictions), len(events))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build()
