"""Command-line entry points."""

import logging

import click

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@click.group()
def ingest():
    """Data ingestion commands."""


@click.group()
def train():
    """Model training commands."""


@click.group()
def predict():
    """Prediction commands."""


@ingest.command("sportsbet")
def ingest_sportsbet():
    """Fetch UFC odds from SportsBet Australia and update predictions.json."""
    import json
    from pathlib import Path

    from ufc_predict.eval.bet_analysis import analyze_all_fights
    from ufc_predict.ingest.sportsbet_scraper import (
        fetch_ufc_markets,
        match_odds_to_predictions,
        save_markets,
    )

    preds_path = Path("data/predictions.json")
    if not preds_path.exists():
        click.echo("No predictions.json found. Run ufc-predict first.", err=True)
        raise SystemExit(1)

    with open(preds_path) as f:
        predictions = json.load(f)

    click.echo("Fetching SportsBet UFC markets…")
    sb_fights = fetch_ufc_markets()

    if not sb_fights:
        click.echo(
            "No markets returned from SportsBet. The API may be unavailable or no fights listed.",
            err=True,
        )
        raise SystemExit(1)

    click.echo(f"Fetched {len(sb_fights)} fights. Matching to predictions…")
    save_markets(sb_fights)
    predictions = match_odds_to_predictions(sb_fights, predictions)

    matched = sum(1 for p in predictions if p.get("sportsbet_odds"))
    click.echo(f"Matched {matched}/{len(predictions)} fights.")

    predictions = analyze_all_fights(predictions)

    with open(preds_path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)

    click.echo("predictions.json updated with SportsBet odds and EV analysis.")


@train.command("props")
@click.option("--db-url", default=None, help="Override DB URL")
def train_props(db_url):
    """Train prop bet models (method of victory + round prediction)."""
    from ufc_predict.models.prop_models import run_training
    run_training(db_url=db_url)


@predict.command("dashboard")
def build_dashboard():
    """Regenerate the static HTML dashboard from predictions.json."""
    from ufc_predict.serve.build_dashboard import build
    build()
    click.echo("Dashboard built → docs/index.html")
