"""Command-line entry points."""

import click


@click.group()
def ingest():
    """Data ingestion commands."""


@click.group()
def train():
    """Model training commands."""


@click.group()
def predict():
    """Prediction commands."""
