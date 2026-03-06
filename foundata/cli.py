import sys
from pathlib import Path

import click
import polars as pl

from foundata import config_validator, verify
from foundata.run import runner

_DEFAULT_CONFIGS_ROOT = Path(__file__).parent.parent / "configs"


@click.group()
def cli():
    """foundata — household travel survey aggregation toolkit."""


@cli.command("run")
@click.option(
    "--data-root",
    "-d",
    required=True,
    type=click.Path(),
    help="Base data directory, e.g. ~/Data/foundata",
)
@click.option(
    "--output",
    "-o",
    default="output",
    show_default=True,
    type=click.Path(),
    help="Directory where CSVs and PNGs are written, defaults to ./output",
)
@click.option(
    "--select",
    "-s",
    multiple=True,
    help="Comma-separated list of sources to process (e.g. --select nhts --select ktdb). ",
    show_default=True,
)
@click.option(
    "--omit",
    "-x",
    multiple=True,
    help="Comma-separated list of sources to omit (e.g. --omit nhts --omit ktdb). ",
    show_default=True,
)
def run(data_root, output, select, omit):
    """Run the data processing pipeline end-to-end."""
    if select and omit:
        click.echo("Cannot use both --select and --omit options.", err=True)
        sys.exit(1)
    runner(data_root, output, select, omit)


@cli.command("validate-config")
@click.argument("source", required=False)
@click.option(
    "--all", "all_sources", is_flag=True, help="Validate all sources."
)
@click.option(
    "--configs-root",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Path to configs directory (default: configs/ in project root).",
)
def validate_config(source, all_sources, configs_root):
    """Validate YAML config(s) against the template schema.

    SOURCE is the survey name (e.g. nhts). Pass --all to validate every source.
    """
    root = Path(configs_root) if configs_root else _DEFAULT_CONFIGS_ROOT

    if all_sources:
        ok = config_validator.validate_all_sources(root)
    elif source:
        ok = config_validator.validate_source(source, root)
    else:
        click.echo("Provide a SOURCE argument or pass --all.", err=True)
        sys.exit(1)

    if not ok:
        sys.exit(1)


@cli.command("validate-table")
@click.argument("attributes_csv", type=click.Path(exists=True))
@click.argument("trips_csv", type=click.Path(exists=True))
def validate_table(attributes_csv, trips_csv):
    """Validate pipeline output CSVs against the template schema.

    ATTRIBUTES_CSV and TRIPS_CSV are paths to the output files produced by the
    pipeline (one row per person and one row per trip, respectively).
    """
    attributes = pl.read_csv(attributes_csv)
    trips = pl.read_csv(trips_csv)
    ok = verify.columns(attributes, trips)
    if not ok:
        sys.exit(1)
