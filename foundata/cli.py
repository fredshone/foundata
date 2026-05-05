import sys
from pathlib import Path
from typing import Optional

import click
import polars as pl

from foundata import config_validator, post_process, verify
from foundata import filter as flt
from foundata.run import runner

_DEFAULT_CONFIGS_ROOT = Path(__file__).parent.parent / "configs"


def _default_out(input_path: str, suffix: str) -> Path:
    """Derive a default output path by appending *suffix* to the input stem."""
    p = Path(input_path)
    return p.parent / f"{p.stem}{suffix}{p.suffix}"


def _resolve_out(
    explicit: Optional[str],
    out_dir: Optional[str],
    input_path: Optional[str],
    suffix: str,
) -> Optional[Path]:
    if explicit:
        path = Path(explicit)
    elif input_path is None:
        return None
    else:
        stem_path = _default_out(input_path, suffix)
        path = Path(out_dir) / stem_path.name if out_dir else stem_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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
@click.option(
    "--home-based/--no-home-based",
    "-hb/-no-hb",
    default=False,
    show_default=True,
    help="Whether to only include home-based trips (i.e. those with 'home' as the origin or destination activity).",
)
@click.option(
    "--filter-consecutive/--no-filter-consecutive",
    "-fc/-no-fc",
    default=False,
    show_default=True,
    help="Whether to filter out consecutive home, work and education activities.",
)
def run(data_root, output, select, omit, home_based, filter_consecutive):
    """Run the data processing pipeline end-to-end."""
    if select and omit:
        click.echo("Cannot use both --select and --omit options.", err=True)
        sys.exit(1)
    runner(data_root, output, select, omit, home_based, filter_consecutive)


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


# ---------------------------------------------------------------------------
# bin command
# ---------------------------------------------------------------------------


@cli.command(
    "bin",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
@click.argument("attributes", type=click.Path(exists=True))
@click.option(
    "--default",
    "-n",
    "n_bins",
    default=5,
    show_default=True,
    type=int,
    help="Default number of bins for all numeric columns.",
)
@click.option(
    "--method",
    "-m",
    default="quantile",
    show_default=True,
    type=click.Choice(["quantile", "uniform"]),
    help="Binning method: quantile (equal-frequency) or uniform (equal-width).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output CSV path (default: <input>_binned.csv alongside input).",
)
@click.option(
    "--select",
    "-s",
    multiple=True,
    help="Columns to bin (e.g. --select year --select month --select weight)",
    show_default=True,
)
@click.option(
    "--omit",
    "-x",
    multiple=True,
    help="Column to omit (e.g. --omit vehicles --omit hh_size)",
    show_default=True,
)
@click.pass_context
def bin_attributes(ctx, attributes, n_bins, method, output, select, omit):
    """Bin numeric columns of an attributes CSV.

    Per-column bin counts can be specified as --COLUMN N, e.g.:

        foundata bin attributes.csv --default 5 --age 10 --hh_income 3
    """
    per_col_bins = {}
    args = list(ctx.args)
    i = 0
    while i < len(args):
        if args[i].startswith("--") and i + 1 < len(args):
            col = args[i][2:]
            try:
                per_col_bins[col] = int(args[i + 1])
                i += 2
                continue
            except ValueError:
                click.echo(
                    f"Warning: ignoring --{col} (expected integer bin count)",
                    err=True,
                )
        i += 1

    df = pl.read_csv(attributes)
    binned = post_process.discretise_numeric(
        df,
        n_bins=n_bins,
        method=method,
        cols=select if select else None,
        exclude_cols=omit if omit else None,
        per_col_bins=per_col_bins or None,
    )

    out = Path(output) if output else _default_out(attributes, "_binned")
    out.parent.mkdir(parents=True, exist_ok=True)
    binned.write_csv(out)
    click.echo(f"Wrote {out}")


# ---------------------------------------------------------------------------
# filter group
# ---------------------------------------------------------------------------

_ATTR_OPT = click.option(
    "--attributes",
    "-a",
    type=click.Path(exists=True),
    default=None,
    help="Path to attributes CSV.",
)
_TRIPS_OPT = click.option(
    "--trips",
    "-t",
    required=True,
    type=click.Path(exists=True),
    help="Path to trips CSV.",
)
_OUT_DIR_OPT = click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output directory. Outputs use suffixed input filenames.",
)
_OUT_ATTR_OPT = click.option(
    "--output-attributes",
    "-oa",
    type=click.Path(),
    default=None,
    help="Explicit output path for attributes CSV (overrides -o).",
)
_OUT_TRIPS_OPT = click.option(
    "--output-trips",
    "-ot",
    type=click.Path(),
    default=None,
    help="Explicit output path for trips CSV (overrides -o).",
)


@cli.group("filter")
def filter_group():
    """Post-process attributes/trips CSVs with built-in filters."""


@filter_group.command("homebased")
@click.option(
    "--attributes",
    "-a",
    required=True,
    type=click.Path(exists=True),
    help="Path to attributes CSV.",
)
@_TRIPS_OPT
@_OUT_DIR_OPT
@_OUT_ATTR_OPT
@_OUT_TRIPS_OPT
def filter_homebased(
    attributes, trips, output, output_attributes, output_trips
):
    """Remove plans whose first or last activity is not home."""
    suffix = "_homebased"
    oa = _resolve_out(output_attributes, output, attributes, suffix)
    ot = _resolve_out(output_trips, output, trips, suffix)

    attrs_df = pl.read_csv(attributes)
    trips_df = pl.read_csv(trips)
    attrs_out, trips_out = flt.home_based(attrs_df, trips_df)

    if oa and attrs_out is not None:
        attrs_out.write_csv(oa)
        click.echo(f"Wrote {oa}")
    trips_out.write_csv(ot)
    click.echo(f"Wrote {ot}")


@filter_group.command("missing-acts-or-modes")
@_ATTR_OPT
@_TRIPS_OPT
@_OUT_DIR_OPT
@_OUT_ATTR_OPT
@_OUT_TRIPS_OPT
def filter_missing_acts_or_modes(
    attributes, trips, output, output_attributes, output_trips
):
    """Remove plans with any missing or unknown activities or modes."""
    suffix = "_clean_modes"
    oa = _resolve_out(output_attributes, output, attributes, suffix)
    ot = _resolve_out(output_trips, output, trips, suffix)

    attrs_df = pl.read_csv(attributes) if attributes else None
    trips_df = pl.read_csv(trips)
    attrs_out, trips_out = flt.missing_acts_or_modes(attrs_df, trips_df)

    if oa and attrs_out is not None:
        attrs_out.write_csv(oa)
        click.echo(f"Wrote {oa}")
    trips_out.write_csv(ot)
    click.echo(f"Wrote {ot}")


@filter_group.command("negative-trips")
@_ATTR_OPT
@_TRIPS_OPT
@_OUT_DIR_OPT
@_OUT_ATTR_OPT
@_OUT_TRIPS_OPT
def filter_negative_trips(
    attributes, trips, output, output_attributes, output_trips
):
    """Remove plans containing trips with negative durations (tst > tet)."""
    suffix = "_no_neg_trips"
    oa = _resolve_out(output_attributes, output, attributes, suffix)
    ot = _resolve_out(output_trips, output, trips, suffix)

    attrs_df = pl.read_csv(attributes) if attributes else None
    trips_df = pl.read_csv(trips)
    attrs_out, trips_out = flt.negative_trips(attrs_df, trips_df)

    if oa and attrs_out is not None:
        attrs_out.write_csv(oa)
        click.echo(f"Wrote {oa}")
    trips_out.write_csv(ot)
    click.echo(f"Wrote {ot}")


@filter_group.command("negative-activities")
@_ATTR_OPT
@_TRIPS_OPT
@_OUT_DIR_OPT
@_OUT_ATTR_OPT
@_OUT_TRIPS_OPT
def filter_negative_activities(
    attributes, trips, output, output_attributes, output_trips
):
    """Remove plans containing activities with negative durations."""
    suffix = "_no_neg_acts"
    oa = _resolve_out(output_attributes, output, attributes, suffix)
    ot = _resolve_out(output_trips, output, trips, suffix)

    attrs_df = pl.read_csv(attributes) if attributes else None
    trips_df = pl.read_csv(trips)
    attrs_out, trips_out = flt.negative_activities(attrs_df, trips_df)

    if oa and attrs_out is not None:
        attrs_out.write_csv(oa)
        click.echo(f"Wrote {oa}")
    trips_out.write_csv(ot)
    click.echo(f"Wrote {ot}")


@filter_group.command("null-times")
@_ATTR_OPT
@_TRIPS_OPT
@_OUT_DIR_OPT
@_OUT_ATTR_OPT
@_OUT_TRIPS_OPT
def filter_null_times(
    attributes, trips, output, output_attributes, output_trips
):
    """Remove plans containing trips with null start or end times."""
    suffix = "_no_null_times"
    oa = _resolve_out(output_attributes, output, attributes, suffix)
    ot = _resolve_out(output_trips, output, trips, suffix)

    attrs_df = pl.read_csv(attributes) if attributes else None
    trips_df = pl.read_csv(trips)
    attrs_out, trips_out = flt.null_times(attrs_df, trips_df)

    if oa and attrs_out is not None:
        attrs_out.write_csv(oa)
        click.echo(f"Wrote {oa}")
    trips_out.write_csv(ot)
    click.echo(f"Wrote {ot}")


@filter_group.command("time-consistent")
@_ATTR_OPT
@_TRIPS_OPT
@_OUT_DIR_OPT
@_OUT_ATTR_OPT
@_OUT_TRIPS_OPT
def filter_time_consistent(
    attributes, trips, output, output_attributes, output_trips
):
    """Apply all time-consistency filters (null pids, negative trips/activities, null times)."""
    suffix = "_time_consistent"
    oa = _resolve_out(output_attributes, output, attributes, suffix)
    ot = _resolve_out(output_trips, output, trips, suffix)

    attrs_df = pl.read_csv(attributes) if attributes else None
    trips_df = pl.read_csv(trips)
    attrs_out, trips_out = flt.time_consistent(attrs_df, trips_df)

    if oa and attrs_out is not None:
        attrs_out.write_csv(oa)
        click.echo(f"Wrote {oa}")
    trips_out.write_csv(ot)
    click.echo(f"Wrote {ot}")


@filter_group.command("consecutive-activities")
@_ATTR_OPT
@_TRIPS_OPT
@click.option(
    "--non-consecutive-types",
    "-n",
    multiple=True,
    default=["home", "work", "education"],
    show_default=True,
    help="Activity types that must not appear consecutively (repeatable).",
)
@_OUT_DIR_OPT
@_OUT_ATTR_OPT
@_OUT_TRIPS_OPT
def filter_consecutive_activities(
    attributes,
    trips,
    non_consecutive_types,
    output,
    output_attributes,
    output_trips,
):
    """Remove plans containing consecutive activities of the same non-consecutive type."""
    suffix = "_no_consec_acts"
    oa = _resolve_out(output_attributes, output, attributes, suffix)
    ot = _resolve_out(output_trips, output, trips, suffix)

    attrs_df = pl.read_csv(attributes) if attributes else None
    trips_df = pl.read_csv(trips)
    attrs_out, trips_out = flt.filter_consecutive_activities(
        attrs_df, trips_df, non_consecutive_types=list(non_consecutive_types)
    )

    if oa and attrs_out is not None:
        attrs_out.write_csv(oa)
        click.echo(f"Wrote {oa}")
    trips_out.write_csv(ot)
    click.echo(f"Wrote {ot}")


@filter_group.command("attributes")
@click.option(
    "--attributes",
    "-a",
    required=True,
    type=click.Path(exists=True),
    help="Path to attributes CSV.",
)
@_TRIPS_OPT
@click.option("--key", "-k", required=True, help="Column name to filter on.")
@click.option(
    "--value",
    "-v",
    multiple=True,
    required=True,
    help="Value(s) to keep (repeatable).",
)
@_OUT_DIR_OPT
@_OUT_ATTR_OPT
@_OUT_TRIPS_OPT
def filter_attributes(
    attributes, trips, key, value, output, output_attributes, output_trips
):
    """Filter attributes on a column value, then restrict trips to surviving persons.

    Example: -k employment -v employed -v ft-employed
    """
    suffix = "_filtered"
    oa = _resolve_out(output_attributes, output, attributes, suffix)
    ot = _resolve_out(output_trips, output, trips, suffix)

    attrs_df = pl.read_csv(attributes)
    trips_df = pl.read_csv(trips)

    total = len(attrs_df)
    col_dtype = attrs_df[key].dtype
    typed_values = pl.Series(list(value)).cast(col_dtype).to_list()
    attrs_out = attrs_df.filter(pl.col(key).is_in(typed_values))
    surviving_pids = attrs_out.select("pid")
    trips_out = trips_df.join(
        surviving_pids, on="pid", how="inner", maintain_order="left"
    )

    click.echo(
        f"Filtered attributes on {key}={list(value)}: kept {len(attrs_out)}/{total} persons"
    )

    if oa:
        attrs_out.write_csv(oa)
        click.echo(f"Wrote {oa}")
    trips_out.write_csv(ot)
    click.echo(f"Wrote {ot}")
