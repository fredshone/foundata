import functools
import inspect
import math
import random
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import polars as pl
import typer
from loguru import logger

from foundata import config_validator, gridjoin, log, post_process, verify
from foundata import filter as flt
from foundata import sample as sampling
from foundata.run import (
    SOURCES,
    SourceDataError,
    batch_source_dir,
    read_table,
    runner,
)

_DEFAULT_CONFIGS_ROOT = Path(__file__).parent.parent / "configs"
_DEFAULT_WORLD = gridjoin.DEFAULT_WORLD

Source = Enum("Source", {name: name for name in SOURCES}, type=str)

OPEN_SOURCES = (
    "nhts",
    "cmap",
    "vista",
    "qhts",
    "ktdb",
    "mexico",
    "brazil",
    "argentina",
    "uruguay",
    "chile",
    "india",
    "mongolia",
)

app = typer.Typer(
    help="foundata — household travel survey aggregation toolkit.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


class LogLevel(str, Enum):
    debug = "DEBUG"
    info = "INFO"
    warning = "WARNING"
    error = "ERROR"


LogLevelOpt = Annotated[
    LogLevel,
    typer.Option(
        "--log-level",
        "-l",
        case_sensitive=False,
        help="Minimum level of log messages to show.",
    ),
]


def _with_log_level(fn):
    """Add a --log-level option to command *fn* that configures logging."""

    @functools.wraps(fn)
    def wrapper(*args, log_level: LogLevel = LogLevel.info, **kwargs):
        log.setup(log_level.value)
        try:
            return fn(*args, **kwargs)
        except SourceDataError as e:
            # missing data is the user's to fix, so no traceback
            for line in str(e).splitlines():
                logger.error(line)
            raise typer.Exit(1)

    sig = inspect.signature(fn)
    wrapper.__signature__ = sig.replace(
        parameters=[
            *sig.parameters.values(),
            inspect.Parameter(
                "log_level",
                inspect.Parameter.KEYWORD_ONLY,
                default=LogLevel.info,
                annotation=LogLevelOpt,
            ),
        ]
    )
    return wrapper


def _split_commas(values: Optional[list[str]]) -> list[str]:
    """Flatten repeated and comma-separated option values."""
    return [v for value in values or [] for v in value.split(",") if v]


class MultiSourceCommand(typer.core.TyperCommand):
    """Let --select/--omit take several space-separated sources.

    Click options take one value each, so `-s nhts ktdb` is rewritten to
    `-s nhts -s ktdb` before parsing. Only known source names are absorbed,
    so a following positional argument (e.g. DATA_ROOT) is left alone.
    """

    MULTI_FLAGS = {"-s", "--select", "-x", "--omit"}

    def parse_args(self, ctx, args):
        expanded, flag, has_value = [], None, False
        for arg in args:
            if has_value and all(v in SOURCES for v in arg.split(",") if v):
                expanded += [flag, arg]
                continue
            expanded.append(arg)
            if flag and not has_value:
                has_value = True  # the flag's own (first) value
                continue
            flag = arg if arg in self.MULTI_FLAGS else None
            has_value = False
        return super().parse_args(ctx, expanded)


def _read(path: str | Path) -> pl.DataFrame:
    """Read a table from a .parquet or .csv file."""
    if Path(path).suffix == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path)


def _write(df: pl.DataFrame, path: str | Path):
    """Write *df* to *path*, as parquet if it ends .parquet, else csv."""
    if Path(path).suffix == ".parquet":
        df.write_parquet(path)
    else:
        df.write_csv(path)
    logger.info(f"Wrote {path}")


def _default_out(input_path: str | Path, suffix: str) -> Path:
    """Derive a default output path by appending *suffix* to the input stem."""
    p = Path(input_path)
    return p.parent / f"{p.stem}{suffix}{p.suffix}"


def _resolve_out(
    explicit: Optional[Path],
    out_dir: Optional[Path],
    input_path: Optional[Path],
    optional_suffix: str,
) -> Optional[Path]:
    if explicit:
        path = Path(explicit)
    elif input_path is None:
        return None
    else:
        stem_path = _default_out(input_path, "")
        path = Path(out_dir) / stem_path.name if out_dir else stem_path
        # check if file exists, if so, add suffix to avoid overwriting
        if path.exists():
            logger.warning(
                f"output path {path} already exists, adding suffix '{optional_suffix}' to avoid overwriting."
            )
            stem_path = _default_out(input_path, optional_suffix)
            path = Path(out_dir) / stem_path.name if out_dir else stem_path
            if path.exists():
                logger.warning(
                    f"output path {path} already exists, overwriting."
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# pipeline commands
# ---------------------------------------------------------------------------

HomeBasedOpt = Annotated[
    bool,
    typer.Option(
        "--home-based/--any-based",
        "-hb/-ab",
        help="Whether to only include home-based trips (i.e. those with 'home' as the origin or destination activity).",
    ),
]
FilterConsecutiveOpt = Annotated[
    bool,
    typer.Option(
        "--filter-consecutive/--any-consecutive",
        "-fc/-ac",
        help="Whether to filter out consecutive home, work and education activities.",
    ),
]
DiagnosticsOpt = Annotated[
    bool,
    typer.Option(
        "--diagnostics",
        "-d",
        help="Also produce the anomaly-detection reports and supplementary activity heatmaps.",
    ),
]
KeepIdsOpt = Annotated[
    bool,
    typer.Option(
        "--keep-ids",
        help=(
            "Also write original_ids, mapping each pid and hid back to "
            "the survey's own ids. It re-identifies records, so keep it private."
        ),
    ),
]
CsvOpt = Annotated[
    bool,
    typer.Option(
        "--csv",
        help="Write output tables as CSV instead of Parquet.",
    ),
]
GridOpt = Annotated[
    bool,
    typer.Option(
        "--grid/--no-grid",
        help="Join WorldTensor data to locations and households (wt_ columns).",
    ),
]
WorldOpt = Annotated[
    Path,
    typer.Option(
        "--world",
        help="WorldTensor data root, holding worldtensor_*/ folders.",
    ),
]
GridConfigOpt = Annotated[
    Path,
    typer.Option(
        "--grid-config",
        help="WorldTensor variables to join and their weights.",
        exists=True,
        dir_okay=False,
    ),
]


@app.command("run")
def run(
    source: Annotated[Source, typer.Argument(help="Source to process.")],
    source_dir: Annotated[
        Path,
        typer.Argument(
            help="Data directory for this source, e.g. ~/Data/foundata/LTDS",
            exists=True,
            file_okay=False,
        ),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Directory where tables and PNGs are written [default: output/<SOURCE>]",
            show_default=False,
        ),
    ] = None,
    home_based: HomeBasedOpt = True,
    filter_consecutive: FilterConsecutiveOpt = True,
    diagnostics: DiagnosticsOpt = False,
    keep_ids: KeepIdsOpt = False,
    csv: CsvOpt = False,
    grid: GridOpt = True,
    world: WorldOpt = _DEFAULT_WORLD,
    grid_config: GridConfigOpt = gridjoin.worldtensor.CONFIG,
):
    """Run the pipeline for a single source, e.g. `foundata run ltds ~/Data/foundata/LTDS`."""
    output = output or Path("output") / source.value
    runner(
        {source.value: source_dir},
        output,
        home_based,
        filter_consecutive,
        diagnostics,
        keep_ids,
        csv=csv,
        world=world if grid else None,
        grid_config=grid_config,
    )


@app.command("batch", cls=MultiSourceCommand)
def batch(
    data_root: Annotated[
        Path,
        typer.Argument(
            help="Base data directory containing one folder per source (LTDS/, NTS/, ...), e.g. ~/Data/foundata",
            exists=True,
            file_okay=False,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Directory where tables and PNGs are written.",
        ),
    ] = Path("output"),
    select: Annotated[
        Optional[list[str]],
        typer.Option(
            "--select",
            "-s",
            metavar="SOURCE...",
            help="Sources to process (e.g. -s nhts ktdb, or -s nhts,ktdb). Defaults to all.",
        ),
    ] = None,
    omit: Annotated[
        Optional[list[str]],
        typer.Option(
            "--omit",
            "-x",
            metavar="SOURCE...",
            help="Sources to omit (e.g. -x nhts ktdb, or -x nhts,ktdb).",
        ),
    ] = None,
    open_only: Annotated[
        bool,
        typer.Option(
            "--open",
            help=(
                "Run only the open-data sources "
                f"({', '.join(OPEN_SOURCES)}). Overrides --select and --omit."
            ),
        ),
    ] = False,
    home_based: HomeBasedOpt = True,
    filter_consecutive: FilterConsecutiveOpt = True,
    diagnostics: DiagnosticsOpt = False,
    keep_ids: KeepIdsOpt = False,
    csv: CsvOpt = False,
    grid: GridOpt = True,
    world: WorldOpt = _DEFAULT_WORLD,
    grid_config: GridConfigOpt = gridjoin.worldtensor.CONFIG,
):
    """Run the pipeline end-to-end over many sources and combine them."""
    select, omit = _split_commas(select), _split_commas(omit)
    unknown = [s for s in select + omit if s not in SOURCES]
    if unknown:
        raise typer.BadParameter(
            f"Unknown source(s) {', '.join(unknown)}, expected any of "
            f"{', '.join(SOURCES)}"
        )
    if open_only:
        if select or omit:
            logger.warning(
                "--open overrides --select and --omit; ignoring them."
            )
        sources = list(OPEN_SOURCES)
    elif select and omit:
        logger.error("Cannot use both --select and --omit options.")
        raise typer.Exit(1)
    elif select:
        sources = [s for s in SOURCES if s in select]
    else:
        sources = [s for s in SOURCES if s not in omit]
    source_dirs = {s: batch_source_dir(data_root, s) for s in sources}
    runner(
        source_dirs,
        output,
        home_based,
        filter_consecutive,
        diagnostics,
        keep_ids,
        csv=csv,
        world=world if grid else None,
        grid_config=grid_config,
    )


@app.command("location-summary")
def location_summary(
    output: Annotated[
        Path,
        typer.Argument(
            help="A batch output directory (attributes, trips and "
            "locations.parquet)"
        ),
    ],
):
    """Print the per-source location coverage table (as in the README)."""
    from foundata import location

    attributes = read_table(
        output, "attributes", columns=["pid", "hid", "source", "hh_location"]
    )
    trips = read_table(output, "trips", columns=["pid", "ocode", "dcode"])
    locations = pl.read_parquet(output / "locations.parquet")
    print(
        location.summary_markdown(
            location.summary(attributes, trips, locations)
        )
    )


@app.command("validate-config")
def validate_config(
    source: Annotated[
        Optional[str], typer.Argument(help="Survey name, e.g. nhts.")
    ] = None,
    all_sources: Annotated[
        bool, typer.Option("--all", help="Validate all sources.")
    ] = False,
    configs_root: Annotated[
        Optional[Path],
        typer.Option(
            exists=True,
            file_okay=False,
            help="Path to configs directory (default: configs/ in project root).",
        ),
    ] = None,
):
    """Validate YAML config(s) against the template schema.

    SOURCE is the survey name (e.g. nhts). Pass --all to validate every source.
    """
    root = configs_root or _DEFAULT_CONFIGS_ROOT

    if all_sources:
        ok = config_validator.validate_all_sources(root)
    elif source:
        ok = config_validator.validate_source(source, root)
    else:
        logger.error("Provide a SOURCE argument or pass --all.")
        raise typer.Exit(1)

    if not ok:
        raise typer.Exit(1)


@app.command("validate-table")
def validate_table(
    attributes: Annotated[Path, typer.Argument(exists=True)],
    trips: Annotated[Path, typer.Argument(exists=True)],
):
    """Validate pipeline output tables (.parquet or .csv) against the template schema.

    ATTRIBUTES and TRIPS are paths to the output files produced by the
    pipeline (one row per person and one row per trip, respectively).
    """
    attributes = _read(attributes)
    trips = _read(trips)
    ok = verify.columns(attributes, trips)
    if not ok:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# bin command
# ---------------------------------------------------------------------------


class BinMethod(str, Enum):
    quantile = "quantile"
    uniform = "uniform"


@app.command(
    "bin",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def bin_attributes(
    ctx: typer.Context,
    attributes: Annotated[Path, typer.Argument(exists=True)],
    n_bins: Annotated[
        int,
        typer.Option(
            "--default",
            "-n",
            help="Default number of bins for all numeric columns.",
        ),
    ] = 5,
    method: Annotated[
        BinMethod,
        typer.Option(
            "--method",
            "-m",
            help="Binning method: quantile (equal-frequency) or uniform (equal-width).",
        ),
    ] = BinMethod.quantile,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output path, .parquet or .csv (default: <input>_binned alongside input, in its format).",
        ),
    ] = None,
    select: Annotated[
        Optional[list[str]],
        typer.Option(
            "--select",
            "-s",
            help="Columns to bin, repeatable or comma-separated (e.g. --select year,month,weight)",
        ),
    ] = None,
    omit: Annotated[
        Optional[list[str]],
        typer.Option(
            "--omit",
            "-x",
            help="Columns to omit, repeatable or comma-separated (e.g. --omit vehicles,hh_size)",
        ),
    ] = None,
):
    """Bin numeric columns of an attributes table (.parquet or .csv).

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
                logger.warning(f"ignoring --{col} (expected integer bin count)")
        i += 1

    df = _read(attributes)
    binned = post_process.discretise_numeric(
        df,
        n_bins=n_bins,
        method=method.value,
        cols=_split_commas(select) or None,
        exclude_cols=_split_commas(omit) or None,
        per_col_bins=per_col_bins or None,
    )

    out = output or _default_out(attributes, "_binned")
    out.parent.mkdir(parents=True, exist_ok=True)
    _write(binned, out)


@app.command("fill-unknown")
def fill_unknown_cmd(
    attributes: Annotated[Path, typer.Argument(exists=True)],
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output path, .parquet or .csv (default: <input>_filled alongside input, in its format).",
        ),
    ] = None,
):
    """Fill null/missing values in an attributes table with 'unknown'.

    Reports the percentage of each column filled and warns when a column
    is entirely unknown or appears to be numeric.
    """
    df = _read(attributes)
    filled, stats = post_process.fill_unknown(df)

    if not stats:
        logger.info("No missing values found.")
    else:
        for col, info in stats.items():
            logger.info(f"  {col}: {info['pct']:.1f}% filled with unknown")
            if info["all_unknown"]:
                logger.warning(f"'{col}' is entirely unknown")
            if info["appears_numeric"]:
                logger.warning(f"'{col}' appears numeric")

    out = output or _default_out(attributes, "_filled")
    out.parent.mkdir(parents=True, exist_ok=True)
    _write(filled, out)


# ---------------------------------------------------------------------------
# filter group
# ---------------------------------------------------------------------------

filter_app = typer.Typer(
    help="Post-process attributes/trips tables (.parquet or .csv) with built-in filters.",
    no_args_is_help=True,
)
app.add_typer(filter_app, name="filter")

AttrOpt = Annotated[
    Optional[Path],
    typer.Option(
        "--attributes", "-a", exists=True, help="Path to attributes table."
    ),
]
RequiredAttrOpt = Annotated[
    Path,
    typer.Option(
        "--attributes", "-a", exists=True, help="Path to attributes table."
    ),
]
TripsOpt = Annotated[
    Path,
    typer.Option("--trips", "-t", exists=True, help="Path to trips table."),
]
OutDirOpt = Annotated[
    Optional[Path],
    typer.Option(
        "--output",
        "-o",
        help="Output directory. Outputs use input filenames, suffixed only to avoid overwriting an existing file.",
    ),
]
OutAttrOpt = Annotated[
    Optional[Path],
    typer.Option(
        "--output-attributes",
        "-oa",
        help="Explicit output path for attributes, .parquet or .csv (overrides -o).",
    ),
]
OutTripsOpt = Annotated[
    Optional[Path],
    typer.Option(
        "--output-trips",
        "-ot",
        help="Explicit output path for trips, .parquet or .csv (overrides -o).",
    ),
]


def _apply_filter(
    filter_fn,
    suffix,
    attributes,
    trips,
    output,
    output_attributes,
    output_trips,
):
    """Read inputs, apply *filter_fn* and write the outputs."""
    oa = _resolve_out(output_attributes, output, attributes, suffix)
    ot = _resolve_out(output_trips, output, trips, suffix)

    attrs_df = _read(attributes) if attributes else None
    trips_df = _read(trips)
    attrs_out, trips_out = filter_fn(attrs_df, trips_df)

    if oa and attrs_out is not None:
        _write(attrs_out, oa)
    _write(trips_out, ot)


@filter_app.command("homebased")
def filter_homebased(
    attributes: RequiredAttrOpt,
    trips: TripsOpt,
    output: OutDirOpt = None,
    output_attributes: OutAttrOpt = None,
    output_trips: OutTripsOpt = None,
):
    """Remove plans whose first or last activity is not home."""
    _apply_filter(
        flt.home_based,
        "_homebased",
        attributes,
        trips,
        output,
        output_attributes,
        output_trips,
    )


@filter_app.command("missing-acts")
def filter_missing_acts(
    trips: TripsOpt,
    attributes: AttrOpt = None,
    output: OutDirOpt = None,
    output_attributes: OutAttrOpt = None,
    output_trips: OutTripsOpt = None,
):
    """Remove plans with any missing or unknown activities."""
    _apply_filter(
        flt.missing_acts,
        "_clean_acts",
        attributes,
        trips,
        output,
        output_attributes,
        output_trips,
    )


@filter_app.command("negative-trips")
def filter_negative_trips(
    trips: TripsOpt,
    attributes: AttrOpt = None,
    output: OutDirOpt = None,
    output_attributes: OutAttrOpt = None,
    output_trips: OutTripsOpt = None,
):
    """Remove plans containing trips with negative durations (tst > tet)."""
    _apply_filter(
        flt.negative_trips,
        "_no_neg_trips",
        attributes,
        trips,
        output,
        output_attributes,
        output_trips,
    )


@filter_app.command("negative-activities")
def filter_negative_activities(
    trips: TripsOpt,
    attributes: AttrOpt = None,
    output: OutDirOpt = None,
    output_attributes: OutAttrOpt = None,
    output_trips: OutTripsOpt = None,
):
    """Remove plans containing activities with negative durations."""
    _apply_filter(
        flt.negative_activities,
        "_no_neg_acts",
        attributes,
        trips,
        output,
        output_attributes,
        output_trips,
    )


@filter_app.command("null-times")
def filter_null_times(
    trips: TripsOpt,
    attributes: AttrOpt = None,
    output: OutDirOpt = None,
    output_attributes: OutAttrOpt = None,
    output_trips: OutTripsOpt = None,
):
    """Remove plans containing trips with null start or end times."""
    _apply_filter(
        flt.null_times,
        "_no_null_times",
        attributes,
        trips,
        output,
        output_attributes,
        output_trips,
    )


@filter_app.command("time-consistent")
def filter_time_consistent(
    trips: TripsOpt,
    attributes: AttrOpt = None,
    output: OutDirOpt = None,
    output_attributes: OutAttrOpt = None,
    output_trips: OutTripsOpt = None,
):
    """Apply all time-consistency filters (null pids, negative trips/activities, null times)."""
    _apply_filter(
        flt.time_consistent,
        "_time_consistent",
        attributes,
        trips,
        output,
        output_attributes,
        output_trips,
    )


@filter_app.command("consecutive-activities")
def filter_consecutive_activities(
    trips: TripsOpt,
    attributes: AttrOpt = None,
    non_consecutive_types: Annotated[
        list[str],
        typer.Option(
            "--non-consecutive-types",
            "-n",
            help="Activity types that must not appear consecutively (repeatable).",
        ),
    ] = ["home", "work", "education"],
    output: OutDirOpt = None,
    output_attributes: OutAttrOpt = None,
    output_trips: OutTripsOpt = None,
):
    """Remove plans containing consecutive activities of the same non-consecutive type."""
    _apply_filter(
        lambda a, t: flt.filter_consecutive_activities(
            a, t, non_consecutive_types=list(non_consecutive_types)
        ),
        "_no_consec_acts",
        attributes,
        trips,
        output,
        output_attributes,
        output_trips,
    )


@filter_app.command("attributes")
def filter_attributes(
    attributes: RequiredAttrOpt,
    trips: TripsOpt,
    key: Annotated[
        str, typer.Option("--key", "-k", help="Column name to filter on.")
    ],
    value: Annotated[
        list[str],
        typer.Option("--value", "-v", help="Value(s) to keep (repeatable)."),
    ],
    output: OutDirOpt = None,
    output_attributes: OutAttrOpt = None,
    output_trips: OutTripsOpt = None,
):
    """Filter attributes on a column value, then restrict trips to surviving persons.

    Example: -k employment -v employed -v ft-employed
    """
    suffix = "_filtered"
    oa = _resolve_out(output_attributes, output, attributes, suffix)
    ot = _resolve_out(output_trips, output, trips, suffix)

    attrs_df = _read(attributes)
    trips_df = _read(trips)

    total = len(attrs_df)
    col_dtype = attrs_df[key].dtype
    typed_values = pl.Series(list(value)).cast(col_dtype).to_list()
    attrs_out = attrs_df.filter(pl.col(key).is_in(typed_values))
    surviving_pids = attrs_out.select("pid")
    trips_out = trips_df.join(
        surviving_pids, on="pid", how="inner", maintain_order="left"
    )

    logger.info(
        f"Filtered attributes on {key}={list(value)}: kept {len(attrs_out)}/{total} persons"
    )

    if oa:
        _write(attrs_out, oa)
    _write(trips_out, ot)


# ---------------------------------------------------------------------------
# split command
# ---------------------------------------------------------------------------


@app.command("split")
def split_cmd(
    inputs: Annotated[list[Path], typer.Argument(exists=True)],
    group: Annotated[
        str,
        typer.Option(
            "--group",
            "-g",
            help="Column to group by (persons stay intact across splits).",
        ),
    ] = "pid",
    split_pct: Annotated[
        float,
        typer.Option(
            "--split",
            "-s",
            help="Test split size as a percentage (e.g. 20 = 20%).",
        ),
    ] = 20,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output directory (default: parent dir of first input).",
        ),
    ] = None,
    seed: Annotated[
        int, typer.Option(help="Random seed for reproducibility.")
    ] = 42,
):
    """Randomly split tables (.parquet or .csv) into train/test while keeping group entities intact."""
    # 1. Read
    dfs = [(path, _read(path)) for path in inputs]
    # 2. Validate group column present
    for path, df in dfs:
        if group not in df.columns:
            raise typer.BadParameter(
                f"Group column '{group}' not found in {path}"
            )
    # 3. Check consistency
    group_sets = [
        (path, set(df[group].drop_nulls().to_list())) for path, df in dfs
    ]
    reference_path, reference_set = group_sets[0]
    inconsistent = [(p, s) for p, s in group_sets[1:] if s != reference_set]
    if inconsistent:
        logger.error(f"Group values for '{group}' differ across inputs:")
        for p, s in inconsistent:
            only_ref = reference_set - s
            only_other = s - reference_set
            logger.error(f"  {reference_path} vs {p}:")
            if only_ref:
                logger.error(
                    f"    Only in {reference_path}: {sorted(only_ref)[:10]}{'...' if len(only_ref) > 10 else ''}"
                )
            if only_other:
                logger.error(
                    f"    Only in {p}: {sorted(only_other)[:10]}{'...' if len(only_other) > 10 else ''}"
                )
        raise typer.Exit(1)
    # 4. Sample test IDs
    all_ids = sorted(reference_set)
    rng = random.Random(seed)
    rng.shuffle(all_ids)
    n_test = math.ceil(len(all_ids) * split_pct / 100)
    test_ids = set(all_ids[:n_test])
    train_ids = set(all_ids[n_test:])
    logger.info(
        f"Split on '{group}': {len(train_ids)} train / {len(test_ids)} test ({split_pct:.0f}%)"
    )
    # 5. Write outputs
    out_dir = output or Path(inputs[0]).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, df in dfs:
        path = Path(path)
        train_df = df.filter(pl.col(group).is_in(train_ids))
        test_df = df.filter(pl.col(group).is_in(test_ids))
        train_path = out_dir / f"{path.stem}_train{path.suffix}"
        test_path = out_dir / f"{path.stem}_test{path.suffix}"
        _write(train_df, train_path)
        _write(test_df, test_path)
        logger.info(
            f"  {path.name:30s} → {len(train_df):>7} train / {len(test_df):>7} test rows"
        )
    logger.info(f"Wrote outputs to {out_dir}")


# ---------------------------------------------------------------------------
# sample command
# ---------------------------------------------------------------------------


@app.command("sample")
def sample_cmd(
    activity_schedules: Annotated[Path, typer.Argument(exists=True)],
    size: Annotated[
        int,
        typer.Option(
            "--size",
            "-n",
            help="Number of persons to sample (split equally across strata).",
        ),
    ],
    attributes: AttrOpt = None,
    trips: Annotated[
        Optional[Path],
        typer.Option("--trips", "-t", exists=True, help="Path to trips table."),
    ] = None,
    strata: Annotated[
        Optional[list[str]],
        typer.Option(
            "--strata",
            "-s",
            help=(
                "Attribute columns to stratify on, repeatable or "
                "comma-separated (e.g. --strata age,hh_income). "
                "Requires --attributes."
            ),
        ),
    ] = None,
    output: OutDirOpt = None,
    output_schedules: Annotated[
        Optional[Path],
        typer.Option(
            "--output-schedules",
            "-os",
            help="Explicit output path for activity schedules, .parquet or .csv (overrides -o).",
        ),
    ] = None,
    output_attributes: OutAttrOpt = None,
    output_trips: OutTripsOpt = None,
    seed: Annotated[
        int, typer.Option(help="Random seed for reproducibility.")
    ] = 42,
    keep_index: Annotated[
        bool,
        typer.Option(
            "--keep-index",
            help=(
                "Keep the original pids and hids instead of reindexing to "
                "integers. Repeat draws of a person get pids suffixed _1, _2, ..."
            ),
        ),
    ] = False,
):
    """Randomly sample persons from activity schedules (.parquet or .csv),
    and optionally their attributes and trips.

    Sampling is by person (pid), not by household: persons are drawn
    independently, so households are not kept whole.

    With --strata, each stratum gets an equal number of persons, sampled
    without replacement. Strata smaller than their share are oversampled
    (with a warning). Numeric strata with more than 5 values are binned
    into 5 categories. Sampled persons are given integer pids 0..n-1, and
    households integer hids, unless --keep-index is passed. Persons sharing
    a household still share one, but repeat draws of a household's persons
    form a new household.

    Example: foundata sample activity_schedules.parquet -n 1000 -a attributes.parquet -s age,sex
    """
    strata = _split_commas(strata)
    if strata and attributes is None:
        raise typer.BadParameter("--strata requires --attributes")

    suffix = "_sample"
    os_ = _resolve_out(output_schedules, output, activity_schedules, suffix)
    oa = _resolve_out(output_attributes, output, attributes, suffix)
    ot = _resolve_out(output_trips, output, trips, suffix)

    schedules_df = _read(activity_schedules)
    attrs_df = _read(attributes) if attributes else None
    try:
        sample = sampling.sample_pids(
            schedules_df["pid"].unique().to_list(),
            size,
            attributes=attrs_df,
            strata=strata,
            seed=seed,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e))
    logger.info(f"Sampled {sample.height} persons")
    trips_df = _read(trips) if trips else None
    if not keep_index:
        # take households from the first table that has them
        hids = next(
            (
                df
                for df in (attrs_df, schedules_df, trips_df)
                if df is not None and "hid" in df.columns
            ),
            None,
        )
        sample = sampling.reindex(sample, hids)

    _write(sampling.apply_sample(schedules_df, sample), os_)
    if attrs_df is not None:
        _write(sampling.apply_sample(attrs_df, sample), oa)
    if trips_df is not None:
        _write(sampling.apply_sample(trips_df, sample), ot)


for _typer in (app, filter_app):
    for _command in _typer.registered_commands:
        _command.callback = _with_log_level(_command.callback)
