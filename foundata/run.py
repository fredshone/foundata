#!/usr/bin/env python3
"""Run the full foundata pipeline and save outputs."""

import time
from pathlib import Path

import polars as pl
from loguru import logger

from foundata import (
    anomaly,
    argentina,
    brazil,
    chile,
    cmap,
    filter,
    fix,
    france,
    gridjoin,
    india,
    ktdb,
    location,
    ltds,
    mexico,
    mon,
    mongolia,
    nhts,
    nts,
    odin,
    ovin,
    plots,
    post_process,
    qhts,
    table_logger,
    tables,
    uruguay,
    utils,
    verify,
    vista,
)

CONFIGS_ROOT = Path(__file__).resolve().parent.parent / "configs"


def log_markdown_table(title: str, table: str):
    logger.info(f"\n### {title}\n\n{table}\n")


def process_source(attributes, trips, source_name):
    n_plans_in = trips["pid"].n_unique()

    attributes = utils.compute_avg_speed(attributes, trips)
    attributes = utils.split_employment_type(attributes)
    attributes = utils.correct_child_employment(attributes)
    attributes, trips = filter.time_consistent(attributes, trips, on="pid")
    utils.check_overlap(
        attributes, trips, on="pid", lhs_name="attributes", rhs_name="trips"
    )
    attributes, trips = filter.missing_acts(attributes, trips)

    attributes, trips = fix.missing_columns(attributes, trips)
    table_logger.record(
        "verify", "ok" if verify.columns(attributes, trips) else "FAIL"
    )
    attributes, trips = filter.columns(attributes, trips)
    attributes, trips = fix.fix_types(attributes, trips)
    attributes = fix.unknown_to_null(attributes)
    attributes = utils.norm_weights(attributes)
    attributes, trips = filter.trips_on_attribute_pids(attributes, trips)
    attributes, trips = filter.activity_consistency(attributes, trips)
    trips = filter.trips_on_endings(trips, time_limit=1440)
    attributes, trips = filter.feasible_trips(attributes, trips)

    attributes = attributes.sort("pid")
    trips = trips.sort(["pid", "seq"])

    n_plans = trips["pid"].n_unique()
    table_logger.record("plans", n_plans)
    table_logger.record("retained", table_logger.pct(n_plans, n_plans_in))
    logger.info(
        f"Processed {source_name}: {len(attributes)} persons, {n_plans} plans, "
        f"{len(trips)} trips ({table_logger.pct(n_plans, n_plans_in)} of plans retained)"
    )
    return attributes, trips


def _configs(source: str, *names: str) -> list[dict]:
    """Load ``configs/<source>/<name>_dictionary.yaml`` for each name."""
    return [
        utils.load_yaml_config(
            CONFIGS_ROOT / source / f"{name}_dictionary.yaml"
        )
        for name in names
    ]


def _load_ktdb(source_dir: Path):
    # hh_dictionary.yaml only declares the home location, which
    # location.LocationFinder reads
    person_config, trips_config = _configs("ktdb", "person", "trip")
    # 2021 personal survey, plus the 2016 and 2010 household (weekday)
    # and weekend surveys
    return ktdb.load_years(
        data_root=source_dir,
        person_config=person_config,
        trips_config=trips_config,
    )


def _load_ltds(source_dir: Path):
    (
        hh_config,
        person_config,
        person_data_config,
        trips_config,
        stages_config,
    ) = _configs("ltds", "hh", "person", "person_data", "trip", "stage")
    return ltds.load_years(
        years=["LTDS2425", "LTDS2324", "LTDS2223", "LTDS1920"],
        data_root=source_dir,
        hh_config=hh_config,
        person_config=person_config,
        person_data_config=person_data_config,
        trips_config=trips_config,
        stages_config=stages_config,
    )


def _load_vista(source_dir: Path):
    hh_config, person_config, trips_config = _configs(
        "vista", "hh", "person", "trip"
    )
    return vista.load_years(
        years=["2012-2020", "2022-2023", "2023-2024"],
        data_root=source_dir,
        hh_config=hh_config,
        person_config=person_config,
        trips_config=trips_config,
    )


def _load_qhts(source_dir: Path):
    hh_config, person_config, trips_config = _configs(
        "qhts", "hh", "person", "trip"
    )
    zone_mapping = qhts.load_zone_mapping(
        CONFIGS_ROOT / "qhts" / "sa1-correspondence-file.csv"
    )
    return qhts.load_years(
        data_root=source_dir,
        years=["2019-22", "2022-25"],
        hh_config=hh_config,
        person_config=person_config,
        trips_config=trips_config,
        zones_mapping=zone_mapping,
    )


def _load_cmap(source_dir: Path):
    hh_config, person_config, trips_config = _configs(
        "cmap", "hh", "person", "trip"
    )
    return cmap.load(
        data_root=source_dir,
        configs_root=CONFIGS_ROOT,
        hh_config=hh_config,
        person_config=person_config,
        trips_config=trips_config,
    )


def _load_nhts(source_dir: Path):
    hh_config, person_config, trips_config = _configs(
        "nhts", "hh", "person", "trip"
    )
    return nhts.load(
        data_root=source_dir,
        years=[2022, 2017, 2009, 2001],
        hh_config=hh_config,
        person_config=person_config,
        trips_config=trips_config,
    )


def _load_nts(source_dir: Path):
    hh_config, person_config, trips_config, stages_config, days_config = (
        _configs("nts", "hh", "person", "trip", "stage", "day")
    )
    return nts.load(
        data_root=source_dir,
        hh_config=hh_config,
        person_config=person_config,
        trips_config=trips_config,
        stages_config=stages_config,
        days_config=days_config,
    )


def _load_odin(source_dir: Path):
    hh_config, person_config, trips_config = _configs(
        "odin", "hh", "person", "trip"
    )
    return odin.load(
        data_root=source_dir,
        configs_root=CONFIGS_ROOT,
        years=[2018, 2019, 2020, 2021, 2022, 2023, 2024],
        hh_config=hh_config,
        person_config=person_config,
        trips_config=trips_config,
    )


def _load_ovin(source_dir: Path):
    hh_config, person_config, trips_config = _configs(
        "ovin", "hh", "person", "trip"
    )
    return ovin.load(
        data_root=source_dir,
        years=list(ovin.DATA_FILES),
        hh_config=hh_config,
        person_config=person_config,
        trips_config=trips_config,
    )


def _load_mon(source_dir: Path):
    hh_config, person_config, trips_config = _configs(
        "mon", "hh", "person", "trip"
    )
    return mon.load(
        data_root=source_dir,
        years=list(mon.DATA_FILES),
        hh_config=hh_config,
        person_config=person_config,
        trips_config=trips_config,
    )


def _load_india(source_dir: Path):
    # time use survey
    hh_config, person_config, trips_config = _configs(
        "india", "hh", "person", "trip"
    )
    return india.load_years(
        data_root=source_dir,
        years=[2019, 2024],
        hh_config=hh_config,
        person_config=person_config,
        trips_config=trips_config,
    )


def _load_brazil(source_dir: Path):
    # sao paulo
    hh_config, person_config, trips_config = _configs(
        "brazil", "hh", "person", "trip"
    )
    return brazil.load(
        data_root=source_dir,
        hh_config=hh_config,
        person_config=person_config,
        trips_config=trips_config,
    )


def _load_mexico(source_dir: Path):
    # mexico city
    hh_config, person_config, trips_config = _configs(
        "mexico", "hh", "person", "trip"
    )
    return mexico.load(
        data_root=source_dir,
        hh_config=hh_config,
        person_config=person_config,
        trips_config=trips_config,
    )


def _load_argentina(source_dir: Path):
    # household mobility surveys
    hh_config, person_config, trips_config = _configs(
        "argentina", "hh", "person", "trip"
    )
    return argentina.load(
        data_root=source_dir,
        hh_config=hh_config,
        person_config=person_config,
        trips_config=trips_config,
    )


def _load_uruguay(source_dir: Path):
    # montevideo
    hh_config, person_config, trips_config = _configs(
        "uruguay", "hh", "person", "trip"
    )
    return uruguay.load(
        data_root=source_dir,
        hh_config=hh_config,
        person_config=person_config,
        trips_config=trips_config,
    )


def _load_mongolia(source_dir: Path):
    # time use survey
    hh_config, person_config, trips_config = _configs(
        "mongolia", "hh", "person", "trip"
    )
    return mongolia.load_years(
        data_root=source_dir,
        years=(2023, 2019, 2011),
        hh_config=hh_config,
        person_config=person_config,
        trips_config=trips_config,
    )


def _load_chile(source_dir: Path):
    # santiago EOD 2012
    hh_config, person_config, trips_config = _configs(
        "chile", "hh", "person", "trip"
    )
    return chile.load(
        data_root=source_dir,
        hh_config=hh_config,
        person_config=person_config,
        trips_config=trips_config,
    )


def _load_france(source_dir: Path):
    # mobisurvstd-standardised surveys: every survey folder under FRANCE/
    # (CEREMA city surveys, EMP 2019, EGT 2020)
    hh_config, person_config, trips_config = _configs(
        "france", "hh", "person", "trip"
    )
    return france.load(
        data_root=source_dir,
        hh_config=hh_config,
        person_config=person_config,
        trips_config=trips_config,
    )


# source name -> raw loader, in processing order. New sources go here. Each
# loader takes the source's own data directory; in a batch run that is
# <data_root>/<NAME>, e.g. ~/Data/foundata/LTDS.
SOURCES = {
    "ktdb": _load_ktdb,
    "ltds": _load_ltds,
    "vista": _load_vista,
    "qhts": _load_qhts,
    "cmap": _load_cmap,
    "nhts": _load_nhts,
    "nts": _load_nts,
    "odin": _load_odin,
    "ovin": _load_ovin,
    "mon": _load_mon,
    "india": _load_india,
    "brazil": _load_brazil,
    "mexico": _load_mexico,
    "argentina": _load_argentina,
    "uruguay": _load_uruguay,
    "mongolia": _load_mongolia,
    "chile": _load_chile,
    "france": _load_france,
}


class SourceDataError(Exception):
    """A source's data is missing or not laid out as its loader expects."""


def _layout_hint(name: str) -> str:
    return f"see data_dir_template/{name.upper()} for the expected layout"


def check_source_dir(name: str, source_dir: str | Path) -> Path:
    """*source_dir* expanded, raising SourceDataError if it is missing or
    empty."""
    path = Path(source_dir).expanduser()
    if not path.is_dir():
        raise SourceDataError(
            f"{name}: data directory {path} not found; {_layout_hint(name)}"
        )
    if not any(path.iterdir()):
        raise SourceDataError(
            f"{name}: data directory {path} is empty; {_layout_hint(name)}"
        )
    return path


def batch_source_dir(data_root: str | Path, name: str) -> Path:
    """The directory holding source *name* within a batch data root."""
    return Path(data_root).expanduser() / name.upper()


def load_source(name: str, source_dir: str | Path):
    """Load and process a single source from its own data directory.

    Returns its attributes, trips and locations, and its original ids (see
    ``utils.enumerate_ids``). Raises SourceDataError if its data is missing.
    """
    if name not in SOURCES:
        raise ValueError(
            f"Unknown source '{name}', expected one of {', '.join(SOURCES)}"
        )
    path = check_source_dir(name, source_dir)
    with table_logger.source(name.upper()):
        try:
            attributes, trips = SOURCES[name](path)
        except FileNotFoundError as e:
            raise SourceDataError(f"{name}: {e}; {_layout_hint(name)}") from e
        if attributes.is_empty():
            raise SourceDataError(
                f"{name}: no persons loaded from {path}; {_layout_hint(name)}"
            )
        # location keys -> this source's location ids (from 0)
        attributes, trips, locations = location.finder(name).index(
            attributes, trips
        )
        # source ids -> this source's pid and hid (from 0)
        attributes, trips, ids = utils.enumerate_ids(attributes, trips)
        logger.info(f"Processing {name.upper()}...")
        attributes, trips = process_source(attributes, trips, name.upper())
        return attributes, trips, locations, ids


def _id_map(col: str, keys: pl.Series, offset: int) -> pl.DataFrame:
    """Combined id (*col*) -> original id (``original_<col>``) for one
    source's *keys*, numbered from *offset*."""
    return pl.DataFrame(
        {
            col: pl.int_range(
                offset, offset + len(keys), dtype=pl.Int32, eager=True
            ),
            f"original_{col}": keys,
        }
    )


def log_locations(
    attributes: pl.DataFrame, trips: pl.DataFrame, locations: pl.DataFrame
) -> None:
    """Log, per source, households without a home location, and the
    location summary table (as in the README)."""
    unlocated = (
        attributes.unique("hid")
        .filter(pl.col("hh_location").is_null())
        .group_by("source")
        .len()
        .sort("source")
    )
    for source, n in unlocated.iter_rows():
        logger.warning(f"{source}: {n} households have no home location")
    log_markdown_table(
        "Locations",
        location.summary_markdown(
            location.summary(attributes, trips, locations)
        ),
    )


def runner(
    source_dirs: dict[str, str | Path],
    output: str | Path,
    home_based: bool = False,
    fix_consecutive: bool = False,
    diagnostics: bool = False,
    keep_ids: bool = False,
    world: str | Path | None = gridjoin.DEFAULT_WORLD,
    grid_cache: str | Path = gridjoin.DEFAULT_CACHE,
):
    """Load, process, combine and write the given sources, then report.

    *source_dirs* maps each source name to its own data directory. The
    summary tables and README plots are always produced; *diagnostics* adds
    the anomaly-detection reports and supplementary heatmaps. *keep_ids*
    also writes ``original_ids.csv``, mapping each pid and hid back to the
    survey's own ids.

    WorldTensor layers under *world* (preprocessed into *grid_cache*) are
    joined to the locations, and from them to each household's
    attributes (``wt_`` columns); with *world* None they stay null.
    """
    output = Path(output).expanduser()
    output.mkdir(exist_ok=True, parents=True)

    sources = [s for s in SOURCES if s in source_dirs]
    if not sources:
        logger.warning("No sources selected. Exiting.")
        return

    logger.info(f"Selected sources: {', '.join(sources)}")
    # fail before loading anything if a source's directory is missing
    problems = []
    for name in sources:
        try:
            check_source_dir(name, source_dirs[name])
        except SourceDataError as e:
            problems.append(str(e))
    if world is not None:
        world = Path(world).expanduser()
        problems += [
            f"{p} (or run without the WorldTensor join: --no-grid)"
            for p in gridjoin.worldtensor.missing_sources(world)
        ]
    if problems:
        raise SourceDataError("\n".join(problems))
    table_logger.reset()

    all_attributes = []
    all_trips = []
    all_locations = []
    all_ids = {"pid": [], "hid": []}
    # each source's location, person and household ids follow the
    # previous source's
    offset = 0
    pid_offset = 0
    hid_offset = 0
    for i, name in enumerate(sources, start=1):
        logger.info(f"===[{i}/{len(sources)}] Loading {name.upper()}===")
        start = time.perf_counter()
        attributes, trips, locations, ids = load_source(name, source_dirs[name])
        logger.info(
            f"{name.upper()} done in {time.perf_counter() - start:.1f}s"
        )
        all_attributes.append(
            attributes.with_columns(
                pl.col("hh_location") + offset,
                pl.col("pid") + pid_offset,
                pl.col("hid") + hid_offset,
            )
        )
        all_trips.append(
            trips.with_columns(
                pl.col("ocode", "dcode") + offset, pl.col("pid") + pid_offset
            )
        )
        all_locations.append(
            locations.with_columns(pl.col("location_id") + offset)
        )
        if keep_ids:
            for col, col_offset in (("pid", pid_offset), ("hid", hid_offset)):
                all_ids[col].append(_id_map(col, ids[col], col_offset))
        offset += locations.height
        pid_offset += len(ids["pid"])
        hid_offset += len(ids["hid"])

    # ------------------------------------------------------------------
    # Concat and write
    # ------------------------------------------------------------------

    logger.info("=== Post processing all sources===")

    logger.debug("Concatenating sources...")
    all_attributes = pl.concat(all_attributes, how="vertical")
    all_trips = pl.concat(all_trips, how="vertical")

    if home_based:
        logger.info("Filtering to home-based trips only...")
        all_attributes, all_trips = filter.home_based(all_attributes, all_trips)

    if fix_consecutive:
        non_consecutive_types = ["home", "work", "education"]
        logger.info(
            f"Fixing consecutive activities of {non_consecutive_types}..."
        )
        all_trips = utils.combine_consecutive_acts(
            all_trips, non_consecutive_types=non_consecutive_types
        )

    # the locations households and trips still refer to
    used = pl.concat(
        [all_attributes["hh_location"], all_trips["ocode"], all_trips["dcode"]]
    ).drop_nulls()
    locations = pl.concat(all_locations).filter(
        pl.col("location_id").is_in(used.implode())
    )
    log_locations(all_attributes, all_trips, locations)
    grid_meta = None
    if world is not None:
        locations, grid_meta = gridjoin.join.join_locations(
            locations, world, Path(grid_cache).expanduser()
        )
        all_attributes = gridjoin.join.to_attributes(all_attributes, locations)

    if keep_ids:
        logger.warning(
            "Writing original_ids.csv: it links records back to the surveys' "
            "own ids, so keep it private"
        )
        (
            all_attributes.select("source", "pid", "hid")
            .join(pl.concat(all_ids["pid"]), on="pid", how="left")
            .join(pl.concat(all_ids["hid"]), on="hid", how="left")
            .write_csv(output / "original_ids.csv")
        )
    del all_ids

    logger.info(f"Writing attributes, trips and locations to {output}...")
    all_attributes.write_csv(output / "attributes.csv")
    location.write(
        locations,
        output / "locations.parquet",
        metadata=grid_meta and {gridjoin.join.METADATA_KEY: grid_meta},
    )
    binned_attributes = post_process.discretise_numeric(
        all_attributes,
        n_bins=5,
        method="quantile",
        exclude_cols=[
            "year",
            "month",
            "weight",
            "vehicles",
            "hh_size",
            "hh_location",
        ],
    )
    binned_attributes = post_process.fill_nulls(binned_attributes)
    binned_attributes.write_csv(output / "binned_attributes.csv")
    all_trips.write_csv(output / "trips.csv")

    if not verify.trips_pids_subset_of_attributes(all_attributes, all_trips):
        raise ValueError("Trips has pids not in attributes")

    activity_schedules = post_process.trips_to_activity_schedules(
        all_attributes, all_trips
    )
    if not verify.activities_pids_match_attributes(
        all_attributes, activity_schedules
    ):
        raise ValueError("Activities pids do not match attributes pids")

    activity_schedules.write_csv(output / "activity_schedules.csv")

    logger.info(f"Written attributes, trips and activities to {output}")

    # ------------------------------------------------------------------
    # Summary tables and README plots
    # ------------------------------------------------------------------
    log_markdown_table(
        "Processing Log (% of plans removed per step)", table_logger.render()
    )
    table_logger.log_notes()

    log_markdown_table(
        "Summary",
        tables.summary_table(all_attributes, all_trips, markdown=True),
    )

    log_markdown_table(
        "Attribute Availability by Source",
        tables.attribute_availability(all_attributes, markdown=True),
    )

    logger.info("Plotting attributes_numeric.png...")
    plots.numeric_hist_grid(
        all_attributes,
        on="source",
        n_cols=3,
        bins=11,
        density=True,
        ignore_cols={"weight", "hh_location"},
        tail_handling="clip",
        tail_ratio_threshold=4,
        outlier_share_max=0.2,
        clip_percentiles=(1.0, 99.0),
        min_unique=5,
        min_group_rows=10,
        verbose=True,
        save_path=output / "attributes_numeric.png",
    )

    logger.info("Plotting attributes_categorical.png...")
    plots.categorical_bar_grid(
        all_attributes,
        on="source",
        save_path=output / "attributes_categorical.png",
    )

    logger.info("Plotting attributes_trends.png...")
    plots.summary_trends(
        all_attributes, on="source", save_path=output / "attributes_trends.png"
    )

    logger.info("Plotting trip_time_of_day.png...")
    plots.time_of_day_profile(
        all_attributes,
        all_trips,
        on="source",
        save_path=output / "trip_time_of_day.png",
    )

    logger.info("Plotting trip_time_heaping.png...")
    plots.time_heaping(
        all_attributes,
        all_trips,
        on="source",
        save_path=output / "trip_time_heaping.png",
    )

    logger.info("Plotting trip_time_diagnostics.png...")
    plots.trip_time_diagnostics(
        all_attributes,
        all_trips,
        on="source",
        save_path=output / "trip_time_diagnostics.png",
    )

    logger.info("Plotting activity_duration_by_type.png...")
    plots.activity_duration_by_type(
        all_attributes,
        activity_schedules,
        on="source",
        save_path=output / "activity_duration_by_type.png",
    )

    logger.info("Plotting activity_counts_grid.png...")
    plots.activities_attributes_grid(
        all_attributes,
        activity_schedules,
        attribute_cols={
            "employment": "bar",
            "hh_income": "line",
            "age": "line",
        },
        on="source",
        save_path=output / "activity_counts_grid.png",
    )

    if diagnostics:
        _diagnostics(all_attributes, all_trips, activity_schedules, output)
    else:
        logger.info(
            "Skipping diagnostics reports and supplementary plots "
            "(enable with --diagnostics)"
        )

    logger.success(f"Done, outputs saved to {output}")


def _diagnostics(
    all_attributes: pl.DataFrame,
    all_trips: pl.DataFrame,
    activities: pl.DataFrame,
    output: Path,
):
    """Supplementary attribute x activity heatmaps and anomaly detection."""
    logger.info("Plotting activity_heatmap_by_employment.png...")
    plots.attribute_activity_heatmap(
        all_attributes,
        activities,
        attribute_col="employment",
        on="source",
        save_path=output / "activity_heatmap_by_employment.png",
    )

    logger.info("Plotting activity_heatmap_by_hh_income.png...")
    # sources such as mexico have no income at all; attribute_activity_heatmap
    # skips plotting (rather than raising) when a column is all null
    plots.attribute_activity_heatmap(
        all_attributes,
        activities,
        attribute_col="hh_income",
        n_bins=8,
        on="source",
        save_path=output / "activity_heatmap_by_hh_income.png",
    )

    logger.info("Plotting activity_heatmap_by_year.png...")
    plots.attribute_activity_heatmap(
        all_attributes,
        activities,
        attribute_col="year",
        on="source",
        save_path=output / "activity_heatmap_by_year.png",
    )

    logger.info("Plotting activity_heatmap_by_day.png...")
    # likewise uruguay never records the travel day
    plots.attribute_activity_heatmap(
        all_attributes,
        activities,
        attribute_col="day",
        on="source",
        save_path=output / "activity_heatmap_by_day.png",
    )

    logger.info("Plotting activity_heatmap_by_age.png...")
    all_attributes_age = post_process.add_age_band(all_attributes)
    plots.attribute_activity_heatmap(
        all_attributes_age,
        activities,
        attribute_col="age_band",
        on="source",
        save_path=output / "activity_heatmap_by_age.png",
    )

    logger.info("Running anomaly detection...")
    # anomaly detection - time quality
    log_markdown_table(
        "Time Quality",
        anomaly.time_quality_summary_table(
            all_attributes, all_trips, markdown=True
        ),
    )

    # anomaly detection - activities
    activity_distribution_shift = anomaly.activity_distribution_shift_matrix(
        all_attributes, activities, on=["source", "year"]
    )
    activity_distribution_shift.write_csv(
        output / "activity_distribution_shift_matrix.csv"
    )
    activity_distribution_shift_outliers = (
        anomaly.flag_distribution_shift_outliers(
            activity_distribution_shift,
            on=["source", "year"],
            top_n=20,
            markdown=True,
        )
    )
    log_markdown_table(
        "20 most unusual activity-participation distributions by source-year "
        "(Jensen-Shannon divergence vs the pooled activity-type mix across "
        "all sources and years)",
        activity_distribution_shift_outliers,
    )

    # anomaly detection - attributes
    distribution_shift = anomaly.attribute_distribution_shift_matrix(
        all_attributes, on=["source", "year"]
    )
    distribution_shift.write_csv(output / "distribution_shift_matrix.csv")
    distribution_shift_outliers = anomaly.flag_distribution_shift_outliers(
        distribution_shift, on=["source", "year"], top_n=20, markdown=True
    )
    log_markdown_table(
        "20 most unusual attribute distributions by source-year (Jensen-Shannon "
        "divergence vs the pooled distribution across all sources and years)",
        distribution_shift_outliers,
    )

    # anomaly detection - conditionality
    conditionality_by_year = anomaly.conditionality_matrix(
        all_attributes, activities, on=["source", "year"]
    )
    conditionality_by_year.write_csv(
        output / "conditionality_matrix_by_year.csv"
    )
    outliers_by_year = anomaly.flag_conditionality_outliers(
        conditionality_by_year, on=["source", "year"], top_n=20, markdown=True
    )
    log_markdown_table(
        "Top 20 conditionality outliers by source-year (attribute x activity "
        "Cramér's V vs peer source-years, most anomalous first; catches "
        "per-year mapping bugs that a source-level check averages away)",
        outliers_by_year,
    )
