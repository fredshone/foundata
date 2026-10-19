"""Montevideo metropolitan area mobility survey (EOD 2016), Uruguay.

Open data from catalogodatos.gub.uy (``eod2016.zip``): four ``;``-separated
CSVs linked by NFORM (hogar), nnper (person), nvj (trip) and netp (stage),
with codebooks in the accompanying ``Dicc *.txt`` files. Blank cells are
missing values and decimals use a comma.

Each person aged 4+ reports the trips made from 4am the previous day (a
Monday-Friday, not recorded which) to 4am on the interview day. The survey
has no income and no distances; distances come from blocks walked, census
segment points (the ``segment_2011`` location lookup) or duration x mode
speed.
"""

from pathlib import Path

import polars as pl
from loguru import logger

from foundata import location, utils

SOURCE = "uruguay"
COUNTRY = "uy"

SURVEY_DIR = "EOD2016"
HOUSEHOLDS_PATH = "Base Hogar Habitos.csv"
PERSONS_PATH = "Base Personas.csv"
TRIPS_PATH = "Base Viajes.csv"
STAGES_PATH = "Base Etapas.csv"


def read_table(path: Path, column_mapping: dict) -> pl.DataFrame:
    """Read one survey CSV as strings, blanks -> null, renamed."""
    data = pl.read_csv(
        path,
        separator=";",
        columns=list(column_mapping),
        infer_schema=False,
        encoding="utf8",
    )
    return data.with_columns(
        pl.all().str.strip_chars().replace("", None)
    ).rename(column_mapping)


def load(
    data_root: str | Path,
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load and normalise Montevideo EOD 2016 survey data.

    Args:
        data_root: Path to the raw data directory for this source.
        hh_config: Parsed hh_dictionary.yaml config.
        person_config: Parsed person_dictionary.yaml config.
        trips_config: Parsed trip_dictionary.yaml config.

    Returns:
        (attributes, trips) DataFrames conforming to the template schema.
    """
    logger.info(f"Loading {SOURCE} data...")
    root = Path(data_root).expanduser() / SURVEY_DIR
    trip_mappings = trips_config["column_mappings"]

    hhs = load_households(
        read_table(root / HOUSEHOLDS_PATH, hh_config["column_mappings"]),
        hh_config,
    )
    persons = load_persons(
        read_table(root / PERSONS_PATH, person_config["column_mappings"]),
        person_config,
    )
    segments = load_segments(hh_config)
    trips = load_trips(
        read_table(root / TRIPS_PATH, trip_mappings["Base Viajes"]),
        trips_config,
        segments,
    )
    stages = read_table(root / STAGES_PATH, trip_mappings["Base Etapas"])
    trips = trips.join(
        access_egress_distances(stages, trips_config),
        on=["pid", "seq"],
        how="left",
    )

    attributes = (
        persons.join(hhs, on="hid", how="left")
        .join(access_egress(trips), on="pid", how="left")
        .with_columns(
            source=pl.lit(SOURCE),
            country=pl.lit(COUNTRY),
            year=pl.lit(trips_config["year"], dtype=pl.Int32),
            month=pl.lit(None, dtype=pl.Int8),
            day=pl.lit("unknown"),
            hh_income=pl.lit(None, dtype=pl.Int32),
            pid=pl.lit(SOURCE) + pl.col("pid"),
            hid=pl.lit(SOURCE) + pl.col("hid"),
        )
    )
    trips = trips.select(
        "seq",
        "ozone",
        "dzone",
        "oact",
        "dact",
        "mode",
        "tst",
        "tet",
        "distance",
        "ocode",
        "dcode",
        pid=pl.lit(SOURCE) + pl.col("pid"),
    ).select("pid", pl.exclude("pid"))
    return attributes, trips


def zone_type(codloc: pl.Expr, config: dict) -> pl.Expr:
    """INE locality code -> urban (Montevideo) / rural (<depto>900) / suburban."""
    return (
        pl.when(codloc.is_null())
        .then(pl.lit("unknown"))
        .when(codloc.is_in(config["urban_localities"]))
        .then(pl.lit("urban"))
        .when(codloc % 1000 == config["rural_locality_suffix"])
        .then(pl.lit("rural"))
        .otherwise(pl.lit("suburban"))
    )


def load_households(hhs: pl.DataFrame, config: dict) -> pl.DataFrame:
    """One row per hogar. vehicles counts cars and mopeds/motorcycles."""
    unknown = config["vehicle_unknown_codes"]

    def count(has: str, n: str) -> pl.Expr:
        return (
            pl.when(pl.col(has) == 2)
            .then(0)
            .when(pl.col(has).is_in(unknown) | pl.col(n).is_in(unknown))
            .then(None)
            .otherwise(pl.col(n))
        )

    hhs = hhs.with_columns(
        pl.col("dwelling", "ownership", "hh_zone").cast(pl.Int32),
        pl.col("hh_size", "hh_size_surveyed").cast(pl.Int32),
        pl.col("has_cars", "cars", "has_mopeds", "mopeds").cast(pl.Int32),
    )
    hhs = hhs.select(
        "hid",
        "home_segment",
        hh_size=pl.when(
            pl.col("hh_size").is_in(config["hh_size_unknown_codes"])
        )
        .then(pl.col("hh_size_surveyed"))
        .otherwise(pl.col("hh_size")),
        dwelling=pl.col("dwelling").replace_strict(
            config["dwelling"], default="unknown"
        ),
        ownership=pl.col("ownership").replace_strict(
            config["ownership"], default="unknown"
        ),
        vehicles=(count("has_cars", "cars") + count("has_mopeds", "mopeds")),
        hh_zone=zone_type(pl.col("hh_zone"), config),
    )
    return location.add(
        hhs.with_columns(home_segment=segment_code("home_segment")),
        source="uruguay",
    )


def segment_code(column: str) -> pl.Expr:
    """An INE 2011 census segment code (0 and blank are unknown)."""
    return location.clean(column, valid=r"[1-9]\d*")


def load_persons(persons: pl.DataFrame, config: dict) -> pl.DataFrame:
    """One row per person asked about travel (aged 4+, E19 answered).

    pid is ``<NFORM>`` + 2-digit ``nnper`` (the survey's IDPER).
    """
    persons = persons.with_columns(
        pl.all().exclude("hid", "pid", "weight").cast(pl.Int32),
        weight=pl.col("weight").str.replace(",", ".").cast(pl.Float32),
        pid=pl.col("hid") + pl.col("pid").str.zfill(2),
    ).filter(pl.col("travelled").is_in(config["travelled_codes"]))

    employed = (pl.col("worked") == config["worked_code"]) | (
        pl.col("has_job") == config["has_job_code"]
    )
    adult_question = pl.col("worked").is_not_null()
    return persons.select(
        "pid",
        "hid",
        "weight",
        "age",
        sex=pl.col("sex").replace_strict(config["sex"], default="unknown"),
        relationship=pl.col("relationship").replace_strict(
            config["relationship"], default="unknown"
        ),
        education=education(config),
        employment=pl.when(employed)
        .then(pl.lit("employed"))
        .otherwise(
            pl.col("employment").replace_strict(
                config["employment"], default="unknown"
            )
        ),
        has_licence=pl.when(pl.col("has_licence").is_null() & ~adult_question)
        .then(pl.lit("void"))
        .otherwise(
            pl.col("has_licence").replace_strict(
                config["has_licence"], default="unknown"
            )
        ),
        occupation=pl.when(employed)
        .then(pl.lit("unknown"))
        .otherwise(pl.lit("void")),
        disability=pl.lit("unknown"),
        can_wfh=pl.lit("unknown"),
        race=pl.lit("unknown"),
    )


def education(config: dict) -> pl.Expr:
    """Highest level from E4, adjusted by completion (E6) and E3."""
    level = pl.col("education")
    by_level = level.replace_strict(config["education"], default="unknown")
    completed = level.replace_strict(
        config["education_if_completed"], default=None, return_dtype=pl.String
    )
    incomplete = level.replace_strict(
        config["education_if_incomplete"], default=None, return_dtype=pl.String
    )
    return (
        pl.when(pl.col("attended") == config["never_attended_code"])
        .then(pl.lit("none"))
        .when(pl.col("completed") == 1)
        .then(completed.fill_null(by_level))
        .when(pl.col("completed") == 2)
        .then(incomplete.fill_null(by_level))
        .otherwise(by_level)
    )


def load_segments(hh_config: dict) -> pl.DataFrame:
    """Census segment points (UTM 21S metres, from the segment_2011
    location lookup) and zone type (from the segment's locality,
    configs/uruguay/segment_localities.csv)."""
    localities = pl.read_csv(
        utils.get_config_path(SOURCE, "segment_localities.csv"),
        schema_overrides={"codseg": pl.String},
    )
    points = location.zone_points(
        SOURCE, "segment_2011", crs="EPSG:32721"
    ).select(codseg="code", x="x", y="y")
    return points.join(localities, on="codseg", how="left").select(
        "codseg", "x", "y", zone=zone_type(pl.col("codloc"), hh_config)
    )


def clock_minutes(col: str, unknown: list[str]) -> pl.Expr:
    """'H,MM' (trailing zero dropped: '7,4' is 7:40) -> minutes, null if unknown."""
    parts = pl.col(col).str.split(",")
    hours = parts.list.get(0).cast(pl.Int32)
    minutes = (
        parts.list.get(1, null_on_oob=True)
        .str.pad_end(2, "0")
        .cast(pl.Int32)
        .fill_null(0)
    )
    return (
        pl.when(pl.col(col).is_in(unknown))
        .then(None)
        .otherwise(hours * 60 + minutes)
    )


def load_trips(
    trips: pl.DataFrame, config: dict, segments: pl.DataFrame
) -> pl.DataFrame:
    """One row per trip, 0-indexed seq, times in minutes since midnight.

    Unknown times are left null, so the plan is dropped downstream.
    """
    trips = location.add_trip_ends(
        trips.with_columns(
            origin_segment=segment_code("ozone"),
            destination_segment=segment_code("dzone"),
        ),
        source="uruguay",
    )
    trips = trips.with_columns(
        pl.col("seq", "oact", "dact", "mode", "blocks").cast(pl.Int32),
        pid=pl.col("hid") + pl.col("pid").str.zfill(2),
        tst=clock_minutes("tst", config["unknown_times"]),
        tet=clock_minutes("tet", config["unknown_times"]),
    )
    for end in ("o", "d"):
        trips = trips.join(
            segments.rename(
                {"x": f"{end}x", "y": f"{end}y", "zone": f"{end}_zone"}
            ),
            left_on=f"{end}zone",
            right_on="codseg",
            how="left",
        )
    trips = trips.with_columns(
        seq=(pl.col("seq") - 1).cast(pl.Int8),
        oact=pl.col("oact").replace_strict(config["dact"], default="unknown"),
        dact=pl.col("dact").replace_strict(config["dact"], default="unknown"),
        ozone=pl.col("o_zone").fill_null("unknown"),
        dzone=pl.col("d_zone").fill_null("unknown"),
        distance=trip_distance(config),
        mode=pl.col("mode").replace_strict(config["mode"], default="unknown"),
    )
    trips = trips.drop(
        "hid", "blocks", "ox", "oy", "dx", "dy", "o_zone", "d_zone"
    )
    # The survey clock runs 4h-28h, so times are already continuous across
    # midnight; fix.day_wrap would instead shift trips that overlap in the
    # raw data by a day rather than leave them for the time filters.
    return trips.sort("pid", "seq")


def trip_distance(config: dict) -> pl.Expr:
    """Trip distance (km); see the distance notes in trip_dictionary.yaml.

    Expects the raw (integer) mode and the segment centroid columns.
    """
    straight_line = (
        (pl.col("ox") - pl.col("dx")) ** 2 + (pl.col("oy") - pl.col("dy")) ** 2
    ).sqrt() / 1000
    duration_h = pl.when(pl.col("tet") > pl.col("tst")).then(
        (pl.col("tet") - pl.col("tst")) / 60
    )
    by_speed = duration_h * pl.col("mode").replace_strict(
        config["speed"], default=None, return_dtype=pl.Float64
    )
    return (
        pl.when(
            pl.col("mode").is_in(config["walk_modes"])
            & pl.col("blocks").is_not_null()
        )
        .then(pl.col("blocks") * config["block_km"])
        .when(straight_line > 0)
        .then(straight_line * config["circuity"])
        .otherwise(by_speed)
        .cast(pl.Float32)
    )


def access_egress_distances(stages: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Per transit trip (pid, seq): km walked to, from and between vehicles.

    A trip counts as transit if any stage is a bus or train.
    """
    block_cols = [
        "blocks_to_vehicle",
        "blocks_from_vehicle",
        "walk_stage_blocks",
    ]
    stages = stages.with_columns(
        pl.col(block_cols).cast(pl.Int32),
        pid=pl.col("hid") + pl.col("pid").str.zfill(2),
        seq=(pl.col("seq").cast(pl.Int32) - 1).cast(pl.Int8),
    ).with_columns(
        pl.when(pl.col(c).is_in(config["unknown_blocks"]))
        .then(None)
        .otherwise(pl.col(c))
        .alias(c)
        for c in block_cols
    )
    stage_transit = (
        pl.col("stage_mode").cast(pl.Int32).is_in(config["transit_stage_modes"])
    )
    return stages.group_by("pid", "seq").agg(
        access_egress_distance=pl.when(stage_transit.any())
        .then(pl.sum_horizontal(block_cols).sum() * config["block_km"])
        .otherwise(None)
        .cast(pl.Float32)
    )


def access_egress(trips: pl.DataFrame) -> pl.DataFrame:
    """Mean per-person access/egress distance (km) over transit trips."""
    return (
        trips.filter(pl.col("access_egress_distance").is_not_null())
        .group_by("pid")
        .agg(pl.col("access_egress_distance").mean().cast(pl.Float32))
    )
