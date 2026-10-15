"""Santiago (Gran Santiago) Origin-Destination Survey (EOD 2012), Chile.

Open data from SECTRA, taken as parquet from
github.com/PLUMAS-research/palomatrix-dtpm (src/palomatrix/eod/datos), a
lossless conversion of the original CSVs: hogares (hogar), personas
(person), viajes (trip), etapas (stage) and vehiculos, linked by Hogar,
Persona and Viaje, with the code tables in codigos.parquet.

Every household member reports the trips of one assigned day (Fecha, any
weekday, normal and summer seasons). Trip distances come from the survey's
straight-line distance between block centroids, EOD zone centroids
(``configs/chile/zones.csv``, from ``scripts/compute_chile_zones.py``) or
duration x mode speed.
"""

from pathlib import Path

import polars as pl
from loguru import logger

from foundata import fix, utils

SOURCE = "chile"
COUNTRY = "cl"

HOUSEHOLDS_PATH = "hogares.parquet"
VEHICLES_PATH = "vehiculos.parquet"
PERSONS_PATH = "personas.parquet"
TRIPS_PATH = "viajes.parquet"
STAGES_PATH = "etapas.parquet"


def read_table(path: Path, column_mapping: dict) -> pl.DataFrame:
    """Read the mapped columns of one survey table, renamed."""
    return pl.read_parquet(path, columns=list(column_mapping)).rename(
        column_mapping
    )


def load(
    data_root: str | Path,
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load and normalise Santiago EOD 2012 survey data.

    Args:
        data_root: Path to the raw data directory for this source.
        hh_config: Parsed hh_dictionary.yaml config.
        person_config: Parsed person_dictionary.yaml config.
        trips_config: Parsed trip_dictionary.yaml config.

    Returns:
        (attributes, trips) DataFrames conforming to the template schema.
    """
    logger.info(f"Loading {SOURCE} data...")
    root = Path(data_root).expanduser()
    hh_mappings = hh_config["column_mappings"]
    trip_mappings = trips_config["column_mappings"]

    persons = load_persons(
        read_table(root / PERSONS_PATH, person_config["column_mappings"]),
        person_config,
    )
    raw_hhs = read_table(root / HOUSEHOLDS_PATH, hh_mappings["hogares"])
    zones = load_zones(raw_hhs, persons, hh_config["density_thresholds"])
    hhs = load_households(
        raw_hhs,
        read_table(root / VEHICLES_PATH, hh_mappings["vehiculos"]),
        persons,
        hh_config,
        zones,
    )
    home_zones = persons.select("pid", "hid").join(
        raw_hhs.select(
            pl.col("hid").cast(pl.String), home_zone=pl.col("hh_zone")
        ),
        on="hid",
    )
    trips = load_trips(
        read_table(root / TRIPS_PATH, trip_mappings["viajes"]),
        read_table(root / STAGES_PATH, trip_mappings["etapas"]),
        trips_config,
        zones,
        home_zones.drop("hid"),
    )

    attributes = (
        persons.drop("income_imputed")
        .join(hhs, on="hid", how="left")
        .join(access_egress(trips), on="pid", how="left")
        .with_columns(
            source=pl.lit(SOURCE),
            country=pl.lit(COUNTRY),
            dwelling=pl.lit("unknown"),
            can_wfh=pl.lit("unknown"),
            race=pl.lit("unknown"),
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
        pid=pl.lit(SOURCE) + pl.col("pid"),
    ).select("pid", pl.exclude("pid"))
    return attributes, trips


def load_zones(
    hhs: pl.DataFrame, persons: pl.DataFrame, thresholds: dict
) -> pl.DataFrame:
    """EOD zones with centroid (UTM metres) and urbanity.

    Urbanity comes from the weighted number of residents per km² of zone
    area. Zones with no sampled residents are unknown.
    """
    residents = (
        persons.join(
            hhs.select(pl.col("hid").cast(pl.String), zone="hh_zone"),
            on="hid",
        )
        .group_by("zone")
        .agg(population=pl.col("weight").cast(pl.Float64).sum())
    )
    zones = pl.read_csv(utils.get_config_path(SOURCE, "zones.csv"))
    density = pl.col("population") / pl.col("area_km2")
    return zones.join(residents, on="zone", how="left").select(
        "zone",
        "x",
        "y",
        zone_type=pl.when(density >= thresholds["urban"])
        .then(pl.lit("urban"))
        .when(density >= thresholds["suburban"])
        .then(pl.lit("suburban"))
        .when(density.is_not_null())
        .then(pl.lit("rural"))
        .otherwise(pl.lit("unknown")),
    )


def zone_type(zone: pl.Expr, zones: pl.DataFrame) -> pl.Expr:
    """EOD zone number -> urban/suburban/rural/unknown."""
    return zone.replace_strict(
        zones["zone"], zones["zone_type"], default="unknown"
    )


def load_households(
    hhs: pl.DataFrame,
    vehicles: pl.DataFrame,
    persons: pl.DataFrame,
    config: dict,
    zones: pl.DataFrame,
) -> pl.DataFrame:
    """One row per hogar, with travel date and annual EUR income."""
    n_vehicles = (
        vehicles.filter(pl.col("vehicle_type").is_in(config["vehicle_types"]))
        .group_by(pl.col("hid").cast(pl.String))
        .agg(vehicles=pl.len().cast(pl.Int32))
    )
    unimputed = persons.group_by("hid").agg(
        income_ok=pl.col("income_imputed")
        .is_in(config["income_imputation_codes"])
        .all()
    )
    date = pl.col("date").str.to_date("%d-%m-%Y")
    hhs = hhs.with_columns(
        hid=pl.col("hid").cast(pl.String),
        year=date.dt.year().cast(pl.Int32),
        month=date.dt.month().cast(pl.Int8),
        day=date.dt.strftime("%A").str.to_lowercase(),
    )
    rate = pl.col("year").replace_strict(
        config["eur_per_clp"], return_dtype=pl.Float64
    )
    return (
        hhs.join(n_vehicles, on="hid", how="left")
        .join(unimputed, on="hid", how="left")
        .select(
            "hid",
            "year",
            "month",
            "day",
            hh_size=pl.col("hh_size").cast(pl.Int32),
            hh_income=pl.when(pl.col("income_ok"))
            .then((pl.col("hh_income") * 12 * rate).round())
            .cast(pl.Int32),
            ownership=pl.col("ownership").replace_strict(
                config["ownership"], default="unknown"
            ),
            vehicles=pl.col("vehicles").fill_null(0),
            hh_zone=zone_type(pl.col("hh_zone"), zones),
        )
    )


def split_codes(col: str) -> pl.Expr:
    """';'-separated multi-valued answer -> list of codes."""
    return pl.col(col).cast(pl.String).str.split(";")


def employment(config: dict) -> pl.Expr:
    """Employment from the first declared activity in employment_priority.

    Workers become ft-employed/pt-employed by work_hours where known.
    """
    activities = split_codes("activity")
    first = pl.coalesce(
        pl.when(activities.list.contains(code)).then(pl.lit(code))
        for code in config["employment_priority"]
    )
    status = first.replace_strict(config["employment"], default="unknown")
    hours = pl.col("work_hours").replace_strict(
        config["work_hours"], default="employed", return_dtype=pl.String
    )
    return pl.when(status == "employed").then(hours).otherwise(status)


def load_persons(persons: pl.DataFrame, config: dict) -> pl.DataFrame:
    """One row per person; pid is Persona (Hogar + 2-digit number)."""
    persons = persons.with_columns(
        pl.col("education", "work_hours", "occupation").cast(pl.Int32),
        pid=pl.col("pid").cast(pl.String),
        hid=pl.col("hid").cast(pl.String),
    ).with_columns(employment=employment(config))

    licences = split_codes("has_licence").list.eval(
        pl.element().cast(pl.Int32).is_in(config["licence_codes"])
    )
    disabilities = split_codes("disability").list.eval(
        pl.element() != config["no_disability_code"]
    )
    return persons.select(
        "pid",
        "hid",
        "employment",
        "income_imputed",
        age=pl.col("age").cast(pl.Int32),
        weight=pl.col("weight").cast(pl.Float32),
        sex=pl.col("sex").replace_strict(config["sex"], default="unknown"),
        relationship=pl.col("relationship").replace_strict(
            config["relationship"], default="unknown"
        ),
        education=pl.col("education").replace_strict(
            config["education"], default="unknown"
        ),
        occupation=pl.when(pl.col("employment").str.ends_with("employed"))
        .then(
            pl.col("occupation").replace_strict(
                config["occupation"], default="unknown"
            )
        )
        .otherwise(pl.lit("void")),
        has_licence=pl.when(licences.list.any())
        .then(pl.lit("yes"))
        .when(licences.is_not_null())
        .then(pl.lit("no"))
        .otherwise(pl.lit("unknown")),
        disability=pl.when(disabilities.list.any())
        .then(pl.lit("yes"))
        .when(disabilities.is_not_null())
        .then(pl.lit("no"))
        .otherwise(pl.lit("unknown")),
    )


def clock_minutes(col: str) -> pl.Expr:
    """'H:MM' -> minutes since midnight (null stays null)."""
    parts = pl.col(col).str.split(":")
    return parts.list.get(0).cast(pl.Int32) * 60 + parts.list.get(1).cast(
        pl.Int32
    )


def trip_modes(stages: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Per trip_id: main mode (by mode_priority) and access/egress blocks.

    blocks_before sums the blocks walked to each stage, for trips with a
    transit stage only; unknown block counts are skipped.
    """
    priority = config["mode_priority"]
    stages = stages.with_columns(
        mode=pl.col("stage_mode").replace_strict(
            config["mode"], default="unknown"
        ),
        blocks_before=pl.when(
            pl.col("blocks_before").is_in(config["unknown_blocks"])
        )
        .then(None)
        .otherwise(pl.col("blocks_before")),
    ).with_columns(
        rank=pl.col("mode").replace_strict(
            priority, list(range(len(priority))), default=len(priority)
        )
    )
    transit = pl.col("stage_mode").is_in(config["transit_stage_modes"])
    return stages.group_by("trip_id").agg(
        mode=pl.col("mode").sort_by("rank").first(),
        transit=transit.any(),
        blocks_before=pl.col("blocks_before").sum(),
    )


def load_trips(
    trips: pl.DataFrame,
    stages: pl.DataFrame,
    config: dict,
    zones: pl.DataFrame,
    home_zones: pl.DataFrame,
) -> pl.DataFrame:
    """One row per trip, 0-indexed seq, times in minutes since midnight.

    A missing start or end time is filled from the other and TiempoViaje;
    trips with neither are left null so the plan is dropped downstream.
    """
    duration = pl.col("duration").cast(pl.Int32)
    trips = (
        trips.with_columns(
            pid=pl.col("pid").cast(pl.String),
            trip_no=pl.col("trip_id") % 100,
            tst=clock_minutes("tst"),
            tet=clock_minutes("tet"),
            dact=pl.col("dact")
            .cast(pl.Int32)
            .replace_strict(config["dact"], default="unknown"),
        )
        .with_columns(
            tst=pl.col("tst").fill_null((pl.col("tet") - duration) % 1440),
            tet=pl.col("tet").fill_null((pl.col("tst") + duration) % 1440),
        )
        .join(trip_modes(stages, config), on="trip_id", how="left")
        .join(home_zones, on="pid", how="left")
        .sort("pid", "trip_no")
    )
    for end in ("o", "d"):
        trips = trips.join(
            zones.select(
                pl.col("zone").alias(f"{end}zone"),
                pl.col("x").alias(f"{end}x"),
                pl.col("y").alias(f"{end}y"),
            ),
            on=f"{end}zone",
            how="left",
        )
    trips = trips.with_columns(
        seq=pl.int_range(pl.len()).over("pid").cast(pl.Int8),
        oact=pl.col("dact")
        .shift(1)
        .over("pid")
        .fill_null(
            pl.when(pl.col("ozone") == pl.col("home_zone"))
            .then(pl.lit("home"))
            .otherwise(pl.lit("other"))
        ),
        mode=pl.col("mode").fill_null("unknown"),
        distance=trip_distance(config),
        access_egress_distance=access_egress_distance(config),
    ).with_columns(
        ozone=zone_type(pl.col("ozone"), zones),
        dzone=zone_type(pl.col("dzone"), zones),
    )
    trips = fix.day_wrap(trips.sort("pid", "seq"))
    return trips.select(
        "pid",
        "seq",
        "ozone",
        "dzone",
        "oact",
        "dact",
        "mode",
        "tst",
        "tet",
        "distance",
        "access_egress_distance",
    )


def trip_distance(config: dict) -> pl.Expr:
    """Trip distance (km); see the distance notes in trip_dictionary.yaml.

    Expects the template mode, raw zones and the zone centroid columns.
    """
    circuity = config["circuity"]
    straight_line = pl.col("straight_line") / 1000
    between_zones = (
        (pl.col("ox") - pl.col("dx")) ** 2 + (pl.col("oy") - pl.col("dy")) ** 2
    ).sqrt() / 1000
    by_speed = (
        pl.col("duration")
        / 60
        * pl.col("mode").replace_strict(
            config["speed"], default=None, return_dtype=pl.Float64
        )
    )
    return (
        pl.when(straight_line > 0)
        .then(straight_line * circuity)
        .when((pl.col("ozone") != pl.col("dzone")) & (between_zones > 0))
        .then(between_zones * circuity)
        .otherwise(by_speed)
        .cast(pl.Float32)
    )


def access_egress_distance(config: dict) -> pl.Expr:
    """Km walked to, between and from vehicles on transit trips."""
    blocks_after = (
        pl.when(pl.col("blocks_after").is_in(config["unknown_blocks"]))
        .then(None)
        .otherwise(pl.col("blocks_after"))
    )
    blocks = pl.sum_horizontal("blocks_before", blocks_after)
    return (
        pl.when(pl.col("transit"))
        .then(blocks * config["block_km"])
        .cast(pl.Float32)
    )


def access_egress(trips: pl.DataFrame) -> pl.DataFrame:
    """Mean per-person access/egress distance (km) over transit trips."""
    return (
        trips.filter(pl.col("access_egress_distance").is_not_null())
        .group_by("pid")
        .agg(pl.col("access_egress_distance").mean().cast(pl.Float32))
    )
