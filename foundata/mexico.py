"""Mexico City Encuesta Origen Destino en Hogares (EOD) 2017, INEGI.

The raw data is a set of dBase tables linked by integer ids: ``THOGAR``
(households, ``ID_HOG``), ``TSDEM`` (persons, ``ID_SOC``), ``TVIAJE`` (trips,
``ID_VIA``) and ``TTRANSPORTE`` (one row per mode used on a trip). Persons
aged 6+ report a weekday diary (Tuesday-Thursday, day not recorded) and a
Saturday diary; each diary is returned as its own plan, with pid
``mexico<ID_SOC><diary>``. The survey has no distances, so they are estimated
from stage durations and mode speeds. Codes are documented in INEGI's
open-data bundle (``eod_2017_csv.zip``).
"""

from pathlib import Path

import polars as pl
from loguru import logger

from foundata import fix, location, utils
from foundata.brazil import read_dbf

SOURCE = "mexico"
COUNTRY = "mx"
CONTINENT = "north_americas"
YEAR = 2017

HOUSEHOLDS_PATH = "THOGAR.DBF"
PERSONS_PATH = "TSDEM.DBF"
TRIPS_PATH = "TVIAJE.DBF"
STAGES_PATH = "TTRANSPORTE.DBF"

UNKNOWN_TIME = 99  # hour/minute code for "no sabe"


def load(
    data_root: str | Path,
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load and normalise Mexico City EOD 2017 survey data.

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

    def read(path: str, column_mapping: dict) -> pl.DataFrame:
        return read_dbf(root / path, columns=list(column_mapping)).rename(
            column_mapping
        )

    hhs = load_households(
        read(HOUSEHOLDS_PATH, hh_config["column_mappings"]), hh_config
    )
    persons = read(PERSONS_PATH, person_config["column_mappings"])
    hhs = hhs.join(
        persons.group_by("hid").agg(hh_size=pl.len().cast(pl.Int32)), on="hid"
    )
    hhs = household_members(hhs, persons, person_config)
    person_days = load_person_days(persons, person_config, trips_config["day"])
    zone_types = district_zone_types(hhs)

    trip_mappings = trips_config["column_mappings"]
    trips = load_trips(
        read(TRIPS_PATH, trip_mappings["TVIAJE"]),
        read(STAGES_PATH, trip_mappings["TTRANSPORTE"]),
        trips_config,
        zone_types,
    )

    attributes = (
        person_days.join(
            hhs.drop("hh_weight", "home_district"), on="hid", how="left"
        )
        .join(access_egress(trips), on="pid", how="left")
        .with_columns(
            source=pl.lit(SOURCE),
            country=pl.lit(COUNTRY),
            continent=pl.lit(CONTINENT),
            year=pl.lit(YEAR, dtype=pl.Int32),
            month=pl.lit(None, dtype=pl.Int8),
            hh_income=pl.lit(None, dtype=pl.Int32),
            pid=pl.col("pid"),
            hid=pl.col("hid"),
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
        pid=pl.col("pid"),
    ).select("pid", pl.exclude("pid"))
    return attributes, trips


def household_members(
    hhs: pl.DataFrame, persons: pl.DataFrame, config: dict
) -> pl.DataFrame:
    """Add hh_adults and hh_children, counted over every member (including
    the under-6s later dropped), before persons become diaries."""
    return utils.add_member_counts(
        hhs, persons.with_columns(age=person_age(config))
    )


def person_age(config: dict) -> pl.Expr:
    """EDAD in years, null for the "no sabe" code."""
    age = pl.col("age").cast(pl.Int32)
    return pl.when(age != config["age_unknown"]).then(age)


def load_households(hhs: pl.DataFrame, config: dict) -> pl.DataFrame:
    """One row per hogar. vehicles counts cars and motorcycles."""
    hhs = hhs.with_columns(
        vehicles=(
            pl.col("cars").cast(pl.Int32) + pl.col("motorcycles").cast(pl.Int32)
        ),
        hh_zone=pl.col("hh_zone")
        .cast(pl.Int32)
        .replace_strict(config["hh_zone"], default="unknown"),
        hh_weight=pl.col("hh_weight").cast(pl.Float64),
        dwelling=pl.lit("unknown"),
        ownership=pl.lit("unknown"),
    ).drop("cars", "motorcycles")
    return location.add(
        hhs.with_columns(home_eod_district=district_code("home_district")),
        source="mexico",
    )


def district_code(column: str) -> pl.Expr:
    """The 3-digit EOD 2017 district code."""
    return location.clean(column, width=3, valid=r"\d{3}")


def district_zone_types(hhs: pl.DataFrame) -> dict[int, str]:
    """Classify each district by the weighted modal hh_zone of its residents.

    hh_zone comes from the dwelling's locality size (TLOC), which can vary
    within a district. Districts with no sampled residents (e.g. 888, outside
    the metro area) are left unmapped (-> unknown).
    """
    zones = (
        hhs.group_by("hh_zone", district=pl.col("home_district").cast(pl.Int32))
        .agg(pl.col("hh_weight").sum())
        .sort(
            "district", "hh_weight", "hh_zone", descending=[False, True, False]
        )
        .group_by("district", maintain_order=True)
        .first()
    )
    return dict(zip(zones["district"], zones["hh_zone"]))


def load_person_days(
    persons: pl.DataFrame, config: dict, day_mapping: dict
) -> pl.DataFrame:
    """One row per person per diary (weekday and Saturday).

    Only persons asked the travel questions (aged 6+) are kept. Each diary
    gets pid ``<ID_SOC><diary>``; persons who didn't travel that day keep a
    row so they get a stay-at-home plan.
    """
    persons = persons.filter(pl.col("weekday_travel").is_not_null())
    persons = persons.with_columns(
        pl.col("sex", "relationship", "education", "employment").cast(pl.Int32),
        age=person_age(config),
        worked_anyway=pl.col("worked_anyway").cast(pl.Int32),
        weight=pl.col("weight").cast(pl.Float32),
    ).with_columns(
        sex=pl.col("sex").replace_strict(config["sex"], default="unknown"),
        relationship=pl.col("relationship").replace_strict(
            config["relationship"], default="unknown"
        ),
        education=pl.col("education").replace_strict(
            config["education"], default="unknown"
        ),
        employment=pl.when(
            pl.col("worked_anyway").is_in(config["worked_anyway_codes"])
        )
        .then(pl.lit("employed"))
        .otherwise(
            pl.col("employment").replace_strict(
                config["employment"], default="unknown"
            )
        ),
    )
    persons = persons.with_columns(
        occupation=pl.when(pl.col("employment") == "employed")
        .then(pl.lit("unknown"))
        .otherwise(pl.lit("void")),
        *[
            pl.lit("unknown").alias(col)
            for col in ["has_licence", "disability", "can_wfh", "race"]
        ],
    ).drop("worked_anyway", "weekday_travel", "saturday_travel")

    return pl.concat(
        persons.with_columns(
            pid=pl.col("pid") + pl.lit(str(diary)), day=pl.lit(day)
        )
        for diary, day in day_mapping.items()
    )


def load_trips(
    trips: pl.DataFrame,
    stages: pl.DataFrame,
    config: dict,
    zone_types: dict[int, str],
) -> pl.DataFrame:
    """One row per trip, 0-indexed seq, times in minutes since midnight.

    Unknown (99) times are left null, so filter.null_times drops the plan.
    """
    trips = location.add_trip_ends(
        trips.with_columns(
            origin_eod_district=district_code("ozone"),
            destination_eod_district=district_code("dzone"),
        ),
        source="mexico",
    )
    trips = trips.with_columns(
        pl.col(c).cast(pl.Int32)
        for c in trips.columns
        if c not in ("pid", "ocode", "dcode")
    ).with_columns(
        pid=pl.col("pid") + pl.col("diary").cast(pl.String),
        seq=(pl.col("seq") - 1).cast(pl.Int8),
        tst=clock_minutes("tst_h", "tst_m"),
        tet=clock_minutes("tet_h", "tet_m"),
        ozone=pl.col("ozone").replace_strict(zone_types, default="unknown"),
        dzone=pl.col("dzone").replace_strict(zone_types, default="unknown"),
    )
    trips = trips.sort("pid", "seq").with_columns(activities(config))

    trips = trips.join(
        trip_modes_and_distances(trips, stages, config), on="tid", how="left"
    )
    trips = trips.drop(
        "tid", "diary", "dest_place", "tst_h", "tst_m", "tet_h", "tet_m"
    )
    return fix.day_wrap(trips)


def clock_minutes(hour: str, minute: str) -> pl.Expr:
    """hour/minute columns -> minutes since midnight, null if either is 99."""
    return (
        pl.when(
            (pl.col(hour) != UNKNOWN_TIME) & (pl.col(minute) != UNKNOWN_TIME)
        )
        .then(pl.col(hour) * 60 + pl.col(minute))
        .otherwise(None)
    )


def activities(config: dict) -> list[pl.Expr]:
    """oact and dact for trips sorted by pid and seq.

    dact comes from the trip purpose, with leisure at another dwelling
    becoming visit. A diary's first oact comes from the origin place type;
    later ones repeat the previous trip's dact.
    """
    dact = (
        pl.when(
            (pl.col("dact") == 5)
            & (pl.col("dest_place") == config["visit_place"])
        )
        .then(pl.lit("visit"))
        .otherwise(
            pl.col("dact").replace_strict(config["dact"], default="unknown")
        )
    )
    first_oact = pl.col("oact").replace_strict(
        config["oact"], default="unknown"
    )
    return [
        dact.alias("dact"),
        pl.when(pl.col("seq") == 0)
        .then(first_oact)
        .otherwise(dact.shift(1).over("pid"))
        .alias("oact"),
    ]


def trip_modes_and_distances(
    trips: pl.DataFrame, stages: pl.DataFrame, config: dict
) -> pl.DataFrame:
    """Per trip (tid): main mode, distance and access/egress distance (km).

    Stage distance is stage duration x mode speed. Stages with an unknown
    duration share whatever trip time the known stages don't account for.
    The main mode is the highest-priority mode in ``config["mode_priority"]``.
    Access/egress distance sums the non-transit stages of transit trips.
    """
    stages = stages.with_columns(
        pl.col(c).cast(pl.Int32) for c in stages.columns
    ).with_columns(
        duration=clock_minutes("duration_h", "duration_m"),
        speed=pl.col("mode").replace_strict(config["speed"]),
        mode=pl.col("mode").replace_strict(config["mode"]),
    )
    trip_duration = trips.select(
        "tid", trip_duration=(pl.col("tet") - pl.col("tst")) % 1440
    )
    stages = (
        stages.join(trip_duration, on="tid", how="left")
        .with_columns(
            unaccounted=(
                pl.col("trip_duration") - pl.col("duration").sum().over("tid")
            ).clip(lower_bound=0),
            n_unknown=pl.col("duration").null_count().over("tid"),
        )
        .with_columns(
            duration=pl.col("duration").fill_null(
                pl.col("unaccounted") / pl.col("n_unknown")
            )
        )
        .with_columns(distance=pl.col("duration") / 60 * pl.col("speed"))
    )

    priority = {mode: rank for rank, mode in enumerate(config["mode_priority"])}
    transit = pl.col("mode").is_in(["bus", "rail"])
    return stages.group_by("tid").agg(
        mode=pl.col("mode")
        .sort_by(pl.col("mode").replace_strict(priority))
        .first(),
        # null when a stage duration couldn't be imputed (unknown trip time)
        distance=pl.when(pl.col("distance").is_not_null().all())
        .then(pl.col("distance").sum())
        .cast(pl.Float32),
        access_egress_distance=pl.when(transit.any())
        .then(pl.col("distance").filter(~transit).sum())
        .otherwise(None),
    )


def access_egress(trips: pl.DataFrame) -> pl.DataFrame:
    """Mean per-person access/egress distance (km) over transit trips."""
    return (
        trips.filter(pl.col("access_egress_distance").is_not_null())
        .group_by("pid")
        .agg(pl.col("access_egress_distance").mean().cast(pl.Float32))
    )
