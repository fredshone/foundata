"""French household travel surveys, standardised by MobiSurvStd.

MobiSurvStd (https://mobisurvstd.github.io/MobiSurvStd/) converts the CEREMA
city surveys (EMD, EDGT, EDVM, EMC²), the national EMP 2019 and the
Île-de-France EGT 2020 into one format: a folder per survey with
households, persons, trips and legs parquet tables linked by household_id,
person_id and trip_id (1..N within each survey), plus metadata.json. Every
folder under data_root with a metadata.json is loaded as one survey (except
those in exclude_surveys: EMP 2019 by default), and
pid and hid are ``france<survey>_<raw id>``.

Only persons with is_surveyed = true reported a travel day (one day, usually
the day before the interview); the rest are dropped but still counted in
hh_size. Zone urbanity comes from the INSEE density grid of the commune.
"""

import datetime as dt
import json
import re
from pathlib import Path

import polars as pl
from loguru import logger

from foundata import fix, location
from foundata.utils import sample_to_euro

SOURCE = "france"
COUNTRY = "fr"

HOUSEHOLDS_PATH = "households.parquet"
PERSONS_PATH = "persons.parquet"
TRIPS_PATH = "trips.parquet"
LEGS_PATH = "legs.parquet"
METADATA_PATH = "metadata.json"


def survey_dirs(root: Path, exclude: list[str]) -> list[Path]:
    """Survey folders under root (those with a metadata.json), sorted."""
    return sorted(
        d
        for d in root.iterdir()
        if (d / METADATA_PATH).exists() and d.name not in exclude
    )


def read_table(path: Path, column_mapping: dict) -> pl.DataFrame:
    """Read the mapped columns of one table, renamed, enums as strings."""
    df = pl.read_parquet(path, columns=list(column_mapping)).rename(
        column_mapping
    )
    return df.with_columns(pl.col(pl.Enum, pl.Categorical).cast(pl.String))


def load(
    data_root: str | Path,
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load and normalise every MobiSurvStd survey under data_root.

    Args:
        data_root: Path to the FRANCE directory (one subfolder per survey).
        hh_config: Parsed hh_dictionary.yaml config.
        person_config: Parsed person_dictionary.yaml config.
        trips_config: Parsed trip_dictionary.yaml config.

    Returns:
        (attributes, trips) DataFrames conforming to the template schema.
    """
    logger.info(f"Loading {SOURCE} data...")
    root = Path(data_root).expanduser()
    attributes, trips = [], []
    for survey in survey_dirs(root, hh_config.get("exclude_surveys") or []):
        a, t = load_survey(survey, hh_config, person_config, trips_config)
        attributes.append(a)
        trips.append(t)
    return pl.concat(attributes), pl.concat(trips)


def load_survey(
    survey: Path,
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load one survey folder; ids are prefixed with the folder name."""
    trip_mappings = trips_config["column_mappings"]
    raw_trips = read_table(survey / TRIPS_PATH, trip_mappings["trips"])
    legs = read_table(survey / LEGS_PATH, trip_mappings["legs"])
    metadata = json.loads((survey / METADATA_PATH).read_text())

    hhs = load_households(
        read_table(survey / HOUSEHOLDS_PATH, hh_config["column_mappings"]),
        raw_trips,
        hh_config,
    )
    persons = load_persons(
        read_table(survey / PERSONS_PATH, person_config["column_mappings"]),
        person_config,
    )
    person_days = raw_trips.group_by("pid").agg(
        person_day=pl.col("weekday").sort_by("seq").first(),
        person_date=pl.col("date").sort_by("seq").first(),
    )
    date = pl.coalesce("person_date", "hh_date")

    prefix = f"{SOURCE}{survey.name}_"
    attributes = (
        persons.join(hhs, on="hid", how="left")
        .join(person_days, on="pid", how="left")
        .join(access_egress(legs, trips_config), on="pid", how="left")
        .with_columns(
            pid=pl.lit(prefix) + pl.col("pid").cast(pl.String),
            hid=pl.lit(prefix) + pl.col("hid").cast(pl.String),
            source=pl.lit(SOURCE),
            country=pl.lit(COUNTRY),
            race=pl.lit("unknown"),
            day=pl.coalesce("person_day", "hh_day").fill_null("unknown"),
            year=date.dt.year()
            .fill_null(fallback_year(metadata))
            .cast(pl.Int32),
            month=date.dt.month().cast(pl.Int8),
            # each survey's weights sum to its own area's population, so
            # they are normalised within the survey before combining
            weight=(pl.col("weight") / pl.col("weight").mean()).cast(
                pl.Float32
            ),
        )
        .pipe(
            location.add,
            hh_config["home_location"],
            source=SOURCE,
            year=survey_year(survey.name, metadata),
            survey=survey.name,
        )
        .drop(
            "person_day",
            "hh_day",
            "person_date",
            "hh_date",
            "home_detailed_zone",
            "point_lng",
            "point_lat",
            "home_insee",
        )
    )
    # a few persons have trips but is_surveyed = false: their trips go too
    trips = (
        load_trips(
            raw_trips,
            hh_config,
            trips_config,
            survey=survey.name,
            year=survey_year(survey.name, metadata),
        )
        .with_columns(pid=pl.lit(prefix) + pl.col("pid").cast(pl.String))
        .join(attributes.select("pid"), on="pid", how="semi")
    )
    return attributes, trips


def fallback_year(metadata: dict) -> int:
    """Year of the midpoint of the survey period, for persons with no date."""
    start = dt.date.fromisoformat(metadata["start_date"])
    end = dt.date.fromisoformat(metadata["end_date"])
    return (start + (end - start) / 2).year


def density_zone(col: str, config: dict) -> pl.Expr:
    """INSEE density grid level (1-7) -> urban/suburban/rural/unknown."""
    return (
        pl.col(col)
        .cast(pl.Int32)
        .replace_strict(config["hh_zone"], default="unknown")
    )


def shared(col: str) -> pl.Expr:
    """The group's value of col if all its non-null values agree, else null."""
    return (
        pl.when(pl.col(col).drop_nulls().n_unique() == 1)
        .then(pl.col(col).drop_nulls().first())
        .alias(col)
    )


def load_households(
    hhs: pl.DataFrame, trips: pl.DataFrame, config: dict
) -> pl.DataFrame:
    """One row per household with zone, vehicles, income and travel day.

    The household travel day (for members without trips) is trips_weekday,
    else the weekday shared by all the household's trips. Its date is the
    date shared by all its trips, else the day before the interview.
    """
    trip_days = trips.group_by("hid").agg(shared("weekday"), shared("date"))
    return hhs.join(trip_days, on="hid", how="left").select(
        "hid",
        hh_size=pl.col("hh_size").cast(pl.Int32),
        dwelling=pl.col("dwelling").replace_strict(
            config["dwelling"], default="unknown"
        ),
        ownership=pl.col("ownership").replace_strict(
            config["ownership"], default="unknown"
        ),
        vehicles=(pl.col("cars") + pl.col("motorcycles").fill_null(0)).cast(
            pl.Int32
        ),
        hh_zone=density_zone("hh_zone", config),
        hh_income=income(config),
        hh_day=pl.coalesce("hh_weekday", "weekday"),
        hh_date=pl.coalesce(
            "date", pl.col("interview_date") - dt.timedelta(days=1)
        ),
        home_detailed_zone="home_detailed_zone",
        point_lng="point_lng",
        point_lat="point_lat",
        home_insee="home_insee",
    )


def survey_year(name: str, metadata: dict | None = None) -> int:
    """The year in a survey folder's name (its last 4-digit number, e.g.
    ``toulouse_2013``, ``EGT2020``), else the survey period's midpoint."""
    if years := re.findall(r"\d{4}", name):
        return int(years[-1])
    if metadata is None:
        raise ValueError(f"no year in survey name {name!r}")
    return fallback_year(metadata)


def income(config: dict) -> pl.Expr:
    """Annual EUR sampled uniformly within the monthly income band x 12.

    The open top band is capped at open_band_upper; no band gives null.
    """
    upper = pl.col("income_upper").fill_null(config["open_band_upper"])
    bounds = pl.concat_list(
        pl.col("income_lower").cast(pl.Int64) * 12, upper.cast(pl.Int64) * 12
    )
    rate = config["eur_rate"]
    return (
        pl.when(pl.col("income_lower").is_not_null())
        .then(bounds)
        .map_elements(
            lambda b: sample_to_euro(b.to_list(), rate),
            return_dtype=pl.Int32,
            skip_nulls=True,
        )
    )


def load_persons(persons: pl.DataFrame, config: dict) -> pl.DataFrame:
    """One row per person surveyed for trips."""
    persons = persons.filter("is_surveyed").with_columns(
        employment=pl.col("employment").replace_strict(
            config["employment"], default="unknown"
        )
    )
    worker = pl.col("employment").str.ends_with("employed")
    category = pl.coalesce("pcs2020", "pcs2003").cast(pl.Int32)
    occupation = pl.coalesce(
        category.replace_strict(
            config["occupation"], default=None, return_dtype=pl.String
        ),
        pl.col("pcs_group")
        .cast(pl.Int32)
        .replace_strict(
            config["occupation_group"], default=None, return_dtype=pl.String
        ),
    ).fill_null("unknown")
    can_wfh = (
        pl.when(pl.col("work_only_at_home"))
        .then(pl.lit("yes"))
        .otherwise(
            pl.col("telework").replace_strict(
                config["can_wfh"], default="unknown"
            )
        )
    )
    return persons.select(
        "pid",
        "hid",
        "employment",
        age=pl.col("age").cast(pl.Int32),
        weight=pl.col("weight").cast(pl.Float64),
        sex=pl.col("sex").replace_strict(config["sex"], default="unknown"),
        relationship=pl.col("relationship").replace_strict(
            config["relationship"], default="unknown"
        ),
        education=pl.col("education").replace_strict(
            config["education"], default="unknown"
        ),
        occupation=pl.when(worker).then(occupation).otherwise(pl.lit("void")),
        can_wfh=pl.when(worker).then(can_wfh).otherwise(pl.lit("void")),
        has_licence=pl.col("has_licence").replace_strict(
            config["has_licence"], default="unknown"
        ),
        disability=pl.col("disability").replace_strict(
            config["disability"], default="unknown"
        ),
    )


def load_trips(
    trips: pl.DataFrame,
    hh_config: dict,
    config: dict,
    survey: str | None = None,
    year: int | None = None,
) -> pl.DataFrame:
    """One row per trip, 0-indexed seq, times in minutes since midnight,
    with origin and destination locations (see trip_dictionary.yaml)."""
    trips = location.add_trip_ends(
        trips, config, source=SOURCE, year=year, survey=survey
    )
    trips = trips.sort("pid", "seq").with_columns(
        seq=(pl.col("seq") - 1).cast(pl.Int8),
        oact=pl.col("oact").replace_strict(config["act"], default="unknown"),
        dact=pl.col("dact").replace_strict(config["act"], default="unknown"),
        mode=pl.col("mode").replace_strict(config["mode"], default="unknown"),
        ozone=density_zone("ozone", hh_config),
        dzone=density_zone("dzone", hh_config),
        tst=pl.col("tst").cast(pl.Int32),
        tet=pl.col("tet").cast(pl.Int32),
        distance=pl.coalesce(
            "travel_distance",
            pl.col("euclidean_distance") * config["circuity"],
        ).cast(pl.Float32),
    )
    trips = fix.day_wrap(trips)
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
        *location.columns("origin"),
        *location.columns("destination"),
    )


def access_egress(legs: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Mean per-person walking distance (km) over their transit trips.

    A walking leg's distance is its travel distance, else its euclidean
    distance x circuity, else its travel time x walk_speed. Transit trips
    with no walking leg are left out rather than counted as 0 km, as in
    ktdb.
    """
    walk_km = pl.coalesce(
        "leg_travel_distance",
        pl.col("leg_euclidean_distance") * config["circuity"],
        pl.col("leg_travel_time") / 60 * config["walk_speed"],
    )
    walking = pl.col("leg_mode_group") == "walking"
    per_trip = (
        legs.group_by("pid", "trip_id")
        .agg(
            transit=(pl.col("leg_mode_group") == "public_transit").any(),
            walked=walking.any(),
            known=walk_km.filter(walking).is_not_null().all(),
            km=walk_km.filter(walking).sum(),
        )
        .filter("transit", "walked", "known")
    )
    return per_trip.group_by("pid").agg(
        access_egress_distance=pl.col("km").mean().cast(pl.Float32)
    )
