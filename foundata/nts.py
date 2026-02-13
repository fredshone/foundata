from pathlib import Path

import polars as pl
import yaml

from .utils import (
    check_overlap,
    get_config_path,
    sample_int_range,
    sample_scaled_range,
    table_joiner,
)


def _expand_root(root: str | Path) -> Path:
    return Path(root).expanduser()


def load_households(
    root: str | Path, config_path: str | Path | None = None
) -> pl.DataFrame:
    root = _expand_root(root)
    config_path = (
        Path(config_path)
        if config_path is not None
        else get_config_path("nts", "hh_dictionary.yaml")
    )
    config = yaml.safe_load(open(config_path))
    columns = config["column_mappings"]

    hhs = pl.read_csv(
        root / "household_eul_2002-2023.tab",
        separator="\t",
        columns=list(columns.keys()),
    ).rename(columns)

    income_config = config["income"]
    hhs = hhs.with_columns(
        pl.col("income")
        .replace_strict(income_config)
        .map_elements(
            lambda bounds: sample_scaled_range(bounds, 0.15),
            return_dtype=pl.Int32,
        )
    )

    hhs = hhs.with_columns(
        pl.col("ownership").replace_strict(config["ownership"]),
        pl.col("property_type").replace_strict(config["property_type"]),
        pl.col("area").replace_strict(config["area"]),
    )

    return hhs


def load_persons(
    root: str | Path, config_path: str | Path | None = None
) -> pl.DataFrame:
    root = _expand_root(root)
    config_path = (
        Path(config_path)
        if config_path is not None
        else get_config_path("nts", "person_dictionary.yaml")
    )
    config = yaml.safe_load(open(config_path))
    columns = config["column_mappings"]

    persons = pl.read_csv(
        root / "individual_eul_2002-2023.tab",
        separator="\t",
        columns=list(columns.keys()),
    ).rename(columns)

    persons = persons.with_columns(
        pl.col("age")
        .replace_strict(config["age"])
        .map_elements(sample_int_range, return_dtype=pl.Int32),
        pl.col("gender").replace_strict(config["gender"]),
        pl.col("education").replace_strict(config["education"]),
        pl.col("license").replace_strict(config["license"]),
        pl.col("work_status").replace_strict(config["work_status"]),
        pl.col("ethnicity").replace_strict(config["ethnicity"]),
        pl.col("wfh").replace_strict(config["wfh"]),
        pl.col("mobility").replace_strict(config["mobility"]),
        pl.col("wheelchair_user").replace_strict(config["wheelchair_user"]),
    )

    return persons


def load_trips(
    root: str | Path, config_path: str | Path | None = None
) -> pl.DataFrame:
    root = _expand_root(root)
    config_path = (
        Path(config_path)
        if config_path is not None
        else get_config_path("nts", "trip_dictionary.yaml")
    )
    config = yaml.safe_load(open(config_path))
    columns = config["column_mappings"]

    trips = pl.read_csv(
        root / "trip_eul_2002-2023.tab",
        separator="\t",
        columns=list(columns.keys()),
    ).rename(columns)

    trips = trips.with_columns(
        pl.col("did").rank(method="dense").over("pid").alias("day")
    )
    trips = trips.with_columns(
        (pl.col("pid") * 100 + pl.col("day")).alias("pdid")
    )

    trips = trips.with_columns(
        pl.col("mode").replace_strict(config["mode"]),
        pl.col("oact").replace_strict(config["act"]),
        pl.col("dact").replace_strict(config["act"]),
        pl.col("tst").cast(pl.Int32, strict=False),
        pl.col("tet").cast(pl.Int32, strict=False),
        (pl.col("distance") * 1.6).alias("distance"),
    )

    trips = trips.filter(pl.col("tst").is_not_null().over("pdid"))
    trips = trips.filter(pl.col("tet").is_not_null().over("pdid"))

    trips = trips.with_columns(
        pl.when(pl.col("tet") < pl.col("tst"))
        .then(pl.col("tet") + 1440)
        .otherwise(pl.col("tet"))
        .alias("tet")
    )

    return trips.sort("hid", "pid", "tid")


def load_days(
    root: str | Path, config_path: str | Path | None = None
) -> pl.DataFrame:
    root = _expand_root(root)
    config_path = (
        Path(config_path)
        if config_path is not None
        else get_config_path("nts", "day_dictionary.yaml")
    )
    config = yaml.safe_load(open(config_path))
    columns = config["column_mappings"]

    days = pl.read_csv(
        root / "day_eul_2002-2023.tab",
        separator="\t",
        columns=list(columns.keys()),
    ).rename(columns)

    return days.with_columns(pl.col("dow").replace_strict(config["dow"]))


def build_attributes(
    persons: pl.DataFrame, households: pl.DataFrame
) -> pl.DataFrame:
    return table_joiner(persons, households, "hid")


def merge_trips_days(trips: pl.DataFrame, days: pl.DataFrame) -> pl.DataFrame:
    return table_joiner(trips, days, "did")


def check_person_trip_overlap(
    trips: pl.DataFrame, attributes: pl.DataFrame
) -> set:
    return check_overlap(trips, attributes, "pid")
