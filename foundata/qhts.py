from pathlib import Path

import polars as pl
import yaml

from .utils import (
    config_for_year,
    get_config_path,
    sample_int_range,
    sample_scaled_range,
)


def _expand_root(root: str | Path) -> Path:
    return Path(root).expanduser()


def preprocess_households(
    hhs: pl.DataFrame, config: dict, year: str
) -> pl.DataFrame:
    column_mapping = config_for_year(config["column_mappings"], year)
    dwell_mapping = config_for_year(config["dwelling_type"], year)
    zone_mapping = config_for_year(config["zone"], year)

    hhs = hhs.select(column_mapping.keys()).rename(column_mapping)

    hhs = hhs.with_columns(
        pl.col("dwelling_type")
        .replace_strict(dwell_mapping)
        .fill_null("unknown")
    )

    hhs = hhs.with_columns(
        pl.col("zone")
        .replace_strict(zone_mapping, default=pl.col("zone"))
        .fill_null("unknown")
        .alias("urban/rural")
    ).drop("zone")

    return hhs.drop_nulls()


def preprocess_persons(
    persons: pl.DataFrame, config: dict, year: str
) -> pl.DataFrame:
    column_mapping = config_for_year(config["column_mappings"], year)
    persons = persons.select(column_mapping.keys()).rename(column_mapping)

    age_mapping = config_for_year(config["age"], year)
    sex_mapping = config_for_year(config["sex"], year)
    relationship_mapping = config_for_year(config["relationship"], year)
    has_licence_mapping = config_for_year(config["has_licence"], year)
    employment_mapping = config_for_year(config["employment"], year)
    occupation_mapping = config_for_year(config["occupation"], year)
    income_mapping = config_for_year(config["income"], year)

    persons = persons.with_columns(
        pl.col("age")
        .replace_strict(age_mapping)
        .map_elements(sample_int_range, pl.Float64)
        .alias("age")
    )

    persons = persons.with_columns(
        pl.col("sex").replace_strict(sex_mapping).alias("sex")
    )

    persons = persons.with_columns(
        pl.col("relationship")
        .replace_strict(relationship_mapping, default=pl.col("relationship"))
        .fill_null("self")
    )

    persons = persons.with_columns(
        pl.col("has_licence").replace_strict(has_licence_mapping)
    )

    persons = persons.with_columns(
        pl.col("employment").replace_strict(employment_mapping)
    )

    persons = persons.with_columns(
        pl.col("occupation")
        .replace_strict(occupation_mapping)
        .fill_null("unknown")
    )

    persons = persons.with_columns(
        pl.col("disability").cast(pl.Boolean).fill_null(False)
    )

    persons = persons.with_columns(
        pl.col("income")
        .replace_strict(income_mapping, default=None)
        .map_elements(
            lambda bounds: sample_scaled_range(bounds, 0.6),
            return_dtype=pl.Int32,
        )
    )

    return persons


def preprocess_trips(
    trips: pl.DataFrame, config: dict, year: str
) -> pl.DataFrame:
    column_mapping = config_for_year(config["column_mappings"], year)
    trips = trips.select(column_mapping.keys()).rename(column_mapping)

    mask = pl.any_horizontal(pl.all().is_null())
    keep = (
        trips.group_by("pid")
        .agg(mask.any().alias("flag"))
        .filter(~pl.col("flag"))
        .select("pid")
    )
    trips = trips.join(keep, on="pid")

    mode_map = config_for_year(config["mode_mappings"], year)
    act_map = config_for_year(config["act_mappings"], year)
    trips = trips.with_columns(
        pl.col("mode").replace_strict(mode_map),
        pl.col("oact").replace_strict(act_map),
        pl.col("dact").replace_strict(act_map),
    )

    return trips


def load_year(
    root: str | Path, year: str, hh_name: str, person_name: str, trips_name: str
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    root = _expand_root(root)
    hh_config = yaml.safe_load(
        open(get_config_path("qhts", "hh_dictionary.yaml"))
    )
    person_config = yaml.safe_load(
        open(get_config_path("qhts", "person_dictionary.yaml"))
    )
    trips_config = yaml.safe_load(
        open(get_config_path("qhts", "trip_dictionary.yaml"))
    )

    hh_columns = list(
        config_for_year(hh_config["column_mappings"], year).keys()
    )
    person_columns = list(
        config_for_year(person_config["column_mappings"], year).keys()
    )
    trips_columns = list(
        config_for_year(trips_config["column_mappings"], year).keys()
    )

    hhs = pl.read_csv(root / year / hh_name, columns=hh_columns)
    hhs = preprocess_households(hhs, hh_config, year=year)

    persons = pl.read_csv(root / year / person_name, columns=person_columns)
    persons = preprocess_persons(persons, person_config, year=year)

    trips = pl.read_csv(root / year / trips_name, columns=trips_columns)
    trips = preprocess_trips(trips, trips_config, year=year)

    return hhs, persons, trips
