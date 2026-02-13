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


def _bounds_from_list(bounds: list[str]) -> tuple[int, int]:
    return int(bounds[0]), int(bounds[1])


def preprocess_households(
    hhs: pl.DataFrame, config: dict, year: str
) -> pl.DataFrame:
    column_mapping = config_for_year(config["column_mappings"], year)
    income_mapping = config_for_year(config["hh_income"], year)
    ownership_mapping = config_for_year(config["home_ownership"], year)
    zone_mapping = config_for_year(config["zone"], year)

    hhs = hhs.select(column_mapping.keys()).rename(column_mapping)
    hhs = hhs.with_columns(
        pl.col("year").str.slice(0, 4).cast(pl.Int32).alias("year")
    )

    if year == "2012-2020":
        hhs = hhs.with_columns(
            pl.when(pl.col("hh_income").is_null())
            .then(0)
            .otherwise(
                (
                    pl.col("hh_income")
                    .str.slice(1)
                    .str.replace_all(",", "")
                    .cast(pl.Int32)
                )
                * 52
                * 0.6
            )
            .alias("hh_income")
        )
    else:
        hhs = hhs.with_columns(
            pl.col("hh_income")
            .replace_strict(
                income_mapping, default=pl.lit((0, 0)), return_dtype=pl.List
            )
            .map_elements(
                lambda bounds: sample_scaled_range(bounds, 0.6),
                return_dtype=pl.Float64,
            )
        )

    hhs = hhs.with_columns(
        pl.col("home_ownership")
        .replace_strict(ownership_mapping)
        .fill_null("unknown")
        .alias("home_ownership")
    )

    hhs = hhs.with_columns(
        pl.col("zone")
        .replace_strict(zone_mapping, default=pl.col("zone"))
        .fill_null("unknown")
        .alias("urban/rural")
    ).drop("zone")

    if year == "2012-2020":
        hhs = hhs.with_columns(
            pl.when(pl.col("wd_weight").is_null())
            .then(pl.col("we_weight"))
            .otherwise(pl.col("wd_weight"))
            .alias("weight")
        ).drop("wd_weight", "we_weight")

    return hhs.drop_nulls()


def preprocess_persons(
    persons: pl.DataFrame, config: dict, year: str
) -> pl.DataFrame:
    column_mapping = config_for_year(config["column_mappings"], year)
    sex_mapping = config_for_year(config["sex"], year)
    relationship_mapping = config_for_year(config["relationship"], year)
    has_license_mapping = config_for_year(config["has_licence"], year)
    occupation_mapping = config_for_year(config["occupation"], year)

    persons = persons.select(column_mapping.keys()).rename(column_mapping)

    if year != "2012-2020":
        persons = persons.with_columns(
            pl.col("age")
            .replace_strict({"100+": "100->100"}, default=pl.col("age"))
            .str.split("->")
            .map_elements(
                lambda bounds: sample_int_range(_bounds_from_list(bounds)),
                pl.Float64,
            )
            .alias("age")
        )

    persons = persons.with_columns(
        pl.col("sex").replace_strict(sex_mapping).alias("sex")
    )

    persons = persons.with_columns(
        pl.col("relationship")
        .replace_strict(relationship_mapping)
        .alias("relationship")
    )

    persons = persons.with_columns(
        pl.col("has_license")
        .replace_strict(has_license_mapping, default=None)
        .alias("has_license")
    )

    persons = persons.with_columns(
        pl.when(pl.col("anywork") == "Y")
        .then(pl.lit("employed"))
        .otherwise(
            pl.when(pl.col("studying") == "No Study")
            .then(pl.lit("unemployed"))
            .otherwise(pl.lit("education"))
        )
        .alias("employment_status")
    ).drop("anywork", "studying")

    persons = persons.with_columns(
        pl.col("occupation")
        .replace_strict(occupation_mapping)
        .alias("occupation")
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
        open(get_config_path("vista", "hh_dictionary.yaml"))
    )
    person_config = yaml.safe_load(
        open(get_config_path("vista", "person_dictionary.yaml"))
    )
    trips_config = yaml.safe_load(
        open(get_config_path("vista", "trip_dictionary.yaml"))
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

    hhs = pl.read_csv(
        root / year / hh_name, columns=hh_columns, null_values="Missing/Refused"
    )
    hhs = preprocess_households(hhs, hh_config, year=year)

    persons = pl.read_csv(root / year / person_name, columns=person_columns)
    persons = preprocess_persons(persons, person_config, year=year)

    trips = pl.read_csv(
        root / year / trips_name, columns=trips_columns, null_values="Missing"
    )
    trips = preprocess_trips(trips, trips_config, year=year)

    return hhs, persons, trips
