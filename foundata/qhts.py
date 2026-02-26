from pathlib import Path
from typing import Optional

import polars as pl

from .fix import day_wrap
from .utils import (
    config_for_year,
    sample_aus_to_euro,
    sample_int_range,
    table_joiner,
)

SOURCE = "qhts"


def default(config, year):
    return config.get(year, config["default"])


def load_years(
    data_root: str | Path,
    years: list[str],
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
    zones_mapping: Optional[pl.DataFrame] = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:

    all_attributes = []
    all_trips = []

    for year in years:
        print(year, ":")

        hh_config_year = config_for_year(hh_config, year)
        person_config_year = config_for_year(person_config, year)
        trips_config_year = config_for_year(trips_config, year)

        hh_columns = list(hh_config_year["column_mappings"].keys())
        person_columns = list(person_config_year["column_mappings"].keys())
        trips_columns = list(trips_config_year["column_mappings"].keys())

        hhs = pl.read_csv(
            data_root / year / "1_QTS_HOUSEHOLDS.csv",
            columns=hh_columns,
            null_values="Missing/Refused",
        )
        hhs = load_households(hhs, hh_config_year, year=year)

        persons = pl.read_csv(
            data_root / year / "2_QTS_PERSONS.csv", columns=person_columns
        )
        persons = load_persons(persons, person_config_year, year=year)

        attributes = table_joiner(hhs, persons, on="hid")

        trips = pl.read_csv(
            data_root / year / "5_QTS_TRIPS.csv",
            columns=trips_columns,
            null_values="Missing",
        )
        trips = load_trips(trips, trips_config_year, year=year)
        trips = day_wrap(trips)

        if zones_mapping is not None:
            trips = (
                trips.join(
                    zones_mapping,
                    left_on="ozone",
                    right_on="SA1_MAINCODE_2021",
                    how="left",
                )
                .drop("ozone")
                .rename({"rurality": "ozone"})
            )
            trips = (
                trips.join(
                    zones_mapping,
                    left_on="dzone",
                    right_on="SA1_MAINCODE_2021",
                    how="left",
                )
                .drop("dzone")
                .rename({"rurality": "dzone"})
            )
            trips = trips.with_columns(
                ozone=pl.col("ozone").fill_null("unknown"),
                dzone=pl.col("dzone").fill_null("unknown"),
            )

        all_attributes.append(attributes)
        all_trips.append(trips)

    attributes = pl.concat(all_attributes)
    trips = pl.concat(all_trips)

    attributes = attributes.with_columns(
        pid=pl.lit(SOURCE) + pl.col("pid").cast(pl.String),
        hid=pl.lit(SOURCE) + pl.col("hid").cast(pl.String),
    )
    trips = trips.with_columns(
        pid=pl.lit(SOURCE) + pl.col("pid").cast(pl.String)
    )

    return attributes, trips


def load_households(hhs: pl.DataFrame, config: dict, year: str) -> pl.DataFrame:
    column_mapping = config["column_mappings"]

    day_mapping = config["day"]
    dwell_mapping = config["dwelling"]
    rurality_mapping = config["rurality"]

    hhs = hhs.select(column_mapping.keys()).rename(column_mapping)

    hhs = hhs.with_columns(
        day=pl.col("day").replace_strict(day_mapping).fill_null("unknown"),
        month=pl.col("month").cast(pl.Int8),
        year=pl.col("year").cast(pl.Int32),
        dwelling=pl.col("dwelling")
        .replace_strict(dwell_mapping)
        .fill_null("unknown"),
        rurality=pl.col("rurality")
        .replace_strict(rurality_mapping, default=pl.col("rurality"))
        .fill_null("unknown"),
    )

    return hhs


def load_persons(
    persons: pl.DataFrame, config: dict, year: str
) -> pl.DataFrame:
    column_mapping = config["column_mappings"]
    persons = persons.select(column_mapping.keys()).rename(column_mapping)

    age_mapping = config["age"]
    sex_mapping = config["sex"]
    relationship_mapping = config["relationship"]
    has_licence_mapping = config["has_licence"]
    employment_mapping = config["employment"]
    occupation_mapping = config["occupation"]
    income_mapping = config["income"]
    disability_mapping = config["disability"]

    persons = persons.with_columns(
        age=pl.col("age")
        .replace_strict(age_mapping)
        .map_elements(sample_int_range, return_dtype=pl.Int32)
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
        pl.col("employment")
        .replace_strict(employment_mapping, default=pl.col("employment"))
        .fill_null("unknown")
    )

    persons = persons.with_columns(
        pl.col("occupation")
        .replace_strict(occupation_mapping)
        .fill_null("unknown")
    )

    persons = persons.with_columns(
        pl.col("disability").replace_strict(
            disability_mapping, default=pl.lit("unknown")
        )
    )

    persons = persons.with_columns(
        country=pl.lit("australia"),
        source=pl.lit("qhts"),
        race=pl.lit("unknown"),
        can_wfh=pl.lit("unknown"),
        ownership=pl.lit("unknown"),
        education=pl.lit("unknown"),
    )

    persons = persons.with_columns(
        income=pl.col("income")
        .replace_strict(
            income_mapping, default=pl.lit([0]), return_dtype=pl.List(pl.Int32)
        )
        .map_elements(
            lambda bounds: sample_aus_to_euro(bounds), return_dtype=pl.Int32
        )
    )

    persons = persons.with_columns(
        hh_income=pl.col("income").sum().over("hid")
    ).drop("income")

    return persons


def load_zone_mapping(path: str | Path) -> pl.DataFrame:

    mapping = {
        "Rural Balance": "rural",
        "Bounded Locality": "rural",
        "Other Urban": "suburban",
        "Major Urban": "urban",
    }
    zones = (
        pl.read_csv(path, columns=["SA1_MAINCODE_2021", "SOS_NAME_2021"])
        .with_columns(
            rurality=pl.col("SOS_NAME_2021")
            .replace_strict(mapping, default="unknown")
            .fill_null("unknown")
        )
        .drop("SOS_NAME_2021")
    )

    return zones


def load_trips(trips: pl.DataFrame, config: dict, year: str) -> pl.DataFrame:
    column_mapping = config["column_mappings"]
    trips = trips.select(column_mapping.keys()).rename(column_mapping)

    mode_map = config["mode_mappings"]
    act_map = config["act_mappings"]
    trips = trips.with_columns(
        pl.col("mode").replace_strict(mode_map),
        pl.col("oact").replace_strict(act_map),
        pl.col("dact").replace_strict(act_map),
    )

    return trips
