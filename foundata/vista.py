from pathlib import Path

import polars as pl

from .fix import day_wrap
from .utils import (
    config_for_year,
    sample_aus_to_euro,
    sample_int_range,
    table_joiner,
)

SOURCE = "vista"


def default(config, year):
    return config.get(year, config["default"])


def _bounds_from_list(bounds: list[str]) -> tuple[int, int]:
    return int(bounds[0]), int(bounds[1])


def load_years(
    data_root: str | Path,
    years: list[str],
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
) -> tuple[pl.DataFrame, pl.DataFrame]:

    hhs_names = [
        "households_vista_2012_2020_lga_v1.csv",
        "household_vista_2022_2023.csv",
        "household_vista_2023_2024.csv",
    ]
    persons_names = [
        "persons_vista_2012_2020_lga_v1.csv",
        "person_vista_2022_2023.csv",
        "person_vista_2023_2024.csv",
    ]

    trips_names = [
        "trips_vista_2012_2020_lga_v1.csv",
        "trips_vista_2022_2023.csv",
        "trips_vista_2023_2024.csv",
    ]

    all_attributes = []
    all_trips = []

    print("Loading VISTA data...")

    for year, hh_name, persons_name, trips_name in zip(
        years, hhs_names, persons_names, trips_names
    ):
        print(year, ":")

        hh_config_year = config_for_year(hh_config, year)
        person_config_year = config_for_year(person_config, year)
        trips_config_year = config_for_year(trips_config, year)

        hh_columns = list(hh_config_year["column_mappings"].keys())
        person_columns = list(person_config_year["column_mappings"].keys())
        trips_columns = list(trips_config_year["column_mappings"].keys())

        hhs = pl.read_csv(
            data_root / year / hh_name,
            columns=hh_columns,
            null_values="Missing/Refused",
        )
        hhs = preprocess_households(hhs, hh_config_year, year=year)

        persons = pl.read_csv(
            data_root / year / persons_name, columns=person_columns
        )
        persons = preprocess_persons(persons, person_config_year, year=year)

        attributes = table_joiner(hhs, persons, on="hid")

        trips = pl.read_csv(
            data_root / year / trips_name,
            columns=trips_columns,
            null_values="Missing",
        )
        trips = preprocess_trips(trips, trips_config_year, year=year)
        trips = day_wrap(trips)

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


def preprocess_households(
    hhs: pl.DataFrame, config: dict, year: str
) -> pl.DataFrame:

    column_mapping = config["column_mappings"]

    month_mapping = config["month"]
    income_mapping = config["hh_income"]
    ownership_mapping = config["ownership"]
    dwelling_mapping = config["dwelling"]
    zone_mapping = config["rurality"]

    hhs = hhs.select(column_mapping.keys()).rename(column_mapping)

    hhs = hhs.with_columns(
        day=pl.col("day").str.to_lowercase(),
        month=pl.col("month").replace_strict(month_mapping).cast(pl.Int8),
        year=pl.col("year").str.slice(0, 4).cast(pl.Int32),
    )

    if year == "2012-2020":
        hhs = hhs.with_columns(
            hh_income=pl.when(pl.col("hh_income").is_null())
            .then(pl.lit(None, pl.Int32))
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
            .cast(pl.Int32)
        )
    else:
        hhs = hhs.with_columns(
            hh_income=pl.col("hh_income")
            .replace_strict(
                income_mapping, default=pl.lit([0]), return_dtype=pl.List
            )
            .map_elements(
                lambda bounds: sample_aus_to_euro(bounds), return_dtype=pl.Int32
            )
        )

    hhs = hhs.with_columns(
        pl.col("ownership")
        .replace_strict(ownership_mapping)
        .fill_null("unknown"),
        pl.col("dwelling")
        .replace_strict(dwelling_mapping, default=pl.col("dwelling"))
        .fill_null("unknown"),
    )

    if year == "2012-2020":
        hhs = hhs.with_columns(
            pl.when(pl.col("wd_weight").is_null())
            .then(pl.col("we_weight"))
            .otherwise(pl.col("wd_weight"))
            .alias("weight")
        ).drop("wd_weight", "we_weight")

    hhs = hhs.with_columns(
        pl.col("rurality")
        .replace_strict(zone_mapping, default=pl.col("rurality"))
        .fill_null("unknown")
    )

    hhs = hhs.with_columns(
        source=pl.lit(SOURCE),
        country=pl.lit("australia"),
        can_wfh=pl.lit("unknown"),
    )

    return hhs


def preprocess_persons(
    persons: pl.DataFrame, config: dict, year: str
) -> pl.DataFrame:
    column_mapping = config["column_mappings"]
    sex_mapping = config["sex"]
    relationship_mapping = config["relationship"]
    has_license_mapping = config["has_licence"]
    occupation_mapping = config["occupation"]

    persons = persons.select(column_mapping.keys()).rename(column_mapping)

    if year != "2012-2020":
        persons = persons.with_columns(
            age=pl.col("age")
            .replace_strict({"100+": "100->100"}, default=pl.col("age"))
            .str.split("->")
            .map_elements(
                lambda bounds: sample_int_range(_bounds_from_list(bounds)),
                pl.Int32,
            )
        )
    else:
        persons = persons.with_columns(age=pl.col("age").cast(pl.Int32))

    persons = persons.with_columns(
        sex=pl.col("sex").replace_strict(sex_mapping)
    )

    persons = persons.with_columns(
        pl.col("relationship")
        .replace_strict(relationship_mapping)
        .alias("relationship")
    )

    persons = persons.with_columns(
        pl.col("has_licence").replace_strict(
            has_license_mapping, default="unknown"
        )
    )

    persons = persons.with_columns(
        employment=pl.when(pl.col("ft") == "Y")
        .then(pl.lit("ft-employed"))
        .when(pl.col("pt") == "Y")
        .then(pl.lit("pt-employed"))
        .when(pl.col("studying") != "No Study")
        .then(pl.lit("student"))
        .when(pl.col("activity") == "Retired")
        .then(pl.lit("retired"))
        .when(pl.col("activity") == "Unemployed")
        .then(pl.lit("unemployed"))
        .otherwise(pl.lit("other"))
    ).drop("ft", "pt", "studying", "activity")

    persons = persons.with_columns(
        pl.col("occupation")
        .replace_strict(occupation_mapping)
        .alias("occupation")
    )

    persons = persons.with_columns(
        education=pl.lit("unknown"),
        race=pl.lit("unknown"),
        disability=pl.lit("unknown"),
    )

    return persons


def preprocess_trips(
    trips: pl.DataFrame, config: dict, year: str
) -> pl.DataFrame:
    column_mapping = config["column_mappings"]
    trips = trips.select(column_mapping.keys()).rename(column_mapping)

    mode_map = config["mode_mappings"]
    act_map = config["act_mappings"]
    rurality_map = config["rurality"]
    trips = trips.with_columns(
        mode=pl.col("mode").replace_strict(mode_map),
        oact=pl.col("oact").replace_strict(act_map),
        dact=pl.col("dact").replace_strict(act_map),
        ozone=pl.col("ozone").replace_strict(
            rurality_map, default=pl.col("ozone")
        ),
        dzone=pl.col("dzone").replace_strict(
            rurality_map, default=pl.col("dzone")
        ),
    )

    return trips
