from pathlib import Path

import polars as pl

from .utils import (
    config_for_year,
    fix_trips,
    sample_aus_to_euro,
    sample_int_range,
    table_joiner,
)


def default(config, year):
    return config.get(year, config["default"])


def _expand_root(root: str | Path) -> Path:
    return Path(root).expanduser()


def _bounds_from_list(bounds: list[str]) -> tuple[int, int]:
    return int(bounds[0]), int(bounds[1])


def preprocess_households(
    hhs: pl.DataFrame, config: dict, year: str
) -> pl.DataFrame:
    column_mapping = config_for_year(config["column_mappings"], year)
    income_mapping = config_for_year(config["hh_income"], year)
    ownership_mapping = config_for_year(config["ownership"], year)
    dwelling_mapping = config_for_year(config["dwelling"], year)
    zone_mapping = config_for_year(config["rurality"], year)

    hhs = hhs.select(column_mapping.keys()).rename(column_mapping)
    hhs = hhs.with_columns(
        pl.col("year").str.slice(0, 4).cast(pl.Int32).alias("year")
    )

    if year == "2012-2020":
        hhs = hhs.with_columns(
            pl.when(pl.col("hh_income").is_null())
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
            .alias("hh_income")
        )
    else:
        hhs = hhs.with_columns(
            pl.col("hh_income")
            .replace_strict(
                income_mapping, default=pl.lit((0, 0)), return_dtype=pl.List
            )
            .map_elements(
                lambda bounds: sample_aus_to_euro(bounds),
                return_dtype=pl.Float64,
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
        source=pl.lit("vista"),
        country=pl.lit("australia"),
        can_wfh=pl.lit("unknown"),
    )

    return hhs


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
                pl.Int64,
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
        pl.col("has_licence").replace_strict(has_license_mapping, default=None)
    )

    persons = persons.with_columns(
        pl.when(pl.col("anywork") == "Y")
        .then(pl.lit("employed"))
        .otherwise(
            pl.when(pl.col("studying") == "No Study")
            .then(pl.lit("unemployed"))
            .otherwise(pl.lit("education"))
        )
        .alias("employment")
    ).drop("anywork", "studying")

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
    rurality_map = config_for_year(config["rurality"], year)
    trips = trips.with_columns(
        pl.col("mode").replace_strict(mode_map),
        pl.col("oact").replace_strict(act_map),
        pl.col("dact").replace_strict(act_map),
        pl.col("ozone").replace_strict(rurality_map, default=pl.col("ozone")),
        pl.col("dzone").replace_strict(rurality_map, default=pl.col("dzone")),
    )

    return trips


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

    for year, hh_name, persons_name, trips_name in zip(
        years, hhs_names, persons_names, trips_names
    ):

        hh_columns = list(default(hh_config["column_mappings"], year).keys())

        person_columns = list(
            default(person_config["column_mappings"], year).keys()
        )

        trips_columns = list(
            default(trips_config["column_mappings"], year).keys()
        )

        print(year, ":")

        hhs = pl.read_csv(
            data_root / year / hh_name,
            columns=hh_columns,
            null_values="Missing/Refused",
        )
        hhs = preprocess_households(hhs, hh_config, year=year)

        persons = pl.read_csv(
            data_root / year / persons_name, columns=person_columns
        )
        persons = preprocess_persons(persons, person_config, year=year)

        attributes = table_joiner(hhs, persons, on="hid")

        trips = pl.read_csv(
            data_root / year / trips_name,
            columns=trips_columns,
            null_values="Missing",
        )
        trips = preprocess_trips(trips, trips_config, year=year)
        trips = fix_trips(trips)

        all_attributes.append(attributes)
        all_trips.append(trips)

    attributes = pl.concat(all_attributes)
    trips = pl.concat(all_trips)
    return attributes, trips
