from pathlib import Path

import polars as pl

from foundata import fix
from foundata.utils import (
    check_overlap,
    compute_avg_speed,
    sample_int_range,
    sample_uk_to_euro,
    table_joiner,
)

SOURCE = "nts"


def load(
    data_root: str | Path,
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
    days_config: dict,
) -> tuple[pl.DataFrame, pl.DataFrame]:

    print("Loading NTS...")

    hhs = load_households(data_root, hh_config)
    persons = load_persons(data_root, person_config)
    attributes = table_joiner(
        hhs, persons, on="hid", lhs_name="households", rhs_name="persons"
    )

    trips = load_trips(data_root, trips_config)
    days = load_days(data_root, days_config)
    trips = table_joiner(
        trips, days, on="did", lhs_name="trips", rhs_name="days"
    )

    trips, attributes = split_days(
        trips, attributes, on_split="pdid", on_base="pid"
    )

    attributes = attributes.with_columns(
        pid=pl.lit(SOURCE) + pl.col("pid").cast(pl.String),
        hid=pl.lit(SOURCE) + pl.col("hid").cast(pl.String),
    )
    trips = trips.with_columns(
        pid=pl.lit(SOURCE) + pl.col("pid").cast(pl.String)
    )

    attributes = compute_avg_speed(attributes, trips)

    return attributes, trips


def load_households(
    root: str | Path, config: dict | None = None
) -> pl.DataFrame:

    print("loading households...")

    columns = config["column_mappings"]

    hhs = pl.read_csv(
        root / "tab" / "household_eul_2002-2023.tab",
        separator="\t",
        columns=list(columns.keys()),
    ).rename(columns)

    income_config = config["hh_income"]
    hhs = hhs.with_columns(
        pl.col("hh_income")
        .replace_strict(income_config)
        .map_elements(
            lambda bounds: sample_uk_to_euro(bounds), return_dtype=pl.Int32
        )
    )

    hhs = hhs.with_columns(
        pl.col("month").cast(pl.Int8),
        pl.col("year").cast(pl.Int32),
        pl.col("ownership").replace_strict(config["ownership"]),
        pl.col("dwelling").replace_strict(config["dwelling"]),
        pl.col("rurality").replace_strict(config["rurality"]),
        pl.lit("nts").alias("source"),
        pl.lit("uk").alias("country"),
    )

    hhs = hhs.filter(pl.col("hid").is_not_null())

    return hhs


def load_persons(root: str | Path, config: dict | None = None) -> pl.DataFrame:

    print("loading persons...")

    columns = config["column_mappings"]

    persons = pl.read_csv(
        root / "tab" / "individual_eul_2002-2023.tab",
        separator="\t",
        columns=list(columns.keys()),
    ).rename(columns)

    persons = persons.with_columns(
        pl.col("age")
        .replace_strict(config["age"])
        .map_elements(sample_int_range, return_dtype=pl.Int32),
        pl.col("sex").replace_strict(config["sex"]),
        pl.col("education").replace_strict(config["education"]),
        pl.col("has_licence").replace_strict(config["has_licence"]),
        pl.col("employment").replace_strict(config["employment"]),
        pl.col("race").replace_strict(config["race"]),
        pl.col("can_wfh").replace_strict(config["can_wfh"]),
        pl.col("disability").replace_strict(config["disability"]),
        pl.col("occupation").replace_strict(config["occupation"]),
        pl.col("relationship").replace_strict(
            config["relationship"], default=pl.col("relationship")
        ),
        # pl.col("wheelchair_user").replace_strict(config["wheelchair_user"]),
    )

    persons = persons.filter(pl.col("pid").is_not_null())

    return persons


def load_trips(root: str | Path, config: dict | None = None) -> pl.DataFrame:

    print("loading trips...")

    columns = config["column_mappings"]

    trips = pl.read_csv(
        root / "tab" / "trip_eul_2002-2023.tab",
        separator="\t",
        columns=list(columns.keys()),
    ).rename(columns)

    trips = (
        trips.with_columns(day=pl.col("did").rank(method="dense").over("pid"))
        .with_columns((pl.col("pid") * 100 + pl.col("day")).alias("pdid"))
        .drop("day")
    )

    trips = trips.with_columns(
        mode=pl.col("mode").replace_strict(config["mode"]),
        oact=pl.col("oact").replace_strict(config["act"]),
        dact=pl.col("dact").replace_strict(config["act"]),
        tst=pl.col("tst").cast(pl.Int32, strict=False),
        tet=pl.col("tet").cast(pl.Int32, strict=False),
        distance=(pl.col("distance") * 1.6).alias("distance"),
        ozone=pl.lit("unknown"),
        dzone=pl.lit("unknown"),
    )

    return trips.sort("hid", "pid", "tid")


def load_days(root: str | Path, config: dict | None = None) -> pl.DataFrame:

    columns = config["column_mappings"]

    days = pl.read_csv(
        root / "tab" / "day_eul_2002-2023.tab",
        separator="\t",
        columns=list(columns.keys()),
    ).rename(columns)

    return days.with_columns(pl.col("day").replace_strict(config["day"]))


def split_days(
    trips: pl.DataFrame,
    attributes: pl.DataFrame,
    on_split: str,
    on_base: str = "pid",
) -> pl.DataFrame:
    mapping = trips.select(on_base, on_split, "day").unique(maintain_order=True)
    trips_split = trips.drop(on_base).rename({on_split: on_base})
    attributes_expanded = (
        mapping.join(attributes, on=on_base, how="left")
        .drop(on_base)
        .rename({on_split: on_base})
    )
    check_overlap(attributes_expanded, trips_split, on=on_base)

    # also
    trips_split = trips_split.drop(["tid", "did", "day"])
    trips_split = fix.day_wrap(trips_split)

    return trips_split, attributes_expanded
