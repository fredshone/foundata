from pathlib import Path

import polars as pl
import yaml

from .utils import (
    check_overlap,
    get_config_path,
    sample_int_range,
    sample_uk_to_euro,
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
        # pl.col("day").replace_strict(config["day"]),
        pl.col("ownership").replace_strict(config["ownership"]),
        pl.col("dwelling").replace_strict(config["dwelling"]),
        pl.col("rurality").replace_strict(config["rurality"]),
        pl.lit("nts").alias("source"),
        pl.lit("uk").alias("country"),
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
        root / "tab" / "trip_eul_2002-2023.tab",
        separator="\t",
        columns=list(columns.keys()),
    ).rename(columns)

    trips = (
        trips.with_columns(
            pl.col("did").rank(method="dense").over("pid").alias("day")
        )
        .with_columns((pl.col("pid") * 100 + pl.col("day")).alias("pdid"))
        .drop("day")
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

    # trips = trips.with_columns(
    #     pl.when(pl.col("tet") < pl.col("tst"))
    #     .then(pl.col("tet") + 1440)
    #     .otherwise(pl.col("tet"))
    #     .alias("tet")
    # )

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
    trips_split = fix_trips(trips_split)

    return trips_split, attributes_expanded


def fix_trips(trips: pl.DataFrame) -> pl.DataFrame:
    # fix trips that pass midnight
    return (
        trips.with_columns(
            flag=pl.when(pl.col("tst") < pl.col("tet").shift(1).over("pid"))
            .then(1)
            .otherwise(0)
        )
        .with_columns(flag=pl.col("flag").cum_sum().over("pid"))
        .with_columns(tst=pl.col("tst") + pl.col("flag") * 1440)
        .drop("flag")
    )
