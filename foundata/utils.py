import random
from pathlib import Path
from typing import Mapping

import polars as pl
import yaml


def expand_root(root: str | Path) -> Path:
    return Path(root).expanduser()


def load_yaml_config(path: str | Path) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle)


def check_overlap(table_a: pl.DataFrame, table_b: pl.DataFrame, on: str) -> set:
    on_a = set(table_a[on])
    on_b = set(table_b[on])
    missing_in_a = on_b - on_a
    missing_in_b = on_a - on_b
    n_a = len(missing_in_a)
    n_b = len(missing_in_b)
    perc_a = n_a / len(on_b) * 100
    perc_b = n_b / len(on_a) * 100

    if missing_in_a:
        print(
            f"Warning: Missing {n_a} ({perc_a:.2f}%) of '{on}' keys in table_a: {missing_in_a}"
        )
    else:
        print(f"All '{on}' keys in table_b are present in table_a")

    if missing_in_b:
        print(
            f"Warning: Missing {n_b} ({perc_b:.2f}%) of '{on}' keys in table_b: {missing_in_b}"
        )
    else:
        print(f"All '{on}' keys in table_a are present in table_b")


def table_joiner(
    table_a: pl.DataFrame, table_b: pl.DataFrame, on: str
) -> pl.DataFrame:
    # check for missing keys
    check_overlap(table_a, table_b, on)

    # check for duplicates
    a_cols = table_a.columns
    b_cols = table_b.columns
    duplicates = set(a_cols) & set(b_cols) - {on}
    if duplicates:
        print(f"Warning: Duplicate columns (other than join key): {duplicates}")

    return table_a.join(table_b, on=on)


def tables_joiner(tables: Mapping[int, pl.DataFrame], on: str) -> pl.DataFrame:
    if not tables:
        raise ValueError("No tables to join")

    result = tables[0]
    for table in tables[1:]:
        result = table_joiner(result, table, on)

    return result


def config_for_year(config: dict, year):
    return config.get(year, config["default"])


def sample_int_range(bounds: tuple[int, int]) -> int:
    a, b = bounds
    return random.randint(int(a), int(b))


def sample_us_to_euro(bounds: tuple[int, int] | None) -> int | None:
    a, b = bounds
    return int(random.randint(int(a), int(b)) / 0.85)


def get_config_path(*parts: str) -> Path:
    return Path(__file__).resolve().parent.parent.joinpath("configs", *parts)


def negative_duration_plans(trips: pl.DataFrame) -> bool:
    bad_trips = trips.filter(pl.col("tst") > pl.col("tet"))
    return trips.join(bad_trips.select("pid").unique(), on="pid", how="inner")


def time_inconsistent_plans(trips: pl.DataFrame) -> bool:
    bad_trips = trips.filter(
        (pl.col("tst") < (pl.col("tet").shift(1)).over("pid"))
    )
    return trips.join(bad_trips.select("pid").unique(), on="pid", how="inner")


def filter_time_consistent(
    attributes: pl.DataFrame, trips: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    print(
        f"Total trips: {len(trips)}, Total plans: {len(trips.select('pid').unique())}, from {len(attributes)} attributes"
    )

    negative_duration_trips = trips.filter(
        (pl.col("tst") < (pl.col("tet").shift(1)).over("pid"))
    )
    clean_trips = trips.join(
        negative_duration_trips.select("pid").unique(), on="pid", how="anti"
    )
    clean_attributes = attributes.join(
        negative_duration_trips.select("pid").unique(), on="pid", how="anti"
    )

    inconsistent_trips = trips.filter(
        (pl.col("tst") < (pl.col("tet").shift(1)).over("pid"))
    )
    clean_trips = clean_trips.join(
        inconsistent_trips.select("pid").unique(), on="pid", how="anti"
    )
    clean_attributes = attributes.join(
        inconsistent_trips.select("pid").unique(), on="pid", how="anti"
    )

    trips_removed = len(trips) - len(clean_trips)
    plans_removed = len(trips.select("pid").unique()) - len(
        clean_trips.select("pid").unique()
    )
    attributes_removed = len(attributes) - len(clean_attributes)

    print(
        f"Removed {trips_removed} trips or {plans_removed} plans and {attributes_removed} attributes due to time inconsistency"
    )

    return clean_attributes, clean_trips
