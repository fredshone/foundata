import random
from pathlib import Path

import polars as pl


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
    if missing_in_b:
        print(
            f"Warning: Missing {n_b} ({perc_b:.2f}%) of '{on}' keys in table_b: {missing_in_b}"
        )

    return missing_in_a & missing_in_b


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


def config_for_year(config: dict, year):
    return config.get(year, config["default"])


def sample_int_range(bounds: tuple[int, int]) -> int:
    a, b = bounds
    return random.randint(int(a), int(b))


def sample_scaled_range(
    bounds: tuple[int, int] | None, scale: float, default: int = 0
) -> int:
    if bounds is None:
        return default
    a, b = bounds
    return int(random.randint(int(a), int(b)) * scale)


def get_config_path(*parts: str) -> Path:
    return Path(__file__).resolve().parent.parent.joinpath("configs", *parts)
