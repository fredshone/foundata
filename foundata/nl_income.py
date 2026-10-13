"""Gross household income for the Dutch sources (odin, ovin, om).

The surveys record disposable household income or personal net income in
bands. These helpers sample a value within a band and convert disposable
income to gross by rank, using the CBS decile boundaries in
configs/core/nl_income.yaml (see there for sources and method).
"""

import random

import numpy as np
import polars as pl

from foundata.utils import get_config_path, load_yaml_config


def load_config() -> dict:
    return load_yaml_config(get_config_path("core", "nl_income.yaml"))


def year_table(config: dict, year: int) -> dict:
    """The CBS table for `year`, or the nearest year available."""
    years = config["years"]
    return years[min(years, key=lambda y: (abs(y - year), y))]


def top_ratio(config: dict, year: int) -> float:
    """Gross/disposable ratio at P90, applied above it."""
    table = year_table(config, year)
    return table["gross"][-1] / table["disposable"][-1]


def disposable_cap(config: dict, year: int) -> float:
    """The disposable income that converts to max_income."""
    return config["max_income"] / top_ratio(config, year)


def to_gross(disposable: pl.Series, config: dict, year: int) -> pl.Series:
    """Convert disposable household income to gross by percentile rank,
    capped at max_income. Nulls stay null."""
    table = year_table(config, year)
    xp = np.array([0, *table["disposable"]], dtype=float)
    fp = np.array([0, *table["gross"]], dtype=float)
    x = disposable.cast(pl.Float64).to_numpy()
    gross = np.where(
        x <= xp[-1], np.interp(x, xp, fp), x * top_ratio(config, year)
    )
    gross = np.minimum(np.round(gross), config["max_income"])
    return pl.Series(disposable.name, gross, nan_to_null=True).cast(pl.Int32)


def gross_decile_bands(config: dict, year: int) -> dict[int, list]:
    """Gross income band of each household-income decile (1-10); the top
    decile is open."""
    edges = [0, *year_table(config, year)["gross"], None]
    return {k: [edges[k - 1], edges[k]] for k in range(1, 11)}


def sample_band(bounds: list | None, alpha: float, cap: float) -> int | None:
    """Sample an income within `bounds`.

    [lo, hi] is sampled uniformly. An open band [lo, None] is sampled from a
    Pareto tail with shape `alpha` starting at lo and truncated at `cap`.
    A single-element list or None means unknown.
    """
    if bounds is None or len(bounds) == 1:
        return None
    lo, hi = bounds
    if hi is not None:
        return random.randint(int(lo), int(hi))
    # inverse CDF of the truncated Pareto
    u = random.random()
    tail = 1 - (lo / cap) ** alpha
    return int(lo * (1 - u * tail) ** (-1 / alpha))
