from typing import Optional

import polars as pl

from foundata import utils


def home_based(
    attributes: pl.DataFrame, trips: pl.DataFrame, on: str = "pid"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Filter out non-home-based plans, where the first and last activity are not home.

    Args:
        attributes: DataFrame of plan attributes, must contain a column matching `on`.
        trips: DataFrame of trips, must contain columns matching `on`, "seq", "oact", and "dact".
        on: Column name to join on (default "pid").
    """
    trips = trips.sort(on, "seq")

    n = len(attributes)

    home_based_plans = (
        trips.group_by(on)
        .agg(
            pl.col("oact").first().alias("first_act"),
            pl.col("dact").last().alias("last_act"),
        )
        .filter(
            (pl.col("first_act") == "home") & (pl.col("last_act") == "home")
        )
        .select(on)
    )

    trips = trips.join(
        home_based_plans, on=on, how="inner", maintain_order="left"
    )
    attributes = attributes.join(
        home_based_plans, on=on, how="inner", maintain_order="left"
    )
    nn = len(attributes)

    print(
        f"Removed {nn}/{n} plans that are not home-based ({100 * nn / n:.1f}%)"
    )
    return attributes, trips


def missing_acts_or_modes(
    attributes: pl.DataFrame, trips: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    n = len(trips.select("pid").unique())
    missing_acts_or_modes = (
        trips.filter(
            pl.col("oact").is_null()
            | pl.col("dact").is_null()
            | pl.col("mode").is_null()
            | (pl.col("oact") == "unknown")
            | (pl.col("dact") == "unknown")
            | (pl.col("mode") == "unknown")
        )
        .select("pid")
        .unique()
    )
    nn = len(missing_acts_or_modes)

    clean_trips = trips.join(
        missing_acts_or_modes, on="pid", how="anti", maintain_order="left"
    )
    clean_attributes = attributes.join(
        missing_acts_or_modes, on="pid", how="anti", maintain_order="left"
    )

    print(
        f"Removed {nn}/{n} plans due to missing activities or modes ({100 * nn / n:.1f}%)"
    )
    return clean_attributes, clean_trips


def negative_trips(
    attributes: pl.DataFrame, trips: pl.DataFrame, on: str = "pid"
) -> tuple[pl.DataFrame, pl.DataFrame]:

    n = len(trips.select(on).unique())
    negative_duration_plans = (
        trips.filter((pl.col("tst") > pl.col("tet"))).select(on).unique()
    )
    nn = len(negative_duration_plans)

    clean_trips = trips.join(
        negative_duration_plans, on=on, how="anti", maintain_order="left"
    )
    clean_attributes = attributes.join(
        negative_duration_plans, on=on, how="anti", maintain_order="left"
    )

    print(
        f"Removed {nn}/{n} plans due to negative trip durations ({100 * nn / n:.1f}%)"
    )
    return clean_attributes, clean_trips


def negative_activities(
    attributes: pl.DataFrame, trips: pl.DataFrame, on: str = "pid"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    n = len(trips.select(on).unique())
    negative_duration_plans = (
        trips.filter((pl.col("tst") < (pl.col("tet").shift(1)).over(on)))
        .select(on)
        .unique()
    )
    nn = len(negative_duration_plans)

    clean_trips = trips.join(
        negative_duration_plans, on=on, how="anti", maintain_order="left"
    )
    clean_attributes = attributes.join(
        negative_duration_plans, on=on, how="anti", maintain_order="left"
    )

    print(
        f"Removed {nn}/{n} plans due to negative activity durations ({100 * nn / n:.1f}%)"
    )
    return clean_attributes, clean_trips


def null_times(
    attributes: pl.DataFrame, trips: pl.DataFrame, on: str = "pid"
) -> tuple[pl.DataFrame, pl.DataFrame]:

    n = len(trips.select(on).unique())
    null_trips = trips.filter(pl.col("tst").is_null() | pl.col("tet").is_null())
    nn = len(null_trips.select(on).unique())

    clean_trips = trips.join(
        null_trips.select(on).unique(), on=on, how="anti", maintain_order="left"
    )
    clean_attributes = attributes.join(
        null_trips.select(on).unique(), on=on, how="anti", maintain_order="left"
    )

    print(
        f"Removed {nn}/{n} plans due to null trip times ({100 * nn / n:.1f}%)"
    )
    return clean_attributes, clean_trips


def null_pids(
    attributes: pl.DataFrame, trips: pl.DataFrame, on: str = "pid"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    n_attr = attributes[on].null_count()
    n_trips = trips[on].null_count()
    if n_attr:
        print(f"Removed {n_attr} rows with null PIDs from attributes")
        attributes = attributes.filter(pl.col(on).is_not_null())
    if n_trips:
        print(f"Removed {n_trips} rows with null PIDs from trips")
        trips = trips.filter(pl.col(on).is_not_null())
    return attributes, trips


def time_consistent(
    attributes: pl.DataFrame, trips: pl.DataFrame, on: str = "pid"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    attributes, trips = null_pids(attributes, trips, on)
    attributes, trips = negative_trips(attributes, trips, on)
    attributes, trips = negative_activities(attributes, trips, on)
    attributes, trips = null_times(attributes, trips, on)
    return attributes, trips


def columns(
    attributes: pl.DataFrame,
    trips: pl.DataFrame,
    template: Optional[dict] = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if template is None:
        attributes_cnfg = utils.get_template_attributes()
        trips_cnfg = utils.get_template_trips()
    else:
        attributes_cnfg = template["attributes"]
        trips_cnfg = template["trips"]

    attributes = attributes.select(attributes_cnfg.keys())
    trips = trips.select(trips_cnfg.keys())
    return attributes, trips
