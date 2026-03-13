from typing import Optional

import polars as pl

from foundata import utils


def home_based(
    attributes: Optional[pl.DataFrame], trips: pl.DataFrame, on: str = "pid"
) -> tuple[Optional[pl.DataFrame], pl.DataFrame]:
    """Filter out non-home-based plans (first and last activity must be home).

    Args:
        attributes: DataFrame of plan attributes with a column matching `on`.
            If None, only trips are filtered.
        trips: DataFrame of trips with columns matching `on`, "seq", "oact", "dact".
        on: Column name to join on (default "pid").

    Returns:
        Tuple of (filtered attributes or None, filtered trips).
    """
    trips = trips.sort(on, "seq")

    n = len(attributes) if attributes is not None else len(trips.select(on).unique())

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
    if attributes is not None:
        attributes = attributes.join(
            home_based_plans, on=on, how="inner", maintain_order="left"
        )
        nn = len(attributes)
    else:
        nn = len(trips.select(on).unique())

    print(
        f"Removed {nn}/{n} plans that are not home-based ({100 * nn / n:.1f}%)"
    )
    return attributes, trips


def missing_acts_or_modes(
    attributes: Optional[pl.DataFrame], trips: pl.DataFrame
) -> tuple[Optional[pl.DataFrame], pl.DataFrame]:
    """Filter out plans with any missing or unknown activities or modes.

    Args:
        attributes: DataFrame of plan attributes. If None, only trips are filtered.
        trips: DataFrame of trips with columns "pid", "oact", "dact", "mode".

    Returns:
        Tuple of (filtered attributes or None, filtered trips).
    """
    n = len(trips.select("pid").unique())
    missing = (
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
    nn = len(missing)

    clean_trips = trips.join(
        missing, on="pid", how="anti", maintain_order="left"
    )
    clean_attributes = (
        attributes.join(missing, on="pid", how="anti", maintain_order="left")
        if attributes is not None
        else None
    )

    print(
        f"Removed {nn}/{n} plans due to missing activities or modes ({100 * nn / n:.1f}%)"
    )
    return clean_attributes, clean_trips


def negative_trips(
    attributes: Optional[pl.DataFrame], trips: pl.DataFrame, on: str = "pid"
) -> tuple[Optional[pl.DataFrame], pl.DataFrame]:
    """Filter out plans containing trips with negative durations (tst > tet).

    Args:
        attributes: DataFrame of plan attributes. If None, only trips are filtered.
        trips: DataFrame of trips with columns matching `on`, "tst", "tet".
        on: Column name to join on (default "pid").

    Returns:
        Tuple of (filtered attributes or None, filtered trips).
    """
    n = len(trips.select(on).unique())
    negative_duration_plans = (
        trips.filter((pl.col("tst") > pl.col("tet"))).select(on).unique()
    )
    nn = len(negative_duration_plans)

    clean_trips = trips.join(
        negative_duration_plans, on=on, how="anti", maintain_order="left"
    )
    clean_attributes = (
        attributes.join(negative_duration_plans, on=on, how="anti", maintain_order="left")
        if attributes is not None
        else None
    )

    print(
        f"Removed {nn}/{n} plans due to negative trip durations ({100 * nn / n:.1f}%)"
    )
    return clean_attributes, clean_trips


def negative_activities(
    attributes: Optional[pl.DataFrame], trips: pl.DataFrame, on: str = "pid"
) -> tuple[Optional[pl.DataFrame], pl.DataFrame]:
    """Filter out plans containing activities with negative durations.

    Args:
        attributes: DataFrame of plan attributes. If None, only trips are filtered.
        trips: DataFrame of trips with columns matching `on`, "tst", "tet".
        on: Column name to join on (default "pid").

    Returns:
        Tuple of (filtered attributes or None, filtered trips).
    """
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
    clean_attributes = (
        attributes.join(negative_duration_plans, on=on, how="anti", maintain_order="left")
        if attributes is not None
        else None
    )

    print(
        f"Removed {nn}/{n} plans due to negative activity durations ({100 * nn / n:.1f}%)"
    )
    return clean_attributes, clean_trips


def null_times(
    attributes: Optional[pl.DataFrame], trips: pl.DataFrame, on: str = "pid"
) -> tuple[Optional[pl.DataFrame], pl.DataFrame]:
    """Filter out plans containing trips with null start or end times.

    Args:
        attributes: DataFrame of plan attributes. If None, only trips are filtered.
        trips: DataFrame of trips with columns matching `on`, "tst", "tet".
        on: Column name to join on (default "pid").

    Returns:
        Tuple of (filtered attributes or None, filtered trips).
    """
    n = len(trips.select(on).unique())
    null_trip_pids = trips.filter(
        pl.col("tst").is_null() | pl.col("tet").is_null()
    ).select(on).unique()
    nn = len(null_trip_pids)

    clean_trips = trips.join(
        null_trip_pids, on=on, how="anti", maintain_order="left"
    )
    clean_attributes = (
        attributes.join(null_trip_pids, on=on, how="anti", maintain_order="left")
        if attributes is not None
        else None
    )

    print(
        f"Removed {nn}/{n} plans due to null trip times ({100 * nn / n:.1f}%)"
    )
    return clean_attributes, clean_trips


def null_pids(
    attributes: Optional[pl.DataFrame], trips: pl.DataFrame, on: str = "pid"
) -> tuple[Optional[pl.DataFrame], pl.DataFrame]:
    """Remove rows with null person IDs from attributes and/or trips.

    Args:
        attributes: DataFrame of plan attributes. If None, only trips are filtered.
        trips: DataFrame of trips.
        on: Column name for person ID (default "pid").

    Returns:
        Tuple of (filtered attributes or None, filtered trips).
    """
    if attributes is not None:
        n_attr = attributes[on].null_count()
        if n_attr:
            print(f"Removed {n_attr} rows with null PIDs from attributes")
            attributes = attributes.filter(pl.col(on).is_not_null())
    n_trips = trips[on].null_count()
    if n_trips:
        print(f"Removed {n_trips} rows with null PIDs from trips")
        trips = trips.filter(pl.col(on).is_not_null())
    return attributes, trips


def time_consistent(
    attributes: Optional[pl.DataFrame], trips: pl.DataFrame, on: str = "pid"
) -> tuple[Optional[pl.DataFrame], pl.DataFrame]:
    """Apply all time-consistency filters: null pids, negative trips/activities, null times.

    Args:
        attributes: DataFrame of plan attributes. If None, only trips are filtered.
        trips: DataFrame of trips.
        on: Column name to join on (default "pid").

    Returns:
        Tuple of (filtered attributes or None, filtered trips).
    """
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
