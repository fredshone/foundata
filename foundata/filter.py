from typing import Optional

import polars as pl

from foundata import utils


def trips_on_endings(trips: pl.DataFrame, time_limit: int = 1440):
    """Filter trips that end before a specified time limit.

    Args:
        trips: DataFrame of trips with columns "pid", "oact", "dact", "tet".
        time_limit: Maximum allowed trip duration (default 1440 minutes).

    Returns:
        trips DataFrame with trips ending before the specified time limit.
    """
    n = len(trips.select("pid").unique())
    clean_trips = trips.filter(pl.col("tet") <= time_limit)
    nn = n - len(clean_trips.select("pid").unique())
    if nn > 0:
        print(
            f"Removed {nn}/{n} trips that end after {time_limit} minutes ({100 * nn / n:.1f}%)"
        )
    return clean_trips


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

    n = (
        len(attributes)
        if attributes is not None
        else len(trips.select(on).unique())
    )

    not_home_based_plans = (
        trips.group_by(on)
        .agg(
            pl.col("oact").sort_by("seq").first().alias("first_act"),
            pl.col("dact").sort_by("seq").last().alias("last_act"),
        )
        .filter(
            (pl.col("first_act") != "home") | (pl.col("last_act") != "home")
        )
        .select(on)
    )

    trips = trips.join(
        not_home_based_plans, on=on, how="anti", maintain_order="left"
    )
    if attributes is not None:
        attributes = attributes.join(
            not_home_based_plans, on=on, how="anti", maintain_order="left"
        )
        nn = n - len(attributes)
    else:
        nn = n - len(trips.select(on).unique())

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
        attributes.join(
            negative_duration_plans, on=on, how="anti", maintain_order="left"
        )
        if attributes is not None
        else None
    )

    print(
        f"Removed {nn}/{n} plans due to negative trip durations ({100 * nn / n:.1f}%)"
    )
    return clean_attributes, clean_trips


def feasible_trips(
    attributes: Optional[pl.DataFrame],
    trips: pl.DataFrame,
    on: str = "pid",
    max_distance: float = 200,
    max_duration: float = 720,
    max_speed: float = 3,
) -> tuple[Optional[pl.DataFrame], pl.DataFrame]:
    """Filter out plans containing trips long too long distances or durations or
    that are too fast.

    Args:
        attributes: DataFrame of plan attributes. If None, only trips are filtered.
        trips: DataFrame of trips with columns matching `on`, "tst", "tet".
        on: Column name to join on (default "pid").
        max_distance: Maximum allowed trip distance in km (default 200).
        max_duration: Maximum allowed trip duration in minutes (default 720).
        max_speed: Maximum allowed average speed in km/min (default 3).

    Returns:
        Tuple of (filtered attributes or None, filtered trips).
    """
    n = len(trips.select(on).unique())
    exclude = (
        trips.filter(
            (pl.col("distance") > max_distance)
            | (pl.col("tet") - pl.col("tst") > max_duration)
            | (
                (pl.col("distance") / (pl.col("tet") - pl.col("tst")))
                > max_speed
            )
        )
        .select(on)
        .unique()
    )
    nn = len(exclude.select(on).unique())
    clean_trips = trips.join(exclude, on=on, how="anti", maintain_order="left")
    clean_attributes = (
        attributes.join(exclude, on=on, how="anti", maintain_order="left")
        if attributes is not None
        else None
    )
    print(
        f"Removed {nn}/{n} plans due to infeasible trips ({100 * nn / n:.1f}%)"
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
        attributes.join(
            negative_duration_plans, on=on, how="anti", maintain_order="left"
        )
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
    null_trip_pids = (
        trips.filter(pl.col("tst").is_null() | pl.col("tet").is_null())
        .select(on)
        .unique()
    )
    nn = len(null_trip_pids)

    clean_trips = trips.join(
        null_trip_pids, on=on, how="anti", maintain_order="left"
    )
    clean_attributes = (
        attributes.join(
            null_trip_pids, on=on, how="anti", maintain_order="left"
        )
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


def trips_on_attribute_pids(
    attributes: pl.DataFrame, trips: pl.DataFrame, on: str = "pid"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Remove trips whose pid does not appear in attributes.

    Args:
        attributes: DataFrame of plan attributes.
        trips: DataFrame of trips.
        on: Column name for person ID (default "pid").

    Returns:
        Tuple of (attributes unchanged, filtered trips).
    """
    n = len(trips.select(on).unique())
    attr_pids = attributes.select(on)
    clean_trips = trips.join(
        attr_pids, on=on, how="inner", maintain_order="left"
    )
    nn = n - len(clean_trips.select(on).unique())
    if nn:
        print(
            f"Removed {nn}/{n} plans with pids not found in attributes ({100 * nn / n:.1f}%)"
        )
    return attributes, clean_trips


def activity_consistency(
    attributes: Optional[pl.DataFrame], trips: pl.DataFrame, on: str = "pid"
) -> tuple[Optional[pl.DataFrame], pl.DataFrame]:
    """Filter out plans where dact[i] != oact[i+1] for any consecutive trip pair.

    Skips null and unknown values (mirrors verify.activity_consistency logic).
    """
    n = len(trips.select(on).unique())
    inconsistent = (
        trips.sort(on, "seq")
        .with_columns(next_oact=pl.col("oact").shift(-1).over(on))
        .filter(pl.col("next_oact").is_not_null())
        # .filter(pl.col("dact") != "unknown")
        # .filter(pl.col("next_oact") != "unknown")
        .filter(pl.col("dact") != pl.col("next_oact"))
        .select(on)
        .unique()
    )
    nn = len(inconsistent)
    clean_trips = trips.join(
        inconsistent, on=on, how="anti", maintain_order="left"
    )
    clean_attributes = (
        attributes.join(inconsistent, on=on, how="anti", maintain_order="left")
        if attributes is not None
        else None
    )
    if nn:
        print(
            f"Removed {nn}/{n} plans due to activity chain inconsistencies ({100 * nn / n:.1f}%)"
        )
    return clean_attributes, clean_trips


def filter_consecutive_activities(
    attributes: Optional[pl.DataFrame],
    trips: pl.DataFrame,
    non_consecutive_types: list[str] = ["home", "work", "education"],
    on: str = "pid",
) -> tuple[Optional[pl.DataFrame], pl.DataFrame]:
    """Remove redundant trips that merge consecutive same-type activities.

    Attributes are passed through unchanged. See
    utils.combine_consecutive_acts for the row-level removal logic.
    """
    clean_trips = utils.combine_consecutive_acts(
        trips, non_consecutive_types=non_consecutive_types, on=on
    )
    return attributes, clean_trips


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
