from typing import Optional

import polars as pl

from foundata import utils


def negative_duration_plans(trips: pl.DataFrame) -> bool:
    bad_trips = trips.filter(pl.col("tst") > pl.col("tet"))
    return trips.join(
        bad_trips.select("pid").unique(),
        on="pid",
        how="inner",
        maintain_order="left_right",
    )


def time_inconsistent_plans(trips: pl.DataFrame) -> bool:
    bad_trips = trips.filter(
        (pl.col("tst") < (pl.col("tet").shift(1)).over("pid"))
    )
    return trips.join(
        bad_trips.select("pid").unique(),
        on="pid",
        how="inner",
        maintain_order="left_right",
    )


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
        f"Removed {nn}/{n} plans due to negative trip durations ({100*nn/n:.1f}%)"
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
        f"Removed {nn}/{n} plans due to negative activity durations ({100*nn/n:.1f}%)"
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

    print(f"Removed {nn}/{n} plans due to null trip times ({100*nn/n:.1f}%)")
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
