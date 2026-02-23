import polars as pl


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


def time_consistent(
    attributes: pl.DataFrame, trips: pl.DataFrame, on: str = "pid"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    attributes, trips = negative_trips(attributes, trips, on)
    attributes, trips = negative_activities(attributes, trips, on)
    return attributes, trips
