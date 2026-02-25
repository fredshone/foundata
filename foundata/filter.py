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


def time_consistent(
    attributes: pl.DataFrame, trips: pl.DataFrame, on: str = "pid"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    attributes, trips = negative_trips(attributes, trips, on)
    attributes, trips = negative_activities(attributes, trips, on)
    attributes, trips = null_times(attributes, trips, on)
    return attributes, trips


def bad_trips(trips: pl.DataFrame) -> pl.DataFrame:
    """
    Remove trips with negative duration or time inconsistencies (overlapping trips).
    But only if plan location consistency is maintained.
    Note that overlapping trip times are calculated based on the original tst and tet,
    and previous tst and tet.
    """
    before = len(trips)
    trips = trips.filter(
        ~(
            (
                (pl.col("tst") < pl.col("tet").shift(1).over("pid"))
                | (pl.col("tet") < pl.col("tet").shift(1).over("pid"))
                | (pl.col("tet") < pl.col("tst"))
            )
            & (pl.col("ozone") == pl.col("dzone").shift(1).over("pid"))
            & (pl.col("dzone") == pl.col("ozone").shift(-1).over("pid"))
        )
    )
    after = len(trips)
    removed = before - after
    perc = 100 * removed / before if before > 0 else 0
    print(
        f"Removed {removed}/{before} trips due to time inconsistencies ({perc:.1f}%)"
    )
    return trips
