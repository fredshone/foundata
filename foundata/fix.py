import polars as pl


def day_wrap(trips: pl.DataFrame) -> pl.DataFrame:
    """
    Look for trips with negative duration or time inconsistencies (overlapping trips).
    If found, add 1440 minutes to the tst and tet of the trip and all subsequent trips of the same pid.
    """
    # first consider case where trip end has moved past midnight.
    # This is identified by tet < tst.
    trips = (
        trips.with_columns(
            flag=pl.when(pl.col("tet") < pl.col("tst")).then(1).otherwise(0)
        )
        .with_columns(flag=pl.col("flag").cum_sum().over("pid"))
        .with_columns(
            tst=pl.col("tst")
            + pl.col("flag").shift(1, fill_value=0).over("pid") * 1440,
            tet=pl.col("tet") + (pl.col("flag") * 1440),
        )
    ).remove("flag")

    # also check for case where tst has moved past midnight.
    trips = (
        trips.with_columns(
            flag=pl.when(pl.col("tst") < pl.col("tet").shift(1).over("pid"))
            .then(1)
            .otherwise(0)
        )
        .with_columns(flag=pl.col("flag").cum_sum().over("pid"))
        .with_columns(
            tst=pl.col("tst") + (pl.col("flag") * 1440),
            tet=pl.col("tet") + (pl.col("flag") * 1440),
        )
    ).remove("flag")

    return trips
