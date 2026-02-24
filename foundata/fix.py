import polars as pl


def day_wrap(trips: pl.DataFrame) -> pl.DataFrame:
    trips = (
        trips.with_columns(
            flag=pl.when(pl.col("tet") < pl.col("tet").shift(1).over("pid"))
            .then(1)
            .when(pl.col("tet") < pl.col("tst"))
            .then(1)
            .otherwise(0)
        )
        .with_columns(flag=pl.col("flag").cum_sum().over("pid"))
        .with_columns(flag2=pl.col("flag").shift(1, fill_value=0).over("pid"))
        .with_columns(
            tst=pl.col("tst")
            + pl.col("flag").shift(1, fill_value=0).over("pid") * 1440,
            tet=pl.col("tet") + (pl.col("flag") * 1440),
        )
    )

    trips = trips.with_columns(
        tst=pl.when(
            pl.col("tst") < pl.col("tet").shift(1, fill_value=0).over("pid")
        )
        .then(pl.col("tst") + 1440)
        .otherwise(pl.col("tst"))
    )

    trips = null_trips(trips)

    return trips


def null_trips(trips: pl.DataFrame) -> pl.DataFrame:
    """Add additional 24 hours to trips that appear to wrap around midnight
    (i.e. tst < previous tet)
    """
    trips = (
        trips.with_columns(
            FLAG1=pl.when(pl.col("tst") < pl.col("tet").shift(1).over("pid"))
            .then(True)
            .when(pl.col("tet") < pl.col("tet").shift(1).over("pid"))
            .then(True)
            .when(pl.col("tet") < pl.col("tst"))
            .then(True)
            .otherwise(False)
        )
        .with_columns(
            FLAG2=pl.when(
                (pl.col("dzone") == pl.col("ozone").shift(-1).over("pid"))
                & pl.col("ozone")
                == pl.col("dzone").shift(1).over("pid")
            )
            .then(True)
            .otherwise(False)
        )
        .filter(~(pl.col("FLAG1") & pl.col("FLAG2")))
        .drop("FLAG1", "FLAG2")
    )
    return trips
