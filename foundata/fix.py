import polars as pl


def fix_trips(trips: pl.DataFrame) -> pl.DataFrame:
    return (
        trips.with_columns(
            flag=pl.when(pl.col("tst") < pl.col("tet").shift(1).over("pid"))
            .then(1)
            .otherwise(0)
        )
        .with_columns(flag=pl.col("flag").cum_sum().over("pid"))
        .with_columns(tst=pl.col("tst") + pl.col("flag") * 1440)
        .drop("flag")
    )
