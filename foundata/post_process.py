import polars as pl


def trips_to_activities(trips: pl.DataFrame) -> pl.DataFrame:
    return (
        trips.sort("pid", "seq")
        .with_columns(aet=pl.col("tst").shift(-1).over("pid"))
        .filter(pl.col("aet").is_not_null())
        .select(
            pl.col("pid"),
            pl.col("seq"),
            pl.col("dact").alias("act"),
            pl.col("dzone").alias("zone"),
            pl.col("tet").alias("ast"),
            pl.col("aet"),
        )
    )
