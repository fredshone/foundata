import polars as pl


def trips_to_activities(
    attributes: pl.DataFrame, trips: pl.DataFrame
) -> pl.DataFrame:
    sorted_trips = trips.sort("pid", "seq")

    first_acts = (
        sorted_trips.group_by("pid").agg(pl.all().first())
        .select(
            pl.col("pid"),
            pl.col("seq"),
            pl.col("oact").alias("act"),
            pl.col("ozone").alias("zone"),
            pl.lit(0, dtype=pl.Int32).alias("ast"),
            pl.col("tst").alias("aet"),
        )
    )

    dest_acts = (
        sorted_trips.filter(pl.col("tet") < 1440)
        .with_columns(
            aet=pl.col("tst").shift(-1).over("pid").fill_null(1440),
            seq=pl.col("seq") + 1,
        )
        .select(
            pl.col("pid"),
            pl.col("seq"),
            pl.col("dact").alias("act"),
            pl.col("dzone").alias("zone"),
            pl.col("tet").alias("ast"),
            pl.col("aet"),
        )
    )

    no_trip_acts = attributes.join(
        trips.select("pid").unique(), on="pid", how="anti"
    ).select(
        pl.col("pid"),
        pl.lit(0, dtype=pl.Int8).alias("seq"),
        pl.lit("home").alias("act"),
        pl.col("rurality").alias("zone"),
        pl.lit(0, dtype=pl.Int32).alias("ast"),
        pl.lit(1440, dtype=pl.Int32).alias("aet"),
    )

    return pl.concat([first_acts, dest_acts, no_trip_acts]).sort("pid", "ast")


def trips_with_following_activity(
    attributes: pl.DataFrame, trips: pl.DataFrame
) -> pl.DataFrame:
    return (
        trips.sort("pid", "seq")
        .with_columns(aet=pl.col("tst").shift(-1).over("pid").fill_null(1440))
        .filter(pl.col("tet") < 1440)
    )
