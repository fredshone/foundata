import polars as pl


def preprocess(trips, year, config: dict):
    column_mapping = config["column_mapping"][year]
    mode_mapping = config["mode_mappings"][year]
    act_mapping = config["act_mappings"][year]

    trips = trips.select(column_mapping.keys()).rename(column_mapping)

    trips = trips.with_columns(
        (str(year) + pl.col("hid").cast(pl.String) + pl.col("pid").cast(pl.String)).cast(pl.Int64).alias("pid")
    )

    mask = pl.any_horizontal(pl.all() < 0)
    keep = trips.group_by("pid").agg(mask.any().alias("flag")).filter(~pl.col("flag")).select("pid")
    trips = trips.join(keep, on="pid")

    def hhmm_to_minutes(i):
        hh = i // 100
        mm = i % 100
        return hh * 60 + mm

    trips = trips.with_columns(
        distance = pl.col("distance") * 1.6,
        tst = pl.col("tst").map_elements(hhmm_to_minutes),
        mode = pl.col("mode").replace_strict(mode_mapping, return_dtype=pl.String),
        oact = pl.col("oact").replace_strict(act_mapping, return_dtype=pl.String),
        dact = pl.col("dact").replace_strict(act_mapping, return_dtype=pl.String),
        ozone = 0,
        dzone = 0
    )

    trips = trips.with_columns(
        tet = (pl.col("tst") + pl.col("duration")),
    )

    return trips