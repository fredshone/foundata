from typing import Optional

import polars as pl

from foundata import utils


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


_DTYPE_MAP = {
    "int8": pl.Int8, "int16": pl.Int16,
    "int32": pl.Int32, "int64": pl.Int64,
    "float32": pl.Float32, "float64": pl.Float64,
    "int": pl.Int32, "integer": pl.Int32,
    "float": pl.Float32,
    "string": pl.String, "str": pl.String,
    "bool": pl.Boolean, "boolean": pl.Boolean,
    "date": pl.Date, "datetime": pl.Datetime,
}


def _cast_df(df: pl.DataFrame, template: dict) -> pl.DataFrame:
    for col, cnfg in template.items():
        if col not in df.columns:
            continue
        polars_type = _DTYPE_MAP.get(cnfg["dtype"])
        if polars_type is None:
            continue
        df = df.with_columns(pl.col(col).cast(polars_type, strict=False))
    return df


def fix_types(
    attributes: pl.DataFrame, trips: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cast attributes and trips columns to the exact Polars dtypes defined in the template."""
    attributes = _cast_df(attributes, utils.get_template_attributes())
    trips = _cast_df(trips, utils.get_template_trips())
    return attributes, trips


def trip_dtypes(
    all_trips: list[pl.DataFrame], template_cnfg: Optional[dict] = None
) -> pl.DataFrame:
    if not template_cnfg:
        template_cnfg = utils.get_template_trips()

    return [_cast_df(trips, template_cnfg) for trips in all_trips]
