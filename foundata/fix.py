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


def _cast_df(df: pl.DataFrame, template: dict) -> pl.DataFrame:
    for col, cnfg in template.items():
        if col not in df.columns:
            continue
        polars_type = utils.DTYPE_MAP.get(cnfg["dtype"])
        if polars_type is None:
            continue
        df = df.with_columns(pl.col(col).cast(polars_type, strict=False))
    return df


def fix_types(
    attributes: pl.DataFrame,
    trips: pl.DataFrame,
    template_attributes: Optional[dict] = None,
    template_trips: Optional[dict] = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cast attributes and trips columns to the exact Polars dtypes defined in the template."""
    if template_attributes is None:
        template_attributes = utils.get_template_attributes()
    if template_trips is None:
        template_trips = utils.get_template_trips()

    attributes = _cast_df(attributes, template_attributes)
    trips = _cast_df(trips, template_trips)
    return attributes, trips


def missing_columns(
    attributes: pl.DataFrame,
    trips: pl.DataFrame,
    template_attributes: Optional[dict] = None,
    template_trips: Optional[dict] = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Add null columns for optional (default=True) template fields absent from data."""
    if template_attributes is None:
        template_attributes = utils.get_template_attributes()
    if template_trips is None:
        template_trips = utils.get_template_trips()

    for col, cnfg in template_attributes.items():
        if cnfg.get("default") and col not in attributes.columns:
            polars_type = utils.DTYPE_MAP.get(cnfg["dtype"])
            print(
                f"WARNING: Optional attributes column '{col}' missing — adding as null {cnfg['dtype']}"
            )
            attributes = attributes.with_columns(
                pl.lit(None).cast(polars_type).alias(col)
            )

    for col, cnfg in template_trips.items():
        if cnfg.get("default") and col not in trips.columns:
            polars_type = utils.DTYPE_MAP.get(cnfg["dtype"])
            print(
                f"WARNING: Optional trips column '{col}' missing — adding as null {cnfg['dtype']}"
            )
            trips = trips.with_columns(
                pl.lit(None).cast(polars_type).alias(col)
            )

    return attributes, trips


def unknown_to_null(df: pl.DataFrame) -> pl.DataFrame:
    """Convert 'unknown' values in string columns to null."""
    for col in df.columns:
        if df.schema[col] == pl.String:
            df = df.with_columns(
                pl.when(pl.col(col) == "unknown")
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )
    return df
