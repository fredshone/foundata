import polars as pl
import polars.selectors as cs


def trips_to_activities(
    attributes: pl.DataFrame, trips: pl.DataFrame
) -> pl.DataFrame:
    sorted_trips = trips.sort("pid", "seq")

    first_acts = (
        sorted_trips.group_by("pid")
        .agg(pl.all().first())
        .select(
            pl.col("pid"),
            pl.col("seq").cast(pl.Int8).alias("seq"),
            pl.col("oact").alias("act"),
            pl.col("ozone").alias("zone"),
            pl.lit(0, dtype=pl.Int32).alias("start"),
            pl.col("tst").alias("end"),
        )
    )

    dest_acts = (
        sorted_trips.filter(pl.col("tet") < 1440)
        .with_columns(
            end=pl.col("tst").shift(-1).over("pid").fill_null(1440),
            seq=pl.col("seq").cast(pl.Int8) + 1,
        )
        .select(
            pl.col("pid"),
            pl.col("seq").cast(pl.Int8).alias("seq"),
            pl.col("dact").alias("act"),
            pl.col("dzone").alias("zone"),
            pl.col("tst").alias("start"),
            pl.col("end").alias("end"),
        )
    )

    no_trip_acts = attributes.join(
        trips.select("pid").unique(), on="pid", how="anti"
    ).select(
        pl.col("pid"),
        pl.lit(0, dtype=pl.Int8).alias("seq"),
        pl.lit("home").alias("act"),
        pl.col("hh_zone").alias("zone"),
        pl.lit(0, dtype=pl.Int32).alias("start"),
        pl.lit(1440, dtype=pl.Int32).alias("end"),
    )

    return pl.concat([first_acts, dest_acts, no_trip_acts]).sort("pid", "seq")


def trips_with_following_activity(
    attributes: pl.DataFrame, trips: pl.DataFrame
) -> pl.DataFrame:
    return (
        trips.sort("pid", "seq")
        .with_columns(aet=pl.col("tst").shift(-1).over("pid").fill_null(1440))
        .filter(pl.col("tet") < 1440)
    )


def _bin_labels(breaks: list[float]) -> list[str]:
    def fmt(v: float) -> str:
        return str(int(v)) if v == int(v) else f"{v:g}"

    result = [f"≤{fmt(breaks[0])}"]
    for i in range(1, len(breaks)):
        result.append(f"{fmt(breaks[i - 1])}-{fmt(breaks[i])}")
    result.append(f">{fmt(breaks[-1])}")
    return result


def fill_nulls(df: pl.DataFrame, fill_value: str = "unknown") -> pl.DataFrame:
    return df.with_columns(
        [pl.col(col).fill_null(fill_value).alias(col) for col in df.columns]
    )


def discretise_numeric(
    df: pl.DataFrame,
    n_bins: int = 5,
    method: str = "quantile",
    cols: list[str] | None = None,
    exclude_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Discretise numeric columns into labelled string bins.

    Args:
        df: Input DataFrame.
        n_bins: Number of bins.
        method: "quantile" (equal-frequency) or "uniform" (equal-width).
        cols: Columns to discretise. If None, all numeric columns are used.
        exclude_cols: Columns to exclude from discretisation.
    Returns:
        DataFrame with selected numeric columns replaced by string bin labels.
        Null values are preserved as null.
    """
    if method not in ("quantile", "uniform"):
        raise ValueError(
            f"method must be 'quantile' or 'uniform', got {method!r}"
        )

    if cols is not None and exclude_cols is not None:
        raise ValueError("Cannot specify both cols and exclude_cols")
    if cols is None:
        cols = df.select(cs.numeric()).columns
    if exclude_cols is not None:
        cols = [col for col in cols if col not in exclude_cols]

    exprs = []
    for col in cols:
        non_null = df[col].drop_nulls()
        if non_null.len() == 0 or non_null.n_unique() < 2:
            continue
        if method == "quantile":
            quantiles = [i / n_bins for i in range(1, n_bins)]
            breaks = sorted({float(non_null.quantile(q)) for q in quantiles})
            if not breaks:
                continue
            labels = _bin_labels(breaks)
            exprs.append(pl.col(col).cut(breaks, labels=labels).cast(pl.String))
        else:  # uniform
            min_val = float(non_null.min())
            max_val = float(non_null.max())
            step = (max_val - min_val) / n_bins
            breaks = [min_val + i * step for i in range(1, n_bins)]
            labels = _bin_labels(breaks)
            exprs.append(pl.col(col).cut(breaks, labels=labels).cast(pl.String))

    if not exprs:
        return df
    return df.with_columns(exprs)
