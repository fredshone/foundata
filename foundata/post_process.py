from typing import Optional

import polars as pl
import polars.selectors as cs


def trips_to_activities(
    attributes: Optional[pl.DataFrame], trips: pl.DataFrame
) -> pl.DataFrame:
    """ "Convert trips to activities by creating an activity for each trip's origin and destination.
    The first activity of each person is created from the first trip's origin, and the last
    activity is created from the last trip's destination. Persons with no trips are assigned a single "home" activity.
    Args:
        attributes: DataFrame with person attributes, must contain "pid" and "hh_zone".
        trips: DataFrame with columns pid, seq, tst, tet, oact, dact, ozone, dzone.
    Returns:
        DataFrame with columns pid, seq, act, zone, start, end.
    """
    print("Converting trips to activities...")
    if attributes is not None:
        print("\tnumber of persons in attributes:", len(attributes))
    print("\tnumber of persons in trips:", len(trips.select("pid").unique()))

    first_acts = (
        trips.sort("pid", "seq")
        .group_by("pid")
        .agg(pl.all().first())
        .select(
            pl.col("pid"),
            pl.col("seq").cast(pl.Int8).alias("seq"),
            pl.col("oact").alias("act"),
            pl.col("ozone").alias("zone"),
            pl.lit(0, dtype=pl.Int32).alias("start"),
            pl.col("tst").alias("end").cast(pl.Int32),
        )
    )

    dest_acts = (
        trips.sort("pid", "seq")
        .with_columns(
            end=pl.col("tst")
            .shift(-1)
            .over("pid")
            .fill_null(1440)
            .cast(pl.Int32),
            seq=pl.col("seq").cast(pl.Int8) + 1,
        )
        .filter(pl.col("tet") <= 1440)
        .select(
            pl.col("pid"),
            pl.col("seq").cast(pl.Int8).alias("seq"),
            pl.col("dact").alias("act"),
            pl.col("dzone").alias("zone"),
            pl.col("tet").alias("start").cast(pl.Int32),
            pl.col("end").alias("end").cast(pl.Int32),
        )
    )

    activities = pl.concat([first_acts, dest_acts]).sort("pid", "seq")

    if attributes is not None:
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
        print(
            f"\tnumber of persons with no trips after anti join: {len(no_trip_acts.select('pid').unique())}"
        )

        activities = pl.concat([activities, no_trip_acts]).sort("pid", "seq")
    print(
        "\tnumber of persons in activities:",
        len(activities.select("pid").unique()),
    )
    return activities


def activities_to_trips(activities: pl.DataFrame) -> pl.DataFrame:
    """Convert activities to trips by pairing each activity with the next one in sequence.
    The last activity of each person is ignored, as it has no following activity to form a trip.
    This cannot recover trip modes or distances!
    Args:
        activities: DataFrame with columns pid, seq, act, zone, start, end.
    Returns:
        DataFrame with columns pid, seq, tst, tet, oact, dact, ozone, dzone.
    """
    print("Converting activities to trips...")
    print(
        "\tnumber of persons in activities:",
        len(activities.select("pid").unique()),
    )

    # filter away plans with no trips (i.e. only one activity)
    activities = activities.filter(pl.len().over("pid") > 1)

    trips = (
        activities.sort("pid", "seq")
        .with_columns(
            seq=pl.col("seq").cast(pl.Int8),
            tst=pl.col("end"),
            tet=pl.col("start").shift(-1).over("pid"),
            oact=pl.col("act"),
            dact=pl.col("act").shift(-1).over("pid"),
            ozone=pl.col("zone"),
            dzone=pl.col("zone").shift(-1).over("pid"),
        )
        .filter(
            pl.col("dact").is_not_null()
        )  # drop the last activity per pid — no trip follows it
        .select(
            pl.col("pid"),
            pl.col("seq"),
            pl.col("tst"),
            pl.col("tet"),
            pl.col("oact"),
            pl.col("dact"),
            pl.col("ozone"),
            pl.col("dzone"),
        )
    )
    print("\tnumber of persons in trips:", len(trips.select("pid").unique()))
    return trips


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
    # fill null strings with "fill value"
    string_cols = [col for col in df.columns if df[col].dtype == pl.String]
    df = df.with_columns(
        [pl.col(col).fill_null(fill_value).alias(col) for col in string_cols]
    )
    # fill empty string cells as well
    for col in string_cols:
        df = df.with_columns(
            pl.when(pl.col(col) == "")
            .then(pl.lit(fill_value))
            .otherwise(pl.col(col))
            .alias(col)
        )
    # fill null numeric values with -1
    numeric_cols = [col for col in df.columns if df[col].dtype.is_numeric()]
    df = df.with_columns(
        [pl.col(col).fill_null(-1).alias(col) for col in numeric_cols]
    )
    # cast any remaining nullable types (Boolean, Date, etc.) to String and fill
    other_cols = [col for col in df.columns if df[col].null_count() > 0]
    if other_cols:
        df = df.with_columns(
            [
                pl.col(col).cast(pl.String).fill_null(fill_value)
                for col in other_cols
            ]
        )
    # assert there are no missing values left
    assert df.null_count().sum_horizontal().sum() == 0, (
        "There are still null values in the DataFrame"
    )
    return df


def fill_unknown(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, dict]]:
    """Fill null/empty values with "unknown", returning per-column fill stats.

    Stats dict keys: pct (% filled), all_unknown (bool), appears_numeric (bool).
    Only columns with at least one fill are included in the stats dict.
    Numeric columns are cast to String before filling.
    """
    n = len(df)
    stats: dict[str, dict] = {}
    exprs = []

    for col in df.columns:
        series = df[col]
        dtype = series.dtype

        if dtype == pl.String:
            null_mask = series.is_null() | (series == "")
            null_count = int(null_mask.sum())
            if null_count == 0:
                continue
            non_empty = series.filter(~null_mask)
            if non_empty.len() == 0:
                appears_numeric = False
                all_unknown = True
            else:
                appears_numeric = bool(
                    non_empty.cast(pl.Float64, strict=False).is_not_null().all()
                )
                all_unknown = bool((non_empty == "unknown").all())
            exprs.append(
                pl.when(pl.col(col).is_null() | (pl.col(col) == ""))
                .then(pl.lit("unknown"))
                .otherwise(pl.col(col))
                .alias(col)
            )
        elif dtype.is_numeric():
            null_count = int(series.null_count())
            if null_count == 0:
                continue
            appears_numeric = True
            all_unknown = null_count == n
            exprs.append(
                pl.col(col).cast(pl.String).fill_null("unknown").alias(col)
            )
        else:
            null_count = int(series.null_count())
            if null_count == 0:
                continue
            appears_numeric = False
            all_unknown = null_count == n
            exprs.append(
                pl.col(col).cast(pl.String).fill_null("unknown").alias(col)
            )

        stats[col] = {
            "pct": 100.0 * null_count / n,
            "all_unknown": all_unknown,
            "appears_numeric": appears_numeric,
        }

    if exprs:
        df = df.with_columns(exprs)
    return df, stats


def discretise_numeric(
    df: pl.DataFrame,
    n_bins: int = 5,
    method: str = "quantile",
    cols: list[str] | None = None,
    exclude_cols: list[str] | None = None,
    per_col_bins: dict[str, int] | None = None,
) -> pl.DataFrame:
    """Discretise numeric columns into labelled string bins.

    Args:
        df: Input DataFrame.
        n_bins: Default number of bins.
        method: "quantile" (equal-frequency) or "uniform" (equal-width).
        cols: Columns to discretise. If None, all numeric columns are used.
        exclude_cols: Columns to exclude from discretisation.
        per_col_bins: Per-column bin count overrides (take precedence over n_bins).
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
        n = per_col_bins.get(col, n_bins) if per_col_bins else n_bins
        if method == "quantile":
            quantiles = [i / n for i in range(1, n)]
            breaks = sorted({float(non_null.quantile(q)) for q in quantiles})
            if not breaks:
                continue
            labels = _bin_labels(breaks)
            exprs.append(pl.col(col).cut(breaks, labels=labels).cast(pl.String))
        else:  # uniform
            min_val = float(non_null.min())
            max_val = float(non_null.max())
            step = (max_val - min_val) / n
            breaks = [min_val + i * step for i in range(1, n)]
            labels = _bin_labels(breaks)
            exprs.append(pl.col(col).cut(breaks, labels=labels).cast(pl.String))

    if not exprs:
        return df
    return df.with_columns(exprs)
