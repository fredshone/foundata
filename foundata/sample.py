import random
from typing import Optional

import polars as pl
import polars.selectors as cs
from loguru import logger

from foundata.post_process import discretise_numeric

MAX_STRATUM_VALUES = 5


def _draw(ids: list, k: int, rng: random.Random) -> list:
    """Draw *k* of *ids*, without replacement until every id is used."""
    reps, rem = divmod(k, len(ids))
    return ids * reps + rng.sample(ids, rem)


def strata_table(
    attributes: pl.DataFrame,
    strata: list[str],
    n_bins: int = MAX_STRATUM_VALUES,
) -> pl.DataFrame:
    """Select pid and the *strata* columns from *attributes*.

    Numeric strata with more than *n_bins* unique values are binned into
    *n_bins* quantile bins, as by `foundata bin`.
    """
    missing = [col for col in strata if col not in attributes.columns]
    if missing:
        raise ValueError(
            f"Strata column(s) {', '.join(missing)} not in attributes"
        )
    table = attributes.select("pid", *strata).unique("pid", keep="first")
    to_bin = [
        col
        for col in table.select(cs.numeric()).columns
        if col in strata and table[col].n_unique() > n_bins
    ]
    for col in to_bin:
        logger.warning(
            f"Numeric strata column '{col}' has more than {n_bins} unique "
            f"values, binning into {n_bins} categories."
        )
    if to_bin:
        table = discretise_numeric(table, n_bins=n_bins, cols=to_bin)
    return table


def sample_pids(
    pids: list,
    n: int,
    attributes: Optional[pl.DataFrame] = None,
    strata: Optional[list[str]] = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Randomly sample *n* persons from *pids*.

    Persons are drawn independently, so households are not kept whole.

    With *strata*, each stratum (unique combination of the strata columns
    in *attributes*) gets an equal share of n // n_strata persons. Persons
    are sampled without replacement, but a stratum (or, unstratified, the
    whole population) smaller than its share is oversampled with a warning.

    Returns:
        DataFrame with columns pid (original), copy and sample_pid, one row
        per sampled person in sample order. copy counts repeat draws of a
        person (0 for the first), whose sample_pid is suffixed _1, _2, ...
        so sample_pids are unique.
    """
    if n < 1:
        raise ValueError(f"Sample size must be at least 1, got {n}")
    rng = random.Random(seed)
    pids = sorted(set(pids))

    if not strata:
        if n > len(pids):
            logger.warning(
                f"Sample size {n} is larger than the {len(pids)} persons "
                "available, oversampling."
            )
        drawn = _draw(pids, n, rng)
    else:
        if attributes is None:
            raise ValueError("Stratified sampling requires attributes")
        table = strata_table(attributes, strata)
        population = pl.DataFrame({"pid": pids}).join(
            table, on="pid", how="inner"
        )
        if population.height < len(pids):
            logger.warning(
                f"{len(pids) - population.height} persons have no attributes "
                "and are excluded from sampling."
            )
        groups = (
            population.sort([*strata, "pid"], nulls_last=True)
            .group_by(strata, maintain_order=True)
            .agg("pid")
        )
        per_stratum = n // groups.height
        if per_stratum == 0:
            raise ValueError(
                f"Sample size {n} is smaller than the {groups.height} strata"
            )
        if per_stratum * groups.height != n:
            logger.info(
                f"Sampling {per_stratum} persons from each of "
                f"{groups.height} strata ({per_stratum * groups.height} total)."
            )
        drawn = []
        for row in groups.iter_rows(named=True):
            stratum_pids = row.pop("pid")
            if per_stratum > len(stratum_pids):
                logger.warning(
                    f"Stratum {row} has {len(stratum_pids)} persons, fewer "
                    f"than the {per_stratum} required, oversampling."
                )
            drawn += _draw(stratum_pids, per_stratum, rng)

    sample = pl.DataFrame({"pid": drawn}).with_columns(
        copy=pl.int_range(pl.len()).over("pid")
    )
    if (sample["copy"] > 0).any():
        sample = sample.with_columns(
            sample_pid=pl.when(pl.col("copy") == 0)
            .then(pl.col("pid").cast(pl.String))
            .otherwise(
                pl.format("{}_{}", pl.col("pid").cast(pl.String), "copy")
            )
        )
    else:
        sample = sample.with_columns(sample_pid=pl.col("pid"))
    return sample.select("pid", "copy", "sample_pid")


def reindex(
    sample: pl.DataFrame, hids: Optional[pl.DataFrame] = None
) -> pl.DataFrame:
    """Replace sample_pids with integers 0..n-1, in sample order.

    With *hids* (a pid to hid lookup), also add a sample_hid of integers
    0..n_households-1. Sampled persons sharing a hid share a sample_hid,
    except that repeat draws of a household's persons (copy 1, 2, ...)
    form a new household per copy, so no household holds the same person
    twice. Persons without a hid get a null sample_hid.
    """
    sample = sample.with_columns(sample_pid=pl.int_range(pl.len()))
    if hids is None:
        return sample
    sample = sample.join(
        hids.select("pid", "hid").unique("pid", keep="first"),
        on="pid",
        how="left",
        maintain_order="left",
    )
    households = (
        sample.select("hid", "copy")
        .drop_nulls("hid")
        .unique(maintain_order=True)
        .with_row_index("sample_hid")
        .with_columns(pl.col("sample_hid").cast(pl.Int64))
    )
    return sample.join(
        households, on=["hid", "copy"], how="left", maintain_order="left"
    ).drop("hid")


def apply_sample(df: pl.DataFrame, sample: pl.DataFrame) -> pl.DataFrame:
    """Restrict *df* to the sampled persons, repeating oversampled ones
    under their sample_pid (and sample_hid, if any), in sample order."""
    renames = {"sample_pid": "pid"}
    if "sample_hid" in sample.columns and "hid" in df.columns:
        renames["sample_hid"] = "hid"
        df_cols = [col for col in df.columns if col != "hid"]
    else:
        df_cols = df.columns
    return (
        sample.select("pid", *renames)
        .join(
            df.select(df_cols),
            on="pid",
            how="inner",
            maintain_order="left_right",
        )
        .drop("pid")
        .rename(renames)
        .select(df.columns)
    )
