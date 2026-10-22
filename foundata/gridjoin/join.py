"""Join WorldTensor to the locations table, then to attributes.

``run.runner`` calls ``join_locations`` on the combined locations table
(one row per zone or point), which adds each location's cell
(``wt_cell``, ``wt_spans_cells``): these do not depend on the year. Then
``to_attributes`` reads the values for each distinct (``hh_location``,
``year``) of the households, from each variable's year nearest the
survey's, and copies them to each person's row (``wt_<var>``).
"""

import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
from loguru import logger

from foundata.gridjoin import worldtensor
from foundata.gridjoin import zones as zones_mod

METADATA_KEY = b"gridjoin"


def join_locations(locations: pl.DataFrame) -> pl.DataFrame:
    """*locations* with ``wt_cell`` and ``wt_spans_cells`` appended (rows
    and other columns unchanged)."""
    table = locations.to_arrow()
    zones = zones_mod.from_table(table)
    for col, values in worldtensor.cells(zones).items():
        table = table.append_column(col, _arrow(values))
    return pl.from_arrow(table)


def to_attributes(
    attributes: pl.DataFrame,
    locations: pl.DataFrame,
    world: Path,
    cache: Path,
    max_fallback_km: float = worldtensor.DEFAULT_MAX_FALLBACK_KM,
) -> tuple[pl.DataFrame, dict]:
    """*attributes* with each WorldTensor value at its home location in
    its survey year (null without either), and the year each is from,
    replacing any ``wt_`` columns it already has; rows and their order are
    unchanged. Also returns what was used (years, units, settings).

    Preprocesses the layers for the survey years into *cache* first if
    needed.
    """
    t0 = time.perf_counter()
    pairs = (
        attributes.select("hh_location", "year")
        .unique()
        .drop_nulls("hh_location")
        .join(
            locations.select(
                pl.col("location_id").alias("hh_location"), "lon", "lat"
            ),
            on="hh_location",
            how="left",
        )
    )
    years = pairs["year"].drop_nulls().unique().sort().to_list()
    worldtensor.preprocess(world, cache, years=years)
    columns, meta = worldtensor.values(
        pairs["lon"].to_numpy().astype(float),
        pairs["lat"].to_numpy().astype(float),
        pairs["year"].cast(pl.Float64).to_numpy(),
        cache,
        max_fallback_km=max_fallback_km,
    )
    values = pairs.select("hh_location", "year").with_columns(
        pl.from_arrow(pa.table({c: _arrow(v) for c, v in columns.items()}))
    )
    names = worldtensor.value_columns()
    stale = [c for c in names if c in attributes.columns]
    out = attributes.drop(stale).join(
        values.select("hh_location", "year", *names),
        on=["hh_location", "year"],
        how="left",
        maintain_order="left",
    )
    logger.info(
        f"Joined WorldTensor to {pairs.height} home locations and years "
        f"in {time.perf_counter() - t0:.1f}s"
    )
    return out, {"worldtensor": meta}


def _arrow(values: np.ndarray) -> pa.Array:
    """Arrow array of *values*, null where NaN or masked."""
    if np.ma.isMaskedArray(values):
        return pa.array(values.data, mask=np.ma.getmaskarray(values))
    if values.dtype.kind == "f":
        return pa.array(values, mask=np.isnan(values))
    return pa.array(values)
