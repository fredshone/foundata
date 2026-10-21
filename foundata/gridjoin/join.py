"""Join WorldTensor to the locations table, then to attributes.

``run.runner`` calls ``join_locations`` on the combined locations table
(one row per zone or point), which adds every ``wt_`` column to it, and
``to_attributes``, which copies the value columns to each household's
row by ``hh_location`` (its ``location_id``). Diagnostics (``wt_cell``,
``wt_spans_cells``, ``wt_<var>_fallback_km``) stay in the locations
table. ``join_file`` does the first step on a locations GeoParquet.
"""

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from foundata.gridjoin import worldtensor
from foundata.gridjoin import zones as zones_mod

METADATA_KEY = b"gridjoin"


def join_locations(
    locations: pl.DataFrame,
    world: Path,
    cache: Path,
    max_fallback_km: float = worldtensor.DEFAULT_MAX_FALLBACK_KM,
) -> tuple[pl.DataFrame, dict]:
    """*locations* with the ``wt_`` columns appended (rows and other
    columns unchanged), and what was used (years, units, settings).

    Preprocesses WorldTensor into *cache* first if needed.
    """
    t0 = time.perf_counter()
    worldtensor.preprocess(world, cache)
    table = _join_table(locations.to_arrow(), cache, max_fallback_km)
    meta = json.loads(table.schema.metadata[METADATA_KEY])
    logger.info(
        f"Joined WorldTensor to {locations.height} locations in "
        f"{time.perf_counter() - t0:.1f}s"
    )
    return pl.from_arrow(table), meta


def to_attributes(
    attributes: pl.DataFrame, locations: pl.DataFrame
) -> pl.DataFrame:
    """*attributes* with each WorldTensor value of its home location
    (null without one), replacing any ``wt_`` columns it already has;
    rows and their order are unchanged."""
    columns = worldtensor.value_columns()
    values = locations.select(
        pl.col("location_id").alias("hh_location"), *columns
    )
    stale = [c for c in columns if c in attributes.columns]
    return attributes.drop(stale).join(
        values, on="hh_location", how="left", maintain_order="left"
    )


def join_file(
    zones_path: str | Path,
    cache: Path,
    out_path: str | Path,
    max_fallback_km: float = worldtensor.DEFAULT_MAX_FALLBACK_KM,
) -> pa.Table:
    """Join WorldTensor (already preprocessed) to a locations GeoParquet
    and write it to *out_path*."""
    table = _join_table(pq.read_table(zones_path), cache, max_fallback_km)
    pq.write_table(table, out_path, compression="zstd")
    logger.info(f"Wrote {out_path}")
    return table


def _join_table(
    table: pa.Table, cache: Path, max_fallback_km: float
) -> pa.Table:
    zones = zones_mod.from_table(table)
    columns, meta = worldtensor.join(
        zones, cache, max_fallback_km=max_fallback_km
    )
    for col, values in columns.items():
        table = table.append_column(col, _arrow(values))
    schema_meta = dict(table.schema.metadata or {})
    schema_meta[METADATA_KEY] = json.dumps({"worldtensor": meta}).encode()
    return table.replace_schema_metadata(schema_meta)


def _arrow(values: np.ndarray) -> pa.Array:
    """Arrow array of *values*, null where NaN or masked."""
    if np.ma.isMaskedArray(values):
        return pa.array(values.data, mask=np.ma.getmaskarray(values))
    if values.dtype.kind == "f":
        return pa.array(values, mask=np.isnan(values))
    return pa.array(values)
