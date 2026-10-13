"""Sample parquet tables by household id, keeping every column unmodified.

Used for open-data sources whose raw tables are parquet files linked by a
household id column (chile).
"""

import random
from pathlib import Path

import polars as pl


def write_parquet_subset(
    src: Path, dst: Path, ids: list[int], hid_col: str
) -> int:
    """Write the rows of src whose hid_col is in ids."""
    rows = pl.read_parquet(src).filter(pl.col(hid_col).is_in(ids))
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(dst)
    return len(rows)


def sample_chile(src_root: Path, dst_root: Path, n: int, seed: int) -> None:
    """Write a Santiago EOD 2012 sample of `n` households across its tables.

    The data is openly published, so records are copied unmodified, along
    with the code tables. The zoning geoparquet isn't copied: the loader
    reads zone areas and centroids from configs/chile/zones.csv.
    """
    from foundata.chile import (
        HOUSEHOLDS_PATH,
        PERSONS_PATH,
        STAGES_PATH,
        TRIPS_PATH,
        VEHICLES_PATH,
    )

    hids = sorted(pl.read_parquet(src_root / HOUSEHOLDS_PATH)["Hogar"])
    sampled = sorted(random.Random(seed).sample(hids, min(n, len(hids))))

    counts = {
        path: write_parquet_subset(
            src_root / path, dst_root / path, sampled, "Hogar"
        )
        for path in (
            HOUSEHOLDS_PATH,
            VEHICLES_PATH,
            PERSONS_PATH,
            TRIPS_PATH,
            STAGES_PATH,
        )
    }
    codes = "codigos.parquet"
    (dst_root / codes).write_bytes((src_root / codes).read_bytes())
    print("  " + ", ".join(f"{p}: {c}" for p, c in counts.items()))
