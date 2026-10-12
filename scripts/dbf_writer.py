"""Write a subset of a dBase III table's records to a new .dbf file.

The header (field descriptors) is copied from the source with only the record
count updated, and the selected records are copied byte for byte, so the
output reads exactly like the original.
"""

import struct
from pathlib import Path

import numpy as np


def write_dbf_subset(src: str | Path, dst: str | Path, rows) -> int:
    """Copy records `rows` (0-indexed, in the given order) from src to dst."""
    data = Path(src).read_bytes()
    n_records, header_len, record_len = struct.unpack("<IHH", data[4:12])
    records = np.frombuffer(
        data, dtype=np.uint8, count=n_records * record_len, offset=header_len
    ).reshape(n_records, record_len)
    subset = records[np.asarray(rows, dtype=np.int64)]

    header = bytearray(data[:header_len])
    header[4:8] = struct.pack("<I", len(subset))
    Path(dst).write_bytes(bytes(header) + subset.tobytes() + b"\x1a")
    return len(subset)


def sample_spod(src_root: Path, dst_root: Path, n: int, seed: int) -> None:
    """Write a São Paulo OD sample of `n` households, plus the zone table.

    The data is openly published, so records are copied unmodified. One
    multi-family household is always included so the family aggregation
    is exercised.
    """
    import polars as pl

    from foundata.spod import OD_PATH, ZONES_PATH, read_dbf

    ids = read_dbf(src_root / OD_PATH, columns=["ID_DOM", "TOT_FAM"])
    ids = ids.with_row_index("row")
    households = ids["ID_DOM"].unique().sort()
    multi_family = ids.filter(pl.col("TOT_FAM").cast(pl.Int32) > 1)["ID_DOM"]
    sampled = set(households.sample(n=n - 1, seed=seed)) | {multi_family[0]}
    rows = ids.filter(pl.col("ID_DOM").is_in(sampled))["row"]

    for path in (OD_PATH, ZONES_PATH):
        (dst_root / path).parent.mkdir(parents=True, exist_ok=True)
    n_rows = write_dbf_subset(src_root / OD_PATH, dst_root / OD_PATH, rows)
    (dst_root / ZONES_PATH).write_bytes((src_root / ZONES_PATH).read_bytes())
    print(f"  households: {len(sampled)}, rows: {n_rows}")


def sample_mxod(src_root: Path, dst_root: Path, n: int, seed: int) -> None:
    """Write a Mexico City EOD sample of `n` households across its tables.

    The data is openly published, so records are copied unmodified: the
    sampled hogares, their persons, those persons' trips and the trips'
    stages.
    """
    import polars as pl

    from foundata.mxod import (
        HOUSEHOLDS_PATH,
        PERSONS_PATH,
        STAGES_PATH,
        TRIPS_PATH,
    )
    from foundata.spod import read_dbf

    def rows(path: str, id_col: str, ids, child_id: str | None = None):
        """Row numbers (and child ids) of records whose id_col is in ids."""
        columns = [id_col] + ([child_id] if child_id else [])
        table = read_dbf(src_root / path, columns=columns).with_row_index("row")
        table = table.filter(pl.col(id_col).is_in(ids))
        return table["row"], table[child_id].implode() if child_id else None

    hids = read_dbf(src_root / HOUSEHOLDS_PATH, columns=["ID_HOG"])["ID_HOG"]
    sampled = hids.sample(n=n, seed=seed).implode()
    hh_rows, _ = rows(HOUSEHOLDS_PATH, "ID_HOG", sampled)
    person_rows, pids = rows(PERSONS_PATH, "ID_HOG", sampled, "ID_SOC")
    trip_rows, tids = rows(TRIPS_PATH, "ID_SOC", pids, "ID_VIA")
    stage_rows, _ = rows(STAGES_PATH, "ID_VIA", tids)

    dst_root.mkdir(parents=True, exist_ok=True)
    for path, selected in (
        (HOUSEHOLDS_PATH, hh_rows),
        (PERSONS_PATH, person_rows),
        (TRIPS_PATH, trip_rows),
        (STAGES_PATH, stage_rows),
    ):
        write_dbf_subset(src_root / path, dst_root / path, selected)
    print(
        f"  households: {len(hh_rows)}, persons: {len(person_rows)}, "
        f"trips: {len(trip_rows)}, stages: {len(stage_rows)}"
    )
