"""Sample delimited text tables by household id, copying lines byte for byte.

Used for open-data sources whose raw tables are CSVs linked by a household
id in the first column (urug), so the sample keeps the original encoding,
BOM, delimiter and line endings.
"""

import random
from pathlib import Path


def _first_field(line: bytes, sep: bytes) -> bytes:
    return line.split(sep, 1)[0].strip()


def write_csv_subset(src: Path, dst: Path, ids: set[bytes], sep: bytes) -> int:
    """Copy the header and the lines whose first field is in ids."""
    lines = src.read_bytes().splitlines(keepends=True)
    kept = [line for line in lines[1:] if _first_field(line, sep) in ids]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(lines[0] + b"".join(kept))
    return len(kept)


def sample_urug(src_root: Path, dst_root: Path, n: int, seed: int) -> None:
    """Write a Montevideo EOD 2016 sample of `n` households across its tables.

    The data is openly published, so records are copied unmodified, along
    with the codebooks.
    """
    from foundata.urug import (
        HOUSEHOLDS_PATH,
        PERSONS_PATH,
        STAGES_PATH,
        SURVEY_DIR,
        TRIPS_PATH,
    )

    src, dst = src_root / SURVEY_DIR, dst_root / SURVEY_DIR
    sep = b";"
    hh_lines = (src / HOUSEHOLDS_PATH).read_bytes().splitlines()[1:]
    hids = sorted({_first_field(line, sep) for line in hh_lines})
    sampled = set(random.Random(seed).sample(hids, min(n, len(hids))))

    counts = {
        path: write_csv_subset(src / path, dst / path, sampled, sep)
        for path in (HOUSEHOLDS_PATH, PERSONS_PATH, TRIPS_PATH, STAGES_PATH)
    }
    for codebook in src.glob("Dicc *.txt"):
        (dst / codebook.name).write_bytes(codebook.read_bytes())
    print("  " + ", ".join(f"{p}: {c}" for p, c in counts.items()))
