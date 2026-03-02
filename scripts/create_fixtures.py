#!/usr/bin/env python3
"""Create anonymised test fixtures from real LTDS and NHTS data.

Usage:
    uv run python scripts/create_fixtures.py

Reads from /home/fred/Data/foundata/{LTDS,NHTS}/ (override with env vars
FOUNDATA_LTDS_DATA and FOUNDATA_NHTS_DATA).
Writes to tests/fixtures/.

Anonymisation: per-column independent shuffle of sensitive fields (seed=42).
This preserves marginal distributions but breaks cross-column correlations.
Join keys (IDs) are never shuffled.
"""
import os
import shutil
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures"
N_HH = 100
SEED = 42

LTDS_DATA_DEFAULT = "/home/fred/Data/foundata/LTDS"
NHTS_DATA_DEFAULT = "/home/fred/Data/foundata/NHTS"

LTDS_HH_SHUFFLE = ["hincome", "hincomei", "hstruct", "hstruct6"]
LTDS_PERSON_SHUFFLE = ["pagei", "pagecat", "pegroup", "psex", "prelrsp", "prelrspi"]
LTDS_PERSON_DATA_SHUFFLE = ["pdlcar", "pwkstat", "poccupa", "pawfh"]
NHTS_HH_SHUFFLE = ["HHFAMINC", "HHFAMINC_IMP", "HH_RACE", "HOMEOWN"]
NHTS_PERSON_SHUFFLE = [
    "R_AGE",
    "R_SEX",
    "R_RELAT",
    "EDUC",
    "MEDCOND",
    "PRMACT",
    "WORKER",
    "DRIVER",
    "R_RACE",
    "R_RACE_IMP",
]


def shuffle_cols(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Shuffle each column independently (only if column exists in df)."""
    return df.with_columns(
        pl.col(c).shuffle(seed=SEED) for c in cols if c in df.columns
    )


def create_ltds_fixture(data_root: Path, out_root: Path) -> None:
    year_dir = "LTDS2425"
    src = data_root / year_dir
    dst = out_root / "ltds" / year_dir
    dst.mkdir(parents=True, exist_ok=True)

    # 1. Household: sample N_HH rows
    hhs = pl.read_csv(src / "Household.csv", ignore_errors=True)
    hhs = hhs.sample(n=min(N_HH, len(hhs)), seed=SEED, shuffle=True)
    hhid_set = set(hhs["hhid"].to_list())
    hhs = shuffle_cols(hhs, LTDS_HH_SHUFFLE)
    hhs.write_csv(dst / "Household.csv")

    # 2. person.csv: filter by phid ∈ hhid_set
    persons = pl.read_csv(src / "person.csv", ignore_errors=True)
    persons = persons.filter(pl.col("phid").is_in(hhid_set))
    ppid_set = set(persons["ppid"].to_list())
    persons = shuffle_cols(persons, LTDS_PERSON_SHUFFLE)
    persons.write_csv(dst / "person.csv")

    # 3. Person_data.csv: filter by ppid ∈ ppid_set
    person_data_path = src / "person data.csv"
    person_data = pl.read_csv(person_data_path, ignore_errors=True)
    person_data = person_data.filter(pl.col("ppid").is_in(ppid_set))
    person_data = shuffle_cols(person_data, LTDS_PERSON_DATA_SHUFFLE)
    person_data.write_csv(dst / "person data.csv")

    # 4. Trip.csv: filter by thid ∈ hhid_set
    trips = pl.read_csv(src / "Trip.csv", ignore_errors=True)
    trips = trips.filter(pl.col("thid").is_in(hhid_set))
    trips.write_csv(dst / "Trip.csv")

    # 5. Stage.csv: filter by spid ∈ ppid_set
    stages = pl.read_csv(src / "Stage.csv", ignore_errors=True)
    stages = stages.filter(pl.col("spid").is_in(ppid_set))
    stages.write_csv(dst / "Stage.csv")

    # 6. HABORO_T.csv: copy unchanged (zone lookup, not personal data)
    shutil.copy(src / "HABORO_T.csv", dst / "HABORO_T.csv")

    n_persons = len(persons)
    n_trips = len(trips)
    print(
        f"LTDS fixture: {len(hhs)} hh, {n_persons} persons, {n_trips} trips"
        f" -> {dst}"
    )


def create_nhts_fixture(data_root: Path, out_root: Path) -> None:
    year_dir = "2022"
    src = data_root / year_dir
    dst = out_root / "nhts" / year_dir
    dst.mkdir(parents=True, exist_ok=True)

    # 1. hhv2pub.csv: sample N_HH rows
    hhs = pl.read_csv(src / "hhv2pub.csv", ignore_errors=True)
    hhs = hhs.sample(n=min(N_HH, len(hhs)), seed=SEED, shuffle=True)
    houseid_set = set(hhs["HOUSEID"].to_list())
    hhs = shuffle_cols(hhs, NHTS_HH_SHUFFLE)
    hhs.write_csv(dst / "hhv2pub.csv")

    # 2. perv2pub.csv: filter by HOUSEID ∈ houseid_set
    persons = pl.read_csv(src / "perv2pub.csv", ignore_errors=True)
    persons = persons.filter(pl.col("HOUSEID").is_in(houseid_set))
    persons = shuffle_cols(persons, NHTS_PERSON_SHUFFLE)
    persons.write_csv(dst / "perv2pub.csv")

    # 3. tripv2pub.csv: filter by HOUSEID ∈ houseid_set
    trips = pl.read_csv(src / "tripv2pub.csv", ignore_errors=True)
    trips = trips.filter(pl.col("HOUSEID").is_in(houseid_set))
    trips.write_csv(dst / "tripv2pub.csv")

    n_persons = len(persons)
    n_trips = len(trips)
    print(
        f"NHTS fixture: {len(hhs)} hh, {n_persons} persons, {n_trips} trips"
        f" -> {dst}"
    )


if __name__ == "__main__":
    ltds_root = Path(os.getenv("FOUNDATA_LTDS_DATA", LTDS_DATA_DEFAULT))
    nhts_root = Path(os.getenv("FOUNDATA_NHTS_DATA", NHTS_DATA_DEFAULT))

    if not ltds_root.exists():
        print(f"WARNING: LTDS data root not found: {ltds_root} — skipping LTDS")
    else:
        create_ltds_fixture(ltds_root, FIXTURE_ROOT)

    if not nhts_root.exists():
        print(f"WARNING: NHTS data root not found: {nhts_root} — skipping NHTS")
    else:
        create_nhts_fixture(nhts_root, FIXTURE_ROOT)
