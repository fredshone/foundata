"""Sample parquet tables by household id.

Used for sources whose raw tables are parquet files linked by a household id
column: chile (open data, every column unmodified) and france (MobiSurvStd
surveys, loader columns only, fresh ids and shuffled attributes).
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
    reads zone areas and points from the eod_zone_2012 location lookup.
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


# MobiSurvStd person / household columns that are shuffled across the sample;
# ids, survey flags, weights and travel-day columns stay with their row so
# persons, trips and legs remain consistent.
FRANCE_PERSON_SHUFFLED = [
    "age",
    "woman",
    "reference_person_link",
    "education_level",
    "detailed_professional_occupation",
    "pcs_category_code2020",
    "pcs_category_code2003",
    "pcs_group_code",
    "work_only_at_home",
    "telework",
    "has_driving_license",
    "has_travel_inconvenience",
]
FRANCE_HH_SHUFFLED = [
    "nb_cars",
    "nb_motorcycles",
    "housing_type",
    "housing_status",
    "home_insee_density",
]


def mapped_columns(mapping: dict) -> list[str]:
    """Raw column names of a (possibly nested) column_mappings dict."""
    cols = []
    for raw, target in mapping.items():
        cols += mapped_columns(target) if isinstance(target, dict) else [raw]
    return cols


def shuffle_columns(
    df: pl.DataFrame, cols: list[str], rng: random.Random
) -> pl.DataFrame:
    """Independently permute each of cols across the rows of df."""
    shuffled = {}
    for col in cols:
        values = df[col].to_list()
        rng.shuffle(values)
        shuffled[col] = pl.Series(col, values, dtype=df[col].dtype)
    return df.with_columns(**shuffled)


def sample_france(
    src_root: Path,
    dst_root: Path,
    surveys: list[str],
    n: int,
    seed: int,
) -> None:
    """Write `n` households from each MobiSurvStd survey folder in surveys.

    Licensing differs between the surveys, so they are all treated as
    restricted: only the columns the loader reads are written (no
    coordinates or original ids), households and persons get fresh
    sequential ids, and household and person attribute columns are shuffled
    independently across the sample. Trip and leg rows are kept intact
    (with remapped ids), and metadata.json is copied.
    """
    from foundata.france import (
        HOUSEHOLDS_PATH,
        LEGS_PATH,
        METADATA_PATH,
        PERSONS_PATH,
        TRIPS_PATH,
    )
    from foundata.utils import get_config_path, load_yaml_config

    def columns(config: str, *keys: str) -> list[str]:
        mapping = load_yaml_config(get_config_path("france", config))[
            "column_mappings"
        ]
        for key in keys:
            mapping = mapping[key]
        return mapped_columns(mapping)

    hh_cols = columns("hh_dictionary.yaml")
    person_cols = columns("person_dictionary.yaml")
    trip_cols = columns("trip_dictionary.yaml", "trips")
    leg_cols = columns("trip_dictionary.yaml", "legs")

    rng = random.Random(seed)
    for survey in surveys:
        src, dst = src_root / survey, dst_root / survey
        hhs = pl.read_parquet(src / HOUSEHOLDS_PATH, columns=hh_cols)
        sampled = sorted(
            rng.sample(hhs["household_id"].to_list(), min(n, len(hhs)))
        )
        hhs = hhs.filter(pl.col("household_id").is_in(sampled))
        persons = pl.read_parquet(src / PERSONS_PATH, columns=person_cols)
        persons = persons.filter(pl.col("household_id").is_in(sampled))
        trips = pl.read_parquet(src / TRIPS_PATH, columns=trip_cols)
        trips = trips.filter(pl.col("household_id").is_in(sampled))
        legs = pl.read_parquet(src / LEGS_PATH, columns=leg_cols)
        legs = legs.filter(pl.col("trip_id").is_in(trips["trip_id"]))

        hh_ids = {old: new for new, old in enumerate(sampled, start=1)}
        person_ids = {
            old: new
            for new, old in enumerate(sorted(persons["person_id"]), start=1)
        }
        trip_ids = {
            old: new
            for new, old in enumerate(sorted(trips["trip_id"]), start=1)
        }

        def remap(df: pl.DataFrame, col: str, ids: dict) -> pl.DataFrame:
            if col not in df.columns:
                return df
            return df.with_columns(
                pl.col(col).replace_strict(ids, return_dtype=df[col].dtype)
            )

        tables = {}
        for path, df in (
            (HOUSEHOLDS_PATH, shuffle_columns(hhs, FRANCE_HH_SHUFFLED, rng)),
            (
                PERSONS_PATH,
                shuffle_columns(persons, FRANCE_PERSON_SHUFFLED, rng),
            ),
            (TRIPS_PATH, trips),
            (LEGS_PATH, legs),
        ):
            df = remap(df, "household_id", hh_ids)
            df = remap(df, "person_id", person_ids)
            df = remap(df, "trip_id", trip_ids)
            tables[path] = df

        dst.mkdir(parents=True, exist_ok=True)
        for path, df in tables.items():
            df.write_parquet(dst / path)
        (dst / METADATA_PATH).write_bytes((src / METADATA_PATH).read_bytes())
        print(
            f"  {survey}: "
            + ", ".join(f"{p}: {len(df)}" for p, df in tables.items())
        )
