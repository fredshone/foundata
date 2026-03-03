"""
Generate anonymised test fixtures for CMAP, NTS, VISTA, and QHTS.

Run once from the project root with real data available:
    uv run python scripts/generate_fixtures.py

Samples 50 households per source, cascades the filter to persons/trips,
then shuffles every non-key column independently to break individual-level
correlations while preserving dtypes and value distributions.

Writes fixture files to tests/fixtures/<source>/ preserving exact filenames.
"""

from pathlib import Path

import polars as pl

DATA_ROOT = Path.home() / "Data" / "foundata"
FIXTURE_ROOT = Path(__file__).parent.parent / "tests" / "fixtures"
N_HOUSEHOLDS = 50
SEED = 42


def shuffle_non_keys(df: pl.DataFrame, key_cols: list[str], seed: int) -> pl.DataFrame:
    """Return df with every non-key column independently shuffled."""
    non_keys = [c for c in df.columns if c not in key_cols]
    shuffled = {
        col: df[col].sample(fraction=1.0, shuffle=True, seed=seed + i)
        for i, col in enumerate(non_keys)
    }
    return df.with_columns([pl.Series(name=c, values=v) for c, v in shuffled.items()])


def sample_hh_ids(df: pl.DataFrame, hh_col: str, n: int, seed: int) -> pl.Series:
    unique_ids = df[hh_col].unique()
    return unique_ids.sample(n=min(n, len(unique_ids)), seed=seed)


# ---------------------------------------------------------------------------
# CMAP
# ---------------------------------------------------------------------------


def generate_cmap():
    src = DATA_ROOT / "CMAP"
    dst = FIXTURE_ROOT / "cmap"
    dst.mkdir(parents=True, exist_ok=True)
    print("Generating CMAP fixtures...")

    hhs = pl.read_csv(src / "household.csv", ignore_errors=True)
    persons = pl.read_csv(
        src / "person.csv",
        ignore_errors=True,
        schema_overrides={"dtype": pl.String},  # contains "1;2;3" multi-values
    )
    places = pl.read_csv(src / "place.csv", ignore_errors=True)
    locations = pl.read_csv(src / "location.csv", ignore_errors=True)

    # Force-include a household where dtype has semicolons, so the column
    # is always inferred as String (not Int64) when the fixture is re-read.
    must_include = (
        persons.filter(pl.col("dtype").str.contains(";"))["sampno"]
        .unique()
        .sample(n=1, seed=SEED)
    )
    extra_ids = sample_hh_ids(
        hhs.filter(~pl.col("sampno").is_in(must_include)), "sampno", N_HOUSEHOLDS - 1, SEED
    )
    sampled_ids = pl.concat([must_include, extra_ids]).unique()

    hhs = hhs.filter(pl.col("sampno").is_in(sampled_ids))
    persons = persons.filter(pl.col("sampno").is_in(sampled_ids))
    places = places.filter(pl.col("sampno").is_in(sampled_ids))
    locations = locations.filter(pl.col("sampno").is_in(sampled_ids))

    hhs = shuffle_non_keys(hhs, key_cols=["sampno"], seed=SEED)
    persons = shuffle_non_keys(persons, key_cols=["sampno", "perno"], seed=SEED + 10)
    places = shuffle_non_keys(
        places,
        # Keep FIPS-linked locno, sequence cols, and distance (avoid spurious negatives).
        key_cols=["sampno", "perno", "placeno", "locno", "placeGroup", "traveldayno", "distance"],
        seed=SEED + 20,
    )
    locations = shuffle_non_keys(
        locations,
        # Keep FIPS and home flag so the rurality join and home-location lookup work.
        key_cols=["sampno", "locno", "state_fips", "county_fips", "tract_fips", "home"],
        seed=SEED + 30,
    )

    hhs.write_csv(dst / "household.csv")
    persons.write_csv(dst / "person.csv")
    places.write_csv(dst / "place.csv")
    locations.write_csv(dst / "location.csv")
    print(f"  households: {len(hhs)}, persons: {len(persons)}, places: {len(places)}, locations: {len(locations)}")


# ---------------------------------------------------------------------------
# NTS
# ---------------------------------------------------------------------------


def generate_nts():
    src = DATA_ROOT / "NTS" / "tab"
    dst = FIXTURE_ROOT / "nts" / "tab"
    dst.mkdir(parents=True, exist_ok=True)
    print("Generating NTS fixtures...")

    read_tab = lambda name: pl.read_csv(  # noqa: E731
        src / name, separator="\t", ignore_errors=True, null_values="NA"
    )

    hhs = read_tab("household_eul_2002-2023.tab")
    individuals = read_tab("individual_eul_2002-2023.tab")
    trips = read_tab("trip_eul_2002-2023.tab")
    days = read_tab("day_eul_2002-2023.tab")

    sampled_hh_ids = sample_hh_ids(hhs, "HouseholdID", N_HOUSEHOLDS, SEED)
    hhs = hhs.filter(pl.col("HouseholdID").is_in(sampled_hh_ids))
    individuals = individuals.filter(pl.col("HouseholdID").is_in(sampled_hh_ids))
    sampled_ind_ids = individuals["IndividualID"].unique()
    trips = trips.filter(pl.col("IndividualID").is_in(sampled_ind_ids))
    days = days.filter(pl.col("IndividualID").is_in(sampled_ind_ids))

    hhs = shuffle_non_keys(hhs, key_cols=["HouseholdID"], seed=SEED)
    individuals = shuffle_non_keys(individuals, key_cols=["IndividualID", "HouseholdID"], seed=SEED + 10)
    trips = shuffle_non_keys(trips, key_cols=["TripID", "JourSeq", "DayID", "IndividualID"], seed=SEED + 20)
    days = shuffle_non_keys(days, key_cols=["DayID", "IndividualID"], seed=SEED + 30)

    hhs.write_csv(dst / "household_eul_2002-2023.tab", separator="\t")
    individuals.write_csv(dst / "individual_eul_2002-2023.tab", separator="\t")
    trips.write_csv(dst / "trip_eul_2002-2023.tab", separator="\t")
    days.write_csv(dst / "day_eul_2002-2023.tab", separator="\t")
    print(f"  households: {len(hhs)}, individuals: {len(individuals)}, trips: {len(trips)}, days: {len(days)}")


# ---------------------------------------------------------------------------
# VISTA
# ---------------------------------------------------------------------------


def generate_vista():
    year = "2012-2020"
    src = DATA_ROOT / "VISTA" / year
    dst = FIXTURE_ROOT / "vista" / year
    dst.mkdir(parents=True, exist_ok=True)
    print("Generating VISTA fixtures...")

    hhs = pl.read_csv(src / "households_vista_2012_2020_lga_v1.csv", null_values="Missing/Refused")
    persons = pl.read_csv(src / "persons_vista_2012_2020_lga_v1.csv", ignore_errors=True)
    trips = pl.read_csv(src / "trips_vista_2012_2020_lga_v1.csv", ignore_errors=True)

    sampled_hh_ids = sample_hh_ids(hhs, "hhid", N_HOUSEHOLDS, SEED)
    hhs = hhs.filter(pl.col("hhid").is_in(sampled_hh_ids))
    persons = persons.filter(pl.col("hhid").is_in(sampled_hh_ids))
    sampled_pers_ids = persons["persid"].unique()
    trips = trips.filter(pl.col("persid").is_in(sampled_pers_ids))

    hhs = shuffle_non_keys(hhs, key_cols=["hhid"], seed=SEED)
    persons = shuffle_non_keys(persons, key_cols=["hhid", "persid"], seed=SEED + 10)
    trips = shuffle_non_keys(trips, key_cols=["hhid", "persid", "tripno"], seed=SEED + 20)

    hhs.write_csv(dst / "households_vista_2012_2020_lga_v1.csv")
    persons.write_csv(dst / "persons_vista_2012_2020_lga_v1.csv")
    trips.write_csv(dst / "trips_vista_2012_2020_lga_v1.csv")
    print(f"  households: {len(hhs)}, persons: {len(persons)}, trips: {len(trips)}")


# ---------------------------------------------------------------------------
# QHTS
# ---------------------------------------------------------------------------


def generate_qhts():
    year = "2017-20"
    src = DATA_ROOT / "QHTS" / year
    dst = FIXTURE_ROOT / "qhts" / year
    dst.mkdir(parents=True, exist_ok=True)
    print("Generating QHTS fixtures...")

    hhs = pl.read_csv(src / "1_QTS_HOUSEHOLDS.csv", ignore_errors=True)
    persons = pl.read_csv(src / "2_QTS_PERSONS.csv", ignore_errors=True)
    trips = pl.read_csv(src / "5_QTS_TRIPS.csv", ignore_errors=True)

    sampled_hh_ids = sample_hh_ids(hhs, "HHID", N_HOUSEHOLDS, SEED)
    hhs = hhs.filter(pl.col("HHID").is_in(sampled_hh_ids))
    persons = persons.filter(pl.col("HHID").is_in(sampled_hh_ids))
    sampled_pers_ids = persons["PERSID"].unique()
    trips = trips.filter(pl.col("PERSID").is_in(sampled_pers_ids))

    hhs = shuffle_non_keys(hhs, key_cols=["HHID"], seed=SEED)
    persons = shuffle_non_keys(persons, key_cols=["HHID", "PERSID"], seed=SEED + 10)
    trips = shuffle_non_keys(trips, key_cols=["HHID", "PERSID", "STARTSTOP"], seed=SEED + 20)

    hhs.write_csv(dst / "1_QTS_HOUSEHOLDS.csv")
    persons.write_csv(dst / "2_QTS_PERSONS.csv")
    trips.write_csv(dst / "5_QTS_TRIPS.csv")
    print(f"  households: {len(hhs)}, persons: {len(persons)}, trips: {len(trips)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_cmap()
    generate_nts()
    generate_vista()
    generate_qhts()
    print("Done.")
