"""
Build a minimal example data directory at data_dir_template/, mirroring the
structure of a real ~/Data/foundata/ tree with a handful of rows per source —
enough to run:

    uv run foundata run -d data_dir_template

Sibling to scripts/generate_fixtures.py (which builds the tests/fixtures/
used by the unit tests) and reuses its sample/shuffle helpers, but targets a
runnable example directory instead of test fixtures, and treats sources
differently based on redistribution licensing:

- cmap, nhts, qhts, vista are public/open datasets: households are sampled
  directly and written with real, unmodified values.
- odin, ktdb, nts, ltds are restricted-access research datasets: real
  respondent IDs are never reused (replaced with fresh synthetic ids,
  consistent across each source's tables), and demographic/attribute columns
  are independently shuffled across the sampled group so no real
  respondent's full attribute combination survives. Trip/stage rows are kept
  internally intact (not column-shuffled) so trip logic — sequencing, times,
  zones — stays coherent; only their household/person id columns are
  remapped to match the shuffled attribute rows.

Run once from the project root with real data available:
    uv run python scripts/build_data_dir_template.py
"""

from pathlib import Path

import polars as pl

DATA_ROOT = Path.home() / "Data" / "foundata"
TEMPLATE_ROOT = Path(__file__).parent.parent / "data_dir_template"
SEED = 7


def shuffle_non_keys(
    df: pl.DataFrame, key_cols: list[str], seed: int
) -> pl.DataFrame:
    """Return df with every non-key column independently shuffled."""
    non_keys = [c for c in df.columns if c not in key_cols]
    shuffled = {
        col: df[col].sample(fraction=1.0, shuffle=True, seed=seed + i)
        for i, col in enumerate(non_keys)
    }
    return df.with_columns(
        [pl.Series(name=c, values=v) for c, v in shuffled.items()]
    )


def sample_ids(df: pl.DataFrame, col: str, n: int, seed: int) -> pl.Series:
    unique_ids = df[col].unique()
    return unique_ids.sample(n=min(n, len(unique_ids)), seed=seed)


def id_map(ids: pl.Series, offset: int) -> dict:
    """Map each real id to a fresh sequential synthetic id, off the real id range."""
    sorted_ids = sorted(ids.to_list())
    return {orig: offset + i for i, orig in enumerate(sorted_ids)}


def remap(df: pl.DataFrame, col: str, mapping: dict) -> pl.DataFrame:
    return df.with_columns(pl.col(col).replace(mapping))


# ---------------------------------------------------------------------------
# CMAP (public data — direct sample, unshuffled)
# ---------------------------------------------------------------------------


def generate_cmap():
    src = DATA_ROOT / "CMAP"
    dst = TEMPLATE_ROOT / "CMAP"
    dst.mkdir(parents=True, exist_ok=True)
    print("Generating CMAP...")

    hhs = pl.read_csv(src / "household.csv", ignore_errors=True)
    persons = pl.read_csv(
        src / "person.csv",
        ignore_errors=True,
        schema_overrides={"dtype": pl.String},
    )
    places = pl.read_csv(src / "place.csv", ignore_errors=True)
    locations = pl.read_csv(src / "location.csv", ignore_errors=True)

    # households with >=2 persons, each person with >=4 place rows (trips are
    # derived by shifting place rows, so each person needs several to yield
    # any real trips)
    place_counts = places.group_by(["sampno", "perno"]).len()
    qualifying_persons = place_counts.filter(pl.col("len") >= 4).select(
        ["sampno", "perno"]
    )
    qualifying_hh_persons = persons.join(
        qualifying_persons, on=["sampno", "perno"], how="inner"
    )
    hh_person_counts = qualifying_hh_persons.group_by("sampno").len()
    qualifying_hh = hh_person_counts.filter(pl.col("len") >= 2)["sampno"]

    sampled_ids = sample_ids(
        hhs.filter(pl.col("sampno").is_in(qualifying_hh)), "sampno", 3, SEED
    )

    hhs = hhs.filter(pl.col("sampno").is_in(sampled_ids))
    persons = persons.filter(pl.col("sampno").is_in(sampled_ids))
    places = places.filter(pl.col("sampno").is_in(sampled_ids))
    locations = locations.filter(pl.col("sampno").is_in(sampled_ids))

    # blank precise coordinates — not used by the loader, but exact home/place
    # addresses are PII even for an otherwise-public dataset
    locations = locations.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("latitude"),
        pl.lit(None, dtype=pl.Float64).alias("longitude"),
    )

    # The real person.csv's "dtype" (-> disability) column has some real
    # semicolon-joined multi-select values (e.g. "6;97"), which is why the
    # loader treats it as a string it can `.str.split(";")`. Our small
    # sample only has single-value codes, which polars would otherwise
    # infer as int64 and break that call. Force one row to a genuine
    # multi-value ("-1;-1" = no listed disabilities) to match production
    # schema inference.
    dtype_values = persons["dtype"].to_list()
    dtype_values[0] = "-1;-1"
    persons = persons.with_columns(
        pl.Series("dtype", dtype_values, dtype=pl.Utf8)
    )

    # Same issue for "indus" (-> occupation): real data has ~11% NAICS
    # range-coded values like "31-33"/"44-45"/"48-49" (config/cmap/
    # person_dictionary.yaml maps these hyphenated strings directly), so
    # the real column infers as string. Force one row to a real range code
    # so a small sample matches production schema inference.
    indus_values = persons["indus"].to_list()
    indus_values[min(1, len(indus_values) - 1)] = "31-33"
    persons = persons.with_columns(
        pl.Series("indus", indus_values, dtype=pl.Utf8)
    )

    hhs.write_csv(dst / "household.csv")
    persons.write_csv(dst / "person.csv")
    places.write_csv(dst / "place.csv")
    locations.write_csv(dst / "location.csv")
    print(
        f"  households: {len(hhs)}, persons: {len(persons)}, "
        f"places: {len(places)}, locations: {len(locations)}"
    )


# ---------------------------------------------------------------------------
# NHTS (public data — direct sample, unshuffled)
# ---------------------------------------------------------------------------

NHTS_FILES = {
    2022: ("hhv2pub.csv", "perv2pub.csv", "tripv2pub.csv"),
    2017: ("hhpub.csv", "perpub.csv", "trippub.csv"),
    2009: ("HHV2PUB.CSV", "PERV2PUB.CSV", "DAYV2PUB.CSV"),
    2001: ("HHPUB.csv", "PERPUB.csv", "DAYPUB.csv"),
}


def generate_nhts():
    print("Generating NHTS...")
    for year, (hh_name, per_name, trip_name) in NHTS_FILES.items():
        src = DATA_ROOT / "NHTS" / str(year)
        dst = TEMPLATE_ROOT / "NHTS" / str(year)
        dst.mkdir(parents=True, exist_ok=True)

        hhs = pl.read_csv(src / hh_name, ignore_errors=True)
        persons = pl.read_csv(src / per_name, ignore_errors=True)
        trips = pl.read_csv(src / trip_name, ignore_errors=True)

        # households with >=2 persons, all of whose trips have non-negative
        # start time / duration (a single negative value drops that
        # person's whole trip chain in the loader)
        clean_pids = (
            trips.group_by("PERSONID", "HOUSEID")
            .agg(
                bad=((pl.col("STRTTIME") < 0) | (pl.col("TRVLCMIN") < 0)).any()
            )
            .filter(~pl.col("bad"))
            .select("HOUSEID", "PERSONID")
        )
        clean_person_counts = clean_pids.group_by("HOUSEID").len()
        person_counts = persons.group_by("HOUSEID").len()
        qualifying = (
            clean_person_counts.join(person_counts, on="HOUSEID", suffix="_all")
            .filter(pl.col("len") == pl.col("len_all"))
            .filter(pl.col("len") >= 2)["HOUSEID"]
        )

        sampled_ids = sample_ids(
            hhs.filter(pl.col("HOUSEID").is_in(qualifying)), "HOUSEID", 3, SEED
        )

        hhs_s = hhs.filter(pl.col("HOUSEID").is_in(sampled_ids))
        persons_s = persons.filter(pl.col("HOUSEID").is_in(sampled_ids))
        trips_s = trips.filter(pl.col("HOUSEID").is_in(sampled_ids))

        hhs_s.write_csv(dst / hh_name)
        persons_s.write_csv(dst / per_name)
        trips_s.write_csv(dst / trip_name)
        print(
            f"  {year}: households: {len(hhs_s)}, persons: {len(persons_s)}, "
            f"trips: {len(trips_s)}"
        )


# ---------------------------------------------------------------------------
# QHTS (public data — direct sample, unshuffled)
# ---------------------------------------------------------------------------

QHTS_YEARS = ["2019-22", "2022-25"]


def generate_qhts():
    print("Generating QHTS...")
    for i, year in enumerate(QHTS_YEARS):
        src = DATA_ROOT / "QHTS" / year
        dst = TEMPLATE_ROOT / "QHTS" / year
        dst.mkdir(parents=True, exist_ok=True)

        hhs = pl.read_csv(
            src / "1_QTS_HOUSEHOLDS.csv",
            ignore_errors=True,
            null_values="Missing/Refused",
        )
        persons = pl.read_csv(src / "2_QTS_PERSONS.csv", ignore_errors=True)
        trips = pl.read_csv(
            src / "5_QTS_TRIPS.csv", ignore_errors=True, null_values="Missing"
        )

        # households with >=2 persons, all of whom have at least one trip
        person_counts = persons.group_by("HHID").len()
        pids_with_trips = trips.select("PERSID").unique()
        persons_with_trips = persons.join(
            pids_with_trips, on="PERSID", how="inner"
        )
        trip_person_counts = persons_with_trips.group_by("HHID").len()
        qualifying = (
            trip_person_counts.join(person_counts, on="HHID", suffix="_all")
            .filter(pl.col("len") == pl.col("len_all"))
            .filter(pl.col("len") >= 2)["HHID"]
        )

        sampled_ids = sample_ids(
            hhs.filter(pl.col("HHID").is_in(qualifying)), "HHID", 4, SEED + i
        )

        hhs_s = hhs.filter(pl.col("HHID").is_in(sampled_ids))
        persons_s = persons.filter(pl.col("HHID").is_in(sampled_ids))
        sampled_persids = persons_s["PERSID"].unique()
        trips_s = trips.filter(pl.col("PERSID").is_in(sampled_persids))

        hhs_s.write_csv(dst / "1_QTS_HOUSEHOLDS.csv")
        persons_s.write_csv(dst / "2_QTS_PERSONS.csv")
        trips_s.write_csv(dst / "5_QTS_TRIPS.csv")
        print(
            f"  {year}: households: {len(hhs_s)}, persons: {len(persons_s)}, "
            f"trips: {len(trips_s)}"
        )


# ---------------------------------------------------------------------------
# VISTA (public data — direct sample, unshuffled)
# ---------------------------------------------------------------------------

VISTA_YEARS = ["2012-2020", "2022-2023", "2023-2024"]
VISTA_FILES = {
    "2012-2020": (
        "households_vista_2012_2020_lga_v1.csv",
        "persons_vista_2012_2020_lga_v1.csv",
        "trips_vista_2012_2020_lga_v1.csv",
    ),
    "2022-2023": (
        "household_vista_2022_2023.csv",
        "person_vista_2022_2023.csv",
        "trips_vista_2022_2023.csv",
    ),
    "2023-2024": (
        "household_vista_2023_2024.csv",
        "person_vista_2023_2024.csv",
        "trips_vista_2023_2024.csv",
    ),
}


def generate_vista():
    print("Generating VISTA...")
    for i, year in enumerate(VISTA_YEARS):
        hh_name, per_name, trip_name = VISTA_FILES[year]
        src = DATA_ROOT / "VISTA" / year
        dst = TEMPLATE_ROOT / "VISTA" / year
        dst.mkdir(parents=True, exist_ok=True)

        hhs = pl.read_csv(
            src / hh_name, ignore_errors=True, null_values="Missing/Refused"
        )
        persons = pl.read_csv(src / per_name, ignore_errors=True)
        trips = pl.read_csv(
            src / trip_name, ignore_errors=True, null_values="Missing"
        )

        person_counts = persons.group_by("hhid").len()
        pids_with_trips = trips.select("persid").unique()
        persons_with_trips = persons.join(
            pids_with_trips, on="persid", how="inner"
        )
        trip_person_counts = persons_with_trips.group_by("hhid").len()
        qualifying = (
            trip_person_counts.join(person_counts, on="hhid", suffix="_all")
            .filter(pl.col("len") == pl.col("len_all"))
            .filter(pl.col("len") >= 2)["hhid"]
        )

        sampled_ids = sample_ids(
            hhs.filter(pl.col("hhid").is_in(qualifying)), "hhid", 4, SEED + i
        )

        if year == "2012-2020":
            # ensure at least one weekend-surveyed household is included:
            # if wehhwgt_LGA is entirely null in the sample, polars infers a
            # different dtype for it than in the (much larger) real file,
            # breaking the cross-year concat of the coalesced weight column
            weekend_ids = hhs.filter(
                pl.col("hhid").is_in(qualifying)
                & pl.col("wehhwgt_LGA").is_not_null()
            )["hhid"]
            if (
                len(weekend_ids) > 0
                and not sampled_ids.is_in(weekend_ids).any()
            ):
                extra = weekend_ids.sample(n=1, seed=SEED)
                sampled_ids = pl.concat(
                    [sampled_ids.slice(0, len(sampled_ids) - 1), extra]
                )

        hhs_s = hhs.filter(pl.col("hhid").is_in(sampled_ids))
        persons_s = persons.filter(pl.col("hhid").is_in(sampled_ids))
        sampled_persids = persons_s["persid"].unique()
        trips_s = trips.filter(pl.col("persid").is_in(sampled_persids))

        # drop approximate DOB columns if present — not used by the loader
        for col in ("monthofbirth", "yearofbirth"):
            if col in persons_s.columns:
                persons_s = persons_s.drop(col)

        hhs_s.write_csv(dst / hh_name)
        persons_s.write_csv(dst / per_name)
        trips_s.write_csv(dst / trip_name)
        print(
            f"  {year}: households: {len(hhs_s)}, persons: {len(persons_s)}, "
            f"trips: {len(trips_s)}"
        )


# ---------------------------------------------------------------------------
# KTDB (restricted — anonymized: shuffled attributes, intact trips, fresh ids)
# ---------------------------------------------------------------------------


def generate_ktdb():
    src = DATA_ROOT / "KTDB"
    dst = TEMPLATE_ROOT / "KTDB"
    dst.mkdir(parents=True, exist_ok=True)
    print("Generating KTDB...")

    persons = pl.read_csv(
        src / "persons.csv", ignore_errors=True, encoding="euc-kr"
    )
    # force zone-code columns to string: the real (much larger) trips.csv
    # infers these as string, but a small sample of purely-numeric-looking
    # codes would otherwise infer as int64 and break the strict-typed join
    # against configs/ktdb/zone_distances.parquet (which stores them as str)
    zone_cols = ["sTP1_1_5", "TP1_1_5"]
    trips = pl.read_csv(
        src / "trips.csv",
        ignore_errors=True,
        encoding="euc-kr",
        schema_overrides={c: pl.Utf8 for c in zone_cols},
    )

    trip_counts = trips.group_by("idx").len().filter(pl.col("len") >= 2)
    sampled_ids = sample_ids(
        persons.filter(pl.col("idx").is_in(trip_counts["idx"])), "idx", 6, SEED
    )

    persons = persons.filter(pl.col("idx").is_in(sampled_ids))
    trips = trips.filter(pl.col("idx").is_in(sampled_ids))

    # drop free-text address columns — not used by the loader, but literal
    # home-address text
    address_cols = [
        "SQ1_4",
        "SQ1_5",
        "SQ1_6",
        "SQ1_7",
        "SQ1_8",
        "R3_2_a",
        "Q2_a",
        "TP2_a",
        "sTP1_1_4",
        "sTP1_1_6",
        "sTP1_1_7",
        "sTP1_1_8",
        "TP1_1_4",
        "TP1_1_6",
        "TP1_1_7",
        "TP1_1_8",
    ]
    persons = persons.drop([c for c in address_cols if c in persons.columns])
    trips = trips.drop([c for c in address_cols if c in trips.columns])

    persons = shuffle_non_keys(persons, key_cols=["idx"], seed=SEED)

    mapping = id_map(sampled_ids, offset=900001)
    persons = remap(persons, "idx", mapping)
    trips = remap(trips, "idx", mapping)

    # The real trips.csv is large enough that some rows have a blank
    # sTP1_1_5/TP1_1_5 zone code, which makes polars infer those columns as
    # string. Our tiny sample only has real 10-digit codes, so it would
    # infer as int64 instead and break the strict-typed join against
    # configs/ktdb/zone_distances.parquet (stored as str). Add one throwaway
    # row (th_seq=0) reproducing that real-world blank-code case — the
    # loader's `seq > 0` filter drops it immediately, but its presence in
    # the file is enough to make schema inference match production.
    dummy = trips.head(1).with_columns(
        th_seq=pl.lit(0).cast(trips.schema["th_seq"]),
        sTP1_1_5=pl.lit(" "),
        TP1_1_5=pl.lit(" "),
    )
    trips = pl.concat([dummy, trips])

    (dst / "persons.csv").write_bytes(
        persons.write_csv().encode("euc-kr", errors="replace")
    )
    (dst / "trips.csv").write_bytes(
        trips.write_csv().encode("euc-kr", errors="replace")
    )
    print(f"  persons: {len(persons)}, trips: {len(trips)}")


# ---------------------------------------------------------------------------
# NTS (restricted — anonymized: shuffled attributes, intact trips, fresh ids)
# ---------------------------------------------------------------------------


def generate_nts():
    src = DATA_ROOT / "NTS" / "tab"
    dst = TEMPLATE_ROOT / "NTS" / "tab"
    dst.mkdir(parents=True, exist_ok=True)
    print("Generating NTS...")

    read_tab = lambda name: pl.read_csv(  # noqa: E731
        src / name, separator="\t", ignore_errors=True, null_values="NA"
    )

    hhs = read_tab("household_eul_2002-2024.tab")
    individuals = read_tab("individual_eul_2002-2024.tab")
    trips = read_tab("trip_eul_2002-2024.tab")
    days = read_tab("day_eul_2002-2024.tab")
    stages = read_tab("stage_eul_2002-2024.tab")

    # households whose individuals have >=1 day with >=2 trips and >=1 stage
    trip_day_counts = trips.group_by("IndividualID", "DayID").len()
    ok_days = trip_day_counts.filter(pl.col("len") >= 2).select(
        "IndividualID", "DayID"
    )
    ok_individuals = ok_days["IndividualID"].unique()
    qualifying_hh = individuals.filter(
        pl.col("IndividualID").is_in(ok_individuals)
    )["HouseholdID"].unique()

    sampled_hh_ids = sample_ids(
        hhs.filter(pl.col("HouseholdID").is_in(qualifying_hh)),
        "HouseholdID",
        6,
        SEED,
    )
    hhs = hhs.filter(pl.col("HouseholdID").is_in(sampled_hh_ids))
    individuals = individuals.filter(
        pl.col("HouseholdID").is_in(sampled_hh_ids)
    )
    sampled_ind_ids = individuals["IndividualID"].unique()

    # keep only the "clean" (>=2 trips) days for the sampled individuals
    ok_days = ok_days.filter(pl.col("IndividualID").is_in(sampled_ind_ids))
    trips = trips.join(ok_days, on=["IndividualID", "DayID"], how="inner")
    days = days.join(ok_days, on=["IndividualID", "DayID"], how="inner")
    sampled_trip_ids = trips["TripID"].unique()
    stages = stages.filter(pl.col("TripID").is_in(sampled_trip_ids))

    # shuffle demographic columns across sampled households/individuals
    hhs = shuffle_non_keys(hhs, key_cols=["HouseholdID"], seed=SEED)
    individuals = shuffle_non_keys(
        individuals, key_cols=["IndividualID", "HouseholdID"], seed=SEED + 10
    )

    # fresh synthetic ids, consistent across all 5 tables — trip/day/stage
    # rows themselves are left fully intact (not column-shuffled) so trip
    # sequencing/timing/purpose stay internally coherent
    hh_map = id_map(sampled_hh_ids, offset=900001)
    ind_map = id_map(sampled_ind_ids, offset=800001)
    day_map = id_map(days["DayID"].unique(), offset=700001)
    trip_map = id_map(trips["TripID"].unique(), offset=600001)
    stage_map = id_map(stages["StageID"].unique(), offset=500001)

    hhs = remap(hhs, "HouseholdID", hh_map)
    individuals = remap(
        remap(individuals, "HouseholdID", hh_map), "IndividualID", ind_map
    )
    days = remap(
        remap(
            remap(days, "HouseholdID", hh_map)
            if "HouseholdID" in days.columns
            else days,
            "IndividualID",
            ind_map,
        ),
        "DayID",
        day_map,
    )
    trips = remap(
        remap(
            remap(remap(trips, "HouseholdID", hh_map), "IndividualID", ind_map),
            "DayID",
            day_map,
        ),
        "TripID",
        trip_map,
    )
    stages = remap(remap(stages, "IndividualID", ind_map), "DayID", day_map)
    stages = remap(stages, "TripID", trip_map)
    stages = remap(stages, "StageID", stage_map)

    hhs.write_csv(dst / "household_eul_2002-2024.tab", separator="\t")
    individuals.write_csv(dst / "individual_eul_2002-2024.tab", separator="\t")
    trips.write_csv(dst / "trip_eul_2002-2024.tab", separator="\t")
    days.write_csv(dst / "day_eul_2002-2024.tab", separator="\t")
    stages.write_csv(dst / "stage_eul_2002-2024.tab", separator="\t")
    print(
        f"  households: {len(hhs)}, individuals: {len(individuals)}, "
        f"trips: {len(trips)}, days: {len(days)}, stages: {len(stages)}"
    )


# ---------------------------------------------------------------------------
# LTDS (restricted — anonymized: shuffled attributes, intact trips, fresh ids)
# ---------------------------------------------------------------------------

LTDS_YEARS = ["LTDS1920", "LTDS2223", "LTDS2324", "LTDS2425"]
LTDS_CANDIDATES = {
    "Household.csv": ["Household.csv"],
    "person.csv": ["person.csv", "Person.csv"],
    "person data.csv": ["person data.csv", "Person_data.csv"],
    "Trip.csv": ["Trip.csv"],
    "Stage.csv": ["Stage.csv"],
}


def _find(year_dir: Path, target: str) -> Path:
    for candidate in LTDS_CANDIDATES[target]:
        p = year_dir / candidate
        if p.exists():
            return p
    raise FileNotFoundError(
        f"none of {LTDS_CANDIDATES[target]} found in {year_dir}"
    )


def generate_ltds():
    print("Generating LTDS...")
    for i, year in enumerate(LTDS_YEARS):
        src = DATA_ROOT / "LTDS" / year
        dst = TEMPLATE_ROOT / "LTDS" / year
        dst.mkdir(parents=True, exist_ok=True)

        hhs = pl.read_csv(_find(src, "Household.csv"), ignore_errors=True)
        persons = pl.read_csv(_find(src, "person.csv"), ignore_errors=True)
        persons_data = pl.read_csv(
            _find(src, "person data.csv"), ignore_errors=True
        )
        trips = pl.read_csv(_find(src, "Trip.csv"), ignore_errors=True)
        stages = pl.read_csv(_find(src, "Stage.csv"), ignore_errors=True)

        # households with >=2 persons who each have >=2 trips
        trip_counts = trips.group_by("tpid").len().filter(pl.col("len") >= 2)
        qualifying_persons = persons.filter(
            pl.col("ppid").is_in(trip_counts["tpid"])
        )
        person_counts = qualifying_persons.group_by("phid").len()
        qualifying_hh = person_counts.filter(pl.col("len") >= 2)["phid"]

        sampled_hh_ids = sample_ids(
            hhs.filter(pl.col("hhid").is_in(qualifying_hh)), "hhid", 4, SEED + i
        )

        hhs_s = hhs.filter(pl.col("hhid").is_in(sampled_hh_ids))
        persons_s = persons.filter(pl.col("phid").is_in(sampled_hh_ids))
        sampled_ppids = persons_s["ppid"].unique()
        persons_data_s = persons_data.filter(
            pl.col("ppid").is_in(sampled_ppids)
        )
        trips_s = trips.filter(pl.col("thid").is_in(sampled_hh_ids))
        stages_s = stages.filter(pl.col("spid").is_in(sampled_ppids))

        # shuffle demographic columns across sampled households/persons
        hhs_s = shuffle_non_keys(hhs_s, key_cols=["hhid"], seed=SEED)
        persons_s = shuffle_non_keys(
            persons_s, key_cols=["phid", "ppid"], seed=SEED + 10
        )
        persons_data_s = shuffle_non_keys(
            persons_data_s, key_cols=["phid", "ppid"], seed=SEED + 20
        )

        hh_map = id_map(sampled_hh_ids, offset=900001)
        ppid_map = id_map(sampled_ppids, offset=800001)

        hhs_s = remap(hhs_s, "hhid", hh_map)
        persons_s = remap(remap(persons_s, "phid", hh_map), "ppid", ppid_map)
        persons_data_s = remap(
            remap(persons_data_s, "phid", hh_map), "ppid", ppid_map
        )
        trips_s = remap(remap(trips_s, "thid", hh_map), "tpid", ppid_map)
        stages_s = remap(stages_s, "spid", ppid_map)

        hhs_s.write_csv(dst / "Household.csv")
        persons_s.write_csv(dst / "person.csv")
        persons_data_s.write_csv(dst / "person data.csv")
        trips_s.write_csv(dst / "Trip.csv")
        stages_s.write_csv(dst / "Stage.csv")
        # HABORO_T.csv is a small public borough-code reference table, not
        # respondent data — copy verbatim so the zone join still resolves
        (dst / "HABORO_T.csv").write_bytes((src / "HABORO_T.csv").read_bytes())

        print(
            f"  {year}: households: {len(hhs_s)}, persons: {len(persons_s)}, "
            f"trips: {len(trips_s)}, stages: {len(stages_s)}"
        )


# ---------------------------------------------------------------------------
# ODIN (restricted — anonymized: shuffled attributes, intact trips, fresh ids)
# ---------------------------------------------------------------------------

ODIN_DATA_FILES = {
    2018: "ODiN2018_Databestand_v2.0.tab",
    2019: "ODiN2019_Databestand_v2.0.tab",
    2020: "ODiN2020_Databestand_v2.0.tab",
    2022: "ODiN2022_Databestand.tab",
    2023: "ODiN2023_Databestand.csv",  # tab-separated despite .csv
    2024: "ODiN2024_DANS_Databestand_v2.0.csv",  # tab-separated despite .csv
}


def generate_odin():
    print("Generating ODIN...")
    for i, (year, name) in enumerate(ODIN_DATA_FILES.items()):
        src = DATA_ROOT / "ODIN" / str(year)
        dst = TEMPLATE_ROOT / "ODIN" / str(year)
        dst.mkdir(parents=True, exist_ok=True)

        data = pl.read_csv(src / name, separator="\t", infer_schema_length=0)

        attr_rows = data.filter(pl.col("OP") == "1")
        trip_rows = data.filter(pl.col("Verpl") == "1")

        # respondents with >=2 kept trips
        trip_counts = (
            trip_rows.group_by("OPID").len().filter(pl.col("len") >= 2)
        )
        sampled_ids = sample_ids(
            attr_rows.filter(pl.col("OPID").is_in(trip_counts["OPID"])),
            "OPID",
            5,
            SEED + i,
        )

        attr_rows = attr_rows.filter(pl.col("OPID").is_in(sampled_ids))
        trip_rows = trip_rows.filter(pl.col("OPID").is_in(sampled_ids))

        # shuffle demographic/household columns across sampled respondents;
        # trip rows are left fully intact so trip logic stays coherent
        attr_rows = shuffle_non_keys(
            attr_rows, key_cols=["OP", "OPID"], seed=SEED
        )

        # A respondent's OP==1 attribute row is very often *also* their
        # first Verpl==1 trip row (same physical row serves both roles in
        # the raw file). After shuffling, that row's own (shuffled) Verpl
        # value could land on "1" again by chance, leaking a bogus,
        # demographically-shuffled "trip" — with a VerplNr that can
        # collide with a real trip's sequence number — into the loader's
        # Verpl==1 selection. Force it off so attr_rows are only ever
        # picked up as attribute rows downstream.
        attr_rows = attr_rows.with_columns(pl.lit("").alias("Verpl"))

        mapping = id_map(sampled_ids, offset=900001)
        attr_rows = remap(attr_rows, "OPID", mapping)
        trip_rows = remap(trip_rows, "OPID", mapping)

        out = pl.concat([attr_rows, trip_rows], how="diagonal")
        out.write_csv(dst / name, separator="\t")
        print(
            f"  {year}: respondents: {len(attr_rows)}, trips: {len(trip_rows)}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_cmap()
    generate_nhts()
    generate_qhts()
    generate_vista()
    generate_ktdb()
    generate_nts()
    generate_ltds()
    generate_odin()
    print("Done.")
