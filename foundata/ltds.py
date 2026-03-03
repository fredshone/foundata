import random
from pathlib import Path

import polars as pl

from foundata import fix
from foundata.utils import (
    bounds_from_list,
    compute_avg_speed,
    config_for_year,
    expand_root,
    fuzzy_loader,
    sample_int_range,
    sample_uk_to_euro,
    table_joiner,
    table_stacker,
)

SOURCE = "ltds"


def load_mapping(path: Path) -> dict:
    zones = pl.read_csv(path)
    mapping = {1: "urban", 2: "suburban", 3: "rural"}
    zones = zones.with_columns(
        pl.col("HIOX").replace_strict(mapping, default="rural")
    )
    return dict(zip(zones["HABORO"], zones["HIOX"]))


def load_years(
    data_root: str | Path,
    years: list[str],
    hh_config: dict,
    person_config: dict,
    person_data_config: dict,
    trips_config: dict,
    stages_config: dict,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    all_attributes = []
    all_trips = []
    print("Loading LTDS...")
    for year in years:
        print(f"Loading {year}...")
        root = expand_root(data_root) / year

        hh_config_year = config_for_year(hh_config, year)
        person_config_year = config_for_year(person_config, year)
        person_data_config_year = config_for_year(person_data_config, year)
        trips_config_year = config_for_year(trips_config, year)
        stages_config_year = config_for_year(stages_config, year)

        hh_columns = list(hh_config_year["column_mappings"].keys())
        hhs = fuzzy_loader(root, "Household.csv", columns=hh_columns)

        person_columns = list(person_config_year["column_mappings"].keys())
        persons = fuzzy_loader(root, "person.csv", columns=person_columns)

        person_data_columns = list(
            person_data_config_year["column_mappings"].keys()
        )
        persons_data = fuzzy_loader(
            root, "person data.csv", columns=person_data_columns
        )

        trips_columns = list(trips_config_year["column_mappings"].keys())
        trips = fuzzy_loader(root, "Trip.csv", columns=trips_columns)

        stages_columns = list(stages_config_year["column_mappings"].keys())
        stages = fuzzy_loader(root, "Stage.csv", columns=stages_columns)

        zone_mapping = load_mapping(root / "HABORO_T.csv")

        print("processing hhs...")
        hhs = preprocess_hhs(hhs, hh_config_year, year, zone_mapping)

        print("processing persons...")
        persons = preprocess_persons(persons, person_config_year)

        print("processing person data...")
        persons_data = preprocess_persons_data(
            persons_data, person_data_config_year, year
        )
        attributes = table_joiner(
            persons,
            persons_data.drop("hid"),
            on="pid",
            lhs_name="persons",
            rhs_name="person_data",
        )
        attributes = table_joiner(
            attributes, hhs, on="hid", lhs_name="attributes", rhs_name="hhs"
        )

        print("processing trips and stages...")
        trips = preprocess_trips(trips, trips_config_year, year, zone_mapping)
        stages = preprocess_stages(stages, stages_config_year, year)
        trips = table_joiner(
            trips, stages, on="tid", lhs_name="trips", rhs_name="stages"
        )

        all_attributes.append(attributes)
        all_trips.append(trips)

    attributes = table_stacker(all_attributes)
    trips = table_stacker(all_trips)

    attributes = attributes.with_columns(
        source=pl.lit(SOURCE),
        country=pl.lit("uk"),
        education=pl.lit("unknown"),
        ownership=pl.lit("unknown"),
        dwelling=pl.lit("unknown"),
        month=pl.lit(None, dtype=pl.Int8),
        disability=pl.lit("unknown"),
    )

    attributes = attributes.with_columns(
        pid=pl.lit(SOURCE) + pl.col("pid").cast(pl.String),
        hid=pl.lit(SOURCE) + pl.col("hid").cast(pl.String),
    )
    trips = trips.with_columns(
        pid=pl.lit(SOURCE) + pl.col("pid").cast(pl.String)
    )

    attributes = compute_avg_speed(attributes, trips)

    return attributes, trips


def preprocess_hhs(
    hhs: pl.DataFrame, config: dict, year: str, zone_mapping: dict
) -> pl.DataFrame:
    column_mapping = config["column_mappings"]
    income_mapping = config["hh_income"]
    struct_mapping = config["hh_structure"]
    day_mapping = config["day"]

    hhs = hhs.select(column_mapping.keys()).rename(column_mapping)

    hhs = hhs.with_columns(year=(pl.col("year") + 2000).cast(pl.Int32))

    hhs = hhs.with_columns(
        hh_income=(
            pl.col("hh_income").replace_strict(
                income_mapping, return_dtype=pl.List(pl.Int32)
            )
        )
    ).with_columns(
        hh_income=(
            pl.col("hh_income").map_elements(
                sample_uk_to_euro, return_dtype=pl.Int32
            )
        )
    )

    hhs = hhs.with_columns(
        pl.col("hh_structure")
        .replace_strict(struct_mapping)
        .fill_null("unknown")
    )

    hhs = hhs.with_columns(
        pl.col("day").replace_strict(day_mapping).fill_null("unknown")
    )

    hhs = hhs.with_columns(
        pl.col("zone")
        .replace_strict(zone_mapping, default=pl.col("zone"))
        .fill_null("unknown")
        .alias("rurality")
    ).drop("zone")

    hhs = hhs.filter(pl.col("hid").is_not_null())

    return hhs


def preprocess_persons(
    persons: pl.DataFrame, persons_config: dict
) -> pl.DataFrame:
    column_mapping = persons_config["column_mappings"]
    sex_mapping = persons_config["sex"]
    relationship_mapping = persons_config["relationship"]
    race_mapping = persons_config["race"]

    persons = persons.select(column_mapping.keys()).rename(column_mapping)

    persons = persons.with_columns(
        pl.col("age")
        .replace_strict({"65+": "65-100"}, default=pl.col("age"))
        .str.split("-")
        .map_elements(
            lambda bounds: sample_int_range(bounds_from_list(bounds)), pl.Int32
        )
    )

    persons = persons.with_columns(pl.col("sex").replace_strict(sex_mapping))

    persons = persons.with_columns(
        pl.col("relationship")
        .replace_strict(relationship_mapping)
        .alias("relationship")
    )

    persons = persons.with_columns(
        pl.col("race")
        .replace_strict(race_mapping, default=pl.col("race"))
        .fill_null("unknown")
    )

    if "employment" in column_mapping.values():
        employment_mapping = persons_config["employment"]
        persons = persons.with_columns(
            pl.col("employment").replace_strict(employment_mapping)
        )

    persons = persons.filter(pl.col("pid").is_not_null())

    return persons


def preprocess_persons_data(
    persons: pl.DataFrame, config: dict, year: str
) -> pl.DataFrame:
    column_mapping = config["column_mappings"]
    has_license_mapping = config["has_licence"]
    can_wfh_mapping = config["can_wfh"]
    occupation_mapping = config["occupation"]

    persons = persons.select(column_mapping.keys()).rename(column_mapping)

    persons = persons.with_columns(
        pl.col("has_licence").replace_strict(has_license_mapping),
        pl.col("can_wfh").replace_strict(can_wfh_mapping),
        pl.col("occupation").replace_strict(occupation_mapping),
    )

    if "employment" in column_mapping.values():
        employment_mapping = config["employment"]
        persons = persons.with_columns(
            pl.col("employment").replace_strict(employment_mapping)
        )

    persons = persons.with_columns((pl.col("no_trips") < 0).alias("no_trips"))

    return persons


def sample_minute(base: int) -> int:
    return random.randint(max(1, int(base)), int(base) + 4)


def sample_tst(row) -> int:
    """
    Sample a start time for a trip based on constraints;
        must start within tst hour
        must finish within tet hour
        trip must have given duration
        must start at least 30 minutes after previous tet hour if exists
        must finish at least 30 minutes before next tst hour if exists
    Note that this can return negative duration activities, for example if duration is too long or short.
    """
    tst, tet, d, ptet, ntst = (
        row["tst"],
        row["tet"],
        row["duration"],
        row["ptet"],
        row["ntst"],
    )
    first_trip = ptet is None
    last_trip = ntst is None
    ptet = ptet if not first_trip else tst - 30
    ntst = ntst if not last_trip else tet + 30
    # if limited by previous trip, start after it ends + 30 minutes
    earliest = max(tst, tet - d, ptet + 30)
    # if limited by next trip, start at least 30 minutes before it starts
    latest = min(tst + 60, tet + 60 - d, ntst + 30 - d)
    if latest < earliest:
        if first_trip:
            return latest
        if last_trip:
            return earliest
        return int((tst + tet + 60 - d) / 2)
    return random.randint(earliest, latest)


def compute_base_intervals(tsts, tets, durs):
    """
    For each trip, compute:
      earliest possible start
      latest possible start
    using hour constraints only.
    """
    earliest = []
    latest = []
    for tst, tet, dur in zip(tsts, tets, durs):
        lo = max(tst, tet - dur)
        hi = min(tst + 60, tet + 60 - dur)
        earliest.append(lo)
        latest.append(hi)
    return earliest, latest


def tighten_intervals(earliest, latest, durations):
    """
    Enforce adjacency:
        s[i] >= s[i-1] + dur[i-1]
        s[i] + dur[i] <= s[i+1]
    Returns tightened (earliest, latest)
    """
    n = len(earliest)
    earliest_new = earliest[:]  # copy
    latest_new = latest[:]  # copy

    # Forward pass
    for i in range(1, n):
        earliest_new[i] = max(
            earliest_new[i], earliest_new[i - 1] + durations[i - 1]
        )

    # Backward pass
    for i in range(n - 2, -1, -1):
        latest_new[i] = min(latest_new[i], latest_new[i + 1] - durations[i])

    return earliest_new, latest_new


def find_infeasible_indices(earliest, latest):
    """Return indices where earliest[i] > latest[i]."""
    return [i for i in range(len(earliest)) if earliest[i] > latest[i]]


def reduce_durations(dur, infeasible, amount=1):
    """
    Reduce durations for infeasible rows.
    Simple policy: reduce by 'amount' minutes (default=1).
    Ensures duration never goes below 0 minutes.
    """
    for i in infeasible:
        dur[i] = max(0, dur[i] - amount)
        if i > 0:
            dur[i - 1] = max(0, dur[i - 1] - amount)
        if i < len(dur) - 1:
            dur[i + 1] = max(0, dur[i + 1] - amount)
    return dur


def sample_start_times(earliest, latest, durations, pid, seed):
    """
    Right-to-left sampling ensuring:
        s[i] + dur[i] <= s[i+1]
    """
    rng = random.Random(seed)
    n = len(earliest)
    starts = [None] * n

    # last trip
    hi = latest[n - 1]
    lo = earliest[n - 1]
    if lo > hi:
        print(f"Warning: Infeasible sample found at pid: {pid} trip: {n - 1}")
        starts[n - 1] = rng.randint(hi, lo)
    else:
        starts[n - 1] = rng.randint(earliest[n - 1], latest[n - 1])

    # right → left
    for i in range(n - 2, -1, -1):
        hi = min(latest[i], starts[i + 1] - durations[i])
        lo = earliest[i]
        if lo > hi:
            print(f"Warning:Negative sample found at pid: {pid} trip: {i}")
            starts[i] = rng.randint(hi, lo)
        else:
            starts[i] = rng.randint(lo, hi)

    return starts


def compute_feasible_schedule(tsts, tets, durations, pid, max_iter=7):
    """
    Recompute constraints repeatedly.
    If infeasible, reduce durations and try again.
    """
    _durations = durations[:]  # copy
    amount = 1
    for _ in range(max_iter):
        earliest, latest = compute_base_intervals(tsts, tets, _durations)
        earliest, latest = tighten_intervals(earliest, latest, _durations)
        infeasible = find_infeasible_indices(earliest, latest)

        if not infeasible:
            # success
            return earliest, latest, _durations

        # Otherwise reduce durations
        print(
            f"Warning: reducing durations by {amount} for infeasible pid: {pid} trips at indices: {infeasible}"
        )
        _durations = reduce_durations(_durations, infeasible, amount=amount)
        amount *= 2  # exponential backoff for faster convergence

    earliest, latest = compute_base_intervals(tsts, tets, durations)
    return earliest, latest, durations


def sample_plan_trip_start_times(trips: pl.DataFrame, seed=42) -> pl.DataFrame:
    # pull columns
    tsts = trips["tst"].to_list()
    tets = trips["tet"].to_list()
    durations = trips["duration"].to_list()

    # compute feasible constraints (auto-repairing durations)
    earliest, latest, durations = compute_feasible_schedule(
        tsts, tets, durations, pid=trips["pid"][0]
    )

    # sample feasible start times
    s = sample_start_times(
        earliest, latest, durations, pid=trips["pid"][0], seed=seed
    )
    e = [s[i] + durations[i] for i in range(len(s))]

    # return new DF
    return trips.with_columns(
        tst=pl.Series("tst", s, dtype=pl.Int64),
        tet=pl.Series("tet", e, dtype=pl.Int64),
        duration=pl.Series("duration", durations, dtype=pl.Int64),
    )


def preprocess_trips(
    trips: pl.DataFrame, config: dict, year: str, zone_mapping: dict
) -> pl.DataFrame:
    column_mapping = config["column_mappings"]
    trips = trips.select(column_mapping.keys()).rename(column_mapping)

    trips = trips.sort(["hid", "pid", "tid"]).with_columns(
        seq=pl.col("tid").rank(method="dense").over("hid", "pid").cast(pl.Int64)
    )

    mode_map = config["mode"]
    act_map = config["act"]

    trips = trips.with_columns(
        pl.col("mode").replace_strict(mode_map, default=pl.col("mode")),
        pl.col("oact").replace_strict(act_map, default=pl.col("oact")),
        pl.col("dact").replace_strict(act_map, default=pl.col("dact")),
    )

    trips = trips.with_columns(
        pl.col("duration").map_elements(sample_minute)
    ).with_columns(tst=pl.col("tst") * 60, tet=pl.col("tet") * 60)

    trips = fix.day_wrap(trips)
    # trips = filter.bad_trips(trips)

    # sample times
    trips = trips.group_by("pid", maintain_order=True).map_groups(
        lambda g: sample_plan_trip_start_times(g, seed=42)
    )

    trips = trips.with_columns(
        ozone=pl.col("ozone").replace_strict(zone_mapping),
        dzone=pl.col("dzone").replace_strict(zone_mapping),
    )

    return trips


def preprocess_stages(
    stages: pl.DataFrame, config: dict, year: str
) -> pl.DataFrame:
    column_mapping = config["column_mappings"]
    stages = (
        stages.select(column_mapping.keys())
        .rename(column_mapping)
        .with_columns(
            distance=pl.when(pl.col("distance") < 0)
            .then(None)
            .otherwise(pl.col("distance") * 1.6)
        )
    )

    stages = stages.group_by(["pid", "tid"]).agg(
        pl.col("distance").sum().alias("distance")
    )

    return stages.drop("pid")
