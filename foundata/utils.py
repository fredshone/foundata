import functools
import random
from pathlib import Path
from typing import Iterable

import polars as pl
import yaml
from loguru import logger
from rapidfuzz import fuzz, process

from foundata import table_logger
from foundata.post_process import (
    activities_to_trips,
    trips_to_activity_schedules,
)

DTYPE_MAP = {
    "int8": pl.Int8,
    "int16": pl.Int16,
    "int32": pl.Int32,
    "int64": pl.Int64,
    "float32": pl.Float32,
    "float64": pl.Float64,
    "int": pl.Int32,
    "integer": pl.Int32,
    "float": pl.Float32,
    "string": pl.String,
    "str": pl.String,
    "bool": pl.Boolean,
    "boolean": pl.Boolean,
    "date": pl.Date,
    "datetime": pl.Datetime,
}


def fuzzy_loader(path: str | Path, target: str, **kwargs) -> pl.DataFrame:
    # look in given path for closest math to target
    candidates = [f.name for f in path.iterdir()]
    if not candidates:
        raise FileNotFoundError(f"No file found in {path}")
    # fuzzy find best candidate
    best_match, score, idx = process.extractOne(
        target, candidates, scorer=fuzz.ratio, score_cutoff=80
    )
    if best_match is None:
        raise FileNotFoundError(
            f"No file found in {path} matching {target} (best match: {best_match} with score {score})"
        )
    logger.debug(
        f"Fuzzy match loading {best_match} from {path} (score: {int(score)}%)"
    )
    return pl.read_csv(path / candidates[idx], **kwargs)


def expand_root(root: str | Path) -> Path:
    return Path(root).expanduser()


def load_yaml_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def check_overlap(
    lhs: pl.DataFrame,
    rhs: pl.DataFrame,
    on: str,
    lhs_name: str = "lhs",
    rhs_name: str = "rhs",
) -> set:
    on_lhs = set(lhs[on])
    on_rhs = set(rhs[on])
    missing_in_lhs = on_rhs - on_lhs
    missing_in_rhs = on_lhs - on_rhs
    n_lhs = len(missing_in_lhs)
    n_rhs = len(missing_in_rhs)
    perc_lhs = n_lhs / len(on_rhs) * 100
    perc_rhs = n_rhs / len(on_lhs) * 100

    if missing_in_lhs:
        n = min(n_lhs, 3)
        table_logger.note(
            f"Missing {n_lhs} ({perc_lhs:.1f}%) of '{on}' keys in '{lhs_name}': {list(missing_in_lhs)[:n]}"
        )
    else:
        logger.debug(
            f"All '{on}' keys in '{rhs_name}' are present in '{lhs_name}'"
        )

    if missing_in_rhs:
        n = min(n_rhs, 3)
        table_logger.note(
            f"Missing {n_rhs} ({perc_rhs:.1f}%) of '{on}' keys in '{rhs_name}': {list(missing_in_rhs)[:n]}"
        )
    else:
        logger.debug(
            f"All '{on}' keys in '{lhs_name}' are present in '{rhs_name}'"
        )


def table_joiner(
    lhs: pl.DataFrame,
    rhs: pl.DataFrame,
    on: str,
    lhs_name: str = "lhs",
    rhs_name: str = "rhs",
    maintain_order: str = "left_right",
) -> pl.DataFrame:
    # check for missing keys
    check_overlap(lhs, rhs, on, lhs_name=lhs_name, rhs_name=rhs_name)

    # check for duplicates
    a_cols = lhs.columns
    b_cols = rhs.columns
    duplicates = set(a_cols) & set(b_cols) - {on}
    if duplicates:
        logger.warning(f"Duplicate columns (other than join key): {duplicates}")

    return lhs.join(rhs, on=on, maintain_order=maintain_order)


def table_stacker(tables: Iterable[pl.DataFrame]) -> pl.DataFrame:
    all_columns = set()
    for table in tables:
        all_columns.update(table.columns)
    for idx, table in enumerate(tables):
        missing_columns = all_columns - set(table.columns)
        if missing_columns:
            raise UserWarning(
                f"Cannot stack tables: Table {idx} is missing columns: {missing_columns}"
            )
    base_order = tables[0].columns
    return pl.concat([table.select(base_order) for table in tables])


def enumerate_ids(
    attributes: pl.DataFrame,
    trips: pl.DataFrame,
    columns: Iterable[str] = ("pid", "hid"),
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, pl.Series]]:
    """*attributes* and *trips* with each of *columns* (a source's string
    ids) replaced by Int32 ids numbered from 0 in sorted id order, and per
    column the original ids, where id ``i`` is ``keys[col][i]``.

    Both tables are numbered from the same keys, so they still join. The
    runner offsets the ids per source to keep them unique once combined.
    """
    all_keys = {}
    for col in columns:
        tables = [df for df in (attributes, trips) if col in df.columns]
        keys = (
            pl.concat([df[col].cast(pl.String) for df in tables])
            .drop_nulls()
            .unique()
            .sort()
        )
        # an Enum of the sorted keys numbers them in order
        ids = pl.Enum(keys)
        to_id = (
            pl.col(col).cast(pl.String).cast(ids).to_physical().cast(pl.Int32)
        )
        if col in attributes.columns:
            attributes = attributes.with_columns(to_id)
        if col in trips.columns:
            trips = trips.with_columns(to_id)
        all_keys[col] = keys
    return attributes, trips, all_keys


def config_for_year(config: dict, year):
    cnfg = {}
    for key, value in config.items():
        if year in value:
            cnfg[key] = value[year]
        else:
            cnfg[key] = value["default"]
    return cnfg


def bounds_from_list(bounds: list[str]) -> tuple[int, int]:
    return int(bounds[0]), int(bounds[1])


def sample_to_euro(bounds, rate=1.0):
    if len(bounds) == 1:
        return None
    a, b = bounds
    return int(random.randint(int(a), int(b)) * rate)


def resolve_activity_chain(
    data: pl.DataFrame, group_cols: list[str]
) -> pl.DataFrame:
    """
    Resolve round-trip destination activities and chain origin activities.

    Some trips have no fixed destination (e.g. ODiN's Doel code 10
    "toeren/wandelen", or NTS's TripPurpTo_B01ID code 15 "Day trip/just
    walk") and are mapped to a null `dact`. These are resolved to the last
    known real activity within `group_cols` (i.e. they return to wherever
    the traveller came from), cascading correctly across consecutive round
    trips. The first trip within a group falls back to its own `oact` if it
    is itself a round trip.

    Each trip's `oact` is then set to the previous trip's (resolved) `dact`,
    chaining activities across the group (e.g. a person's day). If a
    group's first trip is itself a round trip on both ends (no real
    activity anywhere in the group to anchor to — e.g. `oact` shares the
    same round-trip vocabulary as `dact`, unlike ODiN's VertLoc-derived
    `oact`), the remaining nulls are filled with "unknown".
    """
    data = data.sort(group_cols + ["seq"]).with_columns(
        dact=pl.coalesce(
            pl.col("dact"), pl.when(pl.col("seq") == 1).then(pl.col("oact"))
        )
        .forward_fill()
        .over(group_cols)
        .fill_null("unknown")
    )
    return data.with_columns(
        oact=pl.col("dact")
        .shift(1)
        .over(group_cols)
        .fill_null(pl.col("oact"))
        .fill_null("unknown")
    )


def combine_consecutive_acts(
    trips: pl.DataFrame,
    non_consecutive_types: list[str] = ["home", "work", "education"],
    on: str = "pid",
) -> pl.DataFrame:
    """Combine consecutive activities of the same non-consecutive type.

    Where a trip's destination activity is the same as the activity
    immediately before it (either its own origin, for a "there and back"
    trip, or the previous trip's destination, for a chain broken
    elsewhere), that trip is removed rather than filtering out the whole
    plan. Removing it merges the two same-type activities either side of
    it into one, instead of leaving an artificial extra activity in the
    plan.

    Args:
        trips: DataFrame of trips with columns matching `on`, "seq", "oact", "dact".
        non_consecutive_types: List of activity types that are not allowed to appear consecutively (e.g. "work", "education").
        on: Column name to join on (default "pid").

    Returns:
        trips DataFrame with the redundant trips removed.
    """
    n = len(trips)
    redundant = (
        trips.sort(on, "seq")
        .with_columns(prev_dact=pl.col("dact").shift(1).over(on))
        .filter(
            (
                (pl.col("oact") == pl.col("dact"))
                | (pl.col("dact") == pl.col("prev_dact"))
            )
            & pl.col("dact").is_in(non_consecutive_types)
        )
        .select(on, "seq")
    )
    clean_trips = trips.join(
        redundant, on=[on, "seq"], how="anti", maintain_order="left"
    )
    nn = n - len(clean_trips)
    if nn:
        table_logger.record(
            "combined consec",
            f"{100 * nn / n:.1f}%",
            f"Combined {nn}/{n} trips with consecutive activities of the same non-consecutive types ({100 * nn / n:.1f}%)",
        )
    return clean_trips


def odin_equivalence(num_adults, num_children):
    """Compute equivalence factor for household based on number of adults and children.

    Based on ODIN equivalence scale (https://www.odin.dk/en/odin-equivalence-scale/).
    """
    return (num_adults + (0.8 * num_children)) ** 0.5


def get_config_path(*parts: str) -> Path:
    return Path(__file__).resolve().parent.parent.joinpath("configs", *parts)


def template() -> Path:
    return Path(__file__).parent.parent / "configs" / "core" / "template.yaml"


def compute_avg_speed(
    attributes: pl.DataFrame, trips: pl.DataFrame, on: str = "pid"
) -> pl.DataFrame:
    """Add avg_speed (km/h) column to attributes.

    Computed as total trip distance / total trip duration per `on` group
    (default "pid"). Null for groups with no valid trips or zero total
    duration.
    """
    # check distances are not null and durations are positive to avoid skewing avg_speed
    if not trips.select(pl.col("distance").is_not_null()).to_series().all():
        table_logger.note(
            "Some trips have null distance — these will be excluded from avg_speed calculation",
            level="WARNING",
        )
    if not trips.select(pl.col("tet") > pl.col("tst")).to_series().all():
        table_logger.note(
            "Some trips have non-positive duration (tet <= tst) — these will be excluded from avg_speed calculation",
            level="WARNING",
        )

    speed = (
        trips.with_columns(duration=pl.col("tet") - pl.col("tst"))
        .filter(pl.col("duration") > 0)
        .filter(pl.col("distance").is_not_null())
        .group_by(on)
        .agg(
            total_distance=pl.col("distance").sum(),
            total_duration=pl.col("duration").sum(),
        )
        .with_columns(
            avg_speed=(
                pl.col("total_distance") / (pl.col("total_duration") / 60)
            ).cast(pl.Float32)
        )
        .select(on, "avg_speed")
    )
    attributes = attributes.join(speed, on=on, how="left")
    return attributes


ADULT_AGE = 18


def household_composition(
    attributes: pl.DataFrame, adult_age: int = ADULT_AGE
) -> pl.DataFrame:
    """Add hh_adults and hh_children (Int32) counted from household members.

    Loaders that set hh_adults and hh_children themselves (from the source's
    own counts) are left alone. Otherwise each household's persons are
    counted by age (adults are `adult_age` or over), but only where the
    attributes hold the whole household: as many persons as hh_size, each
    with a known age. Other households get null. Sources without a household
    id (hid = pid) therefore only get counts for one-person households.

    Must run before persons are filtered out, and only for sources with one
    row per person (not per person-day).
    """
    if {"hh_adults", "hh_children"} <= set(attributes.columns):
        return attributes.with_columns(
            pl.col("hh_adults", "hh_children").cast(pl.Int32, strict=False)
        )
    hhs = (
        attributes.group_by("hid")
        .agg(pl.col("hh_size").cast(pl.Int32, strict=False).first())
        .filter(pl.col("hh_size").is_not_null())
    )
    counts = add_member_counts(hhs, attributes, adult_age).drop("hh_size")
    return attributes.drop("hh_adults", "hh_children", strict=False).join(
        counts, on="hid", how="left"
    )


def member_counts(
    persons: pl.DataFrame, adult_age: int = ADULT_AGE
) -> pl.DataFrame:
    """hid, hh_adults and hh_children (Int32) of each household in `persons`.

    `persons` holds one row per household member with hid and age. Counts
    are null for households with any member of unknown (null) age. Loaders
    whose person table is the full roster can use this before dropping
    persons who weren't asked the travel questions.
    """
    age = pl.col("age").cast(pl.Int32, strict=False)
    aged = age.is_not_null().all()
    return persons.group_by("hid").agg(
        hh_adults=pl.when(aged).then((age >= adult_age).sum()).cast(pl.Int32),
        hh_children=pl.when(aged).then((age < adult_age).sum()).cast(pl.Int32),
    )


def add_member_counts(
    hhs: pl.DataFrame, persons: pl.DataFrame, adult_age: int = ADULT_AGE
) -> pl.DataFrame:
    """Join hh_adults and hh_children, counted over `persons`, onto `hhs`.

    For loaders whose person table is the household roster: call it before
    dropping members who kept no diary. Households whose roster holds a
    different number of members than their hh_size (where known) get null.
    """
    counts = member_counts(persons, adult_age).join(
        persons.group_by("hid").agg(members=pl.len()), on="hid"
    )
    size = pl.col("hh_size").cast(pl.Int32, strict=False)
    complete = size.is_null() | (pl.col("members") == size)
    return (
        hhs.join(counts, on="hid", how="left")
        .with_columns(
            pl.when(complete).then(pl.col("hh_adults", "hh_children"))
        )
        .drop("members")
    )


def vehicles_per_adult(attributes: pl.DataFrame) -> pl.DataFrame:
    """Add veh_per_adult, household vehicles / hh_adults as a category.

    "0" for no vehicles, "0-1" for fewer vehicles than adults, "1+" for at
    least one vehicle per adult (including vehicles with no adults), and
    "unknown" where either count is (or the source has no vehicles).
    """
    if "vehicles" not in attributes.columns:
        return attributes.with_columns(veh_per_adult=pl.lit("unknown"))
    vehicles = pl.col("vehicles").cast(pl.Int32, strict=False)
    adults = pl.col("hh_adults").cast(pl.Int32, strict=False)
    return attributes.with_columns(
        veh_per_adult=pl.when(vehicles.is_null())
        .then(pl.lit("unknown"))
        .when(vehicles == 0)
        .then(pl.lit("0"))
        .when(adults.is_null())
        .then(pl.lit("unknown"))
        .when(vehicles < adults)
        .then(pl.lit("0-1"))
        .otherwise(pl.lit("1+"))
    )


def split_employment_type(attributes: pl.DataFrame) -> pl.DataFrame:
    """Split the ft/pt distinction out of employment into employed_type.

    "ft-employed" and "pt-employed" collapse to "employed", with the split
    preserved in the new "employed_type" column ("ft"/"pt"). Bare "employed"
    (ft/pt distinction unknown) also maps to "unknown" employed_type, same
    as "unknown"/null employment. Categories the split doesn't apply to
    ("student", "unemployed", "retired", "other", "void") get "void".
    """
    attributes = attributes.with_columns(
        employed_type=pl.when(pl.col("employment") == "ft-employed")
        .then(pl.lit("ft"))
        .when(pl.col("employment") == "pt-employed")
        .then(pl.lit("pt"))
        .when(pl.col("employment") == "employed")
        .then(pl.lit("unknown"))
        .when(
            (pl.col("employment") == "unknown") | pl.col("employment").is_null()
        )
        .then(pl.lit("unknown"))
        .otherwise(pl.lit("void"))
    )
    return attributes.with_columns(
        employment=pl.col("employment").replace(
            {"ft-employed": "employed", "pt-employed": "employed"}
        )
    )


def correct_child_employment(
    attributes: pl.DataFrame, void_age_max: int = 5, student_age_max: int = 16
) -> pl.DataFrame:
    """Force employment/employed_type for children based on age.

    Children aged 0 to `void_age_max` are below school age, so employment
    status doesn't apply and is forced to "void". Children aged
    `void_age_max + 1` to `student_age_max` are of compulsory school age,
    so employment is forced to "student". In both cases the ft/pt split
    doesn't apply, so `employed_type` is forced to "void" too.

    Must run after `split_employment_type()`, which is what populates
    `employed_type` in the first place.
    """
    attributes = attributes.with_columns(
        employment=pl.when(pl.col("age") <= void_age_max)
        .then(pl.lit("void"))
        .when(pl.col("age") <= student_age_max)
        .then(pl.lit("student"))
        .otherwise(pl.col("employment"))
    )
    return attributes.with_columns(
        employed_type=pl.when(pl.col("age") <= student_age_max)
        .then(pl.lit("void"))
        .otherwise(pl.col("employed_type"))
    )


def assign_retired(
    attributes: pl.DataFrame, trips: pl.DataFrame, age_threshold: int = 60
) -> pl.DataFrame:
    """Reclassify unemployed persons as retired based on age and trip purpose.

    Some sources (e.g. ktdb) have no "retired" category and instead code
    everyone who isn't a student or employed as "unemployed". A person is
    reclassified as "retired" when they are "unemployed", aged
    `age_threshold` or older, and make no trips with purpose "work" (dact).
    """
    has_work_trip = (
        trips.filter(pl.col("dact") == "work")
        .select("pid")
        .unique()
        .with_columns(has_work_trip=pl.lit(True))
    )
    attributes = attributes.join(has_work_trip, on="pid", how="left")
    attributes = attributes.with_columns(
        employment=pl.when(
            (pl.col("employment") == "unemployed")
            & (pl.col("age") >= age_threshold)
            & pl.col("has_work_trip").is_null()
        )
        .then(pl.lit("retired"))
        .otherwise(pl.col("employment"))
    )
    return attributes.drop("has_work_trip")


def assign_education_to_escort(trips: pl.DataFrame) -> pl.DataFrame:
    """Assign education trips to escort trips.

    The LTDS data combined education and escort education trip purposes
    into a single "education" category. This function calculates the associated
    activity durations for trips, if an "education" activity has a duration less
    than 60 minutes, it is assumed to be an escort trip and the activity is
    reassigned to "escort". The trips are then updated to reflect the new activity
    assignments.
    """
    # convert trips to activities to calculate durations
    activities = trips_to_activity_schedules(None, trips)
    activities = activities.with_columns(
        duration=pl.col("end") - pl.col("start")
    )
    # where act is "education" and duration < 60, assign to escort
    activities = activities.with_columns(
        pl.when((pl.col("act") == "education") & (pl.col("duration") < 60))
        .then(pl.lit("escort"))
        .otherwise(pl.col("act"))
        .alias("act")
    )

    #  turn back into trips
    new_acts = activities_to_trips(activities).select(
        "oact", "dact", "pid", "seq"
    )

    # join back to trips and update oact/dact
    trips = trips.join(new_acts, on=["pid", "seq"], how="left")
    trips = trips.with_columns(
        oact=pl.col("oact_right"), dact=pl.col("dact_right")
    ).drop(["oact_right", "dact_right"])
    return trips


@functools.lru_cache(maxsize=1)
def _load_template() -> dict:
    with open(template(), encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_template_attributes() -> dict:
    return _load_template()["attributes"]


def get_template_trips() -> dict:
    return _load_template()["trips"]


def norm_weights(
    attributes: pl.DataFrame, weight_col: str = "weight"
) -> pl.DataFrame:
    # norm weights to average 1
    if weight_col not in attributes.columns:
        raise ValueError(
            f"Weight column '{weight_col}' not found in attributes"
        )
    if (
        not attributes.select(pl.col(weight_col).is_not_null())
        .to_series()
        .all()
    ):
        table_logger.note(
            "Some weights are null — these will be treated as zero in normalization",
            level="WARNING",
        )
    # check for non-positive weights to avoid skewing normalization
    if (
        not attributes.select(pl.col(weight_col).fill_null(0).gt(0))
        .to_series()
        .all()
    ):
        table_logger.note(
            "Some weights are non-positive (<= 0) — these will be treated as zero in normalization",
            level="WARNING",
        )
        attributes = attributes.with_columns(
            pl.col(weight_col)
            .fill_null(0)
            .clip(lower_bound=0)
            .alias(weight_col)
        )
    avg_weight = attributes[weight_col].fill_null(0).mean()
    if avg_weight == 0:
        table_logger.note("Total weight is zero — returning all weights as 1")
        return attributes.with_columns(
            pl.lit(1, dtype=pl.Float32).alias(weight_col)
        )
    return attributes.with_columns(
        (pl.col(weight_col).fill_null(0) / avg_weight)
        .cast(pl.Float32)
        .alias(weight_col)
    )


def add_implied_trips(
    episodes: pl.DataFrame,
    implied_trip: pl.Expr,
    trip_mins: pl.Expr,
    message: str,
    null_cols: tuple[str, ...] = ("act",),
) -> pl.DataFrame:
    """Splice a trip between two consecutive stationary episodes wherever
    `implied_trip` is true, taking `trip_mins` minutes (already clipped to
    at most half the earlier episode's duration) from the end of the
    earlier episode. `episodes` must be sorted by pid, tst."""
    episodes = episodes.with_columns(
        implied_trip=implied_trip, trip_mins=trip_mins
    )
    trips = episodes.filter(pl.col("implied_trip")).with_columns(
        tet=pl.col("tst"),
        tst=pl.col("tst") - pl.col("trip_mins"),
        is_travel=pl.lit(True),
        **{c: pl.lit(None, dtype=pl.String) for c in null_cols},
    )
    logger.debug(f"Added {trips.height} {message}")
    # make room for each trip at the end of the episode before it
    episodes = episodes.with_columns(
        tet=pl.col("tet")
        - pl.when(pl.col("implied_trip").shift(-1).over("pid"))
        .then(pl.col("trip_mins").shift(-1).over("pid"))
        .otherwise(0)
    )
    return (
        pl.concat([episodes, trips.select(episodes.columns)])
        .sort("pid", "tst", "tet")
        .drop("implied_trip", "trip_mins")
    )
