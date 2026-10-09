"""Loader for India's Time Use Survey (TUS)

Covers two rounds, each with a different raw layout:

- 2024: one merged household table (tus106HH) and one merged person/diary
  table (tus106PER, one row per diary episode with the person's
  demographics repeated on each).
- 2019: the same information split across four NSS tables -- Block 4
  (household), Block 3 (demographics, one row per member), Block 5 (day of
  week, one row per diary participant) and Block 6 (diary episodes).

Both rounds share one diary design. A person's day (04:00 to 04:00) is cut
into time slots (mostly 30 min, consecutive identical slots merged). Each
slot has one or more activities, each its own row with a full 3-digit
ICATUS 2016 code, flagged:

- major_activity_flag: "1" for the slot's main activity (exactly one per
  slot), "2" for any others ("minor");
- simultaneous_flag: in multi-activity slots, "1" if the activities were
  done at the same time, "2" if one after another (in serial-number order).

ICATUS has one "travelling related to ..." code per division; those rows
are trips, every other row is a stationary activity. See _resolve_slots for
how a multi-activity slot becomes a sequence of episodes, and load_trips
for how that sequence becomes trips.
"""

import mmap
from pathlib import Path

import polars as pl
from nesstar_converter import (
    NESSTAR_MAGIC,
    _parse_resource_layouts,
    extract_block_resource_indexed,
    parse_ddi,
)

from foundata.utils import table_joiner, table_stacker

SOURCE = "tus"
COUNTRY = "ind"

# Average annual EUR/INR rate for each round's survey year -- approximate,
# used only to turn household consumer expenditure into a hh_income proxy
# (see load_households).
INR_TO_EURO = {2024: 0.011, 2019: 0.0127}

ROUNDS = {
    2024: {
        "dir": "TUS2024",
        "nesstar": "TUS2024.Nesstar",
        "ddi": "DDI-IND-NSO-TUS-2024-24.xml",
        "blocks": {"hh": "tus106HH", "person": "tus106PER"},
    },
    2019: {
        "dir": "TUS2019",
        "nesstar": "TUS2019.Nesstar",
        "ddi": "DDI-IND-CSO-TUS-2019-19.xml",
        "blocks": {
            "hh": "LEVEL - 03 (Block 4)",
            "demographics": "LEVEL - 02 (Block 3)",
            "day": "LEVEL - 04 (Block 5)",
            "diary": "LEVEL - 05 (Block 6)",
        },
    },
}

PERSON_COLS = [
    "pid",
    "hid",
    "age",
    "sex",
    "relationship",
    "employment",
    "education",
    "day",
]
EPISODE_COLS = [
    "pid",
    "activity_serial_no",
    "major_activity_flag",
    "simultaneous_flag",
    "time_from",
    "time_to",
    "activity_code",
    "location",
]

# The diary runs 04:00 -> 04:00 next day. Clock times before 04:00 belong
# to the next day, so get +1440 to stay minutes-since-midnight of the diary
# day (as for every other source); the part of the last episode past 1440
# is truncated downstream like any source's last activity of the day.
DIARY_START = 4 * 60

# Cap on the time given to minor travel in a sequential slot -- see
# _resolve_slots.
MAX_MINOR_TRAVEL_MINS = 60

# Implied trips (see _add_implied_trips): activity types that, done outside
# the dwelling, mean the person must have travelled there, and how long
# that unrecorded trip is assumed to take.
IMPLIED_TRIP_ACTS = ["work", "education", "shop", "leisure", "visit", "escort"]
IMPLIED_TRIP_MINS = 10


def load_years(
    data_root: Path,
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
    years: tuple[int, ...] = (2024, 2019),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load and stack several TUS rounds found under `data_root`.

    Returns:
        (attributes, trips) DataFrames conforming to configs/core/template.yaml.
    """
    all_attributes, all_trips = [], []
    for year in years:
        round_dir = Path(data_root) / ROUNDS[year]["dir"]
        attributes, trips = load(
            round_dir / ROUNDS[year]["nesstar"],
            round_dir / ROUNDS[year]["ddi"],
            hh_config,
            person_config,
            trips_config,
            year,
        )
        all_attributes.append(attributes)
        all_trips.append(trips)
    return table_stacker(all_attributes), table_stacker(all_trips)


def load(
    nesstar_path: str | Path,
    ddi_path: str | Path,
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
    year: int = 2024,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load one TUS round (2024 or 2019) from its .Nesstar + DDI XML files.

    Returns:
        (attributes, trips) DataFrames conforming to configs/core/template.yaml.
    """
    if year not in ROUNDS:
        raise ValueError(
            f"Unsupported TUS year: {year} (expected 2024 or 2019)"
        )
    print(f"Loading TUS {year}...")

    blocks = ROUNDS[year]["blocks"]
    hh_mapping = hh_config["column_mappings"][str(year)]
    person_mappings = _person_mappings(person_config, year)
    raw = _load_pruned_blocks(
        Path(nesstar_path).expanduser(),
        Path(ddi_path).expanduser(),
        {
            blocks["hh"]: set(hh_mapping),
            **{blocks[k]: set(m) for k, m in person_mappings.items()},
        },
    )
    # rename raw columns, and build hid/pid on every person-level table
    tables = {
        k: raw[blocks[k]]
        .select(m.keys())
        .rename(m)
        .with_columns(hid=_hid_expr(year), pid=_pid_expr(year))
        for k, m in person_mappings.items()
    }

    if year == 2024:
        # demographics, day and diary all on one row per episode
        persons = tables["person"].unique(
            subset="pid", keep="first", maintain_order=True
        )
        episodes = tables["person"]
    else:
        persons = (
            tables["demographics"]
            .unique(subset="pid", keep="first", maintain_order=True)
            .join(tables["day"].select("pid", "day"), on="pid", how="left")
        )
        episodes = tables["diary"]
    persons = _finish_persons(persons.select(PERSON_COLS), person_config)
    # persons with no diary have an empty activity code
    episodes = episodes.filter(pl.col("activity_code") != "").select(
        EPISODE_COLS
    )

    hhs = load_households(raw[blocks["hh"]], hh_config, year)
    attributes = table_joiner(
        hhs, persons, on="hid", lhs_name="hh", rhs_name="person"
    )
    attributes = _diary_respondents_only(attributes, episodes)
    trips = load_trips(episodes, trips_config, attributes)
    return attributes, trips


def _person_mappings(person_config: dict, year: int) -> dict[str, dict]:
    """Person column mappings keyed by ROUNDS[year]["blocks"] key: 2024's
    single person table, or 2019's demographics/day/diary tables."""
    mappings = person_config["column_mappings"][str(year)]
    return {"person": mappings} if year == 2024 else mappings


def _load_pruned_blocks(
    nesstar_path: Path, ddi_path: Path, columns_by_block: dict[str, set[str]]
) -> dict[str, pl.DataFrame]:
    """Decode only `columns_by_block[name]` of each named DDI block.

    Resource-index matching (_parse_resource_layouts) scores blocks against
    descriptors using each block's *full* ddi_vars list, so that must run
    before pruning -- pruning only trims the list extract_block_resource_indexed
    then iterates over to decode columns.
    """
    blocks = parse_ddi(str(ddi_path))
    fid_by_name = {blk["name"]: fid for fid, blk in blocks.items()}

    result = {}
    with (
        open(nesstar_path, "rb") as f_handle,
        mmap.mmap(f_handle.fileno(), 0, access=mmap.ACCESS_READ) as data,
    ):
        if data[:8] != NESSTAR_MAGIC:
            raise ValueError(f"Not a valid Nesstar file: {nesstar_path}")

        resource_layouts = _parse_resource_layouts(data, blocks)

        for block_name, columns in columns_by_block.items():
            fid = fid_by_name[block_name]
            block_info = blocks[fid]
            if fid not in resource_layouts:
                raise ValueError(
                    f"{block_name} has no resource index -- pruned "
                    "extraction here only supports resource-indexed blocks"
                )

            pruned_vars = [
                v for v in block_info["ddi_vars"] if v["name"] in columns
            ]
            missing = columns - {v["name"] for v in pruned_vars}
            if missing:
                raise ValueError(f"{block_name} missing columns: {missing}")

            df = extract_block_resource_indexed(
                data,
                {**block_info, "ddi_vars": pruned_vars},
                resource_layouts[fid],
            )
            result[block_name] = pl.from_pandas(df)

    return result


def _hid_expr(year: int) -> pl.Expr:
    """hid = source + 2-digit round year + fsu serial + sample hhld no.

    The year prefix keeps 2019 and 2024 household numbering from colliding
    once both rounds are pooled into one dataset (run.py).
    """
    return (
        pl.lit(SOURCE)
        + pl.lit(str(year)[2:4])
        + pl.col("hid_fsu")
        + pl.col("hid_sample")
    )


def _pid_expr(year: int) -> pl.Expr:
    return _hid_expr(year) + pl.col("pid_person")


def load_households(
    hh_raw: pl.DataFrame, config: dict, year: int
) -> pl.DataFrame:
    column_mapping = config["column_mappings"][str(year)]
    hhs = hh_raw.select(column_mapping.keys()).rename(column_mapping)

    if "ownership" not in hhs.columns:
        # 2019's schedule has no dwelling-tenure question at all
        hhs = hhs.with_columns(ownership=pl.lit(None, dtype=pl.String))

    return hhs.with_columns(
        hid=_hid_expr(year),
        hh_size=pl.col("hh_size").cast(pl.Int32, strict=False),
        year=pl.col("year").cast(pl.Int32, strict=False),
        hh_zone=pl.col("hh_zone")
        .replace_strict(config["hh_zone"])
        .fill_null("unknown"),
        ownership=pl.col("ownership")
        .replace_strict(config["ownership"], default="unknown")
        .fill_null("unknown"),
        # Monthly household consumer expenditure (INR) is the closest TUS has
        # to income -- annualise and convert to EUR as a proxy.
        hh_income=(
            pl.col("hh_income")
            .str.strip_chars()
            .replace("", None)
            .cast(pl.Float64, strict=False)
            * 12
            * INR_TO_EURO[year]
        ).cast(pl.Int32),
        weight=pl.col("weight").cast(pl.Float64, strict=False).cast(pl.Float32),
        country=pl.lit(COUNTRY),
        source=pl.lit(SOURCE),
    ).drop("hid_fsu", "hid_sample")


def _finish_persons(persons: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Value mappings and defaults for the one-row-per-person table."""
    return persons.with_columns(
        age=pl.col("age").cast(pl.Int32, strict=False),
        sex=pl.col("sex").replace_strict(config["sex"]).fill_null("unknown"),
        relationship=pl.col("relationship")
        .replace_strict(config["relationship"], default="unknown")
        .fill_null("self"),
        employment=pl.col("employment")
        .replace_strict(config["employment"], default="unknown")
        .fill_null("unknown"),
        education=pl.col("education")
        .replace_strict(config["education"], default="unknown")
        .fill_null("unknown"),
        day=pl.col("day")
        .replace_strict(config["day"], default="unknown")
        .fill_null("unknown"),
        occupation=pl.lit("unknown"),
        disability=pl.lit("unknown"),
        has_licence=pl.lit("unknown"),
        can_wfh=pl.lit("unknown"),
        race=pl.lit("unknown"),
        dwelling=pl.lit("unknown"),
        vehicles=pl.lit(None, dtype=pl.Int32),
        month=pl.lit(None, dtype=pl.Int32),
    )


def _diary_respondents_only(
    attributes: pl.DataFrame, episodes: pl.DataFrame
) -> pl.DataFrame:
    """Drop household members with no diary (mostly under-6s, who are not
    asked to fill one in). Without this they would come out as a default
    "home all day" plan, indistinguishable from a respondent who genuinely
    never left home. hh_size still counts them, as it should.
    """
    n = attributes.height
    attributes = attributes.join(
        episodes.select("pid").unique(),
        on="pid",
        how="semi",
        maintain_order="left",
    )
    nn = n - attributes.height
    print(
        f"Removed {nn}/{n} TUS persons with no time-use diary "
        f"({100 * nn / max(n, 1):.1f}%)"
    )
    return attributes


def load_trips(
    episodes: pl.DataFrame, config: dict, attributes: pl.DataFrame
) -> pl.DataFrame:
    """Turn diary episodes (EPISODE_COLS, all rows of every slot) into trips.

    1. _resolve_slots turns each slot into a short non-overlapping sequence
       of episodes, so the whole diary is one ordered sequence.
    2. _add_implied_trips adds travel the diary leaves out, where the
       person must have moved to or from work, education or a shop.
    3. Every run of consecutive travel episodes becomes one trip (e.g.
       commuting then work-related travel, or the same travel split over
       two slots with different minor activities). Every run of consecutive
       stationary episodes between two trips (a "gap") is squashed to a
       single activity -- the act type with the most total time in that
       gap. A trip's oact is the gap before it and its dact the gap after
       it, read off the same per-gap table, so activity chains are
       consistent by construction.
    """
    travel_codes = pl.Series(config["travel_codes"]["default"]).implode()
    episodes = _resolve_slots(
        episodes.with_columns(
            activity_serial_no=pl.col("activity_serial_no").cast(pl.Int32),
            tst=_diary_minutes("time_from", end=False),
            tet=_diary_minutes("time_to", end=True),
            is_travel=pl.col("activity_code").is_in(travel_codes),
            act=pl.col("activity_code").replace_strict(
                config["act_mappings"]["default"], default="unknown"
            ),
        )
    )
    episodes = _add_implied_trips(episodes)

    # gap id: number of trips started so far, so trip k (all its travel
    # episodes) and the gap after it share id k, and the gap before it is k-1
    episodes = episodes.with_columns(
        trip_start=pl.col("is_travel")
        & ~pl.col("is_travel").shift(1, fill_value=False).over("pid")
    ).with_columns(
        gap=pl.col("trip_start").cast(pl.Int32).cum_sum().over("pid")
    )
    gap_act = (
        episodes.filter(~pl.col("is_travel"))
        .group_by("pid", "gap", "act")
        .agg(dur=(pl.col("tet") - pl.col("tst")).sum())
        .sort(["dur", "act"], descending=[True, False])
        .group_by("pid", "gap", maintain_order=True)
        .agg(act=pl.first("act"))
    )

    trips = (
        episodes.filter(pl.col("is_travel"))
        .group_by("pid", "gap", maintain_order=True)
        .agg(tst=pl.col("tst").first(), tet=pl.col("tet").last())
        .with_columns(
            seq=(pl.int_range(pl.len()).over("pid")).cast(pl.Int8),
            prev_gap=pl.col("gap") - 1,
        )
        .join(
            gap_act.rename({"gap": "prev_gap", "act": "oact"}),
            on=["pid", "prev_gap"],
            how="left",
        )
        .join(gap_act.rename({"act": "dact"}), on=["pid", "gap"], how="left")
        .with_columns(
            oact=pl.col("oact").fill_null("unknown"),
            dact=pl.col("dact").fill_null("unknown"),
            mode=pl.lit("unknown"),
            distance=pl.lit(None, dtype=pl.Float32),
        )
    )

    # TUS has no trip-level geography (only inside/outside the dwelling), so
    # both ends get the household's own zone, on the assumption that most
    # within-day travel does not cross the rural/urban boundary.
    return trips.join(
        attributes.select("pid", ozone="hh_zone"), on="pid", how="left"
    ).select(
        "pid",
        "seq",
        "tst",
        "tet",
        "oact",
        "dact",
        "mode",
        "distance",
        "ozone",
        dzone="ozone",
    )


def _resolve_slots(episodes: pl.DataFrame) -> pl.DataFrame:
    """Reduce each diary slot (all rows sharing pid, tst, tet) to a
    non-overlapping sequence of episodes:

    - no travel in the slot: keep the major activity for the whole slot;
    - major activity is travel, or travel is done simultaneously with the
      major activity (e.g. chatting while travelling): the whole slot is
      one travel episode;
    - minor travel done before/after the major activity (sequential slot,
      e.g. walk to the shop, shop, walk back): keep the major activity and
      the travel rows in serial-number order (which is the order they
      happened in). Travel gets half the slot, capped at
      MAX_MINOR_TRAVEL_MINS, shared equally between the travel rows; the
      major activity gets the rest.

    Other minor (non-travel) activities are dropped; their time goes to the
    major activity.
    """
    slot = ["pid", "tst", "tet"]
    is_major = pl.col("major_activity_flag") == "1"
    slot_has_travel = pl.col("is_travel").any().over(slot)
    whole_slot_travel = (pl.col("is_travel") & is_major).any().over(slot) | (
        slot_has_travel & (pl.col("simultaneous_flag") == "1").any().over(slot)
    )

    episodes = (
        episodes.sort("pid", "tst", "activity_serial_no")
        .with_columns(whole_slot_travel=whole_slot_travel)
        .filter(
            pl.when(pl.col("whole_slot_travel"))
            .then(pl.col("is_travel"))
            .otherwise(is_major | pl.col("is_travel"))
        )
        # a whole-travel slot keeps just its first travel row
        .filter(
            ~pl.col("whole_slot_travel")
            | (pl.int_range(pl.len()).over(slot) == 0)
        )
    )

    # share the slot's duration out along its (1 or more) remaining rows
    slot_dur = pl.col("tet") - pl.col("tst")
    n_travel = pl.col("is_travel").sum().over(slot)
    travel_each = (
        pl.min_horizontal(slot_dur // 2, MAX_MINOR_TRAVEL_MINS) // n_travel
    )
    # a stationary major plus minor travel (every other slot is one row now)
    split = (n_travel > 0) & (n_travel < pl.len().over(slot))
    dur = (
        pl.when(~split)
        .then(slot_dur)
        .when(pl.col("is_travel"))
        .then(travel_each)
        .otherwise(slot_dur - n_travel * travel_each)
    )
    end = pl.col("tst") + pl.col("dur").cum_sum().over(slot)
    return (
        episodes.with_columns(dur=dur)
        .with_columns(tet=end.cast(pl.Int32))
        .with_columns(tst=(pl.col("tet") - pl.col("dur")).cast(pl.Int32))
        .drop("whole_slot_travel", "dur")
    )


def _add_implied_trips(episodes: pl.DataFrame) -> pl.DataFrame:
    """Add the trips the diary leaves out.

    The diary records whether each activity was done inside or outside the
    dwelling, and most moves between the two are not recorded as travel
    (59% in 2024). "Outside" includes the yard or street, so not every
    move is a trip -- one is added only between two consecutive stationary
    episodes where:

    - the location changes (inside <-> outside),
    - the activity type changes (so never home -> home), and
    - the outside one is an IMPLIED_TRIP_ACTS type (e.g. work, shop),
      which must happen somewhere else.

    The trip takes the last IMPLIED_TRIP_MINS of the earlier episode
    (at most half of it).
    """

    def prev(col: str) -> pl.Expr:
        return pl.col(col).shift(1).over("pid")

    outside_trip_act = (pl.col("location") == "2") & pl.col("act").is_in(
        IMPLIED_TRIP_ACTS
    )
    episodes = episodes.with_columns(
        implied_trip=~pl.col("is_travel")
        & ~prev("is_travel")
        & pl.col("location").is_in(["1", "2"])
        & prev("location").is_in(["1", "2"])
        & (pl.col("location") != prev("location"))
        & (pl.col("act") != prev("act"))
        & (outside_trip_act | outside_trip_act.shift(1).over("pid")),
        trip_mins=pl.min_horizontal(
            IMPLIED_TRIP_MINS, (prev("tet") - prev("tst")) // 2
        ),
    )
    trips = episodes.filter(pl.col("implied_trip")).with_columns(
        tet=pl.col("tst"),
        tst=pl.col("tst") - pl.col("trip_mins"),
        is_travel=pl.lit(True),
        act=pl.lit(None, dtype=pl.String),
    )
    print(f"Added {trips.height} TUS trips implied by a change of location")
    # make room for each trip at the end of the episode before it
    episodes = episodes.with_columns(
        tet=pl.col("tet")
        - pl.when(pl.col("implied_trip").shift(-1).over("pid"))
        .then(pl.col("trip_mins").shift(-1).over("pid"))
        .otherwise(0)
    )
    return (
        pl.concat([episodes, trips])
        .sort("pid", "tst")
        .drop("implied_trip", "trip_mins")
    )


def _diary_minutes(col: str, end: bool) -> pl.Expr:
    """ "HH:MM" -> minutes since midnight of the diary day. Times before the
    04:00 diary start are on the next day; an end time of exactly 04:00 is
    the end of the diary, also on the next day."""
    hh = pl.col(col).str.slice(0, 2).cast(pl.Int32)
    mins = hh * 60 + pl.col(col).str.slice(3, 2).cast(pl.Int32)
    next_day = mins <= DIARY_START if end else mins < DIARY_START
    return pl.when(next_day).then(mins + 1440).otherwise(mins)
