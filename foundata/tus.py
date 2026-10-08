"""Loader for India's Time Use Survey (TUS)

Covers two rounds, each with a different raw layout:

- 2024: one merged household table (tus106HH) and one merged person/diary
  table (tus106PER) -- a diary is a sequence of ~15-45 minute activity
  episodes per person, each coded with a 3-digit ICATUS 2016 activity code.
- 2019: the same information split across four separate NSS tables --
  Block 3 (person demographics, one row per member),
  Block 4 (household),
  Block 5 (day of week + type of day, one row per diary participant) and
  Block 6 (diary episodes) -- joined back together here on (hid, pid).

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

from foundata import fix
from foundata.utils import table_joiner, table_stacker

SOURCE = "tus"
COUNTRY = "ind"

# Average annual EUR/INR rate for each round's survey year -- approximate,
# used only to turn household consumer expenditure into a hh_income proxy
# (see load_households).
INR_TO_EURO = {2024: 0.011, 2019: 0.0127}

BLOCKS_2024 = {"hh": "tus106HH", "person": "tus106PER"}
BLOCKS_2019 = {
    "hh": "LEVEL - 03 (Block 4)",
    "demographics": "LEVEL - 02 (Block 3)",
    "day": "LEVEL - 04 (Block 5)",
    "diary": "LEVEL - 05 (Block 6)",
}


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


def load_years(
    data_root: Path,
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
    years: list[int] = [2024, 2019],
) -> dict[int, tuple[pl.DataFrame, pl.DataFrame]]:
    """Load and normalise multiple rounds of TUS survey data.

    Args:
        data_root: path to the root directory containing the TUS data.
        hh_config: parsed hh_dictionary.yaml config.
        person_config: parsed person_dictionary.yaml config.
        trips_config: parsed trip_dictionary.yaml config.
        years: which rounds to load -- 2024 and/or 2019. Picks the matching
            year-keyed column_mappings and raw table layout.
            Returns:
                A dictionary mapping each year to a tuple of (attributes, trips) DataFrames.
    """
    ROUNDS = {
        2024: {
            "nesstar": data_root / "TUS2024" / "TUS2024.Nesstar",
            "ddi": data_root / "TUS2024" / "DDI-IND-NSO-TUS-2024-24.xml",
        },
        2019: {
            "nesstar": data_root / "TUS2019" / "TUS2019.Nesstar",
            "ddi": data_root / "TUS2019" / "DDI-IND-CSO-TUS-2019-19.xml",
        },
    }
    all_attributes = []
    all_trips = []
    for year in years:
        nesstar_path = ROUNDS[year]["nesstar"]
        ddi_path = ROUNDS[year]["ddi"]
        attributes, trips = load(
            nesstar_path, ddi_path, hh_config, person_config, trips_config, year
        )
        all_attributes.append(attributes)
        all_trips.append(trips)
    attributes = table_stacker(all_attributes)
    trips = table_stacker(all_trips)

    return attributes, trips


def load(
    nesstar_path: str | Path,
    ddi_path: str | Path,
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
    year: int = 2024,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load and normalise one round of TUS survey data.

    Args:
        nesstar_path: path to the round's raw .Nesstar binary.
        ddi_path: path to its companion DDI XML.
        hh_config: parsed hh_dictionary.yaml config.
        person_config: parsed person_dictionary.yaml config.
        trips_config: parsed trip_dictionary.yaml config.
        year: which round to load -- 2024 or 2019. Picks the matching
            year-keyed column_mappings and raw table layout.

    Returns:
        (attributes, trips) DataFrames conforming to configs/core/template.yaml.
    """
    print(f"Loading TUS {year}...")

    if year == 2024:
        return _load_2024(
            nesstar_path, ddi_path, hh_config, person_config, trips_config, year
        )
    if year == 2019:
        return _load_2019(
            nesstar_path, ddi_path, hh_config, person_config, trips_config, year
        )
    raise ValueError(f"Unsupported TUS year: {year} (expected 2024 or 2019)")


def _load_2024(
    nesstar_path: str | Path,
    ddi_path: str | Path,
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
    year: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    raw = _load_pruned_blocks(
        Path(nesstar_path).expanduser(),
        Path(ddi_path).expanduser(),
        {
            BLOCKS_2024["hh"]: set(hh_config["column_mappings"]["2024"].keys()),
            BLOCKS_2024["person"]: set(
                person_config["column_mappings"]["2024"].keys()
            ),
        },
    )

    hhs = load_households(raw[BLOCKS_2024["hh"]], hh_config, year)
    persons, episodes = load_persons(
        raw[BLOCKS_2024["person"]], person_config, year
    )
    attributes = table_joiner(
        hhs, persons, on="hid", lhs_name="hh", rhs_name="person"
    )

    trips = load_trips(episodes, trips_config, attributes)

    return attributes, trips


def _load_2019(
    nesstar_path: str | Path,
    ddi_path: str | Path,
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
    year: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    person_mappings = person_config["column_mappings"]["2019"]
    raw = _load_pruned_blocks(
        Path(nesstar_path).expanduser(),
        Path(ddi_path).expanduser(),
        {
            BLOCKS_2019["hh"]: set(hh_config["column_mappings"]["2019"].keys()),
            BLOCKS_2019["demographics"]: set(
                person_mappings["demographics"].keys()
            ),
            BLOCKS_2019["day"]: set(person_mappings["day"].keys()),
            BLOCKS_2019["diary"]: set(person_mappings["diary"].keys()),
        },
    )

    hhs = load_households(raw[BLOCKS_2019["hh"]], hh_config, year)
    persons = load_persons_2019(
        raw[BLOCKS_2019["demographics"]],
        raw[BLOCKS_2019["day"]],
        person_config,
        year,
    )
    episodes = load_episodes_2019(
        raw[BLOCKS_2019["diary"]], person_config, year
    )
    attributes = table_joiner(
        hhs, persons, on="hid", lhs_name="hh", rhs_name="person"
    )

    trips = load_trips(episodes, trips_config, attributes)

    return attributes, trips


def load_households(
    hh_raw: pl.DataFrame, config: dict, year: int
) -> pl.DataFrame:
    column_mapping = config["column_mappings"][str(year)]
    hhs = hh_raw.select(column_mapping.keys()).rename(column_mapping)

    hhs = hhs.with_columns(
        hid=_hid_expr(year),
        hh_size=pl.col("hh_size").cast(pl.Int32, strict=False),
        year=pl.col("year").cast(pl.Int32, strict=False),
        hh_zone=pl.col("hh_zone")
        .replace_strict(config["hh_zone"])
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
    )

    if "ownership" in column_mapping.values():
        # 2024 only -- see configs/tus/hh_dictionary.yaml.
        hhs = hhs.with_columns(
            ownership=pl.col("ownership")
            .replace_strict(config["ownership"], default="unknown")
            .fill_null("unknown")
        )
    else:
        # 2019's schedule has no dwelling-tenure question at all.
        hhs = hhs.with_columns(ownership=pl.lit("unknown"))

    return hhs.drop("hid_fsu", "hid_sample")


def load_persons(
    per_raw: pl.DataFrame, config: dict, year: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split the raw 2024 PER rows into (one row per person) and (one row
    per diary episode), both still carrying the hid/pid-building columns
    needed downstream.
    """
    column_mapping = config["column_mappings"][str(year)]
    raw = per_raw.select(column_mapping.keys()).rename(column_mapping)
    raw = raw.with_columns(hid=_hid_expr(year), pid=_pid_expr(year))

    # one row per person (demographics are constant across a person's episodes)
    persons = (
        raw.select(
            "pid",
            "hid",
            "age",
            "sex",
            "relationship",
            "employment",
            "education",
            "day",
        )
        .unique(subset="pid", keep="first", maintain_order=True)
        .pipe(_finish_persons, config)
    )

    # one row per diary episode (persons with no diary at all -- mostly
    # young children -- have an empty Activity_Serial_No and are dropped
    # here; trips_to_activities gives them a single default "home" activity
    # for the day, same as any other source's trip-less persons)
    episodes = (
        raw.filter(pl.col("activity_code") != "")
        .select(
            "pid",
            "hid",
            "activity_serial_no",
            "major_activity_flag",
            "time_from",
            "time_to",
            "activity_code",
        )
        .filter(pl.col("major_activity_flag") == "1")
    )

    return persons, episodes


def load_persons_2019(
    demographics_raw: pl.DataFrame,
    day_raw: pl.DataFrame,
    config: dict,
    year: int,
) -> pl.DataFrame:
    """Build one row per person for 2019 by joining NSS Block 3
    (demographics, every household member) with Block 5 (day of week,
    diary participants only) on (hid, pid) -- persons with no diary (and so
    no day_of_week) get day="unknown", same as 2024's trip-less persons.
    """
    person_mappings = config["column_mappings"]["2019"]

    demo_mapping = person_mappings["demographics"]
    demo = demographics_raw.select(demo_mapping.keys()).rename(demo_mapping)
    demo = demo.with_columns(hid=_hid_expr(year), pid=_pid_expr(year))

    day_mapping = person_mappings["day"]
    day = day_raw.select(day_mapping.keys()).rename(day_mapping)
    day = day.with_columns(hid=_hid_expr(year), pid=_pid_expr(year)).select(
        "pid", "day"
    )

    persons = (
        demo.select(
            "pid",
            "hid",
            "age",
            "sex",
            "relationship",
            "employment",
            "education",
        )
        .unique(subset="pid", keep="first", maintain_order=True)
        .join(day, on="pid", how="left")
        .pipe(_finish_persons, config)
    )
    return persons


def load_episodes_2019(
    diary_raw: pl.DataFrame, config: dict, year: int
) -> pl.DataFrame:
    """Build the diary-episode table for 2019 from NSS Block 6, matching
    the shape load_trips expects (same as 2024's episodes table).
    """
    diary_mapping = config["column_mappings"]["2019"]["diary"]
    raw = diary_raw.select(diary_mapping.keys()).rename(diary_mapping)
    raw = raw.with_columns(hid=_hid_expr(year), pid=_pid_expr(year))

    episodes = (
        raw.filter(pl.col("activity_code") != "")
        .select(
            "pid",
            "hid",
            "activity_serial_no",
            "major_activity_flag",
            "time_from",
            "time_to",
            "activity_code",
        )
        .filter(pl.col("major_activity_flag") == "1")
    )
    return episodes


def _finish_persons(persons: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Shared value-mapping/defaulting applied to both rounds' one-row-per-person table."""
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


def load_trips(
    episodes: pl.DataFrame, config: dict, attributes: pl.DataFrame
) -> pl.DataFrame:
    """Build trips from the diary: episodes coded as ICATUS travel become
    trips, with oact/dact taken from the (canonical type of the) diary
    activity immediately before/after. Shared by both rounds -- `episodes`
    is already normalised to the same (pid, hid, activity_serial_no,
    major_activity_flag, time_from, time_to, activity_code) shape by
    load_persons/load_persons_2019+load_episodes_2019.
    """
    act_mapping = config["act_mappings"]["default"]
    travel_codes = set(config["travel_codes"]["default"])

    episodes = (
        episodes.with_columns(
            activity_serial_no=pl.col("activity_serial_no").cast(pl.Int32),
            tst=pl.col("time_from").map_elements(
                _parse_time, return_dtype=pl.Int32
            ),
            tet=pl.col("time_to").map_elements(
                _parse_time, return_dtype=pl.Int32
            ),
            is_travel=pl.col("activity_code").is_in(list(travel_codes)),
        )
        .with_columns(
            act=pl.col("activity_code").replace_strict(
                act_mapping, default="unknown"
            )
        )
        .sort("pid", "activity_serial_no")
    )

    # India's TUS diary runs 04:00 -> next day 04:00 (so sleep is never
    # split across midnight), not calendar midnight -> midnight -- but
    # tst/tet must still be true minutes-since-midnight to match every
    # other foundata source (not diary-relative, which is the wrong
    # direction of the fix QHTS needed for its own 4am-anchored raw time
    # *codes* -- see foundata/qhts.py::preprocess_trips). _parse_time
    # parses "HH:MM" directly with no offset, so the one wrap this creates
    # (the diary's last episode crossing back past true midnight) is
    # exactly the kind of ordinary midnight crossing day_wrap already
    # handles for every other source: apply it to the whole diary sequence
    # (activities + trips together) before splitting, so that one wrap is
    # resolved consistently across the full sequence. The portion of the
    # final (always non-travel) episode that lands past minute 1440 is
    # then truncated by trips_to_activities exactly as it is for every
    # other source's last activity of the day -- TUS just happens to have
    # genuine diary data there, where other sources have none to lose.
    episodes = fix.day_wrap(episodes)

    # A diary routinely has several different stationary activities in a
    # row between two trips (e.g. rest, then eat, then nap, all at home,
    # with no trip between them) -- but a plan only has room for a single
    # activity there. So every maximal run of consecutive non-travel
    # episodes (a "gap" -- before the first trip, between two trips, or
    # after the last trip) is squashed into one: the act type with the
    # longest *total* duration within that gap, not just whichever episode
    # happens to be nearest a trip. Each gap is given a "grp" id by taking
    # a running count of travel episodes seen so far (inclusive of the
    # travel row itself) -- every non-travel row between travel episode
    # k-1 and k shares grp=k, so a trip's dact is simply "the gap with its
    # own grp" and its oact is "the gap with grp - 1". Reading oact and
    # dact off the very same per-gap dominant-activity table (rather than
    # deriving them independently from each end) guarantees every plan's
    # activity chain is consistent by construction.
    episodes = episodes.with_columns(
        grp=pl.col("is_travel").cast(pl.Int32).cum_sum().over("pid")
    )

    gap_durations = (
        episodes.filter(~pl.col("is_travel"))
        .with_columns(dur=pl.col("tet") - pl.col("tst"))
        .group_by("pid", "grp", "act")
        .agg(total_dur=pl.col("dur").sum())
    )
    dominant_act = (
        gap_durations.sort(["total_dur", "act"], descending=[True, False])
        .group_by("pid", "grp", maintain_order=True)
        .agg(dominant_act=pl.first("act"))
    )

    trips = episodes.filter(pl.col("is_travel")).with_columns(
        seq=(pl.cum_count("activity_serial_no").over("pid") - 1).cast(pl.Int8),
        mode=pl.lit("unknown"),
        distance=pl.lit(None, dtype=pl.Float32),
    )
    trips = (
        trips.join(
            dominant_act.rename({"dominant_act": "oact"}).with_columns(
                grp=pl.col("grp") + 1
            ),
            on=["pid", "grp"],
            how="left",
        )
        .join(
            dominant_act.rename({"dominant_act": "dact"}),
            on=["pid", "grp"],
            how="left",
        )
        .with_columns(
            oact=pl.col("oact").fill_null("unknown"),
            dact=pl.col("dact").fill_null("unknown"),
        )
        .select("pid", "seq", "tst", "tet", "oact", "dact", "mode", "distance")
    )

    # TUS has no trip-level geography at all (only "inside/outside the
    # dwelling", not an urban/rural classification of *where* outside) --
    # use the person's own household zone for both ends, on the assumption
    # that most within-day travel does not cross the rural/urban boundary.
    hh_zones = attributes.select("pid", ozone=pl.col("hh_zone"))
    trips = trips.join(hh_zones, on="pid", how="left").with_columns(
        dzone=pl.col("ozone")
    )

    return trips


def _parse_time(value: str) -> int:
    """Parse "HH:MM" (already a true clock time, not a time-code) into true
    minutes since midnight. See load_trips for why no anchor shift is
    applied here -- day_wrap handles the one resulting midnight crossing.
    """
    hh, mm = value.split(":")
    return int(hh) * 60 + int(mm)
