"""Loader for Mongolia's Time Use Survey (mongolia), National Statistics Office
of Mongolia: rounds 2023, 2019 and 2011.

Each round has a household table, a person table and a diary table (one row
per episode), in a different layout:

- 2023 (Stata .dta): persons are the diary respondents only (ages 12+,
  usually one per household), each with one 24h diary. Episodes have clock
  times, an ICATUS 2016 activity code, a place (A8) and, when the place is
  "on the road", the mode of travel (A9). The 1 GB dairy_note_1.dta is an
  analysis copy of the diary and isn't needed.
- 2019 (SPSS .sav): the person table is the full household roster; each
  respondent keeps 2-3 diary days. Episodes have clock times, an ICATUS
  2016 code and one of four places, but no mode.
- 2011 (SPSS .sav): roster plus a separate individual questionnaire; 2-3
  diary days per respondent. Episodes have no clock times, only an order
  and a duration from the 04:00 diary start, a national activity code, and
  a place code that doubles as the mode when in transport.

Every diary day runs 04:00 -> 04:00. Each diary day becomes one plan
(pid = person + weekday digit where people kept several). As in the India
TUS (india.py), travel episodes become trips and each run of stationary
episodes between two trips becomes one activity. See load_trips.

The 2015 round isn't loaded: its activity classification isn't published.
"""

from pathlib import Path

import polars as pl
import pyreadstat
from loguru import logger

from foundata import location
from foundata.utils import (
    add_implied_trips,
    config_for_year,
    table_joiner,
    table_stacker,
)

SOURCE = "mongolia"
COUNTRY = "mng"
CONTINENT = "asia"

ROUNDS = {
    2023: {
        "dir": "2023",
        "files": {
            "hh": "hhdata_1.dta",
            "person": "ppdata_1.dta",
            "diary": "dairy_main_1.dta",
        },
    },
    2019: {
        "dir": "2019",
        "files": {
            "hh": "hh [en].sav",
            "person": "hl [en].sav",
            "diary": "dairy [en].sav",
        },
    },
    2011: {
        "dir": "2011",
        "files": {
            "hh": "TUS HH_en.sav",
            "person": "TUS HL_en.sav",
            "questionnaire": "TUS HQ_en.sav",
            "diary": "TUS HD_en.sav",
        },
    },
}

# The diary runs 04:00 -> 04:00 next day. Clock times before 04:00 belong to
# the next day, so get +1440 to stay minutes since midnight of the diary day.
DIARY_START = 4 * 60


def load_years(
    data_root: str | Path,
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
    years: tuple[int, ...] = tuple(ROUNDS),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load and stack the MONGOLIA rounds in `years` from `data_root`/<year>/.

    Returns:
        (attributes, trips) DataFrames conforming to configs/core/template.yaml.
    """
    all_attributes, all_trips = [], []
    for year in years:
        attributes, trips = load(
            Path(data_root).expanduser() / ROUNDS[year]["dir"],
            hh_config,
            person_config,
            trips_config,
            year,
        )
        all_attributes.append(attributes)
        all_trips.append(trips)
    return table_stacker(all_attributes), table_stacker(all_trips)


def load(
    round_dir: str | Path,
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
    year: int = 2023,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load one MONGOLIA round from its folder.

    Returns:
        (attributes, trips) DataFrames conforming to configs/core/template.yaml.
    """
    if year not in ROUNDS:
        raise ValueError(
            f"Unsupported MONGOLIA year: {year} (expected one of {list(ROUNDS)})"
        )
    logger.info(f"Loading MONGOLIA {year}...")
    round_dir = Path(round_dir).expanduser()
    files = ROUNDS[year]["files"]
    hh_cfg, person_cfg, trips_cfg = (
        config_for_year(c, str(year))
        for c in (hh_config, person_config, trips_config)
    )

    hh_raw = read_table(
        round_dir / files["hh"],
        [
            *hh_cfg["column_mappings"],
            *hh_cfg["vehicle_columns"],
            *(hh_cfg["mode_availability"] or {}).values(),
        ],
    )
    diary_raw = read_table(
        round_dir / files["diary"], list(trips_cfg["column_mappings"])
    )

    hhs = load_households(hh_raw, hh_cfg, year)
    persons = load_persons(
        _read_persons(round_dir, files, person_cfg), person_cfg, year
    )
    episodes = _with_ids(
        diary_raw.rename(trips_cfg["column_mappings"]), year
    ).with_columns(
        pid=pl.col("person") + _code("day").cast(pl.String)
        if trips_cfg["pid_day_suffix"]
        else pl.col("person")
    )
    persons = _diary_days_only(persons, diary_days(episodes, trips_cfg))
    attributes = table_joiner(
        hhs, persons, on="hid", lhs_name="hh", rhs_name="person"
    )
    trips = load_trips(episodes, trips_cfg, attributes)
    return attributes.drop("hh_mode"), trips


def read_table(path: Path, columns: list[str]) -> pl.DataFrame:
    """Read `columns` of a Stata (.dta) or SPSS (.sav) file. Some NSO files
    hold UTF-8 text without declaring it (2023's Stata 13 files) or declare
    the wrong encoding (2011's), so readstat's decoding fails on their
    labels; latin1 decodes any byte, and only numeric columns are used."""
    columns = list(dict.fromkeys(columns))
    reader = (
        pyreadstat.read_dta if path.suffix == ".dta" else pyreadstat.read_sav
    )
    try:
        df, _ = reader(str(path), usecols=columns)
    except (pyreadstat.ReadstatError, UnicodeDecodeError):
        df, _ = reader(str(path), usecols=columns, encoding="latin1")
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return pl.from_pandas(df)


def _code(col: str) -> pl.Expr:
    """Codes are stored as doubles; cast to int for the YAML maps."""
    return pl.col(col).cast(pl.Int64, strict=False)


def _map(col: str, mapping: dict) -> pl.Expr:
    return (
        _code(col)
        .replace_strict(mapping, default="unknown", return_dtype=pl.String)
        .fill_null("unknown")
    )


def _digits(col: str, width: int) -> pl.Expr:
    return _code(col).cast(pl.String).str.zfill(width)


def _with_ids(df: pl.DataFrame, year: int) -> pl.DataFrame:
    """hid and person ids: 2-digit round year + the raw ids (2023's
    single hhid/pid, or the older rounds' cluster, household and line
    numbers, zero-padded). The year keeps the rounds' numbering apart."""
    prefix = pl.lit(str(year)[2:])
    if "hid_cluster" in df.columns:
        hid = prefix + _digits("hid_cluster", 4) + _digits("hid_hh", 3)
        df = df.with_columns(hid=hid)
        if "pid_line" in df.columns:
            df = df.with_columns(person=hid + _digits("pid_line", 2))
        return df.drop("hid_cluster", "hid_hh", "pid_line", strict=False)
    if "hid" in df.columns:
        df = df.with_columns(hid=prefix + _code("hid").cast(pl.String))
    if "pid" in df.columns:
        df = df.with_columns(person=prefix + _code("pid").cast(pl.String)).drop(
            "pid"
        )
    return df


def load_households(
    hh_raw: pl.DataFrame, config: dict, year: int
) -> pl.DataFrame:
    """One row per household, plus the helper column hh_mode (the household's
    best available mode, see _trip_modes; null where the round can't tell),
    which load drops once the trips are built."""
    yes = config["vehicle_yes_code"]
    available = config["mode_availability"]
    hhs = _with_ids(
        hh_raw.rename(config["column_mappings"]), year
    ).with_columns(
        hh_size=_code("hh_size").cast(pl.Int32),
        hh_zone=_map("hh_zone", config["hh_zone"]),
        dwelling=_map("dwelling", config["dwelling"]),
        ownership=_map("ownership", config["ownership"]),
        # yes/no ownership per vehicle type, so a count of types owned
        vehicles=pl.sum_horizontal(
            (_code(c) == yes).cast(pl.Int32) for c in config["vehicle_columns"]
        ).cast(pl.Int32),
        hh_income=pl.lit(None, dtype=pl.Int32),
        country=pl.lit(COUNTRY),
        continent=pl.lit(CONTINENT),
        source=pl.lit(SOURCE),
    )
    # home location: aimag + soum (+ bagh) as NSO codes
    soum = location.clean("home_aimag", width=2) + location.clean(
        "home_soum", width=2
    )
    hhs = hhs.with_columns(
        home_soum=location.clean(soum, valid=r"\d{4}"),
        **(
            {
                "home_bagh": location.clean(
                    soum + location.clean("home_bagh", width=2),
                    valid=r"\d{6}",
                )
            }
            if "home_bagh" in hhs.columns
            else {}
        ),
    )
    hhs = location.add(hhs, source=SOURCE, key=year).drop(
        "home_aimag", "home_bagh", strict=False
    )
    if available:
        hh_mode = (
            pl.when(pl.col("vehicles") > 0)
            .then(pl.lit("car"))
            .when(_code(available["bus"]) == yes)
            .then(pl.lit("bus"))
            .when(_code(available["bike"]) == yes)
            .then(pl.lit("bike"))
            .otherwise(pl.lit("walk"))
        )
    else:
        hh_mode = pl.lit(None, dtype=pl.String)
    return hhs.with_columns(hh_mode=hh_mode).select(
        "hid",
        "hh_size",
        "hh_zone",
        "dwelling",
        "ownership",
        "vehicles",
        "hh_income",
        "country",
        "continent",
        "source",
        "hh_mode",
        "hh_location",
    )


def _rule_columns(config: dict) -> list[str]:
    return [col for rule in config["employment_rules"] for col in rule]


def _read_persons(round_dir: Path, files: dict, config: dict) -> pl.DataFrame:
    """The person table renamed, with the raw employment-rule and disability
    columns. 2011 splits it into the roster and the individual questionnaire
    (which holds those columns), joined on the person's ids."""
    mappings = config["column_mappings"]
    extras = [*_rule_columns(config), *config["disability_columns"]]
    if "persons" not in mappings:
        return read_table(
            round_dir / files["person"], [*mappings, *extras]
        ).rename(mappings)
    roster = read_table(
        round_dir / files["person"], list(mappings["persons"])
    ).rename(mappings["persons"])
    questionnaire = read_table(
        round_dir / files["questionnaire"],
        [*mappings["questionnaire"], *extras],
    ).rename(mappings["questionnaire"])
    return roster.join(
        questionnaire, on=["hid_cluster", "hid_hh", "pid_line"], how="left"
    )


def load_persons(
    person_raw: pl.DataFrame, config: dict, year: int
) -> pl.DataFrame:
    """One row per person (column "person"; pid is set per diary day)."""
    # first rule whose column holds one of its codes
    employment = pl.coalesce(
        _code(col).replace_strict(codes, default=None, return_dtype=pl.String)
        for rule in config["employment_rules"]
        for col, codes in rule.items()
    ).fill_null("unknown")
    if config["disability_columns"]:
        difficulty = [_code(c) for c in config["disability_columns"]]
        threshold = config["disability_threshold"]
        disability = (
            pl.when(pl.any_horizontal(d >= threshold for d in difficulty))
            .then(pl.lit("yes"))
            .when(pl.all_horizontal(d.is_not_null() for d in difficulty))
            .then(pl.lit("no"))
            .otherwise(pl.lit("unknown"))
        )
    else:
        disability = _map("disability", config["disability"])
    occupation = (
        _code("occupation") // config["occupation_divisor"]
    ).replace_strict(
        config["occupation"], default="unknown", return_dtype=pl.String
    )
    return (
        _with_ids(person_raw, year)
        .with_columns(
            age=_code("age").cast(pl.Int32),
            year=pl.lit(year, dtype=pl.Int32),
            sex=_map("sex", config["sex"]),
            relationship=_map("relationship", config["relationship"]),
            education=_map("education", config["education"]),
            employment=employment,
            weight=pl.col("weight").cast(pl.Float32),
            disability=disability,
        )
        .with_columns(
            occupation=pl.when(pl.col("employment") == "employed")
            .then(occupation.fill_null("unknown"))
            .otherwise(pl.lit("void")),
            has_licence=pl.lit("unknown"),
            can_wfh=pl.lit("unknown"),
            race=pl.lit("unknown"),
        )
        .select(
            "hid",
            "person",
            "age",
            "year",
            "sex",
            "relationship",
            "education",
            "employment",
            "occupation",
            "disability",
            "weight",
            "has_licence",
            "can_wfh",
            "race",
        )
    )


def diary_days(episodes: pl.DataFrame, config: dict) -> pl.DataFrame:
    """One row per diary day: its pid, person, weekday and month."""
    return episodes.group_by("pid").agg(
        person=pl.col("person").first(),
        day=_map("day", config["day"]).first(),
        month=_code("month").cast(pl.Int32).first(),
    )


def _diary_days_only(persons: pl.DataFrame, days: pl.DataFrame) -> pl.DataFrame:
    """One attributes row per diary day. Household members who kept no diary
    (under 12s, or absent, sick or refusing in the rosters of 2019 and 2011)
    are dropped rather than given a home-all-day plan; hh_size still counts
    them."""
    diarists = persons.join(days, on="person", how="semi").height
    logger.debug(
        f"Removed {persons.height - diarists}/{persons.height} MONGOLIA household "
        f"members with no diary; {days.height} diary days kept"
    )
    return persons.join(days, on="person", how="inner").drop("person")


def load_trips(
    episodes: pl.DataFrame, config: dict, attributes: pl.DataFrame
) -> pl.DataFrame:
    """Turn diary episodes into trips.

    1. Each episode gets diary-day times, a place, an act type and whether
       it is travel (_prepare_episodes).
    2. _add_implied_trips adds travel the diary leaves out, where the
       person changed place without recording any.
    3. Every run of consecutive travel episodes becomes one trip. Every run
       of stationary episodes between two trips (a "gap") is squashed to the
       act type with the most total time in it. A trip's oact is the gap
       before it and its dact the gap after it, so activity chains are
       consistent by construction.
    4. Mode: see _trip_modes. `attributes` needs the hh_mode helper column
       from load_households.
    """
    episodes = _add_implied_trips(_prepare_episodes(episodes, config), config)

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
    rank = {m: i for i, m in enumerate(config["mode_priority"])}
    trips = (
        episodes.filter(pl.col("is_travel"))
        .group_by("pid", "gap", maintain_order=True)
        .agg(
            tst=pl.col("tst").first(),
            tet=pl.col("tet").last(),
            mode_rank=pl.col("mode")
            .replace_strict(rank, default=None, return_dtype=pl.Int32)
            .min(),
        )
        .with_columns(
            seq=pl.int_range(pl.len()).over("pid").cast(pl.Int8),
            prev_gap=pl.col("gap") - 1,
        )
        .join(
            gap_act.rename({"gap": "prev_gap", "act": "oact"}),
            on=["pid", "prev_gap"],
            how="left",
        )
        .join(gap_act.rename({"act": "dact"}), on=["pid", "gap"], how="left")
        # no trip-level geography, so both ends get the household's own
        # zone (as india.py)
        .join(
            attributes.select("pid", "hh_mode", ozone="hh_zone"),
            on="pid",
            how="left",
        )
        .with_columns(
            oact=pl.col("oact").fill_null("unknown"),
            dact=pl.col("dact").fill_null("unknown"),
            mode=_trip_modes(config["mode_priority"]),
            distance=pl.lit(None, dtype=pl.Float32),
        )
    )

    # Times are already monotonic diary minutes (> 1440 after midnight), so
    # fix.day_wrap has nothing to do.
    return trips.select(
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


def _trip_modes(priority: list[str]) -> pl.Expr:
    """A trip's mode, from the best evidence available:

    1. the highest-priority mode among its on-the-road episodes;
    2. else (travel noted at another place, or an implied trip) the
       highest-priority mode the person used on any trip that diary day;
    3. else (no mode recorded all day) the household's best available
       mode, hh_mode, where the round can tell (2023);
    4. else unknown.
    """
    best = pl.col("mode_rank").fill_null(pl.col("mode_rank").min().over("pid"))
    return (
        best.replace_strict(
            dict(enumerate(priority)), default=None, return_dtype=pl.String
        )
        .fill_null(pl.col("hh_mode"))
        .fill_null("unknown")
    )


def _diary_times(config: dict) -> list[pl.Expr]:
    """tst/tet in minutes since midnight of the diary day (> 1440 after
    midnight), from clock times or from durations in episode order. Needs
    episodes sorted by pid and episode."""
    if config["time_format"] == "duration":
        end = DIARY_START + _code("duration").cum_sum().over("pid")
        return [
            (end - _code("duration")).cast(pl.Int32).alias("tst"),
            end.cast(pl.Int32).alias("tet"),
        ]
    clock_st = _code("start_h") * 60 + _code("start_m")
    clock_en = _code("end_h") * 60 + _code("end_m")
    tst = (
        pl.when(clock_st < DIARY_START)
        .then(clock_st + 1440)
        .otherwise(clock_st)
    )
    tet = tst + (clock_en - clock_st) % 1440
    # a handful of diaries overrun 04:00 by an episode; start each episode no
    # earlier than the previous one ended
    tst = pl.max_horizontal(tst, tet.shift(1).over("pid"))
    return [
        tst.cast(pl.Int32).alias("tst"),
        pl.max_horizontal(tst, tet).cast(pl.Int32).alias("tet"),
    ]


def _prepare_episodes(episodes: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Diary-day times, place, act type, travel flag and mode per episode."""
    code = _code("activity_code")
    episodes = (
        episodes.with_columns(
            episode=_code("episode"),
            location=_code("location").fill_null(config["missing_location"]),
        )
        .sort("pid", "episode")
        .with_columns(_diary_times(config))
    )

    act = code.replace_strict(
        config["act_mappings"], default="unknown", return_dtype=pl.String
    )
    away_act = pl.col("location").replace_strict(
        config["home_act_away"], default=None, return_dtype=pl.String
    )
    on_road = pl.col("location").is_in(config["road_locations"])
    mode = pl.lit(None, dtype=pl.String)
    if config["mode_column"]:
        mode = pl.when(on_road).then(
            _code(config["mode_column"]).replace_strict(
                config["mode"], default="unknown", return_dtype=pl.String
            )
        )
    return episodes.with_columns(
        act=pl.when((act == "home") & away_act.is_not_null())
        .then(away_act)
        .otherwise(act),
        is_travel=code.is_in(config["travel_codes"])
        | (code.is_in(config["escort_travel_codes"]) & on_road),
        mode=mode,
    ).select("pid", "tst", "tet", "act", "location", "is_travel", "mode")


def _add_implied_trips(episodes: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Add the trips the diary leaves out: a move between two consecutive
    stationary episodes at different places, with no travel recorded.

    - Between two fixed places (home, work/school, someone's home,
      restaurant, shop, church) it is always a trip.
    - To or from anywhere else ("elsewhere", or on the road / in the
      countryside, which includes herding near the ger), only if the
      activity type changes and the activity there is one of
      implied_trip.acts (e.g. work or shop, which must happen somewhere).

    The trip takes the last implied_trip.minutes of the earlier episode (at
    most half of it).
    """
    implied = config["implied_trip"]

    def prev(col: str | pl.Expr) -> pl.Expr:
        expr = pl.col(col) if isinstance(col, str) else col
        return expr.shift(1).over("pid")

    fixed = pl.col("location").is_in(config["fixed_places"])
    loose_trip_act = ~fixed & pl.col("act").is_in(implied["acts"])
    implied_trip = (
        ~pl.col("is_travel")
        & ~prev("is_travel")
        & (pl.col("location") != prev("location"))
        & (
            (fixed & prev(fixed))
            | (
                (pl.col("act") != prev("act"))
                & (loose_trip_act | prev(loose_trip_act))
            )
        )
    )
    trip_mins = pl.min_horizontal(
        implied["minutes"], (prev("tet") - prev("tst")) // 2
    )
    return add_implied_trips(
        episodes,
        implied_trip,
        trip_mins,
        "MONGOLIA trips implied by a change of place",
        null_cols=("act", "mode"),
    )
