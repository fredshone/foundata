"""OViN (Onderzoek Verplaatsingen in Nederland, 2010-2017).

The predecessor of ODiN, with the same one-row-per-ride layout: a
respondent's first row has OP == 1, a trip's first ride row has Verpl set,
and person attributes are repeated on every row. Differences from ODiN that
matter here:

- distributed as SPSS .sav (numeric codes, NaN for system missing);
- different Hvm (main mode) codes;
- no per-trip departure location (VertLoc), only the person-level
  Vertrekp for the first trip;
- income in absolute EUR bands (HHBestInk) rather than deciles;
- no trip-zone lookup is shipped: trip urbanity comes from the
  respondents' own home gemeente and its Sted (see `gemeente_zone`);
- professional travel is a placeholder row (Verpl 2) rather than trips
  (see `drop_non_regular`).
"""

from pathlib import Path

import polars as pl
import pyreadstat
from loguru import logger

from foundata import fix, location, nl_income
from foundata.utils import expand_root, resolve_activity_chain, table_joiner

SOURCE = "ovin"
CONTINENT = "europe"
DATA_FILES = {
    2010: "OViN2010_Databestand.sav",
    2011: "OViN2011_Databestand_revisie.sav",
    2012: "OViN2012_Databestand_revisie.sav",
    2013: "OViN2013_Databestand.sav",
    2014: "OViN2014_Databestand.sav",
    2015: "OViN2015_Databestand.sav",
    2016: "OViN2016_Databestand.sav",
    2017: "OViN2017_Databestand.sav",
}
FLOAT_COLUMNS = {"FactorP"}


def gemeente_code(column: str) -> pl.Expr:
    """A 4-digit CBS gemeente code (OViN and MON); null for 0 (unknown)."""
    return location.clean(column, width=4, missing=["0000"])


def load(
    data_root: str | Path,
    years: list[int],
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    logger.info("Loading OViN...")
    root = expand_root(data_root)
    columns = ["OP"] + [
        raw
        for config in (hh_config, person_config, trips_config)
        for raw in config["column_mappings"]
    ]
    all_attributes, all_trips = [], []
    for year in years:
        logger.info(f"Loading year {year}...")
        data = read_sav(root / str(year) / DATA_FILES[year], columns)

        hhs = load_households(data, hh_config, year)
        persons = load_persons(data, person_config)
        zones = gemeente_zone(data.filter(pl.col("OP") == 1))
        trips = load_trips(data, trips_config, zones, year)
        persons, trips = drop_non_regular(
            persons, trips, person_config, trips_config
        )

        prefix = pl.lit(str(year))
        attributes = table_joiner(hhs, persons, on="hid").with_columns(
            pid=prefix + pl.col("pid").cast(pl.String),
            hid=prefix + pl.col("hid").cast(pl.String),
            source=pl.lit(SOURCE),
            country=pl.lit("nl"),
            continent=pl.lit(CONTINENT),
        )
        trips = trips.with_columns(
            pid=prefix + pl.col("pid").cast(pl.String),
            hid=prefix + pl.col("hid").cast(pl.String),
        )
        all_attributes.append(attributes)
        all_trips.append(trips)

    attributes = pl.concat(all_attributes, how="diagonal")
    trips = pl.concat(all_trips, how="diagonal")
    return attributes, trips


def read_sav(path: Path, columns: list[str]) -> pl.DataFrame:
    """Read `columns` from an OViN .sav file as polars.

    Variable-name case varies between years (WoGem in 2010, Wogem later), so
    names are matched case-insensitively and returned with the spelling
    asked for. All columns but the weights hold integer codes, which are
    cast strictly so a non-integer value fails loudly.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{SOURCE} data file not found: {path}")
    _, meta = pyreadstat.read_sav(str(path), metadataonly=True)
    by_lower = {c.lower(): c for c in meta.column_names}
    missing = [c for c in dict.fromkeys(columns) if c.lower() not in by_lower]
    if missing:
        raise KeyError(
            f"{path.name} has no column(s) {missing} (matched "
            f"case-insensitively against its {len(by_lower)} variables). "
            "Check the column_mappings for this year."
        )
    rename = {by_lower[c.lower()]: c for c in dict.fromkeys(columns)}
    df, _ = pyreadstat.read_sav(
        str(path), usecols=list(rename), apply_value_formats=False
    )
    data = pl.from_pandas(df).rename(rename)
    return data.with_columns(
        pl.col(c).cast(pl.Float32 if c in FLOAT_COLUMNS else pl.Int64)
        for c in data.columns
    )


def respondents(data: pl.DataFrame, column_mapping: dict) -> pl.DataFrame:
    """One row per respondent (OP == 1), renamed by `column_mapping`."""
    return (
        data.filter(pl.col("OP") == 1)
        .select(column_mapping.keys())
        .rename(column_mapping)
    )


def load_households(
    data: pl.DataFrame, config: dict, year: int
) -> pl.DataFrame:
    data = respondents(data, config["column_mappings"])
    data = data.with_columns(
        hh_income=household_income(data["hh_income"], config, year),
        hh_size=pl.col("hh_size").cast(pl.Int32),
        vehicles=pl.when(pl.col("vehicles") != config["vehicles_unknown"])
        .then(pl.col("vehicles"))
        .cast(pl.Int32),
        hh_adults=pl.col("hh_lft4").cast(pl.Int32),
        hh_children=pl.sum_horizontal("hh_lft1", "hh_lft2", "hh_lft3").cast(
            pl.Int32
        ),
        hh_zone=pl.col("hh_zone").replace_strict(
            config["hh_zone"], default="unknown"
        ),
        day=pl.col("day").replace_strict(config["day"], default="unknown"),
        year=pl.col("year").cast(pl.Int32),
        month=pl.col("month").cast(pl.Int8),
        dwelling=pl.lit("unknown"),
        ownership=pl.lit("unknown"),
    ).drop("day_of_month", "hh_lft1", "hh_lft2", "hh_lft3", "hh_lft4")
    return location.add(
        data.with_columns(home_gemeente=gemeente_code("home_gemeente")),
        source="ovin",
        key=year,
    )


def household_income(codes: pl.Series, config: dict, year: int) -> pl.Series:
    """Gross household income from HHBestInk (disposable income bands).

    A disposable income is sampled within each band (the open top band from
    a Pareto tail) and converted to gross by rank (see foundata.nl_income).
    """
    income_config = nl_income.load_config()
    alpha = nl_income.year_table(income_config, year)["tail_alpha"]
    cap = nl_income.disposable_cap(income_config, year)
    disposable = codes.replace_strict(
        config["hh_income"], return_dtype=pl.List(pl.Int64)
    ).map_elements(
        lambda bounds: nl_income.sample_band(bounds, alpha, cap),
        return_dtype=pl.Int64,
    )
    return nl_income.to_gross(disposable, income_config, year)


def load_persons(data: pl.DataFrame, config: dict) -> pl.DataFrame:
    data = respondents(data, config["column_mappings"])
    return data.with_columns(
        hid=pl.col("pid"),
        age=pl.col("age").cast(pl.Int32),
        weight=pl.col("weight").cast(pl.Float32),
        disability=pl.lit("unknown"),
        can_wfh=pl.lit("unknown"),
        occupation=pl.lit("unknown"),
        **{
            field: pl.col(field).replace_strict(
                config[field], default="unknown"
            )
            for field in (
                "sex",
                "employment",
                "education",
                "has_licence",
                "relationship",
                "race",
            )
        },
    )


def gemeente_zone(respondents: pl.DataFrame) -> pl.DataFrame:
    """Gemeente code -> stedelijkheidsklasse for one survey year, from one
    row per respondent.

    Sted is the stedelijkheidsklasse of the respondent's home gemeente
    (WoGem), so the respondents themselves give a lookup for every gemeente
    someone in the sample lives in, using that year's boundaries. Trips
    to or from other gemeenten (or abroad, code 0) get no match.
    """
    return respondents.group_by(gemeente="WoGem").agg(
        sted=pl.col("Sted").mode().sort().first()
    )


def load_trips(
    data: pl.DataFrame,
    config: dict,
    zones: pl.DataFrame,
    year: int | None = None,
    source: str = "ovin",
) -> pl.DataFrame:
    """All trips (any Verpl type), one row per trip; `trip_type` is kept so
    `drop_non_regular` can use it. Activities are mapped but not chained
    yet (see `finalise_trips`). hid is the person id unless the config maps
    a household id (mon)."""
    column_mapping = config["column_mappings"]
    data = (
        data.filter(pl.col("Verpl") >= 1)
        .select(column_mapping.keys())
        .rename(column_mapping)
    )
    for end, zone in (("o", "ozone"), ("d", "dzone")):
        data = data.join(
            zones.rename({"gemeente": f"{end}_gemeente", "sted": zone}),
            on=f"{end}_gemeente",
            how="left",
            maintain_order="left",
        )
    if "hid" not in data.columns:
        data = data.with_columns(hid=pl.col("pid"))
    data = data.with_columns(
        origin_gemeente=gemeente_code("o_gemeente"),
        destination_gemeente=gemeente_code("d_gemeente"),
    )
    data = location.add_trip_ends(data, source=source, key=year)
    return data.with_columns(
        tst=pl.col("tst_hour") * 60 + pl.col("tst_min"),
        tet=pl.col("tet_hour") * 60 + pl.col("tet_min"),
        distance=(pl.col("distance") * config["distance_scale"]).cast(
            pl.Float32
        ),
        mode=pl.col("mode").replace_strict(config["mode"], default="unknown"),
        oact=pl.col("oact").replace_strict(config["oact"], default="unknown"),
        dact=pl.col("dact").replace_strict(
            config["dact"], default="unknown", return_dtype=pl.String
        ),
        ozone=pl.col("ozone").replace_strict(
            config["stedelijkheidsklasse"], default="unknown"
        ),
        dzone=pl.col("dzone").replace_strict(
            config["stedelijkheidsklasse"], default="unknown"
        ),
    ).drop(
        "tst_hour",
        "tst_min",
        "tet_hour",
        "tet_min",
        "o_gemeente",
        "d_gemeente",
    )


def drop_non_regular(
    persons: pl.DataFrame,
    trips: pl.DataFrame,
    person_config: dict,
    trips_config: dict,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Drop respondents with non-regular travel and keep regular trips.

    Respondents whose only travel was professional, foreign, truck or
    holiday travel (Weggeweest 2-5) have no regular trips but did leave
    home, so a home-all-day plan would be wrong. Respondents who made some
    of those trips alongside regular ones (Verpl 2-5) would be left with an
    incomplete day.

    The exception is the config's `work_travel_block` (OViN Verpl 2,
    beroepsmatige verplaatsingen): a placeholder row with no times, mode or
    distance standing for a stretch of travel made as part of work. When
    the row before it (a trip, or another block) ends at work, the block is
    time spent working, so it is removed and the respondent kept: the next
    trip chains from work. Blocks after any other activity, or at the start
    of the day, can't be placed, and their respondents are dropped.
    """
    regular = pl.col("trip_type") == trips_config["regular_trip"]
    keep = regular
    block_type = trips_config.get("work_travel_block")
    if block_type is not None:
        after_work = (pl.col("dact").shift(1).over("pid") == "work").fill_null(
            False
        )
        keep = regular | ((pl.col("trip_type") == block_type) & after_work)
    trips = trips.sort("pid", "seq")
    drop = pl.concat(
        [
            persons.filter(
                pl.col("away").is_in(person_config["away_non_regular"])
            )["pid"],
            trips.filter(~keep)["pid"],
        ]
    ).implode()
    persons = persons.filter(~pl.col("pid").is_in(drop)).drop("away")
    trips = trips.filter(~pl.col("pid").is_in(drop) & regular).drop("trip_type")
    return persons, finalise_trips(trips)


def finalise_trips(trips: pl.DataFrame) -> pl.DataFrame:
    """Chain activities, renumber seq from 0 and wrap times past midnight.

    Every trip's oact starts as the person-level Vertrekp (where the first
    trip left from); resolve_activity_chain keeps it for the first trip
    only and chains the rest from the previous dact, after resolving
    round trips (Doel 10, toeren/wandelen) to the activity they return to.
    """
    # renumber 1..n first: removed work-travel blocks leave gaps, and
    # resolve_activity_chain finds the first trip by seq == 1
    trips = trips.sort("pid", "seq").with_columns(
        seq=pl.int_range(1, pl.len() + 1).over("pid")
    )
    trips = resolve_activity_chain(trips, group_cols=["pid"])
    trips = trips.with_columns(seq=(pl.col("seq") - 1).cast(pl.Int8))
    return fix.day_wrap(trips)
