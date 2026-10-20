"""MON (Mobiliteitsonderzoek Nederland, 2005-2007).

The predecessor of OViN, with the same one-row-per-ride layout and trip
variables, so trips are loaded by the OViN code. Differences from OViN that
matter here:

- whole households are sampled: a household's first row has Huishoud == 1,
  a person's first row Persoon == 1, with real HHID/PersID keys;
- income is personal net income in bands, summed to the household;
- the diary date is per person (a few households span two days);
- no migration background or position-in-household variables;
- the only non-regular trips are truck trips.
"""

from pathlib import Path

import polars as pl
from loguru import logger

from foundata import location, nl_income
from foundata.ovin import (
    drop_non_regular,
    gemeente_code,
    gemeente_zone,
    load_trips,
    read_sav,
)
from foundata.utils import expand_root, table_joiner

SOURCE = "mon"
DATA_FILES = {
    2005: "Databestand MON 2005.sav",
    2006: "MON2006 eindproducten/bestanden/Databestand MON 2006.sav",
    2007: "MON2007/bestanden/Databestand MON 2007.sav",
}


def load(
    data_root: str | Path,
    years: list[int],
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    logger.info("Loading MON...")
    root = expand_root(data_root)
    columns = ["Huishoud", "Persoon"] + [
        raw
        for config in (hh_config, person_config, trips_config)
        for raw in config["column_mappings"]
    ]
    all_attributes, all_trips = [], []
    for year in years:
        logger.info(f"Loading year {year}...")
        data = read_sav(root / str(year) / DATA_FILES[year], columns)

        persons = load_persons(data, person_config, year)
        hhs = load_households(data, hh_config, persons, year)
        zones = gemeente_zone(data.filter(pl.col("Persoon") == 1))
        trips = load_trips(data, trips_config, zones, year, source="mon")
        persons, trips = drop_non_regular(
            persons.drop("personal_income"),
            trips,
            person_config,
            trips_config,
        )

        prefix = pl.lit(str(year))
        attributes = table_joiner(hhs, persons, on="hid").with_columns(
            pid=prefix + pl.col("pid").cast(pl.String),
            hid=prefix + pl.col("hid").cast(pl.String),
            source=pl.lit(SOURCE),
            country=pl.lit("nl"),
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


def load_persons(data: pl.DataFrame, config: dict, year: int) -> pl.DataFrame:
    """One row per person (Persoon == 1), including those dropped later
    for truck travel, whose income still counts towards the household's.

    personal_income is a net income sampled within the Inkomen band, the
    open top band from that year's Pareto tail (see foundata.nl_income)."""
    income_config = nl_income.load_config()
    alpha = nl_income.year_table(income_config, year)["tail_alpha"]
    cap = nl_income.disposable_cap(income_config, year)
    column_mapping = config["column_mappings"]
    data = (
        data.filter(pl.col("Persoon") == 1)
        .select(column_mapping.keys())
        .rename(column_mapping)
    )
    return data.with_columns(
        age=pl.col("age").cast(pl.Int32),
        personal_income=pl.col("personal_income")
        .replace_strict(
            config["personal_income"], return_dtype=pl.List(pl.Int64)
        )
        .map_elements(
            lambda bounds: nl_income.sample_band(bounds, alpha, cap),
            return_dtype=pl.Int32,
        ),
        year=pl.col("year").cast(pl.Int32),
        month=pl.col("month").cast(pl.Int8),
        weight=pl.col("weight").cast(pl.Float32),
        relationship=pl.lit("unknown"),
        race=pl.lit("unknown"),
        disability=pl.lit("unknown"),
        can_wfh=pl.lit("unknown"),
        occupation=pl.lit("unknown"),
        **{
            field: pl.col(field).replace_strict(
                config[field], default="unknown"
            )
            for field in ("sex", "employment", "education", "has_licence")
        },
        day=pl.col("day").replace_strict(config["day"], default="unknown"),
    ).drop("day_of_month")


def load_households(
    data: pl.DataFrame, config: dict, persons: pl.DataFrame, year: int
) -> pl.DataFrame:
    """Households (Huishoud == 1). hh_income is the members' summed net
    income, standing in for disposable household income, converted to
    gross by rank (see foundata.nl_income)."""
    column_mapping = config["column_mappings"]
    hhs = (
        data.filter(pl.col("Huishoud") == 1)
        .select(column_mapping.keys())
        .rename(column_mapping)
        .join(household_income(persons), on="hid", how="left")
    )
    # complete if every member has a person record; AantPers tops out at a
    # "10 or more" code, so the respondent count is capped the same way.
    # Unknown member incomes are already null via household_income.
    complete = pl.col("respondents").clip(
        upper_bound=config["hh_size_top_code"]
    ) == pl.col("hh_size")
    hhs = hhs.with_columns(
        hh_income=pl.when(complete).then(pl.col("hh_income"))
    )
    hhs = hhs.with_columns(
        hh_income=nl_income.to_gross(
            hhs["hh_income"], nl_income.load_config(), year
        ),
        hh_size=pl.col("hh_size").cast(pl.Int32),
        vehicles=pl.col("vehicles").cast(pl.Int32),
        hh_zone=pl.col("hh_zone").replace_strict(
            config["hh_zone"], default="unknown"
        ),
        dwelling=pl.lit("unknown"),
        ownership=pl.lit("unknown"),
    ).drop("respondents")
    return location.add(
        hhs.with_columns(home_gemeente=gemeente_code("home_gemeente")),
        source="mon",
        key=year,
    )


def household_income(persons: pl.DataFrame) -> pl.DataFrame:
    """Sum of members' personal incomes, null if any is unknown, with the
    number of members on file (the caller nulls incomplete households)."""
    income = pl.col("personal_income")
    return persons.group_by("hid").agg(
        hh_income=pl.when(income.is_not_null().all()).then(income.sum()),
        respondents=pl.len(),
    )
