from pathlib import Path

import polars as pl

from foundata import fix
from foundata.utils import (
    compute_avg_speed,
    config_for_year,
    expand_root,
    sample_us_to_euro,
    table_joiner,
)

SOURCE = "nhts"


def load(
    data_root: str | Path,
    years: list[int],
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
) -> tuple[pl.DataFrame, pl.DataFrame]:

    print("Loading NHTS...")
    hhs = load_households(data_root, hh_config, years=years)
    persons = load_persons(data_root, person_config, years=years)
    attributes = table_joiner(hhs, persons, on="hid")
    trips = load_trips(data_root, trips_config, years=years)

    attributes = attributes.with_columns(
        pid=pl.lit(SOURCE) + pl.col("pid").cast(pl.String),
        hid=pl.lit(SOURCE) + pl.col("hid").cast(pl.String),
        access_egress_distance=pl.lit(None, dtype=pl.Float32),
    )
    trips = trips.with_columns(
        pid=pl.lit(SOURCE) + pl.col("pid").cast(pl.String)
    )

    attributes = compute_avg_speed(attributes, trips)

    return attributes, trips


def load_households(
    root: str | Path,
    hh_config: dict,
    years: list[int] | None = None,
    names: list[str] | None = None,
) -> dict[int, pl.DataFrame]:
    root = expand_root(root)
    years = years or [2022, 2017, 2009, 2001]
    names = names or [
        {
            2022: "hhv2pub.csv",
            2017: "hhpub.csv",
            2009: "HHV2PUB.CSV",
            2001: "HHPUB.csv",
        }[y]
        for y in years
    ]

    hhs: list[pl.DataFrame] = []
    for year, name in zip(years, names):
        print(f"Loading {year}...")
        year_config = config_for_year(hh_config, year)

        column_mapping = year_config["column_mappings"]

        path = root / str(year) / name
        data = pl.read_csv(path, ignore_errors=True)

        select = column_mapping.keys() & set(data.columns)
        data = data.select(select).rename(column_mapping, strict=False)

        if "date" in data.columns:
            data = data.with_columns(
                data["date"]
                .cast(pl.String)
                .str.slice(0, 4)
                .cast(pl.Int32)
                .alias("year"),
                data["date"]
                .cast(pl.String)
                .str.slice(4)
                .cast(pl.Int8)
                .alias("month"),
            ).drop("date")

        # sample income within bounds
        data = data.with_columns(
            hh_income=pl.col("hh_income")
            .replace_strict(year_config["hh_income"])
            .map_elements(sample_us_to_euro, return_dtype=pl.Int32),
            hh_zone=pl.col("hh_zone")
            .cast(pl.String)
            .replace_strict(year_config["hh_zone"]),
            ownership=pl.col("ownership").replace_strict(
                year_config["ownership"]
            ),
            day=pl.col("day").replace_strict(year_config["day"]),
        )

        if "race1" in data.columns:
            data = data.with_columns(
                race=pl.col("race1").replace_strict(year_config["race1"])
            ).drop("race1")
        elif "race2" in data.columns:
            data = data.with_columns(
                race=pl.col("race2").replace_strict(year_config["race2"])
            ).drop("race2")
        else:
            data = data.with_columns(race=pl.lit("unknown"))

        data = data.with_columns(
            hid=(
                pl.lit(year).cast(pl.String) + pl.col("hid").cast(pl.String)
            ).cast(pl.Int64)
        )

        data = data.with_columns(source=pl.lit("nhts"), country=pl.lit("usa"))

        data = data.filter(pl.col("hid").is_not_null())

        hhs.append(data)

    hhs = pl.concat(hhs)

    return hhs


def load_persons(
    root: str | Path,
    person_config: str | Path | None = None,
    years: list[int] | None = None,
    names: list[str] | None = None,
) -> dict[int, pl.DataFrame]:
    root = expand_root(root)
    years = years or [2022, 2017, 2009, 2001]
    names = names or [
        {
            2022: "perv2pub.csv",
            2017: "perpub.csv",
            2009: "PERV2PUB.CSV",
            2001: "PERPUB.csv",
        }[y]
        for y in years
    ]

    persons: list[pl.DataFrame] = []
    for year, name in zip(years, names):
        path = root / str(year) / name
        data = pl.read_csv(path, ignore_errors=True)

        year_config = config_for_year(person_config, year)

        column_mapping = year_config["column_mappings"]

        data = data.select(column_mapping.keys())
        data = data.rename(column_mapping)

        data = data.with_columns(
            pl.col(col)
            .replace_strict(
                year_config[col].get(year, year_config[col]), default=None
            )
            .fill_null(pl.col(col))
            for col in column_mapping.values()
            if col in year_config
        ).with_columns(
            employment=pl.when(pl.col("employed") == 1)
            .then(pl.lit("employed"))
            .otherwise(pl.col("employment"))
        )

        data = data.with_columns(
            age=pl.when(pl.col("age") < 0)
            .then(pl.lit(None, dtype=pl.Int32))
            .otherwise(pl.col("age").cast(pl.Int32)),
            hid=(
                pl.lit(year).cast(pl.String) + pl.col("hid").cast(pl.String)
            ).cast(pl.Int64),
        ).with_columns(
            pid=(
                pl.col("hid").cast(pl.String) + pl.col("pid").cast(pl.String)
            ).cast(pl.Int64)
        )

        data = data.filter(pl.col("pid").is_not_null())

        persons.append(data)

    persons = pl.concat(persons)

    persons = persons.with_columns(
        occupation=pl.lit("unknown"),
        dwelling=pl.lit("unknown"),
        can_wfh=pl.lit("unknown"),
    )

    return persons


def _hhmm_to_minutes(value: int) -> int:
    hh = value // 100
    mm = value % 100
    return hh * 60 + mm


def _preprocess_trips(
    trips: pl.DataFrame, year: int, config: dict
) -> pl.DataFrame:
    column_mapping = config["column_mapping"]
    mode_mapping = config["mode_mappings"]
    act_mapping = config["act_mappings"]
    rurality_mapping = config["hh_zone"]

    trips = trips.select(column_mapping.keys()).rename(column_mapping)

    trips = trips.with_columns(
        pid=(
            str(year)
            + pl.col("hid").cast(pl.String)
            + pl.col("pid").cast(pl.String)
        ).cast(pl.Int64),
        distance=pl.when(pl.col("distance") < 0)
        .then(pl.lit(None))
        .otherwise(pl.col("distance")),
    )

    # todo: remove or also filter associated attributes
    trips = trips.filter(
        ~(
            ((pl.col("tst") < 0) | (pl.col("duration") < 0))
            .any()
            .over("pid")  # group-wise flag aligned to each row
        )
    )

    trips = trips.with_columns(
        distance=pl.col("distance") * 1.6,
        tst=pl.col("tst").map_elements(_hhmm_to_minutes),
        mode=pl.col("mode").replace_strict(
            mode_mapping, return_dtype=pl.String, default=pl.col("mode")
        ),
        oact=pl.col("oact").replace_strict(
            act_mapping, return_dtype=pl.String, default=pl.col("oact")
        ),
        dact=pl.col("dact").replace_strict(
            act_mapping, return_dtype=pl.String, default=pl.col("dact")
        ),
    )
    if "ozone" in trips.columns:
        trips = trips.with_columns(
            ozone=pl.col("ozone")
            .cast(pl.String)
            .replace_strict(
                rurality_mapping,
                return_dtype=pl.String,
                default=pl.col("ozone"),
            )
        )
    else:
        trips = trips.with_columns(ozone=pl.lit("unknown"))

    if "dzone" in trips.columns:
        trips = trips.with_columns(
            dzone=pl.col("dzone")
            .cast(pl.String)
            .replace_strict(
                rurality_mapping,
                return_dtype=pl.String,
                default=pl.col("dzone"),
            )
        )
    else:
        trips = trips.with_columns(dzone=pl.lit("unknown"))

    trips = trips.with_columns(tet=pl.col("tst") + pl.col("duration"))

    trips = fix.day_wrap(trips)

    return trips.with_columns(tet=(pl.col("tst") + pl.col("duration")))


def load_trips(
    root: str | Path,
    trips_config: dict,
    years: list[int] | None = None,
    names: list[str] | None = None,
) -> dict[int, pl.DataFrame]:
    root = expand_root(root)
    years = years or [2022, 2017, 2009, 2001]
    names = names or [
        {
            2022: "tripv2pub.csv",
            2017: "trippub.csv",
            2009: "DAYV2PUB.CSV",
            2001: "DAYPUB.csv",
        }[y]
        for y in years
    ]

    trips_by_year: list[pl.DataFrame] = []
    for year, name in zip(years, names):
        path = root / str(year) / name
        year_config = config_for_year(trips_config, year)
        trips = pl.read_csv(path, ignore_errors=True)
        trips_by_year.append(
            _preprocess_trips(trips, year=year, config=year_config)
        )

    trips = pl.concat(trips_by_year)
    return trips
