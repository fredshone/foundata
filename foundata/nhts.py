from pathlib import Path

import polars as pl
import yaml

from foundata.utils import load_yaml_config, sample_us_to_euro

from .utils import get_config_path


def _expand_root(root: str | Path) -> Path:
    return Path(root).expanduser()


def load_households(
    root: str | Path,
    config_path: str | Path | None = None,
    years: list[int] | None = None,
    names: list[str] | None = None,
) -> dict[int, pl.DataFrame]:
    root = _expand_root(root)
    years = years or [2022, 2017, 2009, 2001]
    names = names or ["hhv2pub.csv", "hhpub.csv", "HHV2PUB.CSV", "HHPUB.csv"]

    config_path = (
        Path(config_path)
        if config_path is not None
        else get_config_path("nhts", "hh_dictionary.yaml")
    )
    config = load_yaml_config(config_path)

    hhs: list[pl.DataFrame] = []
    for year, name in zip(years, names):

        year_config = {}
        for k, v in config.items():
            if year in v:
                year_config[k] = v[year]
            else:
                year_config[k] = v["default"]

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
                .cast(pl.Int32)
                .alias("month"),
            ).drop("date")

        # sample income within bounds
        data = data.with_columns(
            pl.col("hh_income")
            .replace_strict(year_config["hh_income"])
            .map_elements(sample_us_to_euro, return_dtype=pl.Int32),
            # pl.col("rurality").replace_strict(config["rurality"]),
            pl.col("ownership").replace_strict(year_config["ownership"]),
            pl.col("day").replace_strict(year_config["day"]),
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

        hhs.append(data)

    hhs = pl.concat(hhs)

    return hhs


def load_persons(
    root: str | Path,
    config_path: str | Path | None = None,
    years: list[int] | None = None,
    names: list[str] | None = None,
) -> dict[int, pl.DataFrame]:
    root = _expand_root(root)
    years = years or [2022, 2017, 2009, 2001]
    names = names or [
        "perv2pub.csv",
        "perpub.csv",
        "PERV2PUB.CSV",
        "PERPUB.csv",
    ]

    config_path = (
        Path(config_path)
        if config_path is not None
        else get_config_path("nhts", "person_dictionary.yaml")
    )
    with open(config_path) as handle:
        config = yaml.safe_load(handle)

    column_mapping = config["column_mappings"]

    persons: list[pl.DataFrame] = []
    for year, name in zip(years, names):
        path = root / str(year) / name
        data = pl.read_csv(path, ignore_errors=True)

        data = data.select(column_mapping.keys())
        data = data.rename(column_mapping)

        data = data.with_columns(
            pl.col(col)
            .replace_strict(
                config[col].get(year, config[col]["default"]), default=None
            )
            .fill_null(pl.col(col))
            for col in column_mapping.values()
            if col in config
        )
        data = data.with_columns(
            hid=(
                pl.lit(year).cast(pl.String) + pl.col("hid").cast(pl.String)
            ).cast(pl.Int64)
        ).with_columns(
            pid=(
                pl.col("hid").cast(pl.String) + pl.col("pid").cast(pl.String)
            ).cast(pl.Int64)
        )

        persons.append(data)

    persons = pl.concat(persons)

    persons = persons.with_columns(
        industry=pl.lit("unknown"),
        residence=pl.lit("unknown"),
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
    column_mapping = config["column_mapping"][year]
    mode_mapping = config["mode_mappings"][year]
    act_mapping = config["act_mappings"][year]
    rurality_mapping = config["rurality"][year]

    trips = trips.select(column_mapping.keys()).rename(column_mapping)

    trips = trips.with_columns(
        pid=(
            str(year)
            + pl.col("hid").cast(pl.String)
            + pl.col("pid").cast(pl.String)
        ).cast(pl.Int64)
    )

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
            ozone=pl.col("ozone").replace_strict(
                rurality_mapping,
                return_dtype=pl.String,
                default=pl.col("ozone"),
            )
        )
    else:
        trips = trips.with_columns(ozone=pl.lit("unknown"))

    if "dzone" in trips.columns:
        trips = trips.with_columns(
            dzone=pl.col("dzone").replace_strict(
                rurality_mapping,
                return_dtype=pl.String,
                default=pl.col("dzone"),
            )
        )
    else:
        trips = trips.with_columns(dzone=pl.lit("unknown"))

    trips = trips.with_columns(tet=pl.col("tst") + pl.col("duration"))

    # fix trips that pass midnight
    trips = (
        trips.with_columns(
            flag=pl.when(pl.col("tst") < pl.col("tet").shift(1).over("pid"))
            .then(1)
            .otherwise(0)
        )
        .with_columns(flag=pl.col("flag").cum_sum().over("pid"))
        .with_columns(tst=pl.col("tst") + pl.col("flag") * 1440)
        .drop("flag")
    )

    return trips.with_columns(tet=(pl.col("tst") + pl.col("duration")))


def load_trips(
    root: str | Path,
    config_path: str | Path | None = None,
    years: list[int] | None = None,
    names: list[str] | None = None,
) -> dict[int, pl.DataFrame]:
    root = _expand_root(root)
    years = years or [2022, 2017, 2009, 2001]
    names = names or [
        "tripv2pub.csv",
        "trippub.csv",
        "DAYV2PUB.CSV",
        "DAYPUB.csv",
    ]

    config_path = (
        Path(config_path)
        if config_path is not None
        else get_config_path("nhts", "trip_dictionary.yaml")
    )
    with open(config_path) as handle:
        config = yaml.safe_load(handle)

    trips_by_year: list[pl.DataFrame] = []
    for year, name in zip(years, names):
        path = root / str(year) / name
        trips = pl.read_csv(path, ignore_errors=True)
        trips_by_year.append(_preprocess_trips(trips, year=year, config=config))

    trips = pl.concat(trips_by_year)
    return trips
