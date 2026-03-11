from pathlib import Path

import polars as pl

from .utils import (
    compute_avg_speed,
    expand_root,
    get_config_path,
    sample_to_euro,
    table_joiner,
)

USD_TO_EURO = 0.85

SOURCE = "cmap"


def load(
    data_root: str | Path,
    configs_root: str | Path,
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    print("Loading CMAP...")
    rurality = load_rurality(configs_root)

    print("loading households...")
    hhs = load_households(data_root, hh_config)
    hh_locations = load_home_locations(data_root, rurality_table=rurality)
    hhs = hhs.join(hh_locations, on="hid", how="left")

    print("loading persons...")
    persons = load_persons(data_root, person_config)

    attributes = table_joiner(hhs, persons, on="hid").with_columns(
        country=pl.lit("usa"), source=pl.lit("cmap")
    )

    print("loading trips...")
    rurality_mapping = load_locations(data_root, rurality_table=rurality)
    trips = load_trips(
        data_root, trips_config, rurality_mapping=rurality_mapping
    )

    weather = load_weather()
    attributes = attributes.join(
        weather, left_on="survey_date", right_on="date", how="left"
    ).drop("survey_date")

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


def load_households(root: str | Path, config: dict) -> pl.DataFrame:
    column_mapping = config["column_mappings"]
    hhs = pl.read_csv(root / "household.csv", ignore_errors=True)

    hhs = hhs.select(column_mapping.keys()).rename(column_mapping)

    hhs = hhs.with_columns(date=pl.col("date").str.to_datetime("%Y-%m-%d"))

    hhs = hhs.with_columns(
        survey_date=pl.col("date").dt.strftime("%Y-%m-%d"),
        year=pl.col("date").dt.year().cast(pl.Int32),
        month=pl.col("date").dt.month().cast(pl.Int8),
        day=pl.col("date").dt.weekday().replace_strict(config["day"]),
    ).drop("date")

    # sample income within bounds
    hhs = hhs.with_columns(
        pl.col("hh_income")
        .replace_strict(config["hh_income"])
        .map_elements(
            lambda b: sample_to_euro(b, USD_TO_EURO), return_dtype=pl.Int32
        ),
        pl.col("dwelling").replace_strict(config["dwelling"]),
        pl.col("ownership").replace_strict(config["ownership"]),
    )

    hhs = hhs.filter(pl.col("hid").is_not_null())

    return hhs


def load_persons(root: str | Path, config: dict) -> pl.DataFrame:

    column_mapping = config["column_mappings"]
    persons = pl.read_csv(root / "person.csv", ignore_errors=True)

    persons = persons.select(column_mapping.keys()).rename(column_mapping)

    persons = persons.filter(
        pl.col("hid").is_not_null() & pl.col("pid").is_not_null()
    )

    persons = persons.with_columns(
        pid=(
            pl.col("hid").cast(pl.String) + pl.col("pid").cast(pl.String)
        ).cast(pl.Int64),
        age=pl.when(pl.col("age") < 0)
        .then(pl.lit(None, dtype=pl.Int32))
        .otherwise(pl.col("age").cast(pl.Int32)),
        sex=pl.col("sex").replace_strict(config["sex"]),
        disability=pl.when(
            pl.col("disability")
            .str.split(";")
            .cast(pl.List(pl.Int8))
            .list.sum()
            > 0
        )
        .then(pl.lit("yes"))
        .otherwise(pl.lit("no")),
        education=pl.col("education").replace_strict(config["education"]),
        can_wfh=pl.col("can_wfh").replace_strict(config["can_wfh"]),
        employment=pl.when(pl.col("is_employed") == 1)
        .then(pl.lit("employed"))
        .otherwise(pl.col("work_status").replace_strict(config["work_status"])),
        occupation=pl.col("occupation").replace_strict(config["occupation"]),
        race=pl.col("race").replace_strict(config["race"]),
        has_licence=pl.col("has_licence").replace_strict(config["has_licence"]),
        relationship=pl.col("relationship").replace_strict(
            config["relationship"]
        ),
    )

    persons = persons.drop(["is_employed", "work_status"])

    persons = persons.filter(pl.col("pid").is_not_null())

    return persons


def load_weather() -> pl.DataFrame:
    """Load daily weather data for Chicago."""
    csv = get_config_path("cmap", "weather_chicago.csv")
    return (
        pl.read_csv(csv)
        .with_columns(rain=pl.col("precipitation_mm") > 0)
        .drop("precipitation_mm")
    )


def load_rurality(configs_root: Path) -> pl.DataFrame:
    # https://github.com/spaykin/rural-urban-classification/blob/main/data_final/RuralSubUrban_T.csv
    mapping = pl.read_csv(
        configs_root / "cmap" / "RuralSubUrban_T.csv",
        columns=["tractFIPS", "hh_zone"],
    ).with_columns(pl.col("tractFIPS").cast(pl.Utf8).str.zfill(11))
    return mapping


def load_locations(
    root: str | Path, rurality_table: pl.DataFrame
) -> pl.DataFrame:
    root = expand_root(root)
    locations = (
        pl.read_csv(
            root / "location.csv",
            columns=[
                "sampno",
                "locno",
                "state_fips",
                "county_fips",
                "tract_fips",
            ],
        )
        .with_columns(
            pl.col("state_fips").cast(pl.Utf8).str.zfill(2),
            pl.col("county_fips").cast(pl.Utf8).str.zfill(3),
            pl.col("tract_fips").cast(pl.Utf8).str.zfill(6),
        )
        .with_columns(
            fips=(
                pl.col("state_fips")
                + pl.col("county_fips")
                + pl.col("tract_fips")
            ).str.replace_all("-", "0")
        )
        .drop(["state_fips", "county_fips", "tract_fips"])
    )

    locations = (
        locations.join(
            rurality_table,
            left_on="fips",
            right_on="tractFIPS",
            how="left",
            maintain_order="left_right",
        )
        .with_columns(hh_zone=pl.col("hh_zone").fill_null("unknown"))
        .drop("fips")
    )

    return locations


def load_home_locations(
    root: str | Path, rurality_table: pl.DataFrame
) -> pl.DataFrame:
    root = expand_root(root)
    locations = (
        pl.read_csv(
            root / "location.csv",
            columns=[
                "sampno",
                "state_fips",
                "county_fips",
                "tract_fips",
                "home",
            ],
        )
        .filter(pl.col("home") == 1)
        .drop("home")
        .with_columns(
            pl.col("state_fips").cast(pl.Utf8).str.zfill(2),
            pl.col("county_fips").cast(pl.Utf8).str.zfill(3),
            pl.col("tract_fips").cast(pl.Utf8).str.zfill(6),
        )
        .with_columns(
            fips=(
                pl.col("state_fips")
                + pl.col("county_fips")
                + pl.col("tract_fips")
            ).str.replace_all("-", "0")
        )
        .drop(["state_fips", "county_fips", "tract_fips"])
    )

    locations = (
        locations.join(
            rurality_table,
            left_on="fips",
            right_on="tractFIPS",
            how="left",
            maintain_order="left_right",
        )
        .with_columns(hh_zone=pl.col("hh_zone").fill_null("unknown"))
        .drop("fips")
    ).rename({"sampno": "hid"})

    return locations


def load_trips(
    root: str | Path, config: dict, rurality_mapping: pl.DataFrame | None = None
) -> pl.DataFrame:

    trips = pl.read_csv(root / "place.csv", ignore_errors=True)

    column_mapping = config["column_mappings"]
    day_mapping = config["day"]
    mode_mapping = config["mode"]
    act_mapping = config["purpose"]

    trips = trips.select(column_mapping.keys()).rename(column_mapping)

    trips = trips.with_columns(
        pid=(
            pl.col("hid").cast(pl.String) + pl.col("pid").cast(pl.String)
        ).cast(pl.Int64)
    )

    trips = trips.with_columns(
        tst=pl.col("tst").shift(1).over(["hid", "pid"]),
        oact=pl.col("dact").shift(1).over(["hid", "pid"]),
        ozone=pl.col("dzone").shift(1).over(["hid", "pid"]),
        seq=pl.col("seq") - 1,
    ).filter(pl.col("seq") != 0)

    trips = (
        trips.with_columns(
            mode=pl.col("mode").first().over(["hid", "pid", "seq2"]),
            tst=pl.col("tst").first().over(["hid", "pid", "seq2"]),
            tet=pl.col("tet").last().over(["hid", "pid", "seq2"]),
            oact=pl.col("oact").first().over(["hid", "pid", "seq2"]),
            dact=pl.col("dact").last().over(["hid", "pid", "seq2"]),
            ozone=pl.col("ozone").first().over(["hid", "pid", "seq2"]),
            dzone=pl.col("dzone").last().over(["hid", "pid", "seq2"]),
            distance=pl.col("distance").sum().over(["hid", "pid", "seq2"]),
        )
        .unique(
            subset=["hid", "pid", "seq2"], keep="first", maintain_order=True
        )
        .drop("seq2")
    )

    trips = trips.with_columns(
        tst=pl.col("tst").str.to_datetime("%Y-%m-%d %H:%M:%S"),
        tet=pl.col("tet").str.to_datetime("%Y-%m-%d %H:%M:%S"),
    )

    trips = trips.with_columns(
        year=pl.col("tst").dt.year(),
        month=pl.col("tst").dt.month().cast(pl.Int8),
        day=pl.col("tst").dt.weekday().replace_strict(day_mapping),
    )

    trips = trips.with_columns(
        tst=(
            pl.col("tst").dt.hour().cast(pl.Int32) * 60
            + pl.col("tst").dt.minute()
        ),
        tet=(
            pl.col("tet").dt.hour().cast(pl.Int32) * 60
            + pl.col("tet").dt.minute()
        ),
    )

    # deal with trips that span into next day
    trips = trips.with_columns(
        pl.when(pl.col("tet") < pl.col("tst"))
        .then(pl.col("tet") + 1440)
        .otherwise(pl.col("tet"))
        .alias("tet")
    )

    # deal with trips in next day
    trips = trips.with_columns(
        tst=pl.when(pl.col("day") != pl.col("day").first().over("pid"))
        .then(pl.col("tst") + 1440)
        .otherwise(pl.col("tst")),
        tet=pl.when(pl.col("day") != pl.col("day").first().over("pid"))
        .then(pl.col("tet") + 1440)
        .otherwise(pl.col("tet")),
    ).drop("day")

    trips = trips.with_columns(
        mode=pl.col("mode").replace_strict(mode_mapping),
        oact=pl.col("oact").replace_strict(act_mapping),
        dact=pl.col("dact").replace_strict(act_mapping),
    )

    if rurality_mapping is not None:
        trips = (
            trips.rename({"ozone": "ozone_code", "dzone": "dzone_code"})
            .join(
                rurality_mapping.select(
                    ["sampno", "locno", pl.col("hh_zone").alias("ozone")]
                ),
                left_on=["hid", "ozone_code"],
                right_on=["sampno", "locno"],
                how="left",
                maintain_order="left_right",
            )
            .join(
                rurality_mapping.select(
                    ["sampno", "locno", pl.col("hh_zone").alias("dzone")]
                ),
                left_on=["hid", "dzone_code"],
                right_on=["sampno", "locno"],
                how="left",
                maintain_order="left_right",
            )
            .drop(["ozone_code", "dzone_code"])
        )

    return trips
