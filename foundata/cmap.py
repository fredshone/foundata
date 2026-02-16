from pathlib import Path

import polars as pl
import yaml

from .utils import (
    expand_root,
    get_config_path,
    load_yaml_config,
    sample_us_to_euro,
)


def _cast_numeric_columns(frame: pl.DataFrame) -> pl.DataFrame:
    string_cols = [
        col for col in frame.columns if frame[col].dtype == pl.String
    ]
    can_integer_cols = [
        col
        for col in string_cols
        if frame[col].str.to_integer(strict=False).null_count() == 0
    ]
    if can_integer_cols:
        frame = frame.with_columns([pl.col(can_integer_cols).cast(pl.Int32)])
    return frame


def load_households(
    root: str | Path, config_path: str | Path | None = None
) -> pl.DataFrame:
    root = expand_root(root)
    config_path = (
        Path(config_path)
        if config_path is not None
        else get_config_path("cmap", "hh_dictionary.yaml")
    )
    config = load_yaml_config(config_path)
    column_mapping = config["column_mappings"]
    hhs = pl.read_csv(root / "household.csv", ignore_errors=True)

    hhs = hhs.select(column_mapping.keys()).rename(column_mapping)

    hhs = hhs.with_columns(date=pl.col("date").str.to_datetime("%Y-%m-%d"))

    hhs = hhs.with_columns(
        year=pl.col("date").dt.year(),
        month=pl.col("date").dt.month(),
        day=pl.col("date").dt.weekday().replace_strict(config["day"]),
    ).drop("date")

    # sample income within bounds
    hhs = hhs.with_columns(
        pl.col("hh_income")
        .replace_strict(config["hh_income"])
        .map_elements(sample_us_to_euro, return_dtype=pl.Int32),
        pl.col("residence").replace_strict(config["residence"]),
        pl.col("ownership").replace_strict(config["ownership"]),
    )

    return _cast_numeric_columns(hhs)


def load_persons(
    root: str | Path, config_path: str | Path | None = None
) -> pl.DataFrame:
    root = expand_root(root)
    config_path = (
        Path(config_path)
        if config_path is not None
        else get_config_path("cmap", "person_dictionary.yaml")
    )
    config = load_yaml_config(config_path)
    column_mapping = config["column_mappings"]
    persons = pl.read_csv(root / "person.csv", ignore_errors=True)

    persons = persons.select(column_mapping.keys()).rename(column_mapping)

    persons = persons.with_columns(
        (pl.col("hid").cast(pl.String) + pl.col("pid").cast(pl.String))
        .cast(pl.Int64)
        .alias("pid"),
        pl.col("sex").replace_strict(config["sex"]),
        (
            pl.col("disability")
            .str.split(";")
            .cast(pl.List(pl.Int64))
            .list.sum()
            > 0
        ).alias("disability"),
        pl.col("education").replace_strict(config["education"]),
        pl.col("can_wfh").replace_strict(config["can_wfh"]),
        pl.when(pl.col("is_employed") == 1)
        .then(pl.lit("employed"))
        .otherwise(pl.col("work_status").replace_strict(config["work_status"]))
        .alias("employment"),
        pl.col("industry").replace_strict(config["industry"]),
        pl.col("race").replace_strict(config["race"]),
        pl.col("has_licence").replace_strict(config["has_licence"]),
        pl.col("relationship").replace_strict(config["relationship"]),
    )

    persons = persons.drop(["is_employed", "work_status"])

    return _cast_numeric_columns(persons)


def load_rurality(configs_root: Path) -> pl.DataFrame:
    # https://github.com/spaykin/rural-urban-classification/blob/main/data_final/RuralSubUrban_T.csv
    mapping = pl.read_csv(
        configs_root / "cmap" / "RuralSubUrban_T.csv",
        columns=["tractFIPS", "rurality"],
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
            rurality_table, left_on="fips", right_on="tractFIPS", how="left"
        )
        .with_columns(rurality=pl.col("rurality").fill_null("unknown"))
        .drop("fips")
    )

    return locations


def load_trips(
    root: str | Path,
    config_path: str | Path | None = None,
    rurality_mapping: pl.DataFrame | None = None,
) -> pl.DataFrame:
    root = expand_root(root)
    config_path = (
        Path(config_path)
        if config_path is not None
        else get_config_path("cmap", "trip_dictionary.yaml")
    )

    trips = pl.read_csv(root / "place.csv", ignore_errors=True)

    with open(config_path) as handle:
        trip_mapper = yaml.safe_load(handle)
    column_mapping = trip_mapper["column_mappings"]
    day_mapping = trip_mapper["day"]
    mode_mapping = trip_mapper["mode"]
    act_mapping = trip_mapper["purpose"]

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
        month=pl.col("tst").dt.month(),
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
                    ["sampno", "locno", pl.col("rurality").alias("ozone")]
                ),
                left_on=["hid", "ozone_code"],
                right_on=["sampno", "locno"],
                how="left",
            )
            .join(
                rurality_mapping.select(
                    ["sampno", "locno", pl.col("rurality").alias("dzone")]
                ),
                left_on=["hid", "dzone_code"],
                right_on=["sampno", "locno"],
                how="left",
            )
            .drop(["ozone_code", "dzone_code"])
        )

    return trips
