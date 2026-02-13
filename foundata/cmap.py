from pathlib import Path

import polars as pl
import yaml

from .utils import get_config_path, sample_int_range


def _expand_root(root: str | Path) -> Path:
    return Path(root).expanduser()


def _load_yaml_config(path: str | Path) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle)


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
        frame = frame.with_columns([pl.col(can_integer_cols).cast(pl.Int64)])
    return frame


def load_households(
    root: str | Path, config_path: str | Path | None = None
) -> pl.DataFrame:
    root = _expand_root(root)
    config_path = (
        Path(config_path)
        if config_path is not None
        else get_config_path("cmap", "hh_dictionary.yaml")
    )
    config = _load_yaml_config(config_path)
    column_mapping = config["column_mappings"]
    hhs = pl.read_csv(root / "household.csv", ignore_errors=True)

    hhs = hhs.select(column_mapping.keys()).rename(column_mapping)

    hhs = hhs.with_columns(
        pl.col("date")
        .str.split("-")
        .list.get(0)
        .str.to_integer()
        .alias("year"),
        pl.col("date")
        .str.split("-")
        .list.get(1)
        .str.to_integer()
        .alias("month"),
    )

    # sample income within bounds
    hhs = hhs.with_columns(
        pl.col("hh_income")
        .replace_strict(config["hh_income"])
        .map_elements(sample_int_range, return_dtype=pl.Int32),
        pl.col("residence").replace_strict(config["residence"]),
        pl.col("ownership").replace_strict(config["ownership"]),
    )

    return _cast_numeric_columns(hhs)


def load_persons(
    root: str | Path, config_path: str | Path | None = None
) -> pl.DataFrame:
    root = _expand_root(root)
    config_path = (
        Path(config_path)
        if config_path is not None
        else get_config_path("cmap", "person_dictionary.yaml")
    )
    config = _load_yaml_config(config_path)
    column_mapping = config["column_mappings"]
    persons = (
        pl.read_csv(root / "person.csv", ignore_errors=True)
        .fill_nan(-9)
        .fill_null(-9)
    )

    persons = persons.select(column_mapping.keys()).rename(column_mapping)

    persons = persons.with_columns(
        pl.col(col)
        .replace_strict(config[col], default=None)
        .fill_null(pl.col(col))
        for col in column_mapping.keys()
        if col in config
    )

    return _cast_numeric_columns(persons)


def load_trips(
    root: str | Path, config_path: str | Path | None = None
) -> pl.DataFrame:
    root = _expand_root(root)
    config_path = (
        Path(config_path)
        if config_path is not None
        else get_config_path("cmap", "trip_dictionary.yaml")
    )

    trips = (
        pl.read_csv(root / "place.csv", ignore_errors=True)
        .fill_nan(-9)
        .fill_null(-9)
    )

    with open(config_path) as handle:
        trip_mapper = yaml.safe_load(handle)
    column_mapping = trip_mapper["column_mappings"]
    day_mapping = trip_mapper["day"]
    mode_mapping = trip_mapper["mode"]
    act_mapping = trip_mapper["purpose"]

    trips = trips.select(column_mapping.keys()).rename(column_mapping)

    trips = trips.with_columns(
        tst=pl.col("tst").shift(1).over(["hid", "pid"]),
        oact=pl.col("dact").shift(1).over(["hid", "pid"]),
        ozone=pl.col("dzone").shift(1).over(["hid", "pid"]),
        seq=pl.col("seq") - 1,
    ).filter(pl.col("seq") != 0)

    trips = trips.with_columns(
        mode=pl.col("mode").first().over(["hid", "pid", "seq2"]),
        tst=pl.col("tst").first().over(["hid", "pid", "seq2"]),
        tet=pl.col("tet").last().over(["hid", "pid", "seq2"]),
        oact=pl.col("oact").first().over(["hid", "pid", "seq2"]),
        dact=pl.col("dact").last().over(["hid", "pid", "seq2"]),
        ozone=pl.col("ozone").first().over(["hid", "pid", "seq2"]),
        dzone=pl.col("dzone").last().over(["hid", "pid", "seq2"]),
        distance=pl.col("distance").sum().over(["hid", "pid", "seq2"]),
    ).unique(subset=["hid", "pid", "seq2"], keep="first", maintain_order=True)

    trips = trips.with_columns(
        tst=pl.col("tst").str.to_datetime("%Y-%m-%d %H:%M:%S"),
        tet=pl.col("tet").str.to_datetime("%Y-%m-%d %H:%M:%S"),
    )

    trips = trips.with_columns(
        year=pl.col("tst").dt.year(),
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

    trips = trips.with_columns(
        pid=(
            pl.col("hid").cast(pl.String) + pl.col("pid").cast(pl.String)
        ).cast(pl.Int64),
        mode=pl.col("mode").replace_strict(mode_mapping),
        oact=pl.col("oact").replace_strict(act_mapping),
        dact=pl.col("dact").replace_strict(act_mapping),
    )

    return trips
