import random
from pathlib import Path

import polars as pl
import yaml

from .utils import (
    check_overlap,
    config_for_year,
    get_config_path,
    sample_int_range,
    sample_scaled_range,
    table_joiner,
)


def _expand_root(root: str | Path) -> Path:
    return Path(root).expanduser()


def _bounds_from_list(bounds: list[str]) -> tuple[int, int]:
    return int(bounds[0]), int(bounds[1])


def load_mapping(path: Path, key_name: str, value_name: str) -> dict:
    file = pl.read_csv(path)
    return dict(zip(file[key_name], file[value_name]))


def preprocess_households(
    hhs: pl.DataFrame, config: dict, year: str, zone_mapping: dict
) -> pl.DataFrame:
    column_mapping = config_for_year(config["column_mappings"], year)
    income_mapping = config_for_year(config["hh_income"], year)
    struct_mapping = config_for_year(config["hh_structure"], year)

    hhs = hhs.select(column_mapping.keys()).rename(column_mapping)

    hhs = hhs.with_columns(pl.col("year") + 2000)

    hhs = hhs.with_columns(
        pl.col("hh_income")
        .replace_strict(
            income_mapping, default=pl.lit((0, 0)), return_dtype=pl.List
        )
        .map_elements(
            lambda bounds: sample_scaled_range(bounds, 0.9),
            return_dtype=pl.Float64,
        )
    )

    hhs = hhs.with_columns(
        pl.col("hh_structure")
        .replace_strict(struct_mapping)
        .fill_null("unknown")
    )

    hhs = hhs.with_columns(
        pl.col("zone")
        .replace_strict(zone_mapping, default=pl.col("zone"))
        .fill_null("unknown")
        .alias("urban/rural")
    ).drop("zone")

    return hhs.drop_nulls()


def preprocess_persons(
    persons: pl.DataFrame, config: dict, year: str
) -> pl.DataFrame:
    column_mapping = config_for_year(config["column_mappings"], year)
    sex_mapping = config_for_year(config["sex"], year)
    relationship_mapping = config_for_year(config["relationship"], year)
    race_mapping = config_for_year(config["race"], year)

    persons = persons.select(column_mapping.keys()).rename(column_mapping)

    persons = persons.with_columns(
        pl.col("age")
        .replace_strict({"65+": "65-100"}, default=pl.col("age"))
        .str.split("-")
        .map_elements(
            lambda bounds: sample_int_range(_bounds_from_list(bounds)),
            pl.Float64,
        )
    )

    persons = persons.with_columns(pl.col("sex").replace_strict(sex_mapping))

    persons = persons.with_columns(
        pl.col("relationship")
        .replace_strict(relationship_mapping)
        .alias("relationship")
    )

    persons = persons.with_columns(
        pl.col("race")
        .replace_strict(race_mapping, default=pl.col("race"))
        .fill_null("unknown")
    )

    return persons


def preprocess_persons_data(
    persons: pl.DataFrame, config: dict, year: str
) -> pl.DataFrame:
    column_mapping = config_for_year(config["column_mappings"], year)
    has_license_mapping = config_for_year(config["has_licence"], year)
    employment_mapping = config_for_year(config["employment_status"], year)

    persons = persons.select(column_mapping.keys()).rename(column_mapping)

    persons = persons.with_columns(
        pl.col("has_license").replace_strict(
            has_license_mapping, default=None, return_dtype=pl.String
        )
    )

    persons = persons.with_columns(
        pl.col("employment_status").replace_strict(employment_mapping)
    )

    persons = persons.with_columns((pl.col("no_trips") < 0).alias("no_trips"))

    return persons


def sample_minute(base: int) -> int:
    return random.randint(int(base), int(base) + 5)


def sample_tst(row) -> int:
    tst_hr, tet_hr, duration = row["tst"], row["tet"], row["duration"]
    earliest = max(tst_hr, tet_hr - duration)
    latest = min(tst_hr + 60, tet_hr + 60 - duration)
    if latest < earliest:
        return int((tst_hr + tet_hr + duration) / 2)
    return random.randint(earliest, latest)


def preprocess_trips(
    trips: pl.DataFrame, config: dict, year: str, zone_mapping: dict
) -> pl.DataFrame:
    column_mapping = config_for_year(config["column_mappings"], year)
    trips = trips.select(column_mapping.keys()).rename(column_mapping)

    mode_map = config_for_year(config["mode"], year)
    act_map = config_for_year(config["act"], year)
    trips = trips.with_columns(
        pl.col("mode").replace_strict(mode_map),
        pl.col("oact").replace_strict(act_map),
        pl.col("dact").replace_strict(act_map),
    )

    trips = trips.with_columns(pl.col("duration").map_elements(sample_minute))

    trips = trips.with_columns(pl.col("tst") * 60, pl.col("tet") * 60)

    trips = trips.with_columns(
        pl.struct("tst", "tet", "duration")
        .map_elements(sample_tst, return_dtype=pl.Int32)
        .alias("tst")
    )
    trips = trips.with_columns(
        (pl.col("tst") + pl.col("duration")).alias("tet")
    )

    trips = trips.with_columns(
        pl.col("ozone").replace_strict(zone_mapping),
        pl.col("dzone").replace_strict(zone_mapping),
    )

    mask = pl.any_horizontal(pl.all().is_null())
    keep = (
        trips.group_by("pid")
        .agg(mask.any().alias("flag"))
        .filter(~pl.col("flag"))
        .select("pid")
    )
    trips = trips.join(keep, on="pid")

    return trips


def preprocess_stages(
    stages: pl.DataFrame, config: dict, year: str
) -> pl.DataFrame:
    column_mapping = config_for_year(config["column_mappings"], year)
    stages = stages.select(column_mapping.keys()).rename(column_mapping)

    stages = stages.group_by(["pid", "tid"]).agg(
        pl.col("distance").sum().alias("distance")
    )

    return stages.drop("pid")


def load_year(root: str | Path, year: str) -> dict[str, pl.DataFrame]:
    root = _expand_root(root)
    zone_mapping = load_mapping(root / "HABORO_T.csv", "HABORO", "TYPE")

    hh_config = yaml.safe_load(
        open(get_config_path("ltds", "hh_dictionary.yaml"))
    )
    hh_columns = list(
        config_for_year(hh_config["column_mappings"], year).keys()
    )
    hhs = pl.read_csv(root / "Household.csv", columns=hh_columns)
    hhs = preprocess_households(
        hhs, hh_config, year=year, zone_mapping=zone_mapping
    )

    person_config = yaml.safe_load(
        open(get_config_path("ltds", "person_dictionary.yaml"))
    )
    person_columns = list(
        config_for_year(person_config["column_mappings"], year).keys()
    )
    persons = pl.read_csv(root / "person.csv", columns=person_columns)
    persons = preprocess_persons(persons, person_config, year=year)

    person_data_config = yaml.safe_load(
        open(get_config_path("ltds", "person_data_dictionary.yaml"))
    )
    person_data_columns = list(
        config_for_year(person_data_config["column_mappings"], year).keys()
    )
    persons_data = pl.read_csv(
        root / "person data.csv", columns=person_data_columns
    )
    persons_data = preprocess_persons_data(
        persons_data, person_data_config, year=year
    )

    trip_config = yaml.safe_load(
        open(get_config_path("ltds", "trip_dictionary.yaml"))
    )
    trip_columns = list(
        config_for_year(trip_config["column_mappings"], year).keys()
    )
    trips = pl.read_csv(root / "Trip.csv", columns=trip_columns)
    trips = preprocess_trips(
        trips, trip_config, year=year, zone_mapping=zone_mapping
    )

    stage_config = yaml.safe_load(
        open(get_config_path("ltds", "stage_dictionary.yaml"))
    )
    stage_columns = list(
        config_for_year(stage_config["column_mappings"], year).keys()
    )
    stages = pl.read_csv(root / "Stage.csv", columns=stage_columns)
    stages = preprocess_stages(stages, stage_config, year=year)

    return {
        "households": hhs,
        "persons": persons,
        "persons_data": persons_data,
        "trips": trips,
        "stages": stages,
    }


def build_attributes(
    persons: pl.DataFrame, persons_data: pl.DataFrame, households: pl.DataFrame
) -> pl.DataFrame:
    attributes = table_joiner(persons, persons_data.drop("hid"), on="pid")
    return table_joiner(attributes, households, on="hid")


def attach_stage_distances(
    trips: pl.DataFrame, stages: pl.DataFrame
) -> pl.DataFrame:
    return table_joiner(trips, stages, on="tid")


def check_person_trip_overlap(
    attributes: pl.DataFrame, trips: pl.DataFrame
) -> set:
    return check_overlap(attributes, trips, on="pid")
