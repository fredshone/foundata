import random
from pathlib import Path

import polars as pl

from .utils import (
    config_for_year,
    filter_time_consistent,
    fuzzy_loader,
    sample_int_range,
    sample_uk_to_euro,
    table_joiner,
    table_stacker,
)


def _expand_root(root: str | Path) -> Path:
    return Path(root).expanduser()


def _bounds_from_list(bounds: list[str]) -> tuple[int, int]:
    return int(bounds[0]), int(bounds[1])


def load_mapping(path: Path) -> dict:
    zones = pl.read_csv(path)
    mapping = {1: "urban", 2: "suburban", 3: "rural"}
    zones = zones.with_columns(
        pl.col("HIOX").replace_strict(mapping, default="rural")
    )
    return dict(zip(zones["HABORO"], zones["HIOX"]))


def load_years(
    data_root: str | Path,
    years: list[str],
    hh_config: dict,
    person_config: dict,
    person_data_config: dict,
    trips_config: dict,
    stages_config: dict,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    all_attributes = []
    all_trips = []
    for year in years:
        print(f"Loading {year}...")
        root = _expand_root(data_root) / year

        hh_config_year = config_for_year(hh_config, year)
        person_config_year = config_for_year(person_config, year)
        person_data_config_year = config_for_year(person_data_config, year)
        trips_config_year = config_for_year(trips_config, year)
        stages_config_year = config_for_year(stages_config, year)

        hh_columns = list(hh_config_year["column_mappings"].keys())
        hhs = fuzzy_loader(root, "Household.csv", columns=hh_columns)

        person_columns = list(person_config_year["column_mappings"].keys())
        persons = fuzzy_loader(root, "person.csv", columns=person_columns)

        person_data_columns = list(
            person_data_config_year["column_mappings"].keys()
        )
        persons_data = fuzzy_loader(
            root, "person data.csv", columns=person_data_columns
        )

        trips_columns = list(trips_config_year["column_mappings"].keys())
        trips = fuzzy_loader(root, "Trip.csv", columns=trips_columns)

        stages_columns = list(stages_config_year["column_mappings"].keys())
        stages = fuzzy_loader(root, "Stage.csv", columns=stages_columns)

        zone_mapping = load_mapping(root / "HABORO_T.csv")

        hhs = preprocess_households(hhs, hh_config_year, year, zone_mapping)
        persons = preprocess_persons(persons, person_config_year)
        persons_data = preprocess_persons_data(
            persons_data, person_data_config_year, year
        )
        attributes = table_joiner(persons, persons_data.drop("hid"), on="pid")
        attributes = table_joiner(attributes, hhs, on="hid")

        trips = preprocess_trips(trips, trips_config_year, year, zone_mapping)
        stages = preprocess_stages(stages, stages_config_year, year)

        trips = table_joiner(trips, stages, on="tid")

        attributes, trips = filter_time_consistent(attributes, trips, on="pid")

        all_attributes.append(attributes)
        all_trips.append(trips)

    attributes = table_stacker(all_attributes)
    trips = table_stacker(all_trips)

    attributes = attributes.with_columns(
        source=pl.lit("ltds"),
        country=pl.lit("uk"),
        education=pl.lit("unknown"),
        ownership=pl.lit("unknown"),
        dwelling=pl.lit("unknown"),
        month=pl.lit("unknown"),
        disability=pl.lit("unknown"),
    )

    return attributes, trips


def preprocess_households(
    hhs: pl.DataFrame, config: dict, year: str, zone_mapping: dict
) -> pl.DataFrame:
    column_mapping = config["column_mappings"]
    income_mapping = config["hh_income"]
    struct_mapping = config["hh_structure"]

    hhs = hhs.select(column_mapping.keys()).rename(column_mapping)

    hhs = hhs.with_columns(pl.col("year") + 2000)

    hhs = hhs.with_columns(
        hh_income=(
            pl.col("hh_income")
            .replace_strict(income_mapping, return_dtype=pl.List)
            .map_elements(sample_uk_to_euro, return_dtype=pl.Int32)
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
        .alias("rurality")
    ).drop("zone")

    return hhs.drop_nulls()


def preprocess_persons(
    persons: pl.DataFrame, persons_config: dict
) -> pl.DataFrame:
    column_mapping = persons_config["column_mappings"]
    sex_mapping = persons_config["sex"]
    relationship_mapping = persons_config["relationship"]
    race_mapping = persons_config["race"]

    persons = persons.select(column_mapping.keys()).rename(column_mapping)

    persons = persons.with_columns(
        pl.col("age")
        .replace_strict({"65+": "65-100"}, default=pl.col("age"))
        .str.split("-")
        .map_elements(
            lambda bounds: sample_int_range(_bounds_from_list(bounds)), pl.Int32
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

    if "employment" in column_mapping.values():
        employment_mapping = persons_config["employment"]
        persons = persons.with_columns(
            pl.col("employment").replace_strict(employment_mapping)
        )

    return persons


def preprocess_persons_data(
    persons: pl.DataFrame, config: dict, year: str
) -> pl.DataFrame:
    column_mapping = config["column_mappings"]
    has_license_mapping = config["has_licence"]
    can_wfh_mapping = config["can_wfh"]
    occupation_mapping = config["occupation"]

    persons = persons.select(column_mapping.keys()).rename(column_mapping)

    persons = persons.with_columns(
        pl.col("has_licence").replace_strict(has_license_mapping),
        pl.col("can_wfh").replace_strict(can_wfh_mapping),
        pl.col("occupation").replace_strict(occupation_mapping),
    )

    if "employment" in column_mapping.values():
        employment_mapping = config["employment"]
        persons = persons.with_columns(
            pl.col("employment").replace_strict(employment_mapping)
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
    column_mapping = config["column_mappings"]
    trips = trips.select(column_mapping.keys()).rename(column_mapping)

    trips = trips.sort("hid", "pid", "tid").with_columns(
        seq=pl.col("tid").rank(method="dense").over("hid", "pid")
    )

    mode_map = config["mode"]
    act_map = config["act"]

    trips = trips.with_columns(
        pl.col("mode").replace_strict(mode_map, default=pl.col("mode")),
        pl.col("oact").replace_strict(act_map, default=pl.col("oact")),
        pl.col("dact").replace_strict(act_map, default=pl.col("dact")),
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
        ozone=pl.col("ozone").replace_strict(zone_mapping),
        dzone=pl.col("dzone").replace_strict(zone_mapping),
        ltds_source=pl.lit(year),
    )

    return trips


def preprocess_stages(
    stages: pl.DataFrame, config: dict, year: str
) -> pl.DataFrame:
    column_mapping = config["column_mappings"]
    stages = stages.select(column_mapping.keys()).rename(column_mapping)

    stages = stages.group_by(["pid", "tid"]).agg(
        pl.col("distance").sum().alias("distance")
    )

    return stages.drop("pid")
