from pathlib import Path

import polars as pl

from foundata import fix
from foundata.utils import (
    check_overlap,
    compute_avg_speed,
    sample_int_range,
    sample_uk_to_euro,
    table_joiner,
)

SOURCE = "nts"


def load(
    data_root: str | Path,
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
    stages_config: dict,
    days_config: dict,
) -> tuple[pl.DataFrame, pl.DataFrame]:

    print("Loading NTS...")

    hhs = load_households(data_root, hh_config)
    persons = load_persons(data_root, person_config)
    attributes = table_joiner(
        hhs, persons, on="hid", lhs_name="households", rhs_name="persons"
    )

    # add transit access/egress distance using stages
    stages = load_stages(data_root, stages_config)
    attributes = calc_transit_access_egress_distance(attributes, stages)

    trips = load_trips(data_root, trips_config)

    days = load_days(data_root, days_config)
    trips, attributes = split_days(days, trips, attributes)

    attributes = attributes.with_columns(
        pid=pl.lit(SOURCE) + pl.col("pid").cast(pl.String),
        hid=pl.lit(SOURCE) + pl.col("hid").cast(pl.String),
    )
    trips = trips.with_columns(
        pid=pl.lit(SOURCE) + pl.col("pid").cast(pl.String)
    )

    attributes = compute_avg_speed(attributes, trips)

    return attributes, trips


def load_households(
    root: str | Path, config: dict | None = None
) -> pl.DataFrame:

    print("loading households...")

    columns = config["column_mappings"]

    hhs = pl.read_csv(
        root / "tab" / "household_eul_2002-2023.tab",
        separator="\t",
        columns=list(columns.keys()),
    ).rename(columns)

    income_config = config["hh_income"]
    hhs = hhs.with_columns(
        pl.col("hh_income")
        .replace_strict(income_config)
        .map_elements(
            lambda bounds: sample_uk_to_euro(bounds), return_dtype=pl.Int32
        )
    )

    hhs = hhs.with_columns(
        pl.col("month").cast(pl.Int8),
        pl.col("year").cast(pl.Int32),
        pl.col("ownership").replace_strict(config["ownership"]),
        pl.col("dwelling").replace_strict(config["dwelling"]),
        pl.col("hh_zone").replace_strict(config["hh_zone"]),
        pl.lit("nts").alias("source"),
        pl.lit("uk").alias("country"),
    )

    hhs = hhs.filter(pl.col("hid").is_not_null())

    return hhs


def load_persons(root: str | Path, config: dict | None = None) -> pl.DataFrame:

    print("loading persons...")

    columns = config["column_mappings"]

    persons = pl.read_csv(
        root / "tab" / "individual_eul_2002-2023.tab",
        separator="\t",
        columns=list(columns.keys()),
    ).rename(columns)

    persons = persons.with_columns(
        pl.col("age")
        .replace_strict(config["age"])
        .map_elements(sample_int_range, return_dtype=pl.Int32),
        pl.col("sex").replace_strict(config["sex"]),
        pl.col("education").replace_strict(config["education"]),
        pl.col("has_licence").replace_strict(config["has_licence"]),
        pl.col("employment").replace_strict(config["employment"]),
        pl.col("race").replace_strict(config["race"]),
        pl.col("can_wfh").replace_strict(config["can_wfh"]),
        pl.col("disability").replace_strict(config["disability"]),
        pl.col("occupation").replace_strict(config["occupation"]),
        pl.col("relationship").replace_strict(
            config["relationship"], default=pl.col("relationship")
        ),
        # pl.col("wheelchair_user").replace_strict(config["wheelchair_user"]),
    )

    persons = persons.filter(pl.col("pid").is_not_null())

    return persons


def load_trips(root: str | Path, config: dict | None = None) -> pl.DataFrame:

    print("loading trips...")

    columns = config["column_mappings"]

    trips = pl.read_csv(
        root / "tab" / "trip_eul_2002-2023.tab",
        separator="\t",
        columns=list(columns.keys()),
    ).rename(columns)

    trips = trips.with_columns(
        mode=pl.col("mode").replace_strict(config["mode"]),
        oact=pl.col("oact").replace_strict(config["act"]),
        dact=pl.col("dact").replace_strict(config["act"]),
        tst=pl.col("tst").cast(pl.Int32, strict=False),
        tet=pl.col("tet").cast(pl.Int32, strict=False),
        distance=(pl.col("distance") * 1.6).alias("distance"),
        ozone=pl.lit("unknown"),
        dzone=pl.lit("unknown"),
    )

    return trips.sort("hid", "pid", "tid")


def load_days(root: str | Path, config: dict | None = None) -> pl.DataFrame:

    columns = config["column_mappings"]

    days = pl.read_csv(
        root / "tab" / "day_eul_2002-2023.tab",
        separator="\t",
        columns=list(columns.keys()),
    ).rename(columns)

    return days.sort("pid", "did").with_columns(
        day=pl.col("day").replace_strict(config["day"])
    )


def load_stages(root: str | Path, config: dict | None = None) -> pl.DataFrame:

    columns = config["column_mappings"]

    stages = pl.read_csv(
        root / "tab" / "stage_eul_2002-2023.tab",
        separator="\t",
        columns=list(columns.keys()),
    ).rename(columns)

    stages = stages.with_columns(
        mode=pl.col("mode").replace_strict(config["mode"]),
        distance=pl.col("distance") * 1.6,
        main_stage=pl.when(pl.col("main_stage") == 1)
        .then(True)
        .otherwise(False),
    )

    return stages


def calc_transit_access_egress_distance(
    attributes: pl.DataFrame, stages: pl.DataFrame
) -> pl.DataFrame:

    n_trips = stages.select("pid", "tid").unique().shape[0]

    # main stages with at least one transit stage (either bus or rail)
    main_transit_stages = stages.filter(
        pl.col("main_stage") & pl.col("mode").is_in(["bus", "rail"])
    )

    n_main_transit_trips = (
        main_transit_stages.select("pid", "tid").unique().shape[0]
    )
    perc = n_main_transit_trips / n_trips * 100
    print(
        f"{n_main_transit_trips} out of {n_trips} ({perc:.2f}%) trips have a main transit stage."
    )

    # join to get all stages of trips with a main transit stage
    transit_stages = stages.join(
        main_transit_stages.select("pid", "tid"), on=["pid", "tid"], how="left"
    )

    # anti-join to get non-transit stages of those trips, which are by definition access/egress stages
    access_egress_stages = transit_stages.join(
        main_transit_stages.select("pid", "tid", "sid"),
        on=["pid", "tid", "sid"],
        how="anti",
    )

    # sum stage distances to get total access/egress distance per trip
    trip_distance = access_egress_stages.group_by("pid", "tid").agg(
        pl.col("distance")
        .cast(pl.Float32)
        .sum()
        .alias("access_egress_distance")
    )

    # average per person
    person_distance = (
        trip_distance.group_by("pid")
        .agg(pl.col("access_egress_distance").mean())
        .select("pid", "access_egress_distance")
    )

    attributes = attributes.join(person_distance, on="pid", how="left")

    access_egress_distance_nulls = attributes.filter(
        pl.col("access_egress_distance").is_null()
    ).shape[0]
    perc = access_egress_distance_nulls / attributes.shape[0] * 100
    print(
        f"{access_egress_distance_nulls} out of {attributes.shape[0]} ({perc:.2f}%) persons have null access_egress_distance."
    )

    return attributes


def split_days(
    days: pl.DataFrame, trips: pl.DataFrame, attributes: pl.DataFrame
) -> pl.DataFrame:

    # add pdid to days
    days = (
        days.sort("pid", "did")
        .with_columns(seq_in_day=pl.col("did").rank(method="dense").over("pid"))
        .with_columns(
            (pl.col("pid") * 100 + pl.col("seq_in_day")).alias("pdid")
        )
        .drop("seq_in_day")
    )

    # add pdid to trips
    trips = (
        trips.sort("pid", "tid")
        .with_columns(seq_in_day=pl.col("did").rank(method="dense").over("pid"))
        .with_columns(
            (pl.col("pid") * 100 + pl.col("seq_in_day")).alias("pdid")
        )
        .drop("seq_in_day")
    )
    # split using a rename
    trips = trips.drop("pid").rename({"pdid": "pid"})

    # add pdid and dow to attributes via days (split using right join)
    attributes = (
        attributes.join(
            days.select("pid", "pdid", "day"), on="pid", how="right"
        )
        .drop("pid")
        .rename({"pdid": "pid"})
    )

    check_overlap(
        attributes,
        trips,
        on="pid",
        lhs_name="split attributes",
        rhs_name="split trips",
    )

    # also
    trips = trips.drop(["tid", "did"])
    trips = fix.day_wrap(trips)

    return trips, attributes
