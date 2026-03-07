from pathlib import Path

import polars as pl

from foundata import fix, utils

SOURCE = "ktdb"


def load(
    data_root: str | Path, person_config: dict, trips_config: dict
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load and normalise ktdb survey data.

    Args:
        data_root: Path to the raw data directory for this source.
        person_config: Parsed person_dictionary.yaml config.
        trips_config: Parsed trip_dictionary.yaml config.

    Returns:
        (attributes, trips) DataFrames conforming to the template schema.
    """
    attributes = load_persons(data_root, person_config)
    trips = load_trips(data_root, trips_config)

    # Prefix IDs with source name for global uniqueness
    attributes = attributes.with_columns(
        pid=pl.lit(SOURCE) + pl.col("pid").cast(pl.String),
        # hid=pl.lit(SOURCE) + pl.col("pid").cast(pl.String),
    )
    trips = trips.with_columns(
        pid=pl.lit(SOURCE) + pl.col("pid").cast(pl.String)
    )

    return attributes, trips


def load_persons(root: str | Path, config: dict) -> pl.DataFrame:
    """Load and normalise person records."""
    root = Path(root).expanduser()
    column_mapping = config["column_mappings"]

    data = pl.read_csv(root / "persons.csv", ignore_errors=True)
    data = data.select(column_mapping.keys()).rename(column_mapping)

    # Cast coded columns to Int64 so replace_strict can match integer YAML keys;
    # strict=False turns non-numeric values (blanks, spaces) into nulls.
    int_cols = [
        "age",
        "hh_size",
        "cars",
        "vans",
        "motorcycles",
        "sex",
        "has_licence",
        "student",
        "employed",
        "occupation",
        "can_wfh",
        "hh_income",
        "dwelling",
    ]
    data = data.with_columns(
        [pl.col(c).cast(pl.Int64, strict=False) for c in int_cols]
    )

    data = data.with_columns(
        hid=pl.col("pid"),
        year=pl.lit(2021),
        source=pl.lit(SOURCE),
        country=pl.lit("south korea"),
        avg_speed=pl.lit(None).cast(pl.Float64),
        # read to date from YYYYMMDD integer format, e.g. 20210101 for Jan 1, 2021
        date=(pl.lit(20210000) + pl.col("date"))
        .cast(pl.String)
        .str.to_date(format="%Y%m%d"),
        sex=pl.col("sex").replace_strict(config["sex"], default="unknown"),
        has_licence=pl.col("has_licence").replace_strict(
            config["has_licence"], default="unknown"
        ),
        _student=pl.col("student").replace_strict(
            config["student"], default="unknown"
        ),
        _employed=pl.col("employed").replace_strict(
            config["employed"], default="unknown"
        ),
        occupation=pl.col("occupation").replace_strict(
            config["occupation"], default="unknown"
        ),
        can_wfh=pl.col("can_wfh").replace_strict(
            config["can_wfh"], default="unknown"
        ),
        hh_income=pl.col("hh_income")
        .replace_strict(config["hh_income"], default=None)
        .map_elements(utils.sample_krw_to_euro)
        * 1000000
        * 12,
        dwelling=pl.col("dwelling").replace_strict(
            config["dwelling"], default="unknown"
        ),
        vehicles=pl.col("cars").cast(pl.Int64, strict=False).fill_null(0)
        + pl.col("motorcycles").cast(pl.Int64, strict=False).fill_null(0)
        + pl.col("vans").cast(pl.Int64, strict=False).fill_null(0),
        weight=pl.lit(1.0),
    )

    data = data.with_columns(
        month=pl.col("date").dt.month(),
        day=pl.col("date")
        .dt.weekday()
        .replace_strict(config["weekday"], default="unknown"),
        employment=pl.when(pl.col("_student") == "yes")
        .then(pl.lit("student"))
        .when(pl.col("_employed") == "yes")
        .then(pl.lit("employed"))
        .otherwise(pl.lit("unemployed")),
    ).drop(["_student", "_employed"])

    data = data.with_columns(
        pl.lit("unknown").alias(col)
        for col in [
            "education",
            "disability",
            "relationship",
            "race",
            "ownership",
            "rurality",
        ]
    )
    return data


def load_zones() -> pl.DataFrame:
    zones = pl.read_csv(utils.get_config_path("ktdb", "zones.csv"))
    zones = (
        zones.select("Administrative dong code", "town/village name")
        .rename(
            {"Administrative dong code": "zone", "town/village name": "name"}
        )
        .with_columns(
            rurality=pl.when(pl.col("name").str.ends_with("동"))
            .then(pl.lit("urban"))
            .when(pl.col("name").str.ends_with("읍"))
            .then(pl.lit("suburban"))
            .when(pl.col("name").str.ends_with("면"))
            .then(pl.lit("rural"))
            .when(pl.col("name").str.ends_with("리"))
            .then(pl.lit("rural"))
            .otherwise(pl.lit("unknown"))
        )
    )
    return zones


def load_trips(root: str | Path, config: dict) -> pl.DataFrame:
    """Load and normalise trip records.

    TODO: Update file name and path to match raw data layout.
    TODO: Convert tst/tet to minutes since midnight (int).
    TODO: Convert distance to kilometres.
    """
    root = Path(root).expanduser()
    column_mapping = config["column_mappings"]

    data = pl.read_csv(root / "trips.csv", ignore_errors=True)
    data = data.select(column_mapping.keys()).rename(column_mapping)

    data = data.filter(pl.col("seq") > 0)

    data = data.with_columns(
        tst=pl.col("tst-hr").cast(pl.Int16, strict=False) * 60
        + pl.col("tst-min").cast(pl.Int16, strict=False),
        tet=pl.col("tet-hr").cast(pl.Int16, strict=False) * 60
        + pl.col("tet-min").cast(pl.Int16, strict=False),
    ).drop(["tst-hr", "tst-min", "tet-hr", "tet-min"])

    data = data.with_columns(
        origin=pl.col("origin")
        .cast(pl.Int64, strict=False)
        .replace_strict(config["origin"], default="unknown"),
        purpose=pl.col("purpose")
        .cast(pl.Int64, strict=False)
        .replace_strict(config["purpose"], default="unknown"),
    )

    # access trip stages
    stage_cols = ["pid", "seq"]
    for i in range(1, 11):
        stage_cols += [f"mode{i}", f"duration{i}"]

    # modes
    stage_modes = (
        data.select(["pid", "seq"] + [f"mode{i}" for i in range(1, 11)])
        .rename({f"mode{i}": str(i) for i in range(1, 11)})
        .unpivot(index=["pid", "seq"], on=[str(i) for i in range(1, 11)])
        .with_columns(
            variable=pl.col("variable").cast(pl.Int8),
            value=pl.col("value").cast(pl.Int8, strict=False),
        )
        .rename({"variable": "stage", "value": "mode"})
        .filter(pl.col("mode").is_not_null())
        .with_columns(mode=pl.col("mode").replace_strict(config["mode"]))
        .sort(["pid", "seq", "stage"])
    )

    # durations
    stage_durations = (
        data.select(["pid", "seq"] + [f"duration{i}" for i in range(1, 11)])
        .rename({f"duration{i}": str(i) for i in range(1, 11)})
        .unpivot(index=["pid", "seq"], on=[str(i) for i in range(1, 11)])
        .with_columns(
            variable=pl.col("variable").cast(pl.Int8),
            value=pl.col("value").cast(pl.Int16, strict=False),
        )
        .rename({"variable": "stage", "value": "duration"})
        .filter(pl.col("duration").is_not_null())
        .sort(["pid", "seq", "stage"])
    )

    stages = stage_modes.join(stage_durations, on=["pid", "seq", "stage"])

    # filter for trips with pt stages
    # pt_trips = stages.filter(pl.col("mode").is_in(["bus", "rail"])).select("pid", "seq").unique()
    # pt_trips = stages.join(pt_trips, on=["pid", "seq"], how="inner")

    # # get total duration of non pt modes for each trip
    # ae_stages = stages.filter(~pl.col("mode").is_in(["bus", "rail"]))
    # ae_durations = ae_stages.group_by(["pid", "seq"]).agg(pl.col("duration").sum().alias("ae_duration"))
    # pt_ae_durations = pt_trips.drop("duration").join(ae_durations, on=["pid", "seq"], how="left").fill_null(0)

    # stages.join(pt_ae_durations, on=["pid", "seq"], how="left").sort(["pid", "seq", "stage"])

    total_trip_durations = (
        stages.drop("stage")
        .group_by(["pid", "seq"])
        .agg(pl.col("duration").sum())
    )
    # aggregate trip modes based on longest mode across all stages by pid
    main_trip_modes = (
        stages.group_by(["pid", "seq", "mode"])
        .agg(pl.col("duration").sum())
        .group_by(["pid", "seq"])
        .agg(pl.all().sort_by("duration").last())
    ).drop("duration")

    data = (
        data.select(
            "pid", "seq", "ozone", "dzone", "origin", "purpose", "tst", "tet"
        )
        .join(total_trip_durations, on=["pid", "seq"])
        .join(main_trip_modes, on=["pid", "seq"])
    )

    zones = load_zones()
    zone_mapping = dict(zip(zones["zone"], zones["rurality"]))

    data = data.with_columns(
        ozone=pl.col("ozone").replace_strict(zone_mapping, default="unknown"),
        dzone=pl.col("dzone").replace_strict(zone_mapping, default="unknown"),
    )

    data = (
        data.sort(["pid", "seq"])
        .with_columns(
            oact=pl.when(pl.col("seq") > 1)
            .then(pl.col("purpose").shift(1).over("pid"))
            .otherwise(pl.col("origin")),
            dact=pl.col("purpose"),
        )
        .drop(["origin", "purpose"])
    )

    # distance hack
    speeds = {
        "car": 60,
        "bus": 40,
        "rail": 50,
        "walk": 5,
        "bike": 15,
        "other": 30,
        "unknown": 30,
    }
    data = data.with_columns(
        distance=(pl.col("duration") / 60)
        * pl.col("mode").replace_strict(speeds)
    )

    data = data.sort(["pid", "seq"])

    # Handle midnight-crossing trips
    data = fix.day_wrap(data)

    return data
