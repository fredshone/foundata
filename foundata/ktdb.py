from pathlib import Path

import polars as pl

from foundata import fix, utils

SOURCE = "ktdb"

SPEEDS = {
    "car": 60,
    "bus": 40,
    "rail": 50,
    "walk": 5,
    "bike": 15,
    "other": 30,
    "unknown": 30,
}


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
    print(f"Loading {SOURCE} data...")
    attributes = load_persons(data_root, person_config)
    trips = load_trips(data_root, trips_config)

    # transfer access egress distances from trips to attributes
    ae_distances = (
        trips.select(["pid", "access_egress_distance"])
        .unique()
        .filter(pl.col("access_egress_distance").is_not_null())
    )

    attributes = attributes.join(ae_distances, on="pid", how="left")
    trips = trips.drop("access_egress_distance")

    # check for missing access-egress distances and warn
    missing_ae = attributes.filter(pl.col("access_egress_distance").is_null())
    if missing_ae.height > 0:
        perc = missing_ae.height / attributes.height * 100
        print(
            f"WARNING: {missing_ae.height} ({perc:.2f}%) records missing access-egress distance"
        )

    # Derive per-person region from first trip's origin zone prefix
    # todo: specifically find home location from trips based on trip purpose
    person_region = trips.group_by("pid").agg(
        region_code=pl.col("region_code").first()
    )
    trips = trips.drop("region_code")
    attributes = attributes.join(person_region, on="pid", how="left")

    weather = load_weather()

    # check for missing regions in weather data
    weather_regions = set(
        pl.Series(weather.select("region_code").unique()).to_list()
    )
    attribute_regions = set(
        pl.Series(person_region.select("region_code").unique()).to_list()
    )
    missing_regions = attribute_regions - weather_regions
    if missing_regions:
        raise ValueError(f"Missing weather data for regions: {missing_regions}")

    # join weather
    attributes = attributes.join(
        weather,
        left_on=["survey_date", "region_code"],
        right_on=["date", "region_code"],
        how="left",
    ).drop(["survey_date", "region_code"])

    # Prefix IDs with source name for global uniqueness
    attributes = attributes.with_columns(
        pid=pl.lit(SOURCE) + pl.col("pid").cast(pl.String),
        hid=pl.lit(SOURCE) + pl.col("pid").cast(pl.String),  # duplicate of pid
    )
    trips = trips.with_columns(
        pid=pl.lit(SOURCE) + pl.col("pid").cast(pl.String)
    )

    attributes = utils.compute_avg_speed(attributes, trips)

    return attributes, trips


def load_persons(root: str | Path, config: dict) -> pl.DataFrame:
    """Load and normalise person records."""
    root = Path(root).expanduser()
    column_mapping = config["column_mappings"]

    data = pl.read_csv(
        root / "persons.csv", ignore_errors=True, encoding="euc-kr"
    )
    data = data.select(column_mapping.keys()).rename(column_mapping)

    # Cast coded columns to Int32 so replace_strict can match integer YAML keys;
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
        [pl.col(c).cast(pl.Int32, strict=False) for c in int_cols]
    )

    data = data.with_columns(
        hid=pl.col("pid"),
        year=pl.lit(2021, dtype=pl.Int32),
        source=pl.lit(SOURCE),
        country=pl.lit("south korea"),
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
        .map_elements(utils.sample_krw_to_euro, return_dtype=pl.Int32)
        * 1000000
        * 12,
        dwelling=pl.col("dwelling").replace_strict(
            config["dwelling"], default="unknown"
        ),
        vehicles=pl.col("cars").cast(pl.Int32, strict=False).fill_null(0)
        + pl.col("motorcycles").cast(pl.Int32, strict=False).fill_null(0)
        + pl.col("vans").cast(pl.Int32, strict=False).fill_null(0),
        weight=pl.lit(1.0, dtype=pl.Float64),
    )

    data = data.with_columns(
        survey_date=pl.col("date").dt.strftime("%Y-%m-%d"),
        month=pl.col("date").dt.month(),
        day=pl.col("date")
        .dt.weekday()
        .replace_strict(config["weekday"], default="unknown"),
        employment=pl.when(pl.col("_student") == "yes")
        .then(pl.lit("student"))
        .when(pl.col("_employed") == "yes")
        .then(pl.lit("employed"))
        .otherwise(pl.lit("unemployed")),
    ).drop(["_student", "_employed", "date"])

    data = data.with_columns(
        pl.lit("unknown").alias(col)
        for col in [
            "education",
            "disability",
            "relationship",
            "race",
            "ownership",
            "hh_zone",
        ]
    )
    return data


def load_zones() -> pl.DataFrame:
    zones = pl.read_csv(
        utils.get_config_path("ktdb", "zones.csv")
    ).with_columns(
        region_code=pl.col("Administrative dong code")
        .cast(pl.String)
        .str.slice(0, 2)
    )

    zones = (
        zones.select(
            "Administrative dong code", "region_code", "town/village name"
        )
        .rename(
            {"Administrative dong code": "zone", "town/village name": "name"}
        )
        .with_columns(
            hh_zone=pl.when(pl.col("name").str.ends_with("동"))
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


def load_centroids() -> dict[str, tuple[float, float]]:
    """Return {zone_code: (lat, lon)} for all known zones."""
    csv = utils.get_config_path("ktdb", "zone_centroids.csv")
    df = pl.read_csv(csv, schema_overrides={"zone": pl.String})
    return {
        row["zone"]: (row["lat"], row["lon"])
        for row in df.iter_rows(named=True)
    }


def load_distances() -> pl.DataFrame:
    """Load precomputed zone-to-zone haversine distances (km)."""
    path = (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "ktdb"
        / "zone_distances.csv"
    )
    return pl.read_csv(
        path,
        schema={
            "ozone": pl.String,
            "dzone": pl.String,
            "distance_km": pl.Float32,
        },
    )


def load_weather() -> pl.DataFrame:
    csv = utils.get_config_path("ktdb", "weather_regions.csv")
    weather = pl.read_csv(csv, schema_overrides={"region_code": pl.String})
    return weather.with_columns(rain=pl.col("precipitation_mm") > 1).drop(
        "precipitation_mm"
    )


def load_trips(root: str | Path, config: dict) -> pl.DataFrame:
    """Load and normalise trip records.

    TODO: Update file name and path to match raw data layout.
    TODO: Convert tst/tet to minutes since midnight (int).
    TODO: Convert distance to kilometres.
    """
    root = Path(root).expanduser()
    column_mapping = config["column_mappings"]

    data = pl.read_csv(
        root / "trips.csv", ignore_errors=True, encoding="euc-kr"
    )
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

    # access egress for transit
    pt_trips = (
        stages.filter(pl.col("mode").is_in(["bus", "rail"]))
        .select("pid", "seq")
        .unique()
    )
    # pt_trips = stages.join(pt_trips, on=["pid", "seq"], how="inner")

    potential_ae_stages = stages.filter(~pl.col("mode").is_in(["bus", "rail"]))
    ae_stages = (
        potential_ae_stages.join(pt_trips, on=["pid", "seq"], how="inner")
        .with_columns(
            access_egress_distance=(pl.col("duration") / 60)
            * pl.col("mode").replace_strict(SPEEDS)
        )
        .select("pid", "seq", "access_egress_distance")
    )

    ae_distances = (
        (
            ae_stages.group_by(["pid", "seq"]).agg(
                pl.col("access_egress_distance")
                .sum()
                .alias("access_egress_distance")
            )
        )
        .group_by("pid")
        .agg(pl.col("access_egress_distance").mean())
        .select(["pid", "access_egress_distance"])
    )

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
        .join(ae_distances, on="pid", how="left")
    )

    # Distances
    dist_df = load_distances()

    data = data.join(
        dist_df,
        left_on=["ozone", "dzone"],
        right_on=["ozone", "dzone"],
        how="left",
    ).rename({"distance_km": "distance"})

    data = data.with_columns(
        distance=pl.when(pl.col("distance") == 0)
        .then((pl.col("duration") / 60) * pl.col("mode").replace_strict(SPEEDS))
        .otherwise(pl.col("distance") * 1.3)
        .cast(pl.Float32)
    )

    zones = load_zones()
    zone_mapping = dict(zip(zones["zone"], zones["hh_zone"]))

    # Extract 시도 prefix (first 2 chars) before zone codes are overwritten
    data = data.with_columns(
        region_code=pl.col("ozone").cast(pl.String).str.slice(0, 2)
    )

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

    data = data.sort(["pid", "seq"])

    # Handle midnight-crossing trips
    data = fix.day_wrap(data)

    # calc % of pids with access_egress_distance not null
    perc = (
        data.filter(pl.col("access_egress_distance").is_not_null())
        .select("pid")
        .unique()
        .height
        / data.select("pid").unique().height
        * 100
    )
    print(f"Access-egress distance available for {perc:.2f}% of persons")

    return data
