"""KTDB (Korea Transport Database) national travel surveys.

Three survey waves, each in its own subdirectory of the KTDB data folder and
read with the year's part of the nested person/trip dictionaries:

2021: personal travel survey (개인통행실태조사). Separate person and trip
    CSVs; respondents are not linked into households, so hid = pid.
2016: household (가구통행실태조사) and weekend (주말통행실태조사) surveys.
    Both ship one CP949 CSV with a row per linked trip and the household and
    person fields repeated on every row; persons who didn't travel have a
    single row with blank trip fields. Households are identified by (IDX,
    행정구역번호) and persons by household + 가구원구분. In the weekend survey
    each person has a Saturday and a Sunday diary (토일구분), and each diary
    becomes its own plan (pid suffixed _sat/_sun). Members under 5 are not
    surveyed.
2010: household and weekend surveys. Three linked tables (households,
    persons, trip legs) keyed by sheet_code, a 12-digit household id whose
    first 8 digits locate the home 읍면동. The national files code trip ends
    as 251 시군구 zones only. Legs ending at a transfer point are linked into
    trips. The weekday diary has no date, so day is unknown; the weekend
    survey repeats every member as a day-1 (taken to be Saturday) and a day-2
    (Sunday) diary, each its own plan. Members under school age are not
    surveyed. There is no weather. Access/egress distance is left null: walks
    to transit are rarely recorded as legs, so the non-transit legs of
    transit trips are mostly long car or taxi legs to intercity coaches.

None of the surveys have weights.
"""

from pathlib import Path

import polars as pl
from loguru import logger

from foundata import fix, location, utils

SOURCE = "ktdb"
# (year, weekend) of every survey
SURVEYS = [
    (2021, False),
    (2016, False),
    (2016, True),
    (2010, False),
    (2010, True),
]
DIARY_SUFFIX = {1: "_sat", 2: "_sun"}
DIARY_DAY = {1: "saturday", 2: "sunday"}
EMPLOYED = ["employed", "ft-employed", "pt-employed"]
# polars weekday number (1 = Monday) -> template day name
WEEKDAYS = dict(
    enumerate(
        [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ],
        start=1,
    )
)
N_STAGES_2016 = 6


def load_years(
    data_root: str | Path,
    person_config: dict,
    trips_config: dict,
    surveys: list[tuple[int, bool]] = SURVEYS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load and stack the KTDB surveys in `surveys` from `data_root`/<year>/.

    Args:
        data_root: The KTDB data directory, with one subdirectory per year.
        person_config: Parsed (year-keyed) person_dictionary.yaml.
        trips_config: Parsed (year-keyed) trip_dictionary.yaml.
        surveys: (year, weekend) pairs to load.

    Returns:
        (attributes, trips) DataFrames conforming to the template schema.
    """
    all_attributes, all_trips = [], []
    for year, weekend in surveys:
        attributes, trips = load(
            data_root, person_config, trips_config, year, weekend
        )
        attributes = location.add(
            attributes.with_columns(
                **location_codes("home_dong", year, "home")
            ),
            source=SOURCE,
            key=year,
        )
        all_attributes.append(attributes)
        all_trips.append(trips)
    return (
        pl.concat(all_attributes, how="diagonal_relaxed"),
        pl.concat(all_trips, how="diagonal_relaxed"),
    )


def load(
    data_root: str | Path,
    person_config: dict,
    trips_config: dict,
    year: int,
    weekend: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load and normalise one KTDB survey.

    Args:
        data_root: The KTDB data directory, with one subdirectory per year.
        person_config: Parsed (year-keyed) person_dictionary.yaml.
        trips_config: Parsed (year-keyed) trip_dictionary.yaml.
        year: Survey year, 2021, 2016 or 2010.
        weekend: Load the weekend survey instead of the household survey
            (2016 and 2010 only).

    Returns:
        (attributes, trips) DataFrames conforming to the template schema.
    """
    if (year, weekend) not in SURVEYS:
        raise ValueError(
            f"Unsupported KTDB survey: {year}{' weekend' if weekend else ''} "
            f"(expected one of {SURVEYS})"
        )
    root = Path(data_root).expanduser() / str(year)
    person_config = utils.config_for_year(person_config, year)
    trips_config = utils.config_for_year(trips_config, year)
    if year == 2021:
        return load_2021(root, person_config, trips_config)
    if year == 2016:
        return load_2016(root, person_config, trips_config, weekend)
    return load_2010(root, person_config, trips_config, weekend)


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------


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
        .with_columns(hh_zone=zone_class(pl.col("name")))
    )
    return zones


def zone_class(name: pl.Expr) -> pl.Expr:
    """Urbanity of an 읍면동 from its name suffix: 동 (urban neighbourhood),
    읍 (town) or 면/리 (rural township/village)."""
    return (
        pl.when(name.str.ends_with("동"))
        .then(pl.lit("urban"))
        .when(name.str.ends_with("읍"))
        .then(pl.lit("suburban"))
        .when(name.str.ends_with("면") | name.str.ends_with("리"))
        .then(pl.lit("rural"))
        .otherwise(pl.lit("unknown"))
    )


def haversine_km(
    lat1: pl.Expr, lon1: pl.Expr, lat2: pl.Expr, lon2: pl.Expr
) -> pl.Expr:
    """Great-circle distance in km between two points given in degrees."""
    dlat = (lat2 - lat1).radians()
    dlon = (lon2 - lon1).radians()
    a = (dlat / 2).sin() ** 2 + lat1.radians().cos() * lat2.radians().cos() * (
        dlon / 2
    ).sin() ** 2
    return 2 * 6371.0 * a.sqrt().arcsin()


def centroid_distances(
    trips: pl.DataFrame,
    centroids: pl.DataFrame,
    circuity: float,
    speeds: dict[str, float],
) -> pl.DataFrame:
    """Add a distance (km) column from zone centroids.

    trips needs ozone, dzone, mode, tst and tet; centroids has zone, lat and
    lon. The straight line is scaled by circuity. Trips within one zone, with
    an unknown zone, or whose implied speed exceeds 100 km/h fall back to
    duration x mode speed (the trip config's mode_speeds), as in the 2021
    loader.
    """
    o = centroids.rename({"zone": "ozone", "lat": "_olat", "lon": "_olon"})
    d = centroids.rename({"zone": "dzone", "lat": "_dlat", "lon": "_dlon"})
    trips = trips.join(o, on="ozone", how="left").join(
        d, on="dzone", how="left"
    )
    hours = (pl.col("tet") - pl.col("tst")) / 60
    straight = haversine_km(
        pl.col("_olat"), pl.col("_olon"), pl.col("_dlat"), pl.col("_dlon")
    )
    by_speed = hours * mode_speed(speeds)
    return trips.with_columns(
        distance=pl.when(
            straight.is_null()
            | (straight == 0)
            | ((straight * circuity) / hours > 100)
        )
        .then(by_speed)
        .otherwise(straight * circuity)
        .cast(pl.Float32)
    ).drop("_olat", "_olon", "_dlat", "_dlon")


def priority_mode(modes: pl.Expr, priority: list[str]) -> pl.Expr:
    """Aggregate expression picking the highest-priority mode in a group."""
    rank = {m: i for i, m in enumerate(priority)}
    return modes.sort_by(
        modes.replace_strict(rank, default=len(rank), return_dtype=pl.Int8)
    ).first()


def mode_speed(speeds: dict[str, float]) -> pl.Expr:
    """Speed (km/h) of each trip or stage's mode, from a trip config's
    mode_speeds; modes it doesn't list get its "unknown" speed."""
    return pl.col("mode").replace_strict(
        speeds, default=speeds["unknown"], return_dtype=pl.Float64
    )


def access_egress(
    stages: pl.DataFrame,
    keys: list[str],
    person_keys: list[str],
    speeds: dict[str, float],
) -> pl.DataFrame:
    """Mean access/egress distance per person over their transit trips.

    stages has the keys plus mode and duration (minutes) per stage. Non-transit
    stages of transit trips are converted to km at the mode speeds (km/h).
    """
    transit = (
        stages.filter(pl.col("mode").is_in(["bus", "rail"]))
        .select(keys)
        .unique()
    )
    return (
        stages.filter(~pl.col("mode").is_in(["bus", "rail"]))
        .join(transit, on=keys, how="inner")
        .with_columns(
            access_egress_distance=(pl.col("duration").clip(0) / 60)
            * mode_speed(speeds)
        )
        .group_by(keys)
        .agg(pl.col("access_egress_distance").sum())
        .group_by(person_keys)
        .agg(pl.col("access_egress_distance").mean().cast(pl.Float32))
    )


def location_codes(column: str, year: int, role: str) -> dict[str, pl.Expr]:
    """*role*'s 행정동 and 시군구 codes from a household's or trip end's
    dong code (see home_location): for 2021 and 2016 the 10-digit MOIS code
    and its first 5 digits; for 2010 KTDB's 8-digit code mapped to KOSTAT's
    7-digit code, and its first 5 digits."""
    code = location.clean(column)
    if year == 2010:
        dong = location.mapped(column, SOURCE, "dong_2010_kostat.csv")
    else:
        dong = location.clean(column, valid=r"\d{10}")
    return {
        f"{role}_dong": dong,
        f"{role}_sigungu": location.clean(code.str.slice(0, 5), valid=r"\d{5}"),
    }


def zone_points(level: str) -> pl.DataFrame:
    """``zone, lat, lon`` of each unit of a location lookup level."""
    return location.zone_points(SOURCE, level).select(
        zone="code", lat="lat", lon="lon"
    )


def warn_unknown_zones(trips: pl.DataFrame, known: pl.Series) -> None:
    """Warn about trips whose (non-blank) zone code has no point in the
    location lookup; they get no distance."""
    unknown = ~pl.col("ozone").is_in(known.implode()) | ~pl.col("dzone").is_in(
        known.implode()
    )
    blank = (pl.col("ozone").str.strip_chars() == "") | (
        pl.col("dzone").str.strip_chars() == ""
    )
    missing = trips.filter(unknown & blank.not_())
    if missing.height == 0:
        return
    codes = pl.concat([missing["ozone"], missing["dzone"]]).unique().sort()
    codes = codes.filter(~codes.is_in(known.implode())).to_list()
    perc = missing.height / trips.height * 100
    logger.warning(
        f"{missing.height} ({perc:.2f}%) trips use {len(codes)} zone codes "
        f"missing from the location lookup: {', '.join(codes)}"
    )


def load_weather() -> pl.DataFrame:
    csv = utils.get_config_path("ktdb", "weather_regions.csv")
    weather = pl.read_csv(csv, schema_overrides={"region_code": pl.String})
    return weather.with_columns(rain=pl.col("precipitation_mm") > 0).drop(
        "precipitation_mm"
    )


def sample_income(config: dict) -> pl.Expr:
    """Annual EUR household income sampled from the hh_income bracket.

    Config bounds are in millions of KRW per month (e.g. [1, 3] for 1-3
    million KRW/month); scale to actual KRW *before* sampling, since
    sample_to_euro truncates to int(...) after applying the exchange rate,
    so converting afterwards would truncate every bracket to 0.
    """
    rate = config["krw_to_euro"]
    return (
        pl.col("hh_income")
        .replace_strict(config["hh_income"], default=None)
        .map_elements(
            lambda b: utils.sample_to_euro([v * 1_000_000 for v in b], rate),
            return_dtype=pl.Int32,
        )
        * 12
    )


# ----------------------------------------------------------------------
# 2021 personal travel survey
# ----------------------------------------------------------------------


def load_2021(
    root: str | Path, person_config: dict, trips_config: dict
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load the 2021 survey from its year directory, with year-resolved
    configs."""
    logger.info(f"Loading {SOURCE}2021 data...")
    attributes = load_persons_2021(root, person_config)
    trips = load_trips_2021(root, trips_config)
    attributes = utils.assign_retired(attributes, trips)

    # transfer access egress distances from trips to attributes
    ae_distances = (
        trips.select(["pid", "access_egress_distance"])
        .unique()
        .filter(pl.col("access_egress_distance").is_not_null())
    )

    attributes = attributes.join(ae_distances, on="pid", how="left")
    trips = trips.drop("access_egress_distance")

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
    ) - {None}
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
        survey=pl.lit(f"{SOURCE}2021"),
        source=pl.lit(SOURCE),
        country=pl.lit("kr"),
        pid=pl.lit(SOURCE) + pl.col("pid").cast(pl.String),
        # duplicate of pid: the raw data has no household ID linking
        # respondents (each row is an independently surveyed individual),
        # so there is no real household to group by
        hid=pl.lit(SOURCE) + pl.col("pid").cast(pl.String),
    )
    trips = trips.with_columns(
        pid=pl.lit(SOURCE) + pl.col("pid").cast(pl.String),
        hid=pl.lit(SOURCE) + pl.col("pid").cast(pl.String),
    )

    return attributes, trips


def load_persons_2021(root: str | Path, config: dict) -> pl.DataFrame:
    """Load and normalise 2021 person records."""
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
        # Q3 (household income) is answered by only one respondent per
        # household (almost always the non-student adult, R2==99) - other
        # household members (esp. students/children) leave it blank. The
        # raw data has no household ID linking respondents, so there is no
        # way to backfill hh_income for the respondents who weren't asked;
        # it is left null (-> unknown) for them rather than imputed.
        hh_income=sample_income(config),
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
    # occupation only applies to workers, as in 2016 and 2010
    data = data.with_columns(
        occupation=pl.when(pl.col("employment") == "employed")
        .then(pl.col("occupation"))
        .otherwise(pl.lit("void"))
    )

    data = data.with_columns(
        pl.lit("unknown").alias(col)
        for col in [
            "education",
            "disability",
            "relationship",
            "race",
            "ownership",
        ]
    )

    zones = load_zones()
    zone_mapping = dict(zip(zones["zone"], zones["hh_zone"]))
    data = data.with_columns(
        hh_zone=pl.col("home_zone").replace_strict(
            zone_mapping, default="unknown"
        ),
        home_dong=pl.col("home_zone"),
    ).drop("home_zone")
    return data


def load_trips_2021(root: str | Path, config: dict) -> pl.DataFrame:
    """Load and normalise 2021 trip records."""
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
        .with_columns(
            mode=pl.col("mode").replace_strict(
                config["mode"], default="unknown"
            )
        )
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

    ae_distances = access_egress(
        stages, ["pid", "seq"], ["pid"], config["mode_speeds"]
    )

    total_trip_durations = stage_durations.group_by(["pid", "seq"]).agg(
        pl.col("duration").sum()
    )
    # main mode by priority (rail > bus > car > ...), as in 2016 and 2010,
    # so a long walk to the metro doesn't make it a walk trip
    main_trip_modes = stage_modes.group_by(["pid", "seq"]).agg(
        mode=priority_mode(pl.col("mode"), config["mode_priority"])
    )

    # trips with no stage modes or durations are kept, as in 2016: mode
    # unknown, duration from the trip times
    data = (
        data.select(
            "pid", "seq", "ozone", "dzone", "origin", "purpose", "tst", "tet"
        )
        .join(total_trip_durations, on=["pid", "seq"], how="left")
        .join(main_trip_modes, on=["pid", "seq"], how="left")
        .join(ae_distances, on="pid", how="left")
        .with_columns(
            mode=pl.col("mode").fill_null("unknown"),
            duration=pl.col("duration").fill_null(
                (pl.col("tet") - pl.col("tst")) % 1440
            ),
        )
    )

    # Distances: straight line between the dongs' points (from the
    # location lookup)
    points = zone_points("dong_2021")
    data = data.join(
        points.rename({"zone": "ozone", "lat": "_olat", "lon": "_olon"}),
        on="ozone",
        how="left",
    ).join(
        points.rename({"zone": "dzone", "lat": "_dlat", "lon": "_dlon"}),
        on="dzone",
        how="left",
    )
    data = data.with_columns(
        distance=haversine_km(
            pl.col("_olat"), pl.col("_olon"), pl.col("_dlat"), pl.col("_dlon")
        )
    ).drop("_olat", "_olon", "_dlat", "_dlon")
    warn_unknown_zones(data, points["zone"])

    circuity = config["circuity"]
    data = data.with_columns(
        distance=pl.when(
            (pl.col("distance") == 0)
            | (
                (pl.col("distance") * circuity) / (pl.col("duration") / 60)
                > 100
            )
        )
        .then((pl.col("duration") / 60) * mode_speed(config["mode_speeds"]))
        .otherwise(pl.col("distance") * circuity)
        .cast(pl.Float32)
    )

    zones = load_zones()
    zone_mapping = dict(zip(zones["zone"], zones["hh_zone"]))

    data = location.add_trip_ends(
        data.with_columns(
            **location_codes("ozone", 2021, "origin"),
            **location_codes("dzone", 2021, "destination"),
        ),
        source=SOURCE,
        key=2021,
    )
    # Extract 시도 prefix (first 2 chars) before zone codes are overwritten
    data = data.with_columns(
        region_code=pl.when(
            pl.col("ozone").cast(pl.String).str.strip_chars() != ""
        )
        .then(pl.col("ozone").cast(pl.String).str.slice(0, 2))
        .otherwise(None)
    )

    data = data.with_columns(
        ozone=pl.col("ozone").replace_strict(zone_mapping, default="unknown"),
        dzone=pl.col("dzone").replace_strict(zone_mapping, default="unknown"),
    )

    data = (
        data.sort(["pid", "seq"])
        .with_columns(
            oact=pl.when(pl.col("seq") == pl.col("seq").min().over("pid"))
            .then(pl.col("origin"))
            .otherwise(pl.col("purpose").shift(1).over("pid")),
            dact=pl.col("purpose"),
            seq=(pl.col("seq") - 1).cast(pl.Int8),
        )
        .drop(["origin", "purpose"])
    )

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
    logger.debug(f"Access-egress distance available for {perc:.2f}% of persons")

    return data


# ----------------------------------------------------------------------
# 2016 household and weekend surveys
# ----------------------------------------------------------------------


def load_2016(
    root: str | Path,
    person_config: dict,
    trips_config: dict,
    weekend: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load the 2016 household or weekend survey from its year directory,
    with year-resolved configs."""
    survey = f"{SOURCE}2016{'we' if weekend else ''}"
    logger.info(f"Loading {survey} data...")
    path = (
        Path(root).expanduser()
        / person_config["files"]["weekend" if weekend else "household"]
    )
    raw = read_raw_2016(path)

    attributes = load_persons_2016(raw, person_config, survey)
    trips = load_trips_2016(raw, trips_config, survey)
    attributes = utils.assign_retired(attributes, trips)

    ae = trips.group_by("pid").agg(
        pl.col("access_egress_distance").drop_nulls().first()
    )
    attributes = attributes.join(ae, on="pid", how="left")
    trips = trips.drop("access_egress_distance")
    return attributes, trips


def read_raw_2016(path: Path) -> pl.DataFrame:
    raw = pl.read_csv(path, encoding="cp949", infer_schema_length=0)
    # the weekend file writes "(오전/오후)" where the household file has
    # "(오전 오후)"; column_mappings use the household spelling
    return raw.rename({c: c.replace("/", " ") for c in raw.columns})


def with_ids_2016(data: pl.DataFrame, survey: str) -> pl.DataFrame:
    """Add hid and pid. Weekend diaries get a _sat/_sun pid suffix."""
    hid = pl.concat_str(
        pl.lit(survey), pl.col("idx"), pl.col("district"), separator="_"
    )
    suffix = (
        pl.col("weekend_day")
        .cast(pl.Int32, strict=False)
        .replace_strict(DIARY_SUFFIX, default="", return_dtype=pl.String)
        .fill_null("")
    )
    return data.with_columns(
        hid=hid,
        pid=pl.concat_str(hid, pl.col("member"), separator="_") + suffix,
    )


def select_mapped(raw: pl.DataFrame, column_mapping: dict) -> pl.DataFrame:
    """Select and rename mapped columns, adding the ones the file lacks
    (토일구분 exists in the 2016 weekend file only)."""
    cols = [
        pl.col(c).alias(n) if c in raw.columns else pl.lit(None).alias(n)
        for c, n in column_mapping.items()
    ]
    return raw.select(cols)


def load_persons_2016(
    raw: pl.DataFrame, config: dict, survey: str
) -> pl.DataFrame:
    """One row per person (per diary day for the weekend survey)."""
    data = with_ids_2016(select_mapped(raw, config["column_mappings"]), survey)
    data = data.unique(subset="pid", keep="first", maintain_order=True)

    int_cols = [
        "birth_year",
        "sex",
        "has_licence",
        "education",
        "occupation",
        "can_wfh",
        "work_hours",
        "relationship",
        "travel_year",
        "travel_month",
        "travel_dom",
        "weekend_day",
        "hh_size",
        "dwelling",
        "hh_income",
    ]
    vehicle_cols = [f"vehicle{i}" for i in range(1, 7)]
    motorcycle_cols = [f"motorcycles{i}" for i in range(1, 5)]
    data = data.with_columns(
        pl.col(int_cols + vehicle_cols + motorcycle_cols).cast(
            pl.Int32, strict=False
        )
    )

    date = pl.concat_str(
        pl.col("travel_year"),
        pl.col("travel_month"),
        pl.col("travel_dom"),
        separator="-",
    ).str.to_date("%Y-%-m-%-d", strict=False)
    student = pl.col("education").is_in([2, 3, 4])
    employment = (
        pl.when(student)
        .then(pl.lit("student"))
        .otherwise(
            pl.col("occupation").replace_strict(
                config["employment"], default="unknown"
            )
        )
    )
    data = data.with_columns(
        year=pl.lit(2016, dtype=pl.Int32),
        month=pl.col("travel_month"),
        age=pl.lit(2016) - pl.col("birth_year"),
        date=date,
        sex=pl.col("sex").replace_strict(config["sex"], default="unknown"),
        has_licence=pl.col("has_licence").replace_strict(
            config["has_licence"], default="unknown"
        ),
        education=pl.col("education").replace_strict(
            config["education"], default="unknown"
        ),
        occupation=pl.col("occupation").replace_strict(
            config["occupation"], default="unknown"
        ),
        relationship=pl.col("relationship").replace_strict(
            config["relationship"], default="unknown"
        ),
        employment=pl.when(employment == "employed")
        .then(
            pl.col("work_hours").replace_strict(
                config["work_hours"], default="employed"
            )
        )
        .otherwise(employment),
        dwelling=pl.col("dwelling").replace_strict(
            config["dwelling"], default="unknown"
        ),
        hh_income=sample_income(config),
        vehicles=pl.sum_horizontal(
            pl.col(vehicle_cols).is_not_null().cast(pl.Int32)
        )
        + pl.sum_horizontal(pl.col(motorcycle_cols).fill_null(0)),
        ownership=pl.lit("unknown"),
        disability=pl.lit("unknown"),
        race=pl.lit("unknown"),
        weight=pl.lit(1.0, dtype=pl.Float64),
    )
    data = data.with_columns(
        # 재택근무 is only ticked by those working at home
        can_wfh=pl.when(pl.col("can_wfh") == 1)
        .then(pl.lit("yes"))
        .when(pl.col("employment").is_in(EMPLOYED))
        .then(pl.lit("unknown"))
        .otherwise(pl.lit("void")),
        # weekend diaries are labelled Saturday/Sunday; a few dates disagree
        day=pl.coalesce(
            pl.col("weekend_day").replace_strict(
                config["weekend_day"], default=None, return_dtype=pl.String
            ),
            pl.col("date").dt.weekday().replace_strict(WEEKDAYS, default=None),
            pl.col("weekday").replace_strict(config["weekday"], default=None),
            pl.lit("unknown"),
        ),
        survey_date=pl.col("date").dt.strftime("%Y-%m-%d"),
        region_code=pl.col("home_zone").str.slice(0, 2),
    )

    zones = load_zones()
    zone_mapping = dict(zip(zones["zone"].cast(pl.String), zones["hh_zone"]))
    data = data.with_columns(
        hh_zone=pl.col("home_zone").replace_strict(
            zone_mapping, default="unknown"
        ),
        home_dong=pl.col("home_zone"),
    )

    weather = load_weather()
    data = data.join(
        weather,
        left_on=["survey_date", "region_code"],
        right_on=["date", "region_code"],
        how="left",
    )

    return data.with_columns(
        survey=pl.lit(survey), source=pl.lit(SOURCE), country=pl.lit("kr")
    ).drop(
        "idx",
        "district",
        "member",
        "weekend_day",
        "birth_year",
        "work_hours",
        "travel_year",
        "travel_month",
        "travel_dom",
        "weekday",
        "date",
        "survey_date",
        "region_code",
        "home_zone",
        *vehicle_cols,
        *motorcycle_cols,
    )


def clock(ampm: str, hour: str, minute: str, config: dict) -> pl.Expr:
    """2016 minutes since midnight from the AM/PM code, hour (0-11) and
    minute."""
    return (
        pl.col(ampm)
        .cast(pl.Int32, strict=False)
        .replace_strict(config["ampm"], default=None, return_dtype=pl.Int32)
        + pl.col(hour).cast(pl.Int32, strict=False) * 60
        + pl.col(minute).cast(pl.Int32, strict=False)
    )


def load_trips_2016(
    raw: pl.DataFrame, config: dict, survey: str
) -> pl.DataFrame:
    """One row per linked trip, with stages collapsed to a single mode."""
    data = with_ids_2016(select_mapped(raw, config["column_mappings"]), survey)
    data = data.filter(pl.col("travelled") == "1").with_columns(
        seq=pl.col("seq").cast(pl.Int32),
        tst=clock("tst_ampm", "tst_hr", "tst_min", config),
        origin=pl.col("origin")
        .cast(pl.Int32, strict=False)
        .replace_strict(config["origin"], default="unknown"),
        purpose=pl.col("purpose")
        .cast(pl.Int32, strict=False)
        .replace_strict(config["purpose"], default="unknown"),
    )

    # stages: mode and arrival time, in long format
    stages = pl.concat(
        [
            data.select(
                "pid",
                "seq",
                "tst",
                stage=pl.lit(i, dtype=pl.Int8),
                mode=pl.col(f"mode{i}").cast(pl.Int32, strict=False),
                arrival=clock(
                    f"arr_ampm{i}", f"arr_hr{i}", f"arr_min{i}", config
                ),
            )
            for i in range(1, N_STAGES_2016 + 1)
        ]
    ).filter(pl.col("mode").is_not_null())
    stages = stages.sort("pid", "seq", "stage").with_columns(
        mode=pl.col("mode").replace_strict(config["mode"], default="unknown"),
        duration=pl.col("arrival")
        - pl.col("arrival")
        .shift(1)
        .over("pid", "seq")
        .fill_null(pl.col("tst")),
    )
    trip_stages = stages.group_by("pid", "seq").agg(
        mode=priority_mode(pl.col("mode"), config["mode_priority"]),
        tet=pl.col("arrival").last(),
    )
    ae = access_egress(stages, ["pid", "seq"], ["pid"], config["mode_speeds"])

    data = (
        data.select(
            "pid", "hid", "seq", "ozone", "dzone", "origin", "purpose", "tst"
        )
        .join(trip_stages, on=["pid", "seq"], how="left")
        .join(ae, on="pid", how="left")
        .sort("pid", "seq")
        .with_columns(
            oact=pl.when(pl.col("seq") == pl.col("seq").min().over("pid"))
            .then(pl.col("origin"))
            .otherwise(pl.col("purpose").shift(1).over("pid")),
            dact=pl.col("purpose"),
            mode=pl.col("mode").fill_null("unknown"),
        )
        .drop("origin", "purpose")
    )

    data = fix.day_wrap(data)

    data = centroid_distances(
        data,
        zone_points("dong_2016"),
        config["circuity"],
        config["mode_speeds"],
    )

    data = location.add_trip_ends(
        data.with_columns(
            **location_codes("ozone", 2016, "origin"),
            **location_codes("dzone", 2016, "destination"),
        ),
        source=SOURCE,
        key=2016,
    )
    zones = load_zones()
    zone_mapping = dict(zip(zones["zone"].cast(pl.String), zones["hh_zone"]))
    data = data.with_columns(
        ozone=pl.col("ozone").replace_strict(zone_mapping, default="unknown"),
        dzone=pl.col("dzone").replace_strict(zone_mapping, default="unknown"),
        seq=(pl.col("seq") - 1).cast(pl.Int8),
    )
    return data.sort("pid", "seq")


# ----------------------------------------------------------------------
# 2010 household and weekend surveys
# ----------------------------------------------------------------------


def load_2010(
    root: str | Path,
    person_config: dict,
    trips_config: dict,
    weekend: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load the 2010 household or weekend survey from its year directory,
    with year-resolved configs."""
    survey = f"{SOURCE}2010{'we' if weekend else ''}"
    logger.info(f"Loading {survey} data...")
    root = Path(root).expanduser()
    files = person_config["files"]["weekend" if weekend else "household"]
    offset = person_config["weekend_day2_offset"] if weekend else None

    attributes = load_persons_2010(
        read_table(root / files["households"]),
        read_table(root / files["persons"]),
        person_config,
        survey,
        offset,
    )
    trips = load_trips_2010(
        read_table(root / files["trips"]),
        trips_config,
        survey,
        offset,
        zone_classes(attributes),
    )
    attributes = utils.assign_retired(attributes, trips).drop("home_zone")
    return attributes, trips


def read_table(path: Path) -> pl.DataFrame:
    """Read a 2010 tab-separated text table or an xlsx sheet as strings, with
    surrounding whitespace (codes are space-padded) and any BOM removed."""
    if path.suffix == ".xlsx":
        data = pl.read_excel(path, infer_schema_length=0)
    else:
        data = pl.read_csv(
            path, separator="\t", quote_char=None, infer_schema_length=0
        )
    data = data.rename({c: c.lstrip("﻿") for c in data.columns})
    return data.with_columns(pl.all().str.strip_chars()).with_columns(
        pl.when(pl.all() != "").then(pl.all()).name.keep()
    )


def with_ids_2010(data: pl.DataFrame, survey: str, offset: int | None):
    """Pad sheet_code and add hid and pid.

    Weekend member numbers above `offset` are the day-2 diary of member
    number - offset; each diary gets a _sat/_sun pid suffix.
    """
    member = pl.col("member").cast(pl.Int32)
    if offset is None:
        diary = pl.lit(None, dtype=pl.Int32)
    else:
        diary = pl.when(member > offset).then(2).otherwise(1)
        member = (
            pl.when(member > offset).then(member - offset).otherwise(member)
        )
    data = data.with_columns(
        sheet_code=pl.col("sheet_code").str.zfill(12),
        member=member,
        diary=diary,
    )
    hid = pl.lit(f"{survey}_") + pl.col("sheet_code")
    suffix = (
        pl.col("diary")
        .replace_strict(DIARY_SUFFIX, default="", return_dtype=pl.String)
        .fill_null("")
    )
    return data.with_columns(
        hid=hid,
        pid=pl.concat_str(hid, pl.col("member"), separator="_") + suffix,
    )


def load_persons_2010(
    households: pl.DataFrame,
    persons: pl.DataFrame,
    config: dict,
    survey: str,
    offset: int | None,
) -> pl.DataFrame:
    """One row per person (per diary day for the weekend survey)."""
    mappings = config["column_mappings"]
    households = (
        households.select(mappings["households"].keys())
        .rename(mappings["households"])
        .with_columns(sheet_code=pl.col("sheet_code").str.zfill(12))
    )
    persons = persons.select(mappings["persons"].keys()).rename(
        mappings["persons"]
    )
    data = with_ids_2010(persons, survey, offset).join(
        households, on="sheet_code", how="left"
    )

    vehicle_cols = [
        "cars",
        "vans",
        "small_trucks",
        "taxis",
        "motorcycles",
        "trucks",
    ]
    int_cols = [
        "birth_year",
        "sex",
        "has_licence",
        "relationship",
        "occupation",
        "employee_type",
        "hh_size",
        "dwelling",
        "ownership",
        "hh_income",
    ]
    data = data.with_columns(
        pl.col(int_cols + vehicle_cols).cast(pl.Int32, strict=False)
    )

    employment = pl.col("occupation").replace_strict(
        config["employment"], default="unknown"
    )
    data = data.with_columns(
        year=pl.lit(2010, dtype=pl.Int32),
        month=pl.lit(None, dtype=pl.Int32),
        day=pl.col("diary")
        .replace_strict(DIARY_DAY, default=None, return_dtype=pl.String)
        .fill_null("unknown"),
        age=pl.lit(2010) - pl.col("birth_year"),
        sex=pl.col("sex").replace_strict(config["sex"], default="unknown"),
        has_licence=pl.col("has_licence").replace_strict(
            config["has_licence"], default="unknown"
        ),
        relationship=pl.col("relationship").replace_strict(
            config["relationship"], default="unknown"
        ),
        occupation=pl.col("occupation").replace_strict(
            config["occupation"], default="unknown"
        ),
        employment=pl.when(employment == "employed")
        .then(
            pl.col("employee_type").replace_strict(
                config["employee_type"], default="employed"
            )
        )
        .otherwise(employment),
        dwelling=pl.col("dwelling").replace_strict(
            config["dwelling"], default="unknown"
        ),
        ownership=pl.col("ownership").replace_strict(
            config["ownership"], default="unknown"
        ),
        hh_income=sample_income(config),
        vehicles=pl.sum_horizontal(pl.col(vehicle_cols).fill_null(0)),
        education=pl.lit("unknown"),
        disability=pl.lit("unknown"),
        race=pl.lit("unknown"),
        weight=pl.lit(1.0, dtype=pl.Float64),
        home_zone=pl.col("sheet_code").str.slice(2, 3),
    )
    data = data.with_columns(
        can_wfh=pl.when(pl.col("employee_type") == 1)
        .then(pl.lit("yes"))
        .when(pl.col("employment").is_in(EMPLOYED))
        .then(pl.lit("unknown"))
        .otherwise(pl.lit("void"))
    )

    zones = load_zones_2010()
    data = data.with_columns(
        hh_zone=pl.col("sheet_code")
        .str.slice(0, 8)
        .replace_strict(
            dict(zip(zones["code"], zones["hh_zone"])), default="unknown"
        ),
        # KTDB's own 8-digit 2010 읍면동 code (see home_location)
        home_dong=pl.col("sheet_code").str.slice(0, 8),
    )

    return data.with_columns(
        survey=pl.lit(survey), source=pl.lit(SOURCE), country=pl.lit("kr")
    ).drop(
        "sheet_code",
        "member",
        "diary",
        "birth_year",
        "employee_type",
        *vehicle_cols,
    )


def load_zones_2010() -> pl.DataFrame:
    """2010 읍면동 codes (SSGGGDDD) with their urbanity."""
    zones = pl.read_csv(
        utils.get_config_path("ktdb", "zones_2010.csv"), infer_schema_length=0
    )
    return zones.with_columns(hh_zone=zone_class(pl.col("dong")))


def zone_classes(attributes: pl.DataFrame) -> dict[str, str]:
    """Urbanity of each 2010 251 zone: the most common hh_zone of the
    households living there (trip ends are only known to the 시군구)."""
    modal = (
        attributes.unique("hid")
        .filter(pl.col("hh_zone") != "unknown")
        .group_by("home_zone", "hh_zone")
        .len()
        .sort("len", "hh_zone", descending=[True, False])
        .unique("home_zone", keep="first")
    )
    return dict(zip(modal["home_zone"], modal["hh_zone"]))


def hhmm(col: str) -> pl.Expr:
    """Minutes since midnight from an hhmm string (zero padding optional)."""
    s = pl.col(col).str.zfill(4)
    return s.str.slice(0, 2).cast(pl.Int32, strict=False) * 60 + s.str.slice(
        2, 2
    ).cast(pl.Int32, strict=False)


def load_trips_2010(
    legs: pl.DataFrame,
    config: dict,
    survey: str,
    offset: int | None,
    zone_mapping: dict[str, str],
) -> pl.DataFrame:
    """Link 2010 legs into trips. One row per trip."""
    mapping = config["column_mappings"]
    legs = with_ids_2010(
        legs.select(mapping.keys()).rename(mapping), survey, offset
    )
    legs = legs.with_columns(
        leg=pl.col("leg").cast(pl.Int32),
        ozone=pl.col("ozone").str.zfill(3),
        dzone=pl.col("dzone").str.zfill(3),
        tst=hhmm("tst"),
        tet=hhmm("tet"),
        end_type=pl.col("end_type").cast(pl.Int32, strict=False),
        origin=pl.col("origin")
        .cast(pl.Int32, strict=False)
        .replace_strict(config["origin"], default="unknown"),
        purpose=pl.col("purpose")
        .cast(pl.Int32, strict=False)
        .replace_strict(config["purpose"], default="unknown"),
        mode=pl.col("mode")
        .cast(pl.Int32, strict=False)
        .replace_strict(config["mode"], default="unknown"),
    ).sort("pid", "leg")

    # a trip starts at every leg that doesn't continue a transfer
    transfer = config["transfer_end_type"]
    legs = legs.with_columns(
        starts_trip=(
            pl.col("end_type").shift(1).over("pid").fill_null(-1) != transfer
        ).cast(pl.Int32),
        duration=(pl.col("tet") - pl.col("tst")) % 1440,
    ).with_columns(trip=pl.col("starts_trip").cum_sum().over("pid"))

    trips = (
        legs.group_by("pid", "trip", maintain_order=True)
        .agg(
            hid=pl.col("hid").first(),
            ozone=pl.col("ozone").first(),
            dzone=pl.col("dzone").last(),
            origin=pl.col("origin").first(),
            purpose=pl.col("purpose").last(),
            tst=pl.col("tst").first(),
            tet=pl.col("tet").last(),
            mode=priority_mode(pl.col("mode"), config["mode_priority"]),
        )
        .sort("pid", "trip")
        .with_columns(
            seq=(pl.col("trip") - 1).cast(pl.Int8),
            oact=pl.when(pl.col("trip") == 1)
            .then(pl.col("origin"))
            .otherwise(pl.col("purpose").shift(1).over("pid")),
            dact=pl.col("purpose"),
        )
        .drop("trip", "origin", "purpose")
    )

    trips = fix.day_wrap(trips)

    # the 251 zones are the 2010 시군구, located by the sigungu_2010 lookup
    sigungu = pl.read_csv(
        utils.get_config_path("ktdb", "zone_2010_sigungu.csv"),
        infer_schema_length=0,
    )
    points = zone_points("sigungu_2010").join(
        sigungu, left_on="zone", right_on="to"
    )
    trips = centroid_distances(
        trips,
        points.select(zone="from", lat="lat", lon="lon"),
        config["circuity"],
        config["mode_speeds"],
    )
    trips = location.add_trip_ends(
        trips.with_columns(
            origin_sigungu=location.mapped(
                "ozone", SOURCE, "zone_2010_sigungu.csv"
            ),
            destination_sigungu=location.mapped(
                "dzone", SOURCE, "zone_2010_sigungu.csv"
            ),
        ),
        source=SOURCE,
        key=2010,
    )

    return trips.with_columns(
        ozone=pl.col("ozone").replace_strict(zone_mapping, default="unknown"),
        dzone=pl.col("dzone").replace_strict(zone_mapping, default="unknown"),
    ).sort("pid", "seq")
