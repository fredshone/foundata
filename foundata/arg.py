"""Argentine household mobility surveys (Ministerio de Transporte, DGPPSE).

Five surveys are combined into one source, each keyed by a short name:
``salta``, ``rc`` (Resistencia-Corrientes), ``nq`` (Neuquén-Cipolletti),
``sfp`` (Santa Fe-Paraná) and ``enmodo`` (Buenos Aires metropolitan region).
Each ships SPSS .sav tables for households, persons, trips and stages linked
by numeric ids, and records one weekday diary per person (the day before
the interview, 04:00 to 04:00). pid and hid are ``arg<survey><raw id>``.
There are no distances, so they are estimated from stage times and mode
speeds. Codes come from the value labels embedded in the .sav files.
"""

import datetime as dt
import struct
import tempfile
from pathlib import Path

import pandas as pd
import polars as pl
import pyreadstat

from foundata import fix
from foundata.utils import sample_to_euro

SOURCE = "arg"
COUNTRY = "ar"

# raw table paths under data_root, per survey
SURVEYS = {
    "salta": {
        "hh": "encuestas_salta/bases_emd_salta/SALTA_Hogares.sav",
        "persons": "encuestas_salta/bases_emd_salta/SALTA_Personas.sav",
        "trips": "encuestas_salta/bases_emd_salta/SALTA_Viajes.sav",
        "stages": "encuestas_salta/bases_emd_salta/SALTA_Modos.sav",
    },
    "rc": {
        "hh": "emd-resistencia-corrientes/bases_emd_resistencia_corrientes/Bases de Datos EMD Resistencia- Corrientes/CH-CTES_HOGAR.sav",
        "persons": "emd-resistencia-corrientes/bases_emd_resistencia_corrientes/Bases de Datos EMD Resistencia- Corrientes/CH-CTES_PERSONAS.sav",
        "trips": "emd-resistencia-corrientes/bases_emd_resistencia_corrientes/Bases de Datos EMD Resistencia- Corrientes/CH-CTES_VIAJES.sav",
        "stages": "emd-resistencia-corrientes/bases_emd_resistencia_corrientes/Bases de Datos EMD Resistencia- Corrientes/CH-CTES_MODOS.sav",
    },
    "nq": {
        "hh": "encuestas_neuquen-cipolletti/bases_emd_neuquen_cipolletti/Bases de Datos EMD Neuquén- Cippolletti/NEUQUEN_CIPPO_HOGARES.sav",
        "persons": "encuestas_neuquen-cipolletti/bases_emd_neuquen_cipolletti/Bases de Datos EMD Neuquén- Cippolletti/NEUQUEN_CIPPO_PERSONAS.sav",
        "trips": "encuestas_neuquen-cipolletti/bases_emd_neuquen_cipolletti/Bases de Datos EMD Neuquén- Cippolletti/NEUQUEN_CIPPO_VIAJES.sav",
        "stages": "encuestas_neuquen-cipolletti/bases_emd_neuquen_cipolletti/Bases de Datos EMD Neuquén- Cippolletti/NEUQUEN_CIPPO_MODOS.sav",
    },
    "sfp": {
        "hh": "encuestas-santafe-parana/bases_emd_santafe_parana_2012/SANTA FE PARANA_Hogares.sav",
        "persons": "encuestas-santafe-parana/bases_emd_santafe_parana_2012/SANTA FE PARANA_Personas.sav",
        "trips": "encuestas-santafe-parana/bases_emd_santafe_parana_2012/SANTA FE PARANA_Viajes.sav",
        "stages": "encuestas-santafe-parana/bases_emd_santafe_parana_2012/SANTA FE PARANA_Medios.sav",
    },
    "enmodo": {
        "hh": "enmodo/bases_enmodo/ENMODO_HOGARES_pub_20121115.sav",
        "persons": "enmodo/bases_enmodo/ENMODO_PERSONAS_pub_20121115.sav",
        "trips": "enmodo/bases_enmodo/ENMODO_VIAJES_pub_20121115.sav",
        "stages": "enmodo/bases_enmodo/ENMODO_ETAPAS_pub_20121115.sav",
    },
}

WEEKDAYS = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]


def load(
    data_root: str | Path,
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
    surveys: list[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load and normalise the Argentine surveys.

    Args:
        data_root: Path to the raw data directory for this source.
        hh_config: Parsed hh_dictionary.yaml config.
        person_config: Parsed person_dictionary.yaml config.
        trips_config: Parsed trip_dictionary.yaml config.
        surveys: Survey keys to load (default: all in SURVEYS).

    Returns:
        (attributes, trips) DataFrames conforming to the template schema.
    """
    print(f"Loading {SOURCE} data...")
    root = Path(data_root).expanduser()
    loaded = [
        load_survey(root, survey, hh_config, person_config, trips_config)
        for survey in surveys or SURVEYS
    ]
    attributes = pl.concat([a for a, _ in loaded], how="diagonal_relaxed")
    trips = pl.concat([t for _, t in loaded], how="vertical_relaxed")
    return attributes, trips


def load_survey(
    root: Path,
    survey: str,
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(attributes, trips) for one survey, with prefixed pid and hid."""
    paths = SURVEYS[survey]
    trip_mappings = trips_config["column_mappings"][survey]

    def read(table: str, column_mapping: dict) -> pl.DataFrame:
        return read_sav(root / paths[table], list(column_mapping)).rename(
            column_mapping
        )

    hhs = load_households(
        read("hh", hh_config["column_mappings"][survey]), hh_config, survey
    )
    persons = read("persons", person_config["column_mappings"][survey])
    persons = persons.with_columns(pl.col("pid", "hid").cast(pl.Int64))
    hhs = hhs.join(
        persons.group_by("hid").agg(hh_size=pl.len().cast(pl.Int32)), on="hid"
    )
    persons = load_persons(persons, person_config, survey)

    home_zones = persons.join(hhs, on="hid").select("pid", "hh_zone")
    trips = load_trips(
        read("trips", trip_mappings["trips"]),
        read("stages", trip_mappings["stages"]),
        trips_config,
        survey,
        home_zones,
    )

    # persons who said they travelled but have no trips can't be used
    persons = persons.filter(
        (pl.col("travelled") == "no")
        | pl.col("pid").is_in(trips["pid"].unique().implode())
    ).drop("travelled")
    trips = trips.filter(pl.col("pid").is_in(persons["pid"].implode()))

    prefix = SOURCE + survey
    attributes = (
        persons.join(hhs, on="hid", how="left")
        .join(travel_dates(persons, trips, hhs), on="pid", how="left")
        .with_columns(diary_date_columns(hh_config["year"][survey]))
        .drop("travel_date", "fallback_date")
        .with_columns(
            source=pl.lit(SOURCE),
            country=pl.lit(COUNTRY),
            pid=pl.lit(prefix) + pl.col("pid").cast(pl.String),
            hid=pl.lit(prefix) + pl.col("hid").cast(pl.String),
        )
    )
    trips = trips.select(
        "seq",
        "ozone",
        "dzone",
        "oact",
        "dact",
        "mode",
        "tst",
        "tet",
        "distance",
        pid=pl.lit(prefix) + pl.col("pid").cast(pl.String),
    ).select("pid", pl.exclude("pid"))
    return attributes, trips


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_sav(path: Path, columns: list[str]) -> pl.DataFrame:
    """Read columns of an SPSS .sav file into polars, codes left unlabelled.

    Clock times become minutes since midnight and dates become pl.Date.
    Files whose long variable names aren't valid in their declared encoding
    (NEUQUEN_CIPPO_HOGARES.sav has a latin-1 "Año") are read from a copy
    with those names transliterated to ASCII.
    """
    try:
        df, _ = pyreadstat.read_sav(
            str(path), usecols=columns, apply_value_formats=False
        )
    except UnicodeDecodeError:
        df = None
    # usecols silently skips the names it can't decode
    if df is None or set(columns) - set(df.columns):
        with tempfile.TemporaryDirectory() as tmp:
            patched = Path(tmp) / "patched.sav"
            patched.write_bytes(ascii_long_names(path.read_bytes()))
            df, _ = pyreadstat.read_sav(
                str(patched), usecols=columns, apply_value_formats=False
            )
    missing = set(columns) - set(df.columns)
    if missing:
        raise KeyError(f"{path.name} has no columns {sorted(missing)}")
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = _convert_objects(df[col])
    return pl.from_pandas(df[columns])


def _convert_objects(series: pd.Series) -> pd.Series:
    """time -> minutes since midnight, date -> datetime64, else unchanged."""
    values = series.dropna()
    if values.empty:
        return series
    first = values.iloc[0]
    if isinstance(first, dt.time):
        return series.map(
            lambda t: t.hour * 60 + t.minute if isinstance(t, dt.time) else None
        ).astype("float64")
    if isinstance(first, (dt.date, dt.datetime)):
        return pd.to_datetime(series, errors="coerce")
    return series


_ASCII = str.maketrans("ñÑóÓáÁéÉíÍúÚü", "nNoOaAeEiIuUu")


def _ascii(raw: bytes) -> bytes:
    text = raw.decode("latin-1").translate(_ASCII)
    return "".join(c if ord(c) < 128 else "_" for c in text).encode("ascii")


def ascii_long_names(data: bytes) -> bytes:
    """Transliterate variable names to ASCII in a .sav file's dictionary.

    Walks the dictionary records after the 176-byte file header: the 8-byte
    short name of each variable record (type 2) and the long variable names
    record (type 7, subtype 13). Everything else is copied unchanged.
    """
    out = bytearray(data)
    pos = 176
    while pos < len(data):
        (rec_type,) = struct.unpack_from("<i", data, pos)
        if rec_type == 2:
            _, has_label, n_missing = struct.unpack_from("<iii", data, pos + 4)
            out[pos + 24 : pos + 32] = _ascii(data[pos + 24 : pos + 32])
            pos += 32
            if has_label:
                (length,) = struct.unpack_from("<i", data, pos)
                pos += 4 + (length + 3) // 4 * 4
            pos += 8 * abs(n_missing)
        elif rec_type == 3:
            (count,) = struct.unpack_from("<i", data, pos + 4)
            pos += 8
            for _ in range(count):
                length = data[pos + 8]
                pos += 8 + (length + 8) // 8 * 8
        elif rec_type == 4:
            (count,) = struct.unpack_from("<i", data, pos + 4)
            pos += 8 + 4 * count
        elif rec_type == 6:
            (count,) = struct.unpack_from("<i", data, pos + 4)
            pos += 8 + 80 * count
        elif rec_type == 7:
            subtype, size, count = struct.unpack_from("<iii", data, pos + 4)
            body = slice(pos + 16, pos + 16 + size * count)
            if subtype == 13:
                out[body] = _ascii(data[body])
            pos = body.stop
        else:  # 999: end of dictionary
            break
    return bytes(out)


# ---------------------------------------------------------------------------
# Value mapping helpers
# ---------------------------------------------------------------------------


def survey_map(config: dict, key: str, survey: str):
    """A per-survey value map, falling back to the shared default."""
    value = config[key]
    return value.get(survey, value["default"])


def recode(col: str, mapping: dict, default: str | None = "unknown") -> pl.Expr:
    """Map raw codes (numeric codes are cast to int first) to strings."""
    if not mapping:
        return pl.lit(default, dtype=pl.String)
    expr = pl.col(col)
    if not all(isinstance(k, str) for k in mapping):
        expr = expr.cast(pl.Int64, strict=False)
    return expr.replace_strict(mapping, default=default, return_dtype=pl.String)


# ---------------------------------------------------------------------------
# Households
# ---------------------------------------------------------------------------


def load_households(
    hhs: pl.DataFrame, config: dict, survey: str
) -> pl.DataFrame:
    """One row per hogar: dwelling, hh_zone, vehicles, hh_income (annual EUR)
    and fallback_date (the day before the interview, when recorded)."""
    missing = config["missing_codes"]

    def count(col: str) -> pl.Expr:
        # counts are only asked when the hogar has any, so null means 0
        return (
            pl.when(pl.col(col).is_in(missing))
            .then(None)
            .otherwise(pl.col(col).fill_null(0))
            .cast(pl.Int32)
        )

    counts = [count(c) for c in ("cars", "motorcycles", "mopeds") if c in hhs]
    motos = [c for c in hhs.columns if c.startswith("moto_")]
    if motos:
        counts.append(
            pl.sum_horizontal(
                (pl.col(c).is_not_null() & ~pl.col(c).is_in(missing)).cast(
                    pl.Int32
                )
                for c in motos
            )
        )
    vehicles = counts[0]
    for c in counts[1:]:
        vehicles = vehicles + c  # null if any count is missing

    rate = config["eur_per_ars"].get(survey)
    if "hh_income_band" in hhs:
        income = income_from_band(config["hh_income_bands"][survey], rate)
    elif "hh_income" in hhs and rate is not None:
        income = (
            pl.when(
                (pl.col("hh_income") < 0) | pl.col("hh_income").is_in(missing)
            )
            .then(None)
            .otherwise(pl.col("hh_income") * 12 * rate)
            .round()
            .cast(pl.Int32)
        )
    else:
        income = pl.lit(None, dtype=pl.Int32)

    return hhs.select(
        hid=pl.col("hid").cast(pl.Int64),
        dwelling=recode("dwelling", survey_map(config, "dwelling", survey)),
        hh_zone=recode("hh_zone", survey_map(config, "hh_zone", survey)),
        vehicles=vehicles,
        hh_income=income,
        ownership=pl.lit("unknown"),
        fallback_date=interview_eve(hhs),
    )


def income_from_band(bands: dict, rate: float) -> pl.Expr:
    """Annual EUR sampled uniformly within the band's annual ARS bounds."""
    return (
        pl.col("hh_income_band")
        .cast(pl.Int64, strict=False)
        .replace_strict(
            {k: list(v) for k, v in bands.items()},
            default=[0],  # sample_to_euro gives null for a single bound
            return_dtype=pl.List(pl.Int64),
        )
        .map_elements(
            lambda b: sample_to_euro(b.to_list(), rate), return_dtype=pl.Int32
        )
    )


def interview_eve(hhs: pl.DataFrame) -> pl.Expr:
    """The diary day (the day before the interview), if recorded."""
    if "interview_date" in hhs:
        date = pl.col("interview_date").cast(pl.Date)
    elif "interview_day" in hhs:
        date = make_date("interview_year", "interview_month", "interview_day")
    else:
        return pl.lit(None, dtype=pl.Date)
    return date - pl.duration(days=1)


def make_date(year: str, month: str, day: str) -> pl.Expr:
    """Date from numeric columns (2-digit years are 20xx); null if invalid."""
    y = pl.col(year).cast(pl.Int32, strict=False)
    y = pl.when(y < 100).then(y + 2000).otherwise(y)
    return pl.concat_str(
        y,
        pl.col(month).cast(pl.Int32, strict=False),
        pl.col(day).cast(pl.Int32, strict=False),
        separator="-",
    ).str.to_date("%Y-%-m-%-d", strict=False)


# ---------------------------------------------------------------------------
# Persons
# ---------------------------------------------------------------------------


def load_persons(
    persons: pl.DataFrame, config: dict, survey: str
) -> pl.DataFrame:
    """One row per person who answered the travel questions.

    travelled is "yes" or "no"; persons under 3, absent, refusing, or away
    from the study area that day are dropped.
    """

    def mapped(key: str) -> pl.Expr:
        return recode(key, survey_map(config, key, survey))

    if "education_current" in persons:
        education = (
            pl.when(pl.col("education").is_not_null())
            .then(mapped("education"))
            .when(pl.col("education_current").is_not_null())
            .then(mapped("education_current"))
            .otherwise(pl.lit("unknown"))
        )
    else:
        education = mapped("education")

    travelled = recode(
        "travelled", survey_map(config, "travelled", survey), default=None
    )
    if "no_trip_reason" in persons:
        away = survey_map(config, "away_reasons", survey)
        travelled = (
            pl.when(pl.col("no_trip_reason").cast(pl.Int64).is_in(away))
            .then(pl.lit("away"))
            .otherwise(travelled)
        )

    disability = mapped("disability")
    difficulties = [c for c in persons.columns if c.startswith("difficulty_")]
    if difficulties:
        disability = (
            pl.when(pl.any_horizontal(pl.col(c) == 1 for c in difficulties))
            .then(pl.lit("yes"))
            .otherwise(disability)
        )

    age = pl.col("age").cast(pl.Int32)
    persons = (
        persons.with_columns(
            age=pl.when(age <= config["max_age"]).then(age),
            weight=pl.col("weight").cast(pl.Float32),
            sex=mapped("sex"),
            relationship=mapped("relationship"),
            education=education,
            employment=mapped("employment"),
            has_licence=mapped("has_licence"),
            disability=disability,
            travelled=travelled,
        )
        .filter(pl.col("travelled").is_in(["yes", "no"]))
        .filter(~(pl.col("age") < config["min_age"]).fill_null(False))
        .drop(difficulties)
    )

    return persons.with_columns(
        occupation=pl.when(pl.col("employment") == "employed")
        .then(pl.lit("unknown"))
        .otherwise(pl.lit("void")),
        can_wfh=pl.lit("unknown"),
        race=pl.lit("unknown"),
    ).drop(c for c in ("education_current", "no_trip_reason") if c in persons)


def travel_dates(
    persons: pl.DataFrame, trips: pl.DataFrame, hhs: pl.DataFrame
) -> pl.DataFrame:
    """Per pid, the diary date: from the person's trips, else the most common
    trip date in the household, else the day before the interview."""
    trip_dates = trips.group_by("pid").agg(
        pl.col("travel_date").drop_nulls().first()
    )
    hh_dates = (
        trip_dates.join(persons.select("pid", "hid"), on="pid")
        .group_by("hid")
        .agg(hh_date=pl.col("travel_date").drop_nulls().mode().sort().first())
    )
    return (
        persons.select("pid", "hid")
        .join(trip_dates, on="pid", how="left")
        .join(hh_dates, on="hid", how="left")
        .join(hhs.select("hid", "fallback_date"), on="hid", how="left")
        .select(
            "pid",
            travel_date=pl.coalesce("travel_date", "hh_date", "fallback_date"),
        )
    )


def diary_date_columns(survey_year: int) -> list[pl.Expr]:
    """year, month and day (weekday name) from travel_date."""
    date = pl.col("travel_date")
    return [
        pl.coalesce(date.dt.year(), pl.lit(survey_year))
        .cast(pl.Int32)
        .alias("year"),
        date.dt.month().cast(pl.Int8).alias("month"),
        (date.dt.weekday() - 1)
        .replace_strict(
            dict(enumerate(WEEKDAYS)), default="unknown", return_dtype=pl.String
        )
        .fill_null("unknown")
        .alias("day"),
    ]


# ---------------------------------------------------------------------------
# Trips
# ---------------------------------------------------------------------------


def load_trips(
    trips: pl.DataFrame,
    stages: pl.DataFrame,
    config: dict,
    survey: str,
    home_zones: pl.DataFrame,
) -> pl.DataFrame:
    """One row per trip, 0-indexed seq, times in minutes since midnight.

    Unknown times are left null, so filter.null_times drops the plan.
    """
    trips = trips.with_columns(
        pl.col("pid", "tid").cast(pl.Int64),
        tst=clock_minutes(trips, "tst"),
        tet=clock_minutes(trips, "tet"),
        travel_date=travel_date(trips),
    )
    trips = trips.with_columns(diary_times(config["day_start"]))
    if "seq" in trips:
        trips = trips.with_columns(seq=pl.col("seq").cast(pl.Int32) - 1)
    else:  # ENMODO has no trip number
        trips = trips.sort("pid", "tst", "tid", nulls_last=True).with_columns(
            seq=pl.int_range(pl.len()).over("pid")
        )
    trips = trips.with_columns(pl.col("seq").cast(pl.Int8)).sort("pid", "seq")

    zones = survey_map(config, "zone", survey)
    home = config["home_zone_code"]
    trips = trips.join(home_zones, on="pid", how="left").with_columns(
        (
            pl.when(pl.col(end).cast(pl.Int64, strict=False) == home)
            .then(pl.col("hh_zone"))
            .otherwise(recode(end, zones))
            .fill_null("unknown")
            .alias(end)
        )
        for end in ("ozone", "dzone")
    )
    trips = trips.with_columns(activities(survey_map(config, "dact", survey)))

    trips = trips.join(
        trip_modes_and_distances(trips, stages, config, survey),
        on="tid",
        how="left",
    ).with_columns(pl.col("mode").fill_null("unknown"))
    trips = trips.select(
        "pid",
        "seq",
        "ozone",
        "dzone",
        "oact",
        "dact",
        "mode",
        "tst",
        "tet",
        "distance",
        "travel_date",
    )
    return fix.day_wrap(trips)


def clock_minutes(trips: pl.DataFrame, end: str) -> pl.Expr:
    """Minutes since midnight from a clock-time column (already converted by
    read_sav) or from <end>_h and <end>_m columns; null if out of range."""
    if end in trips:
        return pl.col(end).cast(pl.Int32)
    h = pl.col(f"{end}_h").cast(pl.Int32, strict=False)
    m = pl.col(f"{end}_m").cast(pl.Int32, strict=False)
    return pl.when(h.is_between(0, 24) & m.is_between(0, 59)).then(h * 60 + m)


def travel_date(trips: pl.DataFrame) -> pl.Expr:
    if "travel_date" in trips:
        return pl.col("travel_date").cast(pl.Date)
    if "travel_day" in trips:
        return make_date("travel_year", "travel_month", "travel_day")
    return pl.lit(None, dtype=pl.Date)


def diary_times(day_start: int) -> list[pl.Expr]:
    """Shift times before day_start past midnight (the diary runs from 04:00
    to 04:00); tet keeps its clock duration after tst."""
    tst = pl.col("tst")
    shifted = pl.when(tst < day_start).then(tst + 1440).otherwise(tst)
    return [
        shifted.alias("tst"),
        (shifted + (pl.col("tet") - tst) % 1440).alias("tet"),
    ]


def activities(dact_map: dict) -> list[pl.Expr]:
    """oact and dact for trips sorted by pid and seq. The first trip's oact
    comes from the origin activity; later ones repeat the previous dact."""
    dact = recode("dact", dact_map)
    return [
        dact.alias("dact"),
        pl.when(pl.col("seq") == 0)
        .then(recode("oact", dact_map))
        .otherwise(dact.shift(1).over("pid"))
        .alias("oact"),
    ]


def trip_modes_and_distances(
    trips: pl.DataFrame, stages: pl.DataFrame, config: dict, survey: str
) -> pl.DataFrame:
    """Per trip (tid): main mode and distance (km).

    Stage distance is stage time x mode speed. Stage times are capped at the
    trip duration; unknown ones share whatever trip time the known stages
    don't account for. The main mode is the highest-priority mode in
    ``config["mode_priority"]``. Trips without stages get no row.
    """
    missing = config["missing_codes"]
    if "duration" in stages:
        duration = pl.col("duration").cast(pl.Int32, strict=False)
        valid = ~duration.is_in(missing) & (duration >= 0)
    else:
        h = pl.col("duration_h").cast(pl.Int32, strict=False)
        m = pl.col("duration_m").cast(pl.Int32, strict=False)
        duration = h * 60 + m
        valid = ~h.is_in(missing) & (h >= 0) & (m >= 0)
    speeds = survey_map(config, "speed", survey)

    stages = stages.with_columns(
        tid=pl.col("tid").cast(pl.Int64),
        duration=pl.when(valid).then(duration).cast(pl.Float64),
        mode=recode("mode", survey_map(config, "mode", survey)),
    ).with_columns(
        speed=pl.col("mode").replace_strict(
            speeds, default=None, return_dtype=pl.Float64
        )
    )
    trip_duration = trips.select(
        "tid", trip_duration=(pl.col("tet") - pl.col("tst")).cast(pl.Float64)
    )
    stages = (
        stages.join(trip_duration, on="tid", how="left")
        .with_columns(
            duration=pl.when(pl.col("duration").is_not_null()).then(
                pl.min_horizontal("duration", "trip_duration")
            )
        )
        .with_columns(
            unaccounted=(
                pl.col("trip_duration") - pl.col("duration").sum().over("tid")
            ).clip(lower_bound=0),
            n_unknown=pl.col("duration").null_count().over("tid"),
        )
        .with_columns(
            duration=pl.col("duration").fill_null(
                pl.col("unaccounted") / pl.col("n_unknown")
            )
        )
        .with_columns(distance=pl.col("duration") / 60 * pl.col("speed"))
    )

    priority = {mode: rank for rank, mode in enumerate(config["mode_priority"])}
    return stages.group_by("tid").agg(
        mode=pl.col("mode")
        .sort_by(
            pl.col("mode").replace_strict(
                priority, default=len(priority), return_dtype=pl.Int32
            )
        )
        .first(),
        # null when a stage time couldn't be imputed (unknown trip time)
        distance=pl.when(pl.col("distance").is_not_null().all())
        .then(pl.col("distance").sum())
        .cast(pl.Float32),
    )
