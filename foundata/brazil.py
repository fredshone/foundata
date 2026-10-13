"""São Paulo Pesquisa Origem Destino (OD) 2017, Metrô SP.

The raw data is a single flat dBase table, ``OD_2017_v1.dbf``: one row per
trip, with household (``ID_DOM``), family (``ID_FAM``) and person
(``ID_PESS``) columns repeated on every row. Persons who made no trips have a
single row with empty trip columns. ``F_DOM``/``F_FAM``/``F_PESS`` flag the
first row of each household/family/person. Codes are documented in
``LAYOUT OD2017_v1.xlsx``.
"""

import struct
from pathlib import Path

import numpy as np
import polars as pl

from foundata import fix, utils

SOURCE = "brazil"
COUNTRY = "br"

OD_PATH = Path("OD-2017") / "Banco de Dados-OD2017" / "OD_2017_v1.dbf"
ZONES_PATH = (
    Path("OD-2017") / "Mapas-OD2017" / "Shape-OD2017" / "Zonas_2017_region.dbf"
)


def read_dbf(
    path: str | Path, columns: list[str] | None = None
) -> pl.DataFrame:
    """Read a dBase III table into string columns.

    Blank fields and fields filled with asterisks (how the OD files encode
    missing numeric values) are read as null. Records are fixed width, so the
    whole file is sliced with numpy rather than parsed row by row. Callers
    cast columns as needed.
    """
    with open(Path(path).expanduser(), "rb") as f:
        n_records, header_len, record_len = struct.unpack(
            "<IHH", f.read(12)[4:12]
        )
        f.seek(32)
        fields = []
        offset = 1  # first byte of each record is the deletion flag
        while (descriptor := f.read(32))[0] != 0x0D:
            name = descriptor[:11].split(b"\0")[0].decode("ascii")
            length = descriptor[16]
            fields.append((name, offset, length))
            offset += length
        f.seek(header_len)
        records = np.frombuffer(
            f.read(n_records * record_len), dtype=np.uint8
        ).reshape(n_records, record_len)

    if columns is not None:
        missing = set(columns) - {name for name, _, _ in fields}
        if missing:
            raise KeyError(f"Columns not found in {path}: {sorted(missing)}")
        fields = [field for field in fields if field[0] in columns]

    data = {}
    for name, start, length in fields:
        raw = np.ascontiguousarray(records[:, start : start + length])
        values = np.char.decode(raw.view(f"S{length}").ravel(), "latin1")
        data[name] = pl.Series(name, values).str.strip_chars()
    return pl.DataFrame(data).with_columns(
        pl.when(pl.col(name).str.contains(r"^\**$"))
        .then(None)
        .otherwise(pl.col(name))
        .alias(name)
        for name in data
    )


def load(
    data_root: str | Path,
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load and normalise São Paulo OD 2017 survey data.

    Args:
        data_root: Path to the raw data directory for this source.
        hh_config: Parsed hh_dictionary.yaml config.
        person_config: Parsed person_dictionary.yaml config.
        trips_config: Parsed trip_dictionary.yaml config.

    Returns:
        (attributes, trips) DataFrames conforming to the template schema.
    """
    print(f"Loading {SOURCE} data...")
    root = Path(data_root).expanduser()
    columns = {"F_DOM", "F_FAM", "F_PESS"}
    for config in (hh_config, person_config, trips_config):
        columns |= set(config["column_mappings"])
    raw = read_dbf(root / OD_PATH, columns=sorted(columns))

    zone_types = load_zone_types(
        root / ZONES_PATH, raw, hh_config["density_thresholds"]
    )

    hhs = load_households(raw, hh_config, zone_types)
    persons = load_persons(raw, person_config)
    attributes = utils.table_joiner(hhs, persons, on="hid")
    attributes = travel_date(attributes, hh_config["day"])

    trips = load_trips(raw, trips_config, zone_types)
    attributes = attributes.join(
        access_egress(trips), on="pid", how="left"
    ).with_columns(source=pl.lit(SOURCE), country=pl.lit(COUNTRY))
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
    )

    attributes = attributes.with_columns(
        pid=pl.lit(SOURCE) + pl.col("pid"), hid=pl.lit(SOURCE) + pl.col("hid")
    )
    trips = trips.with_columns(pid=pl.lit(SOURCE) + pl.col("pid"))

    return attributes, trips


def load_zone_types(
    path: str | Path, raw: pl.DataFrame, thresholds: dict
) -> dict[int, str]:
    """Classify OD zones as urban/suburban/rural by residential density.

    Density is the weighted (FE_PESS) number of residents per km² of zone
    area. Zones with no sampled residents are left unmapped (-> unknown).
    """
    zones = read_dbf(path, columns=["NumeroZona", "Area_ha_2"]).select(
        zone=pl.col("NumeroZona").cast(pl.Int32),
        area_km2=pl.col("Area_ha_2").cast(pl.Float64) / 100,
    )
    residents = (
        raw.filter(pl.col("F_PESS") == "1")
        .group_by(zone=pl.col("ZONA").cast(pl.Int32))
        .agg(population=pl.col("FE_PESS").cast(pl.Float64).sum())
    )
    zones = zones.join(residents, on="zone", how="inner").with_columns(
        hh_zone=pl.when(
            pl.col("population") / pl.col("area_km2") >= thresholds["urban"]
        )
        .then(pl.lit("urban"))
        .when(
            pl.col("population") / pl.col("area_km2") >= thresholds["suburban"]
        )
        .then(pl.lit("suburban"))
        .otherwise(pl.lit("rural"))
    )
    return dict(zip(zones["zone"], zones["hh_zone"]))


def load_households(
    raw: pl.DataFrame, config: dict, zone_types: dict[int, str]
) -> pl.DataFrame:
    """One row per household, with family-level fields aggregated.

    Income and vehicles are summed over the household's families, and are
    null if any family's value is unknown (for income, if its income code is
    not in ``config["income_codes"]``). Tenure is taken from the first family.
    """
    column_mapping = config["column_mappings"]
    families = (
        raw.filter(pl.col("F_FAM") == "1")
        .select(column_mapping.keys())
        .rename(column_mapping)
        .with_columns(
            pl.col(
                "home_zone",
                "hh_size",
                "family",
                "income_code",
                "cars",
                "motorcycles",
                "ownership",
            ).cast(pl.Int32),
            income=pl.col("income").cast(pl.Float64),
        )
        .with_columns(
            income=pl.when(pl.col("income_code").is_in(config["income_codes"]))
            .then(pl.col("income"))
            .otherwise(None)
        )
    )

    hhs = (
        families.sort("hid", "family")
        .group_by("hid", maintain_order=True)
        .agg(
            pl.col("home_zone", "hh_size", "date", "ownership").first(),
            monthly_income=pl.when(pl.col("income").is_not_null().all())
            .then(pl.col("income").sum())
            .otherwise(None),
            # null when any family did not declare its assets
            vehicles=pl.when(
                pl.col("cars").is_not_null().all()
                & pl.col("motorcycles").is_not_null().all()
            )
            .then((pl.col("cars") + pl.col("motorcycles")).sum())
            .otherwise(None),
        )
    )

    return hhs.with_columns(
        hh_income=(pl.col("monthly_income") * 12 * config["brl_to_euro"]).cast(
            pl.Int32
        ),
        vehicles=pl.col("vehicles").cast(pl.Int32),
        ownership=pl.col("ownership")
        .replace_strict(config["ownership"], default="unknown")
        .fill_null("unknown"),
        hh_zone=pl.col("home_zone")
        .replace_strict(zone_types, default="unknown")
        .fill_null("unknown"),
        date=pl.col("date").str.to_date("%d%m%Y"),
        dwelling=pl.lit("unknown"),
    ).drop("home_zone", "monthly_income")


def load_persons(raw: pl.DataFrame, config: dict) -> pl.DataFrame:
    """One row per person."""
    column_mapping = config["column_mappings"]
    persons = (
        raw.filter(pl.col("F_PESS") == "1")
        .select(column_mapping.keys())
        .rename(column_mapping)
        .with_columns(
            pl.col(
                "age",
                "sex",
                "relationship",
                "education",
                "employment",
                "occupation",
                "day",
            ).cast(pl.Int32),
            weight=pl.col("weight").cast(pl.Float32),
        )
    )

    persons = persons.with_columns(
        sex=pl.col("sex").replace_strict(config["sex"], default="unknown"),
        relationship=pl.col("relationship").replace_strict(
            config["relationship"], default="unknown"
        ),
        education=pl.col("education").replace_strict(
            config["education"], default="unknown"
        ),
        employment=pl.col("employment").replace_strict(
            config["employment"], default="unknown"
        ),
        occupation=pl.col("occupation").replace_strict(
            config["occupation"], default="unknown"
        ),
    ).with_columns(
        occupation=pl.when(pl.col("employment") == "employed")
        .then(pl.col("occupation"))
        .otherwise(pl.lit("void"))
    )

    return persons.with_columns(
        pl.lit("unknown").alias(col)
        for col in ["has_licence", "disability", "can_wfh", "race"]
    )


def travel_date(attributes: pl.DataFrame, day_mapping: dict) -> pl.DataFrame:
    """Derive year/month/day of the travel day from the interview date.

    Travellers report their travel weekday (DIA_SEM, 2=Monday .. 6=Friday),
    which is the last such weekday before the interview. Non-travellers have
    no DIA_SEM, so the previous weekday before the interview is assumed.
    """
    interview_wd = pl.col("date").dt.weekday()  # Monday=1 .. Sunday=7
    reported = (interview_wd - (pl.col("day") - 1) + 6) % 7 + 1
    previous_weekday = (
        pl.when(interview_wd == 1)
        .then(3)
        .when(interview_wd == 7)
        .then(2)
        .otherwise(1)
    )
    offset = pl.coalesce(reported, previous_weekday)
    attributes = attributes.with_columns(
        date=pl.col("date") - pl.duration(days=offset)
    )
    return attributes.with_columns(
        year=pl.col("date").dt.year().cast(pl.Int32),
        month=pl.col("date").dt.month().cast(pl.Int8),
        day=pl.col("date")
        .dt.weekday()
        .replace_strict(
            {k - 1: v for k, v in day_mapping.items()}, default="unknown"
        ),
    ).drop("date")


def load_trips(
    raw: pl.DataFrame, config: dict, zone_types: dict[int, str]
) -> pl.DataFrame:
    """One row per trip, 0-indexed seq, times in minutes since midnight."""
    column_mapping = config["column_mappings"]
    trips = (
        raw.filter(pl.col("N_VIAG").is_not_null())
        .select(column_mapping.keys())
        .rename(column_mapping)
    )
    int_cols = [c for c in trips.columns if c not in ("pid", "distance")]
    trips = trips.with_columns(
        pl.col(int_cols).cast(pl.Int32),
        distance=pl.col("distance").cast(pl.Float64),
    )

    act_mapping = config["dact"]
    trips = trips.with_columns(
        seq=(pl.col("seq") - 1).cast(pl.Int8),
        tst=pl.col("tst_h") * 60 + pl.col("tst_m"),
        tet=pl.col("tet_h") * 60 + pl.col("tet_m"),
        oact=pl.col("oact").replace_strict(act_mapping, default="unknown"),
        dact=pl.col("dact").replace_strict(act_mapping, default="unknown"),
        mode=pl.col("mode").replace_strict(config["mode"], default="unknown"),
        ozone=pl.col("ozone").replace_strict(zone_types, default="unknown"),
        dzone=pl.col("dzone").replace_strict(zone_types, default="unknown"),
        distance=(pl.col("distance") / 1000 * config["circuity"]).cast(
            pl.Float32
        ),
    )

    # Serving a passenger (dropping off / picking up) is recorded as a flag
    # alongside the passenger's purpose, e.g. MOTIVO_D=education, SERVIR_D=1
    trips = trips.with_columns(
        oact=pl.when((pl.col("oescort") == 1) & (pl.col("oact") != "home"))
        .then(pl.lit("escort"))
        .otherwise(pl.col("oact")),
        dact=pl.when((pl.col("descort") == 1) & (pl.col("dact") != "home"))
        .then(pl.lit("escort"))
        .otherwise(pl.col("dact")),
        access_egress_distance=pl.when(pl.col("mode").is_in(["bus", "rail"]))
        .then(
            (pl.col("walk_o").fill_null(0) + pl.col("walk_d").fill_null(0))
            / 60
            * config["walk_speed"]
        )
        .otherwise(None),
    )

    trips = trips.sort("pid", "seq")
    return fix.day_wrap(trips)


def access_egress(trips: pl.DataFrame) -> pl.DataFrame:
    """Mean per-person walking access/egress distance (km) over transit trips."""
    return (
        trips.filter(pl.col("access_egress_distance").is_not_null())
        .group_by("pid")
        .agg(pl.col("access_egress_distance").mean().cast(pl.Float32))
    )
