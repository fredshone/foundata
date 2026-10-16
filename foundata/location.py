"""Locations: where households (and, later, trip ends) are, and how precisely.

**Declaring a location.** Each source declares how to locate a household in
``configs/<source>/hh_dictionary.yaml`` under ``home_location``: a list of
candidate geographies, finest first. A household takes the first candidate
that applies (``when``) and has a valid code (or point). For example::

    home_location:
      - level: district_{year}        # {year} is the survey year
        years: [2019, 2024]           # years with a lookup
        code:                         # parts joined into one code
          - {column: home_nss_region, slice: [0, 2]}
          - {column: home_district, width: 2}
      - level: nss_region
        code: [home_nss_region]

A candidate has ``level`` and either ``code`` (a list of parts) or ``point``
(``{x, y, crs, scale_m}``), and optionally:

- ``when``: ``{year: [...]}`` and/or ``{survey: [...]}``
- ``prefix``: text before the code, e.g. ``"{survey}/"``
- ``valid``: regex the whole code must match; others are unknown
- ``missing``: codes meaning "unknown"
- ``known``: if true, only codes in the level's lookup count, so a code
  the boundaries don't have falls through to the next candidate
- ``years``: the survey years a ``{year}`` level has lookups for

A code part is a column name or ``{column | value, slice: [start, length],
map: {raw: code} or "<file>.csv", width}``, applied in that order: values
are stripped (and ``"12.0"`` read as ``"12"``), sliced, mapped (unmapped
values are unknown; a csv map has ``from, to`` columns and lives in
``configs/<source>/``) and zero-padded. ``add`` applies a declaration;
``validate_spec`` checks one. Loaders map raw columns to intermediate names
(never the reserved ``<role>_level/_code/_lon/_lat/_scale_m``) and call
``add`` once.

**Level names.** A level is a geography with a fixed code list; a
``_<year>`` suffix gives the vintage of the code list where codes change
between rounds (``gemeente_2012``, ``radio_2010``, ``cbsa_2013``). The
precision class comes from the base name (``LEVEL_PRECISION``).

**From households to locations.** ``extract`` splits the location columns
off a table as *refs*, one per household (or trip end). ``index``
collapses refs at the same place into one row of the locations table, with
a stable ``location_key`` (``<source>/<level>/<code>``, or the point) and a
``location_id``. ``resolve`` gives each location its polygon, a point
inside it, its size and provenance from a lookup: the GeoParquet file
``configs/<source>/locations/<level>.parquet``, else
``configs/geo/<country>/<level>.parquet`` for geographies shared by
sources, built by ``scripts/build_location_lookups.py``. ``write`` saves
the table as GeoParquet.

``scale_m`` is the radius of a circle with the polygon's area (or the
point uncertainty), so it can pick a raster resolution or a sampling
radius. ``geom_source`` says what the geometry is:

- ``point``: the survey's coordinate (``geometry`` is that point)
- ``boundary``: the unit's own polygon
- ``parent``: the unit has no published polygon, so the polygon of a unit
  known to contain it (a French zone's draw zone or commune)
- ``estimated``: work in progress; a provisional area that contains the
  unit, while its own boundary is not available (Mongolian baghs outside
  Ulaanbaatar take their soum, Mexico City EOD districts the
  municipalities they overlap)
- ``none``: no geometry (the code is not in the lookup, or there is no
  lookup for the level)

``geom_dataset`` and ``geom_year`` name the boundary dataset and its
vintage (the survey, for points).
"""

import json
import math
import re
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import shapely
from loguru import logger
from pyproj import Transformer

from foundata import utils

ROLES = ("home", "origin", "destination")
LOCATION_FIELDS = ("level", "code", "lon", "lat", "scale_m")

# precision class of each level's base name (without a _<year> suffix)
LEVEL_PRECISION = {
    "point": "coordinate",
    # neighbourhood-scale zones
    "sa1": "small area",
    "tract": "small area",
    "pc4": "small area",
    "dong": "small area",
    "radio": "small area",
    "fraccion": "small area",
    "segment": "small area",
    "bagh": "small area",
    "detailed_zone": "small area",
    "eod_zone": "small area",
    # municipality / district scale
    "lad": "large area",
    "lga": "large area",
    "sigungu": "large area",
    "gemeente": "large area",
    "cbsa": "large area",
    "msa": "large area",
    "district": "large area",
    "eod_district": "large area",
    "soum": "large area",
    "commune": "large area",
    # state / province / region
    "region": "region",
    "state": "region",
    "census_division": "region",
    "nss_region": "region",
}

REF_SCHEMA = {
    "role": pl.String,
    "source": pl.String,
    "country": pl.String,
    "level": pl.String,
    "code": pl.String,
    "lon": pl.Float64,
    "lat": pl.Float64,
    "scale_m": pl.Float64,
}

# a location is identified by these ref fields
LOCATION_KEYS = ["source", "country", "level", "code", "lon", "lat", "scale_m"]

LOCATION_SCHEMA = {
    "location_id": pl.Int32,
    "location_key": pl.String,  # stable: <source>/<level>/<code or lon,lat>
    "source": pl.String,
    "country": pl.String,
    "level": pl.String,
    "code": pl.String,
    "precision": pl.String,
    "lon": pl.Float64,
    "lat": pl.Float64,
    "scale_m": pl.Float64,
    "geom_source": pl.String,
    "geom_dataset": pl.String,  # boundary dataset (or survey, for points)
    "geom_year": pl.Int16,  # its vintage
    "geometry": pl.Binary,  # WKB, WGS84; null where geom_source is none
}

LOOKUP_COLUMNS = [
    "code",
    "lon",
    "lat",
    "area_km2",
    "geom_source",
    "geom_dataset",
    "geom_year",
    "geometry",
]

CANDIDATE_KEYS = {
    "level",
    "code",
    "point",
    "when",
    "prefix",
    "valid",
    "missing",
    "known",
    "years",
}
PART_KEYS = {"column", "value", "slice", "map", "width"}
POINT_KEYS = {"x", "y", "crs", "scale_m"}


def columns(role: str = "home") -> list[str]:
    """The location columns a loader adds for *role*, e.g. ``home_code``."""
    return [f"{role}_{field}" for field in LOCATION_FIELDS]


# -- declarations -------------------------------------------------------------


def validate_spec(spec: object) -> list[str]:
    """Problems with a ``home_location`` declaration (empty if valid)."""
    if not isinstance(spec, list) or not spec:
        return ["home_location must be a non-empty list of candidates"]
    errors = []
    points = 0
    for i, candidate in enumerate(spec):
        where = f"home_location[{i}]"
        if not isinstance(candidate, dict):
            errors.append(f"{where}: not a mapping")
            continue
        if unknown := set(candidate) - CANDIDATE_KEYS:
            errors.append(f"{where}: unknown keys {sorted(unknown)}")
        level = candidate.get("level")
        if not isinstance(level, str):
            errors.append(f"{where}: needs a level")
            continue
        if base_level(level.replace("{year}", "2000")) not in LEVEL_PRECISION:
            errors.append(f"{where}: no precision class for level {level!r}")
        if "{year}" in level and not candidate.get("years"):
            errors.append(f"{where}: {level!r} needs years")
        if ("code" in candidate) == ("point" in candidate):
            errors.append(f"{where}: needs exactly one of code and point")
        if "point" in candidate:
            points += 1
            point = candidate["point"]
            if level != "point":
                errors.append(f"{where}: a point's level must be 'point'")
            if not isinstance(point, dict) or set(point) != POINT_KEYS:
                errors.append(f"{where}: point needs {sorted(POINT_KEYS)}")
        for j, part in enumerate(candidate.get("code") or []):
            if isinstance(part, str):
                continue
            if not isinstance(part, dict):
                errors.append(f"{where}.code[{j}]: not a column or mapping")
            elif unknown := set(part) - PART_KEYS:
                errors.append(
                    f"{where}.code[{j}]: unknown keys {sorted(unknown)}"
                )
            elif ("column" in part) == ("value" in part):
                errors.append(f"{where}.code[{j}]: needs column or value")
        if unknown := set(candidate.get("when") or {}) - {"year", "survey"}:
            errors.append(f"{where}: unknown when keys {sorted(unknown)}")
    if points > 1:
        errors.append("home_location: at most one point candidate")
    return errors


def base_level(level: str) -> str:
    """*level* without its ``_<year>`` vintage suffix."""
    return re.sub(r"_\d{4}$", "", level)


def precision_of(level: str) -> str | None:
    """Precision class of *level* (from its base name)."""
    return LEVEL_PRECISION.get(base_level(level))


def levels(spec: list[dict]) -> list[str]:
    """Every level a declaration can emit, expanding ``{year}``."""
    out = []
    for candidate in spec:
        level = candidate["level"]
        if "{year}" in level:
            out += [level.format(year=y) for y in candidate["years"]]
        else:
            out.append(level)
    return list(dict.fromkeys(out))


def add(
    df: pl.DataFrame,
    spec: list[dict],
    *,
    source: str,
    year: int | None = None,
    survey: str | None = None,
    role: str = "home",
) -> pl.DataFrame:
    """Add *role*'s location columns (``columns(role)``) to *df* from a
    ``home_location``-style declaration (see the module docstring)."""
    if errors := validate_spec(spec):
        raise ValueError(f"{source}: {'; '.join(errors)}")
    applicable = [c for c in spec if _applies(c, year, survey)]
    level_col, code_col, lon_col, lat_col, scale_col = columns(role)

    chosen = pl.lit(None, pl.Int8)
    values = []
    for i, candidate in enumerate(applicable):
        if "point" in candidate:
            point = candidate["point"]
            x, y = (pl.col(point[k]).cast(pl.Float64) for k in ("x", "y"))
            known = (
                x.is_not_null()
                & y.is_not_null()
                & x.is_finite()
                & y.is_finite()
            )
            values.append({"x": x, "y": y})
        else:
            code = _code(candidate, source, year, survey)
            if candidate.get("known"):
                level = candidate["level"].format(year=year)
                code = pl.when(code.is_in(lookup_codes(source, level))).then(
                    code
                )
            known = code.is_not_null()
            values.append({"code": code})
        chosen = (
            pl.when(chosen.is_null() & known)
            .then(pl.lit(i, pl.Int8))
            .otherwise(chosen)
        )

    def pick(key: str, dtype: pl.DataType) -> pl.Expr:
        out = pl.lit(None, dtype)
        for i, value in enumerate(values):
            if key in value:
                out = (
                    pl.when(pl.col("_chosen") == i)
                    .then(value[key])
                    .otherwise(out)
                )
        return out

    level = pl.lit(None, pl.String)
    scale = pl.lit(None, pl.Float64)
    for i, candidate in enumerate(applicable):
        name = candidate["level"].format(year=year)
        if "{year}" in candidate["level"] and year not in candidate["years"]:
            logger.warning(
                f"{source}: {name} is not in its home_location years "
                f"{candidate['years']}, so it has no lookup"
            )
        level = (
            pl.when(pl.col("_chosen") == i).then(pl.lit(name)).otherwise(level)
        )
        if "point" in candidate:
            scale = (
                pl.when(pl.col("_chosen") == i)
                .then(pl.lit(float(candidate["point"]["scale_m"])))
                .otherwise(scale)
            )

    df = df.with_columns(_chosen=chosen).with_columns(
        level.alias(level_col),
        pick("code", pl.String).alias(code_col),
        pick("x", pl.Float64).alias(lon_col),
        pick("y", pl.Float64).alias(lat_col),
        scale.alias(scale_col),
    )
    points = [c["point"] for c in applicable if "point" in c]
    if points and points[0]["crs"].upper() not in ("EPSG:4326", "OGC:CRS84"):
        lon, lat = to_wgs84(df[lon_col], df[lat_col], points[0]["crs"])
        df = df.with_columns(lon.alias(lon_col), lat.alias(lat_col))
    return df.drop("_chosen")


def add_trip_ends(
    trips: pl.DataFrame,
    config: dict,
    *,
    source: str,
    year: int | None = None,
    survey: str | None = None,
) -> pl.DataFrame:
    """Add the origin and destination location columns to *trips* from the
    ``origin_location`` and ``destination_location`` declarations of a
    (year-resolved) trip_dictionary.yaml, where present."""
    for role in ("origin", "destination"):
        if spec := config.get(f"{role}_location"):
            trips = add(
                trips, spec, source=source, year=year, survey=survey, role=role
            )
    return trips


def _applies(candidate: dict, year: int | None, survey: str | None) -> bool:
    when = candidate.get("when") or {}
    if "year" in when and year not in when["year"]:
        return False
    if "survey" in when and survey not in when["survey"]:
        return False
    return True


def _code(candidate: dict, source: str, year, survey) -> pl.Expr:
    code = pl.concat_str([_part(p, source) for p in candidate["code"]])
    if prefix := candidate.get("prefix"):
        code = pl.lit(prefix.format(survey=survey, year=year)) + code
    if valid := candidate.get("valid"):
        code = pl.when(code.str.contains(f"^(?:{valid})$")).then(code)
    if missing := candidate.get("missing"):
        code = pl.when(~code.is_in([str(m) for m in missing])).then(code)
    return code


def _part(part: str | dict, source: str) -> pl.Expr:
    if isinstance(part, str):
        part = {"column": part}
    if "value" in part:
        return pl.lit(str(part["value"]))
    value = (
        pl.col(part["column"])
        .cast(pl.String)
        .str.strip_chars()
        .str.replace(r"^(-?\d+)\.0*$", "${1}")
    )
    value = pl.when(value != "").then(value)
    if "slice" in part:
        value = value.str.slice(*part["slice"])
    if "map" in part:
        value = value.replace_strict(
            _mapping(part["map"], source), default=None, return_dtype=pl.String
        )
    if "width" in part:
        value = value.str.zfill(part["width"])
    return value


def _mapping(mapping: dict | str, source: str) -> dict[str, str]:
    if isinstance(mapping, str):
        table = pl.read_csv(
            utils.get_config_path(source, mapping), infer_schema_length=0
        )
        return dict(zip(table["from"], table["to"]))
    return {str(k): str(v) for k, v in mapping.items()}


def to_wgs84(
    x: pl.Series, y: pl.Series, crs: str
) -> tuple[pl.Series, pl.Series]:
    """Projected coordinates in *crs* as WGS84 lon, lat (nulls kept)."""
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    # lists, as pyproj treats a 1-element array as a scalar point
    lon, lat = transformer.transform(
        x.cast(pl.Float64).fill_null(float("nan")).to_list(),
        y.cast(pl.Float64).fill_null(float("nan")).to_list(),
    )
    return (
        pl.Series(lon, dtype=pl.Float64).fill_nan(None),
        pl.Series(lat, dtype=pl.Float64).fill_nan(None),
    )


# -- refs and locations -------------------------------------------------------


def extract(
    df: pl.DataFrame,
    source: str,
    role: str = "home",
    keys: tuple[str, ...] = ("hid",),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split *role*'s location columns off *df*.

    Returns *df* without them and one ref per *keys* (``keys, role, source,
    country, level, code, lon, lat, scale_m``) where the level is known.
    Rows sharing *keys* (members of a household) should agree; a zone
    level must have a code.
    """
    names = columns(role)
    present = [c for c in names if c in df.columns]
    schema = {k: df.schema[k] for k in keys} | REF_SCHEMA
    if f"{role}_level" not in present:
        return df.drop(present), pl.DataFrame(schema=schema)
    country = (
        pl.col("country").cast(pl.String)
        if "country" in df.columns
        else pl.lit(None, pl.String)
    )
    refs = (
        df.select(*keys, *present, country=country)
        .with_columns(
            pl.lit(None, pl.Float64).alias(c) for c in names if c not in present
        )
        .select(
            *keys,
            role=pl.lit(role),
            source=pl.lit(source),
            country="country",
            level=pl.col(f"{role}_level").cast(pl.String),
            code=pl.col(f"{role}_code").cast(pl.String),
            lon=pl.col(f"{role}_lon").cast(pl.Float64),
            lat=pl.col(f"{role}_lat").cast(pl.Float64),
            scale_m=pl.col(f"{role}_scale_m").cast(pl.Float64),
        )
        .filter(pl.col("level").is_not_null())
    )
    zone_without_code = refs.filter(
        (pl.col("level") != "point") & pl.col("code").is_null()
    )
    if zone_without_code.height:
        raise ValueError(
            f"{source}: {zone_without_code.height} {role} locations have a "
            "level but no code (location.add leaves both null)"
        )
    distinct = refs.unique(maintain_order=True)
    if n := distinct.height - distinct.select(keys).n_unique():
        logger.warning(
            f"{source}: {n} {'/'.join(keys)} have more than one {role} "
            "location; keeping the first"
        )
    refs = distinct.unique(list(keys), keep="first", maintain_order=True)
    return df.drop(present), refs.cast(schema)


def location_key() -> pl.Expr:
    """``<source>/<level>/<code>``, or ``<source>/point/<lon>,<lat>``."""
    point = pl.format("{},{}", pl.col("lon").round(6), pl.col("lat").round(6))
    return pl.concat_str(
        "source",
        pl.lit("/"),
        "level",
        pl.lit("/"),
        pl.when(pl.col("level") == "point")
        .then(point)
        .otherwise(pl.col("code")),
    )


def index(refs: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Distinct locations of *refs*, and the refs with their ``location_id``.

    Locations are numbered in ``location_key`` order, so an id depends only
    on the set of locations in the run; the key is stable across runs.
    """
    locations = (
        refs.select(LOCATION_KEYS)
        .unique()
        .with_columns(location_key=location_key())
        .sort("location_key", "scale_m", nulls_last=True)
        .with_row_index("location_id")
        .with_columns(pl.col("location_id").cast(pl.Int32))
    )
    refs = refs.join(
        locations.select(*LOCATION_KEYS, "location_id"),
        on=LOCATION_KEYS,
        how="left",
        nulls_equal=True,
    )
    return resolve(locations), refs


def load_lookup(
    source: str, level: str, country: str | None = None
) -> pl.DataFrame | None:
    """The lookup for *level*: ``configs/<source>/locations/<level>.parquet``,
    else ``configs/geo/<country>/<level>.parquet``; None if neither."""
    paths = [utils.get_config_path(source, "locations", f"{level}.parquet")]
    if country:
        paths.append(utils.get_config_path("geo", country, f"{level}.parquet"))
    for path in paths:
        if path.exists():
            return pl.read_parquet(path, columns=LOOKUP_COLUMNS)
    return None


def lookup_codes(source: str, level: str) -> list[str]:
    """Codes in *level*'s lookup: the source's own, else the one shared
    lookup ``configs/geo/*/<level>.parquet``; empty if there is none."""
    paths = [utils.get_config_path(source, "locations", f"{level}.parquet")]
    paths = [p for p in paths if p.exists()] or sorted(
        utils.get_config_path("geo").glob(f"*/{level}.parquet")
    )
    if len(paths) > 1:
        raise ValueError(f"{level} has lookups for several countries: {paths}")
    if not paths:
        logger.warning(f"{source}: no lookup for {level}, so no code is known")
        return []
    return pl.read_parquet(paths[0], columns=["code"])["code"].to_list()


def zone_points(
    source: str, level: str, country: str | None = None, crs: str | None = None
) -> pl.DataFrame:
    """Each unit of *level*'s lookup as ``code, area_km2, lon, lat`` (a point
    inside it), or ``code, area_km2, x, y`` projected to *crs*: zone
    centroids and sizes for trip distances and densities, from the same
    lookups as the locations table."""
    lookup = load_lookup(source, level, country)
    if lookup is None:
        raise FileNotFoundError(f"no location lookup for {source} {level}")
    points = lookup.select("code", "area_km2", "lon", "lat")
    if crs is None:
        return points
    x, y = from_wgs84(points["lon"], points["lat"], crs)
    return points.select("code", "area_km2", x=x, y=y)


def from_wgs84(
    lon: pl.Series, lat: pl.Series, crs: str
) -> tuple[pl.Series, pl.Series]:
    """WGS84 lon, lat as coordinates in *crs*."""
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(lon.to_list(), lat.to_list())
    return pl.Series(x, dtype=pl.Float64), pl.Series(y, dtype=pl.Float64)


def equivalent_radius_m(area_km2: pl.Expr) -> pl.Expr:
    """Radius (m) of the circle with the given area (km²)."""
    return (area_km2 / math.pi).sqrt() * 1000


def resolve(locations: pl.DataFrame) -> pl.DataFrame:
    """Fill geometry, centroid, scale, precision and provenance for each
    location."""
    parts = []
    for (source, country, level), group in locations.group_by(
        "source", "country", "level", maintain_order=True
    ):
        if level == "point":
            group = group.with_columns(
                geometry=point_wkb(group["lon"], group["lat"]),
                geom_source=pl.lit("point"),
                geom_dataset=pl.lit(f"{source} survey"),
                geom_year=pl.lit(None, pl.Int16),
            )
        elif (lookup := load_lookup(source, level, country)) is not None:
            group = group.drop("lon", "lat", "scale_m").join(
                lookup.select(
                    "code",
                    "lon",
                    "lat",
                    "geometry",
                    "geom_dataset",
                    "geom_year",
                    scale_m=equivalent_radius_m(pl.col("area_km2")),
                    geom_source=pl.col("geom_source").fill_null("boundary"),
                ),
                on="code",
                how="left",
            )
        else:
            logger.warning(f"No location lookup for {source} {level}")
            group = group.with_columns(
                *(
                    pl.lit(None, pl.Float64).alias(c)
                    for c in ("lon", "lat", "scale_m")
                ),
                geometry=pl.lit(None, pl.Binary),
            )
        parts.append(group)
    if not parts:
        return pl.DataFrame(schema=LOCATION_SCHEMA)
    precision = {
        level: precision_of(level) for level in locations["level"].unique()
    }
    if unknown := {level for level, p in precision.items() if p is None}:
        logger.warning(f"No precision class for levels {unknown}")
    return (
        pl.concat(parts, how="diagonal_relaxed")
        .with_columns(
            precision=pl.col("level").replace_strict(
                precision, default=None, return_dtype=pl.String
            ),
            geom_source=pl.when(pl.col("geometry").is_null())
            .then(pl.lit("none"))
            .otherwise(pl.col("geom_source")),
        )
        .select(LOCATION_SCHEMA.keys())
        .cast(LOCATION_SCHEMA)
        .sort("location_id")
    )


def point_wkb(lon: pl.Series, lat: pl.Series) -> pl.Series:
    """WKB points (null where lon or lat is null)."""
    known = (lon.is_not_null() & lat.is_not_null()).to_numpy()
    points = shapely.points(
        lon.cast(pl.Float64).fill_null(0).to_numpy(),
        lat.cast(pl.Float64).fill_null(0).to_numpy(),
    )
    wkb = shapely.to_wkb(points)
    wkb[~known] = None
    return pl.Series(wkb.tolist(), dtype=pl.Binary)


GEOMETRY_TYPES = {
    0: "Point",
    3: "Polygon",
    6: "MultiPolygon",
    7: "GeometryCollection",
}


def write(locations: pl.DataFrame, path: Path) -> None:
    """Write the locations table as GeoParquet (WKB geometry, WGS84)."""
    table = locations.to_arrow()
    geometry = shapely.from_wkb(locations["geometry"].drop_nulls().to_numpy())
    types = sorted(set(shapely.get_type_id(geometry).tolist()))
    geo = {
        "version": "1.0.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": [
                    GEOMETRY_TYPES.get(t, str(t)) for t in types
                ],
            }
        },
    }
    metadata = {**(table.schema.metadata or {}), b"geo": json.dumps(geo)}
    pq.write_table(
        table.replace_schema_metadata(metadata), path, compression="zstd"
    )


def coverage(locations: pl.DataFrame, refs: pl.DataFrame) -> pl.DataFrame:
    """Per source, role and level: number of refs, and the shares with a
    geometry and with an exact one (a point or the unit's own boundary)."""
    return (
        refs.join(
            locations.select("location_id", "geom_source"),
            on="location_id",
            how="left",
        )
        .group_by("source", "role", "level")
        .agg(
            n=pl.len(),
            with_geometry=(pl.col("geom_source") != "none").mean(),
            exact=pl.col("geom_source").is_in(["point", "boundary"]).mean(),
        )
        .sort("source", "role", "level")
    )


def summary(
    attributes: pl.DataFrame, trips: pl.DataFrame, locations: pl.DataFrame
) -> pl.DataFrame:
    """Per source: home levels and geometry datasets, the shares of
    households with a geometry and with an exact one (a point or the unit's
    own boundary), the median home scale_m, and the share of trip ends with
    a geometry. The README location table is this table."""
    info = locations.select(
        "location_id", "level", "geom_source", "geom_dataset", "scale_m"
    )
    homes = (
        attributes.select("hid", "source", "hh_location")
        .unique("hid")
        .join(info, left_on="hh_location", right_on="location_id", how="left")
    )
    geom_source = pl.col("geom_source").fill_null("none")
    located = geom_source != "none"
    exact = geom_source.is_in(["point", "boundary"])
    by_source = homes.group_by("source").agg(
        households=pl.len(),
        levels=pl.col("level").drop_nulls().unique().sort().str.join(", "),
        geometry=pl.col("geom_dataset")
        .drop_nulls()
        .unique()
        .sort()
        .str.join("; "),
        coverage=located.mean(),
        exact=exact.mean(),
        median_scale_m=pl.col("scale_m").median(),
    )
    sources = attributes.select("pid", "source").unique("pid")
    ends = (
        pl.concat(
            [
                trips.select("pid", id=pl.col(c))
                for c in ("olocation", "dlocation")
            ]
        )
        .join(sources, on="pid", how="left")
        .join(info, left_on="id", right_on="location_id", how="left")
    )
    trip_ends = ends.group_by("source").agg(
        trip_end_coverage=(located & pl.col("id").is_not_null()).mean()
    )
    return by_source.join(trip_ends, on="source", how="left").sort("source")


def summary_markdown(table: pl.DataFrame) -> str:
    """The summary table as a markdown table, for the README."""

    def km(m: float | None) -> str:
        if m is None:
            return ""
        return f"{m / 1000:.1f} km" if m >= 1000 else f"{m:.0f} m"

    def pct(x: float | None) -> str:
        return "" if x is None else f"{100 * x:.1f}%".replace(".0%", "%")

    rows = [
        "| Source | Home levels | Coverage | Exact | Median `scale_m` | Trip ends with geometry |",
        "|---|---|---|---|---|---|",
    ]
    for r in table.iter_rows(named=True):
        rows.append(
            f"| {r['source']} | {r['levels']} | {pct(r['coverage'])} | "
            f"{pct(r['exact'])} | {km(r['median_scale_m'])} | "
            f"{pct(r['trip_end_coverage']) or '—'} |"
        )
    return "\n".join(rows)
