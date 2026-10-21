"""Locations: where households and trip ends are, and how precisely.

**Configs list levels and their lookups.** A source lists, finest first,
the levels it can locate a household at, each with its lookup file in
``configs/<source>/geo/``, under ``home_location`` in hh_dictionary.yaml
(and trip ends under ``trip_location`` in trip_dictionary.yaml), keyed by
survey year (or name), with ``default`` for the rest::

    home_location:
      2023:
        - level: pc4
          geo: pc4.parquet
        - level: gemeente
          geo: gemeente_2023.parquet

A point level has no file but an assumed uncertainty:
``- {level: point, scale_m: 50}``.

**Loaders build codes.** Before ``add``, a loader writes one clean code
column per level, ``<role>_<level>`` (``home_pc4``; ``origin_gemeente``
and ``destination_gemeente`` for trips), null where unknown; ``clean``,
``mapped`` and ``wgs84`` help. A point level reads ``<role>_lon`` and
``<role>_lat`` in WGS84.

**A location is a key.** ``add`` writes, per row, the key of the first
level whose code is in its lookup, ``<source>/<geo>/<code>`` (or
``<source>/point/<lon>,<lat>``), to ``hh_location`` (or ``ocode`` and
``dcode`` for trips), and drops the code columns. A code its lookup lacks
falls through to the next level; the last level keeps it, so the location
has no geometry. A source's ``LocationFinder`` reads and validates its
declarations and loads each lookup once; ``add(df, source=, key=)`` and
``add_trip_ends`` use it.

**Keys become ids.** After a source is loaded, ``LocationFinder.index``
builds its locations table, one row per distinct key with its geometry
from the lookup, and replaces the keys by ``location_id``s (numbered from
0 in key order; the runner offsets them per source), and releases the
lookups. ``write`` saves the combined table as GeoParquet.

``scale_m`` is the radius of a circle with the polygon's area (or the
point uncertainty). ``geom_source`` says what the geometry is:

- ``point``: the survey's coordinate
- ``boundary``: the unit's own polygon
- ``parent``: the polygon of a unit known to contain it (a French zone's
  draw zone or commune), where the unit has none
- ``estimated``: work in progress, a provisional area containing the unit
  (Mongolian baghs outside Ulaanbaatar, Mexico City EOD districts)
- ``none``: no geometry (the code is not in its lookup)
"""

import functools
import json
import math
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import shapely
from loguru import logger
from pyproj import Transformer

from foundata import utils

# the template column each role's location key (then id) goes to
COLUMNS = {"home": "hh_location", "origin": "ocode", "destination": "dcode"}

# precision class of each level
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
    "od_zone": "small area",
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

LOCATION_SCHEMA = {
    "location_id": pl.Int32,
    "location_key": pl.String,  # <source>/<geo>/<code>, or the point
    "source": pl.String,
    "country": pl.String,
    "level": pl.String,
    "code": pl.String,
    "geo": pl.String,  # the lookup the code is from
    "precision": pl.String,
    "lon": pl.Float64,
    "lat": pl.Float64,
    "scale_m": pl.Float64,
    "geom_source": pl.String,
    "geom_dataset": pl.String,  # boundary dataset (or survey, for points)
    "geom_year": pl.Int16,  # its vintage
    "geometry": pl.Binary,  # WKB, WGS84; null where geom_source is none
}


LOOKUP_SCHEMA = {  # configs/<source>/geo/<geo>.parquet
    "code": pl.String,
    "lon": pl.Float64,
    "lat": pl.Float64,
    "area_km2": pl.Float64,
    "geom_source": pl.String,
    "geom_dataset": pl.String,
    "geom_year": pl.Int16,
    "geometry": pl.Binary,
}
LOOKUP_COLUMNS = list(LOOKUP_SCHEMA)

ENTRY_KEYS = {"level", "geo", "scale_m"}


def precision_of(level: str) -> str | None:
    """Precision class of *level*."""
    return LEVEL_PRECISION.get(level)


# -- declarations -------------------------------------------------------------


def for_key(declaration: list | dict | None, key=None) -> list[dict]:
    """The levels a declaration gives for *key* (a survey year or name): its
    block for *key*, else its ``default`` block; a plain list applies to
    every key. Empty if neither."""
    if declaration is None:
        return []
    if isinstance(declaration, list):
        return declaration
    # keys compare as strings: YAML may give 2019 or "2019"
    by_key = {str(k): v for k, v in declaration.items()}
    return by_key.get(str(key), by_key.get("default")) or []


def blocks(declaration: list | dict | None) -> dict:
    """Every block of a declaration, by key (``None`` for a plain list)."""
    if declaration is None:
        return {}
    if isinstance(declaration, list):
        return {None: declaration}
    return dict(declaration)


def geo_stem(entry: dict) -> str | None:
    """The lookup a level reads (its file name without ``.parquet``)."""
    return entry["geo"].removesuffix(".parquet") if "geo" in entry else None


def geo_files(declaration: list | dict | None) -> list[str]:
    """Every lookup a declaration reads, as file stems."""
    stems = [
        geo_stem(entry)
        for entries in blocks(declaration).values()
        for entry in entries or []
    ]
    return list(dict.fromkeys(s for s in stems if s))


def validate(declaration: object, key: str = "home_location") -> list[str]:
    """Problems with a ``home_location`` or ``trip_location`` declaration
    (empty if valid)."""
    if not isinstance(declaration, (list, dict)) or not declaration:
        return [f"{key} must be a list of levels, or blocks of them by survey"]
    errors = []
    for block, entries in blocks(declaration).items():
        where = key if block is None else f"{key}[{block}]"
        if entries is None:
            continue  # no locations for that survey
        if not isinstance(entries, list) or not entries:
            errors.append(f"{where}: must be a non-empty list of levels")
            continue
        levels = []
        for i, entry in enumerate(entries):
            at = f"{where}[{i}]"
            if not isinstance(entry, dict):
                errors.append(f"{at}: not a mapping")
                continue
            if unknown := set(entry) - ENTRY_KEYS:
                errors.append(f"{at}: unknown keys {sorted(unknown)}")
            level = entry.get("level")
            levels.append(level)
            if level not in LEVEL_PRECISION:
                errors.append(f"{at}: no precision class for level {level!r}")
            if level == "point":
                if "geo" in entry or "scale_m" not in entry:
                    errors.append(f"{at}: a point has scale_m and no geo")
            elif not str(entry.get("geo", "")).endswith(".parquet"):
                errors.append(f"{at}: needs geo: <lookup>.parquet")
            elif "scale_m" in entry:
                errors.append(f"{at}: only a point has scale_m")
        if len(levels) != len(set(levels)):
            errors.append(f"{where}: a level is listed twice")
    return errors


# -- finding locations --------------------------------------------------------


class LocationFinder:
    """Finds a source's locations: the key of each household's (or trip
    end's) first known level, then the table of those locations with their
    geometries. Each lookup is loaded once, and released by ``index``.

    *home* and *trip* are the source's ``home_location`` and
    ``trip_location`` declarations, read from its configs if neither is
    given, and validated here.
    """

    def __init__(
        self,
        source: str,
        home: list | dict | None = None,
        trip: list | dict | None = None,
    ):
        self.source = source
        if home is None and trip is None:
            home, trip = (
                utils.load_yaml_config(path).get(key)
                if (path := utils.get_config_path(source, name)).exists()
                else None
                for name, key in (
                    ("hh_dictionary.yaml", "home_location"),
                    ("trip_dictionary.yaml", "trip_location"),
                )
            )
        self.declarations = {"home": home, "trip": trip}
        self.levels: dict[str, str] = {}  # lookup -> level
        self.scale_m: float | None = None  # of the point level
        for name, declaration in self.declarations.items():
            if declaration is None:
                continue
            key = f"{name}_location"
            if errors := validate(declaration, key):
                raise ValueError(f"{source}: {'; '.join(errors)}")
            for entries in blocks(declaration).values():
                for entry in entries or []:
                    if entry["level"] == "point":
                        self.scale_m = float(entry["scale_m"])
                    else:
                        self.levels[geo_stem(entry)] = entry["level"]
        self._lookups: dict[str, pl.DataFrame | None] = {}

    def lookup(self, geo: str) -> pl.DataFrame | None:
        """The lookup *geo* (configs/<source>/geo/<geo>.parquet), loaded
        once; None if there is none."""
        if geo not in self._lookups:
            self._lookups[geo] = load_lookup(self.source, geo)
            if self._lookups[geo] is None:
                logger.warning(f"{self.source}: no lookup {geo}")
        return self._lookups[geo]

    def add(
        self, df: pl.DataFrame, *, key=None, role: str = "home"
    ) -> pl.DataFrame:
        """Write *role*'s location key to its column (``COLUMNS``), from the
        levels its declaration gives for *key* (a survey year or name), and
        drop the code columns read (see the module docstring)."""
        declaration = self.declarations["home" if role == "home" else "trip"]
        entries = for_key(declaration, key)
        inputs = [
            f"{role}_{e['level']}" for e in entries if e["level"] != "point"
        ]
        if any(e["level"] == "point" for e in entries):
            inputs += [f"{role}_lon", f"{role}_lat"]
        if missing := [c for c in inputs if c not in df.columns]:
            raise ValueError(
                f"{self.source}: {role} location needs columns {missing}"
            )
        keys = []
        for i, entry in enumerate(entries):
            if entry["level"] == "point":
                lon, lat = pl.col(f"{role}_lon"), pl.col(f"{role}_lat")
                known = lon.is_finite() & lat.is_finite()
                # + 0.0 turns -0.0 into 0.0, so a point has one key
                value = pl.format(
                    f"{self.source}/point/{{}},{{}}",
                    lon.round(6) + 0.0,
                    lat.round(6) + 0.0,
                )
            else:
                geo = geo_stem(entry)
                code = pl.col(f"{role}_{entry['level']}").cast(pl.String)
                known = code.is_not_null()
                if i < len(entries) - 1:
                    # a code its lookup doesn't have falls through
                    lookup = self.lookup(geo)
                    codes = [] if lookup is None else lookup["code"]
                    known = known & code.is_in(
                        pl.Series(codes, dtype=pl.String).implode()
                    )
                value = pl.lit(f"{self.source}/{geo}/") + code
            keys.append(pl.when(known).then(value))
        location = pl.coalesce(keys) if keys else pl.lit(None, pl.String)
        return df.with_columns(location.alias(COLUMNS[role])).drop(inputs)

    def add_trip_ends(self, trips: pl.DataFrame, *, key=None) -> pl.DataFrame:
        """``ocode`` and ``dcode`` from ``trip_location``, read from
        ``origin_<level>`` and ``destination_<level>`` code columns, if the
        source has one."""
        if self.declarations["trip"] is not None:
            for role in ("origin", "destination"):
                trips = self.add(trips, key=key, role=role)
        return trips

    def locations(
        self, keys: pl.Series, country: str | None = None
    ) -> pl.DataFrame:
        """The locations table of *keys* (distinct and sorted): numbered
        from 0, with each key's level, code, lookup, point, size, geometry
        and provenance."""
        refs = (
            pl.DataFrame({"location_key": keys})
            .with_row_index("location_id")
            .with_columns(
                parts=pl.col("location_key").str.splitn("/", 3),
                source=pl.lit(self.source),
                country=pl.lit(country, pl.String),
            )
            .with_columns(
                geo=pl.col("parts").struct.field("field_1"),
                code=pl.col("parts").struct.field("field_2"),
            )
            .drop("parts")
        )
        parts = []
        points = refs.filter(pl.col("geo") == "point")
        if points.height:
            lonlat = points["code"].str.split_exact(",", 1)
            lon = lonlat.struct.field("field_0").cast(pl.Float64)
            lat = lonlat.struct.field("field_1").cast(pl.Float64)
            parts.append(
                points.with_columns(
                    level=pl.lit("point"),
                    geo=pl.lit(None, pl.String),
                    code=pl.lit(None, pl.String),
                    lon=lon,
                    lat=lat,
                    scale_m=pl.lit(self.scale_m, pl.Float64),
                    geometry=point_wkb(lon, lat),
                    geom_source=pl.lit("point"),
                    geom_dataset=pl.lit(f"{self.source} survey"),
                    geom_year=pl.lit(None, pl.Int16),
                )
            )
        zones = refs.filter(pl.col("geo") != "point")
        for (geo,), group in zones.group_by("geo", maintain_order=True):
            lookup = self.lookup(geo)
            if lookup is None:
                lookup = pl.DataFrame(schema=LOOKUP_SCHEMA)
            parts.append(
                group.join(
                    lookup.select(
                        pl.exclude("area_km2"),
                        scale_m=equivalent_radius_m(pl.col("area_km2")),
                    ),
                    on="code",
                    how="left",
                ).with_columns(level=pl.lit(self.levels.get(geo)))
            )
        if not parts:
            return pl.DataFrame(schema=LOCATION_SCHEMA)
        return (
            pl.concat(parts, how="diagonal_relaxed")
            .with_columns(
                precision=pl.col("level").replace_strict(
                    LEVEL_PRECISION, default=None, return_dtype=pl.String
                ),
                geom_source=pl.when(pl.col("geometry").is_null())
                .then(pl.lit("none"))
                .otherwise(pl.col("geom_source")),
            )
            .select(LOCATION_SCHEMA.keys())
            .cast(LOCATION_SCHEMA)
            .sort("location_id")
        )

    def index(
        self, attributes: pl.DataFrame, trips: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """The source's locations table, and *attributes* and *trips* with
        their location keys replaced by its ``location_id``s. Releases the
        lookups."""
        attributes_keys = [
            c for c in ["hh_location"] if c in attributes.columns
        ]
        trips_keys = [c for c in ["ocode", "dcode"] if c in trips.columns]
        keys = (
            pl.concat(
                [attributes[c].cast(pl.String) for c in attributes_keys]
                + [trips[c].cast(pl.String) for c in trips_keys]
                + [pl.Series([], dtype=pl.String)]
            )
            .drop_nulls()
            .unique()
            .sort()
        )
        country = attributes["country"].drop_nulls().unique()
        locations = self.locations(
            keys, country[0] if len(country) == 1 else None
        )
        self._lookups.clear()
        # an Enum of the sorted keys numbers them as location_id does
        ids = pl.Enum(keys)

        def to_ids(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
            return df.with_columns(
                pl.col(c).cast(pl.String).cast(ids).to_physical().cast(pl.Int32)
                for c in columns
            )

        return (
            to_ids(attributes, attributes_keys),
            to_ids(trips, trips_keys),
            locations,
        )


@functools.cache
def finder(source: str) -> LocationFinder:
    """The source's LocationFinder (one per source and run)."""
    return LocationFinder(source)


def add(
    df: pl.DataFrame, *, source: str, key=None, role: str = "home"
) -> pl.DataFrame:
    """``LocationFinder.add`` with the source's finder."""
    return finder(source).add(df, key=key, role=role)


def add_trip_ends(
    trips: pl.DataFrame, *, source: str, key=None
) -> pl.DataFrame:
    """``LocationFinder.add_trip_ends`` with the source's finder."""
    return finder(source).add_trip_ends(trips, key=key)


# -- helpers for loaders ------------------------------------------------------


def clean(
    expr: pl.Expr | str,
    *,
    width: int | None = None,
    valid: str | None = None,
    missing: list | tuple = (),
) -> pl.Expr:
    """A raw code as a clean string: stripped, ``"12.0"`` read as ``"12"``,
    blank as null, zero-padded to *width*; null where it is in *missing* or
    doesn't fully match the regex *valid*."""
    if isinstance(expr, str):
        expr = pl.col(expr)
    code = (
        expr.cast(pl.String)
        .str.strip_chars()
        .str.replace(r"^(-?\d+)\.0*$", "${1}")
    )
    code = pl.when(code != "").then(code)
    if width:
        code = code.str.zfill(width)
    if missing:
        code = pl.when(~code.is_in([str(m) for m in missing])).then(code)
    if valid:
        code = pl.when(code.str.contains(f"^(?:{valid})$")).then(code)
    return code


def mapping(source: str, name: str) -> dict[str, str]:
    """A ``from, to`` csv in configs/<source>/ as a dict."""
    table = pl.read_csv(
        utils.get_config_path(source, name), infer_schema_length=0
    )
    return dict(zip(table["from"], table["to"]))


def mapped(expr: pl.Expr | str, source: str, name: str) -> pl.Expr:
    """A raw code mapped to a standard code by the ``from, to`` csv *name*
    in configs/<source>/ (null where unmapped)."""
    return clean(expr).replace_strict(
        mapping(source, name), default=None, return_dtype=pl.String
    )


def wgs84(
    df: pl.DataFrame, x: str, y: str, crs: str, role: str = "home"
) -> pl.DataFrame:
    """Add ``<role>_lon`` and ``<role>_lat``: the coordinates *x*, *y* (in
    *crs*) in WGS84, for a point level."""
    xs, ys = df[x].cast(pl.Float64), df[y].cast(pl.Float64)
    if crs.upper() not in ("EPSG:4326", "OGC:CRS84"):
        xs, ys = to_wgs84(xs, ys, crs)
    return df.with_columns(xs.alias(f"{role}_lon"), ys.alias(f"{role}_lat"))


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


# -- lookups and geometry ----------------------------------------------------


def lookup_path(source: str, geo: str) -> Path:
    """Where a lookup lives: ``configs/<source>/geo/<geo>.parquet``."""
    return utils.get_config_path(source, "geo", f"{geo}.parquet")


def load_lookup(source: str, geo: str) -> pl.DataFrame | None:
    """The source's lookup *geo* (a file stem); None if there is none."""
    path = lookup_path(source, geo)
    if not path.exists():
        return None
    return pl.read_parquet(path, columns=LOOKUP_COLUMNS)


def zone_points(source: str, geo: str, crs: str | None = None) -> pl.DataFrame:
    """Each unit of the lookup *geo* as ``code, area_km2, lon, lat`` (a
    point inside it), or ``code, area_km2, x, y`` projected to *crs*: zone
    centroids and sizes for trip distances and densities, from the source
    finder's copy of the lookup."""
    lookup = finder(source).lookup(geo)
    if lookup is None:
        raise FileNotFoundError(f"no location lookup {source} {geo}")
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


def write(
    locations: pl.DataFrame, path: Path, metadata: dict | None = None
) -> None:
    """Write the locations table as GeoParquet (WKB geometry, WGS84), with
    any extra *metadata* (key -> JSON-able value) in its schema."""
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
    extra = {k: json.dumps(v) for k, v in (metadata or {}).items()}
    metadata = {
        **(table.schema.metadata or {}),
        **extra,
        b"geo": json.dumps(geo),
    }
    pq.write_table(
        table.replace_schema_metadata(metadata), path, compression="zstd"
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
            [trips.select("pid", id=pl.col(c)) for c in ("ocode", "dcode")]
        )
        .join(sources, on="pid", how="left")
        .join(info, left_on="id", right_on="location_id", how="left")
    )
    # null for sources without trip-end locations
    trip_ends = ends.group_by("source").agg(
        trip_end_coverage=pl.when(pl.col("id").is_not_null().any()).then(
            (located & pl.col("id").is_not_null()).mean()
        )
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
