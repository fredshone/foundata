"""Build the location lookups that foundata.location resolves codes with.

A lookup is a GeoParquet file with one row per unit of a level: ``code,
lon, lat, area_km2, geom_source, geom_dataset, geom_year`` and a
simplified polygon, written to ``configs/<source>/geo/<level>.parquet``
by the source's builder (national code lists and survey zones alike). Each
file's parquet metadata (key ``foundata``) records how it was built:
the datasets (URL, sha256, retrieval date), the method, alias files used,
and codes that could not be matched or were matched fuzzily.

Which lookups to build comes from each source's ``home_location`` and
``trip_location`` declarations (its hh_ and trip_dictionary.yaml): the
builder for a source writes every ``geo`` file they name
(``location.geo_files``), so a new survey year means adding its block
there and rebuilding.

Boundary files are downloaded on first use to
``<data-root>/boundaries/<COUNTRY>/`` and recorded in
``configs/location_datasets.lock.csv`` (URL, file, sha256, bytes, date); a
file whose hash no longer matches the lock is reported. Some lookups also
read survey files in the data root (France's zone polygons and households,
KTDB's 2010 classification table, Mexico's district catalogue), so they can
only be rebuilt with the data.

Per unit, ``lon``/``lat`` is a point inside the simplified polygon and
``area_km2`` the area of the full polygon in an equal-area projection.
Where a unit has no published boundary, ``geom_source`` says what was used:

- parent: the boundary of a unit containing it (French detailed zones
  without a polygon take their draw zone or commune)
- estimated: work in progress, a provisional area containing the unit
  (Mongolian baghs outside Ulaanbaatar take their soum, and 2019
  Ulaanbaatar khoroos today's khoroo of that number; Mexico City EOD
  districts the municipalities they overlap)

Usage:
    uv run --group scripts python scripts/build_location_lookups.py
    uv run --group scripts python scripts/build_location_lookups.py --only india --only nhts
"""

import csv
import datetime as dt
import difflib
import hashlib
import json
import re
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import shapely
import typer
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from foundata import location  # noqa: E402
from foundata.france import survey_year  # noqa: E402

CONFIGS = Path(__file__).parent.parent / "configs"
LOCK = CONFIGS / "location_datasets.lock.csv"
DEFAULT_DATA_ROOT = Path.home() / "Data/foundata"
EQUAL_AREA_CRS = "EPSG:6933"


# -- datasets -----------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    """A downloadable boundary file. ``name`` and ``year`` are what lookup
    rows report as geom_dataset and geom_year (year: the boundaries'
    vintage, None where the source doesn't say)."""

    url: str
    file: str  # under <data-root>/boundaries/
    name: str
    year: int | None


ONS = "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services"
ONS_QUERY = "FeatureServer/0/query?where=1%3D1&outFields=*&f=geojson&outSR=4326"
CENSUS = "https://www2.census.gov/geo/tiger"
ABS = (
    "https://www.abs.gov.au/statistics/standards/"
    "australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/"
    "access-and-downloads/digital-boundary-files"
)
CBS = "https://www.cbs.nl/-/media"
CBS_MAPS = {
    2005: f"{CBS}/imported/documents/2010/13/2005buurtkaartdata2v2.zip",
    2006: f"{CBS}/imported/documents/2010/13/buurt_2006_gn2.zip",
    2007: f"{CBS}/imported/documents/2009/01/buurt_2007_gen(2).zip",
    2008: "http://download.cbs.nl/regionale-kaarten/2008-buurtkaart-gn-3.zip",
    2009: "http://download.cbs.nl/regionale-kaarten/2009-buurtkaart-gn-3.zip",
    2010: "http://download.cbs.nl/regionale-kaarten/2010-buurtkaart-shape-versie-3.zip",
    2011: "http://download.cbs.nl/regionale-kaarten/2011-buurtkaart-shape-versie-3.zip",
    2012: f"{CBS}/cbs/dossiers/nederland-regionaal/wijk-en-buurtstatistieken/wijkbuurtkaart_2012_v3.zip",
    2013: "http://download.cbs.nl/regionale-kaarten/shape-2013-versie-3-0.zip",
    2014: f"{CBS}/_pdf/2016/35/shape-2014-versie-30.zip",
    2015: f"{CBS}/_pdf/2017/36/buurt_2015.zip",
    2016: f"{CBS}/cbs/dossiers/nederland-regionaal/wijk-en-buurtstatistieken/shape-2016-versie-30.zip",
    2017: f"{CBS}/cbs/dossiers/nederland-regionaal/wijk-en-buurtstatistieken/wijkbuurtkaart_2017_202201.zip",
}  # fmt: skip
PDOK_GEMEENTEN = (
    "https://api.pdok.nl/cbs/gebiedsindelingen/ogc/v1/collections/"
    "gemeente_gegeneraliseerd/items?f=json&limit=1000&jaarcode={year}"
)
ADMDONG = (
    "https://raw.githubusercontent.com/vuski/admdongkor/master/{version}/{name}"
)
KOSTAT_2011 = (
    "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/"
    "kostat/2011/shp/skorea-submunicipalities-2011.{ext}"
)
INDIA = "https://github.com/ramSeraph/indian_admin_boundaries/releases/download"
ETALAB_COMMUNES = (
    "https://etalab-datasets.geo.data.gouv.fr/contours-administratifs/"
    "{year}/geojson/communes-50m.geojson.gz"
)

DATASETS = {
    "ons_regions_2023": Dataset(
        f"{ONS}/Regions_December_2023_Boundaries_EN_BGC/{ONS_QUERY}",
        "UK/Regions_December_2023_Boundaries_EN_BGC.geojson",
        "ONS Regions (December 2023) BGC",
        2023,
    ),
    "ons_countries_2023": Dataset(
        f"{ONS}/Countries_December_2023_Boundaries_UK_BGC/{ONS_QUERY}",
        "UK/Countries_December_2023_Boundaries_UK_BGC.geojson",
        "ONS Countries (December 2023) BGC",
        2023,
    ),
    "ons_lad_2023": Dataset(
        f"{ONS}/Local_Authority_Districts_December_2023_Boundaries_UK_BGC/{ONS_QUERY}",
        "UK/Local_Authority_Districts_December_2023_Boundaries_UK_BGC.geojson",
        "ONS Local Authority Districts (December 2023) BGC",
        2023,
    ),
    "census_cbsa_2013": Dataset(
        f"{CENSUS}/GENZ2013/cb_2013_us_cbsa_500k.zip",
        "USA/cb_2013_us_cbsa_500k.zip",
        "Census cartographic boundary CBSA 2013 (1:500k)",
        2013,
    ),
    "census_cbsa_2018": Dataset(
        f"{CENSUS}/GENZ2018/shp/cb_2018_us_cbsa_500k.zip",
        "USA/cb_2018_us_cbsa_500k.zip",
        "Census cartographic boundary CBSA 2018 (1:500k)",
        2018,
    ),
    "census_cbsa_2009": Dataset(
        f"{CENSUS}/TIGER2009/tl_2009_us_cbsa.zip",
        "USA/tl_2009_us_cbsa.zip",
        "Census TIGER/Line CBSA 2009",
        2009,
    ),
    "census_msa_1999": Dataset(
        f"{CENSUS}/PREVGENZ/ma/ma99/ma99shp/ma99_99_shp.zip",
        "USA/ma99_99_shp.zip",
        "Census cartographic boundary MSA/CMSA 1999",
        1999,
    ),
    "census_cmsa_1999": Dataset(
        f"{CENSUS}/PREVGENZ/ma/ma99/cmsa_99_shp.zip",
        "USA/cmsa_99_shp.zip",
        "Census cartographic boundary CMSA 1999",
        1999,
    ),
    "census_state_2018": Dataset(
        f"{CENSUS}/GENZ2018/shp/cb_2018_us_state_500k.zip",
        "USA/cb_2018_us_state_500k.zip",
        "Census cartographic boundary states 2018 (1:500k)",
        2018,
    ),
    "census_division_2018": Dataset(
        f"{CENSUS}/GENZ2018/shp/cb_2018_us_division_500k.zip",
        "USA/cb_2018_us_division_500k.zip",
        "Census cartographic boundary divisions 2018 (1:500k)",
        2018,
    ),
    **{
        f"census_tract_2010_{state}": Dataset(
            f"{CENSUS}/GENZ2010/gz_2010_{fips}_140_00_500k.zip",
            f"USA/gz_2010_{fips}_140_00_500k.zip",
            f"Census cartographic boundary tracts 2010, {name} (1:500k)",
            2010,
        )
        for state, fips, name in (
            ("il", "17", "Illinois"),
            ("in", "18", "Indiana"),
            ("wi", "55", "Wisconsin"),
        )
    },
    "abs_sa1_2021": Dataset(
        f"{ABS}/SA1_2021_AUST_SHP_GDA2020.zip",
        "AUSTRALIA/SA1_2021_AUST_SHP_GDA2020.zip",
        "ABS ASGS Edition 3 SA1 2021",
        2021,
    ),
    "abs_lga_2021": Dataset(
        f"{ABS}/LGA_2021_AUST_GDA2020_SHP.zip",
        "AUSTRALIA/LGA_2021_AUST_GDA2020_SHP.zip",
        "ABS ASGS Edition 3 LGA 2021",
        2021,
    ),
    **{
        f"cbs_wijkbuurtkaart_{year}": Dataset(
            url,
            f"NETHERLANDS/wijkbuurtkaart/{year}.zip",
            f"CBS Wijk- en buurtkaart {year} (gemeenten)",
            year,
        )
        for year, url in CBS_MAPS.items()
    },
    **{
        f"pdok_gemeenten_{year}": Dataset(
            PDOK_GEMEENTEN.format(year=year),
            f"NETHERLANDS/pdok_gemeenten_{year}.json",
            f"PDOK CBS gebiedsindelingen gemeente {year} (gegeneraliseerd)",
            year,
        )
        for year in range(2016, 2025)
    },
    **{
        f"cbs_pc4_{year}": Dataset(
            f"https://download.cbs.nl/postcode/{release}-cbs_pc4_{year}_vol.zip",
            f"NETHERLANDS/{release}-cbs_pc4_{year}_vol.zip",
            f"CBS kerncijfers per postcode PC4 {year}",
            year,
        )
        for year, release in ((2023, 2026), (2019, 2026))
    },
    **{
        f"admdong_{version}": Dataset(
            ADMDONG.format(version=f"ver{version}", name=name),
            f"KOREA/HangJeongDong_ver{version}.geojson",
            f"행정동 boundaries ver{version} (vuski/admdongkor)",
            int(version[:4]),
        )
        for version, name in (
            ("20161231", "HangJeongDong_ver20161231.geojson"),
            ("20210701", "HangJeongDong_ver20210701.geojson"),
            ("20211001", "HangJeongDong_ver_20211001.geojson"),
            ("20220101", "HangJeongDong_ver20220101.geojson"),
        )
    },
    **{
        f"kostat_2011_{ext}": Dataset(
            KOSTAT_2011.format(ext=ext),
            f"KOREA/kostat2011/skorea-submunicipalities-2011.{ext}",
            "KOSTAT 읍면동 2011 (southkorea/southkorea-maps)",
            2011,
        )
        for ext in ("shp", "shx", "dbf", "prj", "cpg")
    },
    **{
        f"conicet_radios_{year}": Dataset(
            "https://datosdeinvestigacion.conicet.gov.ar/bitstream/handle/"
            f"11336/149711/RADIOS_{year}_V2025-1.zip?sequence={seq}&isAllowed=y",
            f"ARGENTINA/RADIOS_{year}_V2025-1.zip",
            f"INDEC census radios {year}, corrected (CONICET v2025-1)",
            year,
        )
        for year, seq in ((2010, 14), (2001, 8))
    },
    "ine_segments_2011": Dataset(
        "https://mapas.ide.uy/geoserver-vectorial/INE_NO_SEGURO/wfs?service=WFS"
        "&request=GetFeature&version=1.0.0&outputFormat=shape-zip"
        "&typeName=ine_seg_11",
        "URUGUAY/ine_seg_11.zip",
        "INE census segments 2011 (ide.uy)",
        2011,
    ),
    "lgd_districts": Dataset(
        f"{INDIA}/districts/LGD_Districts.parquet",
        "INDIA/LGD_Districts.parquet",
        "LGD districts (ramSeraph/indian_admin_boundaries)",
        None,
    ),
    "census_districts_2011": Dataset(
        f"{INDIA}/census-2011/Districts_2011.parquet",
        "INDIA/Districts_2011.parquet",
        "Census of India districts 2011 (ramSeraph/indian_admin_boundaries)",
        2011,
    ),
    "india_district_splits": Dataset(
        f"{INDIA}/historical/District_splits_and_carveouts_1951_to_2024.csv",
        "INDIA/District_splits_and_carveouts_1951_to_2024.csv",
        "District splits and carve-outs 1951-2024 (ramSeraph)",
        None,
    ),
    "hdx_mongolia_cod": Dataset(
        "https://data.humdata.org/dataset/a9b0a8a6-cb14-448e-b35c-aa5eb51b0557/"
        "resource/de6b11d3-767d-449f-99fc-80c5b4e9abe3/download/"
        "mng_admin_boundaries.geojson.zip",
        "MONGOLIA/mng_admin_boundaries.geojson.zip",
        "Mongolia COD-AB soums (HDX)",
        2020,
    ),
    "ub_khoroos": Dataset(
        "https://raw.githubusercontent.com/Tuvshin-Level/khoroo-map/"
        "bb1bf18f27e57d7a5aa7fe6d4a17234b40e162c1/khoroos.json",
        "MONGOLIA/khoroos.json",
        "Ulaanbaatar khoroos (Tuvshin-Level/khoroo-map)",
        None,
    ),
    "nso_units": Dataset(
        "https://data.1212.mn/api/v1/en/NSO/Population,%20household/"
        "1_Population,%20household/DT_NSO_0300_002V4.px",
        "MONGOLIA/nso_regions_DT_NSO_0300_002V4.json",
        "NSO Mongolia bag and khoroo codes (DT_NSO_0300_002V4)",
        None,
    ),
    "inegi_mg_2020": Dataset(
        "https://www.inegi.org.mx/contenidos/productos/prod_serv/contenidos/"
        "espanol/bvinegi/productos/geografia/marcogeo/889463807469/"
        "mg_2020_integrado.zip",
        "MEXICO/mg_2020_integrado.zip",
        "INEGI Marco Geoestadístico 2020 municipalities",
        2020,
    ),
    **{
        f"etalab_communes_{year}": Dataset(
            ETALAB_COMMUNES.format(year=year),
            f"FRANCE/communes-50m-{year}.geojson.gz",
            f"Etalab contours administratifs communes {year}",
            year,
        )
        for year in (2024, 2019, 2018)
    },
    "osm_communes_2015": Dataset(
        "http://osm13.openstreetmap.fr/~cquest/openfla/export/"
        "communes-20150101-5m-shp.zip",
        "FRANCE/communes-20150101-5m-shp.zip",
        "OpenStreetMap communes 2015-01-01",
        2015,
    ),
}


class Builder:
    """Downloads, the lock file, and the provenance of lookups being built."""

    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.boundaries = data_root / "boundaries"
        self.lock = read_lock()
        self.used: list[str] = []  # dataset ids used for the current lookup
        self.source = ""  # the source being built; lookups go to its geo/

    def fetch(self, dataset_id: str) -> Path:
        """Path of a dataset's file, downloading it if missing, and checking
        it against (or adding it to) the lock file."""
        dataset = DATASETS[dataset_id]
        path = self.boundaries / dataset.file
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {dataset.url}")
            request = urllib.request.Request(
                dataset.url, headers={"User-Agent": "Mozilla/5.0 (foundata)"}
            )
            tmp = path.with_name(path.name + ".part")
            with urllib.request.urlopen(request) as response:
                tmp.write_bytes(response.read())
            tmp.rename(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        entry = self.lock.get(dataset_id)
        if entry is None or entry["url"] != dataset.url:
            self.lock[dataset_id] = {
                "id": dataset_id,
                "url": dataset.url,
                "file": dataset.file,
                "sha256": sha,
                "bytes": path.stat().st_size,
                "retrieved": dt.date.fromtimestamp(
                    path.stat().st_mtime
                ).isoformat(),
            }
            write_lock(self.lock)
        elif entry["sha256"] != sha:
            print(
                f"  WARNING {dataset_id}: {path} differs from the lock "
                f"(sha256 {entry['sha256'][:12]}..., now {sha[:12]}...)"
            )
        if dataset_id not in self.used:
            self.used.append(dataset_id)
        return path

    def polygons(
        self, gdf: gpd.GeoDataFrame, code: pd.Series, dataset_id: str | None
    ) -> gpd.GeoDataFrame:
        """Each unit's polygon (repeated codes dissolved) with a point
        inside it, its area and its dataset."""
        keep = code.notna().to_numpy()
        gdf = gpd.GeoDataFrame(
            {"code": code[keep].astype(str).values},
            geometry=gdf.geometry.values[keep],
            crs=gdf.crs,
        )
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        gdf = gdf.set_geometry(gdf.geometry.make_valid()).dissolve("code")
        out = located(gdf.reset_index())
        dataset = DATASETS.get(dataset_id) if dataset_id else None
        out["geom_dataset"] = dataset.name if dataset else None
        out["geom_year"] = dataset.year if dataset else None
        return out

    def write(
        self,
        df: pd.DataFrame,
        level: str,
        *,
        method: str,
        aliases: list[str] = (),
        unmatched: list[str] = (),
        fuzzy: dict[str, str] | None = None,
        extra_datasets: dict[str, str] | None = None,
    ) -> None:
        """Write a lookup to configs/<source>/geo/<level>.parquet for the
        source being built, with its provenance in the parquet metadata."""
        out = CONFIGS / self.source / "geo" / f"{level}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        for column in ("geom_source", "geom_dataset", "geom_year"):
            if column not in gdf.columns:
                gdf[column] = None
        gdf = (
            gdf.assign(code=gdf.code.astype(str))
            .drop_duplicates("code", keep="first")
            .sort_values("code")
            .reset_index(drop=True)
        )
        gdf = gdf.assign(
            area_km2=gdf.area_km2.astype(float).round(6),
            geom_source=gdf.geom_source.fillna("boundary").astype("string"),
            geom_dataset=gdf.geom_dataset.astype("string"),
            geom_year=gdf.geom_year.astype("Int16"),
        )
        # area from the full polygon; lon/lat inside the simplified one
        gdf = gdf.set_geometry(simplified(gdf))
        points = gdf.geometry.representative_point()
        gdf = gdf.assign(lon=points.x, lat=points.y)
        gdf = gdf[location.LOOKUP_COLUMNS]
        gdf.to_parquet(out, compression="zstd", index=False)

        meta = {
            "level": level,
            "source": self.source,
            "built": dt.date.today().isoformat(),
            "method": method,
            "datasets": [
                {
                    "id": i,
                    "name": DATASETS[i].name,
                    "year": DATASETS[i].year,
                    "url": DATASETS[i].url,
                    "sha256": self.lock[i]["sha256"],
                }
                for i in self.used
            ]
            + [{"id": k, "name": v} for k, v in (extra_datasets or {}).items()],
            "aliases": list(aliases),
            "unmatched": sorted(unmatched),
            "fuzzy_matches": fuzzy or {},
            "simplification": "5% of equivalent radius, 5 m to 1 km",
        }
        table = pq.read_table(out)
        table = table.replace_schema_metadata(
            {
                **table.schema.metadata,
                b"foundata": json.dumps(meta, ensure_ascii=False),
            }
        )
        pq.write_table(table, out, compression="zstd")
        self.used = []
        counts = gdf.geom_source.value_counts().to_dict()
        print(
            f"Wrote {len(gdf)} units {counts} to {out.relative_to(CONFIGS)} "
            f"({out.stat().st_size / 1e6:.1f} MB)"
            + (f"; {len(unmatched)} unmatched" if unmatched else "")
            + (f"; {len(fuzzy)} fuzzy matches" if fuzzy else "")
        )


def read_lock() -> dict[str, dict]:
    if not LOCK.exists():
        return {}
    with LOCK.open() as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def write_lock(lock: dict[str, dict]) -> None:
    with LOCK.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "url", "file", "sha256", "bytes", "retrieved"]
        )
        writer.writeheader()
        writer.writerows(lock[k] for k in sorted(lock))


def located(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """*gdf* (code, geometry) with lon, lat and area_km2, in WGS84."""
    points = gdf.geometry.representative_point().to_crs("EPSG:4326")
    return gpd.GeoDataFrame(
        {
            "code": gdf.code.values,
            "lon": points.x.values,
            "lat": points.y.values,
            "area_km2": gdf.geometry.to_crs(EQUAL_AREA_CRS).area.values / 1e6,
        },
        geometry=gdf.geometry.to_crs("EPSG:4326").values,
        crs="EPSG:4326",
    )


def simplified(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """Polygons simplified to 5% of each unit's equivalent radius (5 m to
    1 km), with coordinates rounded to ~1 m.

    Units are simplified one by one, so neighbours can gap or overlap by up
    to the tolerance along shared borders.
    """
    projected = gdf.geometry.to_crs(EQUAL_AREA_CRS)
    radius_m = np.sqrt(gdf.area_km2.to_numpy() / np.pi) * 1000
    tolerance = np.clip(0.05 * radius_m, 5, 1000)
    geometry = shapely.simplify(
        projected.values, tolerance, preserve_topology=True
    )
    geometry = gpd.GeoSeries(geometry, crs=EQUAL_AREA_CRS).to_crs("EPSG:4326")
    return gpd.GeoSeries(
        shapely.set_precision(geometry.values, 1e-5),
        index=gdf.index,
        crs="EPSG:4326",
    )


def first_of(*lookups: pd.DataFrame) -> pd.DataFrame:
    """Rows of the lookups, keeping the first for each code."""
    non_empty = [lookup for lookup in lookups if len(lookup)] or [lookups[0]]
    return pd.concat(non_empty).drop_duplicates("code", keep="first")


def source_lookups(source: str) -> list[str]:
    """The lookups (file stems) a source's location declarations read:
    home_location (hh_dictionary.yaml) and trip_location
    (trip_dictionary.yaml)."""
    stems = []
    for dictionary, key in (("hh", "home_location"), ("trip", "trip_location")):
        path = CONFIGS / source / f"{dictionary}_dictionary.yaml"
        config = yaml.safe_load(path.read_text()) if path.exists() else {}
        stems += location.geo_files(config.get(key))
    return list(dict.fromkeys(stems))


def unbuilt(source: str, built: set[str]) -> None:
    """Report lookups a source declares that its builder did not write."""
    if missing := [g for g in source_lookups(source) if g not in built]:
        print(f"  WARNING {source}: no builder for lookups {missing}")


# -- builders, one per source ---------------------------------------------------


def nts(b: Builder) -> set[str]:
    regions = gpd.read_file(b.fetch("ons_regions_2023"))
    countries = gpd.read_file(b.fetch("ons_countries_2023"))
    countries = countries[countries.CTRY23CD.str[0].isin(["W", "S"])]
    b.write(
        pd.concat(
            [
                b.polygons(regions, regions.RGN23CD, "ons_regions_2023"),
                b.polygons(countries, countries.CTRY23CD, "ons_countries_2023"),
            ]
        ),
        "region",
        method="ONS English regions, plus the Wales and Scotland country polygons",
    )
    return {"region"}


def ltds(b: Builder) -> set[str]:
    lads = gpd.read_file(b.fetch("ons_lad_2023"))
    b.write(
        b.polygons(lads, lads.LAD23CD, "ons_lad_2023"),
        "lad_2023",
        method="ONS local authority districts (December 2023) by LAD23CD; "
        "London boroughs are the E09 codes",
    )
    return {"lad_2023"}


def nhts(b: Builder) -> set[str]:
    def census(dataset_id: str) -> gpd.GeoDataFrame:
        return gpd.read_file(b.fetch(dataset_id))

    cbsa_2013 = census("census_cbsa_2013")
    b.write(
        b.polygons(cbsa_2013, cbsa_2013.CBSAFP, "census_cbsa_2013"),
        "cbsa_2013",
        method="Census 2013 CBSAs (February 2013 delineations), as NHTS 2017 codes them",
    )
    cbsa_2009 = census("census_cbsa_2009")
    b.write(
        b.polygons(cbsa_2009, cbsa_2009.CBSAFP, "census_cbsa_2009"),
        "cbsa_2003",
        method="TIGER 2009 CBSAs (2003-based delineations), as NHTS 2009 codes them",
    )
    # the 1999 files have no .prj; they are NAD83 longitude/latitude
    msa = census("census_msa_1999").set_crs("EPSG:4269")
    cmsa = census("census_cmsa_1999").set_crs("EPSG:4269")
    aliases = CONFIGS / "nhts" / "geo" / "msa_1999.aliases.yaml"
    nhts_code = {
        c: n for n, c in yaml.safe_load(aliases.read_text())["cmsa"].items()
    }
    unmatched = sorted(set(cmsa.CMSA) - set(nhts_code))
    b.write(
        pd.concat(
            [
                b.polygons(msa, msa.MSACMSA, "census_msa_1999"),
                b.polygons(cmsa, cmsa.CMSA.map(nhts_code), "census_cmsa_1999"),
            ]
        ),
        "msa_1999",
        method="1999 MSAs by MSACMSA code; CMSAs by the 4-digit code NHTS gives them",
        aliases=[str(aliases.relative_to(CONFIGS))],
        unmatched=[f"CMSA {c}" for c in unmatched],
    )
    states = census("census_state_2018")
    b.write(
        b.polygons(states, states.STATEFP, "census_state_2018"),
        "state",
        method="Census 2018 states by FIPS code",
    )
    divisions = census("census_division_2018")
    b.write(
        b.polygons(
            divisions, divisions.DIVISIONCE.str.zfill(2), "census_division_2018"
        ),
        "census_division",
        method="Census divisions by 2-digit code",
    )
    return {"cbsa_2013", "cbsa_2003", "msa_1999", "state", "census_division"}


def cmap(b: Builder) -> set[str]:
    # the CMAP region and its neighbours, for trip ends
    tracts = []
    for state in ("il", "in", "wi"):
        dataset_id = f"census_tract_2010_{state}"
        gdf = gpd.read_file(b.fetch(dataset_id))
        tracts.append(
            b.polygons(gdf, gdf.STATE + gdf.COUNTY + gdf.TRACT, dataset_id)
        )
    b.write(
        pd.concat(tracts),
        "tract_2010",
        method="Census 2010 tracts of Illinois, Indiana and Wisconsin by "
        "11-digit FIPS",
    )
    return {"tract_2010"}


def qhts(b: Builder) -> set[str]:
    sa1 = gpd.read_file(b.fetch("abs_sa1_2021"))
    sa1 = sa1[sa1.STE_CODE21 == "3"]
    b.write(
        b.polygons(sa1, sa1.SA1_CODE21, "abs_sa1_2021"),
        "sa1_2021",
        method="ABS 2021 SA1s in Queensland",
    )
    return {"sa1_2021"}


def vista(b: Builder) -> set[str]:
    lga = gpd.read_file(b.fetch("abs_lga_2021"))
    lga = lga[lga.STE_CODE21 == "2"]
    b.write(
        b.polygons(lga, lga.LGA_CODE21, "abs_lga_2021"),
        "lga_2021",
        method="ABS 2021 LGAs in Victoria by LGA_CODE21 (VISTA's names are "
        "mapped to codes in configs/vista/lga_2021_codes.csv)",
    )
    return {"lga_2021"}


def gemeenten(b: Builder, year: int) -> None:
    """configs/<source>/geo/gemeente_<year>: that year's CBS map (2005-17), PDOK
    gemeenten (2016 on), or the nearest year's CBS map before 2005."""
    if year >= 2018:
        dataset_id = f"pdok_gemeenten_{year}"
        gdf = gpd.read_file(b.fetch(dataset_id))
        code = gdf.statcode.str[2:]
        method = f"PDOK CBS gemeenten {year}"
    else:
        nearest = min(CBS_MAPS, key=lambda y: (abs(y - year), y))
        dataset_id = f"cbs_wijkbuurtkaart_{nearest}"
        archive = b.fetch(dataset_id)
        with zipfile.ZipFile(archive) as zf:
            layer = next(
                n
                for n in zf.namelist()
                if re.search(r"(^|/)(gem|gemeente)_\d{4}[^/]*\.shp$", n)
            )
        gdf = gpd.read_file(f"zip://{archive}!{layer}")
        if "WATER" in gdf.columns:
            gdf = gdf[gdf.WATER != "JA"]
        code = gdf.GM_CODE.str[2:]
        method = f"CBS wijk- en buurtkaart {nearest} gemeenten (land)"
    b.write(
        b.polygons(gdf, code, dataset_id),
        f"gemeente_{year}",
        method=method + "; 4-digit CBS codes",
    )


def nl_source(source: str):
    def build(b: Builder) -> set[str]:
        built = set()
        for level in source_lookups(source):
            if level.startswith("gemeente_"):
                gemeenten(b, int(level.removeprefix("gemeente_")))
                built.add(level)
        if source == "odin":
            pc4 = []
            for year in (2023, 2019):
                archive = b.fetch(f"cbs_pc4_{year}")
                gdf = gpd.read_file(f"zip://{archive}!cbs_pc4_{year}_vol.gpkg")
                pc4.append(
                    b.polygons(
                        gdf,
                        gdf.postcode.astype(int).astype(str).str.zfill(4),
                        f"cbs_pc4_{year}",
                    )
                )
            b.write(
                first_of(*pc4),
                "pc4",
                method="CBS PC4 2023, then 2019 for postcodes since retired",
            )
            built.add("pc4")
        return built

    return build


def ktdb(b: Builder) -> set[str]:
    dong_2021 = [
        (gpd.read_file(b.fetch(f"admdong_{v}")), f"admdong_{v}")
        for v in ("20210701", "20211001", "20220101")
    ]
    b.write(
        first_of(*(b.polygons(g, g.adm_cd2, i) for g, i in dong_2021)),
        "dong_2021",
        method="행정동 of July 2021 by 10-digit code, then October 2021 and "
        "January 2022 for dongs created during the survey",
    )
    dong_2016 = gpd.read_file(b.fetch("admdong_20161231"))
    b.write(
        b.polygons(
            dong_2016, dong_2016.adm_cd, "admdong_20161231"
        ),  # 10-digit here
        "dong_2016",
        method="행정동 of December 2016 by 10-digit code",
    )
    for ext in ("shp", "shx", "dbf", "prj", "cpg"):
        shp = b.fetch(f"kostat_2011_{ext}")
    # its .prj is Korean 1985 / Modified Central Belt without the datum shift
    kostat = gpd.read_file(shp.with_suffix(".shp")).set_crs(
        "EPSG:5174", allow_override=True
    )
    b.write(
        b.polygons(kostat, kostat.code, "kostat_2011_shp"),
        "dong_2010",
        method="KOSTAT 2011 읍면동 by 7-digit KOSTAT code (CRS set to EPSG:5174, "
        "as the file's .prj lacks the datum shift); KTDB 2010 codes are "
        "mapped to these in configs/ktdb/dong_2010_kostat.csv",
    )
    # 시군구 as the union of their dongs (10-digit code's first 5 digits)
    for year, dongs, dataset_id in (
        (
            2021,
            first_of(*(b.polygons(g, g.adm_cd2, i) for g, i in dong_2021)),
            "admdong_20210701",
        ),
        (
            2016,
            b.polygons(dong_2016, dong_2016.adm_cd, "admdong_20161231"),
            "admdong_20161231",
        ),
    ):
        b.write(
            b.polygons(dongs, dongs.code.str[:5], dataset_id),
            f"sigungu_{year}",
            method=f"시군구 of {year} as the union of their 행정동 (first 5 "
            "digits of the 10-digit code)",
        )
    # 2010: KTDB's 시도 + 시군구 prefix, as the union of its dongs' KOSTAT
    # polygons (survey-specific codes, so under configs/ktdb/locations)
    classes = pl.read_csv(
        CONFIGS / "ktdb" / "dong_2010_kostat.csv", infer_schema_length=0
    ).to_pandas()
    by_kostat = b.polygons(kostat, kostat.code, "kostat_2011_shp").set_index(
        "code"
    )
    dongs = classes.join(by_kostat, on="to", how="inner")
    dongs = gpd.GeoDataFrame(dongs, geometry="geometry", crs="EPSG:4326")
    b.used = ["kostat_2011_shp"]
    b.write(
        b.polygons(dongs, dongs["from"].str[:5], "kostat_2011_shp"),
        "sigungu_2010",
        method="KTDB 2010 시도 + 시군구 codes (first 5 digits of the "
        "classification code) as the union of their dongs' KOSTAT 2011 "
        "polygons (configs/ktdb/dong_2010_kostat.csv)",
        aliases=["ktdb/dong_2010_kostat.csv"],
    )
    return {
        "dong_2021",
        "dong_2016",
        "dong_2010",
        "sigungu_2021",
        "sigungu_2016",
        "sigungu_2010",
    }


def argentina(b: Builder) -> set[str]:
    # provinces of the surveys: Chaco, Corrientes, Neuquén, Río Negro, Salta,
    # Santa Fe and Entre Ríos (2010); Ciudad and Buenos Aires (2001)
    for year, provinces in (
        (2010, ("18", "22", "30", "58", "62", "66", "82")),
        (2001, ("02", "06")),
    ):
        archive = b.fetch(f"conicet_radios_{year}")
        radios = gpd.read_file(f"zip://{archive}!Radios {year} v2025-1.shp")
        radios = radios[radios.PROV.isin(provinces)]
        b.write(
            b.polygons(
                radios,
                radios.PROV + radios.DEPTO + radios.FRACC + radios.RADIO,
                f"conicet_radios_{year}",
            ),
            f"radio_{year}",
            method=f"INDEC {year} census radios by 9-digit code, provinces {', '.join(provinces)}",
        )
        b.fetch(f"conicet_radios_{year}")
        b.write(
            b.polygons(
                radios,
                radios.PROV + radios.DEPTO + radios.FRACC,
                f"conicet_radios_{year}",
            ),
            f"fraccion_{year}",
            method=f"INDEC {year} census fracciones (7-digit code) as the "
            f"union of their radios, provinces {', '.join(provinces)}",
        )
    return {"radio_2010", "radio_2001", "fraccion_2010", "fraccion_2001"}


def uruguay(b: Builder) -> set[str]:
    segments = gpd.read_file(f"zip://{b.fetch('ine_segments_2011')}")
    b.write(
        b.polygons(segments, segments.codseg.astype(int), "ine_segments_2011"),
        "segment_2011",
        method="INE 2011 census segments by codseg (multi-part segments dissolved)",
    )
    return {"segment_2011"}


def india_name(name: str, state: str, aliases: dict) -> str:
    """Normalised NSS district name, as LGD names it."""
    name = name.lower().replace("&", "and")
    # keep (rural)/(urban) but drop other bracketed alternatives
    name = re.sub(r"\((rural|urban)\)", r"\1", name)
    name = re.sub(r"\(.*?\)|\bdistrict\b", "", name)
    name = re.sub(r"[^a-z]", "", name)
    if state == "07" and name != "newdelhi":
        name = name.removesuffix("delhi")
    return aliases.get(f"{state}:{name}", aliases.get(name, name))


def india_district_name(name: str) -> str:
    """Normalised boundary-file district name."""
    name = name.replace("*", "").lower().replace("&", "and")
    name = re.sub(r"\(.*?\)|\bdistrict\b", "", name)
    return re.sub(r"[^a-z]", "", name)


def find_district(
    table: pd.DataFrame, state: str, name: str, fuzzy: dict | None = None
) -> int | None:
    """Index of the district called *name* in *state*, allowing small
    spelling differences (recorded in *fuzzy*)."""
    candidates = table[table.st == state]
    exact = candidates.index[candidates.n == name]
    if len(exact) == 1:
        return exact[0]
    close = difflib.get_close_matches(
        name, candidates.n.tolist(), n=1, cutoff=0.85
    )
    if not close:
        return None
    if fuzzy is not None:
        fuzzy[f"{state}:{name}"] = close[0]
    return candidates.index[candidates.n == close[0]][0]


def india(b: Builder) -> set[str]:
    lgd = gpd.read_parquet(b.fetch("lgd_districts"))
    c11 = gpd.read_parquet(b.fetch("census_districts_2011"))
    splits = pd.read_csv(b.fetch("india_district_splits"))
    alias_file = CONFIGS / "india" / "geo" / "district.aliases.yaml"
    aliases = yaml.safe_load(alias_file.read_text())
    names = aliases["names"]
    for table in (lgd, c11):
        table["st"] = table.stcode11.astype(str).str.zfill(2)
        table["n"] = table.dtname.map(india_district_name)

    def split_name(name: str) -> str:
        name = india_district_name(name)
        return names.get(name, name)

    splits = splits.assign(
        before=splits["District-Before"].map(split_name),
        after=splits["District-After"].map(split_name),
    )
    splits = splits[(splits.Year >= 2011) & (splits.before != splits.after)]

    # split rows name their state ("Andhra Pradesh-Telangana" for both);
    # give each row the codes of the LGD states it names
    def state_key(name: str) -> str:
        return re.sub(r"[^a-z]", "", name.lower().replace("&", "and"))

    split_states = splits["State/UT"].map(state_key)
    splits = pd.concat(
        [
            *(
                splits[
                    split_states.str.contains(state_key(name), regex=False)
                ].assign(st=st)
                for st, name in lgd[["st", "stname"]]
                .drop_duplicates()
                .itertuples(index=False)
            ),
            pd.DataFrame(
                [
                    (s["state"], s["year"], s["parent"], s["district"])
                    for s in aliases["extra_splits"]
                ],
                columns=["st", "Year", "before", "after"],
            ),
        ]
    )

    built, by_year = set(), {}
    for level in source_lookups("india"):
        if not level.startswith("district_"):
            continue
        year = int(level.removeprefix("district_"))
        districts = pd.read_csv(
            CONFIGS / "india" / f"districts_{year}.csv", dtype=str
        )
        states = {str(k): v for k, v in aliases["states"].get(year, {}).items()}
        matched, unmatched, fuzzy = {}, [], {}
        for d in districts.itertuples():
            name = india_name(d.name, d.state, names)
            state = states.get(d.state, d.state)
            if name in ("lehladakh", "kargil"):
                state = "38"
            split = (splits.st == state) & (splits.before == name)
            split_before = (split & (splits.Year <= year)).any()
            split_after = (split & (splits.Year > year)).any()
            # the 2011 boundary for a district split after the survey but not
            # between 2011 and the survey, or not in LGD (renamed or merged)
            in_c11 = find_district(c11, d.state, name, fuzzy)
            in_lgd = find_district(lgd, state, name, fuzzy)
            code = d.state + d.district
            if (
                in_c11 is not None
                and not split_before
                and (split_after or in_lgd is None)
            ):
                matched[code] = ("c11", in_c11, name, state)
            elif in_lgd is not None:
                matched[code] = ("lgd", in_lgd, name, state)
            elif in_c11 is not None:
                matched[code] = ("c11", in_c11, name, state)
            else:
                unmatched.append(f"{code} {d.name}")
        taken = {i for table, i, *_ in matched.values() if table == "lgd"}
        rows = []
        for code, (table, i, name, state) in matched.items():
            if table == "c11":
                geometry, dataset = [c11.geometry[i]], "census_districts_2011"
            else:
                # add the districts carved out of this one since 2011 that
                # the survey does not list (created after its district list)
                carved = splits[(splits.st == state) & (splits.before == name)]
                children = {
                    j
                    for child in carved.after
                    if (j := find_district(lgd, state, child)) is not None
                    and j not in taken
                }
                geometry = [
                    lgd.geometry[i],
                    *(lgd.geometry[j] for j in children),
                ]
                dataset = "lgd_districts"
            rows.append(
                {
                    "code": code,
                    "dataset": dataset,
                    "geometry": shapely.union_all(geometry),
                }
            )
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="OGC:CRS84")
        by_year[year] = gdf
        lookup = pd.concat(
            b.polygons(group, group.code, dataset)
            for dataset, group in gdf.groupby("dataset")
        )
        b.used = [
            "lgd_districts",
            "census_districts_2011",
            "india_district_splits",
        ]
        b.write(
            lookup,
            level,
            method="NSS districts matched by name (within state) to LGD "
            "districts, merged with districts carved out of them since 2011 "
            "that the survey doesn't list; Census 2011 districts for those "
            "split after the survey only, or not in LGD",
            aliases=[
                str(alias_file.relative_to(CONFIGS)),
                f"india/districts_{year}.csv",
            ],
            unmatched=unmatched,
            fuzzy=fuzzy,
        )
        built.add(level)

    if "nss_region" in source_lookups("india"):
        # NSS regions as the union of their 2024 districts
        districts = pd.read_csv(
            CONFIGS / "india" / "districts_2024.csv", dtype=str
        )
        region = dict(
            zip(districts.state + districts.district, districts.nss_region)
        )
        gdf = by_year[2024].assign(code=by_year[2024].code.map(region))
        b.used = [
            "lgd_districts",
            "census_districts_2011",
            "india_district_splits",
        ]
        b.write(
            b.polygons(gdf, gdf.code, "lgd_districts"),
            "nss_region",
            method="NSS regions as the union of their districts in the TUS "
            "2024 district list (configs/india/districts_2024.csv)",
        )
        built.add("nss_region")
    return built


def mongolia(b: Builder) -> set[str]:
    cod = b.fetch("hdx_mongolia_cod")
    soums = gpd.read_file(f"zip://{cod}!mng_admin2.geojson")
    soums = soums.assign(code=soums.adm2_pcode.str[2:])
    by_soum = b.polygons(soums, soums.code, "hdx_mongolia_cod").set_index(
        "code"
    )

    # NSO units: region digit + aimag (2) + soum (2) [+ bag (2)]; survey
    # codes leave out the region digit
    units = json.loads(b.fetch("nso_units").read_text())["variables"][0]
    soum_codes = [code[1:] for code in units["values"] if len(code) == 5]
    bags = [code[1:] for code in units["values"] if len(code) == 7]

    # Ulaanbaatar khoroos: district name -> soum code, khoroo n -> bag code
    # 51 + 2(n - 1), as NSO numbers them
    khoroos = khoroo_polygons(b.fetch("ub_khoroos"))
    district = dict(zip(soums.adm2_name1, soums.code))
    khoroos["code"] = khoroos.district.map(district) + (
        51 + 2 * (khoroos.khoroo_number - 1)
    ).astype(str)
    by_khoroo = b.polygons(khoroos, khoroos.code, "ub_khoroos").set_index(
        "code"
    )

    built = set()
    for level in source_lookups("mongolia"):
        year = int(level[-4:]) if level[-4:].isdigit() else None
        rows, unmatched = [], []
        if level == "soum":
            for soum in soum_codes:
                if soum in by_soum.index:
                    rows.append({"code": soum, **by_soum.loc[soum].to_dict()})
                else:
                    unmatched.append(soum)
            method = "HDX COD-AB soums (2020) by NSO aimag + soum code"
        else:
            # other bag boundaries are not published: a bag takes its soum;
            # khoroos have been split since 2019, so 2019 khoroos take
            # today's khoroo of that number (work in progress)
            for bag in bags:
                if bag in by_khoroo.index:
                    row = {"code": bag, **by_khoroo.loc[bag].to_dict()}
                    if year is None or year < 2023:
                        row["geom_source"] = "estimated"
                elif bag[:4] in by_soum.index:
                    row = {
                        "code": bag,
                        **by_soum.loc[bag[:4]].to_dict(),
                        "geom_source": "estimated",
                    }
                else:
                    unmatched.append(bag)
                    continue
                rows.append(row)
            method = (
                "NSO bag codes: Ulaanbaatar khoroos from the khoroo map (exact "
                "for 2023, estimated for earlier years); other bags take "
                "their soum (estimated, work in progress)"
            )
        b.write(
            pd.DataFrame(rows),
            level,
            method=method,
            unmatched=unmatched,
        )
        built.add(level)
    return built


def khoroo_polygons(path: Path) -> gpd.GeoDataFrame:
    """Khoroo map records (district, khoroo_number, geometry) as a frame."""
    from shapely.geometry import shape

    records = json.loads(path.read_text())
    return gpd.GeoDataFrame(
        {
            "district": [r["district"] for r in records],
            "khoroo_number": [int(r["khoroo_number"]) for r in records],
        },
        geometry=[shape(r["geometry"]) for r in records],
        crs="EPSG:4326",
    )


def mexico(b: Builder) -> set[str]:
    archive = b.fetch("inegi_mg_2020")
    muns = gpd.read_file(f"zip://{archive}!conjunto_de_datos/00mun.shp")
    muns = muns.assign(mun=muns.CVE_ENT + muns.CVE_MUN)
    # EOD 2017 district polygons are not published and districts cut across
    # municipalities: a district takes the municipalities it overlaps (work
    # in progress), and is exact only where it is one whole municipality
    catalogue = gpd.read_file(
        b.data_root / "MEXICO" / "TCAT_DISTRITOS.DBF", ignore_geometry=True
    )
    catalogue = catalogue.assign(
        code=catalogue.DISTRITO.str.zfill(3), mun=catalogue.ENT + catalogue.MUN
    )
    districts_per_mun = catalogue.groupby("mun").code.nunique()
    exact = catalogue.groupby("code").mun.agg(
        lambda m: len(m) == 1 and districts_per_mun[m.iloc[0]] == 1
    )
    districts = muns.merge(catalogue[["code", "mun"]], on="mun")
    districts = b.polygons(districts, districts.code, "inegi_mg_2020")
    districts["geom_source"] = districts.code.map(exact).map(
        {True: None, False: "estimated"}
    )
    b.write(
        districts,
        "eod_district",
        method="EOD 2017 districts as the union of the INEGI 2020 "
        "municipalities each overlaps (MEXICO/TCAT_DISTRITOS.DBF); estimated "
        "unless the district is one whole municipality",
        extra_datasets={
            "MEXICO/TCAT_DISTRITOS.DBF": "EOD 2017 district catalogue"
        },
    )
    return {"eod_district"}


def french_communes(b: Builder):
    """A function of survey year giving commune boundaries (by INSEE code)
    of the nearest earlier vintage, then of the others."""
    vintages = {}
    for year in (2024, 2019, 2018):
        dataset_id = f"etalab_communes_{year}"
        gdf = gpd.read_file(f"/vsigzip/{b.fetch(dataset_id)}")
        vintages[year] = b.polygons(gdf, gdf.code, dataset_id).set_index("code")
    gdf = gpd.read_file(f"zip://{b.fetch('osm_communes_2015')}")
    vintages[2015] = b.polygons(gdf, gdf.insee, "osm_communes_2015").set_index(
        "code"
    )

    def communes(year: int) -> pd.DataFrame:
        order = sorted(
            vintages, key=lambda v: (v > year, -v if v <= year else v)
        )
        merged = pd.concat([vintages[v] for v in order])
        return merged[~merged.index.duplicated(keep="first")]

    return communes


def survey_places(survey: Path) -> pl.DataFrame:
    """A CEREMA survey's homes and trip ends: ``home_detailed_zone,
    home_draw_zone, home_insee, has_point`` (named as for homes)."""
    columns = ["detailed_zone", "draw_zone", "insee", "lng", "lat"]
    frames = [
        pl.read_parquet(
            survey / "households.parquet",
            columns=[f"home_{c}" for c in columns],
        ).rename(lambda c: c.removeprefix("home_"))
    ]
    trips = survey / "trips.parquet"
    if trips.exists():
        for end in ("origin", "destination"):
            frames.append(
                pl.read_parquet(
                    trips, columns=[f"{end}_{c}" for c in columns]
                ).rename(lambda c, end=end: c.removeprefix(f"{end}_"))
            )
    return pl.concat(frames, how="vertical_relaxed").select(
        home_detailed_zone="detailed_zone",
        home_draw_zone="draw_zone",
        home_insee="insee",
        has_point=pl.col("lng").is_not_null() & pl.col("lat").is_not_null(),
    )


def france(b: Builder) -> set[str]:
    communes = french_communes(b)
    datasets = list(b.used)
    surveys = sorted(
        p for p in (b.data_root / "FRANCE").iterdir() if p.is_dir()
    )
    zones, unmatched, homes_by_year = [], [], {}
    for survey in surveys:
        households = survey / "households.parquet"
        if not households.exists():
            continue
        year = survey_year(survey.name)
        # places: homes and trip ends, as (zone, draw zone, commune, point)
        homes = survey_places(survey)
        homes_by_year.setdefault(year, []).append(
            homes.filter(
                pl.col("home_detailed_zone").is_null() & ~pl.col("has_point")
            )["home_insee"]
        )
        homes = homes.drop_nulls("home_detailed_zone")
        path = survey / "detailed_zones.geo.parquet"
        if path.exists():
            gdf = gpd.read_parquet(path)
            own = b.polygons(gdf, gdf.detailed_zone_id, None)
            own["geom_dataset"] = f"CEREMA {survey.name} detailed zones"
            own["geom_year"] = year
        else:
            own = gpd.GeoDataFrame(
                columns=["code", "lon", "lat", "area_km2", "geometry"],
                geometry="geometry",
                crs="EPSG:4326",
            )
        # zones without a polygon take their draw zone, else their commune
        # (the most common among the zone's homes and trip ends)
        missing = (
            homes.filter(~pl.col("home_detailed_zone").is_in(own.code.tolist()))
            .group_by("home_detailed_zone")
            .agg(
                pl.col("home_draw_zone").drop_nulls().mode().first(),
                pl.col("home_insee").drop_nulls().mode().first(),
            )
            .to_pandas()
        )
        parents = []
        draw_path = survey / "draw_zones.geo.parquet"
        if draw_path.exists() and len(missing):
            draw = gpd.read_parquet(draw_path)
            by_draw = b.polygons(draw, draw.draw_zone_id, None).set_index(
                "code"
            )
            by_draw["geom_dataset"] = f"CEREMA {survey.name} draw zones"
            by_draw["geom_year"] = year
            parents.append(
                missing.join(by_draw, on="home_draw_zone", how="inner")
            )
        parents.append(
            missing.join(communes(year), on="home_insee", how="inner")
        )
        parent = (
            pd.concat([p for p in parents if len(p)] or [missing.iloc[0:0]])
            .drop_duplicates("home_detailed_zone")
            .rename(columns={"home_detailed_zone": "code"})
            .assign(geom_source="parent")
        )
        unmatched += [
            f"{survey.name}/{z}"
            for z in set(missing.home_detailed_zone) - set(parent.code)
        ]
        unit = first_of(own.assign(geom_source=None), parent)
        unit["code"] = f"{survey.name}/" + unit.code.astype(str)
        zones.append(unit)
    b.used = datasets
    b.write(
        pd.concat([z for z in zones if len(z)]),
        "detailed_zone",
        method="each CEREMA survey's detailed zone polygons (FRANCE/<survey>/"
        "detailed_zones.geo.parquet), coded <survey>/<zone id>; zones "
        "without a polygon take their draw zone, else the commune most of "
        "their households are in (parent)",
        unmatched=unmatched,
        extra_datasets={"FRANCE/<survey>": "CEREMA survey zone files"},
    )

    # communes of homes and trip ends with neither a detailed zone nor a
    # point (the loaders' last fallback), as of the survey year
    built = {"detailed_zone"}
    for level in source_lookups("france"):
        if not level.startswith("commune_"):
            continue
        year = int(level.removeprefix("commune_"))
        codes = pl.concat(
            homes_by_year.get(year, [pl.Series([], dtype=pl.String)])
        )
        lookup = communes(year)
        lookup = lookup[lookup.index.isin(codes.to_list())].reset_index()
        built.add(level)
        if not len(lookup):
            continue  # no household of that year falls back to its commune
        b.used = datasets
        b.write(
            lookup,
            level,
            method=f"communes of {year} (nearest earlier vintage first) "
            "for homes and trip ends of that year's surveys with neither a "
            "detailed zone nor a point",
            unmatched=sorted(set(codes.drop_nulls()) - set(lookup.code)),
        )
    return built


def chile(b: Builder) -> set[str]:
    gdf = gpd.read_parquet(b.data_root / "CHILE" / "zonificacion.parquet")
    zones = b.polygons(gdf, gdf.zona.astype(int).astype(str), None)
    zones["geom_dataset"] = "SECTRA EOD 2012 zonificación"
    zones["geom_year"] = 2012
    b.write(
        zones,
        "eod_zone_2012",
        method="Santiago EOD 2012 zones (CHILE/zonificacion.parquet) by zona; "
        "areas from the polygons (the file's area column doesn't match them)",
        extra_datasets={"CHILE/zonificacion.parquet": "EOD 2012 zones"},
    )
    return {"eod_zone_2012"}


def brazil(b: Builder) -> set[str]:
    gdf = gpd.read_file(
        b.data_root
        / "BRAZIL/OD-2017/Mapas-OD2017/Shape-OD2017/Zonas_2017_region.shp"
    )
    zones = b.polygons(gdf, gdf.NumeroZona.astype(int).astype(str), None)
    zones["geom_dataset"] = "Metrô SP OD 2017 zonas"
    zones["geom_year"] = 2017
    b.write(
        zones,
        "od_zone_2017",
        method="São Paulo OD 2017 zones (Zonas_2017_region.shp, shipped "
        "with the survey) by NumeroZona",
        extra_datasets={
            "BRAZIL/OD-2017/Mapas-OD2017/Shape-OD2017/Zonas_2017_region.shp": (
                "OD 2017 zones"
            )
        },
    )
    return {"od_zone_2017"}


BUILDERS = {
    "nts": nts,
    "ltds": ltds,
    "nhts": nhts,
    "cmap": cmap,
    "qhts": qhts,
    "vista": vista,
    "mon": nl_source("mon"),
    "ovin": nl_source("ovin"),
    "odin": nl_source("odin"),
    "ktdb": ktdb,
    "argentina": argentina,
    "uruguay": uruguay,
    "india": india,
    "mongolia": mongolia,
    "mexico": mexico,
    "france": france,
    "chile": chile,  # homes are points; trip ends are EOD zones
    "brazil": brazil,  # points, else OD zones
}


def main(
    data_root: Path = DEFAULT_DATA_ROOT,
    only: list[str] = typer.Option(
        None, help=f"Build only these sources: {', '.join(BUILDERS)}"
    ),
):
    """Write the location lookups each source's declarations name."""
    builder = Builder(data_root.expanduser())
    for source, build in BUILDERS.items():
        if only and source not in only:
            continue
        print(f"== {source}")
        builder.source = source
        built = build(builder)
        unbuilt(source, built)


if __name__ == "__main__":
    typer.run(main)
