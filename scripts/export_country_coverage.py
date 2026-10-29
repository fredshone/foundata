"""Export world country polygons flagged by foundata coverage, for mapping.

One row per Natural Earth admin-0 country (EPSG:4326) with an
``included`` flag and the foundata sources covering it. Coverage is
found by scanning ``foundata/*.py`` for the country codes each source
writes (``COUNTRY = "..."`` or ``country=pl.lit("...")``), so a new
source is picked up automatically once its code is in ``ISO3``.

Usage:
    uv run --group scripts python scripts/export_country_coverage.py
        [--scale 110m|50m|10m] [--geometry FILE] [--out FILE]

Plot with:
    gdf = geopandas.read_parquet("output/country_coverage.parquet")
    gdf.plot(column="included", legend=True)
"""

import argparse
import re
from pathlib import Path

import geopandas as gpd
from loguru import logger

from foundata import log

NATURAL_EARTH = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_{scale}_admin_0_countries.geojson"
)

# foundata country code -> ISO 3166-1 alpha-3 (Natural Earth ADM0_A3).
ISO3 = {
    "ar": "ARG",
    "aus": "AUS",
    "br": "BRA",
    "cl": "CHL",
    "fr": "FRA",
    "ind": "IND",
    "kr": "KOR",
    "mng": "MNG",
    "mx": "MEX",
    "nl": "NLD",
    "uk": "GBR",
    "usa": "USA",
    "uy": "URY",
}

SOURCE_DIR = Path(__file__).resolve().parents[1] / "foundata"
PATTERNS = [
    re.compile(r'^COUNTRY\s*=\s*"([^"]+)"', re.M),
    re.compile(r'country=pl\.lit\("([^"]+)"'),
    re.compile(r'pl\.lit\("([^"]+)"\)\.alias\("country"\)'),
]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--scale", choices=["110m", "50m", "10m"], default="110m")
parser.add_argument(
    "--geometry", help="local countries file, instead of Natural Earth"
)
parser.add_argument("--out", default="output/country_coverage.parquet")


def source_countries() -> dict[str, list[str]]:
    """Map each foundata country code to the sources that write it."""
    coverage: dict[str, list[str]] = {}
    for path in sorted(SOURCE_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        codes = {c for p in PATTERNS for c in p.findall(text)}
        for code in codes:
            coverage.setdefault(code, []).append(path.stem)
    return coverage


def main():
    args = parser.parse_args()
    log.setup()

    coverage = source_countries()
    unknown = set(coverage) - set(ISO3)
    if unknown:
        raise SystemExit(f"add ISO3 codes for country codes: {sorted(unknown)}")
    by_iso3 = {ISO3[code]: sources for code, sources in coverage.items()}
    logger.info(f"foundata covers {len(by_iso3)} countries: {sorted(by_iso3)}")

    src = args.geometry or NATURAL_EARTH.format(scale=args.scale)
    logger.info(f"reading countries from {src}")
    world = gpd.read_file(src)
    world = world[["ADM0_A3", "NAME", "CONTINENT", "geometry"]].rename(
        columns={"ADM0_A3": "iso3", "NAME": "name", "CONTINENT": "continent"}
    )
    world = world.to_crs("EPSG:4326")

    world["included"] = world["iso3"].isin(by_iso3)
    world["sources"] = world["iso3"].map(
        lambda iso3: ",".join(by_iso3.get(iso3, []))
    )
    missing = set(by_iso3) - set(world["iso3"])
    if missing:
        logger.warning(f"covered countries missing from geometry: {missing}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    world.to_parquet(out)
    logger.info(
        f"wrote {len(world)} countries ({world['included'].sum()} included) to {out}"
    )


if __name__ == "__main__":
    main()
