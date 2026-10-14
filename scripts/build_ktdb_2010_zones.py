"""Build the zone lookups for the KTDB 2010 household and weekend surveys.

The 2010 surveys code home locations as "SS-GGG-DDD" (16 시도, 251 시군구
zones, last 3 digits of the 읍면동 code) and, in the national trip files, trip
ends as the 3-digit 251-zone number only. This script writes:

- configs/ktdb/zones_2010.csv: one row per 읍면동 (code = SSGGGDDD), with
  names, from the classification sheet shipped with the survey.
- configs/ktdb/zone_centroids_2010.csv: one row per 251 zone, the centroid of
  the union of its 읍면동 boundaries (vuski/admdongkor, Dec 2012 version, which
  uses the same 7-digit 행정구역 codes as the classification sheet).

Usage:
    uv run --group scripts python scripts/build_ktdb_2010_zones.py \\
        --classification "/path/to/가구통행실태조사 분류 번호 및 행정코드.xls"
"""

from pathlib import Path

import click
import geopandas as gpd
import pandas as pd
import polars as pl

URL = "https://raw.githubusercontent.com/vuski/admdongkor/master/ver20121210/HangJeongDong_ver20121210.geojson"
OUT_DIR = Path(__file__).parent.parent / "configs/ktdb"

# Districts reorganised between the 2010 survey and the Dec 2012 boundaries,
# so their 2010 codes have no polygons: 251 zone -> 2012 adm_cd prefix of the
# successor. 연기군 became most of 세종특별자치시 (Jul 2012) and 당진군 became
# 당진시 (Jan 2012).
SUCCESSORS = {"159": "2901", "166": "3408"}


@click.command()
@click.option(
    "--classification",
    required=True,
    help="가구통행실태조사 분류 번호 및 행정코드.xls from the 2010 download",
)
@click.option("--geojson", default=URL, show_default=True)
@click.option("--out-dir", default=str(OUT_DIR), show_default=True)
def main(classification, geojson, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zones = (
        pl.read_excel(classification, infer_schema_length=0)
        .rename(
            {
                "시도": "sido",
                "시군구": "sigungu",
                "읍면동": "dong",
                "분류코드": "code",
                "행정구역 코드": "adm_cd",
            }
        )
        .select("code", "sido", "sigungu", "dong", "adm_cd")
        .with_columns(
            code=pl.col("code").str.replace_all("-", ""),
            zone=pl.col("code").str.slice(3, 3),
        )
        .sort("code")
    )
    zones.drop("adm_cd").write_csv(out_dir / "zones_2010.csv")
    print(f"Written {zones.height} 읍면동 to {out_dir / 'zones_2010.csv'}")

    print(f"Reading: {geojson}")
    gdf = gpd.read_file(geojson)[["adm_cd", "geometry"]]
    lookup = zones.select("adm_cd", "zone").to_pandas()
    matched = gdf.merge(lookup, on="adm_cd", how="inner")
    successors = [
        gdf[gdf["adm_cd"].str.startswith(prefix)].assign(zone=zone)
        for zone, prefix in SUCCESSORS.items()
    ]
    gdf = pd.concat([matched, *successors], ignore_index=True)
    missing = set(lookup["zone"]) - set(gdf["zone"])
    if missing:
        raise ValueError(f"No boundaries for 251 zones: {sorted(missing)}")

    # Project to Korea TM (metres) for accurate centroid computation
    districts = gdf.to_crs("EPSG:5179").dissolve(by="zone")
    centroids = districts.geometry.centroid.to_crs("EPSG:4326")
    out = pl.DataFrame(
        {
            "zone": list(centroids.index),
            "lat": list(centroids.y),
            "lon": list(centroids.x),
        }
    ).sort("zone")
    out.write_csv(out_dir / "zone_centroids_2010.csv")
    print(
        f"Written {out.height} centroids to {out_dir / 'zone_centroids_2010.csv'}"
    )


if __name__ == "__main__":
    main()
