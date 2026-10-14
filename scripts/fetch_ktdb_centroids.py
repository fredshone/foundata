"""Download Korean administrative dong boundaries, compute centroids, write CSV.

Usage:
    uv run python scripts/fetch_ktdb_centroids.py
    uv run python scripts/fetch_ktdb_centroids.py --geojson /path/to/file.geojson

    # 2016 survey zones (행정동 codes as of 2016-12-31)
    uv run python scripts/fetch_ktdb_centroids.py \
        --geojson https://raw.githubusercontent.com/vuski/admdongkor/master/ver20161231/HangJeongDong_ver20161231.geojson \
        --out configs/ktdb/zone_centroids_2016.csv

--geojson can be repeated; a zone keeps the centroid from the first source
that has it, so later sources only add zones the earlier ones lack.
"""

from pathlib import Path

import click
import geopandas as gpd
import pandas as pd

BASE = "https://raw.githubusercontent.com/vuski/admdongkor/master"
URLS = [
    f"{BASE}/ver20210701/HangJeongDong_ver20210701.geojson",
    # the 2021 survey also uses a few dongs created after July 2021
    f"{BASE}/ver20220101/HangJeongDong_ver20220101.geojson",
]
DEFAULT_OUT = Path(__file__).parent.parent / "configs/ktdb/zone_centroids.csv"


def centroids(source: str) -> pd.DataFrame:
    print(f"Reading: {source}")
    gdf = gpd.read_file(source)

    # Project to Korea TM (metres) for accurate centroid computation
    gdf = gdf.to_crs("EPSG:5179")
    points = gdf.geometry.centroid

    # Reproject centroids to WGS84 for lat/lon output
    points_wgs = points.to_crs("EPSG:4326")

    return pd.DataFrame(
        {
            # older versions carry the 10-digit code in adm_cd, not adm_cd2
            "zone": gdf.get("adm_cd2", gdf["adm_cd"]).astype(str).str.zfill(10),
            "lat": points_wgs.y,
            "lon": points_wgs.x,
        }
    )


@click.command()
@click.option(
    "--geojson",
    multiple=True,
    help="GeoJSON path or URL, repeatable (default: 2021-07 plus 2022-01)",
)
@click.option("--out", default=str(DEFAULT_OUT), show_default=True)
def main(geojson, out):
    """Compute zone centroids and write configs/ktdb/zone_centroids.csv."""
    frames = []
    for source in geojson or URLS:
        df = centroids(source)
        if frames:
            known = pd.concat(frames)["zone"]
            df = df[~df["zone"].isin(known)]
            print(f"  adding {len(df)} zones")
        frames.append(df)
    out_df = pd.concat(frames, ignore_index=True)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"Written {len(out_df)} centroids to {out}")


if __name__ == "__main__":
    main()
