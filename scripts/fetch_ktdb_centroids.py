"""Download Korean administrative dong boundaries, compute centroids, write CSV.

Usage:
    uv run python scripts/fetch_ktdb_centroids.py
    uv run python scripts/fetch_ktdb_centroids.py --geojson /path/to/file.geojson
"""

from pathlib import Path

import click
import geopandas as gpd
import pandas as pd

URL = "https://raw.githubusercontent.com/vuski/admdongkor/master/ver20210701/HangJeongDong_ver20210701.geojson"
DEFAULT_OUT = Path(__file__).parent.parent / "configs/ktdb/zone_centroids.csv"


@click.command()
@click.option("--geojson", default=None, help="Local GeoJSON path (skips download)")
@click.option("--out", default=str(DEFAULT_OUT), show_default=True)
def main(geojson, out):
    """Compute zone centroids and write configs/ktdb/zone_centroids.csv."""
    source = geojson or URL
    print(f"Reading: {source}")
    gdf = gpd.read_file(source)

    # Project to Korea TM (metres) for accurate centroid computation
    gdf = gdf.to_crs("EPSG:5179")
    centroids = gdf.geometry.centroid

    # Reproject centroids to WGS84 for lat/lon output
    centroids_wgs = centroids.to_crs("EPSG:4326")

    out_df = pd.DataFrame(
        {
            "zone": gdf["adm_cd2"].astype(str).str.zfill(10),
            "lat": centroids_wgs.y,
            "lon": centroids_wgs.x,
        }
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"Written {len(out_df)} centroids to {out}")


if __name__ == "__main__":
    main()
