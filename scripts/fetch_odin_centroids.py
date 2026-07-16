"""Download Dutch municipality (gemeente) boundaries, compute centroids, write CSV.

Usage:
    uv run python scripts/fetch_odin_centroids.py
    uv run python scripts/fetch_odin_centroids.py --years 2018,2019,2020,2023,2024
"""

from pathlib import Path

import click
import geopandas as gpd
import pandas as pd

URL_TEMPLATE = "https://cartomap.github.io/nl/wgs84/gemeente_{year}.geojson"
DEFAULT_OUT = (
    Path(__file__).parent.parent / "configs/odin/gemeente_centroids.csv"
)
DEFAULT_YEARS = "2018,2019,2020,2023,2024"


def load_year_centroids(year: int) -> pd.DataFrame:
    url = URL_TEMPLATE.format(year=year)
    print(f"Reading: {url}")
    gdf = gpd.read_file(url)

    # Project to RD New (metres) for accurate centroid computation
    gdf = gdf.to_crs("EPSG:28992")
    centroids = gdf.geometry.centroid

    # Reproject centroids to WGS84 for lat/lon output
    centroids_wgs = centroids.to_crs("EPSG:4326")

    return pd.DataFrame(
        {
            "year": year,
            "region_code": gdf["statcode"].str.removeprefix("GM").str.zfill(4),
            "region_name": gdf["statnaam"],
            "lat": centroids_wgs.y,
            "lon": centroids_wgs.x,
        }
    )


@click.command()
@click.option(
    "--years",
    default=DEFAULT_YEARS,
    show_default=True,
    help="Comma-separated ODIN survey years",
)
@click.option("--out", default=str(DEFAULT_OUT), show_default=True)
def main(years: str, out: str):
    """Compute gemeente centroids per ODIN survey year and write configs/odin/gemeente_centroids.csv."""
    year_list = [int(y) for y in years.split(",")]
    frames = [load_year_centroids(year) for year in year_list]
    out_df = pd.concat(frames, ignore_index=True)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Written {len(out_df)} centroids to {out_path}")


if __name__ == "__main__":
    main()
