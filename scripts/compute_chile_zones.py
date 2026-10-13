"""Compute area and centroid of the Santiago EOD 2012 zones, write CSV.

The chile survey (Santiago EOD 2012) locates homes and trip ends by EOD
zone (1-866). Its zoning file (CHILE/zonificacion.parquet, a geoparquet in
UTM 19S, EPSG:32719) carries an `area` column that doesn't match the zone
geometry, so this computes each zone's area (km²) and a representative
point (UTM metres) from the polygons. They're used for zone urbanity
(residential density) and fallback trip distances.

Usage:
    uv run --group scripts python scripts/compute_chile_zones.py
    uv run --group scripts python scripts/compute_chile_zones.py --zones /path/to/zonificacion.parquet
"""

from pathlib import Path

import click
import geopandas as gpd
import pandas as pd

DEFAULT_ZONES = Path.home() / "Data/foundata/CHILE/zonificacion.parquet"
DEFAULT_OUT = Path(__file__).parent.parent / "configs/chile/zones.csv"


@click.command()
@click.option("--zones", default=str(DEFAULT_ZONES), show_default=True)
@click.option("--out", default=str(DEFAULT_OUT), show_default=True)
def main(zones, out):
    """Compute zone areas and centroids and write configs/chile/zones.csv."""
    print(f"Reading: {zones}")
    gdf = gpd.read_parquet(zones).to_crs("EPSG:32719")
    points = gdf.geometry.representative_point()

    out_df = pd.DataFrame(
        {
            "zone": gdf["zona"].astype(int),
            "area_km2": (gdf.geometry.area / 1e6).round(4),
            "x": points.x.round(1),
            "y": points.y.round(1),
        }
    )
    out_df.sort_values("zone").to_csv(out, index=False)
    print(f"Wrote {len(out_df)} zones to {out}")


if __name__ == "__main__":
    main()
