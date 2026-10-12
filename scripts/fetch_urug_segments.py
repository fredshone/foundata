"""Download INE Uruguay 2011 census segments, compute centroids, write CSV.

The urug survey (Montevideo EOD 2016) locates homes and trip ends by 2011
census segment (codseg). This writes one row per segment with a
representative point in UTM 21S metres (EPSG:32721) and the INE locality
code (codloc), used for trip distances and zone urbanity.

A few rural segments are split into several polygons (one per locality);
these are dissolved and take the locality of their largest part.

Usage:
    uv run --group scripts python scripts/fetch_urug_segments.py
    uv run --group scripts python scripts/fetch_urug_segments.py --shp /path/to/ine_seg_11.shp
"""

from pathlib import Path

import click
import geopandas as gpd
import pandas as pd

URL = (
    "https://mapas.ide.uy/geoserver-vectorial/INE_NO_SEGURO/wfs?service=WFS"
    "&request=GetFeature&version=1.0.0&outputFormat=shape-zip"
    "&typeName=ine_seg_11"
)
DEFAULT_OUT = Path(__file__).parent.parent / "configs/urug/segments.csv"


@click.command()
@click.option(
    "--shp", default=None, help="Local shapefile path (skips download)"
)
@click.option("--out", default=str(DEFAULT_OUT), show_default=True)
def main(shp, out):
    """Compute segment centroids and write configs/urug/segments.csv."""
    source = shp or URL
    print(f"Reading: {source}")
    gdf = gpd.read_file(source).to_crs("EPSG:32721")

    # locality of each segment's largest part
    largest = gdf.sort_values("area", ascending=False).drop_duplicates("codseg")
    dissolved = gdf[["codseg", "geometry"]].dissolve(by="codseg")
    points = dissolved.geometry.representative_point()

    out_df = pd.DataFrame(
        {
            "codseg": dissolved.index.astype(int),
            "x": points.x.round(1).values,
            "y": points.y.round(1).values,
        }
    ).merge(largest[["codseg", "codloc"]].astype(int), on="codseg", how="left")
    out_df.sort_values("codseg").to_csv(out, index=False)
    print(f"Wrote {len(out_df)} segments to {out}")


if __name__ == "__main__":
    main()
