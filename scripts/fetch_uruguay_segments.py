"""Download INE Uruguay 2011 census segments and write each one's locality.

The uruguay survey (Montevideo EOD 2016) locates homes and trip ends by 2011
census segment (codseg). This writes the INE locality code (codloc) of each
segment, used for zone urbanity; the segments' geometry (for locations and
trip distances) is the segment_2011 location lookup
(scripts/build_location_lookups.py).

A few rural segments are split into several polygons (one per locality);
these take the locality of their largest part.

Usage:
    uv run --group scripts python scripts/fetch_uruguay_segments.py
    uv run --group scripts python scripts/fetch_uruguay_segments.py --shp /path/to/ine_seg_11.shp
"""

from pathlib import Path
from typing import Optional

import geopandas as gpd
import typer

URL = (
    "https://mapas.ide.uy/geoserver-vectorial/INE_NO_SEGURO/wfs?service=WFS"
    "&request=GetFeature&version=1.0.0&outputFormat=shape-zip"
    "&typeName=ine_seg_11"
)
DEFAULT_OUT = (
    Path(__file__).parent.parent / "configs/uruguay/segment_localities.csv"
)


def main(
    shp: Optional[Path] = typer.Option(
        None, help="Local shapefile path (skips download)"
    ),
    out: Path = DEFAULT_OUT,
):
    """Write configs/uruguay/segment_localities.csv (codseg, codloc)."""
    source = shp or URL
    print(f"Reading: {source}")
    gdf = gpd.read_file(source).to_crs("EPSG:32721")

    # locality of each segment's largest part
    largest = gdf.sort_values("area", ascending=False).drop_duplicates("codseg")
    out_df = largest[["codseg", "codloc"]].astype(int).sort_values("codseg")
    out_df.to_csv(out, index=False)
    print(f"Wrote {len(out_df)} segments to {out}")


if __name__ == "__main__":
    typer.run(main)
