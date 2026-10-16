"""Build one location lookup from a local boundary file.

foundata.location resolves codes of a level with
configs/<source>/locations/<level>.parquet (survey-specific zones) or
configs/geo/<country>/<level>.parquet (national geographies). This writes
either from any boundary file geopandas reads, the same way as
scripts/build_location_lookups.py: one simplified polygon per code, a point
inside it as lon/lat, its area in an equal-area projection, and
provenance metadata. To make it reproducible, add the file as a Dataset
and a builder in build_location_lookups.py instead.

Usage:
    uv run --group scripts python scripts/build_location_lookup.py \\
        ~/Data/foundata/CHILE/zonificacion.parquet eod_zone_2012 \\
        --source chile --code-column zona --dataset "EOD 2012 zones" --year 2012
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import typer
from build_location_lookups import DEFAULT_DATA_ROOT, Builder


def main(
    boundaries: Path,
    level: str,
    code_column: str = typer.Option(..., help="Column holding the unit code"),
    source: str = typer.Option(
        None, help="Write to configs/<source>/locations/"
    ),
    country: str = typer.Option(None, help="Write to configs/geo/<country>/"),
    zfill: int = typer.Option(None, help="Zero-pad codes to this width"),
    crs: str = typer.Option(None, help="CRS, for files without one"),
    dataset: str = typer.Option(..., help="Name of the boundary dataset"),
    year: int = typer.Option(None, help="Vintage of the boundaries"),
):
    """Write a lookup for *level* from a boundary file."""
    if (source is None) == (country is None):
        raise typer.BadParameter("give exactly one of --source and --country")
    print(f"Reading: {boundaries}")
    gdf = (
        gpd.read_parquet(boundaries)
        if boundaries.suffix == ".parquet"
        else gpd.read_file(boundaries)
    )
    if crs:
        gdf = gdf.set_crs(crs, allow_override=True)
    code = gdf[code_column].astype(str)
    if zfill:
        code = code.str.zfill(zfill)
    builder = Builder(DEFAULT_DATA_ROOT)
    lookup = builder.polygons(gdf, pd.Series(code.values), None)
    lookup["geom_dataset"] = dataset
    lookup["geom_year"] = year
    builder.write(
        lookup,
        level,
        source=source,
        country=country,
        method=f"{boundaries.name} by {code_column}",
        extra_datasets={str(boundaries): dataset},
    )


if __name__ == "__main__":
    typer.run(main)
