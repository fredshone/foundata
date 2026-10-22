"""Export WorldTensor layers to a GeoParquet of grid cells, for plotting.

One row per 0.25 degree cell, with its polygon (EPSG:4326, lon in
[-180, 180]) and a column per variable from that variable's most recent
year on disk (static layers: their only one). The coordinates in the
files are cell centres, so each cell spans +-0.125 degrees around its
centre, clipped at the poles and at the antimeridian. Cells with no
value in any variable (ocean) are dropped unless ``--keep-empty``.

The year, units and long name of each variable go to a ``.json`` next
to the output.

Usage:
    uv run python scripts/export_worldtensor_grid.py [--variables hdi gini ...]
        [--bbox XMIN YMIN XMAX YMAX] [--keep-empty] [--world DIR] [--out FILE]

Plot with:
    gdf = geopandas.read_parquet("output/worldtensor_grid.parquet")
    gdf.plot(column="hdi", cmap="viridis", legend=True)
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from loguru import logger

from foundata import log
from foundata.gridjoin import DEFAULT_WORLD, worldtensor
from foundata.gridjoin.grid import WORLDTENSOR

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument(
    "--variables",
    nargs="+",
    choices=list(worldtensor.VARIABLES),
    default=list(worldtensor.VARIABLES),
)
parser.add_argument(
    "--bbox",
    nargs=4,
    type=float,
    metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
    help="keep only cells whose centre is in this lon/lat box",
)
parser.add_argument("--keep-empty", action="store_true")
parser.add_argument("--world", type=Path, default=DEFAULT_WORLD)
parser.add_argument(
    "--out", type=Path, default=Path("output/worldtensor_grid.parquet")
)
args = parser.parse_args()

log.setup("INFO")
g = WORLDTENSOR
cell = np.arange(g.size)
lon, lat = g.centre(cell)  # lon in [-180, 180)
half = g.step / 2

keep = np.ones(g.size, bool)
if args.bbox:
    xmin, ymin, xmax, ymax = args.bbox
    keep &= (lon >= xmin) & (lon <= xmax) & (lat >= ymin) & (lat <= ymax)

columns, meta = {}, {}
for var in args.variables:
    source = args.world / worldtensor.VARIABLES[var]
    files = worldtensor.available(source)
    path = source if files is None else files[max(files)]
    data, attrs = worldtensor._read_layer(path, var)  # rows south to north
    columns[var] = data.ravel()
    meta[var] = {
        "year": attrs["reference_year"] if files is None else max(files),
        "units": attrs.get("units"),
        "long_name": attrs.get("long_name"),
        "source": str(path),
    }
    logger.info(f"{var}: {meta[var]['year']} from {path}")

if not args.keep_empty:
    keep &= ~np.all(np.isnan(np.stack(list(columns.values()))), axis=0)

lon, lat = lon[keep], lat[keep]
geometry = shapely.box(
    np.clip(lon - half, -180, 180),
    np.clip(lat - half, -90, 90),
    np.clip(lon + half, -180, 180),
    np.clip(lat + half, -90, 90),
)
gdf = gpd.GeoDataFrame(
    {
        "cell": cell[keep].astype(np.int32),
        "lon": lon,
        "lat": lat,
        **{var: col[keep] for var, col in columns.items()},
    },
    geometry=geometry,
    crs="EPSG:4326",
)

args.out.parent.mkdir(parents=True, exist_ok=True)
gdf.to_parquet(args.out)
args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
logger.info(f"{len(gdf)} cells -> {args.out}")
