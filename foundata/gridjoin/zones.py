"""Zones to join to: a locations GeoParquet as flat numpy columns.

The join needs, per zone, its point (``lon``, ``lat``: the point on the
surface of its polygon, so always inside it), its radius (``scale_m``:
the radius of a circle of the polygon's area, or a point's uncertainty)
and the bounding box of its geometry. Bounds are the one geometry
operation, done once on load (``shapely.bounds``, vectorised).

A geometry split at the antimeridian (Alaska) has parts near -180 and
180, so its plain bounds span the globe; its box is instead the
narrowest one around its parts, with ``xmax`` above 180.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import shapely

from foundata.gridjoin.grid import KM_PER_DEG

# radius of a zone with no better measure (a point's assumed uncertainty)
MIN_RADIUS_M = 50.0


@dataclass
class Zones:
    table: pa.Table  # the original table, untouched
    lon: np.ndarray
    lat: np.ndarray
    radius_m: np.ndarray  # max(scale_m, MIN_RADIUS_M), NaN where unknown
    is_point: np.ndarray
    xmin: np.ndarray  # footprint box: the geometry's bounds for a
    ymin: np.ndarray  # polygon, the point +/- radius for a point;
    xmax: np.ndarray  # xmax may exceed 180 across the antimeridian
    ymax: np.ndarray
    _geometry: np.ndarray | None = None

    def __len__(self) -> int:
        return self.lon.size

    @property
    def located(self) -> np.ndarray:
        return np.isfinite(self.lon) & np.isfinite(self.lat)

    @property
    def geometry(self) -> np.ndarray:
        """Shapely geometries (decoded on first use)."""
        if self._geometry is None:
            self._geometry = shapely.from_wkb(
                self.table["geometry"].to_numpy(zero_copy_only=False)
            )
        return self._geometry


def read(path: str | Path) -> Zones:
    return from_table(pq.read_table(path))


def from_table(table: pa.Table) -> Zones:
    lon = _floats(table["lon"])
    lat = _floats(table["lat"])
    radius = np.fmax(_floats(table["scale_m"]), MIN_RADIUS_M)
    radius[~np.isfinite(lon)] = np.nan
    zones = Zones(table, lon, lat, radius, *([None] * 5))
    geoms = zones.geometry
    types = shapely.get_type_id(geoms)
    zones.is_point = types == shapely.GeometryType.POINT
    bounds = shapely.bounds(geoms)
    _fix_antimeridian(geoms, bounds)
    # points: point +/- radius
    dlat = radius / 1000 / KM_PER_DEG
    dlon = dlat / np.maximum(np.cos(np.radians(lat)), 1e-6)
    point_box = np.column_stack(
        [
            lon - dlon,
            np.maximum(lat - dlat, -90),
            lon + dlon,
            np.minimum(lat + dlat, 90),
        ]
    )
    use_point = zones.is_point | (types < 0)
    bounds[use_point] = point_box[use_point]
    zones.xmin, zones.ymin, zones.xmax, zones.ymax = bounds.T.copy()
    return zones


def _floats(column: pa.ChunkedArray) -> np.ndarray:
    return column.to_numpy(zero_copy_only=False).astype(float)


def _fix_antimeridian(geoms: np.ndarray, bounds: np.ndarray) -> None:
    """Replace globe-spanning bounds by the narrowest box over the parts."""
    for i in np.flatnonzero(bounds[:, 2] - bounds[:, 0] > 180):
        parts = shapely.bounds(shapely.get_parts(geoms[i]))
        west = parts[:, 2] <= 0  # parts that shift east by 360
        shifted = parts.copy()
        shifted[west, 0] += 360
        shifted[west, 2] += 360
        xmin, xmax = shifted[:, 0].min(), shifted[:, 2].max()
        if xmax - xmin < bounds[i, 2] - bounds[i, 0]:
            bounds[i, 0], bounds[i, 2] = xmin, xmax
