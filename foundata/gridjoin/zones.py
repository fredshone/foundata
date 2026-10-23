"""Zones to join to: a locations GeoParquet as flat numpy columns.

The join needs, per zone, its point (``lon``, ``lat``: the point on the
surface of its polygon, so always inside it) and the bounding box of its
geometry (a point's is the point itself: its ``scale_m`` is ignored).
Bounds are done once on load (``shapely.bounds``, vectorised); only
polygons that cross a cell edge are intersected with the cells
(``cell_fractions``).

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

from foundata.gridjoin.grid import Grid


@dataclass
class Zones:
    table: pa.Table  # the original table, untouched
    lon: np.ndarray
    lat: np.ndarray
    is_point: np.ndarray
    xmin: np.ndarray  # footprint box: the geometry's bounds for a
    ymin: np.ndarray  # polygon, the point itself for a point;
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
    zones = Zones(table, lon, lat, *([None] * 5))
    geoms = zones.geometry
    types = shapely.get_type_id(geoms)
    zones.is_point = types == shapely.GeometryType.POINT
    bounds = shapely.bounds(geoms)
    _fix_antimeridian(geoms, bounds)
    use_point = zones.is_point | (types < 0)
    bounds[use_point] = np.column_stack([lon, lat, lon, lat])[use_point]
    zones.xmin, zones.ymin, zones.xmax, zones.ymax = bounds.T.copy()
    return zones


def cell_fractions(
    zones: Zones, grid: Grid, which: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The cells of *grid* that each zone in *which* (indices) overlaps,
    and the fraction of each cell's area (in degrees, so equal-area within
    a cell) that the zone covers. Returns ``(zone, cell, fraction)``, one
    row per overlap.
    """
    which = np.asarray(which, np.int64)
    step = grid.step
    r0, r1 = grid.rows(zones.ymin[which]), grid.rows(zones.ymax[which])
    # unwrapped columns, so each box is in its geometry's own longitudes
    c0 = np.floor((zones.xmin[which] - grid.lon_edge0) / step).astype(np.int64)
    c1 = np.floor((zones.xmax[which] - grid.lon_edge0) / step).astype(np.int64)
    ncol = c1 - c0 + 1
    count = (r1 - r0 + 1) * ncol
    zone = np.repeat(which, count)
    k = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    ncol = np.repeat(ncol, count)
    row = np.repeat(r0, count) + k // ncol
    col = np.repeat(c0, count) + k % ncol
    west = grid.lon_edge0 + col * step
    south = np.maximum(grid.lat_edge0 + row * step, -90.0)
    north = np.minimum(grid.lat_edge0 + (row + 1) * step, 90.0)
    geom = zones.geometry[zone]
    shapely.prepare(zones.geometry[which])
    area = _overlap_area(geom, west, south, north, step)
    # a box past 180 (a zone across the antimeridian) is west of -180 too
    east = west + step > 180
    area[east] += _overlap_area(
        geom[east], west[east] - 360, south[east], north[east], step
    )
    fraction = area / (step * (north - south))
    keep = fraction > 0
    cell = row * grid.nlon + np.mod(col, grid.nlon)
    return zone[keep], cell[keep], np.minimum(fraction[keep], 1.0)


def _overlap_area(geom, west, south, north, step) -> np.ndarray:
    """Area of each geometry inside its box (prepared geometries: boxes
    fully inside or outside need no intersection)."""
    boxes = shapely.box(west, south, west + step, north)
    area = np.where(
        shapely.contains_properly(geom, boxes), shapely.area(boxes), 0.0
    )
    edge = (area == 0) & shapely.intersects(geom, boxes)
    area[edge] = shapely.area(shapely.intersection(geom[edge], boxes[edge]))
    return area


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
