"""Regular global lon/lat grids: cell indices by arithmetic, no geometry.

A grid is ``nlat`` rows by ``nlon`` columns of ``step`` degrees. Row 0 is
the southernmost, column 0 starts at ``lon_edge0``. *Point-registered*
grids (WorldTensor, ERA5 style) have cell centres on the coordinate
values, so their first and last rows are half cells at the poles.

Indexing rules, for every grid:

- a coordinate on a cell edge belongs to the cell to its north (east),
- latitudes are clipped to the first and last rows (so lat = -90 and 90
  are in the grid),
- longitudes wrap: lon = 180 and lon = -180 are the same meridian and
  get the same column.
"""

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_KM = 6371.0088
KM_PER_DEG = EARTH_RADIUS_KM * np.pi / 180


@dataclass(frozen=True)
class Grid:
    """A global regular lon/lat grid."""

    nlat: int
    nlon: int
    step: float
    lat_edge0: float  # southern edge of row 0 (may be below -90)
    lon_edge0: float  # western edge of column 0

    @classmethod
    def point_registered(
        cls, lat0: float, lon0: float, step: float, nlat: int, nlon: int
    ):
        """A grid whose cell centres are at lat0 + i*step, lon0 + j*step."""
        return cls(nlat, nlon, step, lat0 - step / 2, lon0 - step / 2)

    @property
    def size(self) -> int:
        return self.nlat * self.nlon

    def rows(self, lat: np.ndarray) -> np.ndarray:
        r = np.floor((np.asarray(lat, float) - self.lat_edge0) / self.step)
        return np.clip(r, 0, self.nlat - 1).astype(np.int64)

    def cols(self, lon: np.ndarray) -> np.ndarray:
        c = np.floor((np.asarray(lon, float) - self.lon_edge0) / self.step)
        return np.mod(c, self.nlon).astype(np.int64)

    def cell(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Cell id ``row * nlon + col`` of each point."""
        return self.rows(lat) * self.nlon + self.cols(lon)

    def centre(self, cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lon (in [-180, 180)), lat of each cell's centre."""
        row, col = np.divmod(np.asarray(cell, np.int64), self.nlon)
        lon = self.lon_edge0 + (col + 0.5) * self.step
        lat = self.lat_edge0 + (row + 0.5) * self.step
        return np.mod(lon + 180, 360) - 180, lat

    def spans_cells(
        self,
        xmin: np.ndarray,
        ymin: np.ndarray,
        xmax: np.ndarray,
        ymax: np.ndarray,
    ) -> np.ndarray:
        """Whether each box crosses a cell edge (is in more than one cell).

        ``xmax`` may exceed 180 for a box crossing the antimeridian.
        """
        lat_cross = self.rows(ymin) != self.rows(ymax)
        width = np.asarray(xmax, float) - np.asarray(xmin, float)
        lon_cross = (width >= self.step) | (self.cols(xmin) != self.cols(xmax))
        return lat_cross | lon_cross

    def nearest_valid(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        valid: np.ndarray,
        max_km: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """For each point, the nearest cell whose ``valid`` flag is set.

        Distance is great-circle from the point to the cell centre, and
        only cells within *max_km* count. Returns the cell id (-1 where
        none) and its distance in km (NaN where none).
        """
        lon = np.asarray(lon, float)
        lat = np.asarray(lat, float)
        valid = np.asarray(valid, bool).ravel()
        n = lon.size
        best = np.full(n, -1, np.int64)
        best_km = np.full(n, np.inf)
        if n == 0:
            return best, np.full(0, np.nan)
        row0, col0 = self.rows(lat), self.cols(lon)
        dlat = max_km / KM_PER_DEG
        kr = int(np.ceil(dlat / self.step)) + 1
        # widest column window needed, from the most poleward latitude
        coslat = np.cos(np.radians(np.minimum(np.abs(lat).max() + dlat, 90.0)))
        kc = int(np.ceil(dlat / max(coslat, 1e-9) / self.step)) + 1
        kc = min(kc, self.nlon // 2)
        for dr in range(-kr, kr + 1):
            r = row0 + dr
            in_grid = (r >= 0) & (r < self.nlat)
            r = np.clip(r, 0, self.nlat - 1)
            for dc in range(-kc, kc + 1):
                cell = r * self.nlon + np.mod(col0 + dc, self.nlon)
                ok = in_grid & valid[cell]
                if not ok.any():
                    continue
                clon, clat = self.centre(cell[ok])
                km = haversine_km(lon[ok], lat[ok], clon, clat)
                idx = np.flatnonzero(ok)
                better = (km < best_km[idx]) & (km <= max_km)
                best[idx[better]] = cell[ok][better]
                best_km[idx[better]] = km[better]
        best_km[best < 0] = np.nan
        return best, best_km


def haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# WorldTensor: 721 x 1440 cell centres at -90..90, 0..359.75
WORLDTENSOR = Grid.point_registered(-90.0, 0.0, 0.25, 721, 1440)
