"""WorldTensor: 0.25 degree layers joined to zones by their point.

**Which year.** Each variable is taken from its *most recent* year on
disk (``gdp_per_capita`` 2022, ``hdi`` 2015, ``tp_sum`` 2025, ...), not
the zone's survey year; the year used is in the output's metadata and
the cache manifest.

**Preprocess.** Each variable's layer becomes a float32 ``(721, 1440)``
``.npy`` in the cache, rows south to north, columns from lon 0 east
(the files' own layout, checked on read). It is redone only when the
source file changes.

**Join.** A zone takes the value of the cell its point is in, found by
index arithmetic (see ``grid``), looked up once per distinct cell. Where
that cell is NaN (ocean, or masked) it takes the nearest non-NaN cell
whose centre is within ``max_fallback_km`` of its point, else null.
Masks differ by layer: ``gdp_per_capita``'s is eroded at coasts (Nice's
nearest valid cell is 35 km away, Montevideo's 43 km), hence a default
of 50 km, about two cells.
"""

import json
import os
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger

from foundata.gridjoin.grid import WORLDTENSOR
from foundata.gridjoin.zones import Zones

PREFIX = "wt_"

# variable -> its folder of <year>.nc files, or a single static file,
# under the WORLD data root
VARIABLES = {
    "gdp_per_capita": "worldtensor_human_systems/human_systems/gdp_per_capita",
    "ghsl_built_volume": "worldtensor_human_systems/human_systems/ghsl_built_volume",
    "gini": "worldtensor_human_systems/human_systems/gini",
    "hdi": "worldtensor_human_systems/human_systems/hdi",
    "ntl_harmonized": "worldtensor_human_systems/human_systems/ntl_harmonized",
    "population_density": "worldtensor_human_systems/human_systems/population_density",
    "tp_sum": "worldtensor_climate/climate/tp_sum",
    "t2m_mean": "worldtensor_climate/climate/t2m_mean",
    "skt_max": "worldtensor_climate/climate/skt_max",
    "elevation_mean": "worldtensor_static/static/topography/elevation_mean.nc",
    "travel_time_to_cities": "worldtensor_static/static/travel_time_to_cities/travel_time_to_cities.nc",
}


def value_columns() -> list[str]:
    """The columns with a value per variable (as opposed to diagnostics)."""
    return [PREFIX + v for v in VARIABLES]


DEFAULT_MAX_FALLBACK_KM = 50.0


def missing_sources(
    world: Path, variables: dict[str, str] | None = None
) -> list[str]:
    """Problems finding each variable's source under *world* (none: [])."""
    problems = []
    for var, rel in (variables or VARIABLES).items():
        try:
            latest(world / rel)
        except FileNotFoundError as e:
            problems.append(f"WorldTensor {var}: {e}")
    return problems


def latest(source: Path) -> tuple[Path, int | None]:
    """The file to use for a variable, and its year (None if static)."""
    if source.is_file():
        return source, None
    if not source.is_dir():
        raise FileNotFoundError(f"{source} not found")
    years = {
        int(f.stem): f
        for f in source.glob("*.nc")
        if f.stem.isdigit() and not f.name.startswith(".")
    }
    if not years:
        raise FileNotFoundError(f"no <year>.nc files in {source}")
    year = max(years)
    return years[year], year


def preprocess(
    world: Path,
    cache: Path,
    variables: dict[str, str] | None = None,
    force: bool = False,
) -> dict:
    """Write each variable's layer to ``cache/worldtensor``; return the
    manifest (variable -> file, year, units, source). Layers of variables
    no longer in *variables* are removed."""
    variables = variables or VARIABLES
    out = cache / "worldtensor"
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )
    for var in set(manifest) - set(variables):
        (out / manifest.pop(var)["file"]).unlink(missing_ok=True)
        logger.info(f"worldtensor {var}: no longer used, removed")
    for var, rel in variables.items():
        src, year = latest(world / rel)
        stat = src.stat()
        stamp = {
            "source": str(src),
            "mtime": stat.st_mtime,
            "bytes": stat.st_size,
        }
        entry = manifest.get(var, {})
        npy = out / f"{var}.npy"
        if (
            not force
            and npy.exists()
            and all(entry.get(k) == v for k, v in stamp.items())
        ):
            logger.debug(f"worldtensor {var}: up to date ({year})")
            continue
        layer, attrs = _read_layer(src, var)
        _save_npy(npy, layer)
        manifest[var] = {
            **stamp,
            "file": npy.name,
            # a static layer's snapshot year, where the file gives one
            "year": year or attrs.get("reference_year"),
            "static": year is None,
            "units": attrs.get("units"),
            "long_name": attrs.get("long_name"),
            "nan_fraction": round(float(np.isnan(layer).mean()), 4),
        }
        logger.info(f"worldtensor {var}: {year or 'static'} -> {npy}")
    if (
        not manifest_path.exists()
        or json.loads(manifest_path.read_text()) != manifest
    ):
        _write_manifest(manifest_path, manifest)
    return manifest


def _write_manifest(manifest_path: Path, manifest: dict) -> None:
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, manifest_path)


def _read_layer(path: Path, var: str) -> tuple[np.ndarray, dict]:
    with xr.open_dataset(path) as ds:
        da = ds[var].squeeze(drop=True)
        if da.dims != ("lat", "lon"):
            da = da.transpose("lat", "lon")
        lat, lon = da["lat"].values, da["lon"].values
        if lat[0] > lat[-1]:
            da, lat = da.isel(lat=slice(None, None, -1)), lat[::-1]
        g = WORLDTENSOR
        expect_lat = g.lat_edge0 + (np.arange(g.nlat) + 0.5) * g.step
        expect_lon = g.lon_edge0 + (np.arange(g.nlon) + 0.5) * g.step
        if not (np.allclose(lat, expect_lat) and np.allclose(lon, expect_lon)):
            raise ValueError(f"{path}: not the WorldTensor 0.25 degree grid")
        ref = ds.attrs.get("reference_year")
        attrs = {
            "reference_year": None if ref is None else int(ref),
            **da.attrs,
        }
        return da.values.astype(np.float32), attrs


def _save_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def join(
    zones: Zones,
    cache: Path,
    variables: list[str] | None = None,
    max_fallback_km: float = DEFAULT_MAX_FALLBACK_KM,
) -> tuple[dict[str, np.ndarray], dict]:
    """New columns for *zones*, and metadata (the manifest of what was used).

    Columns: ``wt_<var>`` per variable; ``wt_<var>_fallback_km`` for
    each variable whose layer has NaNs (distance to the cell used: 0 for
    the zone's own, null if none in range); ``wt_cell`` (row * 1440 +
    col, null for an unlocated zone); ``wt_spans_cells`` (the footprint box crosses a cell edge, so
    the point's cell is an approximation).
    """
    out = cache / "worldtensor"
    manifest = json.loads((out / "manifest.json").read_text())
    variables = list(variables or VARIABLES)
    missing = [v for v in variables if v not in manifest]
    if missing:
        raise KeyError(f"not preprocessed: {missing} (run preprocess first)")
    g = WORLDTENSOR
    n = len(zones)
    ok = zones.located
    cell = np.full(n, -1, np.int64)
    cell[ok] = g.cell(zones.lon[ok], zones.lat[ok])
    cells, inverse = np.unique(cell[ok], return_inverse=True)
    columns = {}
    for var in variables:
        layer = np.load(out / manifest[var]["file"], mmap_mode="r").ravel()
        valid = ~np.isnan(layer)
        values = np.full(n, np.nan, np.float32)
        values[ok] = np.asarray(layer[cells])[inverse]
        columns[PREFIX + var] = values
        if valid.all():
            continue
        fallback_km = np.where(ok, 0.0, np.nan).astype(np.float32)
        holes = np.flatnonzero(ok & np.isnan(values))
        if holes.size:
            near, km = g.nearest_valid(
                zones.lon[holes], zones.lat[holes], valid, max_fallback_km
            )
            found = near >= 0
            values[holes[found]] = layer[near[found]]
            fallback_km[holes] = km
            logger.debug(
                f"worldtensor {var}: {holes.size} zones on NaN cells, "
                f"{found.sum()} filled within {max_fallback_km} km"
            )
        columns[PREFIX + var + "_fallback_km"] = fallback_km
    columns[PREFIX + "cell"] = np.ma.masked_array(cell.astype(np.int32), ~ok)
    spans = np.zeros(n, bool)
    spans[ok] = g.spans_cells(
        zones.xmin[ok], zones.ymin[ok], zones.xmax[ok], zones.ymax[ok]
    )
    columns[PREFIX + "spans_cells"] = spans
    used = {
        v: {k: manifest[v][k] for k in ("year", "units", "source")}
        for v in variables
    }
    return columns, {"variables": used, "max_fallback_km": max_fallback_km}
