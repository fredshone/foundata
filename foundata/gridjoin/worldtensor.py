"""WorldTensor: 0.25 degree layers joined to zones by their point.

**Which year.** A zone with a survey year takes each variable from the
year on disk nearest to it (the earlier of two equally near), so a 2023
survey gets ``hdi`` 2015 (its last year) and ``t2m_mean`` 2023. Static
layers have one year, their snapshot's, where the file gives one. The
mapping of survey years to years used is in the output's metadata.

**Preprocess.** Each (variable, year) layer becomes a float32 ``(721,
1440)`` ``.npy`` in the cache, rows south to north, columns from lon 0
east (the files' own layout, checked on read). Only the years asked for
are cached, and each is redone only when its source file changes.

**Join.** Space and time are separate: ``cells`` gives each zone the
cell its point is in, found by index arithmetic (see ``grid``), and
``values`` reads each (variable, year) layer at those cells, once per
distinct cell. Where that cell is NaN (ocean, or masked, which can
differ by year) it takes the nearest non-NaN cell whose centre is within
``max_fallback_km`` of its point, else null. Masks differ by layer:
``gdp_per_capita``'s is eroded at coasts (Nice's nearest valid cell is
35 km away, Montevideo's 43 km), hence a default of 50 km, about two
cells.
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
STATIC = "static"  # the layer key of a static variable

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
    """The columns with a value per variable (as opposed to diagnostics),
    as joined to attributes."""
    return [PREFIX + v for v in VARIABLES]


DEFAULT_MAX_FALLBACK_KM = 50.0


def missing_sources(
    world: Path, variables: dict[str, str] | None = None
) -> list[str]:
    """Problems finding each variable's source under *world* (none: [])."""
    problems = []
    for var, rel in (variables or VARIABLES).items():
        try:
            available(world / rel)
        except FileNotFoundError as e:
            problems.append(f"WorldTensor {var}: {e}")
    return problems


def available(source: Path) -> dict[int, Path] | None:
    """A variable's files by year, or None if it is a single static file."""
    if source.is_file():
        return None
    if not source.is_dir():
        raise FileNotFoundError(f"{source} not found")
    years = {
        int(f.stem): f
        for f in source.glob("*.nc")
        if f.stem.isdigit() and not f.name.startswith(".")
    }
    if not years:
        raise FileNotFoundError(f"no <year>.nc files in {source}")
    return dict(sorted(years.items()))


def nearest_year(years: list[int], year: int) -> int:
    """The year in *years* nearest to *year* (the earlier, on a tie)."""
    return min(years, key=lambda y: (abs(y - year), y))


def preprocess(
    world: Path,
    cache: Path,
    years: list[int] | None = None,
    variables: dict[str, str] | None = None,
    force: bool = False,
) -> dict:
    """Write each variable's layers for *years* (each mapped to its
    nearest year on disk; None: the most recent) to ``cache/worldtensor``,
    and return the manifest. Layers already cached for other years are
    kept; those of variables no longer in *variables* are removed.

    The manifest gives, per variable, its ``source``, ``available`` (its years on disk,
    empty if static), ``static``, ``year`` (a static layer's snapshot
    year, where the file gives one), ``units``, ``long_name``, and
    ``layers``: per year (or ``"static"``) its file, source and NaN
    fraction.
    """
    variables = variables or VARIABLES
    out = cache / "worldtensor"
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )
    for var in list(manifest):
        entry = manifest[var]
        if var in variables and "layers" in entry:
            continue
        # no longer used, or cached by a version with one layer each
        layers = entry.get("layers", {"": {"file": entry.get("file")}})
        for layer in layers.values():
            if layer.get("file"):
                (out / layer["file"]).unlink(missing_ok=True)
        del manifest[var]
        if var not in variables:
            logger.info(f"worldtensor {var}: no longer used, removed")
    for var, rel in variables.items():
        files = available(world / rel)
        if files is None:
            wanted = {STATIC: world / rel}
        else:
            need = [max(files)] if years is None else years
            wanted = {
                str(y): files[y]
                for y in {nearest_year(list(files), n) for n in need}
            }
        entry = manifest.setdefault(var, {"layers": {}})
        entry["source"] = str(world / rel)
        entry["available"] = [] if files is None else list(files)
        entry["static"] = files is None
        for key, src in sorted(wanted.items()):
            _preprocess_layer(out, var, key, src, entry, force)
    if (
        not manifest_path.exists()
        or json.loads(manifest_path.read_text()) != manifest
    ):
        _write_manifest(manifest_path, manifest)
    return manifest


def _preprocess_layer(
    out: Path, var: str, key: str, src: Path, entry: dict, force: bool
) -> None:
    stat = src.stat()
    stamp = {"source": str(src), "mtime": stat.st_mtime, "bytes": stat.st_size}
    layer = entry["layers"].get(key, {})
    npy = out / (f"{var}.npy" if key == STATIC else f"{var}_{key}.npy")
    if (
        not force
        and npy.exists()
        and all(layer.get(k) == v for k, v in stamp.items())
    ):
        logger.debug(f"worldtensor {var} {key}: up to date")
        return
    data, attrs = _read_layer(src, var)
    _save_npy(npy, data)
    entry["layers"][key] = {
        **stamp,
        "file": npy.name,
        "nan_fraction": round(float(np.isnan(data).mean()), 4),
    }
    entry["units"] = attrs.get("units")
    entry["long_name"] = attrs.get("long_name")
    if key == STATIC:
        entry["year"] = attrs.get("reference_year")
    logger.info(f"worldtensor {var} {key} -> {npy}")


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


def cells(zones: Zones) -> dict[str, np.ndarray]:
    """Each zone's cell, which does not depend on the year.

    Columns: ``wt_cell`` (row * 1440 + col, null for an unlocated zone);
    ``wt_spans_cells`` (the footprint box crosses a cell edge, so the
    point's cell is an approximation).
    """
    g = WORLDTENSOR
    ok = zones.located
    cell = np.full(len(zones), -1, np.int64)
    cell[ok] = g.cell(zones.lon[ok], zones.lat[ok])
    spans = np.zeros(len(zones), bool)
    spans[ok] = g.spans_cells(
        zones.xmin[ok], zones.ymin[ok], zones.xmax[ok], zones.ymax[ok]
    )
    return {
        PREFIX + "cell": np.ma.masked_array(cell.astype(np.int32), ~ok),
        PREFIX + "spans_cells": spans,
    }


def values(
    lon: np.ndarray,
    lat: np.ndarray,
    year: np.ndarray,
    cache: Path,
    variables: list[str] | None = None,
    max_fallback_km: float = DEFAULT_MAX_FALLBACK_KM,
) -> tuple[dict[str, np.ndarray], dict]:
    """Each variable's value at each point (``lon``, ``lat``) in its
    survey *year* (NaN for any of the three: no value), from the layers
    preprocessed into *cache*.

    Columns, per variable: ``wt_<var>`` (float32, NaN where none).
    Metadata: per variable, its
    units, source and the year used for each survey year, and the number
    of points on NaN cells and how many the fallback filled.
    """
    out = cache / "worldtensor"
    manifest = json.loads((out / "manifest.json").read_text())
    variables = list(variables or VARIABLES)
    missing = [v for v in variables if v not in manifest]
    if missing:
        raise KeyError(f"not preprocessed: {missing} (run preprocess first)")
    g = WORLDTENSOR
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    year = np.asarray(year, float)
    n = lon.size
    ok = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(year)
    cell = np.full(n, -1, np.int64)
    cell[ok] = g.cell(lon[ok], lat[ok])
    survey_years = sorted({int(y) for y in year[ok]})
    columns, used = {}, {}
    for var in variables:
        entry = manifest[var]
        if entry["static"]:
            years = {y: STATIC for y in survey_years}
        else:
            years = {
                y: str(nearest_year(entry["available"], y))
                for y in survey_years
            }
        absent = sorted(set(years.values()) - set(entry["layers"]))
        if absent:
            raise KeyError(
                f"{var} {absent} not preprocessed (run preprocess for "
                f"years {survey_years})"
            )
        vals = np.full(n, np.nan, np.float32)
        holes_n = filled_n = 0
        for key in sorted(set(years.values())):
            surveys = [y for y, k in years.items() if k == key]
            sel = np.flatnonzero(ok & np.isin(year, surveys))
            layer = np.load(
                out / entry["layers"][key]["file"], mmap_mode="r"
            ).ravel()
            cs, inverse = np.unique(cell[sel], return_inverse=True)
            vals[sel] = np.asarray(layer[cs])[inverse]
            holes = sel[np.isnan(vals[sel])]
            if holes.size:
                valid = ~np.isnan(layer)
                near, _ = g.nearest_valid(
                    lon[holes], lat[holes], valid, max_fallback_km
                )
                found = near >= 0
                vals[holes[found]] = layer[near[found]]
                holes_n += holes.size
                filled_n += int(found.sum())
        columns[PREFIX + var] = vals
        if holes_n:
            logger.debug(
                f"worldtensor {var}: {holes_n} points on NaN cells, "
                f"{filled_n} filled within {max_fallback_km} km"
            )
        used[var] = {
            "units": entry.get("units"),
            "source": entry["source"],
            "years": {
                str(y): (entry.get("year") if k == STATIC else int(k))
                for y, k in years.items()
            },
            "on_nan_cells": holes_n,
            "filled": filled_n,
        }
    return columns, {"variables": used, "max_fallback_km": max_fallback_km}
