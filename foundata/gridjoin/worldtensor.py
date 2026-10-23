"""WorldTensor: 0.25 degree layers joined to the cells zones cover.

**Config.** ``configs/core/gridjoin.yaml`` lists the variables, each
with its source and its weight (``Config``, ``load_config``).

**Which year.** A zone with a survey year takes each variable from the
year on disk nearest to it (the earlier of two equally near), so a 2023
survey gets ``hdi`` 2015 (its last year) and ``t2m_mean`` 2023. Static
layers have one year, their snapshot's, where the file gives one. The
mapping of survey years to years used is in the output's metadata.

**Preprocess.** Each (variable, year) layer becomes a float32 ``(721,
1440)`` ``.npy`` in the cache, rows south to north, columns from lon 0
east (the files' own layout, checked on read). Only the years asked for
are cached, and each is redone only when its source file changes.

**Join.** Space and time are separate: ``footprints`` gives each zone
the cells it covers and the fraction of each (a point, or a polygon
inside one cell: its point's cell), and ``values`` reads each (variable,
year) layer at those cells and averages them, by the fraction (weight
``area``) or by a weight layer in the survey's nearest year times the
fraction (``population_count``: the people in each overlap; where there
are none, by area). Cells that are
NaN (ocean, or masked, which can differ by year) are left out; with
none left a zone takes the nearest non-NaN cell whose centre is within
``max_fallback_km`` of its point, else null. Masks differ by layer:
``gdp_per_capita``'s is eroded at coasts (Nice's nearest valid cell is
35 km away, Montevideo's 43 km), hence a default of 50 km, about two
cells.
"""

import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from loguru import logger

from foundata.gridjoin import zones as zones_mod
from foundata.gridjoin.grid import WORLDTENSOR
from foundata.gridjoin.zones import Zones

PREFIX = "wt_"
STATIC = "static"  # the layer key of a static variable

CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "core" / "gridjoin.yaml"
)
AREA = "area"  # the weight by the share of each cell a zone covers


@dataclass(frozen=True)
class Config:
    """The variables to join and how each averages a zone's cells."""

    # variable -> its folder of <year>.nc files, or a single static file,
    # under the WORLD data root
    sources: dict[str, str]
    weight: dict[str, str]  # variable -> AREA or a weight layer
    weights: dict[str, str]  # weight layer -> its source

    @property
    def layers(self) -> dict[str, str]:
        """Every layer to preprocess: the variables and their weights."""
        used = sorted(set(self.weight.values()) - {AREA})
        return {**self.sources, **{w: self.weights[w] for w in used}}


def load_config(path: str | Path = CONFIG) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    weights = raw.get("weights") or {}
    sources, weight, problems = {}, {}, []
    if AREA in weights:
        problems.append(f"{AREA!r} is built in, not a weight layer")
    for var, entry in (raw.get("variables") or {}).items():
        if not isinstance(entry, dict) or set(entry) != {"source", "weight"}:
            problems.append(f"{var}: needs exactly source and weight")
        elif entry["weight"] not in {AREA, *weights}:
            problems.append(
                f"{var}: weight {entry['weight']!r} is not {AREA!r} or one "
                f"of weights ({', '.join(weights)})"
            )
        elif var in weights and weights[var] != entry["source"]:
            problems.append(f"{var}: a weight with another source")
        else:
            sources[var] = entry["source"]
            weight[var] = entry["weight"]
    if not sources:
        problems.append("no variables")
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return Config(sources, weight, weights)


@functools.cache
def default_config() -> Config:
    return load_config()


def value_columns(config: Config | None = None) -> list[str]:
    """The columns with a value per variable (as opposed to diagnostics),
    as joined to attributes."""
    return [PREFIX + v for v in (config or default_config()).sources]


DEFAULT_MAX_FALLBACK_KM = 50.0


def missing_sources(world: Path, config: Config | None = None) -> list[str]:
    """Problems finding each layer's source under *world* (none: [])."""
    problems = []
    for var, rel in (config or default_config()).layers.items():
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
    config: Config | None = None,
    force: bool = False,
) -> dict:
    """Write each of *config*'s layers (variables and weights) for *years*
    (each mapped to its nearest year on disk; None: the most recent) to
    ``cache/worldtensor``, and return the manifest. Layers already cached
    for other years are kept; those no longer in *config* are removed.

    The manifest gives, per layer, its ``source``, ``available`` (its years on disk,
    empty if static), ``static``, ``year`` (a static layer's snapshot
    year, where the file gives one), ``units``, ``long_name``, and
    ``layers``: per year (or ``"static"``) its file, source and NaN
    fraction.
    """
    variables = (config or default_config()).layers
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

    Columns: ``wt_cell`` (row * 1440 + col of its point, null for an
    unlocated zone); ``wt_spans_cells`` (the footprint box crosses a cell
    edge, so its values average several cells).
    """
    g = WORLDTENSOR
    ok = zones.located
    cell = np.full(len(zones), -1, np.int64)
    cell[ok] = g.cell(zones.lon[ok], zones.lat[ok])
    return {
        PREFIX + "cell": np.ma.masked_array(cell.astype(np.int32), ~ok),
        PREFIX + "spans_cells": _spans(zones),
    }


def footprints(zones: Zones) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The cells each located zone covers and the fraction of each cell it
    covers, as ``(zone, cell, fraction)`` rows. A zone in one cell (any
    point) covers its point's cell, with fraction 1.
    """
    g = WORLDTENSOR
    zone, cell, fraction = zones_mod.cell_fractions(
        zones, g, np.flatnonzero(_spans(zones))
    )
    # and any polygon whose geometry meets no cell keeps its point's
    single = zones.located
    single[zone] = False
    s = np.flatnonzero(single)
    return (
        np.concatenate([zone, s]),
        np.concatenate([cell, g.cell(zones.lon[s], zones.lat[s])]),
        np.concatenate([fraction, np.ones(s.size)]),
    )


def _spans(zones: Zones) -> np.ndarray:
    ok = zones.located
    spans = np.zeros(len(zones), bool)
    spans[ok] = WORLDTENSOR.spans_cells(
        zones.xmin[ok], zones.ymin[ok], zones.xmax[ok], zones.ymax[ok]
    )
    return spans


def values(
    lon: np.ndarray,
    lat: np.ndarray,
    year: np.ndarray,
    cache: Path,
    config: Config | None = None,
    max_fallback_km: float = DEFAULT_MAX_FALLBACK_KM,
    footprint: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Each variable's value at each point (``lon``, ``lat``) in its
    survey *year* (NaN for any of the three: no value), from the layers
    preprocessed into *cache*.

    *footprint* gives the cells each point stands for, as ``(point, cell,
    fraction)`` rows (``footprints``, indexed by point); a point's value
    is the mean over its cells weighted by fraction (weight ``area``), or
    by its variable's weight layer times fraction, falling back to
    fraction where those sum to 0. None: each point's own cell.

    Columns, per variable: ``wt_<var>`` (float32, NaN where none).
    Metadata: per variable, its units, source, weight and the year used
    for each survey year, and the number of points with no non-NaN cell
    and how many the fallback filled; and per weight layer, its years.
    """
    config = config or default_config()
    out = cache / "worldtensor"
    manifest = json.loads((out / "manifest.json").read_text())
    missing = [v for v in config.layers if v not in manifest]
    if missing:
        raise KeyError(f"not preprocessed: {missing} (run preprocess first)")
    g = WORLDTENSOR
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    year = np.asarray(year, float)
    n = lon.size
    ok = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(year)
    if footprint is None:
        point = np.flatnonzero(ok)
        cell, fraction = g.cell(lon[ok], lat[ok]), np.ones(point.size)
    else:
        point, cell, fraction = (np.asarray(a) for a in footprint)
        keep = ok[point]
        point, cell, fraction = point[keep], cell[keep], fraction[keep]
    survey_years = sorted({int(y) for y in year[ok]})
    used_weights = sorted(set(config.weight.values()) - {AREA})
    weight_years = {
        w: _layer_years(manifest, w, survey_years) for w in used_weights
    }
    area_years = {y: AREA for y in survey_years}
    weight_layers = {}  # (weight, key) -> layer, loaded on first use
    columns, used = {}, {}
    for var, wvar in config.weight.items():
        entry = manifest[var]
        years = _layer_years(manifest, var, survey_years)
        wyears = area_years if wvar == AREA else weight_years[wvar]
        vals = np.full(n, np.nan, np.float32)
        holes_n = filled_n = 0
        for key in sorted(set(years.values())):
            layer = _load(out, manifest, var, key)
            valid = ~np.isnan(layer)
            for wkey in sorted(
                {wyears[y] for y, k in years.items() if k == key}
            ):
                surveys = [
                    y
                    for y, k in years.items()
                    if k == key and wyears[y] == wkey
                ]
                sel = ok & np.isin(year, surveys)
                rows = sel[point]
                c = cell[rows]
                f = fraction[rows]
                w = f
                if wkey != AREA:
                    if (wvar, wkey) not in weight_layers:
                        weight_layers[wvar, wkey] = np.nan_to_num(
                            _load(out, manifest, wvar, wkey)
                        )
                    w = f * weight_layers[wvar, wkey][c]
                mean = _weighted_mean(n, point[rows], layer[c], w, f)
                vals[sel] = mean[sel]
                holes = np.flatnonzero(sel & np.isnan(vals))
                if holes.size:
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
                f"worldtensor {var}: {holes_n} points with no non-NaN cell, "
                f"{filled_n} filled within {max_fallback_km} km"
            )
        used[var] = {
            "units": entry.get("units"),
            "source": entry["source"],
            "weight": wvar,
            "years": _years_used(entry, years),
            "on_nan_cells": holes_n,
            "filled": filled_n,
        }
    return columns, {
        "variables": used,
        "weights": {
            w: {"years": _years_used(manifest[w], weight_years[w])}
            for w in used_weights
        },
        "max_fallback_km": max_fallback_km,
    }


def _layer_years(manifest: dict, var: str, survey_years: list[int]) -> dict:
    """The layer key of *var* for each survey year (checked cached)."""
    entry = manifest[var]
    if entry["static"]:
        years = {y: STATIC for y in survey_years}
    else:
        years = {
            y: str(nearest_year(entry["available"], y)) for y in survey_years
        }
    absent = sorted(set(years.values()) - set(entry["layers"]))
    if absent:
        raise KeyError(
            f"{var} {absent} not preprocessed (run preprocess for "
            f"years {survey_years})"
        )
    return years


def _years_used(entry: dict, years: dict) -> dict:
    return {
        str(y): (entry.get("year") if k == STATIC else int(k))
        for y, k in years.items()
    }


def _load(out: Path, manifest: dict, var: str, key: str) -> np.ndarray:
    return np.load(out / manifest[var]["layers"][key]["file"]).ravel()


def _weighted_mean(
    n: int,
    point: np.ndarray,
    value: np.ndarray,
    weight: np.ndarray,
    fraction: np.ndarray,
) -> np.ndarray:
    """Per point (of *n*), the mean of its non-NaN *value*s by *weight*,
    or by *fraction* where their weights sum to 0; NaN where it has none
    (float64)."""
    good = ~np.isnan(value)
    point, value = point[good], value[good].astype(float)
    weight, fraction = weight[good], fraction[good]
    by_weight = np.bincount(point, weight, minlength=n)
    by_area = np.bincount(point, fraction, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            by_weight > 0,
            np.bincount(point, weight * value, minlength=n) / by_weight,
            np.bincount(point, fraction * value, minlength=n) / by_area,
        )
