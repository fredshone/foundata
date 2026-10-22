import json

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely
import xarray as xr

from foundata import location, run, utils
from foundata.gridjoin import join, worldtensor, zones
from foundata.gridjoin.grid import WORLDTENSOR

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def zones_table(rows: list[tuple]) -> pa.Table:
    """Zones from ``(lon, lat, scale_m, geometry)`` rows (geometry may be
    None); a point's geometry defaults to the point."""
    lon, lat, scale, wkb = [], [], [], []
    for x, y, s, g in rows:
        if g is None and x is not None:
            g = shapely.Point(x, y)
        lon.append(x)
        lat.append(y)
        scale.append(s)
        wkb.append(None if g is None else shapely.to_wkb(g))
    n = len(rows)
    return pa.table(
        {
            "location_id": pa.array(np.arange(n, dtype=np.int32)),
            "lon": pa.array(lon, pa.float64()),
            "lat": pa.array(lat, pa.float64()),
            "scale_m": pa.array(scale, pa.float64()),
            "geometry": pa.array(wkb, pa.large_binary()),
        }
    )


# ---------------------------------------------------------------------------
# grid indexers
# ---------------------------------------------------------------------------


def test_every_cell_centre_maps_back_to_its_cell():
    g = WORLDTENSOR
    cells = np.arange(g.size)
    lon, lat = g.centre(cells)
    assert (lon >= -180).all() and (lon < 180).all()
    np.testing.assert_array_equal(g.cell(lon, lat), cells)
    # and in the 0..360 convention
    np.testing.assert_array_equal(g.cell(np.mod(lon, 360), lat), cells)


def test_worldtensor_centres_are_its_coordinates():
    lon, lat = WORLDTENSOR.centre(np.arange(WORLDTENSOR.size))
    lat = lat.reshape(721, 1440)
    lon = np.mod(lon.reshape(721, 1440), 360)
    np.testing.assert_allclose(lat[:, 0], np.linspace(-90, 90, 721))
    np.testing.assert_allclose(lon[0], np.arange(1440) * 0.25)


def test_points_on_cell_edges_go_north_and_east():
    # edges are halfway between centres, at 0.125 + k/4
    assert WORLDTENSOR.rows(np.array([-0.125]))[0] == 360
    assert WORLDTENSOR.rows(np.array([0.125]))[0] == 361
    assert WORLDTENSOR.cols(np.array([0.125]))[0] == 1
    assert WORLDTENSOR.cols(np.array([-0.125]))[0] == 0


def test_poles_and_antimeridian_stay_in_the_grid():
    g = WORLDTENSOR
    assert g.rows(np.array([-90.0]))[0] == 0
    assert g.rows(np.array([90.0]))[0] == g.nlat - 1
    assert g.rows(np.array([95.0]))[0] == g.nlat - 1  # clipped
    c = g.cols(np.array([-180.0, 180.0, 540.0]))
    assert c[0] == c[1] == c[2]
    assert 0 <= c[0] < g.nlon


def box_spans(xmin, ymin, xmax, ymax) -> bool:
    return bool(
        WORLDTENSOR.spans_cells(
            *(np.array([v]) for v in (xmin, ymin, xmax, ymax))
        )[0]
    )


def test_spans_cells():
    # the cell centred on (10, 10) runs from 9.875 to 10.125
    assert not box_spans(9.95, 9.95, 10.05, 10.05)
    assert box_spans(10.1, 9.95, 10.2, 10.05)
    assert box_spans(9.95, 10.124, 10.05, 10.126)  # on an edge
    # the cell centred on 180 holds both sides of the antimeridian
    assert not box_spans(179.9, 10.0, 180.1, 10.05)
    assert box_spans(179.8, 10.0, 180.2, 10.05)
    assert box_spans(10.0, 10.0, 10.05 + 360, 10.05)  # all the way round


def test_nearest_valid():
    g = WORLDTENSOR
    valid = np.zeros(g.size, bool)
    target = g.cell(np.array([10.5]), np.array([45.0]))[0]
    valid[target] = True
    near, km = g.nearest_valid(
        np.array([10.0, 10.0]), np.array([45.0, 46.0]), valid, 50
    )
    assert near[0] == target and 35 < km[0] < 45  # 0.5 deg of lon at 45N
    assert near[1] == -1 and np.isnan(km[1])  # too far


def test_nearest_valid_across_the_antimeridian():
    g = WORLDTENSOR
    valid = np.zeros(g.size, bool)
    target = g.cell(np.array([-179.75]), np.array([0.0]))[0]
    valid[target] = True
    near, km = g.nearest_valid(np.array([179.9]), np.array([0.0]), valid, 50)
    assert near[0] == target and km[0] < 40


# ---------------------------------------------------------------------------
# zones
# ---------------------------------------------------------------------------


def test_antimeridian_polygon_gets_a_narrow_box():
    west = shapely.box(-180, 60, -170, 65)
    east = shapely.box(172, 50, 180, 55)
    z = zones.from_table(
        zones_table([(-175, 62, 300e3, shapely.MultiPolygon([west, east]))])
    )
    assert z.xmin[0] == 172 and z.xmax[0] == 190
    assert z.ymin[0] == 50 and z.ymax[0] == 65


def test_point_box_widens_with_latitude():
    z = zones.from_table(
        zones_table([(0, 0, 1000, None), (0, 60, 1000, None), (0, 0, 10, None)])
    )
    w = z.xmax - z.xmin
    h = z.ymax - z.ymin
    np.testing.assert_allclose(w[1] / w[0], 2, rtol=1e-6)
    np.testing.assert_allclose(h[0], h[1])
    assert z.radius_m[2] == zones.MIN_RADIUS_M


# ---------------------------------------------------------------------------
# WorldTensor
# ---------------------------------------------------------------------------


def write_wt(world, var, year, data):
    lat = np.linspace(-90, 90, 721)
    lon = np.arange(1440) * 0.25
    ds = xr.Dataset(
        {var: (("time", "lat", "lon"), data[None].astype(np.float32))},
        coords={"time": [0], "lat": lat, "lon": lon},
    )
    ds[var].attrs["units"] = "u"
    d = world / "wt" / var
    d.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(d / f"{year}.nc")


@pytest.fixture
def wt_cache(tmp_path, monkeypatch):
    """A world with one variable, ``v``: 0 in 2019, row * 1e4 + col in 2020
    and 2022 plus 1e7 in 2022, and NaN at lon 20, lat 10 in 2020 only;
    preprocessed for 2020, and the only variable joined."""
    monkeypatch.setattr(worldtensor, "VARIABLES", {"v": "wt/v"})
    world, cache = tmp_path / "world", tmp_path / "cache"
    field = np.add.outer(
        np.arange(721) * 10000.0, np.arange(1440)
    )  # row * 1e4 + col
    write_wt(world, "v", 2019, field * 0)
    masked = field.copy()
    masked[
        WORLDTENSOR.rows(np.array([10.0]))[0],
        WORLDTENSOR.cols(np.array([20.0]))[0],
    ] = np.nan
    write_wt(world, "v", 2020, masked)
    write_wt(world, "v", 2022, field + 1e7)
    worldtensor.preprocess(world, cache, years=[2020])
    return world, cache


def expected(lon, lat):
    return (
        WORLDTENSOR.rows(np.array([lat]))[0] * 1e4
        + WORLDTENSOR.cols(np.array([lon]))[0]
    )


def test_nearest_year():
    years = [2019, 2020, 2022]
    assert worldtensor.nearest_year(years, 2020) == 2020
    assert worldtensor.nearest_year(years, 2021) == 2020  # tie: earlier
    assert worldtensor.nearest_year(years, 2030) == 2022
    assert worldtensor.nearest_year(years, 1990) == 2019


def test_preprocess_caches_years_asked_for_and_is_idempotent(wt_cache):
    world, cache = wt_cache
    out = cache / "worldtensor"
    assert not (out / "v_2022.npy").exists()
    mtime = (out / "v_2020.npy").stat().st_mtime_ns
    manifest = worldtensor.preprocess(world, cache, years=[2020, 2030])
    assert manifest["v"]["available"] == [2019, 2020, 2022]
    assert set(manifest["v"]["layers"]) == {"2020", "2022"}
    assert (out / "v_2020.npy").stat().st_mtime_ns == mtime
    # no years: the most recent, and earlier layers are kept
    assert set(worldtensor.preprocess(world, cache)["v"]["layers"]) == {
        "2020",
        "2022",
    }


def test_preprocess_replaces_an_old_style_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(worldtensor, "VARIABLES", {"v": "wt/v"})
    world, cache = tmp_path / "world", tmp_path / "cache"
    write_wt(world, "v", 2020, np.zeros((721, 1440)))
    out = cache / "worldtensor"
    out.mkdir(parents=True)
    (out / "v.npy").write_bytes(b"old")
    (out / "manifest.json").write_text(
        json.dumps({"v": {"file": "v.npy", "year": 2020}})
    )
    manifest = worldtensor.preprocess(world, cache)
    assert not (out / "v.npy").exists()
    assert manifest["v"]["layers"]["2020"]["file"] == "v_2020.npy"


def test_worldtensor_values(wt_cache):
    _, cache = wt_cache
    lon = np.array([20.0, -179.99, 180.0, np.nan, 5.0])
    lat = np.array([10.0, -90.0, 90.0, np.nan, 5.0])
    year = np.array([2020, 2020, 2020, 2020, np.nan])
    cols, meta = worldtensor.values(lon, lat, year, cache, max_fallback_km=50)
    v, y = cols["wt_v"], cols["wt_v_year"]
    row, col = WORLDTENSOR.rows(lat[:3]), WORLDTENSOR.cols(lon[:3])
    np.testing.assert_array_equal(v[1:3], row[1:3] * 10000.0 + col[1:3])
    # column 0 is lon 0, so both antimeridian points are in column 720
    assert row[1] == 0 and col[1] == 720 and row[2] == 720 and col[2] == 720
    assert np.isfinite(v[0])  # its cell is NaN: falls back
    assert np.isnan(v[3]) and y.mask[3]  # unlocated
    assert np.isnan(v[4]) and y.mask[4]  # no year
    assert y[:3].tolist() == [2020] * 3
    assert meta["variables"]["v"]["years"] == {"2020": 2020}
    assert meta["variables"]["v"]["on_nan_cells"] == 1
    assert meta["variables"]["v"]["filled"] == 1


def test_worldtensor_values_by_nearest_year(wt_cache):
    world, cache = wt_cache
    worldtensor.preprocess(world, cache, years=[2018, 2025])
    lon, lat = np.full(3, 20.0), np.full(3, 10.0)
    cols, meta = worldtensor.values(
        lon, lat, np.array([2018.0, 2025.0, 2020.0]), cache
    )
    assert cols["wt_v_year"].tolist() == [2019, 2022, 2020]
    # 2019 is all 0; 2022 has no hole here, so needs no fallback
    assert cols["wt_v"][0] == 0
    assert cols["wt_v"][1] == expected(20.0, 10.0) + 1e7
    assert meta["variables"]["v"]["years"] == {
        "2018": 2019,
        "2020": 2020,
        "2025": 2022,
    }


def test_worldtensor_values_needs_its_years_preprocessed(wt_cache):
    _, cache = wt_cache
    with pytest.raises(KeyError, match="2022"):
        worldtensor.values(
            np.array([5.0]), np.array([5.0]), np.array([2024.0]), cache
        )


def test_worldtensor_fallback_respects_radius(wt_cache):
    _, cache = wt_cache
    cols, _ = worldtensor.values(
        np.array([20.0]),
        np.array([10.0]),
        np.array([2020.0]),
        cache,
        max_fallback_km=5,
    )
    assert np.isnan(cols["wt_v"][0]) and cols["wt_v_year"].mask[0]


def test_static_layer_takes_its_reference_year(tmp_path, monkeypatch):
    world, cache = tmp_path / "world", tmp_path / "cache"
    ds = xr.Dataset(
        {"s": (("lat", "lon"), np.ones((721, 1440), np.float32))},
        coords={
            "lat": np.linspace(-90, 90, 721),
            "lon": np.arange(1440) * 0.25,
        },
        attrs={"reference_year": 2015},
    )
    (world / "static").mkdir(parents=True)
    ds.to_netcdf(world / "static" / "s.nc")
    monkeypatch.setattr(worldtensor, "VARIABLES", {"s": "static/s.nc"})
    worldtensor.preprocess(world, cache, years=[2010, 2020])
    cols, meta = worldtensor.values(
        np.array([5.0, 5.0]),
        np.array([5.0, 5.0]),
        np.array([2010.0, 2020.0]),
        cache,
    )
    assert cols["wt_s"].tolist() == [1, 1]
    assert cols["wt_s_year"].tolist() == [2015, 2015]
    assert meta["variables"]["s"]["years"] == {"2010": 2015, "2020": 2015}


def test_preprocess_removes_variables_no_longer_used(wt_cache, monkeypatch):
    world, cache = wt_cache
    write_wt(world, "w", 2000, np.zeros((721, 1440)))
    monkeypatch.setattr(worldtensor, "VARIABLES", {"w": "wt/w"})
    manifest = worldtensor.preprocess(world, cache)
    assert set(manifest) == {"w"}
    assert not (cache / "worldtensor" / "v_2020.npy").exists()
    saved = json.loads((cache / "worldtensor" / "manifest.json").read_text())
    assert set(saved) == {"w"}


def test_missing_sources(wt_cache, tmp_path):
    world, _ = wt_cache
    assert worldtensor.missing_sources(world) == []
    problems = worldtensor.missing_sources(tmp_path / "nowhere")
    assert len(problems) == 1 and "WorldTensor v" in problems[0]


def test_point_outside_its_concave_polygon():
    # a U whose centroid is in the gap: the join uses the zone's own point,
    # and the box flags that the polygon spans cells
    u = shapely.Polygon(
        [
            (30, 0),
            (32, 0),
            (32, 2),
            (31.7, 2),
            (31.7, 0.3),
            (30.3, 0.3),
            (30.3, 2),
            (30, 2),
        ]
    )
    centroid = shapely.centroid(u)
    assert not u.contains(centroid)
    z = zones.from_table(zones_table([(centroid.x, centroid.y, 5e4, u)]))
    cols = worldtensor.cells(z)
    assert (
        cols["wt_cell"][0]
        == WORLDTENSOR.cell(np.array([centroid.x]), np.array([centroid.y]))[0]
    )
    assert cols["wt_spans_cells"][0]


def test_template_lists_every_attribute_column():
    template = utils.get_template_attributes()
    assert [
        c for c in template if c.startswith("wt_")
    ] == worldtensor.attribute_columns()


# ---------------------------------------------------------------------------
# locations, then attributes
# ---------------------------------------------------------------------------


def locations_frame() -> pl.DataFrame:
    return pl.from_arrow(
        zones_table(
            [
                (20.0, 10.0, 50, None),  # a NaN cell in 2020
                (5.0, 5.0, 2000, None),
                (None, None, None, None),  # no geometry
            ]
        )
    ).with_columns(pl.Series("extra", ["a", "b", "c"]))


def test_join_locations_adds_cells(wt_cache):
    locations = locations_frame()
    out = join.join_locations(locations)
    assert out.select(locations.columns).equals(locations)
    assert out.columns[locations.width :] == ["wt_cell", "wt_spans_cells"]
    assert out["wt_cell"][2] is None
    assert (
        out["wt_cell"][1]
        == WORLDTENSOR.cell(np.array([5.0]), np.array([5.0]))[0]
    )


def test_to_attributes_by_home_location_and_year(wt_cache):
    world, cache = wt_cache
    locations = join.join_locations(locations_frame())
    attributes = pl.DataFrame(
        {
            "pid": [0, 1, 2, 3, 4],
            "hh_location": [1, None, 1, 0, 1],
            "year": [2020, 2020, 2020, 2020, 2024],
            # null, as fix.missing_columns adds
            "wt_v": [None] * 5,
            "wt_v_year": [None] * 5,
        },
        schema_overrides={
            "hh_location": pl.Int32,
            "year": pl.Int32,
            "wt_v": pl.Float32,
            "wt_v_year": pl.Int32,
        },
    )
    out, meta = join.to_attributes(attributes, locations, world, cache)
    assert out.columns == ["pid", "hh_location", "year", "wt_v", "wt_v_year"]
    assert out["pid"].to_list() == [0, 1, 2, 3, 4]
    here = expected(5.0, 5.0)
    assert out["wt_v"][[0, 1, 2, 4]].to_list() == [here, None, here, here + 1e7]
    assert out["wt_v"][3] is not None  # fell back
    assert out["wt_v_year"].to_list() == [2020, None, 2020, 2020, 2022]
    assert out["wt_v_year"].dtype == pl.Int32
    assert meta["worldtensor"]["variables"]["v"]["years"] == {
        "2020": 2020,
        "2024": 2022,
    }


def test_locations_metadata_is_written(tmp_path):
    meta = {"worldtensor": {"variables": {}}}
    path = tmp_path / "locations.parquet"
    location.write(
        join.join_locations(locations_frame()),
        path,
        metadata={join.METADATA_KEY: meta},
    )
    schema_meta = pq.read_schema(path).metadata
    assert json.loads(schema_meta[join.METADATA_KEY]) == meta
    assert b"geo" in schema_meta


def test_runner_checks_worldtensor_before_loading(tmp_path, monkeypatch):
    loaded = []
    monkeypatch.setattr(run, "load_source", lambda *a: loaded.append(a))
    (tmp_path / "CHILE").mkdir()
    (tmp_path / "CHILE" / "data.csv").write_text("")
    with pytest.raises(run.SourceDataError, match="--no-grid"):
        run.runner(
            {"chile": tmp_path / "CHILE"},
            tmp_path / "out",
            world=tmp_path / "no_world",
        )
    assert not loaded
