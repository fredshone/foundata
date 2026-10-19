"""Helpers of scripts/build_location_lookups.py (needs the scripts group:
``uv run --group scripts pytest``)."""

import sys
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import pytest
import shapely

gpd = pytest.importorskip("geopandas")
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import build_location_lookups as b  # noqa: E402


def squares(codes) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"code": codes},
        geometry=[shapely.box(i, 0, i + 0.1, 0.1) for i in range(len(codes))],
        crs="EPSG:4326",
    )


def test_polygons_drops_null_codes_and_dissolves(tmp_path):
    builder = b.Builder(tmp_path)
    gdf = squares(["a", None, "a", "b"])
    out = builder.polygons(gdf, gdf.code, None).sort_values("code")
    assert out.code.tolist() == ["a", "b"]
    assert out.geometry.iloc[0].geom_type == "MultiPolygon"
    assert out.area_km2.iloc[0] == pytest.approx(2 * out.area_km2.iloc[1])


def test_write_lookup_with_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "CONFIGS", tmp_path)
    builder = b.Builder(tmp_path)
    builder.source = "xx"
    gdf = squares(["2", "1"])
    lookup = builder.polygons(gdf, gdf.code, None)
    lookup["geom_dataset"] = "test"
    builder.write(lookup, "state", method="squares", unmatched=["9"])
    path = tmp_path / "xx" / "geo" / "state.parquet"
    out = pl.read_parquet(path)
    assert out.columns == b.location.LOOKUP_COLUMNS
    assert out["code"].to_list() == ["1", "2"]
    assert out["geom_source"].to_list() == ["boundary", "boundary"]
    inside = shapely.contains_xy(
        shapely.from_wkb(out["geometry"].to_numpy()),
        out["lon"].to_numpy(),
        out["lat"].to_numpy(),
    )
    assert inside.all()
    meta = pq.read_schema(path).metadata[b"foundata"].decode()
    assert '"method": "squares"' in meta and '"unmatched": ["9"]' in meta


def test_india_name_aliases():
    aliases = {"gurgaon": "gurugram", "27:raigarh": "raigad"}
    assert b.india_name("Gurgaon", "06", aliases) == "gurugram"
    assert b.india_name("Raigarh", "27", aliases) == "raigad"
    assert b.india_name("Raigarh", "22", aliases) == "raigarh"
    assert b.india_name("Bangalore (Rural)", "29", aliases) == "bangalorerural"
    assert b.india_name("North West Delhi", "07", aliases) == "northwest"
    assert b.india_name("New Delhi", "07", aliases) == "newdelhi"


def test_find_district_records_fuzzy_matches():
    table = pd.DataFrame({"st": ["36", "36"], "n": ["jagitial", "jangoan"]})
    fuzzy = {}
    assert b.find_district(table, "36", "jagitial", fuzzy) == 0
    assert fuzzy == {}
    assert b.find_district(table, "36", "jagtial", fuzzy) == 0
    assert fuzzy == {"36:jagtial": "jagitial"}
    assert b.find_district(table, "36", "hyderabad", fuzzy) is None


def test_source_lookups_reads_declarations():
    assert b.source_lookups("ovin") == [
        f"gemeente_{y}" for y in range(2010, 2018)
    ]
    assert "sigungu_2016" in b.source_lookups("ktdb")
    assert b.source_lookups("brazil") == ["od_zone_2017"]
    assert b.source_lookups("odin")[:2] == ["pc4", "gemeente_2018"]


def test_every_source_has_a_builder():
    from foundata import run

    assert set(run.SOURCES) == set(b.BUILDERS)
