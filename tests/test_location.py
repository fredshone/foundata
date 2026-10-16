import json

import polars as pl
import pyarrow.parquet as pq
import pytest
import shapely

from foundata import location, run, utils

# -- declarations ---------------------------------------------------------


def test_add_takes_first_valid_candidate():
    spec = [
        {
            "level": "cbsa_2013",
            "when": {"year": [2017]},
            "code": [{"column": "cbsa", "width": 5}],
            "valid": r"\d{5}",
        },
        {
            "level": "msa_1999",
            "code": [{"column": "msa", "width": 4}],
            "missing": ["9999"],
        },
        {"level": "state", "code": [{"column": "state", "width": 2}]},
    ]
    df = pl.DataFrame(
        {
            "cbsa": ["12060", "XXXX", None, None],
            "msa": ["520", "9999", "9999", None],
            "state": [13.0, 6.0, 6.0, None],
        }
    )
    out = location.add(df, spec, source="t", year=2017)
    assert out["home_level"].to_list() == ["cbsa_2013", "state", "state", None]
    assert out["home_code"].to_list() == ["12060", "06", "06", None]
    # cbsa_2013 doesn't apply in 2009, so the MSA is used
    out = location.add(df, spec, source="t", year=2009)
    assert out["home_level"].to_list() == ["msa_1999", "state", "state", None]
    assert out["home_code"].to_list() == ["0520", "06", "06", None]
    assert out.columns == df.columns + location.columns("home")


def test_add_parts_map_slice_prefix_and_year_level(tmp_path, monkeypatch):
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "map.csv").write_text("from,to\n63,82\n84,30\n")
    monkeypatch.setattr(
        utils, "get_config_path", lambda *parts: tmp_path.joinpath(*parts)
    )
    spec = [
        {
            "level": "radio_{year}",
            "years": [2010],
            "prefix": "{survey}/",
            "code": [
                {"column": "depto", "map": "map.csv"},
                {"column": "depto", "width": 3},
                {"column": "x", "slice": [0, 2]},
                {"value": "9"},
            ],
        }
    ]
    df = pl.DataFrame({"depto": [63, 84, 70], "x": ["01xx", "02yy", "03"]})
    out = location.add(df, spec, source="t", year=2010, survey="sfp")
    assert out["home_level"].to_list() == ["radio_2010", "radio_2010", None]
    assert out["home_code"].to_list() == ["sfp/82063019", "sfp/30084029", None]


def test_add_projects_points_and_falls_back():
    spec = [
        {"level": "detailed_zone", "code": ["zone"]},
        {
            "level": "point",
            "point": {"x": "x", "y": "y", "crs": "EPSG:32719", "scale_m": 50},
        },
        {"level": "commune_{year}", "years": [2015], "code": ["insee"]},
    ]
    df = pl.DataFrame(
        {
            "zone": ["001", None, None],
            "x": [None, 346000.0, None],
            "y": [None, 6297000.0, None],
            "insee": [None, None, "69123"],
        }
    )
    out = location.add(df, spec, source="t", year=2015)
    assert out["home_level"].to_list() == [
        "detailed_zone",
        "point",
        "commune_2015",
    ]
    assert out["home_code"].to_list() == ["001", None, "69123"]
    assert out["home_lon"][1] == pytest.approx(-70.66, abs=0.01)
    assert out["home_lat"][1] == pytest.approx(-33.45, abs=0.01)
    assert out["home_lon"][0] is None
    assert out["home_scale_m"].to_list() == [None, 50.0, None]


def test_add_warns_for_a_year_without_lookup(caplog):
    spec = [{"level": "gemeente_{year}", "years": [2017], "code": ["g"]}]
    out = location.add(
        pl.DataFrame({"g": ["0363"]}), spec, source="t", year=2018
    )
    assert out["home_level"].to_list() == ["gemeente_2018"]


@pytest.mark.parametrize(
    "spec, message",
    [
        ([], "non-empty"),
        ([{"level": "nowhere", "code": ["a"]}], "no precision class"),
        ([{"level": "state"}], "exactly one of code and point"),
        ([{"level": "gemeente_{year}", "code": ["a"]}], "needs years"),
        ([{"level": "state", "code": ["a"], "colour": 1}], "unknown keys"),
        (
            [{"level": "state", "code": [{"column": "a", "pad": 2}]}],
            "unknown keys",
        ),
        ([{"level": "state", "point": {"x": "a"}}], "point's level"),
    ],
)
def test_validate_spec(spec, message):
    assert any(message in e for e in location.validate_spec(spec))


def test_every_source_declares_a_valid_home_location():
    for source in run.SOURCES:
        path = utils.get_config_path(source, "hh_dictionary.yaml")
        config = utils.load_yaml_config(path)
        assert "home_location" in config, path
        spec = config["home_location"]
        specs = spec.values() if isinstance(spec, dict) else [spec]
        for s in specs:
            assert location.validate_spec(s) == [], path


def test_precision_ignores_year_suffix():
    assert location.precision_of("gemeente_2030") == "large area"
    assert location.precision_of("dong_2026") == "small area"
    assert location.precision_of("point") == "coordinate"
    assert location.precision_of("nowhere_2020") is None


# -- extract, index, resolve, write -----------------------------------------


def test_extract_one_ref_per_household_and_drops_columns():
    df = pl.DataFrame(
        {
            "pid": ["a1", "a2", "b1"],
            "hid": ["a", "a", "b"],
            "country": ["aus"] * 3,
            "home_level": ["sa1_2021"] * 3,
            "home_code": ["1", "1", "2"],
        }
    )
    out, refs = location.extract(df, "qhts")
    assert out.columns == ["pid", "hid", "country"]
    assert refs["hid"].to_list() == ["a", "b"]
    assert refs["code"].to_list() == ["1", "2"]
    assert refs["role"].to_list() == ["home", "home"]
    assert refs["country"].to_list() == ["aus", "aus"]
    assert refs["lon"].is_null().all()


def test_extract_without_location_columns_is_empty():
    df = pl.DataFrame({"pid": ["a1"], "hid": ["a"]})
    out, refs = location.extract(df, "nts")
    assert out.equals(df)
    assert refs.is_empty()
    assert refs.columns == ["hid", *location.REF_SCHEMA]


def test_extract_rejects_a_zone_without_a_code():
    df = pl.DataFrame(
        {"hid": ["a"], "home_level": ["lga_2021"], "home_code": [None]}
    )
    with pytest.raises(ValueError, match="level but no code"):
        location.extract(df.cast({"home_code": pl.String}), "vista")


def test_extract_keeps_first_when_members_disagree(caplog):
    df = pl.DataFrame(
        {
            "hid": ["a", "a"],
            "home_level": ["state"] * 2,
            "home_code": ["01", "02"],
        }
    )
    _, refs = location.extract(df, "nhts")
    assert refs["code"].to_list() == ["01"]


def test_extract_trip_ends_by_role():
    trips = pl.DataFrame(
        {
            "pid": ["p", "p"],
            "seq": [1, 2],
            "origin_level": ["sa1_2021", "sa1_2021"],
            "origin_code": ["1", "2"],
        }
    )
    out, refs = location.extract(
        trips, "qhts", role="origin", keys=("pid", "seq")
    )
    assert out.columns == ["pid", "seq"]
    assert refs.select("pid", "seq", "role", "code").rows() == [
        ("p", 1, "origin", "1"),
        ("p", 2, "origin", "2"),
    ]


def square(lon: float, lat: float) -> bytes:
    return shapely.to_wkb(
        shapely.box(lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01)
    )


def lookup_frame(**columns) -> pl.DataFrame:
    n = len(columns["code"])
    columns.setdefault("geom_source", ["boundary"] * n)
    columns.setdefault("geom_dataset", ["test boundaries"] * n)
    columns.setdefault("geom_year", [2021] * n)
    return pl.DataFrame(columns).cast(
        {
            "code": pl.String,
            "lon": pl.Float64,
            "lat": pl.Float64,
            "area_km2": pl.Float64,
            "geom_source": pl.String,
            "geom_dataset": pl.String,
            "geom_year": pl.Int16,
            "geometry": pl.Binary,
        }
    )[location.LOOKUP_COLUMNS]


def refs_frame(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"hid": pl.String} | location.REF_SCHEMA,
        orient="row",
    )


def test_index_shares_rows_resolves_and_keys(monkeypatch):
    lookup = lookup_frame(
        code=["1"],
        lon=[153.0],
        lat=[-27.0],
        area_km2=[3.14159265],
        geometry=[square(153.0, -27.0)],
    )
    monkeypatch.setattr(
        location,
        "load_lookup",
        lambda source, level, country=None: (
            lookup if level == "sa1_2021" else None
        ),
    )
    home = ("home",)
    refs = refs_frame(
        [
            ("p", *home, "chile", "cl", "point", None, -70.6, -33.4, 50.0),
            ("q", *home, "chile", "cl", "point", None, -70.6, -33.4, 50.0),
            ("a", *home, "qhts", "aus", "sa1_2021", "1", None, None, None),
            ("a2", *home, "qhts", "aus", "sa1_2021", "1", None, None, None),
            ("c", *home, "qhts", "aus", "sa1_2021", "9", None, None, None),
            ("d", *home, "vista", "aus", "lga_2021", "20660", None, None, None),
        ]
    )
    locations, refs = location.index(refs)
    assert locations.schema == pl.Schema(location.LOCATION_SCHEMA)
    # numbered in key order
    assert locations["location_key"].to_list() == [
        "chile/point/-70.6,-33.4",
        "qhts/sa1_2021/1",
        "qhts/sa1_2021/9",
        "vista/lga_2021/20660",
    ]
    assert locations["location_id"].to_list() == [0, 1, 2, 3]
    ids = dict(refs.select("hid", "location_id").iter_rows())
    assert ids == {"p": 0, "q": 0, "a": 1, "a2": 1, "c": 2, "d": 3}

    rows = {r["location_id"]: r for r in locations.iter_rows(named=True)}
    assert rows[0]["geom_source"] == "point"
    assert rows[0]["scale_m"] == 50.0
    assert rows[0]["precision"] == "coordinate"
    assert rows[0]["geom_dataset"] == "chile survey"
    assert shapely.from_wkb(rows[0]["geometry"]).equals(
        shapely.Point(-70.6, -33.4)
    )
    assert rows[1]["geom_source"] == "boundary"
    assert rows[1]["scale_m"] == pytest.approx(1000.0)
    assert rows[1]["geom_dataset"] == "test boundaries"
    assert rows[1]["geom_year"] == 2021
    assert shapely.from_wkb(rows[1]["geometry"]).geom_type == "Polygon"
    # a code missing from the lookup, and a level without a lookup
    assert rows[2]["geom_source"] == "none"
    assert rows[2]["geometry"] is None
    assert rows[3]["geom_source"] == "none"
    assert rows[3]["precision"] == "large area"


def test_lookup_geom_source_and_year_suffixed_levels(monkeypatch):
    lookup = lookup_frame(
        code=["0363", "0014"],
        lon=[4.9, 6.6],
        lat=[52.4, 53.2],
        area_km2=[219.0, 197.0],
        geom_source=["boundary", "estimated"],
        geometry=[square(4.9, 52.4), square(6.6, 53.2)],
    )
    monkeypatch.setattr(location, "load_lookup", lambda *a, **k: lookup)
    refs = refs_frame(
        [
            (
                "a",
                "home",
                "ovin",
                "nl",
                "gemeente_2012",
                "0363",
                None,
                None,
                None,
            ),
            (
                "b",
                "home",
                "ovin",
                "nl",
                "gemeente_2012",
                "0014",
                None,
                None,
                None,
            ),
        ]
    )
    locations, _ = location.index(refs)
    # numbered in key order: 0014 first
    assert locations["geom_source"].to_list() == ["estimated", "boundary"]
    assert locations["precision"].to_list() == ["large area"] * 2


def test_load_lookup_falls_back_to_shared_geo(tmp_path, monkeypatch):
    monkeypatch.setattr(
        utils, "get_config_path", lambda *parts: tmp_path.joinpath(*parts)
    )
    lookup = lookup_frame(
        code=["0363"],
        lon=[4.9],
        lat=[52.4],
        area_km2=[219.0],
        geometry=[square(4.9, 52.4)],
    )
    (tmp_path / "geo" / "nl").mkdir(parents=True)
    lookup.write_parquet(tmp_path / "geo" / "nl" / "gemeente_2012.parquet")
    assert location.load_lookup("ovin", "gemeente_2012", "nl").equals(lookup)
    assert location.load_lookup("ovin", "gemeente_2012", "uk") is None
    # a source's own lookup comes first
    (tmp_path / "ovin" / "locations").mkdir(parents=True)
    own = lookup.with_columns(geom_dataset=pl.lit("own"))
    own.write_parquet(tmp_path / "ovin" / "locations" / "gemeente_2012.parquet")
    assert location.load_lookup("ovin", "gemeente_2012", "nl").equals(own)


def test_write_is_geoparquet(tmp_path):
    locations = pl.DataFrame(
        {
            "location_id": [0, 1],
            "location_key": ["chile/point/-70.6,-33.4", "qhts/sa1_2021/1"],
            "source": ["chile", "qhts"],
            "country": ["cl", "aus"],
            "level": ["point", "sa1_2021"],
            "code": [None, "1"],
            "precision": ["coordinate", "small area"],
            "lon": [-70.6, 153.0],
            "lat": [-33.4, -27.0],
            "scale_m": [50.0, 1000.0],
            "geom_source": ["point", "boundary"],
            "geom_dataset": ["chile survey", "test"],
            "geom_year": [None, 2021],
            "geometry": [
                shapely.to_wkb(shapely.Point(-70.6, -33.4)),
                square(153.0, -27.0),
            ],
        }
    ).cast(location.LOCATION_SCHEMA)
    path = tmp_path / "locations.parquet"
    location.write(locations, path)
    geo = json.loads(pq.read_schema(path).metadata[b"geo"])
    assert geo["primary_column"] == "geometry"
    assert geo["columns"]["geometry"]["geometry_types"] == ["Point", "Polygon"]
    assert pl.read_parquet(path).equals(locations)


# -- the committed lookups ----------------------------------------------------

# rough lon/lat bounds of each lookup directory's country, to catch CRS or
# axis-order mistakes
BOUNDS = {
    "uk": (-9, 49, 2, 61),
    "usa": (-180, -15, 180, 72),  # states include the Pacific territories
    "aus": (112, -44, 154, -9),
    "nl": (3, 50.7, 7.3, 53.7),
    "kr": (124, 33, 132, 39),
    "ar": (-74, -56, -53, -21),
    "uy": (-59, -35.1, -53, -30),
    "mng": (87, 41, 120, 53),
    "fr": (-62, -22, 56, 52),  # with the overseas departments
    "india": (68, 6, 98, 38),
    "mexico": (-100, 18, -98, 21),
    "ktdb": (124, 33, 132, 39),
    "chile": (-72, -34.5, -70, -32.5),
    "france": (-62, -22, 56, 52),
}


def lookup_paths():
    root = utils.get_config_path()
    return sorted(
        [*root.glob("*/locations/*.parquet"), *root.glob("geo/*/*.parquet")]
    )


def test_lookups_are_complete_and_plausible():
    paths = lookup_paths()
    assert paths
    for path in paths:
        assert location.precision_of(path.stem) is not None, path
        lookup = pl.read_parquet(path)
        assert lookup.columns == location.LOOKUP_COLUMNS, path
        assert lookup.height > 0, path
        assert lookup["code"].n_unique() == lookup.height, path
        assert lookup["geometry"].is_not_null().all(), path
        assert (lookup["area_km2"] > 0).all(), path
        assert (
            lookup["geom_source"]
            .is_in(["boundary", "parent", "estimated"])
            .all()
        ), path
        assert lookup["geom_dataset"].is_not_null().all(), path
        geometry = shapely.from_wkb(lookup["geometry"].to_numpy())
        inside = shapely.contains_xy(
            geometry, lookup["lon"].to_numpy(), lookup["lat"].to_numpy()
        )
        assert inside.all(), path
        directory = path.parent.name
        key = path.parent.parent.name if directory == "locations" else directory
        x0, y0, x1, y1 = BOUNDS[key]
        assert lookup["lon"].is_between(x0, x1).all(), path
        assert lookup["lat"].is_between(y0, y1).all(), path
        meta = json.loads(pq.read_schema(path).metadata[b"foundata"])
        assert meta["level"] == path.stem and meta["method"], path


def test_lookups_of_different_vintages_differ():
    by_digest = {}
    for path in lookup_paths():
        lookup = pl.read_parquet(path).select("code", "geom_source", "geometry")
        digest = lookup.hash_rows(seed=0).sum()
        assert digest not in by_digest, (path, by_digest.get(digest))
        by_digest[digest] = path
