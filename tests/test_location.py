import json

import polars as pl
import pyarrow.parquet as pq
import pytest
import shapely

from foundata import location, run, utils


def square(lon: float, lat: float) -> bytes:
    return shapely.to_wkb(
        shapely.box(lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01)
    )


def lookup_frame(codes: list[str], lon: float = 4.9, lat: float = 52.4):
    """A lookup of small squares, one per code, the first at lon, lat."""
    n = len(codes)
    return pl.DataFrame(
        {
            "code": codes,
            "lon": [lon + i for i in range(n)],
            "lat": [lat] * n,
            "area_km2": [3.14159265] * n,
            "geom_source": ["boundary"] * n,
            "geom_dataset": ["test boundaries"] * n,
            "geom_year": [2021] * n,
            "geometry": [square(lon + i, lat) for i in range(n)],
        },
        schema=location.LOOKUP_SCHEMA,
    )


@pytest.fixture
def lookups(tmp_path, monkeypatch):
    """Write configs/t/geo/<stem>.parquet with the given codes."""
    monkeypatch.setattr(
        utils, "get_config_path", lambda *parts: tmp_path.joinpath(*parts)
    )
    (tmp_path / "t" / "geo").mkdir(parents=True)

    def write(stem, codes):
        lookup_frame(codes).write_parquet(
            tmp_path / "t" / "geo" / f"{stem}.parquet"
        )

    return write


NHTS_LIKE = {
    2017: [
        {"level": "cbsa", "geo": "cbsa_2013.parquet"},
        {"level": "state", "geo": "state.parquet"},
    ],
    "default": [
        {"level": "msa", "geo": "msa_1999.parquet"},
        {"level": "state", "geo": "state.parquet"},
    ],
}


# -- finding keys -------------------------------------------------------------


def test_add_keys_the_first_level_with_a_known_code(lookups):
    lookups("cbsa_2013", ["12060"])
    lookups("msa_1999", ["0520"])
    lookups("state", ["06", "13"])
    finder = location.LocationFinder("t", home=NHTS_LIKE)
    df = pl.DataFrame(
        {
            "home_cbsa": ["12060", "99999", None, None],
            "home_msa": ["0520", "0520", None, None],
            "home_state": ["13", "06", "06", None],
        }
    )
    out = finder.add(df, key=2017)
    # a code its lookup lacks (99999) falls through to the next level
    assert out["hh_location"].to_list() == [
        "t/cbsa_2013/12060",
        "t/state/06",
        "t/state/06",
        None,
    ]
    # the code columns it read are dropped; home_msa wasn't read in 2017
    assert out.columns == ["home_msa", "hh_location"]
    # other keys take the default block
    out = finder.add(df, key=2009)
    assert out["hh_location"].to_list() == [
        "t/msa_1999/0520",
        "t/msa_1999/0520",
        "t/state/06",
        None,
    ]


def test_the_last_level_keeps_an_unknown_code(lookups):
    lookups("gemeente_2018", ["0363"])
    declaration = [{"level": "gemeente", "geo": "gemeente_2018.parquet"}]
    df = pl.DataFrame({"home_gemeente": ["0363", "9999"]})
    out = location.LocationFinder("t", home=declaration).add(df)
    assert out["hh_location"].to_list() == [
        "t/gemeente_2018/0363",
        "t/gemeente_2018/9999",
    ]


def test_add_points_then_zones(lookups):
    lookups("commune_2015", ["69123"])
    declaration = [
        {"level": "point", "scale_m": 50},
        {"level": "commune", "geo": "commune_2015.parquet"},
    ]
    df = location.wgs84(
        pl.DataFrame(
            {
                "x": [346000.0, None],
                "y": [6297000.0, None],
                "home_commune": ["69001", "69123"],
            }
        ),
        "x",
        "y",
        "EPSG:32719",
    )
    out = location.LocationFinder("t", home=declaration).add(df)
    point, commune = out["hh_location"].to_list()
    assert commune == "t/commune_2015/69123"
    lon, lat = map(float, point.removeprefix("t/point/").split(","))
    assert lon == pytest.approx(-70.66, abs=0.01)
    assert lat == pytest.approx(-33.45, abs=0.01)
    assert out.columns == ["x", "y", "hh_location"]


def test_add_needs_the_code_columns(lookups):
    with pytest.raises(ValueError, match="home_state"):
        location.LocationFinder("t", home=NHTS_LIKE).add(
            pl.DataFrame({"x": [1]}), key=2017
        )


def test_add_trip_ends_keys_both_ends(lookups):
    lookups("sa1_2021", ["1", "2"])
    declaration = [{"level": "sa1", "geo": "sa1_2021.parquet"}]
    trips = pl.DataFrame(
        {"origin_sa1": ["1", None], "destination_sa1": ["2", "1"]}
    )
    out = location.LocationFinder("t", trip=declaration).add_trip_ends(trips)
    assert out.columns == ["ocode", "dcode"]
    assert out["ocode"].to_list() == ["t/sa1_2021/1", None]
    assert out["dcode"].to_list() == ["t/sa1_2021/2", "t/sa1_2021/1"]


def test_clean_and_mapped(lookups, tmp_path):
    (tmp_path / "t" / "map.csv").write_text("from,to\n63,82\n")
    df = pl.DataFrame(
        {"a": [" 7.0", "", "63", "1234"], "b": ["63", "1", None, "63"]}
    )
    out = df.select(
        a=location.clean("a", width=3, valid=r"\d{3}", missing=["063"]),
        b=location.mapped("b", "t", "map.csv"),
    )
    assert out["a"].to_list() == ["007", None, None, None]
    assert out["b"].to_list() == ["82", None, None, "82"]


# -- keys to locations ----------------------------------------------------------


def test_index_builds_the_locations_and_swaps_keys_for_ids(lookups):
    lookups("sa1_2021", ["1", "2"])
    finder = location.LocationFinder(
        "t",
        home=[
            {"level": "point", "scale_m": 50},
            {"level": "sa1", "geo": "sa1_2021.parquet"},
        ],
    )
    attributes = pl.DataFrame(
        {
            "hid": ["a", "a", "b", "c"],
            "country": ["aus"] * 4,
            "hh_location": [
                "t/sa1_2021/1",
                "t/sa1_2021/1",
                "t/point/-70.6,-33.4",
                "t/sa1_2021/9",  # not in the lookup
            ],
        }
    )
    trips = pl.DataFrame(
        {"ocode": ["t/sa1_2021/1", None], "dcode": ["t/sa1_2021/2", None]}
    )
    attributes, trips, locations = finder.index(attributes, trips)
    assert locations.schema == pl.Schema(location.LOCATION_SCHEMA)
    # numbered from 0 in key order; households share a row
    assert locations["location_key"].to_list() == [
        "t/point/-70.6,-33.4",
        "t/sa1_2021/1",
        "t/sa1_2021/2",
        "t/sa1_2021/9",
    ]
    assert attributes["hh_location"].to_list() == [1, 1, 0, 3]
    assert trips["ocode"].to_list() == [1, None]
    assert trips["dcode"].to_list() == [2, None]

    rows = list(locations.iter_rows(named=True))
    assert rows[0]["level"] == "point"
    assert rows[0]["geom_source"] == "point"
    assert rows[0]["scale_m"] == 50.0
    assert rows[0]["precision"] == "coordinate"
    assert shapely.from_wkb(rows[0]["geometry"]).equals(
        shapely.Point(-70.6, -33.4)
    )
    assert (rows[1]["level"], rows[1]["code"], rows[1]["geo"]) == (
        "sa1",
        "1",
        "sa1_2021",
    )
    assert rows[1]["country"] == "aus"
    assert rows[1]["precision"] == "small area"
    assert rows[1]["geom_source"] == "boundary"
    assert rows[1]["scale_m"] == pytest.approx(1000.0)
    assert rows[1]["geom_year"] == 2021
    # a code missing from the lookup has no geometry
    assert rows[3]["geom_source"] == "none"
    assert rows[3]["geometry"] is None


def test_codes_with_slashes_and_vintages(lookups):
    lookups("gemeente_2012", ["0363"])
    lookups("gemeente_2013", ["0363"])
    lookups("detailed_zone", ["lyon_2015/001"])
    finder = location.LocationFinder(
        "t",
        home={
            2012: [{"level": "gemeente", "geo": "gemeente_2012.parquet"}],
            2013: [{"level": "gemeente", "geo": "gemeente_2013.parquet"}],
            2015: [{"level": "detailed_zone", "geo": "detailed_zone.parquet"}],
        },
    )
    locations = finder.locations(
        pl.Series(
            [
                "t/detailed_zone/lyon_2015/001",
                "t/gemeente_2012/0363",
                "t/gemeente_2013/0363",
            ]
        )
    )
    # the same code in two vintages is two locations
    assert locations.select("geo", "code").rows() == [
        ("detailed_zone", "lyon_2015/001"),
        ("gemeente_2012", "0363"),
        ("gemeente_2013", "0363"),
    ]
    assert (locations["geom_source"] == "boundary").all()


def test_lookups_are_loaded_once_until_index(lookups, monkeypatch):
    lookups("state", ["06"])
    finder = location.LocationFinder("t", home=NHTS_LIKE)
    calls = []
    load = location.load_lookup
    monkeypatch.setattr(
        location,
        "load_lookup",
        lambda *args: calls.append(args) or load(*args),
    )
    for _ in range(3):
        finder.lookup("state")
    assert finder.lookup("state")["code"].to_list() == ["06"]
    assert calls == [("t", "state")]
    assert finder.lookup("missing") is None
    # index releases them
    finder.index(
        pl.DataFrame({"country": ["us"], "hh_location": ["t/state/06"]}),
        pl.DataFrame(),
    )
    finder.lookup("state")
    assert calls[-1] == ("t", "state") and len(calls) == 3


def test_blocks_match_string_and_int_keys(lookups):
    lookups("bagh_2019", ["110151"])
    lookups("soum", ["1101"])
    home = {
        "2019": [
            {"level": "bagh", "geo": "bagh_2019.parquet"},
            {"level": "soum", "geo": "soum.parquet"},
        ],
        "default": [{"level": "soum", "geo": "soum.parquet"}],
    }
    finder = location.LocationFinder("t", home=home)
    df = pl.DataFrame({"home_bagh": ["110151"], "home_soum": ["1101"]})
    assert (
        finder.add(df, key=2019)["hh_location"].item() == "t/bagh_2019/110151"
    )
    df = pl.DataFrame({"home_soum": ["1101"]})
    assert finder.add(df, key=2011)["hh_location"].item() == "t/soum/1101"


def test_finder_rejects_an_invalid_declaration():
    with pytest.raises(ValueError, match="needs geo"):
        location.LocationFinder("t", home=[{"level": "state"}])


@pytest.mark.parametrize(
    "declaration, message",
    [
        ([], "list of levels"),
        ([{"level": "nowhere", "geo": "x.parquet"}], "no precision class"),
        ([{"level": "state"}], "needs geo"),
        ([{"level": "state", "geo": "state.csv"}], "needs geo"),
        ([{"level": "state", "geo": "s.parquet", "colour": 1}], "unknown keys"),
        ([{"level": "point"}], "scale_m"),
        (
            [{"level": "state", "geo": "s.parquet", "scale_m": 5}],
            "only a point",
        ),
        (
            {2019: [{"level": "state", "geo": "a.parquet"}] * 2},
            "listed twice",
        ),
    ],
)
def test_validate(declaration, message):
    assert any(message in e for e in location.validate(declaration))


def test_every_source_declares_valid_locations():
    trips = []
    for source in run.SOURCES:
        for dictionary, key in (
            ("hh_dictionary.yaml", "home_location"),
            ("trip_dictionary.yaml", "trip_location"),
        ):
            config = utils.load_yaml_config(
                utils.get_config_path(source, dictionary)
            )
            if key == "trip_location" and key not in config:
                continue
            assert location.validate(config[key], key) == [], (source, key)
            for geo in location.geo_files(config[key]):
                assert location.lookup_path(source, geo).exists(), (source, geo)
            if key == "trip_location":
                trips.append(source)
    assert len(trips) == 13


def test_precision_of_levels():
    assert location.precision_of("gemeente") == "large area"
    assert location.precision_of("dong") == "small area"
    assert location.precision_of("point") == "coordinate"
    assert location.precision_of("nowhere") is None


def test_write_is_geoparquet(tmp_path):
    locations = pl.DataFrame(
        {
            "location_id": [0, 1],
            "location_key": ["chile/point/-70.6,-33.4", "qhts/sa1_2021/1"],
            "source": ["chile", "qhts"],
            "country": ["cl", "aus"],
            "level": ["point", "sa1"],
            "code": [None, "1"],
            "geo": [None, "sa1_2021"],
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

# rough lon/lat bounds of each source's lookups, to catch CRS or axis-order
# mistakes
UK = (-9, 49, 2, 61)
NL = (3, 50.7, 7.3, 53.7)
BOUNDS = {
    "nts": UK,
    "ltds": UK,
    "nhts": (-180, -15, 180, 72),  # states include the Pacific territories
    "cmap": (-93, 36.5, -84.5, 47.5),  # all of IL, IN and WI
    "qhts": (137, -30, 154, -9),
    "vista": (140, -39.5, 150.5, -33.8),
    "mon": NL,
    "ovin": NL,
    "odin": NL,
    "ktdb": (124, 33, 132, 39),
    "argentina": (-74, -56, -53, -21),
    "uruguay": (-59, -35.1, -53, -30),
    "mongolia": (87, 41, 120, 53),
    "france": (-62, -22, 56, 52),  # with the overseas departments
    "india": (68, 6, 98, 38),
    "mexico": (-100, 18, -98, 21),
    "chile": (-72, -34.5, -70, -32.5),
    "brazil": (-47.5, -24.1, -45.6, -23.1),  # São Paulo metropolitan area
}


def declared_lookups(source: str) -> list[str]:
    """The lookups a source's home_location and trip_location read."""
    stems = []
    for dictionary, key in (
        ("hh_dictionary.yaml", "home_location"),
        ("trip_dictionary.yaml", "trip_location"),
    ):
        config = utils.load_yaml_config(
            utils.get_config_path(source, dictionary)
        )
        stems += location.geo_files(config.get(key))
    return stems


def lookup_paths():
    return sorted(utils.get_config_path().glob("*/geo/*.parquet"))


def test_lookups_are_complete_and_plausible():
    paths = lookup_paths()
    assert paths
    for path in paths:
        # every lookup is read by its source's declarations
        assert path.stem in declared_lookups(path.parent.parent.name), path
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
        x0, y0, x1, y1 = BOUNDS[path.parent.parent.name]
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
