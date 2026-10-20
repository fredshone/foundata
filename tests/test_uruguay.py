import os
from pathlib import Path

import polars as pl

from foundata import uruguay
from foundata.utils import get_config_path, load_yaml_config

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
DATA_ROOT = os.getenv("FOUNDATA_URUGUAY_DATA", str(FIXTURE_ROOT / "uruguay"))
CONFIGS_ROOT = get_config_path()


def load_configs():
    return [
        load_yaml_config(CONFIGS_ROOT / "uruguay" / name)
        for name in (
            "hh_dictionary.yaml",
            "person_dictionary.yaml",
            "trip_dictionary.yaml",
        )
    ]


def test_uruguay_load():
    hh_cfg, person_cfg, trips_cfg = load_configs()

    attrs, trips = uruguay.load(Path(DATA_ROOT), hh_cfg, person_cfg, trips_cfg)

    root = Path(DATA_ROOT) / uruguay.SURVEY_DIR
    persons = uruguay.read_table(
        root / uruguay.PERSONS_PATH, {"E19": "travelled"}
    )
    raw_trips = uruguay.read_table(root / uruguay.TRIPS_PATH, {"nvj": "seq"})
    # persons not asked about travel (age 0-3) or with no answer are dropped
    assert len(attrs) == persons["travelled"].is_in(["1", "2"]).sum()
    assert len(trips) == len(raw_trips)
    assert attrs["pid"].is_unique().all()
    assert set(trips["pid"]).issubset(set(attrs["pid"]))
    assert attrs["source"].unique().to_list() == ["uruguay"]
    assert attrs["age"].drop_nulls().min() >= 4
    assert (attrs["hh_size"] < 99).all()
    assert trips.group_by("pid").agg(pl.col("seq").min())["seq"].max() == 0
    assert trips["mode"].is_in(["unknown"]).sum() == 0
    assert (trips["distance"].drop_nulls() > 0).all()
    assert trips["tst"].min() >= 4 * 60


def test_clock_minutes():
    df = pl.DataFrame({"t": ["7", "7,4", "7,05", "17,3", "25,15", "99,99"]})
    result = df.select(
        uruguay.clock_minutes("t", ["88,88", "99,99"]).alias("t")
    )["t"]
    assert result.to_list() == [420, 460, 425, 1050, 1515, None]


def test_zone_type():
    hh_cfg, _, _ = load_configs()
    df = pl.DataFrame({"codloc": [1020, 3320, 3900, 1900, None]})
    result = df.select(uruguay.zone_type(pl.col("codloc"), hh_cfg).alias("z"))[
        "z"
    ]
    assert result.to_list() == [
        "urban",
        "suburban",
        "rural",
        "rural",
        "unknown",
    ]


def test_load_households():
    hh_cfg, _, _ = load_configs()
    hhs = pl.DataFrame(
        {
            "hid": ["1", "2", "3"],
            "dwelling": ["1", "3", "9"],
            "ownership": ["1", "2", "9"],
            "has_cars": ["1", "2", "1"],
            "cars": ["2", None, "99"],
            "has_mopeds": ["1", "2", "2"],
            "mopeds": ["1", None, None],
            "hh_size": ["3", "99", "2"],
            "hh_size_surveyed": ["3", "4", "2"],
            "hh_zone": ["1020", "3320", "3900"],
            "home_segment": ["109009", "121027", None],
        }
    )

    result = uruguay.load_households(hhs, hh_cfg)

    assert result["vehicles"].to_list() == [3, 0, None]
    # D5 99 (sin dato) falls back to the number of persons surveyed
    assert result["hh_size"].to_list() == [3, 4, 2]
    assert result["dwelling"].to_list() == ["house", "flat", "unknown"]
    assert result["ownership"].to_list() == ["owned", "rented", "unknown"]
    assert result["hh_zone"].to_list() == ["urban", "suburban", "rural"]
    assert result["hh_location"].to_list() == [
        "uruguay/segment_2011/109009",
        "uruguay/segment_2011/121027",
        None,
    ]


def test_load_persons():
    _, person_cfg, _ = load_configs()
    persons = pl.DataFrame(
        {
            "hid": ["10"] * 6,
            "pid": ["1", "2", "3", "4", "5", "6"],
            "weight": ["597,16"] * 6,
            "age": ["42", "70", "8", "2", "30", "25"],
            "sex": ["1", "2", "1", "2", "1", "2"],
            "relationship": ["1", "2", "3", "4", "7", "13"],
            "attended": ["2", "2", "1", None, "3", "1"],
            "education": ["3", "6", "2", None, None, "6"],
            "completed": ["1", "1", "2", None, None, "2"],
            "worked": ["2", "2", None, None, "1", "2"],
            "has_job": ["2", "1", None, None, None, "1"],
            "employment": [None, "4", None, None, None, "2"],
            "has_licence": ["1", "2", None, None, "3", "9"],
            "travelled": ["1", "2", "1", None, "1", "9"],
        }
    )

    result = uruguay.load_persons(persons, person_cfg)

    # the 2-year-old (not asked) and the E19 sin dato person are dropped
    assert result["pid"].to_list() == ["1001", "1002", "1003", "1005"]
    assert result["weight"][0] == pl.Series([597.16], dtype=pl.Float32)[0]
    # completed secundaria -> high-school, completed university -> degree,
    # never attended -> none
    assert result["education"].to_list() == [
        "high-school",
        "degree",
        "school",
        "none",
    ]
    # has a job to return to (E9 = 2) counts as employed
    assert result["employment"].to_list() == [
        "employed",
        "retired",
        "unknown",
        "employed",
    ]
    assert result["has_licence"].to_list() == ["yes", "no", "void", "no"]
    assert result["occupation"].to_list() == [
        "unknown",
        "void",
        "void",
        "unknown",
    ]
    assert result["relationship"].to_list() == [
        "self",
        "partner",
        "child",
        "parent",
    ]


def segments() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "codseg": ["101001", "101002", "303107"],
            "x": [0.0, 3000.0, 0.0],
            "y": [0.0, 4000.0, 10000.0],
            "zone": ["urban", "urban", "rural"],
        }
    )


def test_load_trips_distances():
    _, _, trips_cfg = load_configs()
    trips = pl.DataFrame(
        {
            "hid": ["10"] * 4,
            "pid": ["1"] * 4,
            "seq": ["1", "2", "3", "4"],
            "oact": ["1", "2", "7", "13"],
            "dact": ["2", "7", "13", "1"],
            "tst": ["7", "12", "13", "18"],
            "tet": ["7,3", "12,1", "14", "18,3"],
            "mode": ["13", "0", "8", "8"],
            "blocks": [None, "4", None, None],
            "ozone": ["101001", "101002", "101002", "101002"],
            "dzone": ["101002", "101002", "101002", "999999"],
        }
    )

    result = uruguay.load_trips(trips, trips_cfg, segments())

    distance = result["distance"].to_list()
    # 5 km between segments x circuity 1.3
    assert abs(distance[0] - 6.5) < 1e-4
    # walk trip: 4 blocks x 100 m
    assert abs(distance[1] - 0.4) < 1e-4
    # same segment: 60 min by car at 20 km/h
    assert abs(distance[2] - 20.0) < 1e-4
    # destination not in the lookup: 30 min at 20 km/h, zone unknown
    assert abs(distance[3] - 10.0) < 1e-4
    assert result["dzone"].to_list()[3] == "unknown"
    assert result["seq"].to_list() == [0, 1, 2, 3]
    assert result["mode"].to_list() == ["bus", "walk", "car", "car"]
    assert result["oact"].to_list() == ["home", "work", "shop", "visit"]
    assert result["dact"].to_list() == ["work", "shop", "visit", "home"]
    assert result["pid"].unique().to_list() == ["1001"]


def test_access_egress_distances():
    _, _, trips_cfg = load_configs()
    stages = pl.DataFrame(
        {
            "hid": ["10"] * 4,
            "pid": ["1"] * 4,
            "seq": ["1", "1", "1", "2"],
            "stage_mode": ["2", "51", "2", "11"],
            "blocks_to_vehicle": [None, "2", None, "1"],
            "blocks_from_vehicle": [None, "0", None, "1"],
            "walk_stage_blocks": ["7", None, "99", None],
        }
    )

    result = uruguay.access_egress_distances(stages, trips_cfg).sort("seq")

    # bus trip: 7 + 2 + 0 blocks (99 sin dato ignored); car trip: none
    assert abs(result["access_egress_distance"][0] - 0.9) < 1e-4
    assert result["access_egress_distance"][1] is None
