import os
from pathlib import Path

import polars as pl

from foundata import chile
from foundata.utils import get_config_path, load_yaml_config

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
DATA_ROOT = os.getenv("FOUNDATA_CHILE_DATA", str(FIXTURE_ROOT / "chile"))
CONFIGS_ROOT = get_config_path()


def load_configs():
    return [
        load_yaml_config(CONFIGS_ROOT / "chile" / name)
        for name in (
            "hh_dictionary.yaml",
            "person_dictionary.yaml",
            "trip_dictionary.yaml",
        )
    ]


def test_chile_load():
    hh_cfg, person_cfg, trips_cfg = load_configs()

    attrs, trips = chile.load(Path(DATA_ROOT), hh_cfg, person_cfg, trips_cfg)

    root = Path(DATA_ROOT)
    persons = pl.read_parquet(root / chile.PERSONS_PATH)
    raw_trips = pl.read_parquet(root / chile.TRIPS_PATH)
    # every person is kept, including those with no trips
    assert len(attrs) == len(persons)
    assert len(trips) == len(raw_trips)
    assert attrs["pid"].is_unique().all()
    assert set(trips["pid"]).issubset(set(attrs["pid"]))
    assert attrs["pid"].str.starts_with("chile").all()
    assert attrs["hid"].str.starts_with("chile").all()
    assert attrs["source"].unique().to_list() == ["chile"]
    assert attrs["country"].unique().to_list() == ["cl"]
    assert set(attrs["year"].unique()) <= {2012, 2013}
    assert (
        attrs["day"]
        .is_in(
            [
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            ]
        )
        .all()
    )
    assert attrs["weight"].dtype == pl.Float32
    assert trips.group_by("pid").agg(pl.col("seq").min())["seq"].max() == 0
    assert trips["seq"].dtype == pl.Int8
    assert trips["distance"].dtype == pl.Float32
    assert (trips["distance"].drop_nulls() >= 0).all()
    assert trips["mode"].is_in(["unknown"]).sum() <= 1


def test_clock_minutes():
    df = pl.DataFrame({"t": ["7:05", "13:00", "0:20", "23:59", None]})
    result = df.select(chile.clock_minutes("t"))["t"]
    assert result.to_list() == [425, 780, 20, 1439, None]


def test_employment():
    _, person_cfg, _ = load_configs()
    df = pl.DataFrame(
        {
            "activity": ["A", "A;B", "B;D", "C;D", "D", "F", "G", None, "A"],
            "work_hours": [1, 2, None, None, None, None, None, None, None],
        }
    )
    result = df.select(chile.employment(person_cfg).alias("e"))["e"]
    assert result.to_list() == [
        "ft-employed",
        "pt-employed",
        "student",
        "retired",
        "unemployed",
        "unemployed",
        "unknown",
        "unknown",
        "employed",
    ]


def test_load_persons():
    _, person_cfg, _ = load_configs()
    persons = pl.DataFrame(
        {
            "hid": [1, 1, 2],
            "pid": [101, 102, 201],
            "age": [40, 8, 70],
            "sex": [1, 2, 2],
            "relationship": [1, 3, 7],
            "education": [11.0, 4.0, 99.0],
            "activity": ["A", "B", "C"],
            "work_hours": [1.0, None, None],
            "occupation": [6.0, None, None],
            "has_licence": ["1;3", "1", "5"],
            "disability": ["J", "J", "D;H"],
            "income_imputed": [0, 0, 1],
            "weight": [10.5, 20.0, 5.0],
        }
    )
    result = chile.load_persons(persons, person_cfg)
    assert result["pid"].to_list() == ["101", "102", "201"]
    assert result["relationship"].to_list() == ["self", "child", "unrelated"]
    assert result["education"].to_list() == ["degree", "school", "unknown"]
    assert result["occupation"].to_list() == ["service", "void", "void"]
    assert result["has_licence"].to_list() == ["yes", "no", "no"]
    assert result["disability"].to_list() == ["no", "no", "yes"]


def test_load_households():
    hh_cfg, _, _ = load_configs()
    hhs = pl.DataFrame(
        {
            "hid": [1, 2],
            "hh_zone": [1, 999],
            "date": ["14-04-2013", "10-08-2012"],
            "hh_size": [2, 1],
            "ownership": [3, 5],
            "hh_income": [100_000, 50_000],
        }
    )
    vehicles = pl.DataFrame({"hid": [1, 1, 1], "vehicle_type": [1, 2, 8]})
    persons = pl.DataFrame(
        {"hid": ["1", "1", "2"], "income_imputed": [0, 2, 0]}
    )
    zones = pl.DataFrame(
        {"zone": [1], "x": [0.0], "y": [0.0], "zone_type": ["urban"]}
    )
    result = chile.load_households(hhs, vehicles, persons, hh_cfg, zones)
    assert result["day"].to_list() == ["sunday", "friday"]
    assert result["month"].to_list() == [4, 8]
    assert result["year"].to_list() == [2013, 2012]
    # taxis (8) aren't household vehicles
    assert result["vehicles"].to_list() == [2, 0]
    assert result["ownership"].to_list() == ["rented", "other"]
    assert result["hh_zone"].to_list() == ["urban", "unknown"]
    rate = hh_cfg["eur_per_clp"]
    assert result["hh_income"].to_list() == [
        round(100_000 * 12 * rate[2013]),
        round(50_000 * 12 * rate[2012]),
    ]

    # dropping imputation code 2 nulls the first household's income
    strict = {**hh_cfg, "income_imputation_codes": [0]}
    result = chile.load_households(hhs, vehicles, persons, strict, zones)
    assert result["hh_income"].to_list()[0] is None


def test_trip_modes():
    _, _, trips_cfg = load_configs()
    stages = pl.DataFrame(
        {
            "trip_id": [1, 1, 2, 3, 3],
            "stage_mode": [1, 4, 8, 3, 2],  # car+metro, walk, bus+bus
            "blocks_before": [0.0, 3.0, 0.0, 2.0, 99.0],
        }
    )
    result = chile.trip_modes(stages, trips_cfg).sort("trip_id")
    assert result["mode"].to_list() == ["rail", "walk", "bus"]
    assert result["transit"].to_list() == [True, False, True]
    # unknown block counts (99) are skipped
    assert result["blocks_before"].to_list() == [3.0, 0.0, 2.0]


def test_trip_distance():
    _, _, trips_cfg = load_configs()
    df = pl.DataFrame(
        {
            "straight_line": [2000, -1, -1, 0],
            "ozone": [1, 1, 5, 5],
            "dzone": [2, 2, 5, 5],
            "ox": [0.0, 0.0, 0.0, 0.0],
            "oy": [0.0, 0.0, 0.0, 0.0],
            "dx": [3000.0, 3000.0, 0.0, 0.0],
            "dy": [4000.0, 4000.0, 0.0, 0.0],
            "duration": [30.0, 30.0, 30.0, 20.0],
            "mode": ["car", "car", "walk", "bus"],
        }
    )
    result = df.select(chile.trip_distance(trips_cfg).alias("d"))["d"]
    circuity = trips_cfg["circuity"]
    speed = trips_cfg["speed"]
    expected = [
        2 * circuity,  # survey straight line
        5 * circuity,  # zone centroids
        0.5 * speed["walk"],  # same zone: duration x speed
        20 / 60 * speed["bus"],  # same block (0 m): duration x speed
    ]
    assert [round(x, 4) for x in result.to_list()] == [
        round(x, 4) for x in expected
    ]
