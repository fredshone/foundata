from pathlib import Path

import polars as pl
import pytest

from foundata import post_process

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "post_process"
TRIPS_CSV = FIXTURE_DIR / "trips.csv"

TRIPS_SCHEMA = {
    "pid": pl.String,
    "seq": pl.Int32,
    "ozone": pl.String,
    "dzone": pl.String,
    "oact": pl.String,
    "dact": pl.String,
    "mode": pl.String,
    "tst": pl.Int32,
    "tet": pl.Int32,
    "distance": pl.Float32,
}


def _make_trips(rows):
    return pl.DataFrame(
        rows,
        schema={
            "pid": pl.String,
            "seq": pl.Int32,
            "oact": pl.String,
            "dact": pl.String,
            "ozone": pl.String,
            "dzone": pl.String,
            "tst": pl.Int32,
            "tet": pl.Int32,
        },
    )


def _make_attributes(rows):
    return pl.DataFrame(
        rows,
        schema={
            "pid": pl.String,
            "hh_zone": pl.String,
        },
    )


def test_trips_to_activities_basic():
    attrs = _make_attributes([{"pid": "p1", "hh_zone": "urban"}])
    trips = _make_trips([
        {"pid": "p1", "seq": 1, "oact": "home", "dact": "work", "ozone": "z1", "dzone": "z2", "tst": 480, "tet": 540},
        {"pid": "p1", "seq": 2, "oact": "work", "dact": "shop", "ozone": "z2", "dzone": "z3", "tst": 900, "tet": 920},
        {"pid": "p1", "seq": 3, "oact": "shop", "dact": "home", "ozone": "z3", "dzone": "z1", "tst": 960, "tet": 1020},
    ])
    acts = post_process.trips_to_activities(attrs, trips)
    assert len(acts) == 4

    row0 = acts.row(0, named=True)
    assert row0["act"] == "home"
    assert row0["zone"] == "z1"
    assert row0["ast"] == 0
    assert row0["aet"] == 480

    row1 = acts.row(1, named=True)
    assert row1["act"] == "work"
    assert row1["zone"] == "z2"
    assert row1["ast"] == 540
    assert row1["aet"] == 900

    row2 = acts.row(2, named=True)
    assert row2["act"] == "shop"
    assert row2["zone"] == "z3"
    assert row2["ast"] == 920
    assert row2["aet"] == 960

    row3 = acts.row(3, named=True)
    assert row3["act"] == "home"
    assert row3["zone"] == "z1"
    assert row3["ast"] == 1020
    assert row3["aet"] == 1440


def test_trips_to_activities_single_trip():
    attrs = _make_attributes([{"pid": "p1", "hh_zone": "urban"}])
    trips = _make_trips([
        {"pid": "p1", "seq": 1, "oact": "home", "dact": "work", "ozone": "z1", "dzone": "z2", "tst": 480, "tet": 540},
    ])
    acts = post_process.trips_to_activities(attrs, trips)
    assert len(acts) == 2


def test_trips_to_activities_multi_person():
    attrs = _make_attributes([{"pid": "p1", "hh_zone": "urban"}, {"pid": "p2", "hh_zone": "urban"}])
    trips = _make_trips([
        {"pid": "p1", "seq": 1, "oact": "home", "dact": "work", "ozone": "z1", "dzone": "z2", "tst": 480, "tet": 540},
        {"pid": "p1", "seq": 2, "oact": "work", "dact": "home", "ozone": "z2", "dzone": "z1", "tst": 1020, "tet": 1080},
        {"pid": "p2", "seq": 1, "oact": "home", "dact": "shop", "ozone": "z1", "dzone": "z3", "tst": 600, "tet": 630},
        {"pid": "p2", "seq": 2, "oact": "shop", "dact": "home", "ozone": "z3", "dzone": "z1", "tst": 700, "tet": 730},
    ])
    acts = post_process.trips_to_activities(attrs, trips)
    assert len(acts) == 6
    pids = acts.get_column("pid").to_list()
    assert pids.count("p1") == 3
    assert pids.count("p2") == 3


def test_trips_to_activities_columns():
    attrs = _make_attributes([{"pid": "p1", "hh_zone": "urban"}])
    trips = _make_trips([
        {"pid": "p1", "seq": 1, "oact": "home", "dact": "work", "ozone": "z1", "dzone": "z2", "tst": 480, "tet": 540},
        {"pid": "p1", "seq": 2, "oact": "work", "dact": "home", "ozone": "z2", "dzone": "z1", "tst": 1020, "tet": 1080},
    ])
    acts = post_process.trips_to_activities(attrs, trips)
    assert set(acts.columns) == {"pid", "seq", "act", "zone", "ast", "aet"}


def test_trips_to_activities_includes_start_of_day():
    attrs = _make_attributes([{"pid": "p1", "hh_zone": "urban"}])
    trips = _make_trips([
        {"pid": "p1", "seq": 1, "oact": "home", "dact": "work", "ozone": "z1", "dzone": "z2", "tst": 480, "tet": 540},
        {"pid": "p1", "seq": 2, "oact": "work", "dact": "home", "ozone": "z2", "dzone": "z1", "tst": 1020, "tet": 1080},
    ])
    acts = post_process.trips_to_activities(attrs, trips)
    row = acts.row(0, named=True)
    assert row["ast"] == 0
    assert row["aet"] == 480
    assert row["act"] == "home"
    assert row["zone"] == "z1"


def test_trips_to_activities_includes_end_of_day():
    attrs = _make_attributes([{"pid": "p1", "hh_zone": "urban"}])
    trips = _make_trips([
        {"pid": "p1", "seq": 1, "oact": "home", "dact": "work", "ozone": "z1", "dzone": "z2", "tst": 480, "tet": 540},
        {"pid": "p1", "seq": 2, "oact": "work", "dact": "home", "ozone": "z2", "dzone": "z1", "tst": 1020, "tet": 1080},
    ])
    acts = post_process.trips_to_activities(attrs, trips)
    row = acts.row(-1, named=True)
    assert row["aet"] == 1440
    assert row["ast"] == 1080
    assert row["act"] == "home"
    assert row["zone"] == "z1"


def test_trips_with_following_activity_basic():
    attrs = _make_attributes([{"pid": "p1", "hh_zone": "urban"}])
    trips = _make_trips([
        {"pid": "p1", "seq": 1, "oact": "home", "dact": "work", "ozone": "z1", "dzone": "z2", "tst": 480, "tet": 540},
        {"pid": "p1", "seq": 2, "oact": "work", "dact": "shop", "ozone": "z2", "dzone": "z3", "tst": 900, "tet": 920},
        {"pid": "p1", "seq": 3, "oact": "shop", "dact": "home", "ozone": "z3", "dzone": "z1", "tst": 960, "tet": 1020},
    ])
    result = post_process.trips_with_following_activity(attrs, trips)
    assert len(result) == 3
    last = result.row(-1, named=True)
    assert last["aet"] == 1440


def test_trips_with_following_activity_columns():
    attrs = _make_attributes([{"pid": "p1", "hh_zone": "urban"}])
    trips = _make_trips([
        {"pid": "p1", "seq": 1, "oact": "home", "dact": "work", "ozone": "z1", "dzone": "z2", "tst": 480, "tet": 540},
        {"pid": "p1", "seq": 2, "oact": "work", "dact": "home", "ozone": "z2", "dzone": "z1", "tst": 1020, "tet": 1080},
    ])
    result = post_process.trips_with_following_activity(attrs, trips)
    assert set(result.columns) == set(trips.columns) | {"aet"}


@pytest.fixture
def fixture_trips():
    return pl.read_csv(TRIPS_CSV, schema_overrides=TRIPS_SCHEMA)


@pytest.fixture
def fixture_attrs(fixture_trips):
    return fixture_trips.select("pid", "ozone").unique("pid").rename({"ozone": "hh_zone"})


def test_trips_to_activities_fixture(fixture_attrs, fixture_trips):
    acts = post_process.trips_to_activities(fixture_attrs, fixture_trips)

    assert set(acts.columns) == {"pid", "seq", "act", "zone", "ast", "aet"}
    assert (acts["ast"] >= 0).all()
    assert (acts["aet"] <= 1440).all()
    assert (acts["ast"] <= acts["aet"]).all()  # zero-duration activities are valid (tet[i]==tst[i+1])

    # Each person with N trips produces N+1 activities
    trip_counts = fixture_trips.group_by("pid").len()
    act_counts = acts.group_by("pid").len()
    joined = trip_counts.join(act_counts, on="pid", suffix="_acts")
    for row in joined.iter_rows(named=True):
        assert row["len_acts"] == row["len"] + 1, f"pid={row['pid']}: {row['len']} trips → expected {row['len'] + 1} activities, got {row['len_acts']}"

    # Activities sorted by ast within each person
    for pid in fixture_trips["pid"].unique():
        person_acts = acts.filter(pl.col("pid") == pid).sort("ast")
        ast_vals = person_acts["ast"].to_list()
        assert ast_vals == sorted(ast_vals), f"pid={pid} activities not sorted by ast"


def test_trips_with_following_activity_fixture(fixture_attrs, fixture_trips):
    result = post_process.trips_with_following_activity(fixture_attrs, fixture_trips)

    # All input columns preserved plus aet
    assert set(fixture_trips.columns) | {"aet"} == set(result.columns)

    # All valid trips (tet < 1440) are included
    valid_trip_count = fixture_trips.filter(pl.col("tet") < 1440).shape[0]
    assert len(result) == valid_trip_count

    # Last trip per person has aet == 1440
    last_trips = result.sort("pid", "seq").group_by("pid").last()
    assert (last_trips["aet"] == 1440).all()

    # For non-last trips: aet == next trip's tst
    sorted_result = result.sort("pid", "seq")
    next_tst = sorted_result.with_columns(
        next_tst=pl.col("tst").shift(-1).over("pid")
    ).filter(pl.col("aet") != 1440)
    mismatches = next_tst.filter(pl.col("aet") != pl.col("next_tst"))
    assert len(mismatches) == 0, f"{len(mismatches)} non-last trips have aet != next trip's tst"


def test_trips_to_activities_no_trips_person():
    attrs = _make_attributes([
        {"pid": "p1", "hh_zone": "urban"},
        {"pid": "p2", "hh_zone": "suburban"},
    ])
    trips = _make_trips([
        {"pid": "p1", "seq": 1, "oact": "home", "dact": "work", "ozone": "z1", "dzone": "z2", "tst": 480, "tet": 540},
        {"pid": "p1", "seq": 2, "oact": "work", "dact": "home", "ozone": "z2", "dzone": "z1", "tst": 1020, "tet": 1080},
    ])
    acts = post_process.trips_to_activities(attrs, trips)
    assert len(acts) == 4

    p1_acts = acts.filter(pl.col("pid") == "p1")
    assert len(p1_acts) == 3

    p2_acts = acts.filter(pl.col("pid") == "p2")
    assert len(p2_acts) == 1
    row = p2_acts.row(0, named=True)
    assert row["ast"] == 0
    assert row["aet"] == 1440
    assert row["act"] == "home"
    assert row["zone"] == "suburban"
