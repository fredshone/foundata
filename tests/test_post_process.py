import polars as pl

from foundata import post_process


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


def test_trips_to_activities_basic():
    trips = _make_trips([
        {"pid": "p1", "seq": 1, "oact": "home", "dact": "work", "ozone": "z1", "dzone": "z2", "tst": 480, "tet": 540},
        {"pid": "p1", "seq": 2, "oact": "work", "dact": "shop", "ozone": "z2", "dzone": "z3", "tst": 900, "tet": 920},
        {"pid": "p1", "seq": 3, "oact": "shop", "dact": "home", "ozone": "z3", "dzone": "z1", "tst": 960, "tet": 1020},
    ])
    acts = post_process.trips_to_activities(trips)
    assert len(acts) == 2
    row0 = acts.row(0, named=True)
    assert row0["act"] == "work"
    assert row0["zone"] == "z2"
    assert row0["ast"] == 540
    assert row0["aet"] == 900
    row1 = acts.row(1, named=True)
    assert row1["act"] == "shop"
    assert row1["zone"] == "z3"
    assert row1["ast"] == 920
    assert row1["aet"] == 960


def test_trips_to_activities_single_trip():
    trips = _make_trips([
        {"pid": "p1", "seq": 1, "oact": "home", "dact": "work", "ozone": "z1", "dzone": "z2", "tst": 480, "tet": 540},
    ])
    acts = post_process.trips_to_activities(trips)
    assert len(acts) == 0


def test_trips_to_activities_multi_person():
    trips = _make_trips([
        {"pid": "p1", "seq": 1, "oact": "home", "dact": "work", "ozone": "z1", "dzone": "z2", "tst": 480, "tet": 540},
        {"pid": "p1", "seq": 2, "oact": "work", "dact": "home", "ozone": "z2", "dzone": "z1", "tst": 1020, "tet": 1080},
        {"pid": "p2", "seq": 1, "oact": "home", "dact": "shop", "ozone": "z1", "dzone": "z3", "tst": 600, "tet": 630},
        {"pid": "p2", "seq": 2, "oact": "shop", "dact": "home", "ozone": "z3", "dzone": "z1", "tst": 700, "tet": 730},
    ])
    acts = post_process.trips_to_activities(trips)
    assert len(acts) == 2
    pids = acts.get_column("pid").to_list()
    assert pids.count("p1") == 1
    assert pids.count("p2") == 1


def test_trips_to_activities_columns():
    trips = _make_trips([
        {"pid": "p1", "seq": 1, "oact": "home", "dact": "work", "ozone": "z1", "dzone": "z2", "tst": 480, "tet": 540},
        {"pid": "p1", "seq": 2, "oact": "work", "dact": "home", "ozone": "z2", "dzone": "z1", "tst": 1020, "tet": 1080},
    ])
    acts = post_process.trips_to_activities(trips)
    assert set(acts.columns) == {"pid", "seq", "act", "zone", "ast", "aet"}
