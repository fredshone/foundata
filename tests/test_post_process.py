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
    return pl.DataFrame(rows, schema={"pid": pl.String, "hh_zone": pl.String})


def test_trips_to_activities_basic():
    attrs = _make_attributes([{"pid": "p1", "hh_zone": "urban"}])
    trips = _make_trips(
        [
            {
                "pid": "p1",
                "seq": 1,
                "oact": "home",
                "dact": "work",
                "ozone": "z1",
                "dzone": "z2",
                "tst": 480,
                "tet": 540,
            },
            {
                "pid": "p1",
                "seq": 2,
                "oact": "work",
                "dact": "shop",
                "ozone": "z2",
                "dzone": "z3",
                "tst": 900,
                "tet": 920,
            },
            {
                "pid": "p1",
                "seq": 3,
                "oact": "shop",
                "dact": "home",
                "ozone": "z3",
                "dzone": "z1",
                "tst": 960,
                "tet": 1020,
            },
        ]
    )
    acts = post_process.trips_to_activities(attrs, trips)
    assert len(acts) == 4

    row0 = acts.row(0, named=True)
    assert row0["act"] == "home"
    assert row0["zone"] == "z1"
    assert row0["start"] == 0
    assert row0["end"] == 480

    row1 = acts.row(1, named=True)
    assert row1["act"] == "work"
    assert row1["zone"] == "z2"
    assert row1["start"] == 480
    assert row1["end"] == 900

    row2 = acts.row(2, named=True)
    assert row2["act"] == "shop"
    assert row2["zone"] == "z3"
    assert row2["start"] == 900
    assert row2["end"] == 960

    row3 = acts.row(3, named=True)
    assert row3["act"] == "home"
    assert row3["zone"] == "z1"
    assert row3["start"] == 960
    assert row3["end"] == 1440


def test_trips_to_activities_single_trip():
    attrs = _make_attributes([{"pid": "p1", "hh_zone": "urban"}])
    trips = _make_trips(
        [
            {
                "pid": "p1",
                "seq": 1,
                "oact": "home",
                "dact": "work",
                "ozone": "z1",
                "dzone": "z2",
                "tst": 480,
                "tet": 540,
            }
        ]
    )
    acts = post_process.trips_to_activities(attrs, trips)
    assert len(acts) == 2


def test_trips_to_activities_multi_person():
    attrs = _make_attributes(
        [{"pid": "p1", "hh_zone": "urban"}, {"pid": "p2", "hh_zone": "urban"}]
    )
    trips = _make_trips(
        [
            {
                "pid": "p1",
                "seq": 1,
                "oact": "home",
                "dact": "work",
                "ozone": "z1",
                "dzone": "z2",
                "tst": 480,
                "tet": 540,
            },
            {
                "pid": "p1",
                "seq": 2,
                "oact": "work",
                "dact": "home",
                "ozone": "z2",
                "dzone": "z1",
                "tst": 1020,
                "tet": 1080,
            },
            {
                "pid": "p2",
                "seq": 1,
                "oact": "home",
                "dact": "shop",
                "ozone": "z1",
                "dzone": "z3",
                "tst": 600,
                "tet": 630,
            },
            {
                "pid": "p2",
                "seq": 2,
                "oact": "shop",
                "dact": "home",
                "ozone": "z3",
                "dzone": "z1",
                "tst": 700,
                "tet": 730,
            },
        ]
    )
    acts = post_process.trips_to_activities(attrs, trips)
    assert len(acts) == 6
    pids = acts.get_column("pid").to_list()
    assert pids.count("p1") == 3
    assert pids.count("p2") == 3


def test_trips_to_activities_columns():
    attrs = _make_attributes([{"pid": "p1", "hh_zone": "urban"}])
    trips = _make_trips(
        [
            {
                "pid": "p1",
                "seq": 1,
                "oact": "home",
                "dact": "work",
                "ozone": "z1",
                "dzone": "z2",
                "tst": 480,
                "tet": 540,
            },
            {
                "pid": "p1",
                "seq": 2,
                "oact": "work",
                "dact": "home",
                "ozone": "z2",
                "dzone": "z1",
                "tst": 1020,
                "tet": 1080,
            },
        ]
    )
    acts = post_process.trips_to_activities(attrs, trips)
    assert set(acts.columns) == {"pid", "seq", "act", "zone", "start", "end"}


def test_trips_to_activities_includes_start_of_day():
    attrs = _make_attributes([{"pid": "p1", "hh_zone": "urban"}])
    trips = _make_trips(
        [
            {
                "pid": "p1",
                "seq": 1,
                "oact": "home",
                "dact": "work",
                "ozone": "z1",
                "dzone": "z2",
                "tst": 480,
                "tet": 540,
            },
            {
                "pid": "p1",
                "seq": 2,
                "oact": "work",
                "dact": "home",
                "ozone": "z2",
                "dzone": "z1",
                "tst": 1020,
                "tet": 1080,
            },
        ]
    )
    acts = post_process.trips_to_activities(attrs, trips)
    row = acts.row(0, named=True)
    assert row["start"] == 0
    assert row["end"] == 480
    assert row["act"] == "home"
    assert row["zone"] == "z1"


def test_trips_to_activities_includes_end_of_day():
    attrs = _make_attributes([{"pid": "p1", "hh_zone": "urban"}])
    trips = _make_trips(
        [
            {
                "pid": "p1",
                "seq": 1,
                "oact": "home",
                "dact": "work",
                "ozone": "z1",
                "dzone": "z2",
                "tst": 480,
                "tet": 540,
            },
            {
                "pid": "p1",
                "seq": 2,
                "oact": "work",
                "dact": "home",
                "ozone": "z2",
                "dzone": "z1",
                "tst": 1020,
                "tet": 1080,
            },
        ]
    )
    acts = post_process.trips_to_activities(attrs, trips)
    row = acts.row(-1, named=True)
    assert row["end"] == 1440
    assert row["start"] == 1020
    assert row["act"] == "home"
    assert row["zone"] == "z1"


@pytest.fixture
def fixture_trips():
    return pl.read_csv(TRIPS_CSV, schema_overrides=TRIPS_SCHEMA)


@pytest.fixture
def fixture_attrs(fixture_trips):
    return (
        fixture_trips.select("pid", "ozone")
        .unique("pid")
        .rename({"ozone": "hh_zone"})
    )


def test_trips_to_activities_fixture(fixture_attrs, fixture_trips):
    acts = post_process.trips_to_activities(fixture_attrs, fixture_trips)

    assert set(acts.columns) == {"pid", "seq", "act", "zone", "start", "end"}
    assert (acts["start"] >= 0).all()
    assert (acts["end"] <= 1440).all()
    assert (
        acts["start"] <= acts["end"]
    ).all()  # zero-duration activities are valid (tet[i]==tst[i+1])

    # Each person with N trips produces N+1 activities
    trip_counts = fixture_trips.group_by("pid").len()
    act_counts = acts.group_by("pid").len()
    joined = trip_counts.join(act_counts, on="pid", suffix="_acts")
    for row in joined.iter_rows(named=True):
        assert (
            row["len_acts"] == row["len"] + 1
        ), f"pid={row['pid']}: {row['len']} trips → expected {row['len'] + 1} activities, got {row['len_acts']}"

    # Activities sorted by start within each person
    for pid in fixture_trips["pid"].unique():
        person_acts = acts.filter(pl.col("pid") == pid).sort("start")
        start_vals = person_acts["start"].to_list()
        assert start_vals == sorted(
            start_vals
        ), f"pid={pid} activities not sorted by start time"


def test_trips_to_activities_no_trips_person():
    attrs = _make_attributes(
        [
            {"pid": "p1", "hh_zone": "urban"},
            {"pid": "p2", "hh_zone": "suburban"},
        ]
    )
    trips = _make_trips(
        [
            {
                "pid": "p1",
                "seq": 1,
                "oact": "home",
                "dact": "work",
                "ozone": "z1",
                "dzone": "z2",
                "tst": 480,
                "tet": 540,
            },
            {
                "pid": "p1",
                "seq": 2,
                "oact": "work",
                "dact": "home",
                "ozone": "z2",
                "dzone": "z1",
                "tst": 1020,
                "tet": 1080,
            },
        ]
    )
    acts = post_process.trips_to_activities(attrs, trips)
    assert len(acts) == 4

    p1_acts = acts.filter(pl.col("pid") == "p1")
    assert len(p1_acts) == 3

    p2_acts = acts.filter(pl.col("pid") == "p2")
    assert len(p2_acts) == 1
    row = p2_acts.row(0, named=True)
    assert row["start"] == 0
    assert row["end"] == 1440
    assert row["act"] == "home"
    assert row["zone"] == "suburban"


def test_discretise_numeric_quantile_basic():
    df = pl.DataFrame(
        {"pid": ["a", "b", "c", "d", "e"], "age": [10, 20, 30, 40, 50]}
    )
    result = post_process.discretise_numeric(df, n_bins=2, method="quantile")
    assert result["age"].dtype == pl.String
    assert result["pid"].dtype == pl.String  # non-numeric untouched


def test_discretise_numeric_uniform_basic():
    df = pl.DataFrame({"age": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
    result = post_process.discretise_numeric(df, n_bins=5, method="uniform")
    assert result["age"].dtype == pl.String
    assert result["age"].null_count() == 0


def test_discretise_numeric_preserves_nulls():
    df = pl.DataFrame(
        {"age": pl.Series([10, None, 30, None, 50], dtype=pl.Int32)}
    )
    result = post_process.discretise_numeric(df, n_bins=2, method="quantile")
    assert result["age"].null_count() == 2


def test_discretise_numeric_cols_subset():
    df = pl.DataFrame(
        {"age": [10, 20, 30], "vehicles": [0, 1, 2], "pid": ["a", "b", "c"]}
    )
    result = post_process.discretise_numeric(df, cols=["age"])
    assert result["age"].dtype == pl.String
    assert result["vehicles"].dtype != pl.String  # vehicles unchanged


def test_discretise_numeric_label_format():
    df = pl.DataFrame({"age": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
    result = post_process.discretise_numeric(df, n_bins=3, method="uniform")
    labels = set(result["age"].drop_nulls().to_list())
    assert all(
        "-" in l or l.startswith("≤") or l.startswith(">") for l in labels
    )


def test_fill_nulls_string_cols():
    df = pl.DataFrame(
        {
            "mode": pl.Series(["walk", None, "car"], dtype=pl.String),
            "age": pl.Series(["25-40", None, "40-60"], dtype=pl.Utf8),
        }
    )
    result = post_process.fill_nulls(df)
    assert result["mode"].to_list() == ["walk", "unknown", "car"]
    assert result["age"].to_list() == ["25-40", "unknown", "40-60"]


def test_fill_nulls_empty_strings():
    df = pl.DataFrame({"mode": pl.Series(["walk", "", "car"], dtype=pl.String)})
    result = post_process.fill_nulls(df)
    assert result["mode"].to_list() == ["walk", "unknown", "car"]


def test_fill_nulls_numeric_filled_with_minus_one():
    df = pl.DataFrame(
        {
            "vehicles": pl.Series([1, None, 3], dtype=pl.Int32),
            "weight": pl.Series([1.5, None, 2.0], dtype=pl.Float32),
        }
    )
    result = post_process.fill_nulls(df)
    assert result["vehicles"].to_list() == [1, -1, 3]
    assert result["weight"].to_list() == [1.5, -1.0, 2.0]


def test_discretise_numeric_invalid_method():
    df = pl.DataFrame({"age": [10, 20, 30]})
    with pytest.raises(ValueError, match="method must be"):
        post_process.discretise_numeric(df, method="bad")
