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


def _make_activities(rows):
    return pl.DataFrame(
        rows,
        schema={
            "pid": pl.String,
            "seq": pl.Int32,
            "act": pl.String,
            "zone": pl.String,
            "start": pl.Int32,
            "end": pl.Int32,
        },
    )


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
    assert row1["start"] == 540
    assert row1["end"] == 900

    row2 = acts.row(2, named=True)
    assert row2["act"] == "shop"
    assert row2["zone"] == "z3"
    assert row2["start"] == 920
    assert row2["end"] == 960

    row3 = acts.row(3, named=True)
    assert row3["act"] == "home"
    assert row3["zone"] == "z1"
    assert row3["start"] == 1020
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
    assert row["start"] == 1080
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
        assert row["len_acts"] == row["len"] + 1, (
            f"pid={row['pid']}: {row['len']} trips → expected {row['len'] + 1} activities, got {row['len_acts']}"
        )

    # Activities sorted by start within each person
    for pid in fixture_trips["pid"].unique():
        person_acts = acts.filter(pl.col("pid") == pid).sort("start")
        start_vals = person_acts["start"].to_list()
        assert start_vals == sorted(start_vals), (
            f"pid={pid} activities not sorted by start time"
        )


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


def test_trips_to_activities_last_trip_at_midnight_generates_final_activity():
    """Last trip ends at tet=1440 — its destination activity must still be created.

    Pattern seen in NTS: plan has two trips (home→work, work→home) where the
    return-home trip ends exactly at midnight. Without the fix, the home
    destination is silently dropped and the plan appears to end at 'work'.
    """
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
                "tet": 1440,  # ends exactly at midnight
            },
        ]
    )
    acts = post_process.trips_to_activities(attrs, trips)
    assert len(acts) == 3
    last = acts.sort("seq").row(-1, named=True)
    assert last["act"] == "home"
    assert last["end"] == 1440


def test_trips_to_activities_multi_trip_last_at_midnight_generates_final_activity():
    """Multi-trip plan where last trip (other→home, tet=1440) must appear in output.

    Pattern: nts200200005603-style plan where 6 intermediate trips all have
    tet < 1440 and are visible, but the 7th return-home trip has tet=1440 and
    was previously dropped — leaving 'other' as the apparent last activity.
    """
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
                "tst": 505,
                "tet": 510,
            },
            {
                "pid": "p1",
                "seq": 2,
                "oact": "work",
                "dact": "other",
                "ozone": "z2",
                "dzone": "z3",
                "tst": 800,
                "tet": 802,
            },
            {
                "pid": "p1",
                "seq": 3,
                "oact": "other",
                "dact": "shop",
                "ozone": "z3",
                "dzone": "z4",
                "tst": 808,
                "tet": 810,
            },
            {
                "pid": "p1",
                "seq": 4,
                "oact": "shop",
                "dact": "work",
                "ozone": "z4",
                "dzone": "z2",
                "tst": 825,
                "tet": 830,
            },
            {
                "pid": "p1",
                "seq": 5,
                "oact": "work",
                "dact": "home",
                "ozone": "z2",
                "dzone": "z1",
                "tst": 1080,
                "tet": 1090,
            },
            {
                "pid": "p1",
                "seq": 6,
                "oact": "home",
                "dact": "other",
                "ozone": "z1",
                "dzone": "z3",
                "tst": 1175,
                "tet": 1180,
            },
            {
                "pid": "p1",
                "seq": 7,
                "oact": "other",
                "dact": "home",
                "ozone": "z3",
                "dzone": "z1",
                "tst": 1300,
                "tet": 1440,
            },
        ]
    )
    acts = post_process.trips_to_activities(attrs, trips)
    assert len(acts) == 8
    last = acts.sort("seq").row(-1, named=True)
    assert last["act"] == "home"
    assert last["end"] == 1440


# ---------------------------------------------------------------------------
# trips_to_activities: malformed / odd inputs
# ---------------------------------------------------------------------------


def test_trips_to_activities_seq_contradicts_chronological_tst_order():
    """seq order and chronological (tst) order disagree.

    trips_to_activities orders purely by seq, not by tst, so this produces a
    nonsensical activity (end < start) rather than raising — documents that
    no timing validation happens here (that's filter.py's job upstream).
    """
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
                "tst": 900,
                "tet": 950,
            },
            {
                "pid": "p1",
                "seq": 2,
                "oact": "work",
                "dact": "home",
                "ozone": "z2",
                "dzone": "z1",
                "tst": 480,
                "tet": 500,
            },
        ]
    )
    acts = post_process.trips_to_activities(attrs, trips)
    assert len(acts) == 3
    row = acts.filter(pl.col("seq") == 2).row(0, named=True)
    assert row["act"] == "work"
    assert row["start"] == 950
    assert row["end"] == 480
    assert row["end"] < row["start"]


def test_trips_to_activities_non_contiguous_seq_numbers():
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
                "seq": 3,
                "oact": "work",
                "dact": "shop",
                "ozone": "z2",
                "dzone": "z3",
                "tst": 900,
                "tet": 920,
            },
            {
                "pid": "p1",
                "seq": 5,
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
    assert len(acts) == len(trips) + 1
    # seq gaps are passed through unrenumbered
    assert acts.sort("seq").get_column("seq").to_list() == [1, 2, 4, 6]


def test_trips_to_activities_duplicate_seq():
    """Duplicate seq values per pid are not deduplicated or rejected."""
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
                "seq": 1,
                "oact": "work",
                "dact": "shop",
                "ozone": "z2",
                "dzone": "z3",
                "tst": 600,
                "tet": 650,
            },
        ]
    )
    acts = post_process.trips_to_activities(attrs, trips)
    assert len(acts) == len(trips) + 1
    seqs = acts.get_column("seq").to_list()
    assert seqs.count(2) == 2  # duplicate seq propagates into the output


def test_trips_to_activities_tet_over_1440_dropped_silently():
    """Regression test for the dest_acts filter/shift ordering fix.

    When a trip's tet > 1440 is dropped, the *preceding* surviving
    destination activity's end must still be computed from the real next
    trip's tst (not cascaded to fill_null(1440) because the shift ran after
    filtering).
    """
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
                "tst": 900,
                "tet": 1500,  # invalid, > 1440
            },
        ]
    )
    acts = post_process.trips_to_activities(attrs, trips)
    assert len(acts) == 2  # the invalid trip's destination activity is dropped
    row = acts.sort("seq").row(-1, named=True)
    assert row["act"] == "work"
    assert row["start"] == 540
    assert row["end"] == 900  # not 1440 — uses the real next trip's tst


def test_trips_to_activities_tet_less_than_tst():
    """tet < tst (bad/negative duration) doesn't crash; no validation happens here."""
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
                "tst": 500,
                "tet": 480,
            }
        ]
    )
    acts = post_process.trips_to_activities(attrs, trips)
    assert len(acts) == 2
    row = acts.sort("seq").row(-1, named=True)
    assert row["act"] == "work"
    assert row["start"] == 480
    assert row["end"] == 1440


def test_trips_to_activities_pid_missing_from_attributes():
    """A pid present in trips but absent from attributes is still processed
    normally — attributes only gates the anti-join for no-trip pids."""
    attrs = _make_attributes([{"pid": "pX", "hh_zone": "suburban"}])
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
    assert len(acts) == 3
    p1_acts = acts.filter(pl.col("pid") == "p1")
    assert len(p1_acts) == 2

    pX_acts = acts.filter(pl.col("pid") == "pX")
    assert len(pX_acts) == 1
    row = pX_acts.row(0, named=True)
    assert row["act"] == "home"
    assert row["zone"] == "suburban"
    assert row["start"] == 0
    assert row["end"] == 1440


def test_trips_to_activities_empty_trips_all_get_home_activity():
    attrs = _make_attributes(
        [
            {"pid": "p1", "hh_zone": "urban"},
            {"pid": "p2", "hh_zone": "suburban"},
        ]
    )
    trips = _make_trips([])
    acts = post_process.trips_to_activities(attrs, trips)
    assert len(acts) == 2
    for row in acts.iter_rows(named=True):
        assert row["act"] == "home"
        assert row["start"] == 0
        assert row["end"] == 1440


def test_trips_to_activities_both_empty():
    attrs = _make_attributes([])
    trips = _make_trips([])
    acts = post_process.trips_to_activities(attrs, trips)
    assert len(acts) == 0
    assert set(acts.columns) == {"pid", "seq", "act", "zone", "start", "end"}


# ---------------------------------------------------------------------------
# activities_to_trips
# ---------------------------------------------------------------------------


def test_activities_to_trips_basic():
    activities = _make_activities(
        [
            {
                "pid": "p1",
                "seq": 1,
                "act": "home",
                "zone": "z1",
                "start": 0,
                "end": 480,
            },
            {
                "pid": "p1",
                "seq": 2,
                "act": "work",
                "zone": "z2",
                "start": 480,
                "end": 900,
            },
            {
                "pid": "p1",
                "seq": 3,
                "act": "home",
                "zone": "z1",
                "start": 900,
                "end": 1440,
            },
        ]
    )
    trips = post_process.activities_to_trips(activities)
    assert len(trips) == 2

    row0 = trips.row(0, named=True)
    assert row0["oact"] == "home"
    assert row0["dact"] == "work"
    assert row0["ozone"] == "z1"
    assert row0["dzone"] == "z2"
    assert row0["tst"] == 480
    assert row0["tet"] == 480

    row1 = trips.row(1, named=True)
    assert row1["oact"] == "work"
    assert row1["dact"] == "home"
    assert row1["ozone"] == "z2"
    assert row1["dzone"] == "z1"
    assert row1["tst"] == 900
    assert row1["tet"] == 900


def test_activities_to_trips_single_activity_dropped():
    activities = _make_activities(
        [
            {
                "pid": "p1",
                "seq": 0,
                "act": "home",
                "zone": "z1",
                "start": 0,
                "end": 1440,
            }
        ]
    )
    trips = post_process.activities_to_trips(activities)
    assert len(trips) == 0


def test_activities_to_trips_mixed_single_and_multi_activity_persons():
    activities = _make_activities(
        [
            {
                "pid": "p1",
                "seq": 0,
                "act": "home",
                "zone": "z1",
                "start": 0,
                "end": 1440,
            },
            {
                "pid": "p2",
                "seq": 0,
                "act": "home",
                "zone": "z1",
                "start": 0,
                "end": 480,
            },
            {
                "pid": "p2",
                "seq": 1,
                "act": "work",
                "zone": "z2",
                "start": 480,
                "end": 1440,
            },
        ]
    )
    trips = post_process.activities_to_trips(activities)
    assert trips.get_column("pid").to_list() == ["p2"]


def test_activities_to_trips_columns():
    activities = _make_activities(
        [
            {
                "pid": "p1",
                "seq": 0,
                "act": "home",
                "zone": "z1",
                "start": 0,
                "end": 480,
            },
            {
                "pid": "p1",
                "seq": 1,
                "act": "work",
                "zone": "z2",
                "start": 480,
                "end": 1440,
            },
        ]
    )
    trips = post_process.activities_to_trips(activities)
    assert set(trips.columns) == {
        "pid",
        "seq",
        "tst",
        "tet",
        "oact",
        "dact",
        "ozone",
        "dzone",
    }


def test_activities_to_trips_seq_cast_to_int8():
    activities = _make_activities(
        [
            {
                "pid": "p1",
                "seq": 0,
                "act": "home",
                "zone": "z1",
                "start": 0,
                "end": 480,
            },
            {
                "pid": "p1",
                "seq": 1,
                "act": "work",
                "zone": "z2",
                "start": 480,
                "end": 1440,
            },
        ]
    )
    trips = post_process.activities_to_trips(activities)
    assert trips.schema["seq"] == pl.Int8


def test_activities_to_trips_out_of_order_rows():
    ordered = _make_activities(
        [
            {
                "pid": "p1",
                "seq": 1,
                "act": "home",
                "zone": "z1",
                "start": 0,
                "end": 480,
            },
            {
                "pid": "p1",
                "seq": 2,
                "act": "work",
                "zone": "z2",
                "start": 480,
                "end": 900,
            },
            {
                "pid": "p1",
                "seq": 3,
                "act": "home",
                "zone": "z1",
                "start": 900,
                "end": 1440,
            },
        ]
    )
    shuffled = _make_activities(
        [
            {
                "pid": "p1",
                "seq": 3,
                "act": "home",
                "zone": "z1",
                "start": 900,
                "end": 1440,
            },
            {
                "pid": "p1",
                "seq": 1,
                "act": "home",
                "zone": "z1",
                "start": 0,
                "end": 480,
            },
            {
                "pid": "p1",
                "seq": 2,
                "act": "work",
                "zone": "z2",
                "start": 480,
                "end": 900,
            },
        ]
    )
    expected = post_process.activities_to_trips(ordered)
    result = post_process.activities_to_trips(shuffled)
    assert result.rows() == expected.rows()


def test_activities_to_trips_duplicate_seq():
    """Duplicate seq values per pid are not deduplicated or rejected."""
    activities = _make_activities(
        [
            {
                "pid": "p1",
                "seq": 0,
                "act": "home",
                "zone": "z1",
                "start": 0,
                "end": 480,
            },
            {
                "pid": "p1",
                "seq": 0,
                "act": "weird",
                "zone": "z9",
                "start": 100,
                "end": 200,
            },
            {
                "pid": "p1",
                "seq": 1,
                "act": "work",
                "zone": "z2",
                "start": 480,
                "end": 1440,
            },
        ]
    )
    trips = post_process.activities_to_trips(activities)
    assert len(trips) == 2
    row0 = trips.row(0, named=True)
    assert row0["oact"] == "home"
    assert row0["dact"] == "weird"
    assert row0["tst"] == 480
    assert (
        row0["tet"] == 100
    )  # tet < tst: nonsensical, but produced without error


def test_activities_to_trips_empty_input():
    activities = _make_activities([])
    trips = post_process.activities_to_trips(activities)
    assert len(trips) == 0
    assert set(trips.columns) == {
        "pid",
        "seq",
        "tst",
        "tet",
        "oact",
        "dact",
        "ozone",
        "dzone",
    }


# ---------------------------------------------------------------------------
# round-trip consistency between trips_to_activities and activities_to_trips
# ---------------------------------------------------------------------------


def test_round_trip_trips_to_activities_to_trips_is_exact_identity():
    """On well-formed input, trips -> activities -> trips reproduces the
    original trips exactly (mod seq dtype). This depends on dest_acts using
    tet (arrival) rather than tst (departure) for the activity start."""
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
    activities = post_process.trips_to_activities(attrs, trips)
    round_trip = post_process.activities_to_trips(activities)

    assert len(round_trip) == len(trips)
    for orig_row, rt_row in zip(
        trips.sort("seq").iter_rows(named=True),
        round_trip.sort("seq").iter_rows(named=True),
    ):
        assert rt_row["tst"] == orig_row["tst"]
        assert rt_row["tet"] == orig_row["tet"]
        assert rt_row["oact"] == orig_row["oact"]
        assert rt_row["dact"] == orig_row["dact"]
        assert rt_row["ozone"] == orig_row["ozone"]
        assert rt_row["dzone"] == orig_row["dzone"]


def test_round_trip_trips_to_activities_to_trips_no_trip_pid_yields_zero_trips():
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
            }
        ]
    )
    activities = post_process.trips_to_activities(attrs, trips)
    round_trip = post_process.activities_to_trips(activities)
    assert round_trip.get_column("pid").to_list() == ["p1"]


def test_round_trip_activities_to_trips_to_activities_is_exact_identity_even_with_gaps():
    """Regression test for the dest_acts start=tet fix: activities -> trips
    -> activities is a clean identity even when the activity chain has real
    gaps (travel time) between activities, not just for contiguous chains.
    Before the fix, gaps collapsed to zero on the return trip."""
    attrs = _make_attributes([{"pid": "p1", "hh_zone": "urban"}])

    contiguous = _make_activities(
        [
            {
                "pid": "p1",
                "seq": 0,
                "act": "home",
                "zone": "z1",
                "start": 0,
                "end": 480,
            },
            {
                "pid": "p1",
                "seq": 1,
                "act": "work",
                "zone": "z2",
                "start": 480,
                "end": 900,
            },
            {
                "pid": "p1",
                "seq": 2,
                "act": "home",
                "zone": "z1",
                "start": 900,
                "end": 1440,
            },
        ]
    )
    gapped = _make_activities(
        [
            {
                "pid": "p1",
                "seq": 0,
                "act": "home",
                "zone": "z1",
                "start": 0,
                "end": 480,
            },
            {
                "pid": "p1",
                "seq": 1,
                "act": "work",
                "zone": "z2",
                "start": 540,
                "end": 900,
            },
            {
                "pid": "p1",
                "seq": 2,
                "act": "home",
                "zone": "z1",
                "start": 960,
                "end": 1440,
            },
        ]
    )

    for activities in (contiguous, gapped):
        round_trip = post_process.activities_to_trips(activities)
        round_activities = post_process.trips_to_activities(attrs, round_trip)
        assert round_activities.sort("seq").select(
            "act", "zone", "start", "end"
        ).rows() == (
            activities.sort("seq").select("act", "zone", "start", "end").rows()
        )


def test_round_trip_fixture_csv_exact_identity(fixture_attrs, fixture_trips):
    """tst/tet/dact/dzone always survive the round trip exactly, since they map
    1:1 from a trip's own fields. oact/ozone are reconstructed from the
    *previous* trip's dact/dzone (the activity chain), so they only survive
    where the original data's chain is internally consistent — real NTS data
    isn't always (verify.activity_consistency exists for exactly this)."""
    activities = post_process.trips_to_activities(fixture_attrs, fixture_trips)
    round_trip = post_process.activities_to_trips(activities)

    orig = fixture_trips.select(
        "pid",
        pl.col("seq").cast(pl.Int8).alias("seq"),
        "tst",
        "tet",
        "oact",
        "dact",
        "ozone",
        "dzone",
    )
    assert len(round_trip) == len(orig)

    joined = orig.join(round_trip, on=["pid", "seq"], suffix="_rt")
    assert len(joined) == len(
        orig
    )  # every original trip has a round-tripped match

    for col in ("tst", "tet", "dact", "dzone"):
        assert (joined[col] == joined[f"{col}_rt"]).all(), (
            f"{col} mismatch after round-trip"
        )

    # oact/ozone only round-trip where the original chain is self-consistent
    # (previous trip's dact/dzone matches this trip's oact/ozone)
    chained = orig.sort("pid", "seq").with_columns(
        prev_dact=pl.col("dact").shift(1).over("pid"),
        prev_dzone=pl.col("dzone").shift(1).over("pid"),
    )
    consistent_pids_seqs = chained.filter(
        pl.col("prev_dact").is_null() | (pl.col("prev_dact") == pl.col("oact"))
    ).select("pid", "seq")
    consistent_joined = joined.join(consistent_pids_seqs, on=["pid", "seq"])
    assert len(consistent_joined) > 0
    assert (consistent_joined["oact"] == consistent_joined["oact_rt"]).all()
    assert (consistent_joined["ozone"] == consistent_joined["ozone_rt"]).all()


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
        "-" in label or label.startswith("≤") or label.startswith(">")
        for label in labels
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


def test_fill_nulls_boolean_col():
    df = pl.DataFrame(
        {"rain": pl.Series([True, None, False], dtype=pl.Boolean)}
    )
    result = post_process.fill_nulls(df)
    assert result["rain"].dtype == pl.String
    assert result["rain"].to_list() == ["true", "unknown", "false"]


def test_discretise_numeric_invalid_method():
    df = pl.DataFrame({"age": [10, 20, 30]})
    with pytest.raises(ValueError, match="method must be"):
        post_process.discretise_numeric(df, method="bad")


# ---------------------------------------------------------------------------
# fill_unknown
# ---------------------------------------------------------------------------


def test_fill_unknown_string_null_and_empty():
    df = pl.DataFrame(
        {"sex": pl.Series(["male", None, "", "female"], dtype=pl.String)}
    )
    filled, stats = post_process.fill_unknown(df)
    assert filled["sex"].to_list() == ["male", "unknown", "unknown", "female"]
    assert "sex" in stats
    assert stats["sex"]["pct"] == pytest.approx(50.0)
    assert stats["sex"]["all_unknown"] is False
    assert stats["sex"]["appears_numeric"] is False


def test_fill_unknown_numeric_column():
    df = pl.DataFrame({"age": pl.Series([25, None, 40], dtype=pl.Int32)})
    filled, stats = post_process.fill_unknown(df)
    assert filled["age"].dtype == pl.String
    assert filled["age"].to_list() == ["25", "unknown", "40"]
    assert stats["age"]["appears_numeric"] is True
    assert stats["age"]["all_unknown"] is False
    assert stats["age"]["pct"] == pytest.approx(100.0 / 3)


def test_fill_unknown_all_null_column():
    df = pl.DataFrame({"zone": pl.Series([None, None, None], dtype=pl.String)})
    filled, stats = post_process.fill_unknown(df)
    assert all(v == "unknown" for v in filled["zone"].to_list())
    assert stats["zone"]["all_unknown"] is True
    assert stats["zone"]["pct"] == pytest.approx(100.0)


def test_fill_unknown_all_unknown_preexisting():
    df = pl.DataFrame(
        {"zone": pl.Series(["unknown", None, "unknown"], dtype=pl.String)}
    )
    filled, stats = post_process.fill_unknown(df)
    assert all(v == "unknown" for v in filled["zone"].to_list())
    assert stats["zone"]["all_unknown"] is True


def test_fill_unknown_appears_numeric_string_column():
    df = pl.DataFrame(
        {"income": pl.Series(["1000", None, "2500"], dtype=pl.String)}
    )
    filled, stats = post_process.fill_unknown(df)
    assert stats["income"]["appears_numeric"] is True


def test_fill_unknown_not_appears_numeric_text_column():
    df = pl.DataFrame(
        {"mode": pl.Series(["walk", None, "car"], dtype=pl.String)}
    )
    filled, stats = post_process.fill_unknown(df)
    assert stats["mode"]["appears_numeric"] is False


def test_fill_unknown_no_nulls_absent_from_stats():
    df = pl.DataFrame({"sex": pl.Series(["male", "female"], dtype=pl.String)})
    _, stats = post_process.fill_unknown(df)
    assert "sex" not in stats


def test_fill_unknown_empty_stats_when_no_nulls():
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    _, stats = post_process.fill_unknown(df)
    assert stats == {}
