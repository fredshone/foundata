import os
from pathlib import Path

import polars as pl
import pytest

from foundata import filter, fix, tus, verify
from foundata.utils import (
    get_config_path,
    load_yaml_config,
    split_employment_type,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
DATA_ROOT = os.getenv("FOUNDATA_TUS_DATA", str(FIXTURE_ROOT / "tus"))
CONFIGS_ROOT = get_config_path()


@pytest.fixture(scope="module")
def hh_cfg():
    return load_yaml_config(CONFIGS_ROOT / "tus" / "hh_dictionary.yaml")


@pytest.fixture(scope="module")
def person_cfg():
    return load_yaml_config(CONFIGS_ROOT / "tus" / "person_dictionary.yaml")


@pytest.fixture(scope="module")
def trips_cfg():
    return load_yaml_config(CONFIGS_ROOT / "tus" / "trip_dictionary.yaml")


@pytest.fixture(scope="module")
def loaded(hh_cfg, person_cfg, trips_cfg):
    return tus.load_years(
        Path(DATA_ROOT),
        years=[2019, 2024],
        hh_config=hh_cfg,
        person_config=person_cfg,
        trips_config=trips_cfg,
    )


def test_tus_load(loaded):
    attrs, trips = loaded

    assert len(attrs) > 0
    assert len(trips) > 0
    assert "tus" in attrs["source"].unique().to_list()
    assert set(attrs["year"].unique()) == {2019, 2024}
    assert set(trips["pid"]).issubset(set(attrs["pid"]))
    attrs = split_employment_type(attrs)
    attrs, trips = fix.missing_columns(attrs, trips)
    attrs, trips = filter.columns(attrs, trips)
    attrs, trips = fix.fix_types(attrs, trips)
    assert verify.columns(attrs, trips)


def test_tus_ids_prefixed_by_round_year(loaded):
    """Both rounds number FSUs/households independently, so hid/pid carry a
    2-digit round prefix to stay unique once the rounds are stacked."""
    attrs, _ = loaded

    assert attrs["pid"].is_unique().all()
    for year, prefix in [(2019, "tus19"), (2024, "tus24")]:
        rows = attrs.filter(pl.col("year") == year)
        assert rows["hid"].str.starts_with(prefix).all()
        assert rows["pid"].str.starts_with(prefix).all()


def test_tus_trip_chains_consistent(loaded):
    """Each trip's dact is the next trip's oact by construction (see
    load_trips), so filter.activity_consistency should never bite."""
    _, trips = loaded

    nxt = trips.sort("pid", "seq").with_columns(
        next_oact=pl.col("oact").shift(-1).over("pid")
    )
    linked = nxt.filter(pl.col("next_oact").is_not_null())
    assert (linked["dact"] == linked["next_oact"]).all()
    assert (trips["mode"] == "unknown").all()
    assert trips["distance"].is_null().all()


def _hh_raw_2019(**overrides):
    row = {
        "fsu": "10202",
        "b1q4": "01",
        "b4q1": "4",
        "sector": "2",
        "b4q9": "1000",
        "mult": "123.5",
        "survey_yr": "2019",
    }
    row.update(overrides)
    return pl.DataFrame({k: [v] for k, v in row.items()})


def test_load_households_2019(hh_cfg):
    hhs = tus.load_households(_hh_raw_2019(), hh_cfg, 2019)

    row = hhs.row(0, named=True)
    assert row["hid"] == "tus191020201"
    assert row["hh_size"] == 4
    assert row["hh_zone"] == "urban"
    # monthly INR expenditure -> annual EUR proxy
    assert row["hh_income"] == int(1000 * 12 * tus.INR_TO_EURO[2019])
    # 2019 has no tenure question
    assert row["ownership"] == "unknown"
    assert "hid_fsu" not in hhs.columns
    assert "hid_sample" not in hhs.columns


def test_load_households_blank_expenditure_is_null(hh_cfg):
    hhs = tus.load_households(_hh_raw_2019(b4q9="   "), hh_cfg, 2019)

    assert hhs["hh_income"].to_list() == [None]


def _episodes(rows):
    """rows: (time_from, time_to, code[, major_flag[, simultaneous_flag
    [, location]]]). Defaults: major, single-activity slot (blank
    simultaneous flag), inside the dwelling ("1")."""
    return pl.DataFrame(
        [
            {
                "pid": "p1",
                "activity_serial_no": f"{i + 1:03d}",
                "major_activity_flag": row[3] if len(row) > 3 else "1",
                "simultaneous_flag": row[4] if len(row) > 4 else "",
                "time_from": row[0],
                "time_to": row[1],
                "activity_code": row[2],
                "location": row[5] if len(row) > 5 else "1",
            }
            for i, row in enumerate(rows)
        ]
    )


_ATTRS = pl.DataFrame({"pid": ["p1"], "hh_zone": ["rural"]})


def test_load_trips_gap_takes_longest_activity(trips_cfg):
    """A run of several non-travel episodes between two trips collapses to
    the act type with the most total time in that run."""
    episodes = _episodes(
        [
            ("04:00", "07:00", "911"),  # home (sleep)
            ("07:00", "07:30", "182"),  # travel (commute)
            ("07:30", "08:00", "921"),  # home (eat), 30 min
            ("08:00", "12:00", "110"),  # work, 240 min
            ("12:00", "12:30", "371"),  # shop, 30 min
            ("12:30", "13:00", "182"),  # travel (commute)
            ("13:00", "04:00", "911"),  # home (sleep), wraps past midnight
        ]
    )

    trips = tus.load_trips(episodes, trips_cfg, _ATTRS)

    assert trips["seq"].to_list() == [0, 1]
    assert trips["oact"].to_list() == ["home", "work"]
    assert trips["dact"].to_list() == ["work", "home"]
    assert trips["tst"].to_list() == [420, 750]
    assert trips["tet"].to_list() == [450, 780]
    assert trips["ozone"].to_list() == ["rural", "rural"]
    assert trips["dzone"].to_list() == ["rural", "rural"]


def test_load_trips_wraps_trip_crossing_midnight(trips_cfg):
    """The diary runs 04:00 -> 04:00, so a trip spanning midnight must come
    out as minutes-since-midnight > 1440, not a negative duration."""
    episodes = _episodes(
        [
            ("04:00", "20:00", "911"),
            ("20:00", "20:30", "750"),  # travel to socialise
            ("20:30", "23:30", "712"),  # visit
            ("23:30", "00:30", "750"),  # travel home, across midnight
            ("00:30", "04:00", "911"),
        ]
    )

    trips = tus.load_trips(episodes, trips_cfg, _ATTRS)

    assert trips["tst"].to_list() == [1200, 1410]
    assert trips["tet"].to_list() == [1230, 1470]
    assert trips["oact"].to_list() == ["home", "visit"]
    assert trips["dact"].to_list() == ["visit", "home"]


def test_load_trips_no_travel_gives_no_trips(trips_cfg):
    episodes = _episodes([("04:00", "12:00", "911"), ("12:00", "04:00", "311")])

    trips = tus.load_trips(episodes, trips_cfg, _ATTRS)

    assert trips.is_empty()


def test_load_trips_minor_travel_splits_sequential_slot(trips_cfg):
    """Minor travel done before/after the major activity gets half the slot
    (shared between travel rows), in serial-number order."""
    episodes = _episodes(
        [
            ("04:00", "12:00", "911"),
            ("12:00", "12:30", "380", "2", "2"),  # travel...
            ("12:00", "12:30", "371", "1", "2"),  # ...then shop
            ("12:30", "13:00", "372"),  # more shopping
            ("13:00", "13:30", "750", "2", "2"),  # travel
            ("13:00", "13:30", "742", "1", "2"),  # religious practice
            ("13:00", "13:30", "711", "2", "2"),  # non-travel minor: dropped
            ("13:00", "13:30", "750", "2", "2"),  # travel back
            ("13:30", "04:00", "911"),
        ]
    )

    trips = tus.load_trips(episodes, trips_cfg, _ATTRS)

    # 12:00 slot: travel 15 min, shop 15. 13:00 slot: 2 x 7 min travel,
    # 16 min religious practice in between
    assert trips["tst"].to_list() == [720, 780, 803]
    assert trips["tet"].to_list() == [735, 787, 810]
    assert trips["oact"].to_list() == ["home", "shop", "visit"]
    assert trips["dact"].to_list() == ["shop", "visit", "home"]
    assert trips["seq"].to_list() == [0, 1, 2]


def test_load_trips_minor_travel_capped_at_an_hour(trips_cfg):
    """In a long slot, minor travel gets at most 60 min in total."""
    episodes = _episodes(
        [
            ("04:00", "16:00", "911"),
            ("16:00", "19:00", "380", "2", "2"),
            ("16:00", "19:00", "371", "1", "2"),
            ("16:00", "19:00", "380", "2", "2"),
            ("19:00", "04:00", "911"),
        ]
    )

    trips = tus.load_trips(episodes, trips_cfg, _ATTRS)

    assert trips["tst"].to_list() == [960, 1110]
    assert trips["tet"].to_list() == [990, 1140]


def test_load_trips_simultaneous_travel_takes_whole_slot(trips_cfg):
    """Travel done at the same time as the major activity (e.g. chatting
    while travelling) makes the whole slot travel."""
    episodes = _episodes(
        [
            ("04:00", "08:00", "911"),
            ("08:00", "09:00", "711", "1", "1"),  # chatting...
            ("08:00", "09:00", "750", "2", "1"),  # ...while travelling
            ("09:00", "04:00", "712"),
        ]
    )

    trips = tus.load_trips(episodes, trips_cfg, _ATTRS)

    assert trips["tst"].to_list() == [480]
    assert trips["tet"].to_list() == [540]
    assert trips["oact"].to_list() == ["home"]
    assert trips["dact"].to_list() == ["visit"]


def test_load_trips_major_travel_takes_whole_slot(trips_cfg):
    episodes = _episodes(
        [
            ("04:00", "08:00", "911"),
            ("08:00", "08:30", "182", "1", "2"),
            ("08:00", "08:30", "921", "2", "2"),  # minor, dropped
            ("08:00", "08:30", "181", "2", "2"),  # minor travel, dropped
            ("08:30", "04:00", "110"),
        ]
    )

    trips = tus.load_trips(episodes, trips_cfg, _ATTRS)

    assert trips["tst"].to_list() == [480]
    assert trips["tet"].to_list() == [510]


def test_load_trips_merges_consecutive_travel(trips_cfg):
    """Back-to-back travel episodes (here commuting then work-related
    travel, and a minor travel at the start of the next slot) are one trip,
    not several trips with no activity between them."""
    episodes = _episodes(
        [
            ("04:00", "08:30", "911"),
            ("08:30", "09:00", "182"),  # commuting
            ("09:00", "09:30", "181"),  # work-related travel
            ("09:30", "10:00", "182", "2", "2"),  # last bit of travel...
            ("09:30", "10:00", "110", "1", "2"),  # ...then work
            ("10:00", "15:00", "110"),
            ("15:00", "15:30", "182"),
            ("15:30", "04:00", "911"),
        ]
    )

    trips = tus.load_trips(episodes, trips_cfg, _ATTRS)

    assert trips["seq"].to_list() == [0, 1]
    assert trips["tst"].to_list() == [510, 900]
    assert trips["tet"].to_list() == [585, 930]
    assert trips["oact"].to_list() == ["home", "work"]
    assert trips["dact"].to_list() == ["work", "home"]


def test_load_trips_adds_trip_for_unrecorded_move_to_work(trips_cfg):
    """Working outside the dwelling straight after being at home, with no
    travel recorded, implies a trip there and back (10 min each, taken from
    the end of the earlier activity)."""
    episodes = _episodes(
        [
            ("04:00", "09:00", "911", "1", "", "1"),  # home, inside
            ("09:00", "13:00", "121", "1", "", "2"),  # work in the fields
            ("13:00", "04:00", "911", "1", "", "1"),  # home, inside
        ]
    )

    trips = tus.load_trips(episodes, trips_cfg, _ATTRS)

    assert trips["tst"].to_list() == [530, 770]
    assert trips["tet"].to_list() == [540, 780]
    assert trips["oact"].to_list() == ["home", "work"]
    assert trips["dact"].to_list() == ["work", "home"]


def test_load_trips_no_implied_trip_for_other_moves(trips_cfg):
    """No trip is added for: moving outside without changing activity type
    (home -> home), going out for an activity type not in
    IMPLIED_TRIP_ACTS ("other", e.g. tending a kitchen garden), or changing
    activity without changing location."""
    episodes = _episodes(
        [
            ("04:00", "07:00", "911", "1", "", "1"),  # home, inside
            ("07:00", "08:00", "931", "1", "", "2"),  # home, outside
            ("08:00", "09:00", "211", "1", "", "2"),  # other, outside
            ("09:00", "10:00", "921", "1", "", "1"),  # home, inside
            ("10:00", "12:00", "110", "1", "", "1"),  # work, inside
            ("12:00", "04:00", "911", "1", "", "1"),  # home, inside
        ]
    )

    trips = tus.load_trips(episodes, trips_cfg, _ATTRS)

    assert trips.is_empty()


def test_load_trips_adds_trip_for_unrecorded_move_to_visit(trips_cfg):
    episodes = _episodes(
        [
            ("04:00", "08:00", "911", "1", "", "1"),  # home, inside
            ("08:00", "09:00", "711", "1", "", "2"),  # chatting, outside
            ("09:00", "04:00", "911", "1", "", "1"),  # home, inside
        ]
    )

    trips = tus.load_trips(episodes, trips_cfg, _ATTRS)

    assert trips["oact"].to_list() == ["home", "visit"]
    assert trips["dact"].to_list() == ["visit", "home"]


def test_load_trips_no_implied_trip_next_to_recorded_travel(trips_cfg):
    episodes = _episodes(
        [
            ("04:00", "08:00", "911", "1", "", "1"),
            ("08:00", "08:30", "182", "1", "", "2"),  # recorded commute
            ("08:30", "13:00", "110", "1", "", "2"),
            ("13:00", "04:00", "911", "1", "", "1"),  # implied trip back
        ]
    )

    trips = tus.load_trips(episodes, trips_cfg, _ATTRS)

    assert trips["tst"].to_list() == [480, 770]
    assert trips["tet"].to_list() == [510, 780]


def test_diary_minutes():
    df = pl.DataFrame({"t": ["04:00", "23:30", "00:30", "03:59"]})

    starts = df.select(tus._diary_minutes("t", end=False))["t"].to_list()
    ends = df.select(tus._diary_minutes("t", end=True))["t"].to_list()

    assert starts == [240, 1410, 1470, 1679]
    # an end time of 04:00 closes the diary, on the next day
    assert ends == [1680, 1410, 1470, 1679]


def test_diary_respondents_only():
    attrs = pl.DataFrame({"pid": ["a", "b", "c"], "age": [30, 3, 40]})
    episodes = pl.DataFrame({"pid": ["a", "a", "c"]})

    result = tus._diary_respondents_only(attrs, episodes)

    assert result["pid"].to_list() == ["a", "c"]


def test_tus_load_excludes_persons_without_diary(loaded):
    attrs, _ = loaded

    assert (attrs["day"] != "unknown").all()


def test_load_unsupported_year_raises(hh_cfg, person_cfg, trips_cfg):
    with pytest.raises(ValueError, match="Unsupported TUS year"):
        tus.load("x", "y", hh_cfg, person_cfg, trips_cfg, year=2010)
