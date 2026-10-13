import os
from pathlib import Path

import polars as pl
import pytest

from foundata import mxod
from foundata.braz import read_dbf
from foundata.utils import get_config_path, load_yaml_config

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
DATA_ROOT = os.getenv("FOUNDATA_MXOD_DATA", str(FIXTURE_ROOT / "mxod"))
CONFIGS_ROOT = get_config_path()


def load_configs():
    return [
        load_yaml_config(CONFIGS_ROOT / "mxod" / name)
        for name in (
            "hh_dictionary.yaml",
            "person_dictionary.yaml",
            "trip_dictionary.yaml",
        )
    ]


def test_mxod_load():
    hh_cfg, person_cfg, trips_cfg = load_configs()

    attrs, trips = mxod.load(Path(DATA_ROOT), hh_cfg, person_cfg, trips_cfg)

    persons = read_dbf(Path(DATA_ROOT) / mxod.PERSONS_PATH, columns=["P4_2"])
    raw_trips = read_dbf(Path(DATA_ROOT) / mxod.TRIPS_PATH, columns=["ID_VIA"])
    # one plan per diary (weekday, Saturday) per person asked about travel
    assert len(attrs) == 2 * persons["P4_2"].is_not_null().sum()
    assert len(trips) == len(raw_trips)
    assert attrs["pid"].is_unique().all()
    assert set(trips["pid"]).issubset(set(attrs["pid"]))
    assert attrs["pid"].str.starts_with("mxod").all()
    assert attrs["hid"].str.starts_with("mxod").all()
    assert attrs["source"].unique().to_list() == ["mxod"]
    assert set(attrs["day"]) == {"unknown", "saturday"}
    assert attrs["age"].min() >= 6
    assert trips.group_by("pid").agg(pl.col("seq").min())["seq"].max() == 0
    assert trips["mode"].is_in(["unknown"]).sum() == 0
    assert (trips["distance"].drop_nulls() > 0).all()


def raw_persons(**overrides) -> pl.DataFrame:
    """Raw (string) TSDEM columns after column_mappings renaming."""
    data = {
        "pid": ["1", "2", "3"],
        "hid": ["10", "10", "10"],
        "weight": ["100", "100", "100"],
        "age": ["40", "35", "03"],
        "sex": ["1", "2", "1"],
        "relationship": ["1", "2", "3"],
        "education": ["08", "03", None],
        "employment": ["1", "5", None],
        "worked_anyway": [None, "2", None],
        "weekday_travel": ["1", "2", None],
        "saturday_travel": ["2", "2", None],
    }
    data.update(overrides)
    return pl.DataFrame(data)


def test_load_person_days():
    _, person_cfg, trips_cfg = load_configs()

    result = mxod.load_person_days(
        raw_persons(), person_cfg, trips_cfg["day"]
    ).sort("pid")

    # the 3-year-old wasn't asked about travel, so is dropped
    assert result["pid"].to_list() == ["11", "12", "21", "22"]
    assert result["day"].to_list() == [
        "unknown",
        "saturday",
        "unknown",
        "saturday",
    ]
    # homemaker who sold goods last week (P3_8 = 2) counts as employed
    assert result["employment"].to_list() == ["employed"] * 4
    assert result["education"].to_list() == [
        "degree",
        "degree",
        "school",
        "school",
    ]


def test_district_zone_types_weighted_mode():
    hhs = pl.DataFrame(
        {
            "home_district": ["001", "001", "001", "002"],
            "hh_zone": ["urban", "suburban", "suburban", "rural"],
            "hh_weight": [300.0, 100.0, 100.0, 50.0],
        }
    )

    assert mxod.district_zone_types(hhs) == {1: "urban", 2: "rural"}


def test_clock_minutes_unknown_is_null():
    df = pl.DataFrame({"h": [8, 99, 23], "m": [30, 0, 99]})

    result = df.select(mxod.clock_minutes("h", "m"))["h"].to_list()

    assert result == [510, None, None]


def test_activities():
    _, _, trips_cfg = load_configs()
    trips = pl.DataFrame(
        {
            "pid": ["a", "a", "a", "b"],
            "seq": [0, 1, 2, 0],
            "oact": [1, 5, 7, 3],  # origin place types
            "dact": [5, 5, 1, 2],  # purposes
            "dest_place": [5, 7, 1, 3],
        }
    )

    result = trips.with_columns(mxod.activities(trips_cfg))

    # leisure at another dwelling becomes visit, and the next trip starts
    # from it; each diary's first origin comes from the place type
    assert result["dact"].to_list() == ["leisure", "visit", "home", "work"]
    assert result["oact"].to_list() == ["home", "leisure", "visit", "work"]


@pytest.fixture
def trips_cfg():
    return {
        **load_configs()[2],
        "speed": {14: 6, 5: 30, 1: 20},
    }


def test_trip_modes_and_distances(trips_cfg):
    trips = pl.DataFrame({"tid": [1, 2], "tst": [480, 600], "tet": [540, 630]})
    stages = pl.DataFrame(
        {
            "tid": ["1", "1", "1", "2"],
            "mode": ["14", "05", "14", "01"],  # walk, metro, walk / car
            "stage": ["1", "2", "3", "1"],
            "duration_h": ["00", "00", "99", "00"],
            "duration_m": ["10", "40", "99", "30"],
        }
    )

    result = mxod.trip_modes_and_distances(trips, stages, trips_cfg).sort("tid")

    assert result["mode"].to_list() == ["rail", "car"]
    # the unknown walk stage gets the 10 of 60 minutes not accounted for
    walk = 20 / 60 * 6
    assert result["distance"].to_list() == pytest.approx(
        [walk + 40 / 60 * 30, 30 / 60 * 20]
    )
    assert result["access_egress_distance"].to_list() == [
        pytest.approx(walk),
        None,
    ]
