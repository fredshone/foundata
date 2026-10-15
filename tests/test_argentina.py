import datetime as dt
import os
from pathlib import Path

import pandas as pd
import polars as pl
import pyreadstat
import pytest

from foundata import argentina
from foundata.utils import get_config_path, load_yaml_config

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
DATA_ROOT = os.getenv(
    "FOUNDATA_ARGENTINA_DATA", str(FIXTURE_ROOT / "argentina")
)
CONFIGS_ROOT = get_config_path()


def load_configs():
    return [
        load_yaml_config(CONFIGS_ROOT / "argentina" / name)
        for name in (
            "hh_dictionary.yaml",
            "person_dictionary.yaml",
            "trip_dictionary.yaml",
        )
    ]


def test_argentina_load():
    hh_cfg, person_cfg, trips_cfg = load_configs()

    attrs, trips = argentina.load(
        Path(DATA_ROOT), hh_cfg, person_cfg, trips_cfg
    )

    assert attrs["pid"].is_unique().all()
    assert set(trips["pid"]).issubset(set(attrs["pid"]))
    assert attrs["pid"].str.starts_with("argentina").all()
    assert attrs["hid"].str.starts_with("argentina").all()
    assert attrs["source"].unique().to_list() == ["argentina"]
    for survey in argentina.SURVEYS:
        assert attrs["pid"].str.starts_with("argentina" + survey).any()
    assert attrs["age"].drop_nulls().min() >= 3
    assert trips.group_by("pid").agg(pl.col("seq").min())["seq"].max() == 0
    # some stages record 0 minutes
    assert (trips["distance"].drop_nulls() >= 0).all()
    assert set(attrs["day"]) <= set(argentina.WEEKDAYS) | {"unknown"}
    # ENMODO records no travel date
    enmodo = attrs.filter(pl.col("pid").str.starts_with("argentinaenmodo"))
    assert enmodo["day"].unique().to_list() == ["unknown"]
    # sampled from the income quintile's range
    assert enmodo["hh_income"].is_not_null().all()
    assert attrs["hh_income"].drop_nulls().min() >= 0
    for col in ("dact", "mode", "ozone", "dzone"):
        assert trips[col].null_count() == 0
    # weights are normalised within each survey
    means = attrs.group_by(
        pl.col("pid").str.extract(
            "^argentina(" + "|".join(argentina.SURVEYS) + ")"
        )
    ).agg(pl.col("weight").mean())["weight"]
    assert ((means - 1).abs() < 1e-4).all()


def hh_config(**overrides):
    config = {
        "missing_codes": [-98, -95, 99999],
        "eur_per_ars": {"s": 0.5},
        "dwelling": {"default": {1: "house", 4: "flat"}},
        "hh_zone": {"default": {}, "s": {"CC": "urban", "CO": "suburban"}},
    }
    return config | overrides


def test_households_vehicles_and_income():
    hhs = pl.DataFrame(
        {
            "hid": [1.0, 2.0, 3.0],
            "dwelling": [1.0, 4.0, 7.0],
            "hh_zone": ["CC", "CO", "XX"],
            "cars": [None, 2.0, 1.0],  # null: not asked because none
            "motorcycles": [1.0, None, 99999.0],  # 99999: Ns/Nc
            "hh_income": [1000.0, -95.0, 0.0],
        }
    )

    out = argentina.load_households(hhs, hh_config(), "s")

    assert out["dwelling"].to_list() == ["house", "flat", "unknown"]
    assert out["hh_zone"].to_list() == ["urban", "suburban", "unknown"]
    assert out["vehicles"].to_list() == [1, 2, None]
    assert out["hh_income"].to_list() == [6000, None, 0]
    assert out["fallback_date"].is_null().all()


def test_households_income_sampled_from_band():
    bands = {1: [1000, 2000], 2: [2000, 4000]}
    hhs = pl.DataFrame(
        {
            "hid": [1.0, 2.0, 3.0],
            "dwelling": [1.0, 1.0, 1.0],
            "hh_zone": ["CC", "CC", "CC"],
            "cars": [0.0, 0.0, 0.0],
            "hh_income_band": [1.0, 2.0, None],
        }
    )

    out = argentina.load_households(
        hhs, hh_config(hh_income_bands={"s": bands}), "s"
    )

    income = out["hh_income"].to_list()
    assert 500 <= income[0] <= 1000
    assert 1000 <= income[1] <= 2000
    assert income[2] is None


def test_households_count_motos_and_interview_eve():
    hhs = pl.DataFrame(
        {
            "hid": [1.0, 2.0],
            "dwelling": [1.0, 1.0],
            "hh_zone": ["CC", "CC"],
            "cars": [None, 1.0],
            "moto_1": [1.0, None],
            "moto_2": [2.0, None],
            "hh_income": [100.0, 100.0],
            "interview_day": [5.0, 1.0],
            "interview_month": [12.0, 11.0],
            "interview_year": [12.0, 2012.0],
        }
    )

    out = argentina.load_households(hhs, hh_config(), "s")

    assert out["vehicles"].to_list() == [2, 1]
    assert out["fallback_date"].to_list() == [
        dt.date(2012, 12, 4),
        dt.date(2012, 10, 31),
    ]


def test_persons_filters_and_education():
    _, person_cfg, _ = load_configs()
    persons = pl.DataFrame(
        {
            "pid": [1, 2, 3, 4, 5],
            "hid": [9, 9, 9, 9, 9],
            "weight": [1.0] * 5,
            "age": [40.0, 10.0, 2.0, 30.0, 25.0],
            "sex": [1.0, 2.0, 1.0, 2.0, 1.0],
            "relationship": [1.0, 3.0, 3.0, 2.0, 4.0],
            "education": [6.0, None, None, 4.0, None],
            "education_current": [None, 4.0, None, None, None],
            "employment": [1.0, None, None, 7.0, 5.0],
            "has_licence": [1.0, None, None, 2.0, 2.0],
            "disability": [0.0, 1.0, 1.0, 1.0, 0.0],
            "travelled": [1.0, 1.0, 2.0, 2.0, 2.0],
            "no_trip_reason": [None, None, None, 6.0, 11.0],
            "difficulty_walking": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    out = argentina.load_persons(persons, person_cfg, "rc")

    # pid 3 is under 3; pid 4 was outside the study area (rc reason 6)
    assert out["pid"].to_list() == [1, 2, 5]
    assert out["education"].to_list() == ["degree", "school", "unknown"]
    assert out["employment"].to_list() == ["employed", "unknown", "student"]
    assert out["has_licence"].to_list() == ["yes", "unknown", "no"]
    # rc: "ninguna" 0 with no difficulty flagged is unknown
    assert out["disability"].to_list() == ["unknown", "no", "unknown"]
    assert out["occupation"].to_list() == ["unknown", "void", "void"]
    assert "difficulty_walking" not in out.columns


def test_persons_difficulty_flags():
    _, person_cfg, _ = load_configs()
    persons = pl.DataFrame(
        {
            "pid": [1],
            "hid": [9],
            "weight": [1.0],
            "age": [70.0],
            "sex": [1.0],
            "relationship": [1.0],
            "education": [2.0],
            "employment": [7.0],
            "has_licence": [2.0],
            "disability": [0.0],
            "travelled": [2.0],
            "difficulty_walking": [1.0],
        }
    )

    out = argentina.load_persons(persons, person_cfg, "rc")

    assert out["disability"].to_list() == ["yes"]


def test_enmodo_person_codes():
    _, person_cfg, _ = load_configs()
    persons = pl.DataFrame(
        {
            "pid": [1, 2, 3, 4],
            "hid": [9, 9, 9, 9],
            "weight": [1.0] * 4,
            "age": [999.0, 1.0, 30.0, 30.0],
            "sex": ["Femenino", "Masculino", "Masculino", "Femenino"],
            "relationship": [1.0, 3.0, 5.0, 2.0],
            "education": [9.0, 1.0, 2.0, 5.0],
            "employment": [1.0, 0.0, 3.0, 7.0],
            "has_licence": [1.0, 0.0, 4.0, 4.0],
            "disability": [1.0, 1.0, 0.0, 1.0],
            "travelled": [1.0, 0.0, 4.0, 11.0],
        }
    )

    out = argentina.load_persons(persons, person_cfg, "enmodo")

    # 2: under 3 (VIAJE 0); 4: outside the study area (VIAJE 11)
    assert out["pid"].to_list() == [1, 3]
    assert out["age"].to_list() == [None, 30]
    assert out["sex"].to_list() == ["female", "male"]
    assert out["relationship"].to_list() == ["self", "parent"]
    assert out["disability"].to_list() == ["no", "yes"]


def test_diary_times_anchor_at_4am():
    trips = pl.DataFrame(
        {"tst": [470, 1430, 30, 200], "tet": [500, 10, 50, 190]}
    )

    out = trips.with_columns(argentina.diary_times(240))

    assert out["tst"].to_list() == [470, 1430, 1470, 1640]
    # 03:20 -> 03:10 is read as a 23h50 trip, which feasible_trips drops
    assert out["tet"].to_list() == [500, 1450, 1490, 3070]


def test_clock_minutes_from_hour_minute():
    trips = pl.DataFrame(
        {"tst_h": [7.0, 24.0, 99999.0, 8.0], "tst_m": [30.0, 0.0, 0.0, 60.0]}
    )

    out = trips.select(tst=argentina.clock_minutes(trips, "tst"))

    assert out["tst"].to_list() == [450, 1440, None, None]


def test_enmodo_seq_from_start_times():
    _, _, trips_cfg = load_configs()
    trips = pl.DataFrame(
        {
            "tid": [10.0, 11.0, 12.0],
            "pid": [1.0, 1.0, 1.0],
            "oact": [2.0, 1.0, 7.0],
            "dact": [7.0, 2.0, 1.0],
            "tst_h": [17.0, 8.0, 1.0],
            "tst_m": [0.0, 0.0, 0.0],
            "tet_h": [17.0, 8.0, 1.0],
            "tet_m": [30.0, 45.0, 20.0],
            "ozone": [427.0, 901.0, 427.0],
            "dzone": [427.0, 427.0, 901.0],
        }
    )
    stages = pl.DataFrame(
        {
            "tid": [10.0, 11.0, 11.0, 12.0],
            "stage": [1.0, 1.0, 2.0, 1.0],
            "mode": [13.0, 13.0, 7.0, 3.0],
            "duration_h": [0.0, 0.0, 99.0, 0.0],
            "duration_m": [30.0, 10.0, 0.0, 20.0],
        }
    )
    home = pl.DataFrame({"pid": [1], "hh_zone": ["urban"]})

    out = argentina.load_trips(trips, stages, trips_cfg, "enmodo", home)

    # 01:00 is the end of the diary day
    assert out["seq"].to_list() == [0, 1, 2]
    assert out["tst"].to_list() == [480, 1020, 1500]
    assert out["oact"].to_list() == ["home", "work", "shop"]
    assert out["dact"].to_list() == ["work", "shop", "home"]
    # walk 10 min + unknown rail stage gets the remaining 35 min
    assert out["mode"].to_list() == ["rail", "walk", "car"]
    assert out["distance"].to_list() == pytest.approx(
        [10 / 60 * 5 + 35 / 60 * 30, 30 / 60 * 5, 20 / 60 * 22]
    )
    assert out["travel_date"].is_null().all()


def test_emd_trips_home_zone_and_unstaged_trips():
    _, _, trips_cfg = load_configs()
    trips = pl.DataFrame(
        {
            "tid": [1.0, 2.0],
            "pid": [5.0, 5.0],
            "seq": [1.0, 2.0],
            "oact": [1.0, 2.0],
            "dact": [2.0, 1.0],
            "tst": [480.0, 1080.0],
            "tet": [500.0, None],
            "ozone": [96.0, 13.0],
            "dzone": [13.0, 96.0],
            "travel_day": [6.0, 6.0],
            "travel_month": [11.0, 11.0],
            "travel_year": [12.0, 12.0],
        }
    )
    stages = pl.DataFrame(
        {"tid": [1.0], "stage": [1.0], "mode": [3.0], "duration": [-96.0]}
    )
    home = pl.DataFrame({"pid": [5], "hh_zone": ["urban"]})

    out = argentina.load_trips(trips, stages, trips_cfg, "salta", home)

    assert out["ozone"].to_list() == ["urban", "suburban"]
    assert out["dzone"].to_list() == ["suburban", "urban"]
    assert out["mode"].to_list() == ["bus", "unknown"]
    # unknown stage time takes the whole 20 min trip
    assert out["distance"].to_list() == [pytest.approx(20 / 60 * 16), None]
    assert out["travel_date"].to_list() == [dt.date(2012, 11, 6)] * 2


def test_travel_dates_fall_back_to_household_then_interview():
    persons = pl.DataFrame({"pid": [1, 2, 3], "hid": [7, 7, 8]})
    trips = pl.DataFrame({"pid": [1], "travel_date": [dt.date(2012, 11, 6)]})
    hhs = pl.DataFrame(
        {"hid": [7, 8], "fallback_date": [None, dt.date(2012, 11, 8)]}
    )

    out = argentina.travel_dates(persons, trips, hhs).sort("pid")

    assert out["travel_date"].to_list() == [
        dt.date(2012, 11, 6),
        dt.date(2012, 11, 6),
        dt.date(2012, 11, 8),
    ]
    days = out.select(argentina.diary_date_columns(2012))
    assert days["day"].to_list() == ["tuesday", "tuesday", "thursday"]


def test_read_sav_with_latin1_variable_names(tmp_path):
    path = tmp_path / "hogares.sav"
    pyreadstat.write_sav(
        pd.DataFrame({"AXO": [12.0], "Hora": [dt.time(7, 30)]}), str(path)
    )
    # the real NEUQUEN_CIPPO_HOGARES.sav names this variable "Año" in latin-1
    path.write_bytes(path.read_bytes().replace(b"AXO", b"A\xd1O"))

    out = argentina.read_sav(path, ["ANO", "Hora"])

    assert out.to_dicts() == [{"ANO": 12.0, "Hora": 450.0}]
