import os
from pathlib import Path

import polars as pl

from foundata import qhts
from foundata.utils import get_config_path, load_yaml_config

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
DATA_ROOT = os.getenv("FOUNDATA_QHTS_DATA", str(FIXTURE_ROOT / "qhts"))
CONFIGS_ROOT = get_config_path()


def test_qhts_load():
    hh_cfg = load_yaml_config(CONFIGS_ROOT / "qhts" / "hh_dictionary.yaml")
    person_cfg = load_yaml_config(
        CONFIGS_ROOT / "qhts" / "person_dictionary.yaml"
    )
    trips_cfg = load_yaml_config(CONFIGS_ROOT / "qhts" / "trip_dictionary.yaml")

    attrs, trips = qhts.load_years(
        Path(DATA_ROOT),
        years=["2017-20"],
        hh_config=hh_cfg,
        person_config=person_cfg,
        trips_config=trips_cfg,
    )

    assert len(attrs) > 0
    assert len(trips) > 0
    assert "qhts" in attrs["source"].unique().to_list()
    assert set(trips["pid"]).issubset(set(attrs["pid"]))


def test_qhts_preprocess_trips_applies_4am_offset():
    """QHTS's raw STARTIME/ARRTIME are TIME-codes anchored at 04:00 (see the
    survey's R_TIME.csv: TIME=0 -> 04:00:00), not literal minutes-since-
    midnight. preprocess_trips must shift them by +240 minutes to align with
    the template's midnight-relative tst/tet convention."""
    config = {
        "column_mappings": {
            "HHID": "hid",
            "PERSID": "pid",
            "STARTSTOP": "seq",
            "MAINMODE": "mode",
            "CUMDIST": "distance",
            "STARTIME": "tst",
            "ARRTIME": "tet",
            "ORIGPURP": "oact",
            "DESTPURP": "dact",
            "ORIGSA1_2021": "ozone",
            "DESTSA1_2021": "dzone",
        },
        "mode_mappings": {"Car driver": "car"},
        "act_mappings": {"At Home": "home", "Work (my workplace)": "work"},
    }
    raw = pl.DataFrame(
        {
            "HHID": [1],
            "PERSID": ["1/1"],
            "STARTSTOP": [1],
            "MAINMODE": ["Car driver"],
            "CUMDIST": [5.0],
            "STARTIME": [180],  # 07:00 real clock time, per R_TIME.csv
            "ARRTIME": [200],
            "ORIGPURP": ["At Home"],
            "DESTPURP": ["Work (my workplace)"],
            "ORIGSA1_2021": ["z1"],
            "DESTSA1_2021": ["z2"],
        }
    )

    result = qhts.preprocess_trips(raw, config, year="2019-22")

    assert result["tst"].to_list() == [180 + 240]
    assert result["tet"].to_list() == [200 + 240]


def test_qhts_preprocess_persons_hh_income_null_when_all_unanswered():
    """A household where every member's INCOME response is a non-answer
    ("Select One" / "Prefer not to say") must get a null hh_income, not 0.

    `sample_to_euro` already encodes "no real answer" as a single-element
    bounds list -> None per person; but summing a group of all-null values
    with polars' `.sum()` silently produces 0 rather than null, which
    previously miscoded "nobody answered" households as literally €0
    income, indistinguishable from a household that actually reported nil
    income ("Nil or negative income": [0, 0]).
    """
    config = {
        "column_mappings": {
            "PERSID": "pid",
            "HHID": "hid",
            "AGEGROUP": "age",
            "SEX": "sex",
            "RELATIONSHIP": "relationship",
            "CARLICENCE": "has_licence",
            "MAINACT": "employment",
            "ANZSCO_1-digit": "occupation",
            "ASSISTANY": "disability",
            "INCOME": "income",
        },
        "age": {1: [0, 4]},
        "sex": {"M": "male", "F": "female"},
        "relationship": {"spouse": "partner"},
        "has_licence": {1: "yes", 0: "no"},
        "employment": {"Full-time Work": "ft-employed"},
        "occupation": {1: "managerial"},
        "disability": {1: "yes", 0: "no"},
        "income": {
            "Select One": [0],
            "Prefer not to say / can't say": [0],
            "Nil or negative income": [0, 0],
        },
    }
    raw = pl.DataFrame(
        {
            "PERSID": ["h1/1", "h1/2", "h2/1", "h2/2"],
            "HHID": [1, 1, 2, 2],
            "AGEGROUP": [1, 1, 1, 1],
            "SEX": ["M", "F", "M", "F"],
            "RELATIONSHIP": ["spouse", "spouse", "spouse", "spouse"],
            "CARLICENCE": [1, 1, 1, 1],
            "MAINACT": ["Full-time Work"] * 4,
            "ANZSCO_1-digit": [1, 1, 1, 1],
            "ASSISTANY": [0, 0, 0, 0],
            "INCOME": [
                "Select One",
                "Prefer not to say / can't say",
                "Nil or negative income",
                "Prefer not to say / can't say",
            ],
        }
    )

    result = qhts.preprocess_persons(raw, config, year="2019-22")

    h1 = result.filter(pl.col("hid") == 1)
    h2 = result.filter(pl.col("hid") == 2)
    assert h1["hh_income"].to_list() == [None, None]
    assert h2["hh_income"].to_list() == [0, 0]
