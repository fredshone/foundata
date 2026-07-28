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
