import os
from pathlib import Path

import polars as pl

from foundata import filter, fix, verify, vista
from foundata.utils import config_for_year, get_config_path, load_yaml_config

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
DATA_ROOT = os.getenv("FOUNDATA_VISTA_DATA", str(FIXTURE_ROOT / "vista"))
CONFIGS_ROOT = get_config_path()


def test_vista_load():
    hh_cfg = load_yaml_config(CONFIGS_ROOT / "vista" / "hh_dictionary.yaml")
    person_cfg = load_yaml_config(
        CONFIGS_ROOT / "vista" / "person_dictionary.yaml"
    )
    trips_cfg = load_yaml_config(
        CONFIGS_ROOT / "vista" / "trip_dictionary.yaml"
    )

    attrs, trips = vista.load_years(
        Path(DATA_ROOT),
        years=["2012-2020"],
        hh_config=hh_cfg,
        person_config=person_cfg,
        trips_config=trips_cfg,
    )

    assert len(attrs) > 0
    assert len(trips) > 0
    assert "vista" in attrs["source"].unique().to_list()
    assert set(trips["pid"]).issubset(set(attrs["pid"]))
    attrs, trips = fix.missing_columns(attrs, trips)
    attrs, trips = filter.columns(attrs, trips)
    attrs, trips = fix.fix_types(attrs, trips)
    assert verify.columns(attrs, trips)


def test_vista_employment_handles_yes_no_encoding():
    """The 2012-2020 wave encodes fulltimework/parttimework as Y/N, but the
    2022-2023 and 2023-2024 waves switched to Yes/No/Not applicable — a
    strict "== Y" check would silently misclassify every full/part-time
    worker surveyed since 2022 as never employed (see
    foundata/vista.py::preprocess_persons)."""
    person_cfg = load_yaml_config(
        CONFIGS_ROOT / "vista" / "person_dictionary.yaml"
    )
    cfg = config_for_year(person_cfg, "2022-2023")

    raw = pl.DataFrame(
        {
            "persid": [1, 2, 3],
            "hhid": [1, 1, 2],
            "agegroup": ["25->34", "35->44", "45->54"],
            "sex": ["M", "F", "M"],
            "relationship": ["Self", "Spouse", "Self"],
            "nolicence": ["Some Licence", "Some Licence", "No Licence"],
            "fulltimework": ["Yes", "No", "No"],
            "parttimework": ["No", "Yes", "No"],
            "studying": ["No Study", "No Study", "No Study"],
            "activities": [
                "No other activity",
                "No other activity",
                "Retired",
            ],
            "anzsco1": ["Managers", "Managers", "Managers"],
        }
    )

    persons = vista.preprocess_persons(raw, cfg, year="2022-2023")
    assert persons["employment"].to_list() == [
        "ft-employed",
        "pt-employed",
        "retired",
    ]
