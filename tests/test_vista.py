import os

import pytest

from foundata import vista, verify
from foundata.utils import get_config_path, load_yaml_config

DATA_ROOT = os.getenv("FOUNDATA_VISTA_DATA")
CONFIGS_ROOT = get_config_path()


@pytest.mark.skipif(not DATA_ROOT, reason="FOUNDATA_VISTA_DATA not set")
def test_vista_load():
    hh_cfg = load_yaml_config(CONFIGS_ROOT / "vista" / "hh_dictionary.yaml")
    person_cfg = load_yaml_config(CONFIGS_ROOT / "vista" / "person_dictionary.yaml")
    trips_cfg = load_yaml_config(CONFIGS_ROOT / "vista" / "trip_dictionary.yaml")

    attrs, trips = vista.load_years(
        DATA_ROOT,
        years=["2012_2020"],
        hh_config=hh_cfg,
        person_config=person_cfg,
        trips_config=trips_cfg,
    )

    assert len(attrs) > 0
    assert len(trips) > 0
    assert "vista" in attrs["source"].unique().to_list()
    assert set(trips["pid"]).issubset(set(attrs["pid"]))
    assert verify.columns(attrs, trips)
