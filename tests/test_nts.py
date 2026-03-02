import os

import pytest

from foundata import nts, verify
from foundata.utils import get_config_path, load_yaml_config

DATA_ROOT = os.getenv("FOUNDATA_NTS_DATA")
CONFIGS_ROOT = get_config_path()


@pytest.mark.skipif(not DATA_ROOT, reason="FOUNDATA_NTS_DATA not set")
def test_nts_load():
    hh_cfg = load_yaml_config(CONFIGS_ROOT / "nts" / "hh_dictionary.yaml")
    person_cfg = load_yaml_config(CONFIGS_ROOT / "nts" / "person_dictionary.yaml")
    trips_cfg = load_yaml_config(CONFIGS_ROOT / "nts" / "trip_dictionary.yaml")
    days_cfg = load_yaml_config(CONFIGS_ROOT / "nts" / "days_dictionary.yaml")

    attrs, trips = nts.load(DATA_ROOT, hh_cfg, person_cfg, trips_cfg, days_cfg)

    assert len(attrs) > 0
    assert len(trips) > 0
    assert "nts" in attrs["source"].unique().to_list()
    assert set(trips["pid"]).issubset(set(attrs["pid"]))
    assert verify.columns(attrs, trips)
