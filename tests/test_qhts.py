import os

import pytest

from foundata import qhts, verify
from foundata.utils import get_config_path, load_yaml_config

DATA_ROOT = os.getenv("FOUNDATA_QHTS_DATA")
CONFIGS_ROOT = get_config_path()


@pytest.mark.skipif(not DATA_ROOT, reason="FOUNDATA_QHTS_DATA not set")
def test_qhts_load():
    hh_cfg = load_yaml_config(CONFIGS_ROOT / "qhts" / "hh_dictionary.yaml")
    person_cfg = load_yaml_config(CONFIGS_ROOT / "qhts" / "person_dictionary.yaml")
    trips_cfg = load_yaml_config(CONFIGS_ROOT / "qhts" / "trip_dictionary.yaml")

    attrs, trips = qhts.load_years(
        DATA_ROOT,
        years=["2017"],
        hh_config=hh_cfg,
        person_config=person_cfg,
        trips_config=trips_cfg,
    )

    assert len(attrs) > 0
    assert len(trips) > 0
    assert "qhts" in attrs["source"].unique().to_list()
    assert set(trips["pid"]).issubset(set(attrs["pid"]))
    assert verify.columns(attrs, trips)
