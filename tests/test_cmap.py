import os
from pathlib import Path

import pytest

from foundata import cmap, verify
from foundata.utils import get_config_path, load_yaml_config

DATA_ROOT = os.getenv("FOUNDATA_CMAP_DATA")
CONFIGS_ROOT = get_config_path()


@pytest.mark.skipif(not DATA_ROOT, reason="FOUNDATA_CMAP_DATA not set")
def test_cmap_load():
    hh_cfg = load_yaml_config(CONFIGS_ROOT / "cmap" / "hh_dictionary.yaml")
    person_cfg = load_yaml_config(CONFIGS_ROOT / "cmap" / "person_dictionary.yaml")
    trips_cfg = load_yaml_config(CONFIGS_ROOT / "cmap" / "trip_dictionary.yaml")

    attrs, trips = cmap.load(DATA_ROOT, hh_cfg, person_cfg, trips_cfg)

    assert len(attrs) > 0
    assert len(trips) > 0
    assert "cmap" in attrs["source"].unique().to_list()
    assert set(trips["pid"]).issubset(set(attrs["pid"]))
    assert verify.columns(attrs, trips)
