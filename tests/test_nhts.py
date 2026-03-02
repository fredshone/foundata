import os
from pathlib import Path

from foundata import nhts, verify
from foundata.utils import get_config_path, load_yaml_config

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
DATA_ROOT = os.getenv("FOUNDATA_NHTS_DATA", str(FIXTURE_ROOT / "nhts"))
CONFIGS_ROOT = get_config_path()


def test_nhts_load():
    hh_cfg = load_yaml_config(CONFIGS_ROOT / "nhts" / "hh_dictionary.yaml")
    person_cfg = load_yaml_config(CONFIGS_ROOT / "nhts" / "person_dictionary.yaml")
    trips_cfg = load_yaml_config(CONFIGS_ROOT / "nhts" / "trip_dictionary.yaml")

    attrs, trips = nhts.load(
        DATA_ROOT,
        years=[2022],
        hh_config=hh_cfg,
        person_config=person_cfg,
        trips_config=trips_cfg,
    )

    assert len(attrs) > 0
    assert len(trips) > 0
    assert "nhts" in attrs["source"].unique().to_list()
    assert set(trips["pid"]).issubset(set(attrs["pid"]))
    assert verify.columns(attrs, trips)
