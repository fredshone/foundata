import os
from pathlib import Path

from foundata import filter, fix, nts, verify
from foundata.utils import get_config_path, load_yaml_config

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
DATA_ROOT = os.getenv("FOUNDATA_NTS_DATA", str(FIXTURE_ROOT / "nts"))
CONFIGS_ROOT = get_config_path()


def test_nts_load():
    hh_cfg = load_yaml_config(CONFIGS_ROOT / "nts" / "hh_dictionary.yaml")
    person_cfg = load_yaml_config(CONFIGS_ROOT / "nts" / "person_dictionary.yaml")
    trips_cfg = load_yaml_config(CONFIGS_ROOT / "nts" / "trip_dictionary.yaml")
    days_cfg = load_yaml_config(CONFIGS_ROOT / "nts" / "day_dictionary.yaml")

    attrs, trips = nts.load(Path(DATA_ROOT), hh_cfg, person_cfg, trips_cfg, days_cfg)

    assert len(attrs) > 0
    assert len(trips) > 0
    assert "nts" in attrs["source"].unique().to_list()
    assert set(trips["pid"]).issubset(set(attrs["pid"]))
    attrs, trips = filter.columns(attrs, trips)
    attrs, trips = fix.fix_types(attrs, trips)
    assert verify.columns(attrs, trips)
