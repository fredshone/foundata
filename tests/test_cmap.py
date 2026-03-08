import os
from pathlib import Path

from foundata import cmap, filter, fix, verify
from foundata.utils import get_config_path, load_yaml_config

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
DATA_ROOT = os.getenv("FOUNDATA_CMAP_DATA", str(FIXTURE_ROOT / "cmap"))
CONFIGS_ROOT = get_config_path()


def test_cmap_load():
    hh_cfg = load_yaml_config(CONFIGS_ROOT / "cmap" / "hh_dictionary.yaml")
    person_cfg = load_yaml_config(CONFIGS_ROOT / "cmap" / "person_dictionary.yaml")
    trips_cfg = load_yaml_config(CONFIGS_ROOT / "cmap" / "trip_dictionary.yaml")

    attrs, trips = cmap.load(Path(DATA_ROOT), CONFIGS_ROOT, hh_cfg, person_cfg, trips_cfg)

    assert len(attrs) > 0
    assert len(trips) > 0
    assert "cmap" in attrs["source"].unique().to_list()
    assert set(trips["pid"]).issubset(set(attrs["pid"]))
    attrs, trips = filter.columns(attrs, trips)
    attrs, trips = fix.fix_types(attrs, trips)
    assert verify.columns(attrs, trips)
