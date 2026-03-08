import os
from pathlib import Path

from foundata import filter, fix, ktdb, verify
from foundata.utils import get_config_path, load_yaml_config

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
DATA_ROOT = os.getenv("FOUNDATA_KTDB_DATA", str(FIXTURE_ROOT / "ktdb"))
CONFIGS_ROOT = get_config_path()


def test_ktdb_load():
    person_cfg = load_yaml_config(CONFIGS_ROOT / "ktdb" / "person_dictionary.yaml")
    trips_cfg = load_yaml_config(CONFIGS_ROOT / "ktdb" / "trip_dictionary.yaml")

    attrs, trips = ktdb.load(
        Path(DATA_ROOT),
        person_config=person_cfg,
        trips_config=trips_cfg,
    )

    assert len(attrs) > 0
    assert len(trips) > 0
    assert "ktdb" in attrs["source"].unique().to_list()
    assert set(trips["pid"]).issubset(set(attrs["pid"]))
    attrs, trips = filter.columns(attrs, trips)
    attrs, trips = fix.fix_types(attrs, trips)
    assert verify.columns(attrs, trips)
