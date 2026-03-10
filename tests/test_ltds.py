import os
from pathlib import Path

from foundata import filter, fix, ltds, verify
from foundata.utils import get_config_path, load_yaml_config

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
DATA_ROOT = os.getenv("FOUNDATA_LTDS_DATA", str(FIXTURE_ROOT / "ltds"))
CONFIGS_ROOT = get_config_path()


def test_ltds_load():
    hh_cfg = load_yaml_config(CONFIGS_ROOT / "ltds" / "hh_dictionary.yaml")
    person_cfg = load_yaml_config(CONFIGS_ROOT / "ltds" / "person_dictionary.yaml")
    person_data_cfg = load_yaml_config(
        CONFIGS_ROOT / "ltds" / "person_data_dictionary.yaml"
    )
    trips_cfg = load_yaml_config(CONFIGS_ROOT / "ltds" / "trip_dictionary.yaml")
    stages_cfg = load_yaml_config(CONFIGS_ROOT / "ltds" / "stage_dictionary.yaml")

    attrs, trips = ltds.load_years(
        DATA_ROOT,
        years=["LTDS2425"],
        hh_config=hh_cfg,
        person_config=person_cfg,
        person_data_config=person_data_cfg,
        trips_config=trips_cfg,
        stages_config=stages_cfg,
    )

    assert len(attrs) > 0
    assert len(trips) > 0
    assert "ltds" in attrs["source"].unique().to_list()
    assert set(trips["pid"]).issubset(set(attrs["pid"]))
    attrs, trips = fix.missing_columns(attrs, trips)
    attrs, trips = filter.columns(attrs, trips)
    attrs, trips = fix.fix_types(attrs, trips)
    assert verify.columns(attrs, trips)
