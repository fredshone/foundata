import os
from pathlib import Path

import polars as pl

from foundata import filter, fix, ktdb, verify
from foundata.utils import (
    get_config_path,
    load_yaml_config,
    split_employment_type,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
DATA_ROOT = os.getenv("FOUNDATA_KTDB_DATA", str(FIXTURE_ROOT / "ktdb"))
CONFIGS_ROOT = get_config_path()


def test_ktdb_load():
    person_cfg = load_yaml_config(
        CONFIGS_ROOT / "ktdb" / "person_dictionary.yaml"
    )
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
    attrs = split_employment_type(attrs)
    attrs, trips = fix.missing_columns(attrs, trips)
    attrs, trips = filter.columns(attrs, trips)
    attrs, trips = fix.fix_types(attrs, trips)
    assert verify.columns(attrs, trips)


def test_ktdb_load_persons_hh_income_not_truncated_to_zero(monkeypatch):
    """Config brackets for hh_income (person_dictionary.yaml `hh_income:`)
    are in millions of KRW, e.g. code 3 -> [3, 5]. The exchange-rate
    conversion must happen after scaling those bounds to actual KRW, not
    before: applying `KRW_TO_EURO` (0.00058) directly to a raw sample like
    3 and then `int(...)`-truncating (as `sample_to_euro` does) collapses
    every real income bracket to 0, leaving only the "don't know" code
    (99, a single-element bound) correctly null — i.e. exactly the
    null-or-0 pattern this test guards against.
    """
    monkeypatch.setattr("foundata.utils.random.randint", lambda a, b: b)

    person_cfg = load_yaml_config(
        CONFIGS_ROOT / "ktdb" / "person_dictionary.yaml"
    )
    persons = ktdb.load_persons(Path(DATA_ROOT), person_cfg)

    known = persons.filter(pl.col("hh_income").is_not_null())
    assert known.height > 0
    assert (known["hh_income"] > 0).all()
