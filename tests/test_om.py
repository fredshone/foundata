import os
from pathlib import Path

import polars as pl

from foundata import nl_income, om
from foundata.utils import get_config_path, load_yaml_config

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
DATA_ROOT = os.getenv("FOUNDATA_OM_DATA", str(FIXTURE_ROOT / "om"))
CONFIGS_ROOT = get_config_path()


def load_configs():
    return [
        load_yaml_config(CONFIGS_ROOT / "om" / name)
        for name in (
            "hh_dictionary.yaml",
            "person_dictionary.yaml",
            "trip_dictionary.yaml",
        )
    ]


def test_om_load():
    hh_cfg, person_cfg, trips_cfg = load_configs()

    attrs, trips = om.load(
        DATA_ROOT, list(om.DATA_FILES), hh_cfg, person_cfg, trips_cfg
    )

    assert attrs["pid"].is_unique().all()
    assert set(trips["pid"]).issubset(set(attrs["pid"]))
    assert attrs["pid"].str.starts_with("om").all()
    assert attrs["source"].unique().to_list() == ["om"]
    assert attrs["country"].unique().to_list() == ["nl"]
    assert set(attrs["year"]) == set(om.DATA_FILES)
    # whole households: PersID is HHID * 100 + RegelNr
    assert attrs.select(
        pl.col("pid").str.starts_with(pl.col("hid")).all()
    ).item()
    assert attrs.group_by("hid").len()["len"].max() > 1
    assert attrs["month"].is_between(1, 12).all()
    assert (
        attrs["day"]
        .is_in(
            ["monday", "tuesday", "wednesday", "thursday", "friday"]
            + ["saturday", "sunday"]
        )
        .all()
    )
    assert attrs["hh_income"].drop_nulls().min() >= 0
    assert attrs["weight"].dtype == pl.Float32
    assert not {"away", "personal_income"} & set(attrs.columns)

    joined = trips.join(attrs.select("pid", hh="hid"), on="pid")
    assert (joined["hid"] == joined["hh"]).all()
    assert trips.group_by("pid").agg(pl.col("seq").min())["seq"].max() == 0
    assert (trips["distance"] > 0).all()
    for col in ("oact", "dact", "mode", "ozone", "dzone"):
        assert trips[col].null_count() == 0


def raw(**columns) -> pl.DataFrame:
    return pl.DataFrame(columns, schema_overrides={"FactorP": pl.Float32})


def persons_raw(**overrides):
    rows = dict(
        Persoon=[1, 1, 1, 1, 0],
        PersID=[101, 102, 201, 301, 301],
        HHID=[1, 1, 2, 3, 3],
        Leeftijd=[40, 8, 30, 70, 70],
        Geslacht=[1, 2, 2, 1, 1],
        MaatPart=[2, 5, 7, 8, 8],
        Opleid=[4, 0, 3, 6, 6],
        Rijbewij=[1, 0, 2, 3, 3],
        Inkomen=[6, 0, 3, 7, 7],
        Responsc=[1, 1, 3, 2, 2],
        Weekdag=[2, 2, 7, 1, 1],
        Jaar=[2006, 2006, 2006, 2006, 2006],
        Maand=[5, 5, 6, 7, 7],
        Dag=[1, 1, 3, 2, 2],
        FactorP=[100.0, 110.0, 120.0, 130.0, 130.0],
    )
    return raw(**(rows | overrides))


def test_persons_mapped_one_row_each():
    _, person_cfg, _ = load_configs()

    persons = om.load_persons(persons_raw(), person_cfg, 2006)

    assert persons["pid"].to_list() == [101, 102, 201, 301]
    assert persons["employment"].to_list() == [
        "ft-employed",
        "student",
        "unemployed",  # WAO
        "retired",
    ]
    assert persons["education"].to_list() == [
        "degree",
        "unknown",  # under 12
        "high-school",
        "unknown",
    ]
    assert persons["has_licence"].to_list() == ["yes", "no", "no", "unknown"]
    assert persons["day"].to_list() == [
        "monday",
        "monday",
        "saturday",
        "sunday",
    ]
    assert persons["relationship"].unique().to_list() == ["unknown"]
    income = persons["personal_income"].to_list()
    assert 30000 <= income[0] <= 500000  # open band, Pareto tail
    assert income[1] == 0  # under 12, no own income
    assert 7500 <= income[2] <= 15000
    assert income[3] is None


def test_household_income_summed_and_nulled():
    hh_cfg, person_cfg, _ = load_configs()
    persons = om.load_persons(
        persons_raw(
            Persoon=[1, 1, 1, 1, 1],
            PersID=[101, 102, 201, 301, 302],
            HHID=[1, 1, 2, 3, 3],
            Inkomen=[3, 1, 4, 2, 7],
        ),
        person_cfg,
        2006,
    )
    data = raw(
        Huishoud=[1, 1, 1],
        HHID=[1, 2, 3],
        AantPers=[2, 2, 2],  # household 2 has a non-responding member
        HAuto=[1, 0, 2],
        Sted=[1, 3, 5],
        WoGem=[363, 344, 1680],
    )

    hhs = om.load_households(data, hh_cfg, persons, 2006).sort("hid")

    income = hhs["hh_income"].to_list()
    # net 7.5-15k (plus no own income), converted to gross by rank
    lo, hi = nl_income.to_gross(
        pl.Series([7500, 15000]), nl_income.load_config(), 2006
    )
    assert lo <= income[0] <= hi
    assert income[1] is None  # incomplete household
    assert income[2] is None  # a member's income unknown
    assert hhs["hh_zone"].to_list() == ["urban", "suburban", "rural"]
    assert hhs["vehicles"].to_list() == [1, 0, 2]
    assert "respondents" not in hhs.columns
