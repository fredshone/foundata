import os
from pathlib import Path

import pandas as pd
import polars as pl
import pyreadstat
import pytest

from foundata import nl_income, ovin
from foundata.utils import get_config_path, load_yaml_config

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
DATA_ROOT = os.getenv("FOUNDATA_OVIN_DATA", str(FIXTURE_ROOT / "ovin"))
CONFIGS_ROOT = get_config_path()


def load_configs():
    return [
        load_yaml_config(CONFIGS_ROOT / "ovin" / name)
        for name in (
            "hh_dictionary.yaml",
            "person_dictionary.yaml",
            "trip_dictionary.yaml",
        )
    ]


def test_ovin_load():
    hh_cfg, person_cfg, trips_cfg = load_configs()

    attrs, trips = ovin.load(
        DATA_ROOT, list(ovin.DATA_FILES), hh_cfg, person_cfg, trips_cfg
    )

    assert attrs["pid"].is_unique().all()
    assert set(trips["pid"]).issubset(set(attrs["pid"]))
    assert (attrs["pid"] == attrs["hid"]).all()
    assert attrs["source"].unique().to_list() == ["ovin"]
    assert attrs["country"].unique().to_list() == ["nl"]
    assert set(attrs["year"]) == set(ovin.DATA_FILES)
    for year in ovin.DATA_FILES:
        assert attrs["pid"].str.starts_with(str(year)).any()
    assert (
        attrs["day"]
        .is_in(
            ["monday", "tuesday", "wednesday", "thursday", "friday"]
            + ["saturday", "sunday"]
        )
        .all()
    )
    assert attrs["month"].is_between(1, 12).all()
    assert attrs["weight"].dtype == pl.Float32
    assert attrs["hh_income"].drop_nulls().min() >= 0
    assert "away" not in attrs.columns

    assert trips.group_by("pid").agg(pl.col("seq").min())["seq"].max() == 0
    assert trips["seq"].dtype == pl.Int8
    assert (trips["distance"] > 0).all()
    assert "trip_type" not in trips.columns
    for col in ("oact", "dact", "mode", "ozone", "dzone"):
        assert trips[col].null_count() == 0
    # chained: each trip leaves from the previous trip's destination
    chained = trips.sort("pid", "seq").with_columns(
        prev=pl.col("dact").shift(1).over("pid")
    )
    assert (
        chained.filter(pl.col("prev").is_not_null())
        .select((pl.col("oact") == pl.col("prev")).all())
        .item()
    )


def raw(**columns) -> pl.DataFrame:
    return pl.DataFrame(columns, schema_overrides={"FactorP": pl.Float32})


def test_read_sav_matches_names_case_insensitively(tmp_path):
    path = tmp_path / "ovin.sav"
    pyreadstat.write_sav(
        pd.DataFrame({"OP": [1.0], "Wogem": [363.0], "FactorP": [1.5]}),
        str(path),
    )

    data = ovin.read_sav(path, ["OP", "WoGem", "FactorP"])

    assert data.columns == ["OP", "WoGem", "FactorP"]
    assert data.schema["WoGem"] == pl.Int64
    assert data.schema["FactorP"] == pl.Float32
    assert data["WoGem"].to_list() == [363]


def test_read_sav_names_missing_columns(tmp_path):
    path = tmp_path / "ovin.sav"
    pyreadstat.write_sav(pd.DataFrame({"OP": [1.0]}), str(path))

    with pytest.raises(
        KeyError, match=r"ovin\.sav has no column\(s\) \['Doel'\]"
    ):
        ovin.read_sav(path, ["OP", "Doel"])
    with pytest.raises(FileNotFoundError, match="missing.sav"):
        ovin.read_sav(tmp_path / "missing.sav", ["OP"])


def test_households_unknown_income_and_vehicles():
    hh_cfg, _, _ = load_configs()
    data = raw(
        OP=[1, 1, 1, 0],
        OPID=[1, 2, 3, 3],
        HHPers=[2, 1, 3, 3],
        HHBestInk=[3, 7, None, None],
        HHAuto=[1, 10, 2, 2],
        Sted=[1, 3, 5, 5],
        WoGem=[363, 344, 1680, 1680],
        Weekdag=[1, 2, 7, 7],
        Jaar=[2015, 2015, 2015, 2015],
        Maand=[3, 3, 3, 3],
        Dag=[1, 2, 7, 7],
    )

    hhs = ovin.load_households(data, hh_cfg, 2015).sort("hid")

    assert hhs["hid"].to_list() == [1, 2, 3]
    income = hhs["hh_income"].to_list()
    # disposable band 20-30k, converted to gross at 2015 ranks
    lo, hi = nl_income.to_gross(
        pl.Series([20000, 30000]), nl_income.load_config(), 2015
    )
    assert 20000 < lo <= income[0] <= hi
    assert income[1:] == [None, None]
    assert hhs["vehicles"].to_list() == [1, None, 2]
    assert hhs["hh_zone"].to_list() == ["urban", "suburban", "rural"]
    assert hhs["day"].to_list() == ["sunday", "monday", "saturday"]
    assert "home_gemeente" not in hhs.columns


def test_gemeente_zone_from_respondents_home_gemeente():
    respondents = raw(
        WoGem=[363, 363, 1680, 1680, 1680, 344],
        Sted=[1, 1, 5, 4, 5, 3],
    )

    zones = ovin.gemeente_zone(respondents).sort("gemeente")

    # the most common Sted per gemeente
    assert zones.rows() == [(344, 3), (363, 1), (1680, 5)]


def trip_rows(**overrides):
    """Raw trip rows for respondent 1, all Verpl == 1 by default."""
    rows = dict(
        OPID=[1, 1, 1, 1],
        Verpl=[1, 1, 1, 1],
        VerplNr=[1, 2, 3, 4],
        Hvm=[6, 15, 22, 5],
        Vertrekp=[2, 2, 2, 2],
        Doel=[2, 10, 1, 8],
        VertUur=[8, 12, 17, 23],
        VertMin=[0, 0, 0, 50],
        AankUur=[8, 12, 17, 24],
        AankMin=[30, 30, 30, 10],
        AfstV=[150, 20, 140, 35],
        VertGem=[363, 344, 344, 0],
        AankGem=[344, 344, 363, 363],
    )
    return raw(**(rows | overrides))


def test_trips_mapped_and_chained():
    _, person_cfg, trips_cfg = load_configs()
    zones = pl.DataFrame({"gemeente": [363, 344], "sted": [1, 3]})
    trips = ovin.load_trips(trip_rows(), trips_cfg, zones)
    persons = pl.DataFrame({"pid": [1], "away": [1]})

    _, trips = ovin.drop_non_regular(persons, trips, person_cfg, trips_cfg)

    assert trips["seq"].to_list() == [0, 1, 2, 3]
    assert trips["mode"].to_list() == ["car", "bike", "walk", "bus"]
    # Vertrekp 2 (ander adres) for the first trip, then chained; the
    # toeren/wandelen round trip returns to work
    assert trips["oact"].to_list() == ["other", "work", "work", "home"]
    assert trips["dact"].to_list() == ["work", "work", "home", "shop"]
    assert trips["ozone"].to_list() == [
        "urban",
        "suburban",
        "suburban",
        "unknown",  # VertGem 0: abroad
    ]
    assert trips["dzone"].to_list() == [
        "suburban",
        "suburban",
        "urban",
        "urban",
    ]
    assert trips["distance"].to_list() == [15.0, 2.0, 14.0, 3.5]
    # AankUur 24 already runs past midnight
    assert trips["tst"].to_list() == [480, 720, 1020, 1430]
    assert trips["tet"].to_list() == [510, 750, 1050, 1450]


def test_drop_non_regular_travellers():
    _, person_cfg, trips_cfg = load_configs()
    zones = pl.DataFrame(
        {"gemeente": [], "sted": []},
        schema=[
            ("gemeente", pl.Int64),
            ("sted", pl.Int64),
        ],
    )
    # 1: regular trips only; 2: a holiday trip among regular ones;
    # 3: no trips; 4: only professional trips (Weggeweest 2)
    rows = trip_rows(
        OPID=[1, 1, 2, 2],
        VerplNr=[1, 2, 1, 2],
        Verpl=[1, 1, 1, 5],
    )
    trips = ovin.load_trips(rows, trips_cfg, zones)
    persons = pl.DataFrame({"pid": [1, 2, 3, 4], "away": [1, 1, 0, 2]})

    persons, trips = ovin.drop_non_regular(
        persons, trips, person_cfg, trips_cfg
    )

    assert persons["pid"].to_list() == [1, 3]
    assert "away" not in persons.columns
    assert trips["pid"].unique().to_list() == [1]
    assert "trip_type" not in trips.columns


def test_work_travel_blocks_absorbed_after_work():
    _, person_cfg, trips_cfg = load_configs()
    zones = pl.DataFrame(
        {"gemeente": [], "sted": []},
        schema=[("gemeente", pl.Int64), ("sted", pl.Int64)],
    )
    # 1: home -> work, two work-travel blocks, work -> home: kept
    # 2: a block at the start of the day: dropped
    # 3: home -> shop, then a block: dropped
    rows = raw(
        OPID=[1, 1, 1, 1, 2, 2, 3, 3],
        Verpl=[1, 2, 2, 1, 2, 1, 1, 2],
        VerplNr=[1, 2, 3, 4, 1, 2, 1, 2],
        Hvm=[6, None, None, 6, None, 6, 6, None],
        Vertrekp=[1] * 8,
        Doel=[2, 4, 4, 1, 4, 1, 8, 4],
        VertUur=[8, None, None, 17, None, 17, 9, None],
        VertMin=[0, None, None, 0, None, 0, 0, None],
        AankUur=[8, None, None, 17, None, 17, 9, None],
        AankMin=[30, None, None, 30, None, 30, 30, None],
        AfstV=[150, None, None, 150, None, 150, 20, None],
        VertGem=[363] * 8,
        AankGem=[363] * 8,
    )
    trips = ovin.load_trips(rows, trips_cfg, zones)
    persons = pl.DataFrame({"pid": [1, 2, 3], "away": [1, 1, 1]})

    persons, trips = ovin.drop_non_regular(
        persons, trips, person_cfg, trips_cfg
    )

    assert persons["pid"].to_list() == [1]
    assert trips["seq"].to_list() == [0, 1]
    assert trips["oact"].to_list() == ["home", "work"]
    assert trips["dact"].to_list() == ["work", "home"]
    assert trips["tst"].to_list() == [480, 1020]
