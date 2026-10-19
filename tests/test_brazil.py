import os
import struct
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from foundata import brazil
from foundata.utils import get_config_path, load_yaml_config

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
DATA_ROOT = os.getenv("FOUNDATA_BRAZIL_DATA", str(FIXTURE_ROOT / "brazil"))
CONFIGS_ROOT = get_config_path()


def load_configs():
    return [
        load_yaml_config(CONFIGS_ROOT / "brazil" / name)
        for name in (
            "hh_dictionary.yaml",
            "person_dictionary.yaml",
            "trip_dictionary.yaml",
        )
    ]


def write_dbf(path: Path, fields: list[tuple[str, int]], rows: list[list[str]]):
    """Minimal dBase III writer: character fields, right-aligned values."""
    header_len = 32 + 32 * len(fields) + 1
    record_len = 1 + sum(length for _, length in fields)
    header = struct.pack("<B3xIHH20x", 3, len(rows), header_len, record_len)
    for name, length in fields:
        header += struct.pack("<11sc4xB15x", name.encode(), b"C", length)
    header += b"\x0d"
    records = b"".join(
        b" "
        + b"".join(
            value.rjust(length).encode()
            for value, (_, length) in zip(row, fields)
        )
        for row in rows
    )
    path.write_bytes(header + records + b"\x1a")


def test_read_dbf_blank_and_asterisk_fields_are_null(tmp_path):
    path = tmp_path / "t.dbf"
    write_dbf(
        path,
        [("ID", 3), ("N", 4), ("X", 2)],
        [["001", "12.5", "**"], ["002", "", "7"], ["003", "****", "*"]],
    )

    result = brazil.read_dbf(path)

    assert result["ID"].to_list() == ["001", "002", "003"]
    assert result["N"].to_list() == ["12.5", None, None]
    assert result["X"].to_list() == [None, "7", None]
    assert brazil.read_dbf(path, columns=["X"]).columns == ["X"]


def test_brazil_load():
    hh_cfg, person_cfg, trips_cfg = load_configs()

    attrs, trips = brazil.load(Path(DATA_ROOT), hh_cfg, person_cfg, trips_cfg)

    raw = brazil.read_dbf(
        Path(DATA_ROOT) / brazil.OD_PATH, columns=["ID_PESS", "N_VIAG"]
    )
    assert attrs["pid"].is_unique().all()
    assert len(attrs) == raw["ID_PESS"].n_unique()
    assert len(trips) == raw["N_VIAG"].is_not_null().sum()
    assert set(trips["pid"]).issubset(set(attrs["pid"]))
    assert attrs["pid"].str.starts_with("brazil").all()
    assert attrs["hid"].str.starts_with("brazil").all()
    assert attrs["source"].unique().to_list() == ["brazil"]
    assert trips.group_by("pid").agg(pl.col("seq").min())["seq"].max() == 0
    assert set(attrs["day"]) <= {
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
    }
    assert trips["oact"].is_in(["unknown"]).sum() == 0
    assert trips["mode"].is_in(["unknown"]).sum() == 0


def raw_families(**overrides) -> pl.DataFrame:
    """Two families in one household, as raw (string) DBF columns."""
    data = {
        "F_FAM": ["1", "1"],
        "ID_DOM": ["00010001", "00010001"],
        "ZONA": ["1", "1"],
        "NO_MORAD": ["3", "3"],
        "DATA": ["06092017", "06092017"],
        "FAMILIA": ["1", "2"],
        "RENDA_FA": ["1000.00", "500.00"],
        "CD_RENFA": ["1", "1"],
        "QT_AUTO": ["1", "0"],
        "QT_MOTO": ["0", "1"],
        "CONDMORA": ["2", "1"],
        "CO_DOM_X": ["333743", "333743"],
        "CO_DOM_Y": ["7394463", "7394463"],
    }
    data.update(overrides)
    return pl.DataFrame(data)


def test_load_households_aggregates_families():
    hh_cfg, _, _ = load_configs()

    hhs = brazil.load_households(raw_families(), hh_cfg, {1: "urban"})

    assert len(hhs) == 1
    row = hhs.row(0, named=True)
    assert row["hh_income"] == int(1500 * 12 * hh_cfg["brl_to_euro"])
    assert row["vehicles"] == 2
    assert row["ownership"] == "owned"  # first family
    assert row["hh_zone"] == "urban"
    assert row["date"] == date(2017, 9, 6)
    # dwelling point in central São Paulo
    source, level, point = row["hh_location"].split("/")
    assert (source, level) == ("brazil", "point")
    lon, lat = map(float, point.split(","))
    assert lon == pytest.approx(-46.63, abs=0.01)
    assert lat == pytest.approx(-23.55, abs=0.01)


def test_load_households_unknown_family_values_null_household():
    hh_cfg = {**load_configs()[0], "income_codes": [1, 2]}

    hhs = brazil.load_households(
        raw_families(CD_RENFA=["1", "3"], QT_AUTO=["1", None]), hh_cfg, {}
    )

    row = hhs.row(0, named=True)
    assert row["hh_income"] is None
    assert row["vehicles"] is None
    assert row["hh_zone"] == "unknown"


def test_travel_date():
    hh_cfg, _, _ = load_configs()
    attributes = pl.DataFrame(
        {
            "date": [
                date(2017, 9, 6),  # Wednesday interview, Tuesday travel
                date(2017, 9, 4),  # Monday interview, Friday travel
                date(2017, 9, 4),  # Monday interview, no trips
                date(2017, 9, 6),  # Wednesday interview, no trips
            ],
            "day": [3, 6, None, None],
        }
    )

    result = brazil.travel_date(attributes, hh_cfg["day"])

    assert result["day"].to_list() == [
        "tuesday",
        "friday",
        "friday",
        "tuesday",
    ]
    assert result["month"].to_list() == [9, 9, 9, 9]
    assert result["year"].to_list() == [2017] * 4


def test_load_trips_times_escort_and_access_egress():
    _, _, trips_cfg = load_configs()
    raw = pl.DataFrame(
        {
            "ID_PESS": ["p1", "p1", "p1"],
            "N_VIAG": ["1", "2", "3"],
            "ZONA_O": ["1", "2", "1"],
            "ZONA_D": ["2", "1", "2"],
            "CO_O_X": ["333743", None, "333743"],
            "CO_O_Y": ["7394463", None, "7394463"],
            "CO_D_X": ["333104", "333743", "327503"],
            "CO_D_Y": ["7394476", "7394463", "7392159"],
            "MOTIVO_O": ["8", "4", "8"],
            "MOTIVO_D": ["4", "8", "3"],
            "SERVIR_O": ["2", "1", "2"],
            "SERVIR_D": ["1", "2", "2"],
            "MODOPRIN": ["9", "9", "1"],
            "H_SAIDA": ["7", "8", "23"],
            "MIN_SAIDA": ["30", "0", "40"],
            "H_CHEG": ["7", "8", "0"],
            "MIN_CHEG": ["45", "15", "20"],
            "DISTANCIA": ["1000.00", "1000.00", "2000.00"],
            "ANDA_O": [None, None, "6"],
            "ANDA_D": [None, None, "12"],
        }
    )

    trips = brazil.load_trips(raw, trips_cfg, {1: "urban", 2: "rural"})

    assert trips["seq"].to_list() == [0, 1, 2]
    assert trips["oact"].to_list() == ["home", "escort", "home"]
    assert trips["dact"].to_list() == ["escort", "home", "work"]
    assert trips["mode"].to_list() == ["car", "car", "rail"]
    assert trips["ozone"].to_list() == ["urban", "rural", "urban"]
    assert trips["tst"].to_list() == [450, 480, 1420]
    assert trips["tet"].to_list() == [465, 495, 1460]  # wrapped past midnight
    assert trips["distance"][0] == pytest.approx(trips_cfg["circuity"])
    assert trips["access_egress_distance"].to_list() == [
        None,
        None,
        18 / 60 * trips_cfg["walk_speed"],
    ]
    # trip ends: the coordinate, else the OD zone
    assert trips["ocode"][1] == "brazil/od_zone_2017/2"
    assert trips["ocode"][0].startswith("brazil/point/-46.6")
    assert trips["dcode"].str.starts_with("brazil/point/").all()
