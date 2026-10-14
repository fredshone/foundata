import os
from pathlib import Path

import polars as pl
import pytest

from foundata import (
    config_validator,
    filter,
    fix,
    ktdb,
    verify,
)
from foundata.utils import (
    config_for_year,
    get_config_path,
    load_yaml_config,
    split_employment_type,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
# the KTDB folder, with one subdirectory per survey year
DATA_ROOT = os.getenv("FOUNDATA_KTDB_DATA", str(FIXTURE_ROOT / "ktdb"))
CONFIGS_ROOT = get_config_path()


def nested_configs() -> tuple[dict, dict]:
    return tuple(
        load_yaml_config(CONFIGS_ROOT / "ktdb" / f"{n}_dictionary.yaml")
        for n in ("person", "trip")
    )


def configs(year: int) -> tuple[dict, dict]:
    """Person and trip configs resolved for one year."""
    return tuple(config_for_year(c, year) for c in nested_configs())


def check_template(attrs: pl.DataFrame, trips: pl.DataFrame):
    attrs = split_employment_type(attrs)
    attrs, trips = fix.missing_columns(attrs, trips)
    attrs, trips = filter.columns(attrs, trips)
    attrs, trips = fix.fix_types(attrs, trips)
    assert verify.columns(attrs, trips)


def test_ktdb_load():
    person_cfg, trips_cfg = nested_configs()

    attrs, trips = ktdb.load(
        DATA_ROOT,
        person_config=person_cfg,
        trips_config=trips_cfg,
        year=2021,
    )

    assert len(attrs) > 0
    assert len(trips) > 0
    assert "ktdb" in attrs["source"].unique().to_list()
    assert attrs["survey"].unique().to_list() == ["ktdb2021"]
    assert set(trips["pid"]).issubset(set(attrs["pid"]))
    check_template(attrs, trips)


def test_ktdb_load_persons_hh_income_not_truncated_to_zero(monkeypatch):
    """Config brackets for hh_income (person_dictionary.yaml `hh_income:`)
    are in millions of KRW, e.g. code 3 -> [3, 5]. The exchange-rate
    conversion must happen after scaling those bounds to actual KRW, not
    before: applying `krw_to_euro` (0.00058) directly to a raw sample like
    3 and then `int(...)`-truncating (as `sample_to_euro` does) collapses
    every real income bracket to 0, leaving only the "don't know" code
    (99, a single-element bound) correctly null — i.e. exactly the
    null-or-0 pattern this test guards against.
    """
    monkeypatch.setattr("foundata.utils.random.randint", lambda a, b: b)

    person_cfg, _ = configs(2021)
    persons = ktdb.load_persons_2021(Path(DATA_ROOT) / "2021", person_cfg)

    known = persons.filter(pl.col("hh_income").is_not_null())
    assert known.height > 0
    assert (known["hh_income"] > 0).all()


@pytest.mark.parametrize("year", [2016, 2010], ids=["2016", "2010"])
@pytest.mark.parametrize("weekend", [False, True], ids=["weekday", "weekend"])
def test_ktdb_wave_load(year, weekend):
    person_cfg, trips_cfg = nested_configs()
    attrs, trips = ktdb.load(
        DATA_ROOT, person_cfg, trips_cfg, year=year, weekend=weekend
    )

    survey = f"ktdb{year}{'we' if weekend else ''}"
    assert len(attrs) > 0
    assert len(trips) > 0
    assert attrs["pid"].is_unique().all()
    assert attrs["survey"].unique().to_list() == [survey]
    assert attrs["source"].unique().to_list() == ["ktdb"]
    assert attrs["pid"].str.starts_with(f"{survey}_").all()
    assert attrs["hid"].str.starts_with(f"{survey}_").all()
    assert set(trips["pid"]).issubset(set(attrs["pid"]))
    assert (trips.group_by("pid").agg(pl.col("seq").min())["seq"] == 0).all()
    assert (attrs["year"] == year).all()
    if weekend:
        assert set(attrs["day"]) <= {"saturday", "sunday"}
        assert attrs["pid"].str.contains("_(sat|sun)$").all()
        # every person has both diaries
        persons = attrs["pid"].str.replace("_(sat|sun)$", "")
        assert (persons.value_counts()["count"] == 2).all()
    elif year == 2010:
        assert (attrs["day"] == "unknown").all()
    check_template(attrs, trips)


def test_ktdb_2010_ids_split_weekend_diaries():
    legs = pl.DataFrame(
        {"sheet_code": ["10010530202", "10010530202"], "member": ["2", "8"]}
    )
    out = ktdb.with_ids_2010(legs, "ktdb2010we", offset=6)
    assert out["hid"].to_list() == ["ktdb2010we_010010530202"] * 2
    assert out["pid"].to_list() == [
        "ktdb2010we_010010530202_2_sat",
        "ktdb2010we_010010530202_2_sun",
    ]


def test_ktdb_2010_links_transfer_legs():
    _, trips_cfg = configs(2010)
    legs = pl.DataFrame(
        {
            "sheet_code": ["010010530202"] * 3,
            "member": ["1"] * 3,
            "leg": ["1", "2", "3"],
            "purpose": ["3 ", "3 ", "2 "],  # 출근, 출근, 귀가 (space-padded)
            "mode": ["1 ", "4 ", "4 "],  # walk, bus, bus
            "origin": ["1", "4", "2"],  # 집, 환승, 직장
            "ozone": ["001", "001", "2"],  # unpadded, as in the weekend xlsx
            "tst": ["0800", "0815", "1800"],
            "end_type": ["4", "2", "1"],  # 환승, 직장, 집
            "dzone": ["001", "002", "001"],
            "tet": ["815", "0850", "1830"],
        }
    ).with_columns(pl.all().str.strip_chars())
    mapping = {c: c for c in legs.columns}
    trips_cfg = {**trips_cfg, "column_mappings": mapping}

    trips = ktdb.load_trips_2010(legs, trips_cfg, "ktdb2010", None, {})

    assert trips["seq"].to_list() == [0, 1]
    assert trips["tst"].to_list() == [480, 1080]
    assert trips["tet"].to_list() == [530, 1110]
    assert trips["mode"].to_list() == ["bus", "bus"]
    assert trips["oact"].to_list() == ["home", "work"]
    assert trips["dact"].to_list() == ["work", "home"]
    # different 251 zones: centroid distance; the second trip is 002 -> 001
    assert trips["distance"][0] == pytest.approx(trips["distance"][1])


def test_ktdb_2016_clock_handles_next_day():
    _, trips_cfg = configs(2016)
    df = pl.DataFrame(
        {
            "ampm": ["1", "2", "3"],
            "hr": ["8", "0", "1"],
            "min": ["5", "30", "0"],
        }
    )
    out = df.select(t=ktdb.clock("ampm", "hr", "min", trips_cfg))
    assert out["t"].to_list() == [485, 750, 1500]


def test_ktdb_2016_stages_set_end_time_mode_and_access():
    _, trips_cfg = configs(2016)
    columns = {
        c: [None] for c in trips_cfg["column_mappings"] if c != "토일구분"
    }
    row = {
        "IDX": "1",
        "행정구역번호": "1101053-0001-01",
        "가구원구분": "1",
        "통행유무": "1",
        "최초출발지_유형": "1",
        "목적통행_통행순서": "1",
        "통행목적": "4",  # 출근
        "목적통행_출발시각(오전 오후)": "1",
        "목적통행_출발시각(시)": "8",
        "목적통행_출발시각(분)": "0",
        "목적통행_출발지_행정동코드": "1111053000",
        "목적통행_도착지_행정동코드": "1111054000",
        "수단통행1_교통수단": "1",  # walk, 12 min
        "수단통행1_도착시각(오전 오후)": "1",
        "수단통행1_도착시각(시)": "8",
        "수단통행1_도착시각(분)": "12",
        "수단통행2_교통수단": "4",  # bus
        "수단통행2_도착시각(오전 오후)": "1",
        "수단통행2_도착시각(시)": "8",
        "수단통행2_도착시각(분)": "40",
    }
    raw = pl.DataFrame({**columns, **{k: [v] for k, v in row.items()}})

    trips = ktdb.load_trips_2016(raw, trips_cfg, "ktdb2016")

    assert trips["pid"].to_list() == ["ktdb2016_1_1101053-0001-01_1"]
    assert trips["seq"].to_list() == [0]
    assert trips["tst"].to_list() == [480]
    assert trips["tet"].to_list() == [520]
    assert trips["mode"].to_list() == ["bus"]
    assert trips["oact"].to_list() == ["home"]
    assert trips["dact"].to_list() == ["work"]
    assert trips["access_egress_distance"][0] == pytest.approx(1.0)


def test_ktdb_centroid_distances_fall_back_to_speed_within_a_zone():
    centroids = pl.DataFrame(
        {"zone": ["a", "b"], "lat": [37.5, 37.6], "lon": [127.0, 127.0]}
    )
    trips = pl.DataFrame(
        {
            "ozone": ["a", "a", "a"],
            "dzone": ["b", "a", "x"],
            "mode": ["car", "car", "walk"],
            "tst": [0, 0, 0],
            "tet": [30, 30, 60],
        }
    )
    speeds = {"car": 16, "walk": 5, "unknown": 10}

    out = ktdb.centroid_distances(trips, centroids, 1.3, speeds)

    # 0.1 degree of latitude is ~11.1 km
    assert out["distance"][0] == pytest.approx(11.12 * 1.3, rel=1e-3)
    assert out["distance"][1] == pytest.approx(8.0)  # same zone: 0.5 h x 16
    assert out["distance"][2] == pytest.approx(5.0)  # unknown zone


def test_ktdb_configs_validate():
    assert config_validator.validate_source("ktdb", CONFIGS_ROOT)


@pytest.mark.parametrize("year", [2021, 2016, 2010])
def test_ktdb_configs_resolve_for_every_year(year):
    person_cfg, trips_cfg = configs(year)
    for cfg in (person_cfg, trips_cfg):
        assert cfg["column_mappings"]
    assert person_cfg["krw_to_euro"] > 0


def test_ktdb_mode_speeds_consistent_across_years():
    speeds = [configs(y)[1]["mode_speeds"] for y in (2010, 2016, 2021)]
    assert speeds[0] == speeds[1] == speeds[2]


def test_ktdb_load_years_stacks_all_surveys():
    person_cfg, trips_cfg = nested_configs()
    attrs, trips = ktdb.load_years(DATA_ROOT, person_cfg, trips_cfg)
    assert set(attrs["survey"]) == {
        "ktdb2021",
        "ktdb2016",
        "ktdb2016we",
        "ktdb2010",
        "ktdb2010we",
    }
    assert attrs["pid"].is_unique().all()
    assert set(trips["pid"]).issubset(set(attrs["pid"]))
    check_template(attrs, trips)


def test_ktdb_load_rejects_unknown_surveys():
    person_cfg, trips_cfg = nested_configs()
    with pytest.raises(ValueError, match="Unsupported"):
        ktdb.load(DATA_ROOT, person_cfg, trips_cfg, year=2021, weekend=True)


def test_ktdb_zone_distances_store_ordered_pairs():
    # load_trips looks pairs up as (min, max), so every stored pair must be
    # ordered and every centroid zone must appear
    dist = ktdb.load_distances()
    assert (dist["ozone"] <= dist["dzone"]).all()
    zones = set(dist["ozone"]) | set(dist["dzone"])
    assert zones == set(ktdb.load_centroids())


def test_ktdb_load_trips_finds_distance_for_known_zone_pairs():
    # either zone order must hit the table; blank zones are left to the
    # speed fallback
    _, trips_cfg = configs(2021)
    root = Path(DATA_ROOT) / "2021"
    zones = pl.read_csv(
        root / "trips.csv", ignore_errors=True, encoding="euc-kr"
    ).select(
        pid="idx",
        seq="th_seq",
        ozone="sTP1_1_5",
        dzone="TP1_1_5",
    )
    known = list(ktdb.load_centroids())
    trips = ktdb.load_trips_2021(root, trips_cfg).join(
        zones.filter(
            pl.col("ozone").is_in(known) & pl.col("dzone").is_in(known)
        ),
        on=["pid", "seq"],
    )
    assert (trips["ozone"] > trips["dzone"]).any()
    assert trips["distance"].null_count() == 0


def test_ktdb_warns_about_zones_missing_from_distances(capsys):
    distances = pl.DataFrame({"_lo": ["1111053000"], "_hi": ["1111054000"]})
    trips = pl.DataFrame(
        {
            "ozone": ["1111053000", "9999999999", " "],
            "dzone": ["1111054000", "1111053000", "1111053000"],
        }
    )
    ktdb.warn_unknown_zones(trips, distances)
    out = capsys.readouterr().out
    # the blank zone is left to the speed fallback, not reported
    assert "1 (33.33%) trips use 1 zone codes" in out
    assert "9999999999" in out

    ktdb.warn_unknown_zones(trips.head(1), distances)
    assert capsys.readouterr().out == ""
