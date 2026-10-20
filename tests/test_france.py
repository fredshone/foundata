import datetime as dt
import json
import os
from pathlib import Path

import polars as pl

from foundata import france, location
from foundata.utils import get_config_path, load_yaml_config

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
DATA_ROOT = os.getenv("FOUNDATA_FRANCE_DATA", str(FIXTURE_ROOT / "france"))
CONFIGS_ROOT = get_config_path()

WEEKDAYS = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]


def load_configs():
    return [
        load_yaml_config(CONFIGS_ROOT / "france" / name)
        for name in (
            "hh_dictionary.yaml",
            "person_dictionary.yaml",
            "trip_dictionary.yaml",
        )
    ]


def test_france_load():
    hh_cfg, person_cfg, trips_cfg = load_configs()

    attrs, trips = france.load(Path(DATA_ROOT), hh_cfg, person_cfg, trips_cfg)

    root = Path(DATA_ROOT)
    surveys = france.survey_dirs(root, [])
    assert surveys
    persons = pl.concat(
        pl.read_parquet(s / france.PERSONS_PATH, columns=["is_surveyed"])
        for s in surveys
    )
    # only persons surveyed for trips are kept, with or without trips
    assert len(attrs) == persons["is_surveyed"].sum()
    assert attrs["pid"].is_unique().all()
    assert set(trips["pid"]).issubset(set(attrs["pid"]))
    assert attrs["source"].unique().to_list() == ["france"]
    assert attrs["country"].unique().to_list() == ["fr"]
    assert attrs["day"].is_in(WEEKDAYS + ["unknown"]).all()
    assert attrs["year"].is_not_null().all()
    assert attrs["weight"].dtype == pl.Float32
    assert attrs["hh_income"].dtype == pl.Int32
    assert (attrs["hh_size"] >= 1).all()
    assert trips.group_by("pid").agg(pl.col("seq").min())["seq"].max() == 0
    assert trips["seq"].dtype == pl.Int8
    assert trips["distance"].dtype == pl.Float32
    assert set(trips.columns) == {
        "pid",
        "seq",
        "ozone",
        "dzone",
        "oact",
        "dact",
        "mode",
        "tst",
        "tet",
        "distance",
        "ocode",
        "dcode",
    }
    # location keys (ids once the runner indexes them)
    assert trips["dcode"].is_not_null().mean() > 0.9


def test_exclude_surveys():
    hh_cfg, person_cfg, trips_cfg = load_configs()
    root = Path(DATA_ROOT)
    first = france.survey_dirs(root, [])[0].name
    hh_cfg = {**hh_cfg, "exclude_surveys": [first]}

    attrs, _ = france.load(root, hh_cfg, person_cfg, trips_cfg)

    assert not attrs["pid"].str.starts_with(f"{first}_").any()


def test_fallback_year_is_survey_midpoint():
    meta = {"start_date": "2018-05-01", "end_date": "2019-04-30"}
    assert france.fallback_year(meta) == 2018
    meta = {"start_date": "2019-11-05", "end_date": "2020-10-17"}
    assert france.fallback_year(meta) == 2020


def test_density_zone():
    hh_cfg = load_configs()[0]
    df = pl.DataFrame(
        {"z": [1, 2, 3, 4, 5, 6, 7, None]}, schema={"z": pl.UInt8}
    )
    result = df.select(france.density_zone("z", hh_cfg))["z"].to_list()
    assert result == [
        "urban",
        "urban",
        "suburban",
        "suburban",
        "rural",
        "rural",
        "rural",
        "unknown",
    ]


def test_income_samples_annual_band():
    hh_cfg = load_configs()[0]
    df = pl.DataFrame(
        {
            "income_lower": [800, 5500, None],
            "income_upper": [1200, None, None],
        },
        schema={"income_lower": pl.UInt16, "income_upper": pl.UInt16},
    )
    result = df.select(income=france.income(hh_cfg))["income"]
    assert result.dtype == pl.Int32
    assert 800 * 12 <= result[0] <= 1200 * 12
    # open top band is capped at open_band_upper
    assert 5500 * 12 <= result[1] <= hh_cfg["open_band_upper"] * 12
    assert result[2] is None


def test_household_day_and_date():
    hh_cfg = load_configs()[0]
    hhs = pl.DataFrame(
        {
            "hid": [1, 2, 3],
            "hh_size": [2, 2, 1],
            "cars": [1, 0, 2],
            "motorcycles": [1, None, 0],
            "dwelling": ["house", "apartment", None],
            "ownership": ["owner:unspecified", "tenant:private", None],
            "hh_zone": [1, 6, None],
            "income_lower": [None, None, None],
            "income_upper": [None, None, None],
            "hh_weekday": [None, "friday", None],
            "interview_date": [
                dt.date(2015, 3, 10),
                dt.date(2015, 3, 7),
                None,
            ],
            "home_detailed_zone": ["001", None, None],
            "point_lng": [None, 2.35, None],
            "point_lat": [None, 48.85, None],
            "home_insee": ["69123", "75056", "69123"],
        },
        schema_overrides={
            "income_lower": pl.UInt16,
            "income_upper": pl.UInt16,
        },
    )
    trips = pl.DataFrame(
        {
            "hid": [1, 1, 2, 2],
            "weekday": ["monday", "monday", "tuesday", "friday"],
            "date": [
                dt.date(2015, 3, 9),
                dt.date(2015, 3, 9),
                dt.date(2015, 3, 3),
                dt.date(2015, 3, 6),
            ],
        }
    )
    result = france.load_households(hhs, trips, hh_cfg).sort("hid")
    # household 1: weekday and date shared by all its trips
    # household 2: trips_weekday, and the day before the interview since
    # its trips' dates differ
    # household 3: no trips and no dates
    assert result["hh_day"].to_list() == ["monday", "friday", None]
    assert result["hh_date"].to_list() == [
        dt.date(2015, 3, 9),
        dt.date(2015, 3, 6),
        None,
    ]
    assert result["vehicles"].to_list() == [2, 0, 2]
    assert result["dwelling"].to_list() == ["house", "flat", "unknown"]
    assert result["ownership"].to_list() == ["owned", "rented", "unknown"]
    assert result["hh_zone"].to_list() == ["urban", "rural", "unknown"]


def test_home_location_zone_then_point_then_commune():
    hhs = pl.DataFrame(
        {
            # 99999999 is not a Lyon 2015 zone, so falls through
            "home_detailed_zone": ["00101001", "99999999", None, None],
            "point_lng": [4.8, 2.35, None, None],
            "point_lat": [45.7, 48.85, None, None],
            "home_insee": ["69123", "75056", "69123", None],
        }
    )
    hhs = hhs.with_columns(
        **france.location_codes(
            "home",
            "lyon_2015",
            "home_detailed_zone",
            "point_lng",
            "point_lat",
            "home_insee",
        )
    )
    out = location.add(hhs, source="france", key=2015)
    assert out["hh_location"].to_list() == [
        "france/detailed_zone/lyon_2015/00101001",
        "france/point/2.35,48.85",
        "france/commune_2015/69123",
        None,
    ]


def test_survey_year():
    assert france.survey_year("toulouse_2013") == 2013
    assert france.survey_year("EGT2020") == 2020
    assert france.survey_year("Angouleme_2023") == 2023


def test_load_persons():
    person_cfg = load_configs()[1]
    persons = pl.DataFrame(
        {
            "pid": [1, 2, 3, 4, 5],
            "hid": [1, 1, 2, 2, 3],
            "age": [40, 38, 70, 12, 30],
            "sex": [True, False, False, True, None],
            "relationship": [
                "reference_person",
                "spouse",
                "reference_person",
                "child",
                "reference_person",
            ],
            "education": [
                "higher:at_least_bac+3",
                "secondary:bac",
                "primary",
                None,
                None,
            ],
            "employment": [
                "worker:full_time",
                "worker:part_time",
                "other:retired",
                "student:primary_or_secondary",
                "worker:unspecified",
            ],
            "pcs2020": [37, None, None, None, None],
            "pcs2003": [None, 64, None, None, None],
            "pcs_group": [3, 6, 7, None, 5],
            "work_only_at_home": [False, True, None, None, False],
            "telework": ["yes:weekly", None, None, None, None],
            "has_licence": ["yes", "in_progress", "no", None, "yes"],
            "disability": [None, False, True, None, None],
            "is_surveyed": [True, True, True, True, False],
            "weight": [1.0, 2.0, 3.0, 4.0, None],
        }
    )
    result = france.load_persons(persons, person_cfg).sort("pid")
    # the person not surveyed for trips is dropped
    assert result["pid"].to_list() == [1, 2, 3, 4]
    assert result["sex"].to_list() == ["female", "male", "male", "female"]
    assert result["employment"].to_list() == [
        "ft-employed",
        "pt-employed",
        "retired",
        "student",
    ]
    # PCS category (2020, else 2003) wins over the group
    assert result["occupation"].to_list() == [
        "managerial",
        "operator",
        "void",
        "void",
    ]
    assert result["can_wfh"].to_list() == ["yes", "yes", "void", "void"]
    assert result["education"].to_list() == [
        "degree",
        "high-school",
        "school",
        "unknown",
    ]
    assert result["has_licence"].to_list() == ["yes", "no", "no", "unknown"]
    assert result["disability"].to_list() == ["unknown", "no", "yes", "unknown"]


def test_occupation_group_fallback_and_unknown():
    person_cfg = load_configs()[1]
    persons = pl.DataFrame(
        {
            "pid": [1, 2],
            "hid": [1, 2],
            "age": [40, 40],
            "sex": [True, True],
            "relationship": ["reference_person"] * 2,
            "education": [None, None],
            "employment": ["worker:full_time"] * 2,
            "pcs2020": [None, None],
            "pcs2003": [None, None],
            "pcs_group": [5, None],
            "work_only_at_home": [False, None],
            "telework": ["no", None],
            "has_licence": ["yes", "yes"],
            "disability": [None, None],
            "is_surveyed": [True, True],
            "weight": [1.0, 1.0],
        },
        schema_overrides={
            "pcs2020": pl.UInt8,
            "pcs2003": pl.UInt8,
            "pcs_group": pl.UInt8,
            "education": pl.String,
            "disability": pl.Boolean,
        },
    )
    result = france.load_persons(persons, person_cfg).sort("pid")
    assert result["occupation"].to_list() == ["clerical", "unknown"]
    assert result["can_wfh"].to_list() == ["no", "unknown"]


def test_load_trips():
    hh_cfg, _, trips_cfg = load_configs()
    trips = pl.DataFrame(
        {
            "trip_id": [1, 2, 3],
            "pid": [1, 1, 1],
            "hid": [1, 1, 1],
            "seq": [2, 1, 3],
            "oact": ["work:usual", "home:main", "leisure:restaurant"],
            "dact": ["leisure:restaurant", "work:usual", "home:main"],
            "mode": [
                "walking",
                "public_transit:urban:metro",
                "public_transit:urban:bus",
            ],
            "ozone": [1, 4, 1],
            "dzone": [1, 1, 4],
            "tst": [720, 480, 1430],
            "tet": [730, 510, 1460],
            "travel_distance": [None, 12.0, None],
            "euclidean_distance": [1.0, 10.0, None],
            "weekday": ["monday"] * 3,
            "date": [None, None, None],
            "o_detailed_zone": ["00101002", "00101001", None],
            "d_detailed_zone": ["00101001", "00101002", None],
            "o_lng": [None, None, None],
            "o_lat": [None, None, None],
            "d_lng": [None, None, 2.35],
            "d_lat": [None, None, 48.85],
            "o_insee": ["69123"] * 3,
            "d_insee": ["69123"] * 3,
        },
        schema_overrides={"tst": pl.UInt16, "tet": pl.UInt16},
    )
    result = france.load_trips(trips, hh_cfg, trips_cfg, "lyon_2015", 2015)
    # sorted by seq
    assert result["ocode"].to_list() == [
        "france/detailed_zone/lyon_2015/00101001",
        "france/detailed_zone/lyon_2015/00101002",
        "france/commune_2015/69123",
    ]
    assert result["dcode"][2] == "france/point/2.35,48.85"
    assert result["seq"].to_list() == [0, 1, 2]
    assert result["oact"].to_list() == ["home", "work", "shop"]
    assert result["dact"].to_list() == ["work", "shop", "home"]
    assert result["mode"].to_list() == ["rail", "walk", "bus"]
    assert result["ozone"].to_list() == ["suburban", "urban", "urban"]
    distance = result["distance"].to_list()
    assert distance[0] == 12.0
    # euclidean distance x circuity when there is no travel distance
    assert abs(distance[1] - trips_cfg["circuity"]) < 1e-6
    assert distance[2] is None
    # times already run past midnight
    assert result["tet"].to_list() == [510, 730, 1460]


def test_access_egress():
    trips_cfg = load_configs()[2]
    legs = pl.DataFrame(
        {
            "trip_id": [1, 1, 1, 2, 3, 3],
            "pid": [1, 1, 1, 1, 2, 2],
            "leg_mode_group": [
                "walking",
                "public_transit",
                "walking",
                "car_driver",
                "public_transit",
                "walking",
            ],
            "leg_travel_distance": [0.4, 5.0, None, 3.0, 2.0, None],
            "leg_euclidean_distance": [None, None, 0.5, None, None, None],
            "leg_travel_time": [5, 10, 6, 10, 10, 20],
        }
    )
    result = france.access_egress(legs, trips_cfg).sort("pid")
    walk_speed = trips_cfg["walk_speed"]
    expected = [
        0.4 + 0.5 * trips_cfg["circuity"],  # travel, else euclidean x circuity
        20 / 60 * walk_speed,  # else travel time x walk speed
    ]
    assert result["pid"].to_list() == [1, 2]
    for got, want in zip(result["access_egress_distance"], expected):
        assert abs(got - want) < 1e-5


def test_access_egress_skips_transit_trips_without_walking():
    # trip 2 is transit with no walking leg: left out, not averaged as 0 km
    trips_cfg = load_configs()[2]
    legs = pl.DataFrame(
        {
            "trip_id": [1, 1, 2],
            "pid": [1, 1, 1],
            "leg_mode_group": ["walking", "public_transit", "public_transit"],
            "leg_travel_distance": [0.6, 5.0, 8.0],
            "leg_euclidean_distance": [None, None, None],
            "leg_travel_time": [8, 10, 15],
        }
    )
    result = france.access_egress(legs, trips_cfg)
    assert result["pid"].to_list() == [1]
    assert abs(result["access_egress_distance"][0] - 0.6) < 1e-5


def test_metadata_present():
    for survey in france.survey_dirs(Path(DATA_ROOT), []):
        meta = json.loads((survey / france.METADATA_PATH).read_text())
        assert {"start_date", "end_date"} <= set(meta)
