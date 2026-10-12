import os
from pathlib import Path

import polars as pl
import pytest

from foundata import filter, fix, mong, verify
from foundata.utils import (
    config_for_year,
    get_config_path,
    load_yaml_config,
    split_employment_type,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
DATA_ROOT = os.getenv("FOUNDATA_MONG_DATA", str(FIXTURE_ROOT / "mong"))
CONFIGS_ROOT = get_config_path()


def _config(name):
    return load_yaml_config(CONFIGS_ROOT / "mong" / f"{name}_dictionary.yaml")


@pytest.fixture(scope="module")
def configs():
    return _config("hh"), _config("person"), _config("trip")


@pytest.fixture(scope="module")
def loaded(configs):
    return mong.load_years(Path(DATA_ROOT), *configs)


def _year(config, year):
    return config_for_year(config, str(year))


def test_mong_load(loaded):
    attrs, trips = loaded

    assert len(attrs) > 0
    assert len(trips) > 0
    assert attrs["source"].unique().to_list() == ["mong"]
    assert set(attrs["year"]) == {2023, 2019, 2011}
    assert attrs["pid"].is_unique().all()
    assert set(trips["pid"]).issubset(set(attrs["pid"]))
    assert "hh_mode" not in attrs.columns
    assert attrs["hh_income"].is_null().all()
    assert attrs["month"].is_between(1, 12).all()
    assert trips.group_by("pid").agg(pl.col("seq").min())["seq"].max() == 0
    assert trips["distance"].is_null().all()
    attrs = split_employment_type(attrs)
    attrs, trips = fix.missing_columns(attrs, trips)
    attrs, trips = filter.columns(attrs, trips)
    attrs, trips = fix.fix_types(attrs, trips)
    assert verify.columns(attrs, trips)


def test_mong_ids_carry_round_year(loaded):
    attrs, _ = loaded

    for year in (2023, 2019, 2011):
        rows = attrs.filter(pl.col("year") == year)
        prefix = f"mong{str(year)[2:]}"
        assert rows["hid"].str.starts_with(prefix).all()
        assert rows["pid"].str.starts_with(prefix).all()


def test_mong_modes_by_round(loaded):
    """2023 always has a mode (household fallback), 2019 never records one,
    2011 reads it from the place code."""
    attrs, trips = loaded
    trips = trips.join(attrs.select("pid", "year"), on="pid")

    assert (trips.filter(pl.col("year") == 2023)["mode"] != "unknown").all()
    assert (trips.filter(pl.col("year") == 2019)["mode"] == "unknown").all()
    assert (trips.filter(pl.col("year") == 2011)["mode"] != "unknown").any()


def test_mong_several_diary_days_per_person(loaded):
    """2019 and 2011 respondents keep 2-3 diary days, each its own plan
    (pid = person + weekday digit) with that day's weekday."""
    attrs, _ = loaded
    older = attrs.filter(pl.col("year") < 2023).with_columns(
        person=pl.col("pid").str.head(-1)
    )

    per_person = older.group_by("person").agg(
        n=pl.len(), days=pl.col("day").n_unique()
    )
    assert per_person["n"].max() > 1
    assert (per_person["n"] == per_person["days"]).all()


def test_mong_trip_chains_consistent(loaded):
    """Each trip's dact is the next trip's oact by construction."""
    _, trips = loaded

    nxt = trips.sort("pid", "seq").with_columns(
        next_oact=pl.col("oact").shift(-1).over("pid")
    )
    linked = nxt.filter(pl.col("next_oact").is_not_null())
    assert (linked["dact"] == linked["next_oact"]).all()


def test_load_households_2023(configs):
    hhs = pl.DataFrame(
        {
            "hhid": [1.0, 2.0, 3.0, 4.0],
            "HH14": [3.0, 1.0, 5.0, 2.0],
            "HH8": [1.0, 3.0, 4.0, 2.0],
            "HC1": [11.0, 12.0, 14.0, 96.0],
            "HC5": [1.0, 2.0, 3.0, None],
            "CA1M": [1.0, 2.0, 2.0, 2.0],  # motorbike
            "CA1N": [1.0, 2.0, 2.0, 2.0],  # car
            "CA1O": [2.0, 2.0, 2.0, 2.0],
            "CA1P": [2.0, 2.0, 2.0, 2.0],
            "CA1Q": [2.0, 2.0, 2.0, 2.0],
            "TP1": [1.0, 1.0, 2.0, 2.0],  # uses the bus
            "CA1L": [1.0, 1.0, 1.0, 2.0],  # bicycle
        }
    )

    out = mong.load_households(hhs, _year(configs[0], 2023), 2023)

    assert out["hid"].to_list() == ["mong231", "mong232", "mong233", "mong234"]
    assert out["hh_zone"].to_list() == ["urban", "rural", "rural", "urban"]
    assert out["dwelling"].to_list() == ["other", "flat", "house", "other"]
    assert out["ownership"].to_list() == ["owned", "rented", "other", "unknown"]
    assert out["vehicles"].to_list() == [2, 0, 0, 0]
    assert out["hh_income"].is_null().all()
    # car > bus > bike > walk
    assert out["hh_mode"].to_list() == ["car", "bus", "bike", "walk"]


def test_load_households_2011_ids_and_no_mode_fallback(configs):
    hhs = pl.DataFrame(
        {
            "HH1": [12.0],
            "HH2": [3.0],
            "HH15": [4.0],
            "HH3": [4.0],
            "HC1": [16.0],  # ger
            "HC6": [1.0],
            "CA5_K": [2.0],
            "CA5_L": [2.0],
            "CA5_M": [2.0],
            "CA5_N": [1.0],  # motorcycle
        }
    )

    out = mong.load_households(hhs, _year(configs[0], 2011), 2011)

    assert out["hid"].to_list() == ["mong110012003"]
    assert out["dwelling"].to_list() == ["other"]
    assert out["vehicles"].to_list() == [1]
    assert out["hh_mode"].to_list() == [None]


def test_load_persons_2023(configs):
    cfg = _year(configs[1], 2023)
    persons = pl.DataFrame(
        {
            "hhid": [1.0, 1.0, 2.0, 3.0],
            "pid": [11.0, 12.0, 21.0, 31.0],
            "HL6": [40.0, 13.0, 70.0, 30.0],
            "HL4": [1.0, 2.0, 2.0, 1.0],
            "HL3": [1.0, 3.0, 6.0, 11.0],
            "ED4": [7.0, 2.0, 1.0, 4.0],
            # herder producing for own use, but NSO counts them employed
            "EP18": [15.0, None, 19.0, 12.0],
            "p_work": [1.0, 0.0, 0.0, 0.0],
            "occ_class_d1": [6.0, None, None, None],
            "weight_pop": [100.0, 50.0, 20.0, 10.0],
            "HE5": [1.0, 1.0, 1.0, 1.0],
            "HE7": [1.0, 1.0, 1.0, 2.0],
            "HE8": [2.0, 1.0, 3.0, 1.0],  # walking: 3 = a lot of difficulty
            "HE9": [1.0, 1.0, 1.0, 1.0],
            "HE10": [1.0, 1.0, 1.0, 1.0],
            "HE11": [1.0, 1.0, 1.0, None],
        }
    ).rename(cfg["column_mappings"])

    out = mong.load_persons(persons, cfg, 2023)

    assert out["person"].to_list() == [
        "mong2311",
        "mong2312",
        "mong2321",
        "mong2331",
    ]
    assert out["hid"].to_list() == ["mong231", "mong231", "mong232", "mong233"]
    assert out["relationship"].to_list() == [
        "self",
        "child",
        "relative",
        "unrelated",
    ]
    assert out["education"].to_list() == [
        "degree",
        "school",
        "none",
        "high-school",
    ]
    assert out["employment"].to_list() == [
        "employed",
        "unknown",
        "retired",
        "unemployed",
    ]
    assert out["occupation"].to_list() == ["manual", "void", "void", "void"]
    assert out["disability"].to_list() == ["no", "no", "yes", "unknown"]
    assert out["year"].to_list() == [2023] * 4


def test_load_persons_2019_employment_rules_in_order(configs):
    """The first matching rule wins: working beats studying, and someone
    not working or studying is classed by why they aren't working."""
    cfg = _year(configs[1], 2019)
    n = 4
    persons = pl.DataFrame(
        {
            "HH2": [1.0] * n,
            "HH3": [2.0] * n,
            "HL1": [1.0, 2.0, 3.0, 4.0],
            "HL6": [30.0, 20.0, 65.0, 40.0],
            "HL4": [1.0] * n,
            "HL3": [1.0] * n,
            "ED3": [8.0, 4.0, 2.0, 98.0],
            "EP6_iscocode": [2411.0, None, None, None],
            "pweight": [1.0] * n,
            "EP3": [1.0, 2.0, 2.0, 2.0],
            "EP4": [None, 2.0, 2.0, 2.0],
            "ED1": [1.0, 1.0, 2.0, 2.0],  # 1: studying
            "EP14": [None, None, 7.0, None],  # 7: retired
            "EP16": [None, None, None, 4.0],  # 4: household responsibilities
            "EP13": [None, None, 2.0, 2.0],
            **{
                c: [1.0] * n for c in ["HE2", "HE4", "HE5", "HE6", "HE7", "HE8"]
            },
        }
    ).rename(cfg["column_mappings"])

    out = mong.load_persons(persons, cfg, 2019)

    assert out["person"].to_list() == [
        f"mong190001002{i:02d}" for i in range(1, 5)
    ]
    assert out["employment"].to_list() == [
        "employed",
        "student",
        "retired",
        "unemployed",
    ]
    assert out["occupation"].to_list() == [
        "professional",
        "void",
        "void",
        "void",
    ]
    assert out["education"].to_list() == [
        "degree",
        "high-school",
        "school",
        "unknown",
    ]
    assert out["disability"].to_list() == ["no"] * 4


def test_load_persons_2011_direct_disability_question(configs):
    cfg = _year(configs[1], 2011)
    persons = pl.DataFrame(
        {
            "hid_cluster": [1.0, 1.0, 1.0],
            "hid_hh": [1.0, 1.0, 1.0],
            "pid_line": [1.0, 2.0, 3.0],
            "age": [50.0, 45.0, 8.0],
            "sex": [1.0, 2.0, 1.0],
            "relationship": [1.0, 2.0, 9.0],
            "weight": [10.0] * 3,
            "education": [7.0, 4.0, None],
            "occupation": [6121.0, None, None],
            "disability": [
                2.0,
                1.0,
                None,
            ],  # HE2: 1 yes, 2 no; not asked of under 12s
            "EP1": [2.0, 2.0, None],
            "EP2": [1.0, 2.0, None],  # herding for the household
            "EP4": [None, 2.0, None],
            "ED2": [2.0, 2.0, None],
            "EP20": [None, 10.0, None],  # household duties
        }
    )

    out = mong.load_persons(persons, cfg, 2011)

    assert out["disability"].to_list() == ["no", "yes", "unknown"]
    assert out["employment"].to_list() == ["employed", "unemployed", "unknown"]
    assert out["occupation"].to_list() == ["manual", "void", "void"]
    assert out["relationship"].to_list() == ["self", "partner", "relative"]


def _episodes(rows, pid="p1", **cols):
    """rows: (start "HH:MM", end "HH:MM", activity code, place[, mode code]).
    place None = not asked (as for sleep)."""
    return pl.DataFrame(
        [
            {
                "pid": pid,
                "episode": float(i + 1),
                "start_h": float(row[0][:2]),
                "start_m": float(row[0][3:]),
                "end_h": float(row[1][:2]),
                "end_m": float(row[1][3:]),
                "activity_code": float(row[2]),
                "location": None if row[3] is None else float(row[3]),
                "mode_code": float(row[4]) if len(row) > 4 else None,
            }
            for i, row in enumerate(rows)
        ],
        schema_overrides={"location": pl.Float64, "mode_code": pl.Float64},
    )


def _attrs(hh_mode="walk"):
    return pl.DataFrame(
        {"pid": ["p1"], "hh_zone": ["rural"], "hh_mode": [hh_mode]}
    )


@pytest.fixture(scope="module")
def trips23(configs):
    return _year(configs[2], 2023)


def test_load_trips_recorded_travel(trips23):
    episodes = _episodes(
        [
            ("04:00", "07:00", 911, None),  # sleep, place not asked = home
            ("07:00", "07:30", 182, 8, 4),  # commute by bus
            ("07:30", "12:00", 110, 2),  # work
            ("12:00", "12:20", 182, 8, 1),  # walk home
            ("12:20", "04:00", 921, 1),  # eat at home
        ]
    )

    trips = mong.load_trips(episodes, trips23, _attrs())

    assert trips["seq"].to_list() == [0, 1]
    assert trips["tst"].to_list() == [420, 720]
    assert trips["tet"].to_list() == [450, 740]
    assert trips["oact"].to_list() == ["home", "work"]
    assert trips["dact"].to_list() == ["work", "home"]
    assert trips["mode"].to_list() == ["bus", "walk"]
    assert trips["ozone"].to_list() == ["rural", "rural"]
    assert trips["dzone"].to_list() == ["rural", "rural"]


def test_load_trips_wraps_past_midnight(trips23):
    """The diary runs 04:00 -> 04:00: times after midnight are > 1440."""
    episodes = _episodes(
        [
            ("04:00", "20:00", 911, 1),
            ("20:00", "20:30", 750, 8, 2),
            ("20:30", "23:30", 712, 3),  # socialising at someone's home
            ("23:30", "00:30", 750, 8, 2),
            ("00:30", "04:00", 911, None),
        ]
    )

    trips = mong.load_trips(episodes, trips23, _attrs())

    assert trips["tst"].to_list() == [1200, 1410]
    assert trips["tet"].to_list() == [1230, 1470]
    assert trips["dact"].to_list() == ["visit", "home"]


def test_load_trips_repairs_diary_overrunning_4am(trips23):
    """A last episode recorded after a 04:10 end starts where the previous
    one ended rather than jumping back to the morning."""
    episodes = _episodes(
        [
            ("04:00", "08:00", 911, None),
            ("08:00", "08:30", 182, 8, 2),
            ("08:30", "17:00", 110, 2),
            ("17:00", "17:30", 182, 8, 2),
            ("17:30", "04:10", 911, None),
            ("04:10", "04:00", 911, None),
        ]
    )

    out = mong._prepare_episodes(episodes, trips23)

    assert out["tst"].to_list()[-2:] == [1050, 1690]
    assert out["tet"].to_list()[-2:] == [1690, 1690]


def test_load_trips_home_activity_away_retyped(trips23):
    """Eating (a "home" code) at a restaurant is a meal out -> shop."""
    episodes = _episodes(
        [
            ("04:00", "12:00", 911, 1),
            ("12:00", "12:20", 950, 8, 1),
            ("12:20", "13:00", 921, 4),  # eating at a restaurant
            ("13:00", "13:20", 950, 8, 1),
            ("13:20", "04:00", 921, 1),
        ]
    )

    trips = mong.load_trips(episodes, trips23, _attrs())

    assert trips["dact"].to_list() == ["shop", "home"]


def test_load_trips_implied_between_fixed_places(trips23):
    """Home -> work/school -> shop -> home with no travel recorded: every
    change between buildings is a trip, taking the last 10 min of the
    earlier activity. With no mode recorded all day, the household's best
    available mode is used."""
    episodes = _episodes(
        [
            ("04:00", "09:00", 911, None),
            ("09:00", "13:00", 110, 2),
            ("13:00", "13:30", 371, 5),
            ("13:30", "04:00", 921, 1),
        ]
    )

    trips = mong.load_trips(episodes, trips23, _attrs(hh_mode="car"))

    assert trips["tst"].to_list() == [530, 770, 800]
    assert trips["tet"].to_list() == [540, 780, 810]
    assert trips["oact"].to_list() == ["home", "work", "shop"]
    assert trips["dact"].to_list() == ["work", "shop", "home"]
    assert trips["mode"].to_list() == ["car", "car", "car"]


def test_load_trips_no_implied_trip_for_herding(trips23):
    """Herding in the countryside (A8 = 8, "other") or fetching water
    elsewhere is not a trip-worthy activity, so moving to it from home
    implies no trip."""
    episodes = _episodes(
        [
            ("04:00", "08:00", 911, None),
            ("08:00", "12:00", 212, 8, 7),  # herding on horseback
            ("12:00", "13:00", 921, 1),
            ("13:00", "14:00", 242, 7),  # fetching water elsewhere (home)
            ("14:00", "04:00", 911, None),
        ]
    )

    trips = mong.load_trips(episodes, trips23, _attrs())

    assert trips.is_empty()


def test_load_trips_implied_to_elsewhere_for_trip_activity(trips23):
    episodes = _episodes(
        [
            ("04:00", "10:00", 911, None),
            ("10:00", "12:00", 832, 7),  # exercising elsewhere: leisure
            ("12:00", "04:00", 921, 1),
        ]
    )

    trips = mong.load_trips(episodes, trips23, _attrs())

    assert trips["oact"].to_list() == ["home", "leisure"]
    assert trips["dact"].to_list() == ["leisure", "home"]


def test_load_trips_escort_codes_travel_only_on_road(trips23):
    """Accompanying a child on the road is travel (with its mode); at
    school it is an escort activity."""
    episodes = _episodes(
        [
            ("04:00", "08:00", 911, None),
            ("08:00", "08:20", 442, 8, 2),  # driving the child
            ("08:20", "08:40", 442, 2),  # at the school
            ("08:40", "09:00", 442, 8, 2),
            ("09:00", "04:00", 921, 1),
        ]
    )

    trips = mong.load_trips(episodes, trips23, _attrs())

    assert trips["oact"].to_list() == ["home", "escort"]
    assert trips["dact"].to_list() == ["escort", "home"]
    assert trips["mode"].to_list() == ["car", "car"]


def test_load_trips_mode_priority_and_person_fallback(trips23):
    """A trip walking then by bus is a bus trip (car > bus > other > bike >
    walk); a travel episode noted at home has no mode of its own and takes
    the person's best mode that day."""
    episodes = _episodes(
        [
            ("04:00", "08:00", 911, None),
            ("08:00", "08:10", 182, 8, 1),  # walk to the stop
            ("08:10", "08:40", 182, 8, 4),  # bus
            ("08:40", "17:00", 110, 2),
            ("17:00", "17:30", 182, 1),  # commute recorded "at home"
            ("17:30", "04:00", 921, 1),
        ]
    )

    trips = mong.load_trips(episodes, trips23, _attrs(hh_mode="car"))

    assert trips["tst"].to_list() == [480, 1020]
    assert trips["tet"].to_list() == [520, 1050]
    assert trips["mode"].to_list() == ["bus", "bus"]


def test_load_trips_no_travel_gives_no_trips(trips23):
    episodes = _episodes(
        [("04:00", "12:00", 911, None), ("12:00", "04:00", 311, 1)]
    )

    trips = mong.load_trips(episodes, trips23, _attrs())

    assert trips.is_empty()


def test_load_trips_2019_has_no_mode(configs):
    """2019 records "on the way" (ID7 = 3) but never a mode, and has no
    household fallback, so its trips are unknown."""
    episodes = _episodes(
        [
            ("04:00", "08:00", 911, 1),
            ("08:00", "08:30", 182, 3),
            ("08:30", "17:00", 110, 2),
            ("17:00", "17:30", 182, 3),
            ("17:30", "04:00", 921, 1),
        ]
    )

    trips = mong.load_trips(
        episodes, _year(configs[2], 2019), _attrs(hh_mode=None)
    )

    assert trips["dact"].to_list() == ["work", "home"]
    assert trips["mode"].to_list() == ["unknown", "unknown"]


def test_load_trips_2011_times_from_durations_and_mode_from_place(configs):
    """2011 gives only durations, in order from 04:00, and the place code
    doubles as the mode in transport (21 bus, 27 walk)."""
    rows = [  # (duration, national activity code, place/mode code)
        (180, 1011, 11),  # sleep at home
        (40, 119, 21),  # going to work by bus
        (480, 111, 13),  # main job at work
        (30, 119, 27),  # walking home
        (710, 1021, 11),  # eating at home, to 04:00
    ]
    episodes = pl.DataFrame(
        {
            "pid": ["p1"] * len(rows),
            "episode": [float(i) for i in range(len(rows))],
            "duration": [float(r[0]) for r in rows],
            "activity_code": [float(r[1]) for r in rows],
            "location": [float(r[2]) for r in rows],
        }
    )

    trips = mong.load_trips(
        episodes, _year(configs[2], 2011), _attrs(hh_mode=None)
    )

    assert trips["tst"].to_list() == [420, 940]
    assert trips["tet"].to_list() == [460, 970]
    assert trips["oact"].to_list() == ["home", "work"]
    assert trips["dact"].to_list() == ["work", "home"]
    assert trips["mode"].to_list() == ["bus", "walk"]


def test_diary_days_one_row_per_diary(configs):
    episodes = pl.DataFrame(
        {
            "pid": ["a3", "a3", "a4", "b7"],
            "person": ["a", "a", "a", "b"],
            "day": [3.0, 3.0, 4.0, 7.0],
            "month": [6.0, 6.0, 6.0, 12.0],
        }
    )

    out = mong.diary_days(episodes, _year(configs[2], 2019)).sort("pid")

    assert out["person"].to_list() == ["a", "a", "b"]
    assert out["day"].to_list() == ["wednesday", "thursday", "sunday"]
    assert out["month"].to_list() == [6, 6, 12]


def test_load_unsupported_year_raises(configs):
    with pytest.raises(ValueError, match="Unsupported MONG year"):
        mong.load(Path(DATA_ROOT), *configs, year=2015)
