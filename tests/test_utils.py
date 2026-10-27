import polars as pl
import pytest

from foundata import utils

# --- sample_to_euro ---


def test_sample_to_euro_no_conversion():
    for _ in range(1000):
        result = utils.sample_to_euro((10, 100))
        assert 10 <= result <= 100


def test_sample_to_euro_single_elem_returns_none():
    result = utils.sample_to_euro((0,))
    assert result is None


def test_sample_to_euro_usd_conversion():
    results = [utils.sample_to_euro((10000, 10000), 0.85) for _ in range(10)]
    # 10000 * 0.85 = 8500
    for r in results:
        assert r == pytest.approx(8500, abs=1)


def test_sample_to_euro_gbp_conversion():
    results = [utils.sample_to_euro((10000, 10000), 1.14) for _ in range(10)]
    # 10000 * 1.14 = 11400
    for r in results:
        assert r == pytest.approx(11400, abs=1)


# --- config_for_year ---


def test_config_for_year_uses_year_specific():
    config = {"column_mappings": {2017: {"A": "hid"}, "default": {"B": "hid"}}}
    result = utils.config_for_year(config, 2017)
    assert result["column_mappings"] == {"A": "hid"}


def test_config_for_year_falls_back_to_default():
    config = {"column_mappings": {2017: {"A": "hid"}, "default": {"B": "hid"}}}
    result = utils.config_for_year(config, 2009)
    assert result["column_mappings"] == {"B": "hid"}


# --- compute_avg_speed ---


def test_compute_avg_speed_basic():
    # person "p1": 10 km in 30 min + 20 km in 60 min → 30 km / 1.5 h = 20 km/h
    attributes = pl.DataFrame({"pid": ["p1"]})
    trips = pl.DataFrame(
        {
            "pid": ["p1", "p1"],
            "tst": [0, 30],
            "tet": [30, 90],
            "distance": [10.0, 20.0],
        }
    )
    result = utils.compute_avg_speed(attributes, trips)
    assert "avg_speed" in result.columns
    assert result["avg_speed"][0] == pytest.approx(20.0, rel=1e-4)


def test_compute_avg_speed_null_for_no_trips():
    attributes = pl.DataFrame({"pid": ["p1", "p2"]})
    trips = pl.DataFrame(
        {"pid": ["p1"], "tst": [0], "tet": [60], "distance": [30.0]}
    )
    result = utils.compute_avg_speed(attributes, trips)
    assert result.filter(pl.col("pid") == "p2")["avg_speed"][0] is None


def test_compute_avg_speed_filters_zero_duration():
    # trip with tet == tst should be excluded
    attributes = pl.DataFrame({"pid": ["p1"]})
    trips = pl.DataFrame(
        {"pid": ["p1"], "tst": [60], "tet": [60], "distance": [10.0]}
    )
    result = utils.compute_avg_speed(attributes, trips)
    assert result["avg_speed"][0] is None


def test_compute_avg_speed_filters_null_distance():
    attributes = pl.DataFrame({"pid": ["p1"]})
    trips = pl.DataFrame(
        {"pid": ["p1"], "tst": [0], "tet": [60], "distance": [None]}
    )
    result = utils.compute_avg_speed(attributes, trips)
    assert result["avg_speed"][0] is None


def test_compute_avg_speed_non_negative():
    attributes = pl.DataFrame({"pid": ["p1", "p2", "p3"]})
    trips = pl.DataFrame(
        {
            "pid": ["p1", "p2", "p3"],
            "tst": [0, 0, 0],
            "tet": [30, 60, 120],
            "distance": [5.0, 40.0, 100.0],
        }
    )
    result = utils.compute_avg_speed(attributes, trips)
    speeds = result["avg_speed"].drop_nulls().to_list()
    assert all(s >= 0 for s in speeds)


# --- split_employment_type ---


def test_split_employment_type_collapses_ft_pt():
    attributes = pl.DataFrame({"employment": ["ft-employed", "pt-employed"]})
    result = utils.split_employment_type(attributes)
    assert result["employment"].to_list() == ["employed", "employed"]
    assert result["employed_type"].to_list() == ["ft", "pt"]


def test_split_employment_type_void_for_non_split_categories():
    attributes = pl.DataFrame(
        {"employment": ["student", "unemployed", "retired"]}
    )
    result = utils.split_employment_type(attributes)
    assert result["employment"].to_list() == [
        "student",
        "unemployed",
        "retired",
    ]
    assert result["employed_type"].to_list() == ["void", "void", "void"]


def test_split_employment_type_unknown():
    # bare "employed" has no ft/pt distinction recorded, same as
    # "unknown"/null employment.
    attributes = pl.DataFrame({"employment": ["unknown", None, "employed"]})
    result = utils.split_employment_type(attributes)
    assert result["employed_type"].to_list() == [
        "unknown",
        "unknown",
        "unknown",
    ]


# --- household_composition / add_member_counts / vehicles_per_adult ---


def test_household_composition_counts_complete_households():
    # h1 complete (2 of 2), h2 incomplete (1 of 3), h3 has an unknown age
    attributes = pl.DataFrame(
        {
            "pid": ["a", "b", "c", "d", "e"],
            "hid": ["h1", "h1", "h2", "h3", "h3"],
            "age": [40, 17, 30, 50, None],
            "hh_size": [2, 2, 3, 2, 2],
        }
    )
    result = utils.household_composition(attributes).sort("pid")
    assert result["hh_adults"].to_list() == [1, 1, None, None, None]
    assert result["hh_children"].to_list() == [1, 1, None, None, None]
    assert result["hh_adults"].dtype == pl.Int32


def test_household_composition_keeps_loader_counts():
    attributes = pl.DataFrame(
        {
            "pid": ["a"],
            "hid": ["h1"],
            "age": [40],
            "hh_size": [3],
            "hh_adults": [2],
            "hh_children": [None],
        }
    )
    result = utils.household_composition(attributes)
    assert result["hh_adults"].to_list() == [2]
    assert result["hh_children"].to_list() == [None]


def test_household_composition_null_hh_size_is_unknown():
    attributes = pl.DataFrame(
        {"pid": ["a"], "hid": ["h1"], "age": [40], "hh_size": [None]}
    )
    result = utils.household_composition(attributes)
    assert result["hh_adults"].to_list() == [None]


def test_add_member_counts_nulls_short_rosters():
    hhs = pl.DataFrame({"hid": ["h1", "h2", "h3"], "hh_size": [3, 2, None]})
    persons = pl.DataFrame(
        {
            "hid": ["h1", "h1", "h1", "h2", "h3"],
            "age": [5, 18, 70, 30, 10],
        }
    )
    result = utils.add_member_counts(hhs, persons).sort("hid")
    assert result["hh_adults"].to_list() == [2, None, 0]
    assert result["hh_children"].to_list() == [1, None, 1]
    assert result.columns == ["hid", "hh_size", "hh_adults", "hh_children"]


def test_vehicles_per_adult_categories():
    attributes = pl.DataFrame(
        {
            "vehicles": [0, 0, 1, 2, 3, 1, None, 2],
            "hh_adults": [2, None, 2, 2, 2, 0, 2, None],
        }
    )
    result = utils.vehicles_per_adult(attributes)
    assert result["veh_per_adult"].to_list() == [
        "0",
        "0",  # no vehicles, whatever the adults
        "0-1",
        "1+",  # exactly one per adult
        "1+",
        "1+",  # vehicles but no adults
        "unknown",
        "unknown",
    ]


# --- assign_retired ---


def test_assign_retired_reclassifies_unemployed_over_60_with_no_work_trips():
    attributes = pl.DataFrame(
        {"pid": ["p1"], "employment": ["unemployed"], "age": [65]}
    )
    trips = pl.DataFrame({"pid": ["p1"], "dact": ["shop"]})
    result = utils.assign_retired(attributes, trips)
    assert result["employment"].to_list() == ["retired"]


def test_assign_retired_keeps_unemployed_with_work_trip():
    attributes = pl.DataFrame(
        {"pid": ["p1"], "employment": ["unemployed"], "age": [65]}
    )
    trips = pl.DataFrame({"pid": ["p1"], "dact": ["work"]})
    result = utils.assign_retired(attributes, trips)
    assert result["employment"].to_list() == ["unemployed"]


def test_assign_retired_keeps_unemployed_under_age_threshold():
    attributes = pl.DataFrame(
        {"pid": ["p1"], "employment": ["unemployed"], "age": [59]}
    )
    trips = pl.DataFrame({"pid": ["p1"], "dact": ["shop"]})
    result = utils.assign_retired(attributes, trips)
    assert result["employment"].to_list() == ["unemployed"]


def test_assign_retired_leaves_non_unemployed_untouched():
    attributes = pl.DataFrame(
        {
            "pid": ["p1", "p2"],
            "employment": ["employed", "student"],
            "age": [65, 65],
        }
    )
    trips = pl.DataFrame(
        {"pid": [], "dact": []}, schema={"pid": pl.String, "dact": pl.String}
    )
    result = utils.assign_retired(attributes, trips)
    assert result["employment"].to_list() == ["employed", "student"]


def test_assign_retired_person_with_no_trips_is_reclassified():
    attributes = pl.DataFrame(
        {"pid": ["p1"], "employment": ["unemployed"], "age": [65]}
    )
    trips = pl.DataFrame(
        {"pid": [], "dact": []}, schema={"pid": pl.String, "dact": pl.String}
    )
    result = utils.assign_retired(attributes, trips)
    assert result["employment"].to_list() == ["retired"]


# --- resolve_activity_chain ---


def test_resolve_activity_chain_groups_by_multiple_columns():
    # NTS's JourSeq restarts each DayID, so chaining must group by
    # (pid, did) rather than pid alone — otherwise a round trip on day 2
    # would wrongly inherit day 1's last activity.
    data = pl.DataFrame(
        {
            "pid": ["1", "1", "1", "1"],
            "did": ["a", "a", "b", "b"],
            "seq": [1, 2, 1, 2],
            "oact": ["home", "unset", "home", "unset"],
            "dact": ["work", "home", "shop", None],
        }
    )

    result = utils.resolve_activity_chain(data, group_cols=["pid", "did"])
    result = result.sort(["did", "seq"])

    assert result["dact"].to_list() == ["work", "home", "shop", "shop"]
    assert result["oact"].to_list() == ["home", "work", "home", "shop"]


def test_combine_consecutive_acts_removes_self_loop_trip():
    # p1: single "there and back" trip (home -> home) → removed entirely
    trips = pl.DataFrame(
        {
            "pid": ["p1", "p2", "p2"],
            "seq": [0, 0, 1],
            "oact": ["home", "home", "work"],
            "dact": ["home", "work", "home"],
        }
    )
    result = utils.combine_consecutive_acts(trips)
    assert "p1" not in set(result["pid"])
    assert set(result["pid"]) == {"p2"}
    assert len(result) == 2


def test_combine_consecutive_acts_removes_middle_trip_within_day():
    # p1: home -> shop -> home -> home -> other; the redundant home->home
    # trip (seq 2) is dropped, merging the two home activities either side
    trips = pl.DataFrame(
        {
            "pid": ["p1"] * 5,
            "seq": [0, 1, 2, 3, 4],
            "oact": ["home", "shop", "home", "home", "other"],
            "dact": ["shop", "home", "home", "other", "home"],
        }
    )
    result = utils.combine_consecutive_acts(trips)
    assert result["seq"].to_list() == [0, 1, 3, 4]


def test_combine_consecutive_acts_keeps_types_not_in_list():
    trips = pl.DataFrame(
        {
            "pid": ["p1", "p1"],
            "seq": [0, 1],
            "oact": ["shop", "shop"],
            "dact": [
                "other",
                "other",
            ],  # consecutive, but "other" not restricted
        }
    )
    result = utils.combine_consecutive_acts(trips)
    assert len(result) == 2


def test_combine_consecutive_acts_keeps_non_consecutive_plan():
    trips = pl.DataFrame(
        {
            "pid": ["p1", "p1"],
            "seq": [0, 1],
            "oact": ["home", "work"],
            "dact": ["work", "home"],
        }
    )
    result = utils.combine_consecutive_acts(trips)
    assert len(result) == 2


def test_combine_consecutive_acts_custom_non_consecutive_types():
    trips = pl.DataFrame(
        {
            "pid": ["p1", "p1", "p2", "p2"],
            "seq": [0, 1, 0, 1],
            "oact": ["home", "home", "work", "home"],
            "dact": ["shop", "shop", "home", "home"],
        }
    )
    result = utils.combine_consecutive_acts(
        trips, non_consecutive_types=["shop"]
    )
    assert result.filter(pl.col("pid") == "p1")["seq"].to_list() == [0]
    assert result.filter(pl.col("pid") == "p2").height == 2


def test_combine_consecutive_acts_inconsistent_oact_dact_chain():
    """Regression: consecutive dacts are detected even when oact[i] != dact[i-1]."""
    trips = pl.DataFrame(
        {
            "pid": ["p1", "p1", "p1"],
            "seq": [0, 1, 2],
            "oact": ["home", "shop", "work"],
            "dact": ["shop", "home", "home"],  # dact[1]==dact[2]==home
        }
    )
    result = utils.combine_consecutive_acts(
        trips, non_consecutive_types=["home"]
    )
    assert result["seq"].to_list() == [0, 1]


def test_resolve_activity_chain_leading_round_trip_both_ends_unknown():
    # Unlike ODiN's VertLoc-derived oact, some sources (e.g. NTS) draw oact
    # from the same round-trip vocabulary as dact — so a group's first trip
    # can have both ends unresolved, with no real activity anywhere in the
    # group to anchor to. This should fall back to "unknown" rather than
    # leaving a null that fails downstream validation.
    data = pl.DataFrame(
        {
            "pid": ["1", "1"],
            "seq": [1, 2],
            "oact": [None, "unset"],
            "dact": [None, "shop"],
        }
    )

    result = utils.resolve_activity_chain(data, group_cols=["pid"]).sort("seq")

    assert result["dact"].to_list() == ["unknown", "shop"]
    assert result["oact"].to_list() == ["unknown", "unknown"]


def test_assign_education_to_escort():
    trips = pl.DataFrame(
        {
            "pid": ["p1", "p1"],
            "seq": [0, 1],
            "tst": [0, 50],
            "tet": [30, 110],
            "oact": ["home", "education"],
            "dact": ["education", "home"],
            "distance": [5.0, 5.0],
            "mode": ["walk", "walk"],
            "ozone": ["A", "B"],
            "dzone": ["B", "A"],
        },
        schema_overrides={"tst": pl.Int32, "tet": pl.Int32},
    )
    result = utils.assign_education_to_escort(trips)
    assert result["oact"].to_list() == ["home", "escort"]
    assert result["dact"].to_list() == ["escort", "home"]


def test_do_not_assign_education_to_escort():
    trips = pl.DataFrame(
        {
            "pid": ["p1", "p1"],
            "seq": [0, 1],
            "tst": [0, 180],
            "tet": [60, 230],
            "oact": ["home", "education"],
            "dact": ["education", "home"],
            "distance": [5.0, 5.0],
            "mode": ["walk", "walk"],
            "ozone": ["A", "B"],
            "dzone": ["B", "A"],
        },
        schema_overrides={"tst": pl.Int32, "tet": pl.Int32},
    )
    result = utils.assign_education_to_escort(trips)
    assert result["oact"].to_list() == ["home", "education"]
    assert result["dact"].to_list() == ["education", "home"]


# --- enumerate_ids ---


def test_enumerate_ids_numbers_from_zero_in_sorted_order():
    attrs = pl.DataFrame({"pid": ["x2", "x0", "x1"], "hid": ["h1", "h0", "h1"]})
    trips = pl.DataFrame({"pid": ["x1", "x1", "x2"], "seq": [0, 1, 0]})
    attrs, trips, keys = utils.enumerate_ids(attrs, trips)
    assert attrs["pid"].dtype == pl.Int32
    assert keys["pid"].to_list() == ["x0", "x1", "x2"]
    assert keys["hid"].to_list() == ["h0", "h1"]
    assert attrs["pid"].to_list() == [2, 0, 1]
    assert attrs["hid"].to_list() == [1, 0, 1]
    assert trips["pid"].to_list() == [1, 1, 2]


def test_enumerate_ids_covers_trip_only_pids():
    # a trip pid missing from attributes still gets an id (filtered later)
    attrs = pl.DataFrame({"pid": ["b"], "hid": ["h"]})
    trips = pl.DataFrame({"pid": ["a", "b"]})
    attrs, trips, _ = utils.enumerate_ids(attrs, trips)
    assert attrs["pid"].to_list() == [1]
    assert trips["pid"].to_list() == [0, 1]


def test_vehicles_per_adult_no_vehicles_column():
    attributes = pl.DataFrame({"hh_adults": [2]})
    result = utils.vehicles_per_adult(attributes)
    assert result["veh_per_adult"].to_list() == ["unknown"]
