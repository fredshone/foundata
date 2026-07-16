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
    config = {
        "column_mappings": {
            2017: {"A": "hid"},
            "default": {"B": "hid"},
        }
    }
    result = utils.config_for_year(config, 2017)
    assert result["column_mappings"] == {"A": "hid"}


def test_config_for_year_falls_back_to_default():
    config = {
        "column_mappings": {
            2017: {"A": "hid"},
            "default": {"B": "hid"},
        }
    }
    result = utils.config_for_year(config, 2009)
    assert result["column_mappings"] == {"B": "hid"}


# --- table_joiner / check_overlap ---


def test_table_joiner_warns_on_missing_keys(capsys):
    lhs = pl.DataFrame({"pid": ["a", "b", "c"]})
    rhs = pl.DataFrame({"pid": ["a", "b", "x"]})
    utils.check_overlap(lhs, rhs, on="pid", lhs_name="attrs", rhs_name="trips")
    captured = capsys.readouterr()
    assert "Warning" in captured.out or "Missing" in captured.out


def test_check_overlap_all_present(capsys):
    lhs = pl.DataFrame({"pid": ["a", "b", "c"]})
    rhs = pl.DataFrame({"pid": ["a", "b", "c"]})
    utils.check_overlap(lhs, rhs, on="pid", lhs_name="attrs", rhs_name="trips")
    captured = capsys.readouterr()
    assert "Warning" not in captured.out


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
        {
            "pid": ["p1"],
            "tst": [0],
            "tet": [60],
            "distance": [30.0],
        }
    )
    result = utils.compute_avg_speed(attributes, trips)
    assert result.filter(pl.col("pid") == "p2")["avg_speed"][0] is None


def test_compute_avg_speed_filters_zero_duration():
    # trip with tet == tst should be excluded
    attributes = pl.DataFrame({"pid": ["p1"]})
    trips = pl.DataFrame(
        {
            "pid": ["p1"],
            "tst": [60],
            "tet": [60],
            "distance": [10.0],
        }
    )
    result = utils.compute_avg_speed(attributes, trips)
    assert result["avg_speed"][0] is None


def test_compute_avg_speed_filters_null_distance():
    attributes = pl.DataFrame({"pid": ["p1"]})
    trips = pl.DataFrame(
        {
            "pid": ["p1"],
            "tst": [0],
            "tet": [60],
            "distance": [None],
        }
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
        {"employment": ["employed", "student", "unemployed", "retired"]}
    )
    result = utils.split_employment_type(attributes)
    assert result["employment"].to_list() == [
        "employed",
        "student",
        "unemployed",
        "retired",
    ]
    assert result["employed_type"].to_list() == ["void", "void", "void", "void"]


def test_split_employment_type_unknown():
    attributes = pl.DataFrame({"employment": ["unknown", None]})
    result = utils.split_employment_type(attributes)
    assert result["employed_type"].to_list() == ["unknown", "unknown"]


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
