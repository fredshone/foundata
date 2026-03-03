import io
from contextlib import redirect_stdout
from pathlib import Path

import polars as pl
import pytest

from foundata import utils


# --- sample_int_range ---

def test_sample_int_range_in_bounds():
    for _ in range(1000):
        result = utils.sample_int_range((10, 100))
        assert 10 <= result <= 100


def test_sample_int_range_single_elem_returns_none():
    result = utils.sample_int_range((0,))
    assert result is None


# --- sample_us_to_euro ---

def test_sample_us_to_euro_conversion():
    results = [utils.sample_us_to_euro((10000, 10000)) for _ in range(10)]
    # 10000 * 0.85 = 8500
    for r in results:
        assert r == pytest.approx(8500, abs=1)


def test_sample_us_to_euro_single_elem_returns_none():
    assert utils.sample_us_to_euro((0,)) is None


# --- sample_uk_to_euro ---

def test_sample_uk_to_euro_conversion():
    results = [utils.sample_uk_to_euro((10000, 10000)) for _ in range(10)]
    # 10000 * 1.14 = 11400
    for r in results:
        assert r == pytest.approx(11400, abs=1)


def test_sample_uk_to_euro_single_elem_returns_none():
    assert utils.sample_uk_to_euro((0,)) is None


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
    trips = pl.DataFrame({
        "pid": ["p1", "p1"],
        "tst": [0, 30],
        "tet": [30, 90],
        "distance": [10.0, 20.0],
    })
    result = utils.compute_avg_speed(attributes, trips)
    assert "avg_speed" in result.columns
    assert result["avg_speed"][0] == pytest.approx(20.0, rel=1e-4)


def test_compute_avg_speed_null_for_no_trips():
    attributes = pl.DataFrame({"pid": ["p1", "p2"]})
    trips = pl.DataFrame({
        "pid": ["p1"],
        "tst": [0],
        "tet": [60],
        "distance": [30.0],
    })
    result = utils.compute_avg_speed(attributes, trips)
    assert result.filter(pl.col("pid") == "p2")["avg_speed"][0] is None


def test_compute_avg_speed_filters_zero_duration():
    # trip with tet == tst should be excluded
    attributes = pl.DataFrame({"pid": ["p1"]})
    trips = pl.DataFrame({
        "pid": ["p1"],
        "tst": [60],
        "tet": [60],
        "distance": [10.0],
    })
    result = utils.compute_avg_speed(attributes, trips)
    assert result["avg_speed"][0] is None


def test_compute_avg_speed_filters_null_distance():
    attributes = pl.DataFrame({"pid": ["p1"]})
    trips = pl.DataFrame({
        "pid": ["p1"],
        "tst": [0],
        "tet": [60],
        "distance": [None],
    })
    result = utils.compute_avg_speed(attributes, trips)
    assert result["avg_speed"][0] is None


def test_compute_avg_speed_non_negative():
    attributes = pl.DataFrame({"pid": ["p1", "p2", "p3"]})
    trips = pl.DataFrame({
        "pid": ["p1", "p2", "p3"],
        "tst": [0, 0, 0],
        "tet": [30, 60, 120],
        "distance": [5.0, 40.0, 100.0],
    })
    result = utils.compute_avg_speed(attributes, trips)
    speeds = result["avg_speed"].drop_nulls().to_list()
    assert all(s >= 0 for s in speeds)
