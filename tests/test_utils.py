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
