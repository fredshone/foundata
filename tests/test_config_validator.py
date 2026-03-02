from pathlib import Path

import pytest

from foundata.config_validator import (
    validate_all_sources,
    validate_column_mappings,
    validate_value_mappings,
)
from foundata import utils


@pytest.fixture(scope="session")
def attr_template():
    return utils.get_template_attributes()


@pytest.fixture(scope="session")
def trip_template():
    return utils.get_template_trips()


# --- validate_column_mappings ---

def test_validate_column_mappings_valid(attr_template):
    config = {"column_mappings": {"RAW_HH_ID": "hid", "RAW_SIZE": "hh_size"}}
    errors = validate_column_mappings(config, attr_template)
    assert errors == []


def test_validate_column_mappings_invalid_field_name(attr_template):
    config = {"column_mappings": {"RAW_COL": "hh_incme"}}  # typo — flagged as warning
    warnings = validate_column_mappings(config, attr_template)
    assert len(warnings) == 1
    assert "hh_incme" in warnings[0]


def test_validate_column_mappings_year_keyed(attr_template):
    config = {
        "column_mappings": {
            2022: {"RAW_HH": "hid"},
            "default": {"RAW_HH": "hid"},
        }
    }
    errors = validate_column_mappings(config, attr_template)
    assert errors == []


def test_validate_column_mappings_year_keyed_invalid(attr_template):
    config = {
        "column_mappings": {
            2022: {"RAW_HH": "bad_field"},
            "default": {"RAW_HH": "hid"},
        }
    }
    warnings = validate_column_mappings(config, attr_template)
    assert any("bad_field" in w for w in warnings)


# --- validate_value_mappings ---

def test_validate_value_mappings_valid(attr_template):
    config = {
        "column_mappings": {"RAW_SEX": "sex"},
        "sex": {1: "male", 2: "female", 9: "unknown"},
    }
    errors = validate_value_mappings(config, attr_template)
    assert errors == []


def test_validate_value_mappings_invalid_value(attr_template):
    config = {
        "column_mappings": {"RAW_SEX": "sex"},
        "sex": {1: "Male"},  # wrong case
    }
    errors = validate_value_mappings(config, attr_template)
    assert len(errors) == 1
    assert "Male" in errors[0]


def test_validate_value_mappings_skips_non_string(attr_template):
    config = {
        "column_mappings": {"RAW_INC": "hh_income"},
        "hh_income": {1: [0, 10000], 2: [10000, 20000]},
    }
    # hh_income has no 'set', so skipped entirely
    errors = validate_value_mappings(config, attr_template)
    assert errors == []


# --- validate_all_sources (regression guard) ---

def test_all_existing_sources_valid():
    configs_root = utils.get_config_path()
    result = validate_all_sources(configs_root)
    assert result is True, "One or more existing source configs are invalid"
