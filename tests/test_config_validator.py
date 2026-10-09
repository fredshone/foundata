import pytest

from foundata import utils
from foundata.config_validator import (
    check_required_fields,
    validate_column_mappings,
    validate_value_mappings,
)


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
    config = {
        "column_mappings": {"RAW_COL": "hh_incme"}
    }  # typo — flagged as warning
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


def test_validate_column_mappings_year_keyed_without_default(attr_template):
    # TUS style: year keys only, no "default", with 2019 nested one level
    # further by raw table
    config = {
        "column_mappings": {
            "2024": {"Household_Size": "hh_size"},
            "2019": {
                "demographics": {"b3q5": "age"},
                "day": {"b5q5": "bad_field"},
            },
        }
    }
    warnings = validate_column_mappings(config, attr_template)
    assert len(warnings) == 1
    assert "bad_field" in warnings[0]


def test_check_required_fields_nested_year_keyed():
    hh = {"column_mappings": {"2024": {"A": "hh_size"}}}
    person = {"column_mappings": {"2019": {"demographics": {"B": "age"}}}}
    warnings = check_required_fields(hh, person)
    assert not any("'hh_size'" in w for w in warnings)
    assert not any("'age'" in w for w in warnings)
    assert any("'sex'" in w for w in warnings)


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
