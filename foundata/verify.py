from pathlib import Path

import polars as pl
import yaml


def template() -> Path:
    return Path(__file__).parent.parent / "configs" / "core" / "template.yaml"


def template_attributes_columns() -> set[str]:
    # load yaml config and return set of expected columns
    with open(template()) as f:
        config = yaml.safe_load(f)
    return set(config["attribute_columns"])


def template_trips_columns() -> set[str]:
    # load yaml config and return set of expected columns
    with open(template()) as f:
        config = yaml.safe_load(f)
    return set(config["trip_columns"])


def columns(attributes: pl.DataFrame, trips: pl.DataFrame) -> bool:
    expected_attributes = template_attributes_columns()
    expected_trips = template_trips_columns()

    actual_attributes = set(attributes.columns)
    actual_trips = set(trips.columns)

    missing_attributes = expected_attributes - actual_attributes
    missing_trips = expected_trips - actual_trips

    if missing_attributes:
        print(f"Missing columns in attributes: {missing_attributes}")
    if missing_trips:
        print(f"Missing columns in trips: {missing_trips}")

    return not missing_attributes and not missing_trips
