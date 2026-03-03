from pathlib import Path

import polars as pl

from foundata import fix, utils
from foundata.utils import table_joiner

SOURCE = "ktdb"


def load(
    data_root: str | Path,
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load and normalise ktdb survey data.

    Args:
        data_root: Path to the raw data directory for this source.
        hh_config: Parsed hh_dictionary.yaml config.
        person_config: Parsed person_dictionary.yaml config.
        trips_config: Parsed trip_dictionary.yaml config.

    Returns:
        (attributes, trips) DataFrames conforming to the template schema.
    """
    hhs = load_households(data_root, hh_config)
    persons = load_persons(data_root, person_config)
    attributes = table_joiner(hhs, persons, on="hid")
    trips = load_trips(data_root, trips_config)

    # Prefix IDs with source name for global uniqueness
    attributes = attributes.with_columns(
        pid=pl.lit(SOURCE) + pl.col("pid").cast(pl.String),
        hid=pl.lit(SOURCE) + pl.col("hid").cast(pl.String),
    )
    trips = trips.with_columns(
        pid=pl.lit(SOURCE) + pl.col("pid").cast(pl.String),
    )

    attributes = utils.compute_avg_speed(attributes, trips)

    return attributes, trips


def load_households(
    root: str | Path,
    config: dict,
) -> pl.DataFrame:
    """Load and normalise household records.

    TODO: Update file name and path to match raw data layout.
    TODO: Apply income range sampling via utils.sample_*_to_euro().
    TODO: Parse survey date into year/month columns if needed.
    """
    root = Path(root).expanduser()
    column_mapping = config["column_mappings"]

    # TODO: Replace "households.csv" with the actual file name.
    data = pl.read_csv(root / "households.csv", ignore_errors=True)
    data = data.select(column_mapping.keys()).rename(column_mapping)

    # TODO: Map coded values to canonical labels for each categorical field.
    # Example:
    # data = data.with_columns(
    #     rurality=pl.col("rurality").replace_strict(config["rurality"]),
    # )

    data = data.with_columns(
        source=pl.lit(SOURCE),
        country=pl.lit("unknown"),  # TODO: set ISO country code, e.g. "aus"
    )

    return data


def load_persons(
    root: str | Path,
    config: dict,
) -> pl.DataFrame:
    """Load and normalise person records.

    TODO: Update file name and path to match raw data layout.
    """
    root = Path(root).expanduser()
    column_mapping = config["column_mappings"]

    # TODO: Replace "persons.csv" with the actual file name.
    data = pl.read_csv(root / "persons.csv", ignore_errors=True)
    data = data.select(column_mapping.keys()).rename(column_mapping)

    # TODO: Map coded values to canonical labels.
    # TODO: Fill unmapped categorical fields with "unknown".
    # Example:
    # data = data.with_columns(
    #     sex=pl.col("sex").replace_strict(config["sex"]),
    #     occupation=pl.lit("unknown"),
    # )

    return data


def load_trips(
    root: str | Path,
    config: dict,
) -> pl.DataFrame:
    """Load and normalise trip records.

    TODO: Update file name and path to match raw data layout.
    TODO: Convert tst/tet to minutes since midnight (int).
    TODO: Convert distance to kilometres.
    """
    root = Path(root).expanduser()
    column_mapping = config["column_mappings"]

    # TODO: Replace "trips.csv" with the actual file name.
    data = pl.read_csv(root / "trips.csv", ignore_errors=True)
    data = data.select(column_mapping.keys()).rename(column_mapping)

    # TODO: Map mode, oact, dact, ozone, dzone to canonical values.
    # TODO: Compute tst/tet in minutes since midnight.
    # TODO: Convert distance to km (e.g. * 1.60934 for miles).

    # Handle midnight-crossing trips
    data = fix.day_wrap(data)

    return data
