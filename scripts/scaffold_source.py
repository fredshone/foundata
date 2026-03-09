#!/usr/bin/env python3
"""Generate boilerplate for a new foundata source.

Usage:
    python scripts/scaffold_source.py <source_name>

Creates:
    configs/<source>/hh_dictionary.yaml
    configs/<source>/person_dictionary.yaml
    configs/<source>/trip_dictionary.yaml
    foundata/<source>.py
"""

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_ROOT = REPO_ROOT / "configs"
FOUNDATA_ROOT = REPO_ROOT / "foundata"
TEMPLATE_PATH = CONFIGS_ROOT / "core" / "template.yaml"


def load_template() -> dict:
    with open(TEMPLATE_PATH) as f:
        return yaml.safe_load(f)


def format_set_comment(values: list) -> str:
    return "  # Valid values: " + ", ".join(str(v) for v in values)


def build_hh_yaml(template_attributes: dict) -> str:
    lines = [
        "# Household-level dictionary for <source>",
        "# Map raw CSV column names to template field names under column_mappings.",
        "# Add value mappings for categorical fields below.",
        "",
        "column_mappings:",
    ]

    # List all hh-relevant attribute fields as commented stubs
    hh_fields = [
        "hid", "hh_size", "hh_income", "dwelling", "ownership",
        "vehicles", "hh_zone", "weight",
    ]
    for field in hh_fields:
        if field in template_attributes:
            lines.append(f"  # RAW_COL_NAME: {field}")

    lines.append("")

    # Add categorical value mapping stubs
    for field in hh_fields:
        if field not in template_attributes:
            continue
        cfg = template_attributes[field]
        if "set" not in cfg:
            continue
        lines.append(f"# {field} mapping")
        lines.append(format_set_comment(cfg["set"]))
        lines.append(f"{field}:")
        for val in cfg["set"]:
            lines.append(f"  # 1: {val}")
        lines.append("")

    return "\n".join(lines)


def build_person_yaml(template_attributes: dict) -> str:
    lines = [
        "# Person-level dictionary for <source>",
        "# Map raw CSV column names to template field names under column_mappings.",
        "# Add value mappings for categorical fields below.",
        "",
        "column_mappings:",
    ]

    person_fields = [
        "pid", "hid", "age", "sex", "employment", "occupation",
        "education", "has_licence", "disability", "can_wfh",
        "relationship", "race",
    ]
    for field in person_fields:
        if field in template_attributes:
            lines.append(f"  # RAW_COL_NAME: {field}")

    lines.append("")

    for field in person_fields:
        if field not in template_attributes:
            continue
        cfg = template_attributes[field]
        if "set" not in cfg:
            continue
        lines.append(f"# {field} mapping")
        lines.append(format_set_comment(cfg["set"]))
        lines.append(f"{field}:")
        for val in cfg["set"]:
            lines.append(f"  # 1: {val}")
        lines.append("")

    return "\n".join(lines)


def build_trip_yaml(template_trips: dict) -> str:
    lines = [
        "# Trip-level dictionary for <source>",
        "# Map raw CSV column names to template field names under column_mappings.",
        "# Add value mappings for categorical fields below.",
        "",
        "column_mappings:",
    ]

    trip_fields = list(template_trips.keys())
    for field in trip_fields:
        lines.append(f"  # RAW_COL_NAME: {field}")

    lines.append("")

    for field in trip_fields:
        cfg = template_trips[field]
        if "set" not in cfg:
            continue
        lines.append(f"# {field} mapping")
        lines.append(format_set_comment(cfg["set"]))
        lines.append(f"{field}:")
        for val in cfg["set"]:
            lines.append(f"  # 1: {val}")
        lines.append("")

    return "\n".join(lines)


def build_python_module(source_name: str) -> str:
    return f'''from pathlib import Path

import polars as pl

from foundata import fix, utils
from foundata.utils import table_joiner

SOURCE = "{source_name}"


def load(
    data_root: str | Path,
    hh_config: dict,
    person_config: dict,
    trips_config: dict,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load and normalise {source_name} survey data.

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
    #     hh_zone=pl.col("hh_zone").replace_strict(config["hh_zone"]),
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
'''


def scaffold_source(source_name: str) -> None:
    template = load_template()
    template_attributes = template["attributes"]
    template_trips = template["trips"]

    # Create config directory
    config_dir = CONFIGS_ROOT / source_name
    if config_dir.exists():
        print(f"ERROR: Config directory already exists: {config_dir}")
        sys.exit(1)
    config_dir.mkdir(parents=True)

    # Write YAML configs
    hh_yaml = build_hh_yaml(template_attributes)
    person_yaml = build_person_yaml(template_attributes)
    trip_yaml = build_trip_yaml(template_trips)

    (config_dir / "hh_dictionary.yaml").write_text(hh_yaml)
    (config_dir / "person_dictionary.yaml").write_text(person_yaml)
    (config_dir / "trip_dictionary.yaml").write_text(trip_yaml)

    # Write Python module
    py_path = FOUNDATA_ROOT / f"{source_name}.py"
    if py_path.exists():
        print(f"ERROR: Python module already exists: {py_path}")
        sys.exit(1)
    py_path.write_text(build_python_module(source_name))

    print(f"Scaffolded new source: {source_name!r}")
    print(f"  {config_dir}/hh_dictionary.yaml")
    print(f"  {config_dir}/person_dictionary.yaml")
    print(f"  {config_dir}/trip_dictionary.yaml")
    print(f"  {py_path}")
    print()
    print("Next steps:")
    print("  1. Fill in column_mappings in each YAML (raw col -> template field)")
    print("  2. Add value mappings for categorical fields")
    print("  3. Implement load_households/load_persons/load_trips in the .py module")
    print(f"  4. Run: python -c \"from foundata.config_validator import validate_source; validate_source('{source_name}')\"")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    scaffold_source(sys.argv[1])
