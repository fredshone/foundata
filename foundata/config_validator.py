from pathlib import Path

from foundata import utils

PROGRAMMATIC_FIELDS = {"country", "source", "year", "month", "day"}


def validate_column_mappings(config: dict, template_section: dict) -> list[str]:
    """Warn about column_mappings values that are not valid template field names.

    Returns a list of warning strings. Non-template target names may be
    legitimate intermediate fields used in processing, so these are warnings
    rather than hard errors.
    """
    warnings = []
    col_mappings = config.get("column_mappings") or {}

    # Collect all target field names across all year keys
    # Handle year-keyed configs (NHTS/NTS style with 'default' key)
    if isinstance(col_mappings, dict) and "default" in col_mappings:
        all_targets: list[tuple[str, str]] = []
        for year_key, year_mappings in col_mappings.items():
            if isinstance(year_mappings, dict):
                for raw_col, mapped_col in year_mappings.items():
                    all_targets.append((str(raw_col), mapped_col))
    else:
        all_targets = [
            (str(raw_col), mapped_col)
            for raw_col, mapped_col in col_mappings.items()
        ]

    valid_fields = set(template_section.keys())
    seen = set()
    for raw_col, mapped_col in all_targets:
        if not isinstance(mapped_col, str):
            continue
        if mapped_col in seen:
            continue
        seen.add(mapped_col)
        if mapped_col not in valid_fields:
            warnings.append(
                f"column_mappings: '{mapped_col}' is not a template field (from raw col '{raw_col}') — may be intermediate"
            )
    return warnings


def validate_value_mappings(config: dict, template_section: dict) -> list[str]:
    """Check that mapped string values are in the template's allowed set."""
    errors = []
    for key, mapping in config.items():
        if key == "column_mappings":
            continue

        # Skip if this field is not in the template
        if key not in template_section:
            continue

        field_config = template_section[key]
        if "set" not in field_config:
            continue

        allowed = set(field_config["set"])

        # Handle year-keyed configs
        if isinstance(mapping, dict) and "default" in mapping:
            all_values = {}
            for year_key, year_mapping in mapping.items():
                if isinstance(year_mapping, dict):
                    all_values.update(year_mapping)
        else:
            all_values = mapping

        if not isinstance(all_values, dict):
            continue

        for coded_val, mapped_val in all_values.items():
            # Skip non-string values (e.g. income bounds [0, 10000])
            if not isinstance(mapped_val, str):
                continue
            if mapped_val not in allowed:
                errors.append(
                    f"'{key}': value '{mapped_val}' (for code {coded_val!r}) not in allowed set {sorted(allowed)}"
                )
    return errors


def check_required_fields(hh_config: dict, person_config: dict) -> list[str]:
    """Warn about template attribute fields missing from both configs combined."""
    template_attributes = utils.get_template_attributes()
    warnings = []

    def get_mapped_fields(config: dict) -> set[str]:
        fields = set()
        col_mappings = config.get("column_mappings") or {}
        if isinstance(col_mappings, dict) and "default" in col_mappings:
            for year_key, year_mappings in col_mappings.items():
                if isinstance(year_mappings, dict):
                    fields.update(year_mappings.values())
        else:
            fields.update(col_mappings.values())
        return fields

    hh_fields = get_mapped_fields(hh_config)
    person_fields = get_mapped_fields(person_config)
    all_mapped = hh_fields | person_fields

    for field in template_attributes:
        if field not in all_mapped and field not in PROGRAMMATIC_FIELDS:
            warnings.append(
                f"Template attribute '{field}' not found in hh or person column_mappings (may be programmatically set)"
            )
    return warnings


def validate_source(
    source_name: str, configs_root: str | Path | None = None
) -> bool:
    """Load all YAML dicts for a source and run all checks. Returns True if valid."""
    if configs_root is None:
        configs_root = utils.get_config_path()

    configs_root = Path(configs_root)
    source_dir = configs_root / source_name

    if not source_dir.exists():
        print(f"ERROR: Config directory not found: {source_dir}")
        return False

    template_attributes = utils.get_template_attributes()
    template_trips = utils.get_template_trips()

    errors = []
    warnings = []

    # Load available configs
    hh_config = None
    person_config = None
    trip_config = None

    hh_path = source_dir / "hh_dictionary.yaml"
    person_path = source_dir / "person_dictionary.yaml"
    trip_path = source_dir / "trip_dictionary.yaml"

    if hh_path.exists():
        hh_config = utils.load_yaml_config(hh_path)
    else:
        warnings.append(f"Missing hh_dictionary.yaml in {source_dir}")

    if person_path.exists():
        person_config = utils.load_yaml_config(person_path)
    else:
        warnings.append(f"Missing person_dictionary.yaml in {source_dir}")

    if trip_path.exists():
        trip_config = utils.load_yaml_config(trip_path)
    else:
        warnings.append(f"Missing trip_dictionary.yaml in {source_dir}")

    # Validate column mappings (unknown targets are warnings — may be intermediate fields)
    if hh_config:
        warns = validate_column_mappings(hh_config, template_attributes)
        warnings.extend([f"[hh] {w}" for w in warns])

    if person_config:
        warns = validate_column_mappings(person_config, template_attributes)
        warnings.extend([f"[person] {w}" for w in warns])

    if trip_config:
        warns = validate_column_mappings(trip_config, template_trips)
        warnings.extend([f"[trips] {w}" for w in warns])

    # Validate value mappings
    if hh_config:
        errs = validate_value_mappings(hh_config, template_attributes)
        errors.extend([f"[hh] {e}" for e in errs])

    if person_config:
        errs = validate_value_mappings(person_config, template_attributes)
        errors.extend([f"[person] {e}" for e in errs])

    if trip_config:
        errs = validate_value_mappings(trip_config, template_trips)
        errors.extend([f"[trips] {e}" for e in errs])

    # Check required fields coverage
    if hh_config and person_config:
        field_warnings = check_required_fields(hh_config, person_config)
        warnings.extend(field_warnings)

    # Print summary
    print(f"\n=== Validating source: {source_name} ===")
    if warnings:
        for w in warnings:
            print(f"  WARN: {w}")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        print(
            f"  RESULT: INVALID ({len(errors)} errors, {len(warnings)} warnings)"
        )
    else:
        print(f"  RESULT: OK ({len(warnings)} warnings)")

    return len(errors) == 0


def validate_all_sources(configs_root: str | Path | None = None) -> bool:
    """Run validate_source on all existing sources. Returns True if all valid."""
    if configs_root is None:
        configs_root = utils.get_config_path()

    configs_root = Path(configs_root)
    sources = [
        d.name
        for d in configs_root.iterdir()
        if d.is_dir() and d.name != "core"
    ]

    if not sources:
        print("No sources found.")
        return True

    all_valid = True
    for source in sorted(sources):
        valid = validate_source(source, configs_root)
        if not valid:
            all_valid = False

    print(f"\n=== Summary: {'ALL VALID' if all_valid else 'SOME INVALID'} ===")
    return all_valid
