import polars as pl

from foundata import utils


def null_pids(
    attributes: pl.DataFrame, trips: pl.DataFrame, on: str = "pid"
) -> bool:
    n_attr = attributes[on].null_count()
    n_trips = trips[on].null_count()
    if n_attr:
        print(f"ERROR: {n_attr} null PIDs in attributes")
    if n_trips:
        print(f"ERROR: {n_trips} null PIDs in trips")
    return n_attr == 0 and n_trips == 0


def columns(
    attributes: pl.DataFrame,
    trips: pl.DataFrame,
    template_attributes: dict | None = None,
    template_trips: dict | None = None,
) -> bool:
    if template_attributes is None:
        template_attributes = utils.get_template_attributes()
    if template_trips is None:
        template_trips = utils.get_template_trips()

    expected_attributes = template_attributes.keys()
    expected_trips = template_trips.keys()

    # ckeck for missing columns
    actual_attributes = set(attributes.columns)
    actual_trips = set(trips.columns)

    missing_attributes = expected_attributes - actual_attributes
    missing_trips = expected_trips - actual_trips

    if missing_attributes:
        print(f"Missing columns in attributes: {missing_attributes}")
    if missing_trips:
        print(f"Missing columns in trips: {missing_trips}")

    # check for columns types
    attributes_good = check_col_cnfg(attributes, template_attributes)
    trips_good = check_col_cnfg(trips, template_trips)

    # check for extra columns
    extra_attributes = actual_attributes - expected_attributes
    extra_trips = actual_trips - expected_trips

    if extra_attributes:
        print(f"Extra columns in attributes: {extra_attributes}")
    if extra_trips:
        print(f"Extra columns in trips: {extra_trips}")

    return (
        not missing_attributes
        and not missing_trips
        and attributes_good
        and trips_good
    )


def check_dtype(expected_dtype: str, actual_dtype: pl.DataType) -> bool:
    exact = {
        "int8": pl.Int8,
        "int16": pl.Int16,
        "int32": pl.Int32,
        "int64": pl.Int64,
        "float32": pl.Float32,
        "float64": pl.Float64,
    }
    if expected_dtype in exact:
        return actual_dtype == exact[expected_dtype]
    if expected_dtype in ["integer", "int"]:
        return actual_dtype.is_integer()
    elif expected_dtype == "float":
        return actual_dtype.is_float()
    elif expected_dtype in ["string", "str"]:
        return actual_dtype == pl.String
    elif expected_dtype in ["boolean", "bool"]:
        return actual_dtype == pl.Boolean
    elif expected_dtype == "numeric":
        return actual_dtype.is_numeric()
    elif expected_dtype == "any":
        return True
    elif expected_dtype == "date":
        return actual_dtype == pl.Date
    elif expected_dtype == "datetime":
        return actual_dtype == pl.Datetime
    else:
        raise ValueError(f"Unknown expected dtype: {expected_dtype}")


def check_no_default(actual: pl.Series) -> bool:
    if actual.dtype.is_numeric():
        return actual.null_count() == 0
    elif actual.dtype == pl.String:
        return "unknown" not in set(actual.unique())
    else:
        raise ValueError(
            f"Unsupported dtype for col: {actual.name}, default value check: {actual.dtype}"
        )


def check_min(expected_min: float, actual_series: pl.Series) -> bool:
    if actual_series.dtype.is_numeric():
        return actual_series.fill_null(expected_min).min() >= expected_min
    else:
        raise ValueError(
            f"Expected numeric series '{actual_series.name}', got {actual_series.dtype}"
        )


def check_max(expected_max: float, actual_series: pl.Series) -> bool:
    if actual_series.dtype.is_numeric():
        return actual_series.fill_null(expected_max).max() <= expected_max
    else:
        raise ValueError(
            f"Expected numeric series '{actual_series.name}', got {actual_series.dtype}"
        )


def check_set(expected_set: set, actual_series: pl.Series) -> bool:
    actual_set = set(actual_series.unique())
    missing = expected_set - actual_set
    extra = actual_set - expected_set
    if missing:
        print(f"Warning: Missing values in '{actual_series.name}': {missing}")
    if extra:
        print(f"Unexpected values in '{actual_series.name}': {extra}")
        return False
    return True


def activity_consistency(trips: pl.DataFrame) -> bool:
    inconsistent = (
        trips.sort("pid", "seq")
        .with_columns(next_oact=pl.col("oact").shift(-1).over("pid"))
        .filter(pl.col("next_oact").is_not_null())
        .filter(pl.col("dact") != "unknown")
        .filter(pl.col("next_oact") != "unknown")
        .filter(pl.col("dact") != pl.col("next_oact"))
        .select("pid")
        .unique()
    )
    n = len(inconsistent)
    total = trips.select("pid").n_unique()
    if n:
        print(
            f"Warning: {n}/{total} persons have activity chain inconsistencies ({100*n/total:.1f}%)"
        )
        return False
    return True


def location_consistency(trips: pl.DataFrame) -> bool:
    inconsistent = (
        trips.sort("pid", "seq")
        .with_columns(next_ozone=pl.col("ozone").shift(-1).over("pid"))
        .filter(pl.col("next_ozone").is_not_null())
        .filter(pl.col("dzone") != "unknown")
        .filter(pl.col("next_ozone") != "unknown")
        .filter(pl.col("dzone") != pl.col("next_ozone"))
        .select("pid")
        .unique()
    )
    n = len(inconsistent)
    total = trips.select("pid").n_unique()
    if n:
        print(
            f"Warning: {n}/{total} persons have location chain inconsistencies ({100*n/total:.1f}%)"
        )
        return False
    return True


def check_col_cnfg(
    actual: pl.DataFrame, template: dict, auto_cast: bool = True
) -> None:
    actual_cols = set(actual.columns)
    template_cols = set(template.keys())

    fails = 0
    for col in actual_cols & template_cols:
        cnfg = template[col]

        expected_dtype = cnfg["dtype"]
        actual_dtype = actual[col].dtype
        good_dtype = check_dtype(expected_dtype, actual_dtype)
        if auto_cast and not good_dtype:
            print(
                f"Auto-casting column '{col}' from {actual_dtype} to expected {expected_dtype}"
            )
            polars_type = utils.DTYPE_MAP.get(expected_dtype)
            if polars_type is not None:
                actual = actual.with_columns(
                    pl.col(col).cast(polars_type, strict=True)
                )
                actual_dtype = actual[col].dtype
                good_dtype = check_dtype(expected_dtype, actual_dtype)

        elif not good_dtype:
            print(
                f"ERROR: Column '{col}' has dtype {actual_dtype} but expected {expected_dtype}"
            )
            fails += 1

        if "default" not in cnfg or not cnfg["default"]:
            good_default = check_no_default(actual[col])
            if not good_default:
                print(
                    f"ERROR: Column '{col}' appears to be using default values but expected none."
                )
                fails += 1

        if "min" in cnfg:
            expected_min = cnfg["min"]
            good_min = check_min(expected_min, actual[col])
            if not good_min:
                print(
                    f"ERROR: Column '{col}' has min {actual[col].min()} but expected at least {expected_min}"
                )
                fails += 1

        if "max" in cnfg:
            expected_max = cnfg["max"]
            good_max = check_max(expected_max, actual[col])
            if not good_max:
                print(
                    f"ERROR: Column '{col}' has max {actual[col].max()} but expected at most {expected_max}"
                )
                fails += 1

        if "set" in cnfg:
            expected_set = set(cnfg["set"])
            good_set = check_set(expected_set, actual[col])
            if not good_set:
                print(
                    f"ERROR: Column '{col}' expected set {expected_set} but found ^."
                )
                fails += 1

    return fails == 0
