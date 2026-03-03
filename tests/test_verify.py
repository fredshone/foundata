import polars as pl
import pytest

from foundata import verify


# --- check_dtype ---

def test_check_dtype_int_pass():
    assert verify.check_dtype("int", pl.Int32)

def test_check_dtype_int_fail():
    assert not verify.check_dtype("int", pl.String)

def test_check_dtype_float_pass():
    assert verify.check_dtype("float", pl.Float64)

def test_check_dtype_float_fail():
    assert not verify.check_dtype("float", pl.Int32)

def test_check_dtype_string_pass():
    assert verify.check_dtype("string", pl.String)
    assert verify.check_dtype("str", pl.String)

def test_check_dtype_string_fail():
    assert not verify.check_dtype("string", pl.Int32)

def test_check_dtype_numeric_pass():
    assert verify.check_dtype("numeric", pl.Int64)
    assert verify.check_dtype("numeric", pl.Float32)

def test_check_dtype_numeric_fail():
    assert not verify.check_dtype("numeric", pl.String)

def test_check_dtype_boolean_pass():
    dtype = pl.Series([True, False]).dtype
    assert verify.check_dtype("boolean", dtype)
    assert verify.check_dtype("bool", dtype)

def test_check_dtype_boolean_fail():
    dtype = pl.Series([1, 2], dtype=pl.Int32).dtype
    assert not verify.check_dtype("boolean", dtype)

def test_check_dtype_any_always_pass():
    assert verify.check_dtype("any", pl.String)
    assert verify.check_dtype("any", pl.Int32)

def test_check_dtype_unknown_raises():
    with pytest.raises(ValueError):
        verify.check_dtype("badtype", pl.String)


# --- check_min / check_max ---

def test_check_min_valid():
    s = pl.Series("age", [10, 20, 30])
    assert verify.check_min(0, s)

def test_check_min_invalid():
    s = pl.Series("age", [-1, 10, 20])
    assert not verify.check_min(0, s)

def test_check_max_valid():
    s = pl.Series("age", [10, 20, 30])
    assert verify.check_max(120, s)

def test_check_max_invalid():
    s = pl.Series("age", [10, 20, 200])
    assert not verify.check_max(120, s)


# --- check_set ---

def test_check_set_valid(capsys):
    s = pl.Series("mode", ["car", "bus", "walk"])
    result = verify.check_set({"car", "bus", "walk", "bike", "rail", "other", "unknown"}, s)
    assert result is True

def test_check_set_invalid(capsys):
    s = pl.Series("mode", ["Car", "Bus"])  # wrong case
    result = verify.check_set({"car", "bus"}, s)
    assert result is False

def test_check_set_extra_values(capsys):
    s = pl.Series("mode", ["car", "hovercraft"])
    result = verify.check_set({"car", "bus"}, s)
    assert result is False


# --- check_no_default ---

def test_check_no_default_numeric_pass():
    s = pl.Series("age", [10, 20, 30], dtype=pl.Int32)
    assert verify.check_no_default(s)

def test_check_no_default_numeric_fail():
    s = pl.Series("age", [10, None, 30], dtype=pl.Int32)
    assert not verify.check_no_default(s)

def test_check_no_default_string_pass():
    s = pl.Series("pid", ["a001", "a002"], dtype=pl.String)
    assert verify.check_no_default(s)

def test_check_no_default_string_fail():
    s = pl.Series("pid", ["a001", "unknown"], dtype=pl.String)
    assert not verify.check_no_default(s)


# --- verify.columns ---

def test_columns_pass(sample_attributes_df, sample_trips_df):
    result = verify.columns(sample_attributes_df, sample_trips_df)
    assert result is True


def test_columns_missing(sample_attributes_df, sample_trips_df):
    attrs = sample_attributes_df.drop("age")
    result = verify.columns(attrs, sample_trips_df)
    assert result is False


def test_columns_extra(sample_attributes_df, sample_trips_df, capsys):
    attrs = sample_attributes_df.with_columns(pl.lit("extra").alias("extra_col"))
    # extra columns don't cause failure, just a warning
    result = verify.columns(attrs, sample_trips_df)
    captured = capsys.readouterr()
    assert "extra_col" in captured.out


# --- verify.activity_consistency ---

def _make_trips(rows):
    return pl.DataFrame(rows, schema={"pid": pl.String, "seq": pl.Int32, "oact": pl.String, "dact": pl.String, "ozone": pl.String, "dzone": pl.String})


def test_activity_consistency_pass():
    trips = _make_trips([
        {"pid": "p1", "seq": 1, "oact": "home", "dact": "work", "ozone": "z1", "dzone": "z2"},
        {"pid": "p1", "seq": 2, "oact": "work", "dact": "home", "ozone": "z2", "dzone": "z1"},
    ])
    assert verify.activity_consistency(trips) is True


def test_activity_consistency_fail(capsys):
    trips = _make_trips([
        {"pid": "p1", "seq": 1, "oact": "home", "dact": "work", "ozone": "z1", "dzone": "z2"},
        {"pid": "p1", "seq": 2, "oact": "shop", "dact": "home", "ozone": "z2", "dzone": "z1"},
    ])
    result = verify.activity_consistency(trips)
    assert result is False
    assert "inconsistencies" in capsys.readouterr().out


def test_activity_consistency_skips_unknown():
    trips = _make_trips([
        {"pid": "p1", "seq": 1, "oact": "home", "dact": "unknown", "ozone": "z1", "dzone": "z2"},
        {"pid": "p1", "seq": 2, "oact": "shop", "dact": "home", "ozone": "z2", "dzone": "z1"},
    ])
    assert verify.activity_consistency(trips) is True


# --- verify.location_consistency ---

def test_location_consistency_pass():
    trips = _make_trips([
        {"pid": "p1", "seq": 1, "oact": "home", "dact": "work", "ozone": "z1", "dzone": "z2"},
        {"pid": "p1", "seq": 2, "oact": "work", "dact": "home", "ozone": "z2", "dzone": "z1"},
    ])
    assert verify.location_consistency(trips) is True


def test_location_consistency_fail(capsys):
    trips = _make_trips([
        {"pid": "p1", "seq": 1, "oact": "home", "dact": "work", "ozone": "z1", "dzone": "z2"},
        {"pid": "p1", "seq": 2, "oact": "work", "dact": "home", "ozone": "z9", "dzone": "z1"},
    ])
    result = verify.location_consistency(trips)
    assert result is False
    assert "inconsistencies" in capsys.readouterr().out


def test_location_consistency_skips_unknown():
    trips = _make_trips([
        {"pid": "p1", "seq": 1, "oact": "home", "dact": "work", "ozone": "z1", "dzone": "unknown"},
        {"pid": "p1", "seq": 2, "oact": "work", "dact": "home", "ozone": "z9", "dzone": "z1"},
    ])
    assert verify.location_consistency(trips) is True
