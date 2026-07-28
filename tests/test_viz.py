import matplotlib

matplotlib.use("Agg")

import polars as pl
import pytest

from foundata import viz

TRIPS_SCHEMA = {
    "pid": pl.String,
    "seq": pl.Int32,
    "oact": pl.String,
    "dact": pl.String,
    "ozone": pl.String,
    "dzone": pl.String,
    "mode": pl.String,
    "tst": pl.Int32,
    "tet": pl.Int32,
    "distance": pl.Float32,
}


@pytest.fixture
def attrs():
    return pl.DataFrame(
        {
            "pid": ["p1", "p2", "p3", "p4"],
            "source": ["a", "a", "b", "b"],
            "hh_zone": ["urban"] * 4,
        }
    )


@pytest.fixture
def trips():
    return pl.DataFrame(
        {
            "pid": ["p1", "p2", "p3", "p4"],
            "seq": [0, 0, 0, 0],
            "oact": ["home", "home", "home", "home"],
            "dact": ["education", "work", "work", "work"],
            "ozone": ["urban"] * 4,
            "dzone": ["urban"] * 4,
            "mode": ["car", "car", "car", "car"],
            "tst": [480, 1500, 490, 600],  # p2 day-wraps, p3 has tst > tet
            "tet": [510, 1530, 480, 605],
            "distance": [5.0, 2.0, 1.0, 50.0],  # p4 implies ~600 km/h
        },
        schema=TRIPS_SCHEMA,
    )


@pytest.fixture
def attrs_employment(attrs):
    return attrs.with_columns(
        pl.Series(
            "employment", ["employed", "student", "unemployed", "employed"]
        )
    )


# ---------------------------------------------------------------------------
# time_quality_summary_table
# ---------------------------------------------------------------------------


def test_time_quality_summary_table_values(attrs, trips):
    table = viz.time_quality_summary_table(attrs, trips).sort("source")

    a = table.filter(pl.col("source") == "a").row(0, named=True)
    b = table.filter(pl.col("source") == "b").row(0, named=True)

    assert a["n_trips"] == 2
    assert a["non_positive_duration_pct"] == pytest.approx(0.0)
    assert a["day_wrap_pct"] == pytest.approx(50.0)
    assert a["median_duration_min"] == pytest.approx(30.0)
    assert a["implausible_speed_pct"] == pytest.approx(0.0)
    assert a["median_speed_kmh"] == pytest.approx(7.0)

    assert b["n_trips"] == 2
    assert b["non_positive_duration_pct"] == pytest.approx(50.0)
    assert b["day_wrap_pct"] == pytest.approx(0.0)
    assert b["median_duration_min"] == pytest.approx(5.0)
    assert b["implausible_speed_pct"] == pytest.approx(100.0)
    assert b["median_speed_kmh"] == pytest.approx(600.0)


def test_time_quality_summary_table_markdown(attrs, trips):
    md = viz.time_quality_summary_table(attrs, trips, markdown=True)
    assert isinstance(md, str)
    assert md.startswith("| source |")
    assert "a" in md and "b" in md


# ---------------------------------------------------------------------------
# plot functions — smoke tests: they should run end-to-end and save a
# non-empty file without raising, for both normal and pathological
# (day-wrap / negative-duration / high-implied-speed) input.
# ---------------------------------------------------------------------------


def _assert_saved(path):
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_time_of_day_profile(attrs, trips, tmp_path):
    out = tmp_path / "time_of_day.png"
    viz.plot_time_of_day_profile(attrs, trips, save_path=out)
    _assert_saved(out)


def test_plot_time_heaping(attrs, trips, tmp_path):
    out = tmp_path / "heaping.png"
    viz.plot_time_heaping(attrs, trips, save_path=out)
    _assert_saved(out)


def test_plot_trip_time_diagnostics(attrs, trips, tmp_path):
    out = tmp_path / "trip_time_diagnostics.png"
    viz.plot_trip_time_diagnostics(attrs, trips, save_path=out)
    _assert_saved(out)


def test_plot_activity_duration_by_type(attrs, trips, tmp_path):
    out = tmp_path / "activity_duration.png"
    viz.plot_activity_duration_by_type(attrs, trips, save_path=out)
    _assert_saved(out)


# ---------------------------------------------------------------------------
# attribute x activity diagnostics
# ---------------------------------------------------------------------------


def test_activity_counts_per_person(attrs_employment, trips):
    counts = viz._activity_counts_per_person(
        attrs_employment, trips, ["work", "education"]
    ).sort("pid")
    assert counts["pid"].to_list() == ["p1", "p2", "p3", "p4"]
    # p2's trip day-wraps (tet=1530 > 1440), so trips_to_activities drops its
    # destination activity — p2 contributes 0 to every activity type here.
    assert counts["work"].to_list() == [0, 0, 1, 1]
    assert counts["education"].to_list() == [1, 0, 0, 0]


def test_plot_activity_count_by_attribute(attrs_employment, trips, tmp_path):
    out = tmp_path / "activity_count_by_attribute.png"
    viz.plot_activity_count_by_attribute(
        attrs_employment,
        trips,
        attribute_col="employment",
        act_types=["work", "education"],
        save_path=out,
    )
    _assert_saved(out)


def test_plot_attribute_activity_heatmap(attrs_employment, trips, tmp_path):
    out = tmp_path / "attribute_activity_heatmap.png"
    viz.plot_attribute_activity_heatmap(
        attrs_employment, trips, attribute_col="employment", save_path=out
    )
    _assert_saved(out)


def test_plot_functions_raise_on_no_groups(trips, tmp_path):
    empty_attrs = pl.DataFrame(
        {
            "pid": pl.Series([], dtype=pl.String),
            "source": pl.Series([], dtype=pl.String),
            "hh_zone": pl.Series([], dtype=pl.String),
        }
    )
    with pytest.raises(ValueError):
        viz.plot_time_of_day_profile(
            empty_attrs, trips, save_path=tmp_path / "x.png"
        )
