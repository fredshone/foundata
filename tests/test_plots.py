import matplotlib

matplotlib.use("Agg")

import polars as pl
import pytest

from foundata import plots, post_process

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


@pytest.fixture
def attrs_conditionality():
    return pl.DataFrame(
        {
            "pid": ["p1", "p2", "p3", "p4", "p5", "p6"],
            "source": ["a", "a", "a", "a", "b", "b"],
            "employment": [
                "employed",
                "student",
                "retired",
                "employed",
                "unemployed",
                "ft-employed",
            ],
            "age": [35, 19, 70, 40, 22, 45],
            "hh_zone": ["urban"] * 6,
        }
    )


@pytest.fixture
def attrs_full(attrs_conditionality):
    return attrs_conditionality.with_columns(
        pl.Series("hh_income", [20000, 45000, 15000, 60000, 30000, 80000])
    )


@pytest.fixture
def trips_conditionality():
    return pl.DataFrame(
        {
            "pid": ["p1", "p2", "p3", "p4", "p5", "p6"],
            "seq": [0] * 6,
            "oact": ["home"] * 6,
            "dact": ["work", "education", "home", "work", "home", "work"],
            "ozone": ["urban"] * 6,
            "dzone": ["urban"] * 6,
            "mode": ["car"] * 6,
            "tst": [480] * 6,
            "tet": [510] * 6,
            "distance": [5.0] * 6,
        },
        schema=TRIPS_SCHEMA,
    )


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
    plots.time_of_day_profile(attrs, trips, save_path=out)
    _assert_saved(out)


def test_plot_time_heaping(attrs, trips, tmp_path):
    out = tmp_path / "heaping.png"
    plots.time_heaping(attrs, trips, save_path=out)
    _assert_saved(out)


def test_plot_trip_time_diagnostics(attrs, trips, tmp_path):
    out = tmp_path / "trip_time_diagnostics.png"
    plots.trip_time_diagnostics(attrs, trips, save_path=out)
    _assert_saved(out)


def test_plot_activity_duration_by_type(attrs, trips, tmp_path):
    out = tmp_path / "activity_duration.png"
    activities = post_process.trips_to_activities(attrs, trips)
    plots.activity_duration_by_type(attrs, activities, save_path=out)
    _assert_saved(out)


def test_plot_categorical_bar_grid(attrs_employment, tmp_path):
    out = tmp_path / "categorical_bar_grid.png"
    plots.categorical_bar_grid(attrs_employment, save_path=out)
    _assert_saved(out)


# ---------------------------------------------------------------------------
# _stratified_sample
# ---------------------------------------------------------------------------


def test_stratified_sample_keeps_small_groups_intact():
    # source "a" is large, source "b" is small and would be swamped by a
    # single global sample over the pooled data.
    df = pl.DataFrame(
        {
            "pid": [f"p{i}" for i in range(3000)],
            "source": ["a"] * 2900 + ["b"] * 100,
            "employment": ["employed"] * 2900
            + ["void"] * 2
            + ["employed"] * 98,
        }
    )
    sampled = plots._stratified_sample(df, "source", max_per_group=1000)

    assert sampled.filter(pl.col("source") == "a").height == 1000
    # "b" is under the cap, so it's kept whole — including the rare "void"
    # rows that a global sample could easily have dropped.
    b = sampled.filter(pl.col("source") == "b")
    assert b.height == 100
    assert (b["employment"] == "void").sum() == 2


# ---------------------------------------------------------------------------
# attribute x activity diagnostics
# ---------------------------------------------------------------------------


def test_plot_activity_count_by_attribute(attrs_employment, trips, tmp_path):
    out = tmp_path / "activity_count_by_attribute.png"
    activities = post_process.trips_to_activities(attrs_employment, trips)
    plots.activity_count_by_attribute(
        attrs_employment,
        activities,
        attribute_col="employment",
        act_types=["work", "education"],
        save_path=out,
    )
    _assert_saved(out)


def test_plot_attribute_activity_heatmap(attrs_employment, trips, tmp_path):
    out = tmp_path / "attribute_activity_heatmap.png"
    activities = post_process.trips_to_activities(attrs_employment, trips)
    plots.attribute_activity_heatmap(
        attrs_employment, activities, attribute_col="employment", save_path=out
    )
    _assert_saved(out)


def test_plot_activity_count_by_attribute_age_band(
    attrs_conditionality, trips_conditionality, tmp_path
):
    out = tmp_path / "activity_count_by_age.png"
    banded = post_process.add_age_band(attrs_conditionality)
    activities = post_process.trips_to_activities(banded, trips_conditionality)
    plots.activity_count_by_attribute(
        banded,
        activities,
        attribute_col="age_band",
        act_types=["work", "education"],
        save_path=out,
    )
    _assert_saved(out)


def test_plot_attribute_activity_heatmap_binned(
    attrs_full, trips_conditionality, tmp_path
):
    out = tmp_path / "attribute_activity_heatmap_binned.png"
    activities = post_process.trips_to_activities(
        attrs_full, trips_conditionality
    )
    plots.attribute_activity_heatmap(
        attrs_full,
        activities,
        attribute_col="hh_income",
        n_bins=3,
        save_path=out,
    )
    _assert_saved(out)


def test_plot_activity_count_by_attribute_name_collides_with_act_type(
    attrs_conditionality, trips_conditionality, tmp_path
):
    # "education" is both an act_type and, here, the attribute being
    # faceted on — activity_counts_per_person's "education" count column
    # must not shadow the attribute column after the join (see
    # tests/test_anomaly.py for the same collision on conditionality_matrix).
    attrs = attrs_conditionality.with_columns(
        education=pl.Series(
            ["degree", "none", "degree", "none", "degree", "none"]
        )
    )
    activities = post_process.trips_to_activities(attrs, trips_conditionality)

    out = tmp_path / "activity_count_by_attribute_education.png"
    plots.activity_count_by_attribute(
        attrs,
        activities,
        attribute_col="education",
        act_types=["work", "education"],
        save_path=out,
    )
    _assert_saved(out)

    out2 = tmp_path / "attribute_activity_heatmap_education.png"
    plots.attribute_activity_heatmap(
        attrs, activities, attribute_col="education", save_path=out2
    )
    _assert_saved(out2)

    out3 = tmp_path / "activities_attributes_grid_education.png"
    plots.activities_attributes_grid(
        attrs,
        activities,
        attribute_cols={"education": "bar"},
        save_path=out3,
    )
    _assert_saved(out3)


def test_plot_activity_counts_grid(attrs_full, trips_conditionality, tmp_path):
    out = tmp_path / "activity_counts_grid.png"
    activities = post_process.trips_to_activities(
        attrs_full, trips_conditionality
    )
    plots.activities_attributes_grid(
        attrs_full,
        activities,
        attribute_cols={
            "employment": "bar",
            "hh_income": "line",
            "age": "line",
        },
        save_path=out,
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
        plots.time_of_day_profile(
            empty_attrs, trips, save_path=tmp_path / "x.png"
        )
