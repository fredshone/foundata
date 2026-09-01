import math

import polars as pl
import pytest

from foundata import anomaly, post_process

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


@pytest.fixture
def activities_conditionality(attrs_conditionality, trips_conditionality):
    return post_process.trips_to_activities(
        attrs_conditionality, trips_conditionality
    )


# ---------------------------------------------------------------------------
# conditionality_matrix
# ---------------------------------------------------------------------------


def test_conditionality_matrix_values(
    attrs_conditionality, activities_conditionality
):
    matrix = anomaly.conditionality_matrix(
        attrs_conditionality,
        activities_conditionality,
        attribute_cols=["employment", "age"],
        act_types=["work", "education"],
        min_group_n=1,
    )

    # source "a": p1 (35, employed, work), p2 (19, student, education),
    # p3 (70, retired, home), p4 (40, employed, work) — both employment and
    # age perfectly predict both work and education here (n=4, V=1.0).
    a = matrix.filter(pl.col("source") == "a").sort("attribute", "act_type")
    assert a["n"].to_list() == [4, 4, 4, 4]
    assert a["cramers_v"].to_list() == pytest.approx([1.0, 1.0, 1.0, 1.0])

    # source "b": p5 (unemployed, home), p6 (ft-employed, work) — neither
    # has an education activity, so has_education has no variation and
    # Cramér's V against it is NaN (undefined, not zero).
    b_educ = matrix.filter(
        (pl.col("source") == "b") & (pl.col("act_type") == "education")
    )
    assert all(math.isnan(v) for v in b_educ["cramers_v"].to_list())


def test_conditionality_matrix_skips_missing_attribute_cols(
    attrs_conditionality, activities_conditionality
):
    # "hh_income" isn't in the fixture — should be silently skipped rather
    # than raising, since `attribute_cols` is filtered against the actual
    # attributes columns present.
    matrix = anomaly.conditionality_matrix(
        attrs_conditionality,
        activities_conditionality,
        attribute_cols=["employment", "hh_income"],
        act_types=["work"],
        min_group_n=1,
    )
    assert set(matrix["attribute"].unique().to_list()) == {"employment"}


def test_conditionality_matrix_respects_min_group_n(
    attrs_conditionality, activities_conditionality
):
    # source "b" only has 2 persons; with a min_group_n above that, its
    # rows should be dropped entirely while source "a" (4 persons) survives.
    matrix = anomaly.conditionality_matrix(
        attrs_conditionality,
        activities_conditionality,
        attribute_cols=["employment"],
        act_types=["work"],
        min_group_n=3,
    )
    assert matrix["source"].to_list() == ["a"]


def test_conditionality_matrix_attribute_name_collides_with_act_type(
    attrs_conditionality, activities_conditionality
):
    # "education" is both an act_type below and, here, a person attribute
    # name. `activity_counts_per_person` returns a count column literally
    # named "education", and a naive join against the attribute of the same
    # name lets the count column silently win — the attribute's own values
    # (all null for source "a") must not be shadowed by that count.
    attrs = attrs_conditionality.with_columns(
        education=pl.Series([None, None, None, None, "degree", "none"])
    )
    matrix = anomaly.conditionality_matrix(
        attrs,
        activities_conditionality,
        attribute_cols=["education"],
        act_types=["work", "education"],
        min_group_n=1,
    )

    # Source "a" has a fully-null "education" attribute (n=0 non-null),
    # so it must be skipped entirely — if the count column shadowed it,
    # rows would wrongly appear since the count is never null.
    assert matrix.filter(pl.col("source") == "a").is_empty()

    # Source "b" has real "education" attribute values and should compute
    # against them normally.
    b = matrix.filter(pl.col("source") == "b")
    assert set(b["act_type"].to_list()) == {"work", "education"}
    assert b["n"].to_list() == [2, 2]


def test_conditionality_matrix_composite_on(
    attrs_conditionality, activities_conditionality
):
    # Grouping by ["source", "year"] instead of just "source" should key
    # rows by both columns, catching a per-year anomaly a source-level
    # grouping would fold into the source's aggregate.
    attrs = attrs_conditionality.with_columns(
        year=pl.Series([2019, 2019, 2020, 2020, 2019, 2020])
    )
    matrix = anomaly.conditionality_matrix(
        attrs,
        activities_conditionality,
        on=["source", "year"],
        attribute_cols=["employment"],
        act_types=["work"],
        min_group_n=1,
    )
    assert set(matrix.columns) >= {"source", "year", "attribute", "act_type"}
    assert sorted(matrix.select("source", "year").unique().rows()) == [
        ("a", 2019),
        ("a", 2020),
        ("b", 2019),
        ("b", 2020),
    ]


# ---------------------------------------------------------------------------
# flag_conditionality_outliers
# ---------------------------------------------------------------------------


def test_flag_conditionality_outliers_composite_on():
    matrix = pl.DataFrame(
        {
            "source": ["a", "a", "b", "b"],
            "year": [2019, 2020, 2019, 2020],
            "attribute": ["employment"] * 4,
            "act_type": ["work"] * 4,
            "n": [100] * 4,
            "cramers_v": [0.7, 0.72, 0.68, 0.05],
        }
    )
    flagged = anomaly.flag_conditionality_outliers(
        matrix, on=["source", "year"], z_threshold=1.5, min_peers=3
    )
    assert flagged.height == 1
    row = flagged.row(0, named=True)
    assert (row["source"], row["year"]) == ("b", 2020)


def test_flag_conditionality_outliers_markdown_empty_when_nothing_flagged():
    matrix = pl.DataFrame(
        {
            "source": ["a", "b", "c"],
            "attribute": ["employment"] * 3,
            "act_type": ["work"] * 3,
            "n": [100] * 3,
            "cramers_v": [0.7, 0.71, 0.69],
        }
    )
    md = anomaly.flag_conditionality_outliers(
        matrix, z_threshold=1.5, markdown=True
    )
    assert md == "No anomalies flagged."


def test_flag_conditionality_outliers():
    # "d" is a clear outlier against three close peers for the same
    # (attribute, act_type) pair.
    matrix = pl.DataFrame(
        {
            "source": ["a", "b", "c", "d"],
            "attribute": ["employment"] * 4,
            "act_type": ["work"] * 4,
            "n": [100] * 4,
            "cramers_v": [0.7, 0.72, 0.68, 0.05],
        }
    )
    flagged = anomaly.flag_conditionality_outliers(
        matrix, z_threshold=1.5, min_peers=3
    )
    assert flagged.height == 1
    row = flagged.row(0, named=True)
    assert row["source"] == "d"
    assert row["z"] < -1.5


def test_flag_conditionality_outliers_skips_pairs_with_too_few_peers():
    matrix = pl.DataFrame(
        {
            "source": ["a", "b"],
            "attribute": ["employment"] * 2,
            "act_type": ["work"] * 2,
            "n": [100, 100],
            "cramers_v": [0.7, 0.05],
        }
    )
    flagged = anomaly.flag_conditionality_outliers(
        matrix, z_threshold=1.0, min_peers=3
    )
    assert flagged.height == 0


def test_flag_conditionality_outliers_sorted_by_abs_z_and_capped_to_top_n():
    # Each (attribute, act_type) pair has its own set of 4 peers with one
    # outlier; "weak" (low V) and "strong" (high V) outliers should both be
    # eligible, ranked purely by |z| regardless of direction, and capped to
    # top_n.
    matrix = pl.DataFrame(
        {
            "source": ["a", "b", "c", "d"] * 3,
            "attribute": ["employment"] * 4 + ["age"] * 4 + ["sex"] * 4,
            "act_type": ["work"] * 12,
            "n": [100] * 12,
            "cramers_v": (
                [0.7, 0.72, 0.68, 0.05]  # d: weak outlier
                + [0.1, 0.12, 0.08, 0.9]  # d: strong outlier, larger |z|
                + [
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                ]  # no variation -> z undefined, not flagged
            ),
        }
    )
    flagged = anomaly.flag_conditionality_outliers(
        matrix, z_threshold=1.5, min_peers=3, top_n=1
    )
    assert flagged.height == 1
    row = flagged.row(0, named=True)
    assert (row["attribute"], row["source"]) == ("age", "d")

    flagged_all = anomaly.flag_conditionality_outliers(
        matrix, z_threshold=1.5, min_peers=3, top_n=20
    )
    assert flagged_all.height == 2
    # most anomalous (largest |z|) first
    assert flagged_all["z"].abs().to_list() == sorted(
        flagged_all["z"].abs().to_list(), reverse=True
    )
    assert flagged_all.row(0, named=True)["attribute"] == "age"


def test_flag_conditionality_outliers_markdown():
    matrix = pl.DataFrame(
        {
            "source": ["a", "b", "c", "d"],
            "attribute": ["employment"] * 4,
            "act_type": ["work"] * 4,
            "n": [100] * 4,
            "cramers_v": [0.7, 0.72, 0.68, 0.05],
        }
    )
    md = anomaly.flag_conditionality_outliers(
        matrix, z_threshold=1.5, markdown=True
    )
    assert isinstance(md, str)
    assert md.startswith("| source |")
    assert "d" in md


# ---------------------------------------------------------------------------
# distribution_shift_matrix
# ---------------------------------------------------------------------------


def test_distribution_shift_matrix_values():
    attrs = pl.DataFrame(
        {
            "pid": [f"p{i}" for i in range(1, 9)],
            "source": ["a"] * 4 + ["b"] * 4,
            "year": [2019] * 8,
            "employment": (
                ["employed", "employed", "employed", "student"]
                + ["employed", "student", "student", "student"]
            ),
        }
    )
    matrix = anomaly.attribute_distribution_shift_matrix(
        attrs,
        on=["source", "year"],
        attribute_cols=["employment"],
        min_group_n=1,
    )
    assert matrix.height == 2

    a = matrix.filter(pl.col("source") == "a").row(0, named=True)
    b = matrix.filter(pl.col("source") == "b").row(0, named=True)

    # source "a" is 75% employed against the pooled 50/50 split, so
    # "employed" is over-represented and "student" under-represented by the
    # same 25pp margin; the reverse holds for source "b" (25% employed).
    assert a["top_over_category"] == "employed"
    assert a["top_over_delta_pct"] == pytest.approx(25.0)
    assert a["top_under_category"] == "student"
    assert a["top_under_delta_pct"] == pytest.approx(-25.0)

    assert b["top_over_category"] == "student"
    assert b["top_under_category"] == "employed"

    # by symmetry (a is 75/25, b is 25/75, pooled is 50/50) both groups'
    # divergence from the pooled distribution should be identical.
    assert a["jsd"] == pytest.approx(b["jsd"])
    assert a["jsd"] > 0


def test_distribution_shift_matrix_zero_for_matching_distribution():
    attrs = pl.DataFrame(
        {
            "pid": [f"p{i}" for i in range(1, 9)],
            "source": ["a", "a", "b", "b"] * 2,
            "year": [2019] * 8,
            "employment": ["employed", "student"] * 4,
        }
    )
    matrix = anomaly.attribute_distribution_shift_matrix(
        attrs,
        on=["source", "year"],
        attribute_cols=["employment"],
        min_group_n=1,
    )
    # both groups match the 50/50 pooled distribution exactly.
    assert matrix["jsd"].to_list() == pytest.approx([0.0, 0.0])


def test_distribution_shift_matrix_respects_min_group_n():
    attrs = pl.DataFrame(
        {
            "pid": [f"p{i}" for i in range(1, 7)],
            "source": ["a"] * 4 + ["b"] * 2,
            "year": [2019] * 6,
            "employment": [
                "employed",
                "student",
                "employed",
                "student",
                "employed",
                "employed",
            ],
        }
    )
    matrix = anomaly.attribute_distribution_shift_matrix(
        attrs,
        on=["source", "year"],
        attribute_cols=["employment"],
        min_group_n=3,
    )
    assert matrix["source"].to_list() == ["a"]


def test_distribution_shift_matrix_skips_missing_attribute_cols():
    attrs = pl.DataFrame(
        {
            "pid": ["p1", "p2"],
            "source": ["a", "a"],
            "year": [2019, 2019],
            "employment": ["employed", "student"],
        }
    )
    matrix = anomaly.attribute_distribution_shift_matrix(
        attrs,
        on=["source", "year"],
        attribute_cols=["employment", "hh_income"],
        min_group_n=1,
    )
    assert set(matrix["attribute"].unique().to_list()) == {"employment"}


# ---------------------------------------------------------------------------
# activity_distribution_shift_matrix
# ---------------------------------------------------------------------------


def test_activity_distribution_shift_matrix_values():
    attrs = pl.DataFrame(
        {
            "pid": [f"p{i}" for i in range(1, 9)],
            "source": ["a"] * 4 + ["b"] * 4,
            "year": [2019] * 8,
        }
    )
    activities = pl.DataFrame(
        {
            "pid": [f"p{i}" for i in range(1, 9)],
            "act": (
                ["work", "work", "work", "education"]
                + ["work", "education", "education", "education"]
            ),
        }
    )
    matrix = anomaly.activity_distribution_shift_matrix(
        attrs, activities, on=["source", "year"], min_group_n=1
    )
    assert matrix.height == 2
    assert set(matrix["attribute"].unique().to_list()) == {"act"}

    a = matrix.filter(pl.col("source") == "a").row(0, named=True)
    b = matrix.filter(pl.col("source") == "b").row(0, named=True)

    # source "a" is 75% work against the pooled 50/50 split, so "work" is
    # over-represented and "education" under-represented by the same 25pp
    # margin; the reverse holds for source "b" (25% work).
    assert a["top_over_category"] == "work"
    assert a["top_over_delta_pct"] == pytest.approx(25.0)
    assert a["top_under_category"] == "education"
    assert a["top_under_delta_pct"] == pytest.approx(-25.0)

    assert b["top_over_category"] == "education"
    assert b["top_under_category"] == "work"

    # by symmetry (a is 75/25, b is 25/75, pooled is 50/50) both groups'
    # divergence from the pooled distribution should be identical.
    assert a["jsd"] == pytest.approx(b["jsd"])
    assert a["jsd"] > 0


def test_activity_distribution_shift_matrix_zero_for_matching_distribution():
    attrs = pl.DataFrame(
        {
            "pid": [f"p{i}" for i in range(1, 9)],
            "source": ["a", "a", "b", "b"] * 2,
            "year": [2019] * 8,
        }
    )
    activities = pl.DataFrame(
        {
            "pid": [f"p{i}" for i in range(1, 9)],
            "act": ["work", "education"] * 4,
        }
    )
    matrix = anomaly.activity_distribution_shift_matrix(
        attrs, activities, on=["source", "year"], min_group_n=1
    )
    assert matrix["jsd"].to_list() == pytest.approx([0.0, 0.0])


def test_activity_distribution_shift_matrix_respects_min_group_n():
    attrs = pl.DataFrame(
        {
            "pid": ["p1", "p2", "p3", "p4"],
            "source": ["a", "a", "b", "b"],
            "year": [2019] * 4,
        }
    )
    activities = pl.DataFrame(
        {
            # source "a" contributes 3 activities, "b" only 1
            "pid": ["p1", "p1", "p2", "p3"],
            "act": ["work", "education", "work", "work"],
        }
    )
    matrix = anomaly.activity_distribution_shift_matrix(
        attrs, activities, on=["source", "year"], min_group_n=2
    )
    assert matrix["source"].to_list() == ["a"]


def test_activity_distribution_shift_matrix_counts_activities_not_persons():
    # p1 contributes two activities, so source "a" should have n=3 activities
    # from 2 persons, not n=2.
    attrs = pl.DataFrame(
        {"pid": ["p1", "p2"], "source": ["a", "a"], "year": [2019, 2019]}
    )
    activities = pl.DataFrame(
        {"pid": ["p1", "p1", "p2"], "act": ["work", "shop", "work"]}
    )
    matrix = anomaly.activity_distribution_shift_matrix(
        attrs, activities, on=["source", "year"], min_group_n=1
    )
    assert matrix.row(0, named=True)["n"] == 3


# ---------------------------------------------------------------------------
# flag_distribution_shift_outliers
# ---------------------------------------------------------------------------


def test_flag_distribution_shift_outliers_top_n():
    matrix = pl.DataFrame(
        {
            "source": ["a", "b", "c", "d"],
            "year": [2019] * 4,
            "attribute": ["employment"] * 4,
            "n": [100] * 4,
            "jsd": [0.1, 0.5, 0.3, 0.9],
            "top_over_category": ["x"] * 4,
            "top_over_delta_pct": [1.0] * 4,
            "top_under_category": ["y"] * 4,
            "top_under_delta_pct": [-1.0] * 4,
        }
    )
    top = anomaly.flag_distribution_shift_outliers(
        matrix, on=["source", "year"], top_n=2
    )
    assert top["source"].to_list() == ["d", "b"]


def test_flag_distribution_shift_outliers_markdown_empty_when_nothing_flagged():
    matrix = pl.DataFrame(
        {
            "source": ["a"],
            "year": [2019],
            "attribute": ["employment"],
            "n": [100],
            "jsd": [float("nan")],
            "top_over_category": ["x"],
            "top_over_delta_pct": [1.0],
            "top_under_category": ["y"],
            "top_under_delta_pct": [-1.0],
        }
    )
    md = anomaly.flag_distribution_shift_outliers(
        matrix, on=["source", "year"], markdown=True
    )
    assert md == "No anomalies flagged."


def test_flag_distribution_shift_outliers_markdown():
    matrix = pl.DataFrame(
        {
            "source": ["a", "b"],
            "year": [2019, 2020],
            "attribute": ["employment"] * 2,
            "n": [100, 100],
            "jsd": [0.1, 0.9],
            "top_over_category": ["employed", "student"],
            "top_over_delta_pct": [10.0, 30.0],
            "top_under_category": ["student", "employed"],
            "top_under_delta_pct": [-10.0, -30.0],
        }
    )
    md = anomaly.flag_distribution_shift_outliers(
        matrix, on=["source", "year"], top_n=1, markdown=True
    )
    assert isinstance(md, str)
    assert md.startswith("| source |")
    assert "b" in md
    assert "student (+30pp)" in md


# ---------------------------------------------------------------------------
# time_quality_summary_table
# ---------------------------------------------------------------------------


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


def test_time_quality_summary_table_values(attrs, trips):
    table = anomaly.time_quality_summary_table(attrs, trips).sort("source")

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
    md = anomaly.time_quality_summary_table(attrs, trips, markdown=True)
    assert isinstance(md, str)
    assert md.startswith("| source |")
    assert "a" in md and "b" in md
