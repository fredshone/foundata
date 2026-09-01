import polars as pl
import pytest

from foundata import post_process, tables

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


# ---------------------------------------------------------------------------
# render_markdown_table
# ---------------------------------------------------------------------------


def test_render_markdown_table_pads_columns_to_widest_cell():
    md = tables.render_markdown_table(
        ["source", "trips"], [["ltds", "10"], ["vista_long_name", "200,000"]]
    )
    lines = md.splitlines()
    # every line has identical total width once padded — that's what makes
    # columns line up when printed straight to a terminal.
    widths = {len(line) for line in lines}
    assert len(widths) == 1
    # header/data cells for the wider column are padded out to match the
    # widest entry ("vista_long_name" / "200,000") rather than the header.
    assert lines[0] == f"| {'source'.ljust(15)} | {'trips'.ljust(7)} |"
    assert lines[2] == f"| {'ltds'.ljust(15)} | {'10'.ljust(7)} |"
    assert (
        lines[3] == f"| {'vista_long_name'.ljust(15)} | {'200,000'.ljust(7)} |"
    )


def test_render_markdown_table_separator_matches_column_widths():
    md = tables.render_markdown_table(["a", "bb"], [["1", "22"]])
    header, sep, row = md.splitlines()
    assert len(sep) == len(header) == len(row)


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


# ---------------------------------------------------------------------------
# attribute_availability
# ---------------------------------------------------------------------------


def test_attribute_availability_values():
    attrs = pl.DataFrame(
        {
            "pid": ["p1", "p2", "p3", "p4"],
            "hid": ["h1", "h2", "h3", "h4"],
            "source": ["a", "a", "b", "b"],
            # nulls in both a and b => all-availability 50%
            "age": [25, None, None, 50],
            # nulls only in a => all-availability 75%
            "employment": ["employed", "unknown", "unemployed", "employed"],
        }
    )

    table = tables.attribute_availability(attrs)

    assert table.columns == ["attribute", "a", "b", "all"]
    age_row = table.filter(pl.col("attribute") == "age").row(0, named=True)
    assert age_row["a"] == pytest.approx(50.0)
    assert age_row["b"] == pytest.approx(50.0)
    assert age_row["all"] == pytest.approx(50.0)

    # "unknown" strings are treated the same as null
    employment_row = table.filter(pl.col("attribute") == "employment").row(
        0, named=True
    )
    assert employment_row["a"] == pytest.approx(50.0)
    assert employment_row["b"] == pytest.approx(100.0)
    assert employment_row["all"] == pytest.approx(75.0)

    # rows sorted by "all" availability, highest first
    assert table["attribute"].to_list() == ["employment", "age"]


def test_attribute_availability_markdown():
    attrs = pl.DataFrame(
        {
            "pid": ["p1", "p2"],
            "hid": ["h1", "h2"],
            "source": ["a", "b"],
            "age": [25, None],
        }
    )

    md = tables.attribute_availability(attrs, markdown=True)
    assert isinstance(md, str)
    assert "| Attribute | a" in md
    assert "age" in md and "0%" in md and "100%" in md


# ---------------------------------------------------------------------------
# activity_summary_table
# ---------------------------------------------------------------------------


def test_activity_summary_table_values(attrs, trips):
    activities = post_process.trips_to_activities(attrs, trips)
    table = tables.activity_summary_table(attrs, activities).sort(
        "source", "act"
    )

    # source "a": p1 home->education, p2 home->work (day-wraps, dest dropped)
    a_home = table.filter(
        (pl.col("source") == "a") & (pl.col("act") == "home")
    ).row(0, named=True)
    assert a_home["n_activities"] == 2
    assert a_home["n_participants"] == 2
    assert a_home["participation_prob_pct"] == pytest.approx(100.0)
    assert a_home["participation_rate_pct"] == pytest.approx(100.0)

    a_education = table.filter(
        (pl.col("source") == "a") & (pl.col("act") == "education")
    ).row(0, named=True)
    assert a_education["n_activities"] == 1
    assert a_education["n_participants"] == 1
    assert a_education["participation_prob_pct"] == pytest.approx(50.0)
    assert a_education["participation_rate_pct"] == pytest.approx(50.0)
    assert a_education["median_duration_min"] == pytest.approx(1440 - 510)


def test_activity_summary_table_markdown(attrs, trips):
    activities = post_process.trips_to_activities(attrs, trips)
    md = tables.activity_summary_table(attrs, activities, markdown=True)
    assert isinstance(md, str)
    # grouped into one sub-table per activity type, headed by its name, with
    # source as the row key so values are easy to compare across sources
    assert "**education**" in md and "**home**" in md
    assert "| source |" in md
    # education block appears before the home block ("**education**" < "**home**")
    assert md.index("**education**") < md.index("**home**")
    assert "a" in md and "b" in md


def test_activity_summary_table_prob_vs_rate_diverge():
    # p1 makes two "shop" activities in the day, p2 makes none: probability
    # of >=1 shop activity is 50% (1 of 2 persons), but the mean count per
    # person (rate) is 100% (2 shop activities / 2 persons) — the two
    # metrics should genuinely differ here.
    attrs = pl.DataFrame(
        {
            "pid": ["p1", "p2"],
            "source": ["a", "a"],
            "hh_zone": ["urban", "urban"],
        }
    )
    trips = pl.DataFrame(
        {
            "pid": ["p1", "p1", "p2"],
            "seq": [0, 1, 0],
            "oact": ["home", "shop", "home"],
            "dact": ["shop", "shop", "work"],
            "ozone": ["urban"] * 3,
            "dzone": ["urban"] * 3,
            "mode": ["car"] * 3,
            "tst": [480, 520, 480],
            "tet": [500, 540, 500],
            "distance": [1.0, 1.0, 1.0],
        },
        schema=TRIPS_SCHEMA,
    )

    activities = post_process.trips_to_activities(attrs, trips)
    table = tables.activity_summary_table(attrs, activities)
    shop = table.filter(pl.col("act") == "shop").row(0, named=True)
    assert shop["n_activities"] == 2
    assert shop["n_participants"] == 1
    assert shop["participation_prob_pct"] == pytest.approx(50.0)
    assert shop["participation_rate_pct"] == pytest.approx(100.0)
