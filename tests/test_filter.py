import polars as pl
from click.testing import CliRunner

from foundata import filter
from foundata.cli import cli


def make_attrs(pids: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"pid": pids})


def make_trips(
    pids: list[str],
    tsts: list[int],
    tets: list[int],
    seqs: list[int] | None = None,
) -> pl.DataFrame:
    n = len(pids)
    seqs = seqs or list(range(n))
    return pl.DataFrame({"pid": pids, "seq": seqs, "tst": tsts, "tet": tets})


# --- trips_on_endings ---


def make_trips_with_tet(pids: list[str], tets: list[int]) -> pl.DataFrame:
    n = len(pids)
    return pl.DataFrame(
        {"pid": pids, "seq": list(range(n)), "tst": [0] * n, "tet": tets}
    )


def test_trips_on_endings_removes_trips_over_limit():
    trips = make_trips_with_tet(["p1", "p1", "p2"], [500, 1500, 800])
    result = filter.trips_on_endings(trips, time_limit=1440)
    assert len(result) == 2
    assert set(result["pid"]) == {"p1", "p2"}
    assert list(result["tet"]) == [500, 800]


def test_trips_on_endings_keeps_trip_at_limit():
    """tet == time_limit is kept (inclusive boundary)."""
    trips = make_trips_with_tet(["p1", "p1"], [1440, 1441])
    result = filter.trips_on_endings(trips, time_limit=1440)
    assert len(result) == 1
    assert result["tet"][0] == 1440


def test_trips_on_endings_default_limit_is_1440():
    trips = make_trips_with_tet(["p1", "p1"], [1440, 1441])
    result = filter.trips_on_endings(trips)
    assert len(result) == 1


def test_trips_on_endings_custom_limit():
    trips = make_trips_with_tet(["p1", "p1", "p1"], [200, 500, 900])
    result = filter.trips_on_endings(trips, time_limit=500)
    assert len(result) == 2
    assert all(t <= 500 for t in result["tet"])


def test_trips_on_endings_partial_plan_keeps_pid():
    """Plan with some trips over limit loses those trips but stays in result."""
    trips = make_trips_with_tet(["p1", "p1", "p1"], [400, 1500, 1600])
    result = filter.trips_on_endings(trips, time_limit=1440)
    assert len(result) == 1
    assert result["pid"][0] == "p1"


def test_trips_on_endings_all_trips_over_limit_removes_pid():
    trips = make_trips_with_tet(["p1", "p1", "p2"], [1500, 1600, 800])
    result = filter.trips_on_endings(trips, time_limit=1440)
    assert set(result["pid"]) == {"p2"}


def test_trips_on_endings_keeps_all_when_none_exceed_limit():
    trips = make_trips_with_tet(["p1", "p2", "p3"], [100, 500, 1440])
    result = filter.trips_on_endings(trips, time_limit=1440)
    assert len(result) == 3


# --- negative_trips ---


def test_negative_trips_removes_plans():
    attrs = make_attrs(["p1", "p2"])
    trips = make_trips(
        ["p1", "p1", "p2"],
        [100, 500, 200],
        [200, 400, 300],  # p1 second trip: tst=500 > tet=400 → bad
    )
    clean_attrs, clean_trips = filter.negative_trips(attrs, trips)
    assert set(clean_attrs["pid"]) == {"p2"}
    assert set(clean_trips["pid"]) == {"p2"}


def test_negative_trips_keeps_good_plans():
    attrs = make_attrs(["p1"])
    trips = make_trips(["p1", "p1"], [100, 200], [150, 250])
    clean_attrs, clean_trips = filter.negative_trips(attrs, trips)
    assert len(clean_attrs) == 1
    assert len(clean_trips) == 2


# --- negative_activities ---


def test_negative_activities_removes_plans():
    attrs = make_attrs(["p1", "p2"])
    # p1: second trip starts before first ends (tst < previous tet)
    trips = pl.DataFrame(
        {
            "pid": ["p1", "p1", "p2"],
            "seq": [0, 1, 0],
            "tst": [100, 150, 200],  # p1: tst[1]=150 < tet[0]=200 → overlap
            "tet": [200, 300, 300],
        }
    )
    clean_attrs, clean_trips = filter.negative_activities(attrs, trips)
    assert set(clean_attrs["pid"]) == {"p2"}


# --- null_times ---


def test_null_times_removes_plans():
    attrs = make_attrs(["p1", "p2"])
    trips = pl.DataFrame(
        {
            "pid": ["p1", "p2"],
            "seq": [0, 0],
            "tst": [None, 100],
            "tet": [200, 300],
        }
    )
    clean_attrs, clean_trips = filter.null_times(attrs, trips)
    assert set(clean_attrs["pid"]) == {"p2"}
    assert set(clean_trips["pid"]) == {"p2"}


# --- time_consistent (all three combined) ---


def test_time_consistent_chains_all_three():
    attrs = make_attrs(["p1", "p2", "p3"])
    trips = pl.DataFrame(
        {
            "pid": ["p1", "p2", "p2", "p3"],
            "seq": [0, 0, 1, 0],
            "tst": [None, 100, 90, 200],  # p1: null tst; p2: overlap; p3: ok
            "tet": [200, 200, 300, 300],
        }
    )
    clean_attrs, clean_trips = filter.time_consistent(attrs, trips)
    assert set(clean_attrs["pid"]) == {"p3"}


# --- home_based ---


def make_trips_with_acts(
    pids: list[str], seqs: list[int], oacts: list[str], dacts: list[str]
) -> pl.DataFrame:
    return pl.DataFrame(
        {"pid": pids, "seq": seqs, "oact": oacts, "dact": dacts}
    )


def test_home_based_keeps_home_based_plans():
    attrs = make_attrs(["p1"])
    trips = make_trips_with_acts(
        ["p1", "p1"], [0, 1], ["home", "work"], ["work", "home"]
    )
    clean_attrs, clean_trips = filter.home_based(attrs, trips)
    assert set(clean_attrs["pid"]) == {"p1"}
    assert len(clean_trips) == 2


def test_home_based_keeps_home_to_home_plans():
    attrs = make_attrs(["p1"])
    trips = make_trips_with_acts(["p1"], [0], ["home"], ["home"])
    clean_attrs, clean_trips = filter.home_based(attrs, trips)
    assert set(clean_attrs["pid"]) == {"p1"}
    assert len(clean_trips) == 1


def test_home_based_removes_non_home_start():
    attrs = make_attrs(["p1", "p2"])
    trips = make_trips_with_acts(
        ["p1", "p1", "p2", "p2"],
        [0, 1, 0, 1],
        ["work", "home", "home", "work"],  # p1 starts at work → removed
        ["home", "home", "work", "home"],
    )
    clean_attrs, clean_trips = filter.home_based(attrs, trips)
    assert set(clean_attrs["pid"]) == {"p2"}
    assert set(clean_trips["pid"]) == {"p2"}


def test_home_based_removes_non_home_end():
    attrs = make_attrs(["p1", "p2"])
    trips = make_trips_with_acts(
        ["p1", "p1", "p2", "p2"],
        [0, 1, 0, 1],
        ["home", "work", "home", "work"],
        ["work", "work", "work", "home"],  # p1 ends at work → removed
    )
    clean_attrs, clean_trips = filter.home_based(attrs, trips)
    assert set(clean_attrs["pid"]) == {"p2"}
    assert set(clean_trips["pid"]) == {"p2"}


def test_home_based_uses_seq_order():
    """Trips provided out of seq order should still be evaluated correctly."""
    attrs = make_attrs(["p1"])
    trips = pl.DataFrame(
        {
            "pid": ["p1", "p1"],
            "seq": [1, 0],  # reversed
            "oact": ["work", "home"],
            "dact": ["home", "work"],
        }
    )
    clean_attrs, clean_trips = filter.home_based(attrs, trips)
    assert set(clean_attrs["pid"]) == {"p1"}


def test_home_based_none_attributes():
    trips = make_trips_with_acts(
        ["p1", "p1", "p2"],
        [0, 1, 0],
        ["home", "work", "work"],
        ["work", "home", "home"],
    )
    clean_attrs, clean_trips = filter.home_based(None, trips)
    assert clean_attrs is None
    assert set(clean_trips["pid"]) == {"p1"}


# --- filter_consecutive_activities ---


def test_consecutive_activities_removes_plans_with_consecutive_home_only():
    attrs = make_attrs(["p1", "p2"])
    trips = make_trips_with_acts(
        ["p1", "p2", "p2"],
        [0, 0, 1],
        ["home", "home", "work"],  # p1: two consecutive home acts → removed
        ["home", "work", "home"],
    )
    clean_attrs, clean_trips = filter.filter_consecutive_activities(
        attrs, trips
    )
    assert set(clean_attrs["pid"]) == {"p2"}
    assert set(clean_trips["pid"]) == {"p2"}


def test_consecutive_activities_removes_plans_with_consecutive_home_within_day():
    attrs = make_attrs(["p1", "p2"])
    trips = make_trips_with_acts(
        ["p1", "p1", "p1", "p1", "p1", "p2", "p2"],
        [0, 1, 2, 3, 4, 0, 1],
        [
            "home",
            "shop",
            "home",
            "home",
            "other",
            "home",
            "work",
        ],  # p1: two consecutive home acts → removed
        ["shop", "home", "home", "other", "home", "work", "home"],
    )
    clean_attrs, clean_trips = filter.filter_consecutive_activities(
        attrs, trips
    )
    assert set(clean_attrs["pid"]) == {"p2"}
    assert set(clean_trips["pid"]) == {"p2"}


def test_consecutive_activities_removes_plans_with_consecutive_work():
    attrs = make_attrs(["p1", "p2"])
    trips = make_trips_with_acts(
        ["p1", "p1", "p2", "p2"],
        [0, 1, 0, 1],
        [
            "work",
            "work",
            "home",
            "work",
        ],  # p1: two consecutive work oacts → removed
        ["home", "home", "work", "home"],
    )
    clean_attrs, clean_trips = filter.filter_consecutive_activities(
        attrs, trips
    )
    assert set(clean_attrs["pid"]) == {"p2"}
    assert set(clean_trips["pid"]) == {"p2"}


def test_consecutive_activities_keeps_plans_without_consecutive_nonconsecutive():
    attrs = make_attrs(["p1"])
    trips = make_trips_with_acts(
        ["p1", "p1"],
        [0, 1],
        ["home", "work"],  # different oacts → keep
        ["work", "home"],
    )
    clean_attrs, clean_trips = filter.filter_consecutive_activities(
        attrs, trips
    )
    assert len(clean_attrs) == 1
    assert len(clean_trips) == 2


def test_consecutive_activities_keeps_consecutive_acts_not_in_list():
    attrs = make_attrs(["p1"])
    trips = make_trips_with_acts(
        ["p1", "p1"],
        [0, 1],
        [
            "shop",
            "shop",
        ],  # consecutive, but "other" not in default non_consecutive_types
        ["other", "other"],
    )
    clean_attrs, clean_trips = filter.filter_consecutive_activities(
        attrs, trips
    )
    assert len(clean_attrs) == 1
    assert len(clean_trips) == 2


def test_consecutive_activities_removes_consecutive_education():
    attrs = make_attrs(["p1", "p2"])
    trips = make_trips_with_acts(
        ["p1", "p1", "p2"],
        [0, 1, 0],
        [
            "education",
            "education",
            "work",
        ],  # p1: consecutive education → removed
        ["home", "home", "home"],
    )
    clean_attrs, clean_trips = filter.filter_consecutive_activities(
        attrs, trips
    )
    assert set(clean_attrs["pid"]) == {"p2"}


def test_consecutive_activities_custom_non_consecutive_types():
    attrs = make_attrs(["p1", "p2"])
    trips = make_trips_with_acts(
        ["p1", "p1", "p2", "p2"],
        [0, 1, 0, 1],
        [
            "home",
            "home",
            "work",
            "home",
        ],  # p1: consecutive shop destinations, custom list → removed
        ["shop", "shop", "home", "home"],
    )
    clean_attrs, clean_trips = filter.filter_consecutive_activities(
        attrs, trips, non_consecutive_types=["shop"]
    )
    assert set(clean_attrs["pid"]) == {"p2"}


def test_consecutive_activities_none_attributes():
    trips = make_trips_with_acts(
        ["p1", "p1", "p2"],
        [0, 1, 0],
        ["work", "work", "home"],
        ["home", "home", "work"],
    )
    clean_attrs, clean_trips = filter.filter_consecutive_activities(None, trips)
    assert clean_attrs is None
    assert set(clean_trips["pid"]) == {"p2"}


def test_consecutive_activities_inconsistent_oact_dact_chain():
    """Regression: consecutive dacts are detected even when oact[i] != dact[i-1]."""
    attrs = make_attrs(["p1", "p2"])
    trips = make_trips_with_acts(
        ["p1", "p1", "p1", "p2"],
        [0, 1, 2, 0],
        [
            "home",
            "shop",
            "work",
            "home",
        ],  # p1: oact[2]=work but dact[1]==dact[2]==home
        ["shop", "home", "home", "work"],
    )
    clean_attrs, clean_trips = filter.filter_consecutive_activities(
        attrs, trips, non_consecutive_types=["home"]
    )
    assert set(clean_attrs["pid"]) == {"p2"}
    assert set(clean_trips["pid"]) == {"p2"}


# --- activity_consistency ---


def test_activity_consistency_removes_inconsistent():
    attrs = make_attrs(["p1", "p2"])
    trips = pl.DataFrame(
        {
            "pid": ["p1", "p1", "p2", "p2"],
            "seq": [0, 1, 0, 1],
            "oact": ["home", "work", "home", "work"],
            "dact": [
                "shop",
                "home",
                "work",
                "home",
            ],  # p1: dact[0]=shop, oact[1]=work → mismatch
        }
    )
    clean_attrs, clean_trips = filter.activity_consistency(attrs, trips)
    assert set(clean_attrs["pid"]) == {"p2"}
    assert set(clean_trips["pid"]) == {"p2"}


def test_activity_consistency_keeps_consistent():
    attrs = make_attrs(["p1"])
    trips = make_trips_with_acts(
        ["p1", "p1"], [0, 1], ["home", "work"], ["work", "home"]
    )
    clean_attrs, clean_trips = filter.activity_consistency(attrs, trips)
    assert len(clean_attrs) == 1
    assert len(clean_trips) == 2


def test_activity_consistency_skips_unknown():
    attrs = make_attrs(["p1"])
    trips = pl.DataFrame(
        {
            "pid": ["p1", "p1"],
            "seq": [0, 1],
            "oact": ["home", "unknown"],
            "dact": ["unknown", "home"],  # dact[0]=unknown → skip check
        }
    )
    clean_attrs, clean_trips = filter.activity_consistency(attrs, trips)
    assert len(clean_attrs) == 1


def test_activity_consistency_none_attributes():
    trips = pl.DataFrame(
        {
            "pid": ["p1", "p1", "p2"],
            "seq": [0, 1, 0],
            "oact": ["home", "work", "home"],
            "dact": [
                "shop",
                "home",
                "work",
            ],  # p1: dact[0]=shop, oact[1]=work → mismatch
        }
    )
    clean_attrs, clean_trips = filter.activity_consistency(None, trips)
    assert clean_attrs is None
    assert set(clean_trips["pid"]) == {"p2"}


# --- filter.columns ---


def test_columns_trims_to_template(sample_attributes_df, sample_trips_df):
    attrs_extra = sample_attributes_df.with_columns(
        pl.lit("junk").alias("raw_col_x")
    )
    trips_extra = sample_trips_df.with_columns(pl.lit(99).alias("raw_seq_x"))
    clean_attrs, clean_trips = filter.columns(attrs_extra, trips_extra)
    assert "raw_col_x" not in clean_attrs.columns
    assert "raw_seq_x" not in clean_trips.columns
    # All template columns are present
    from foundata import utils

    assert set(utils.get_template_attributes().keys()).issubset(
        set(clean_attrs.columns)
    )
    assert set(utils.get_template_trips().keys()).issubset(
        set(clean_trips.columns)
    )


def test_filter_attributes_numeric_key(tmp_path):
    attrs = pl.DataFrame({"pid": ["a", "b", "c"], "year": [2022, 2023, 2023]})
    trips = pl.DataFrame({"pid": ["a", "b", "c"], "seq": [1, 1, 2]})
    attrs_path = tmp_path / "attrs.csv"
    trips_path = tmp_path / "trips.csv"
    attrs.write_csv(attrs_path)
    trips.write_csv(trips_path)

    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "filter",
            "attributes",
            "-a",
            str(attrs_path),
            "-t",
            str(trips_path),
            "-k",
            "year",
            "-v",
            "2023",
            "-o",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    out_attrs = pl.read_csv(out_dir / "attrs_filtered.csv")
    assert len(out_attrs) == 2
    assert set(out_attrs["pid"].to_list()) == {"b", "c"}
