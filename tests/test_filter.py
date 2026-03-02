import polars as pl

from foundata import filter


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


# --- filter.columns ---

def test_columns_trims_to_template(sample_attributes_df, sample_trips_df):
    attrs_extra = sample_attributes_df.with_columns(
        pl.lit("junk").alias("raw_col_x")
    )
    trips_extra = sample_trips_df.with_columns(
        pl.lit(99).alias("raw_seq_x")
    )
    clean_attrs, clean_trips = filter.columns(attrs_extra, trips_extra)
    assert "raw_col_x" not in clean_attrs.columns
    assert "raw_seq_x" not in clean_trips.columns
    # All template columns are present
    from foundata import utils
    assert set(utils.get_template_attributes().keys()).issubset(set(clean_attrs.columns))
    assert set(utils.get_template_trips().keys()).issubset(set(clean_trips.columns))
