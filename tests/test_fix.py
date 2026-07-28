import polars as pl

from foundata import fix


def test_day_wrap_preserves_row_count():
    """Regression test for a bug where `.remove("flag")` (filters rows) was
    used instead of `.drop("flag")` (drops the helper column), silently
    deleting every trip from a person's first inconsistency onward."""
    trips = pl.DataFrame(
        {
            "pid": ["p1", "p1", "p2", "p2", "p3", "p3", "p3"],
            "seq": [1, 2, 1, 2, 1, 2, 3],
            "tst": [1380, 100, 100, 150, 1000, 1000, 1000],
            "tet": [30, 200, 200, 250, 900, 900, 900],
        }
    )
    result = fix.day_wrap(trips)
    assert result.height == trips.height
    assert "flag" not in result.columns


def test_day_wrap_corrects_midnight_crossing_in_place():
    trips = pl.DataFrame(
        {"pid": ["p1"], "seq": [1], "tst": [1380], "tet": [30]}
    )
    result = fix.day_wrap(trips)
    assert result.height == 1
    row = result.row(0, named=True)
    assert row["tst"] == 1380
    assert row["tet"] == 1470
    assert row["tet"] > row["tst"]


def test_day_wrap_corrects_activity_overlap():
    trips = pl.DataFrame(
        {
            "pid": ["p2", "p2"],
            "seq": [1, 2],
            "tst": [100, 150],
            "tet": [200, 250],
        }
    )
    result = fix.day_wrap(trips).sort("seq")
    assert result.height == 2
    first, second = result.row(0, named=True), result.row(1, named=True)
    assert first["tst"] == 100
    assert first["tet"] == 200
    assert second["tst"] == 1590
    assert second["tet"] == 1690


def test_day_wrap_does_not_cascade_beyond_one_day():
    """Repeated inconsistencies for one pid should not stack multiple
    1440-minute shifts — a real single-day diary never needs more than one
    midnight-crossing correction. Trips still inconsistent after that are
    left for downstream filters (`filter.time_consistent`, etc.) to drop."""
    trips = pl.DataFrame(
        {
            "pid": ["p3", "p3", "p3"],
            "seq": [1, 2, 3],
            "tst": [1000, 1000, 1000],
            "tet": [900, 900, 900],
        }
    )
    result = fix.day_wrap(trips)
    assert result.height == 3
    # a single day's worth of correction is at most +1440; anything at or
    # beyond +2880 would mean a second (cascading) shift was applied
    assert result["tst"].max() < 1000 + 2 * 1440
    assert result["tet"].max() < 900 + 2 * 1440
