import polars as pl

from foundata import filter
from foundata.table_logger import TableLogger


def test_record_collects_under_source():
    log = TableLogger()
    with log.source("A"):
        log.record("step1", "1.0%")
        log.record("step2", "2.0%")
    with log.source("B"):
        log.record("step2", "3.0%")
    out = log.render()
    assert "step1" in out and "step2" in out
    # missing steps are rendered as "-"
    b_row = next(line for line in out.splitlines() if line.startswith("| B"))
    assert "-" in b_row and "3.0%" in b_row


def test_record_without_source_not_collected():
    log = TableLogger()
    log.record("step", "1.0%", "full message")
    log.record("quiet", "0.0%")
    assert log.render() == ""


def test_notes_not_rendered_in_table():
    log = TableLogger()
    with log.source("A"):
        log.note("something odd")
        log.note("very odd", level="ERROR")
    assert "something odd" not in log.render()


def test_filter_records_into_global_logger():
    from foundata import table_logger

    table_logger.reset()
    trips = pl.DataFrame(
        {
            "pid": [1, 2],
            "oact": ["home", None],
            "dact": ["work", "home"],
        }
    )
    with table_logger.source("X"):
        filter.missing_acts(None, trips)
    assert "50.0%" in table_logger.render()
    table_logger.reset()
