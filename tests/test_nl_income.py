import random

import polars as pl

from foundata import nl_income

CONFIG = nl_income.load_config()


def test_to_gross_maps_percentiles_by_rank():
    table = CONFIG["years"][2018]
    gross = nl_income.to_gross(
        pl.Series([0, *table["disposable"], None]), CONFIG, 2018
    )
    assert gross.to_list() == [0, *table["gross"], None]
    assert gross.dtype == pl.Int32


def test_to_gross_above_p90_uses_top_ratio_and_cap():
    ratio = nl_income.top_ratio(CONFIG, 2018)
    gross = nl_income.to_gross(pl.Series([100000, 10**7]), CONFIG, 2018)
    assert gross[0] == round(100000 * ratio)
    assert gross[1] == CONFIG["max_income"]


def test_year_table_uses_nearest_year():
    assert nl_income.year_table(CONFIG, 2005) is CONFIG["years"][2011]
    assert nl_income.year_table(CONFIG, 2030) is CONFIG["years"][2024]


def test_gross_decile_bands():
    bands = nl_income.gross_decile_bands(CONFIG, 2020)
    gross = CONFIG["years"][2020]["gross"]
    assert bands[1] == [0, gross[0]]
    assert bands[5] == [gross[3], gross[4]]
    assert bands[10] == [gross[8], None]


def test_sample_band():
    random.seed(0)
    assert nl_income.sample_band([0], 3.0, 500000) is None
    assert nl_income.sample_band(None, 3.0, 500000) is None
    closed = [nl_income.sample_band([10, 20], 3.0, 500000) for _ in range(200)]
    assert min(closed) >= 10 and max(closed) <= 20
    tail = [
        nl_income.sample_band([50000, None], 3.0, 200000) for _ in range(5000)
    ]
    assert min(tail) >= 50000 and max(tail) <= 200000
    # truncated Pareto(3) from 50k: mean ~ 74k, far below a uniform's 125k
    assert 68000 < sum(tail) / len(tail) < 78000
