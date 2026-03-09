import os

import polars as pl
import pytest

from foundata import fix, utils


@pytest.fixture(scope="session")
def template_attributes():
    return utils.get_template_attributes()


@pytest.fixture(scope="session")
def template_trips():
    return utils.get_template_trips()


@pytest.fixture
def sample_attributes_df():
    """Minimal valid attributes DataFrame matching the template schema."""
    attrs = pl.DataFrame(
        {
            "hid": ["nhts001", "nhts002"],
            "pid": ["nhts0011", "nhts0021"],
            "age": [30, 45],
            "hh_size": [2, 3],
            "hh_income": [40000, 55000],
            "sex": ["male", "female"],
            "dwelling": ["house", "flat"],
            "ownership": ["owned", "rented"],
            "vehicles": [1, 2],
            "disability": ["no", "no"],
            "education": ["degree", "high-school"],
            "can_wfh": ["yes", "no"],
            "occupation": ["professional", "clerical"],
            "race": ["white", "unknown"],
            "has_licence": ["yes", "yes"],
            "relationship": ["self", "self"],
            "employment": ["ft-employed", "pt-employed"],
            "country": ["usa", "usa"],
            "source": ["nhts", "nhts"],
            "year": [2022, 2022],
            "month": [6, 7],
            "day": ["monday", "tuesday"],
            "hh_zone": ["urban", "suburban"],
            "weight": [1.0, 1.5],
            "avg_speed": [30.0, 25.0],
            "access_egress_distance": [None, None],
        }
    )
    trips = pl.DataFrame(
        {
            "pid": ["nhts0011", "nhts0011", "nhts0021"],
            "seq": [0, 1, 0],
            "ozone": ["urban", "urban", "suburban"],
            "dzone": ["urban", "suburban", "urban"],
            "oact": ["home", "work", "home"],
            "dact": ["work", "home", "work"],
            "mode": ["car", "car", "bus"],
            "tst": [480, 1020, 540],
            "tet": [510, 1050, 570],
            "distance": [5.0, 5.0, 3.2],
        }
    )
    attrs, _ = fix.fix_types(attrs, trips)
    return attrs


@pytest.fixture
def sample_trips_df():
    """Minimal valid trips DataFrame matching the template schema."""
    attrs = pl.DataFrame(
        {
            "hid": ["nhts001", "nhts002"],
            "pid": ["nhts0011", "nhts0021"],
            "age": [30, 45],
            "hh_size": [2, 3],
            "hh_income": [40000, 55000],
            "sex": ["male", "female"],
            "dwelling": ["house", "flat"],
            "ownership": ["owned", "rented"],
            "vehicles": [1, 2],
            "disability": ["no", "no"],
            "education": ["degree", "high-school"],
            "can_wfh": ["yes", "no"],
            "occupation": ["professional", "clerical"],
            "race": ["white", "unknown"],
            "has_licence": ["yes", "yes"],
            "relationship": ["self", "self"],
            "employment": ["ft-employed", "pt-employed"],
            "country": ["usa", "usa"],
            "source": ["nhts", "nhts"],
            "year": [2022, 2022],
            "month": [6, 7],
            "day": ["monday", "tuesday"],
            "hh_zone": ["urban", "suburban"],
            "weight": [1.0, 1.5],
            "avg_speed": [30.0, 25.0],
            "access_egress_distance": [None, None],
        }
    )
    trips = pl.DataFrame(
        {
            "pid": ["nhts0011", "nhts0011", "nhts0021"],
            "seq": [0, 1, 0],
            "ozone": ["urban", "urban", "suburban"],
            "dzone": ["urban", "suburban", "urban"],
            "oact": ["home", "work", "home"],
            "dact": ["work", "home", "work"],
            "mode": ["car", "car", "bus"],
            "tst": [480, 1020, 540],
            "tet": [510, 1050, 570],
            "distance": [5.0, 5.0, 3.2],
        }
    )
    _, trips = fix.fix_types(attrs, trips)
    return trips


def data_root_for(source: str):
    """Return data root path for a source, or None if env var not set."""
    env_var = f"FOUNDATA_{source.upper()}_DATA"
    return os.getenv(env_var)
