import os

import polars as pl
import pytest

from foundata import utils


@pytest.fixture(scope="session")
def template_attributes():
    return utils.get_template_attributes()


@pytest.fixture(scope="session")
def template_trips():
    return utils.get_template_trips()


@pytest.fixture
def sample_attributes_df():
    """Minimal valid attributes DataFrame matching the template schema."""
    return pl.DataFrame(
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
            "rurality": ["urban", "suburban"],
            "weight": [1.0, 1.5],
        }
    )


@pytest.fixture
def sample_trips_df():
    """Minimal valid trips DataFrame matching the template schema."""
    return pl.DataFrame(
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


def data_root_for(source: str):
    """Return data root path for a source, or None if env var not set."""
    env_var = f"FOUNDATA_{source.upper()}_DATA"
    return os.getenv(env_var)
