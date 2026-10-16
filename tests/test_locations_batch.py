"""End-to-end home locations: run the batch on data_dir_template and check
that every household's hh_location is in locations.parquet, and that each
source's locations resolve against a lookup."""

from pathlib import Path

import polars as pl
import pytest

from foundata import location, run
from foundata.cli import batch_source_dir

TEMPLATE = Path(__file__).parent.parent / "data_dir_template"

# share of each source's template households with a geometry, at least
# (one of the dozen NHTS template households has no home geography)
MIN_COVERAGE = {"nhts": 0.9}


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    output = tmp_path_factory.mktemp("batch")
    source_dirs = {s: batch_source_dir(TEMPLATE, s) for s in run.SOURCES}
    run.runner(source_dirs, output, home_based=True, fix_consecutive=True)
    attributes = pl.read_csv(
        output / "attributes.csv", infer_schema_length=None
    )
    locations = pl.read_parquet(output / "locations.parquet")
    trips = pl.read_csv(output / "trips.csv", infer_schema_length=None)
    return attributes, locations, trips


def test_hh_location_references_locations(batch):
    attributes, locations, _ = batch
    assert locations["location_id"].is_unique().all()
    assert locations["location_key"].is_unique().all()
    ids = attributes["hh_location"].drop_nulls().unique()
    assert ids.is_in(locations["location_id"].implode()).all()
    # a household's members share its location
    assert (
        attributes.group_by("hid").agg(pl.col("hh_location").n_unique())[
            "hh_location"
        ]
        == 1
    ).all()
    # a location belongs to the source of its households
    joined = attributes.join(
        locations, left_on="hh_location", right_on="location_id"
    )
    assert (joined["source"] == joined["source_right"]).all()


def test_every_source_is_located_with_geometry(batch):
    attributes, locations, _ = batch
    households = attributes.unique("hid").join(
        locations, left_on="hh_location", right_on="location_id", how="left"
    )
    for source, group in households.group_by("source"):
        source = source[0]
        share = (group["geom_source"].fill_null("none") != "none").mean()
        assert share >= MIN_COVERAGE.get(source, 1.0), (source, share)


def test_every_level_has_a_lookup(batch):
    _, locations, _ = batch
    for (source, country, level), _ in locations.group_by(
        "source", "country", "level"
    ):
        if level != "point":
            assert location.load_lookup(source, level, country) is not None, (
                source,
                level,
            )
        assert location.precision_of(level) is not None, level


def test_codes_match_their_lookups_format(batch):
    """A location's code has the length of its lookup's codes, catching
    padding mistakes in a source's home_location declaration."""
    _, locations, _ = batch
    for (source, country, level), group in locations.filter(
        pl.col("level") != "point"
    ).group_by("source", "country", "level"):
        lookup = location.load_lookup(source, level, country)
        lengths = set(lookup["code"].str.len_chars().unique())
        if len(lengths) == 1:
            assert set(group["code"].str.len_chars().unique()) <= lengths, (
                source,
                level,
            )


def test_trip_ends_reference_locations(batch):
    attributes, locations, trips = batch
    for column in ("olocation", "dlocation"):
        ids = trips[column].drop_nulls().unique()
        assert ids.is_in(locations["location_id"].implode()).all()
    # QHTS declares its trip ends, and a trip home ends in the home SA1
    qhts = trips.filter(pl.col("pid").str.starts_with("qhts"))
    assert qhts["dlocation"].is_not_null().mean() > 0.9
    home = qhts.filter(pl.col("dact") == "home").join(
        attributes.select("pid", "hh_location"), on="pid"
    )
    assert (home["dlocation"] == home["hh_location"]).mean() > 0.9
