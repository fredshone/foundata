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
    run.runner(
        source_dirs,
        output,
        home_based=True,
        fix_consecutive=True,
        keep_ids=True,
        world=None,
    )
    attributes = pl.read_csv(
        output / "attributes.csv", infer_schema_length=None
    )
    locations = pl.read_parquet(output / "locations.parquet")
    trips = pl.read_csv(output / "trips.csv", infer_schema_length=None)
    activities = pl.read_csv(
        output / "activity_schedules.csv", infer_schema_length=None
    )
    return attributes, locations, trips, activities


def test_hh_location_references_locations(batch):
    attributes, locations, _, _ = batch
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
    attributes, locations, _, _ = batch
    households = attributes.unique("hid").join(
        locations, left_on="hh_location", right_on="location_id", how="left"
    )
    for source, group in households.group_by("source"):
        source = source[0]
        share = (group["geom_source"].fill_null("none") != "none").mean()
        assert share >= MIN_COVERAGE.get(source, 1.0), (source, share)


def test_every_level_has_a_lookup(batch):
    _, locations, _, _ = batch
    for (source, level, geo), _ in locations.group_by("source", "level", "geo"):
        if level != "point":
            assert location.load_lookup(source, geo) is not None, (source, geo)
        assert location.precision_of(level) is not None, level


def test_codes_match_their_lookups_format(batch):
    """A location's code has the length of its lookup's codes, catching
    padding mistakes in a loader's codes."""
    _, locations, _, _ = batch
    for (source, geo), group in locations.filter(
        pl.col("level") != "point"
    ).group_by("source", "geo"):
        lookup = location.load_lookup(source, geo)
        lengths = set(lookup["code"].str.len_chars().unique())
        if len(lengths) == 1:
            assert set(group["code"].str.len_chars().unique()) <= lengths, (
                source,
                geo,
            )


def test_trip_ends_reference_locations(batch):
    attributes, locations, trips, _ = batch
    for column in ("ocode", "dcode"):
        ids = trips[column].drop_nulls().unique()
        assert ids.is_in(locations["location_id"].implode()).all()
    # QHTS declares its trip ends, and a trip home ends in the home SA1
    qhts_pids = attributes.filter(pl.col("source") == "qhts")["pid"]
    qhts = trips.filter(pl.col("pid").is_in(qhts_pids.implode()))
    assert qhts["dcode"].is_not_null().mean() > 0.9
    home = qhts.filter(pl.col("dact") == "home").join(
        attributes.select("pid", "hh_location"), on="pid"
    )
    assert (home["dcode"] == home["hh_location"]).mean() > 0.9


def test_activity_locations(batch):
    attributes, locations, trips, activities = batch
    ids = activities["location"].drop_nulls().unique()
    assert ids.is_in(locations["location_id"].implode()).all()
    # every home activity of a located household is located
    home = activities.filter(pl.col("act") == "home").join(
        attributes.select("pid", "hh_location"), on="pid"
    )
    located = home.filter(pl.col("hh_location").is_not_null())
    assert located["location"].is_not_null().all()
    # trip destinations carry over to the activity that follows
    dests = trips.select("pid", seq=pl.col("seq") + 1, dcode="dcode").join(
        activities, on=["pid", "seq"]
    )
    assert (
        dests.filter(pl.col("dcode").is_not_null())
        .select((pl.col("dcode") == pl.col("location")).all())
        .item()
    )


def test_ids_are_unique_across_sources(batch):
    attributes, _, trips, _ = batch
    assert attributes["pid"].dtype.is_integer()
    assert attributes["pid"].is_unique().all()
    # a household belongs to one source
    assert (
        attributes.group_by("hid").agg(pl.col("source").n_unique())["source"]
        == 1
    ).all()
    assert trips["pid"].is_in(attributes["pid"].implode()).all()


def test_original_ids_map_back_to_each_source(batch, tmp_path_factory):
    attributes, _, _, _ = batch
    output = next(
        p
        for p in tmp_path_factory.getbasetemp().glob("batch*")
        if (p / "original_ids.csv").exists()
    )
    ids = pl.read_csv(output / "original_ids.csv", infer_schema_length=None)
    assert ids.height == attributes.height
    assert ids["pid"].is_unique().all()
    assert ids["original_pid"].is_not_null().all()
    assert ids["original_hid"].is_not_null().all()
    # original ids are unique within a source
    assert ids.select(
        pl.struct("source", "original_pid").is_unique().all()
    ).item()
    # a household's members share its original id
    assert (
        ids.group_by("hid").agg(pl.col("original_hid").n_unique())[
            "original_hid"
        ]
        == 1
    ).all()
