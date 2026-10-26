import polars as pl
import pytest
from loguru import logger
from typer.testing import CliRunner

from foundata import cli
from foundata.sample import apply_sample, reindex, sample_pids, strata_table


@pytest.fixture
def warnings():
    messages = []
    handler = logger.add(
        lambda m: messages.append(m.record["message"]), level="WARNING"
    )
    yield messages
    logger.remove(handler)


@pytest.fixture
def attributes():
    # 8 women, 2 men, in households of 2
    return pl.DataFrame(
        {
            "pid": [f"p{i}" for i in range(10)],
            "hid": [f"h{i // 2}" for i in range(10)],
            "sex": ["female"] * 8 + ["male"] * 2,
            "age": [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
        }
    )


def test_unstratified_sample_is_without_replacement(warnings):
    pids = [f"p{i}" for i in range(10)]
    sample = sample_pids(pids, 5, seed=1)
    assert sample.height == 5
    assert sample["pid"].n_unique() == 5
    assert sample["pid"].to_list() == sample["sample_pid"].to_list()
    assert not warnings


def test_sample_is_reproducible():
    pids = list(range(100))
    assert sample_pids(pids, 10, seed=3).equals(sample_pids(pids, 10, seed=3))


def test_unstratified_oversample_warns_and_relabels(warnings):
    sample = sample_pids(["a", "b"], 5)
    assert sample.height == 5
    assert sample["sample_pid"].n_unique() == 5
    assert set(sample["pid"]) == {"a", "b"}
    assert any("oversampling" in w for w in warnings)


def test_strata_are_equal_size(attributes, warnings):
    sample = sample_pids(
        attributes["pid"].to_list(), 4, attributes=attributes, strata=["sex"]
    )
    counts = sample.join(attributes, on="pid").group_by("sex").len().sort("sex")
    assert counts["len"].to_list() == [2, 2]
    assert sample["pid"].n_unique() == 4
    assert not warnings


def test_small_stratum_is_oversampled(attributes, warnings):
    sample = sample_pids(
        attributes["pid"].to_list(), 8, attributes=attributes, strata=["sex"]
    )
    men = sample.join(attributes, on="pid").filter(pl.col("sex") == "male")
    assert men.height == 4
    # each man is drawn twice
    assert men["pid"].value_counts()["count"].to_list() == [2, 2]
    assert sample["sample_pid"].n_unique() == 8
    women = sample.join(attributes, on="pid").filter(pl.col("sex") == "female")
    assert women["pid"].n_unique() == 4
    assert any("male" in w and "oversampling" in w for w in warnings)


def test_numeric_strata_with_many_values_are_binned(attributes, warnings):
    table = strata_table(attributes, ["age"])
    assert table["age"].dtype == pl.String
    assert table["age"].n_unique() == 5
    assert any("binning" in w for w in warnings)


def test_numeric_strata_with_few_values_are_kept(warnings):
    attrs = pl.DataFrame({"pid": ["a", "b", "c"], "hh_size": [1, 2, 2]})
    assert strata_table(attrs, ["hh_size"])["hh_size"].dtype == pl.Int64
    assert not warnings


def test_missing_strata_column_raises(attributes):
    with pytest.raises(ValueError, match="income"):
        sample_pids(["p0"], 1, attributes=attributes, strata=["income"])


def test_too_many_strata_for_size_raises(attributes):
    with pytest.raises(ValueError, match="smaller than"):
        sample_pids(
            attributes["pid"].to_list(),
            1,
            attributes=attributes,
            strata=["sex"],
        )


def test_apply_sample_repeats_persons_with_new_pids():
    trips = pl.DataFrame({"pid": ["a", "a", "b"], "seq": [1, 2, 1]})
    sample = pl.DataFrame({"pid": ["a", "a"], "sample_pid": ["a", "a_1"]})
    out = apply_sample(trips, sample)
    assert out.columns == ["pid", "seq"]
    assert out.rows() == [("a", 1), ("a", 2), ("a_1", 1), ("a_1", 2)]


@pytest.fixture
def tables(tmp_path, attributes):
    schedules = pl.DataFrame(
        {"pid": attributes["pid"].to_list() * 2, "act": ["home"] * 20}
    )
    trips = attributes.select("pid", "hid", seq=pl.lit(1))

    def write(ext):
        for name, df in (
            ("activity_schedules", schedules),
            ("attributes", attributes),
            ("trips", trips),
        ):
            getattr(df, f"write_{ext}")(tmp_path / f"{name}.{ext}")
        return tmp_path

    return write


def _sample(root, ext, *args):
    result = CliRunner().invoke(
        cli.app,
        [
            "sample",
            str(root / f"activity_schedules.{ext}"),
            "-a",
            str(root / f"attributes.{ext}"),
            "-t",
            str(root / f"trips.{ext}"),
            "-o",
            str(root / "sampled"),
            *args,
        ],
    )
    assert result.exit_code == 0, result.output
    read = pl.read_parquet if ext == "parquet" else pl.read_csv
    return {
        name: read(root / "sampled" / f"{name}.{ext}")
        for name in ("activity_schedules", "attributes", "trips")
    }


def test_reindex_keeps_households_together():
    sample = pl.DataFrame(
        {
            "pid": ["a", "b", "c", "a", "b", "d"],
            "copy": [0, 0, 0, 1, 1, 0],
            "sample_pid": ["a", "b", "c", "a_1", "b_1", "d"],
        }
    )
    hids = pl.DataFrame(
        {"pid": ["a", "b", "c", "d"], "hid": ["h1", "h1", "h2", None]}
    )
    out = reindex(sample, hids)
    assert out["sample_pid"].to_list() == list(range(6))
    assert out["sample_hid"].to_list() == [0, 0, 1, 2, 2, None]


def test_apply_sample_replaces_hid():
    attrs = pl.DataFrame({"pid": ["a", "b"], "hid": ["h", "h"], "age": [1, 2]})
    sample = pl.DataFrame(
        {"pid": ["b", "a"], "sample_pid": [0, 1], "sample_hid": [0, 0]}
    )
    out = apply_sample(attrs, sample)
    assert out.rows() == [(0, 0, 2), (1, 0, 1)]


@pytest.mark.parametrize("ext", ["parquet", "csv"])
def test_sample_command(tables, ext):
    out = _sample(tables(ext), ext, "-n", "6", "-s", "sex")
    assert out["attributes"]["sex"].value_counts()["count"].to_list() == [3, 3]
    assert out["activity_schedules"].height == 12
    assert out["trips"].height == 6


def test_sample_command_reindexes_pids(tables):
    # 4 per sex, so the 2 men are each drawn twice
    out = _sample(tables("parquet"), "parquet", "-n", "8", "-s", "sex")
    assert out["attributes"]["pid"].to_list() == list(range(8))
    assert out["trips"]["pid"].to_list() == list(range(8))
    assert set(out["activity_schedules"]["pid"]) == set(range(8))


def test_sample_command_reindexes_hids(tables):
    out = _sample(tables("parquet"), "parquet", "-n", "8", "-s", "sex")
    attrs = out["attributes"]
    assert attrs["hid"].dtype == pl.Int64
    # the 2 men share a household: each draw of the pair is a household
    men = attrs.filter(pl.col("sex") == "male")
    assert men.group_by("hid").len()["len"].to_list() == [2, 2]
    assert out["trips"].select("pid", "hid").equals(attrs.select("pid", "hid"))


def test_sample_command_keep_index(tables):
    out = _sample(
        tables("parquet"), "parquet", "-n", "8", "-s", "sex", "--keep-index"
    )
    pids = out["attributes"]["pid"]
    assert pids.n_unique() == 8
    assert {"p8_1", "p9_1"} <= set(pids)
    assert set(out["trips"]["pid"]) == set(pids)


def test_sample_command_strata_require_attributes(tmp_path):
    path = tmp_path / "activity_schedules.csv"
    pl.DataFrame({"pid": ["a"]}).write_csv(path)
    result = CliRunner().invoke(
        cli.app, ["sample", str(path), "-n", "1", "-s", "sex"]
    )
    assert result.exit_code != 0
