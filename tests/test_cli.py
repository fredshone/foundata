import polars as pl
import pytest
from typer.testing import CliRunner

from foundata import cli
from foundata.gridjoin import worldtensor


@pytest.fixture
def batch_sources(monkeypatch, tmp_path):
    """Invoke `batch` with the pipeline stubbed, returning the sources run."""
    calls = []
    options = []

    def runner(source_dirs, *args, **kwargs):
        calls.append(list(source_dirs))
        options.append(kwargs)

    monkeypatch.setattr(cli, "runner", runner)

    def invoke(*args):
        result = CliRunner().invoke(cli.app, ["batch", *args])
        assert result.exit_code == 0, result.output
        return calls[-1]

    invoke.root = str(tmp_path)
    invoke.options = options
    return invoke


def test_select_space_separated_before_data_root(batch_sources):
    assert batch_sources("-s", "chile", "brazil", batch_sources.root) == [
        "brazil",
        "chile",
    ]


def test_select_comma_separated_and_repeated(batch_sources):
    sources = batch_sources(
        batch_sources.root, "-s", "brazil,chile", "-s", "nts"
    )
    assert sources == ["nts", "brazil", "chile"]


def test_omit_space_separated(batch_sources):
    sources = batch_sources(batch_sources.root, "-x", "ktdb", "ltds")
    assert "ktdb" not in sources and "ltds" not in sources
    assert "nts" in sources


def test_unknown_source_rejected(tmp_path):
    result = CliRunner().invoke(
        cli.app, ["batch", str(tmp_path), "-s", "brazil,nope"]
    )
    assert result.exit_code != 0


def test_every_command_has_log_level():
    for typer_app in (cli.app, cli.filter_app):
        for command in typer_app.registered_commands:
            assert "log_level" in command.callback.__signature__.parameters


def test_grid_join_is_on_unless_no_grid(batch_sources, tmp_path):
    batch_sources("-s", "chile", "--world", str(tmp_path), batch_sources.root)
    assert batch_sources.options[-1]["world"] == tmp_path
    batch_sources("-s", "chile", "--no-grid", batch_sources.root)
    assert batch_sources.options[-1]["world"] is None


def test_grid_config_option(batch_sources, tmp_path):
    batch_sources("-s", "chile", batch_sources.root)
    assert batch_sources.options[-1]["grid_config"] == worldtensor.CONFIG
    config = tmp_path / "gridjoin.yaml"
    config.write_text("")
    batch_sources(
        "-s", "chile", "--grid-config", str(config), batch_sources.root
    )
    assert batch_sources.options[-1]["grid_config"] == config


def test_tables_are_parquet_unless_csv(batch_sources):
    batch_sources("-s", "chile", batch_sources.root)
    assert batch_sources.options[-1]["csv"] is False
    batch_sources("-s", "chile", "--csv", batch_sources.root)
    assert batch_sources.options[-1]["csv"] is True


@pytest.fixture
def tables(tmp_path):
    """Write small attributes and trips tables as both parquet and csv."""
    attrs = pl.DataFrame({"pid": [1, 2, 3, 4], "age": [10, 20, None, 40]})
    trips = pl.DataFrame({"pid": [1, 2, 3, 4], "seq": [1, 1, 1, 2]})
    for ext, write in (("parquet", "write_parquet"), ("csv", "write_csv")):
        getattr(attrs, write)(tmp_path / f"attributes.{ext}")
        getattr(trips, write)(tmp_path / f"trips.{ext}")
    return tmp_path


def _invoke(*args):
    result = CliRunner().invoke(cli.app, [str(a) for a in args])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("ext", ["parquet", "csv"])
def test_commands_write_their_input_format(tables, ext):
    attrs, trips = tables / f"attributes.{ext}", tables / f"trips.{ext}"
    read = pl.read_parquet if ext == "parquet" else pl.read_csv

    _invoke("bin", attrs)
    assert read(tables / f"attributes_binned.{ext}").height == 4
    _invoke("fill-unknown", attrs)
    assert read(tables / f"attributes_filled.{ext}")["age"].null_count() == 0

    out = tables / "filtered"
    _invoke(
        "filter",
        "attributes",
        "-a",
        attrs,
        "-t",
        trips,
        "-k",
        "pid",
        "-v",
        "1",
        "-o",
        out,
    )
    assert read(out / f"attributes.{ext}")["pid"].to_list() == [1]
    assert read(out / f"trips.{ext}")["pid"].to_list() == [1]

    _invoke("split", attrs, trips, "-o", tables / "split")
    for stem in ("attributes_train", "attributes_test", "trips_test"):
        assert (tables / "split" / f"{stem}.{ext}").exists()


def test_explicit_output_format_follows_its_extension(tables):
    _invoke("bin", tables / "attributes.parquet", "-o", tables / "b.csv")
    assert pl.read_csv(tables / "b.csv").height == 4
