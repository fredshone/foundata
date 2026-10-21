import pytest
from typer.testing import CliRunner

from foundata import cli


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
