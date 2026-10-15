import pytest
from typer.testing import CliRunner

from foundata import cli


@pytest.fixture
def batch_sources(monkeypatch, tmp_path):
    """Invoke `batch` with the pipeline stubbed, returning the sources run."""
    calls = []
    monkeypatch.setattr(
        cli,
        "runner",
        lambda source_dirs, *args: calls.append(list(source_dirs)),
    )

    def invoke(*args):
        result = CliRunner().invoke(cli.app, ["batch", *args])
        assert result.exit_code == 0, result.output
        return calls[-1]

    invoke.root = str(tmp_path)
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
