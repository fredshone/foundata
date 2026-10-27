"""Missing source data fails early, with an error naming what is missing."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from foundata import cli, france, run


def test_load_source_missing_dir(tmp_path):
    with pytest.raises(run.SourceDataError, match="nhts: data directory"):
        run.load_source("nhts", tmp_path / "NHTS")


def test_load_source_empty_dir(tmp_path):
    with pytest.raises(run.SourceDataError, match="is empty"):
        run.load_source("nhts", tmp_path)


def test_load_source_missing_file_names_it(tmp_path):
    (tmp_path / "2022").mkdir()
    with pytest.raises(run.SourceDataError, match=r"nhts: .*hhv2pub\.csv"):
        run.load_source("nhts", tmp_path)


def test_runner_checks_every_source_before_loading(tmp_path, monkeypatch):
    loaded = []
    monkeypatch.setattr(run, "load_source", lambda *a: loaded.append(a))
    (tmp_path / "CHILE").mkdir()
    (tmp_path / "CHILE" / "data.csv").write_text("")
    source_dirs = {
        s: run.batch_source_dir(tmp_path, s)
        for s in ("nhts", "chile", "mexico")
    }
    with pytest.raises(run.SourceDataError) as e:
        run.runner(source_dirs, tmp_path / "out")
    assert "nhts" in str(e.value) and "mexico" in str(e.value)
    assert "chile" not in str(e.value)
    assert not loaded


def test_cli_reports_missing_data_without_traceback(tmp_path):
    result = CliRunner().invoke(
        cli.app, ["run", "nhts", str(tmp_path), "-o", str(tmp_path / "out")]
    )
    assert result.exit_code == 1
    assert not isinstance(result.exception, run.SourceDataError)


def _survey(root: Path, name: str, tables=True) -> Path:
    survey = root / name
    survey.mkdir()
    (survey / france.METADATA_PATH).write_text(json.dumps({}))
    if tables:
        for table in (
            france.HOUSEHOLDS_PATH,
            france.PERSONS_PATH,
            france.TRIPS_PATH,
            france.LEGS_PATH,
        ):
            (survey / table).write_text("")
    return survey


def test_france_no_surveys(tmp_path):
    (tmp_path / "not_a_survey").mkdir()
    with pytest.raises(FileNotFoundError, match="no survey folders"):
        france.survey_dirs(tmp_path, [])


def test_france_only_excluded_surveys(tmp_path):
    _survey(tmp_path, "EMP2019")
    with pytest.raises(FileNotFoundError, match="no survey folders"):
        france.survey_dirs(tmp_path, ["EMP2019"])


def test_france_survey_missing_table_fails_up_front(tmp_path):
    _survey(tmp_path, "a_2015")
    _survey(tmp_path, "b_2016", tables=False)
    with pytest.raises(FileNotFoundError, match=r"b_2016/households\.parquet"):
        france.survey_dirs(tmp_path, [])


def test_france_survey_dirs(tmp_path):
    a = _survey(tmp_path, "a_2015")
    _survey(tmp_path, "EMP2019", tables=False)
    assert france.survey_dirs(tmp_path, ["EMP2019"]) == [a]


def test_runner_writes_metadata(tmp_path, monkeypatch):
    repo = Path(__file__).parent.parent
    monkeypatch.chdir(repo)
    template = repo / "data_dir_template"
    source_dirs = {"chile": run.batch_source_dir(template, "chile")}
    run.runner(source_dirs, tmp_path, home_based=True, world=None)
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["started"] <= meta["created"]
    assert meta["options"]["home_based"] is True
    assert meta["options"]["world"] is None
    chile = meta["sources"]["chile"]
    assert chile["data_dir"] == "data_dir_template/CHILE"
    assert chile["persons"] > 0 and chile["trips"] > 0
    assert chile["years"] and all(isinstance(y, int) for y in chile["years"])
    assert meta["command"]["argv"]
    assert str(Path.home()) not in json.dumps(meta)
    assert meta["code"]["commit"]
