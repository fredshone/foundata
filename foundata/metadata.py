"""Describe a pipeline run in a ``metadata.json`` beside its outputs: when
it ran, the command and options used, the data sources read, and the code
version."""

import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import polars as pl

FILENAME = "metadata.json"

REPO_ROOT = Path(__file__).resolve().parent.parent


def _home_relative(path: Path) -> str:
    """Absolute *path* as ``~/...`` if under the home directory."""
    if path.is_relative_to(Path.home()):
        return str(Path("~") / path.relative_to(Path.home()))
    return str(path)


def relative(path: str | Path | None) -> str | None:
    """*path* relative to the working directory if within it, else to the
    home directory (as ``~/...``), so metadata doesn't expose the local
    layout; absolute only if it is under neither."""
    if path is None:
        return None
    path = Path(path).expanduser().resolve()
    if path.is_relative_to(Path.cwd()):
        return str(path.relative_to(Path.cwd()))
    return _home_relative(path)


def now() -> str:
    """The current UTC time, ISO formatted to the second."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git(*args: str) -> str | None:
    """Output of git *args* in the repo, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def code_version() -> dict:
    """The foundata package version and, in a git checkout, its commit,
    branch and whether tracked files have uncommitted changes."""
    try:
        version = importlib_metadata.version("foundata")
    except importlib_metadata.PackageNotFoundError:
        version = None
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "version": version,
        "commit": commit,
        "branch": commit and _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
        "python": platform.python_version(),
    }


def command() -> dict:
    """The command line this process was started with, and where."""
    argv = [Path(sys.argv[0]).name, *sys.argv[1:]]
    return {
        "argv": argv,
        "command": shlex.join(argv),
        "cwd": _home_relative(Path.cwd()),
    }


def source_summary(
    source_dirs: dict[str, Path], attributes: pl.DataFrame, trips: pl.DataFrame
) -> dict:
    """Per source: its data directory, survey years, and the persons,
    households and trips it contributes to the outputs."""
    persons = attributes.group_by("source").agg(
        pl.len().alias("persons"),
        pl.col("hid").n_unique().alias("households"),
        pl.col("year").drop_nulls().unique().sort().alias("years"),
    )
    n_trips = dict(
        trips.join(attributes.select("pid", "source"), on="pid")
        .group_by("source")
        .len()
        .iter_rows()
    )
    rows = {row["source"]: row for row in persons.iter_rows(named=True)}
    summary = {}
    for name, source_dir in source_dirs.items():
        row = rows.get(name, {})
        summary[name] = {
            "data_dir": relative(source_dir),
            "years": row.get("years", []),
            "persons": row.get("persons", 0),
            "households": row.get("households", 0),
            "trips": n_trips.get(name, 0),
        }
    return summary


def write(output: Path, **fields) -> Path:
    """Write ``metadata.json`` to *output*: the command, code version and
    end time, plus *fields* (e.g. the start time, options and sources)."""
    meta = {
        "created": now(),
        **fields,
        "command": command(),
        "code": code_version(),
    }
    path = Path(output) / FILENAME
    path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return path
