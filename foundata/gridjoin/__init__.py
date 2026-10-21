"""Attach WorldTensor's gridded data to locations, and from them to
households, by cell index on its regular 0.25 degree grid. See
docs/gridjoin.md."""

from pathlib import Path

DEFAULT_WORLD = Path.home() / "Data" / "foundata" / "WORLD"
DEFAULT_CACHE = Path.home() / "Data" / "foundata" / "cache" / "gridjoin"

from foundata.gridjoin import join, worldtensor  # noqa: E402
