"""Preprocess the WorldTensor layers joined to locations (``foundata.gridjoin``).

Writes each variable's most recent year to the cache as a ``.npy``
(about 4 MB each), redoing one only when its source file changes. ``run``
and ``batch`` do this themselves when needed; this is for doing it ahead.

Usage:
    uv run python scripts/preprocess_grids.py [--world DIR] [--cache DIR] [--force]
"""

import argparse
from pathlib import Path

from foundata import log
from foundata.gridjoin import DEFAULT_CACHE, DEFAULT_WORLD, worldtensor

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--world", type=Path, default=DEFAULT_WORLD)
parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
parser.add_argument(
    "--force", action="store_true", help="redo even if up to date"
)
args = parser.parse_args()

log.setup("INFO")
worldtensor.preprocess(args.world, args.cache, force=args.force)
