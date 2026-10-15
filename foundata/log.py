"""Loguru configuration for foundata.

Library modules log through ``from loguru import logger``; this module only
configures where those logs go. The CLI calls `setup` once at start-up.

Levels used across the pipeline:

- DEBUG: fine-grained detail (sub-step progress, per-record fixes, stats)
- INFO: progress through sources and steps, written outputs, report tables
- WARNING: data quality problems worth a look, but the run continues
- ERROR: checks that failed (a run may raise shortly after)
"""

import sys

from loguru import logger

FORMAT = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
    "<level>{message}</level>"
)


def setup(level: str = "INFO") -> None:
    """Replace loguru's default handler with a concise stderr handler."""
    logger.remove()
    logger.add(sys.stderr, level=level.upper(), format=FORMAT)
