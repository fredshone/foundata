"""Collect per-source, per-step log values and render them as one table.

Pipeline steps call `record` (for a short tabular value, typically the % of
plans removed) or `note` (for a free-text warning). While a source is active
(inside `with source(name):`) these are collected rather than logged:
`render` builds a single summary table (logged at INFO at the end of a run)
and `log_notes` then logs each note at its own level, tagged with its source.

Outside an active source (e.g. calling a filter directly, or on the combined
data after concatenation) messages are logged straight away instead: record messages at
INFO and notes at their given level.
"""

from contextlib import contextmanager

from loguru import logger

from foundata.tables import render_markdown_table


class TableLogger:
    def __init__(self):
        self.reset()

    def reset(self):
        self.rows: dict[str, dict[str, str]] = {}
        self.notes: dict[str, list[str]] = {}
        self.steps: list[str] = []
        self.current: str | None = None

    @contextmanager
    def source(self, name: str):
        """Collect records and notes under *name* for the duration."""
        previous = self.current
        self.current = name
        self.rows.setdefault(name, {})
        self.notes.setdefault(name, [])
        try:
            yield self
        finally:
            self.current = previous

    def record(self, step: str, value, message: str | None = None):
        """Record a short *value* for *step* under the active source.

        With no active source, log *message* at INFO instead (if given).
        """
        if self.current is None:
            if message:
                logger.info(message)
            return
        if step not in self.steps:
            self.steps.append(step)
        self.rows[self.current][step] = str(value)

    def note(self, message: str, level: str = "WARNING"):
        """Collect a note under the active source, else log it at *level*."""
        if self.current is None:
            logger.log(level, message)
            return
        self.notes[self.current].append((level, message))

    def render(self) -> str:
        """Render collected records as a markdown table."""
        if not self.rows:
            return ""
        headers = ["source", *self.steps]
        rows = [
            [name, *(values.get(step, "-") for step in self.steps)]
            for name, values in self.rows.items()
        ]
        return render_markdown_table(headers, rows)

    def log_notes(self):
        """Log each collected note at its own level, prefixed by its source."""
        logger.debug("processing notes collected during run:")
        for name, msgs in self.notes.items():
            for level, msg in msgs:
                logger.log(level, f"{name}: {msg}")


LOGGER = TableLogger()

source = LOGGER.source
record = LOGGER.record
note = LOGGER.note
render = LOGGER.render
log_notes = LOGGER.log_notes
reset = LOGGER.reset


def pct(nn: int, n: int) -> str:
    """Format nn/n as a percentage string, '0.0%' when n is zero."""
    return f"{100 * nn / n:.1f}%" if n else "0.0%"
