"""The line a finished Fit leaves in the header, in plain words.

    Done: 253 of 256 frames measured · 15 to check in Review

Green unless something failed.  Problems (frames that could not be read, a
failed autosave, tracking that failed) follow after a dash and turn it amber;
routine notes (what tracking re-measured, low-confidence fits, ML assist's
counts) are details: the line's tooltip and the log.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bubbletrack.ui.wording import plural

OK_COLOUR, WARN_COLOUR = "#10b981", "#f59e0b"


@dataclass(frozen=True)
class RunOutcome:
    """What a finished (or stopped) run did."""

    total: int                 # frames the run was asked to measure
    measured: int              # of those, frames with a result now
    stopped: bool = False
    frames_done: int = 0       # frames processed before Stop
    ml: bool = False           # measured with ML assist
    to_check: int = 0          # frames in Review rows to check (red or amber)
    problems: tuple[str, ...] = field(default_factory=tuple)
    details: tuple[str, ...] = field(default_factory=tuple)


def summary_line(outcome: RunOutcome) -> tuple[str, str, str]:
    """(text, colour, details) for the header."""
    if outcome.stopped:
        text = f"Stopped after {plural(outcome.frames_done, 'frame')}"
    else:
        how = " with ML assist" if outcome.ml else ""
        text = f"Done{how}: {outcome.measured} of {plural(outcome.total, 'frame')} measured"
    if outcome.to_check:
        text += f" · {outcome.to_check} to check in Review"
    if outcome.problems:
        text += " — " + "; ".join(outcome.problems)
    warn = outcome.stopped or bool(outcome.problems)
    return text, WARN_COLOUR if warn else OK_COLOUR, "\n".join(outcome.details)
