"""What the Review header and its empty page say (pure).

Frames in red and amber rows are *to check*; grey rows are *for
information* and counted apart.  Progress counts from the frames listed
since the last fit; skipped frames are never counted as checked.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from bubbletrack.controller.review_segments import ReviewSegment
from bubbletrack.ui.review_text import done_text, summary_text

CHECK_TIERS = frozenset({"high", "medium"})  # red and amber rows
EMPTY_BEFORE_FIT = "Frames to check show up here after a fit."
EMPTY_ALL_CLEAR = "Nothing to check: Auto is sure of every frame.\nNext: export R(t)."
EMPTY_ALL_DONE = done_text(0)
EMPTY_FILTERED = "Nothing of the selected types."
NOTHING_YET = "Nothing to check yet"


def check_frames(segments: Sequence[ReviewSegment]) -> set[int]:
    """Frames listed in a row to check (red or amber)."""
    return {f for s in segments if s.tier in CHECK_TIERS for f in s.frames}


class ReviewProgress:
    """Frames listed (and listed to check) since the last fit."""

    def __init__(self) -> None:
        self.listed: set[int] = set()
        self.checks: set[int] = set()

    def restart(self, segments: Sequence[ReviewSegment] = ()) -> None:
        """Count from the frames *segments* list now (a fit, another recording)."""
        self.listed = {f for s in segments for f in s.frames}
        self.checks = check_frames(segments)

    def note(self, segments: Sequence[ReviewSegment]) -> None:
        """The queue changed: remember every frame it listed."""
        self.listed |= {f for s in segments for f in s.frames}
        self.checks |= check_frames(segments)

    def progress(self, segments: Sequence[ReviewSegment]) -> tuple[int, int]:
        """(frames still to check, frames to check since the last fit)."""
        return len(check_frames(segments)), len(self.checks)

    def summary(self, segments: Sequence[ReviewSegment], skipped: Collection[int], *,
                fitted: bool) -> str:
        """'12 of 48 frames to check · 11 for information'."""
        if not (fitted or segments):
            return NOTHING_YET
        left, total = self.progress(segments)
        info = sum(s.count for s in segments if s.tier not in CHECK_TIERS)
        return summary_text(left, total, info, skipped=len(self.checks & set(skipped)))

    def empty_page(self, segments: Sequence[ReviewSegment], skipped: Collection[int], *,
                   fitted: bool) -> tuple[str, bool]:
        """(message, offer Export) for the list's place when no row is on show."""
        if segments:
            return EMPTY_FILTERED, False
        if self.listed:
            return done_text(len(self.listed & set(skipped))), True
        if fitted:
            return EMPTY_ALL_CLEAR, True
        return EMPTY_BEFORE_FIT, False
