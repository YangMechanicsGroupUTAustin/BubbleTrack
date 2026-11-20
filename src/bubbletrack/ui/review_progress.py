"""What the Review header and its empty page say (pure).

Every listed frame is a frame *to check*, whatever its row's dot colour:
it stays listed until the user settles it, so the header, the status line
and ML assist all count the same frames.  Progress counts from the frames
listed since the last fit; skipped frames are never counted as checked.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from bubbletrack.controller.review_segments import ReviewSegment
from bubbletrack.ui.review_text import done_text, summary_text

EMPTY_BEFORE_FIT = "Frames to check show up here after a fit."
EMPTY_ALL_CLEAR = "Nothing to check: Auto is sure of every frame.\nNext: export R(t)."
EMPTY_FILTERED = "Nothing of the selected types."
NOTHING_YET = "Nothing to check yet"


def check_frames(segments: Sequence[ReviewSegment]) -> set[int]:
    """The frames to check: every frame the rows list."""
    return {f for s in segments for f in s.frames}


class ReviewProgress:
    """Frames listed since the last fit."""

    def __init__(self) -> None:
        self.listed: set[int] = set()

    def restart(self, segments: Sequence[ReviewSegment] = ()) -> None:
        """Count from the frames *segments* list now (a fit, another recording)."""
        self.listed = check_frames(segments)

    def note(self, segments: Sequence[ReviewSegment]) -> None:
        """The queue changed: remember every frame it listed."""
        self.listed |= check_frames(segments)

    def progress(self, segments: Sequence[ReviewSegment]) -> tuple[int, int]:
        """(frames still to check, frames to check since the last fit)."""
        return len(check_frames(segments)), len(self.listed)

    def summary(self, segments: Sequence[ReviewSegment], skipped: Collection[int], *,
                fitted: bool) -> str:
        """'12 of 48 frames to check'."""
        if not (fitted or segments):
            return NOTHING_YET
        left, total = self.progress(segments)
        return summary_text(left, total, skipped=len(self.listed & set(skipped)))

    def empty_page(self, segments: Sequence[ReviewSegment], skipped: Collection[int], *,
                   fitted: bool) -> tuple[str, bool]:
        """(message, offer Export) for the list's place when no row is on show."""
        if segments:
            return EMPTY_FILTERED, False
        if self.listed:
            left_out = sorted(self.listed & set(skipped))
            return done_text(len(self.listed) - len(left_out), len(left_out),
                             left_out[0] if left_out else None), True
        if fitted:
            return EMPTY_ALL_CLEAR, True
        return EMPTY_BEFORE_FIT, False
