"""What Review remembers between refreshes (pure): merged lists, skips, guesses seen.

* Extra items (e.g. the sequence solver's) hold for the result they were
  made from; once their frame gets another result they drop out.
* Skipped frames stay out of the queue until their result changes.
* A guess counts as looked at once its frame was on screen while it showed
  that very guess (Accept all asks about the others).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from bubbletrack.controller.review_queue import ReviewItem, build_review_items
from bubbletrack.model.state import AppState
from bubbletrack.model.suggestion import shown_suggestion


def result_key(state: AppState, frame: int) -> tuple:
    """What identifies frame *frame*'s current result (radius, confidence, source)."""
    conf = state.fit_confidence
    c = float(conf[frame]) if conf is not None else float("nan")
    source = int(state.fit_source[frame]) if state.fit_source is not None else 0
    return float(state.radius[frame]), (c if np.isfinite(c) else None), source


class ReviewMemory:
    """Extra items, skipped frames and seen guesses of one loaded source."""

    def __init__(self) -> None:
        self._extra: tuple[ReviewItem, ...] = ()
        self._extra_keys: dict[int, tuple] = {}  # frame -> result the item was made for
        self._dismissed: dict[int, tuple] = {}  # frame -> result it was skipped with
        self._seen: dict[int, object] = {}  # frame -> the guess it showed on screen

    def clear(self) -> None:
        """Forget everything (another source was loaded)."""
        self._extra, self._extra_keys = (), {}
        self._dismissed.clear()
        self._seen.clear()

    # -- the queue --------------------------------------------------------- #

    def set_extra(self, state: AppState, items: Iterable[ReviewItem]) -> None:
        n = 0 if state.radius is None else len(state.radius)
        self._extra = tuple(item for item in items if 0 <= item.frame < n)
        self._extra_keys = {item.frame: result_key(state, item.frame) for item in self._extra}

    def items(self, state: AppState) -> list[ReviewItem]:
        """The queue for the current results: extra items still valid, no skips."""
        n = 0 if state.radius is None else len(state.radius)
        extra = [item for item in self._extra if item.frame < n
                 and self._extra_keys.get(item.frame) == result_key(state, item.frame)]
        return [item for item in build_review_items(state, extra)
                if not self.is_dismissed(state, item.frame)]

    # -- skips ------------------------------------------------------------- #

    def dismiss(self, state: AppState, frames: Iterable[int]) -> None:
        for f in frames:
            self._dismissed[f] = result_key(state, f)

    def undismiss(self, frames: Iterable[int]) -> None:
        for f in frames:
            self._dismissed.pop(f, None)

    def is_dismissed(self, state: AppState, frame: int) -> bool:
        """Skipped frames stay hidden until their result changes."""
        key = self._dismissed.get(frame)
        return key is not None and key == result_key(state, frame)

    # -- guesses looked at ---------------------------------------------------- #

    def note_shown(self, state: AppState, frame: int) -> None:
        guess = shown_suggestion(state, frame)
        if guess is not None:
            self._seen[frame] = guess

    def was_seen(self, state: AppState, frame: int) -> bool:
        """True if the guess *frame* shows now was on screen (or is on screen)."""
        guess = shown_suggestion(state, frame)
        return frame == state.image_no or (guess is not None and self._seen.get(frame) is guess)
