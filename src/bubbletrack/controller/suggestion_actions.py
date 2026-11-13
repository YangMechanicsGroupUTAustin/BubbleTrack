"""Accept (A) in Review and Auto's guesses: a guess becomes a result only once seen.

* The frame on screen shows a guess: A stores that guess only (an Auto
  result, locked: the user accepted it) as one undoable edit in the shared
  history, then Review shows the row's next frame with a guess.
* The frame on screen has none: A accepts the row as before, but leaves
  every frame that shows a guess alone (each needs its own look).

Guesses of frames the user has not displayed are never stored.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from bubbletrack.controller import edge_support
from bubbletrack.controller.display_mixin import display_frame
from bubbletrack.model.result_history import FrameResult, record_edit, snapshot
from bubbletrack.model.suggestion import shown_suggestion


@dataclass(frozen=True)
class Promotion:
    """The guess stored for one frame, and the frame's result before it."""

    frame: int
    before: Mapping[int, FrameResult]

    @property
    def label(self) -> str:
        return f"accept of Auto's guess on frame {self.frame + 1}"


def accept_shown_guess(state, frame: int | None) -> Promotion | None:
    """Store the guess of *frame* (the one on screen); None if it shows none."""
    if frame is None or shown_suggestion(state, frame) is None:
        return None
    before = snapshot(state, [frame])
    if not edge_support.accept_suggestion(state, frame):
        return None
    return Promotion(frame, before)


def guessed_frames(state, segment) -> tuple[int, ...]:
    """*segment*'s listed frames that show a guess, in frame order."""
    return tuple(f for f in segment.frames if shown_suggestion(state, f) is not None)


def next_guess(state, segment, frame: int) -> int | None:
    """The row's next frame with a guess after *frame* (then from its start)."""
    others = [f for f in guessed_frames(state, segment) if f != frame]
    return next((f for f in others if f > frame), others[0] if others else None)


def finish_promotion(app, promotion: Promotion) -> str:
    """After the Accept: push its edit, redraw, and return the status text."""
    state = app._get_state()
    app.history.push(record_edit(state, promotion.label, promotion.before))
    if state.image_no == promotion.frame:
        display_frame(state, app.w, state.image_no, app._set_state, app._image_cache)
    return (f"Frame {promotion.frame + 1}: Auto's guess accepted "
            f"(R = {state.radius[promotion.frame]:.1f} px) and locked — Ctrl+Z undoes")
