"""Accept in Review and Auto's guesses: a guess becomes a result once accepted.

* Accept (A) on a frame showing a guess stores that guess only (an Auto
  result, locked: the user accepted it) as one undoable edit in the shared
  history.
* Accept all (Shift+A) stores every guess of the row; it asks first when
  some of them were never on screen (``ReviewController``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from bubbletrack.controller import edge_support
from bubbletrack.controller.display_mixin import display_frame
from bubbletrack.model.result_history import FrameResult, record_edit, snapshot
from bubbletrack.model.suggestion import shown_suggestion

UNDO_HINT = " — Ctrl+Z undoes"  # every Review edit is one Ctrl+Z


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


def finish_promotion(app, promotion: Promotion) -> str:
    """After the Accept: push its edit, redraw, and return the status text."""
    state = app._get_state()
    app.history.push(record_edit(state, promotion.label, promotion.before))
    if state.image_no == promotion.frame:
        display_frame(state, app.w, state.image_no, app._set_state, app._image_cache)
    return (f"Frame {promotion.frame + 1}: Auto's guess accepted "
            f"(R = {state.radius[promotion.frame]:.1f} px) and locked{UNDO_HINT}")
