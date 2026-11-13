"""Accept (A) in Review on frames that show Auto's guess: store the guesses.

``promote_guesses`` turns every listed frame's shown guess into its result
(an Auto result, locked: the user accepted it) before the row is locked;
``finish_promotion`` records the whole Accept as one undoable edit in the
shared history (Ctrl+Z puts the guesses and the unlocked frames back) and
redraws the frame on screen.  Frames without a guess keep today's meaning.
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
    """Guesses stored by one Accept, and the span's results before it."""

    frames: tuple[int, ...]
    before: Mapping[int, FrameResult]

    @property
    def label(self) -> str:
        if len(self.frames) == 1:
            return f"accept of Auto's guess on frame {self.frames[0] + 1}"
        return f"accept of Auto's guesses on {len(self.frames)} frames"


def promote_guesses(state, segment) -> Promotion | None:
    """Store the shown guesses of *segment*'s listed frames; None if it has none.

    The snapshot covers the segment's whole span, so the edit also holds
    the locks the Accept adds afterwards.
    """
    guessed = [f for f in segment.frames if shown_suggestion(state, f) is not None]
    if not guessed:
        return None
    before = snapshot(state, range(segment.start, segment.end + 1))
    stored = tuple(f for f in guessed if edge_support.accept_suggestion(state, f))
    return Promotion(stored, before) if stored else None


def finish_promotion(app, promotion: Promotion, text: str) -> str:
    """After the Accept: push its edit, redraw, and return the status text."""
    state = app._get_state()
    app.history.push(record_edit(state, promotion.label, promotion.before))
    if state.image_no in promotion.frames:
        display_frame(state, app.w, state.image_no, app._set_state, app._image_cache)
    if len(promotion.frames) == 1:
        frame = promotion.frames[0]
        text = (f"Frame {frame + 1}: Auto's guess accepted "
                f"(R = {state.radius[frame]:.1f} px) and locked")
    else:
        text += f"; Auto's guess stored on {len(promotion.frames)} frames"
    return text + " — Ctrl+Z undoes"
