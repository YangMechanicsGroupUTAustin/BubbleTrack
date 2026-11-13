"""Review's result edits: Accept (A), Accept all (Shift+A), No bubble (N, Shift+N).

Each is one undoable edit in the shared ``ResultHistory`` (Ctrl+Z, like a
fix).  Row actions touch exactly the row's frames ("Accept all 23" means
those 23), never other rows' frames inside its span.  Nothing here deletes
a result: "no bubble" is the explicit way to say a frame is empty.  Each
function returns the status text, or None if nothing changed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from bubbletrack.controller import edge_support, suggestion_actions
from bubbletrack.controller.display_mixin import display_frame, refresh_chart
from bubbletrack.controller.review_segments import ReviewSegment
from bubbletrack.controller.suggestion_actions import UNDO_HINT
from bubbletrack.event_bus import RESULTS_CHANGED
from bubbletrack.model.result_history import record_edit, snapshot
from bubbletrack.model.suggestion import shown_suggestion
from bubbletrack.ui.review_text import span_text

__all__ = ["UNDO_HINT", "accept_frame", "accept_segment", "is_empty", "mark_no_bubble"]


def is_empty(state, frame: int) -> bool:
    """No result and no guess on *frame*: nothing to accept."""
    return not state.radius[frame] > 0 and shown_suggestion(state, frame) is None


def accept_frame(app, frame: int) -> str | None:
    """Accept *frame*: the guess it shows (that guess only), else its result,
    locked against Fit All.  An empty frame means "no bubble" (as N)."""
    state = app._get_state()
    if state.fit_locked is None:
        return None
    if is_empty(state, frame):
        return mark_no_bubble(app, [frame], f"Frame {frame + 1}")
    promotion = suggestion_actions.accept_shown_guess(state, frame)
    if promotion is not None:
        app._on_lock_range(frame, frame, locked=True)  # chart, autosave
        return suggestion_actions.finish_promotion(app, promotion)
    before = snapshot(state, [frame])
    state.fit_locked[frame] = True  # fit_source kept
    edit = record_edit(state, f"accept of frame {frame + 1}", before)
    if edit is None:
        return None  # already accepted
    app._on_lock_range(frame, frame, locked=True)  # chart, autosave
    app.history.push(edit)
    return f"Frame {frame + 1} accepted and locked{UNDO_HINT}"


def accept_segment(app, segment: ReviewSegment) -> str | None:
    """Accept every frame of *segment*: each guess shown becomes the frame's
    result, every frame is locked; frames with nothing to accept are
    "no bubble" (locked empty)."""
    state = app._get_state()
    locked = state.fit_locked
    if locked is None:
        return None
    frames = list(segment.frames)
    before = snapshot(state, frames)
    empty = [f for f in frames if is_empty(state, f)]
    stored = [f for f in suggestion_actions.guessed_frames(state, segment)
              if edge_support.accept_suggestion(state, f)]
    locked[frames] = True  # fit_source kept
    edit = record_edit(state, f"accept of {span_text(segment).lower()}", before)
    if edit is None:
        return None
    app.history.push(edit)
    _changed(app, state, stored)
    text = f"{span_text(segment)} accepted and locked"
    notes = ([f"{len(stored)} of Auto's guesses stored"] if stored else []) + \
        ([f"{len(empty)} empty frame(s) marked as no bubble"] if empty else [])
    return text + (f" ({'; '.join(notes)})" if notes else "") + UNDO_HINT


def mark_no_bubble(app, frames: Iterable[int], what: str) -> str | None:
    """No bubble on *frames* ("Frame 5" / "Frames 10–44" as *what*): cleared
    and locked empty; frames locked with a result are kept."""
    state = app._get_state()
    frames = sorted(set(frames))
    if state.radius is None or not frames:
        return None
    before = snapshot(state, frames)
    for start, end in _runs(frames):
        app._on_mark_no_bubble(start, end)
    edit = record_edit(state, f"no-bubble mark of {what.lower()}", before)
    if edit is None:
        return None  # e.g. every frame was locked with a result (its status says so)
    app.history.push(edit)
    return f"{what} marked as no bubble{UNDO_HINT}"


def _runs(frames: Sequence[int]) -> list[tuple[int, int]]:
    """Sorted *frames* as (start, end) runs of consecutive frames."""
    runs: list[tuple[int, int]] = []
    for f in frames:
        if runs and f == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], f)
        else:
            runs.append((f, f))
    return runs


def _changed(app, state, stored: Sequence[int]) -> None:
    """Chart, autosave, queue (RESULTS_CHANGED), and the frame on screen if a
    guess there became its result."""
    refresh_chart(state, app.w)
    if state.image_no in stored:
        display_frame(state, app.w, state.image_no, app._set_state, app._image_cache)
    app.bus.emit(RESULTS_CHANGED)
