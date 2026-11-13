"""Review's result edits: Accept (A), Accept all (Shift+A), No bubble (N, Shift+N).

Each is one undoable edit in the shared ``ResultHistory`` (Ctrl+Z, like a
fix); the chart, the autosave and the queue follow through the
AppController's range handlers (``_on_lock_range``, ``_on_mark_no_bubble``).
Nothing here deletes a result: "no bubble" is the explicit way to say a
frame is empty.  Each function returns the status text, or None if nothing
changed.
"""

from __future__ import annotations

from bubbletrack.controller import edge_support, suggestion_actions
from bubbletrack.controller.display_mixin import display_frame
from bubbletrack.controller.review_segments import ReviewSegment
from bubbletrack.model.result_history import record_edit, snapshot
from bubbletrack.ui.review_text import span_text

UNDO_HINT = " — Ctrl+Z undoes"


def accept_frame(app, frame: int) -> str | None:
    """Accept *frame*: the guess it shows (that guess only), else its result
    (or, without one, "no bubble here"), locked against Fit All."""
    state = app._get_state()
    if state.fit_locked is None:
        return None
    promotion = suggestion_actions.accept_shown_guess(state, frame)
    if promotion is not None:
        app._on_lock_range(frame, frame, locked=True)  # chart, autosave
        return suggestion_actions.finish_promotion(app, promotion)
    before = snapshot(state, [frame])
    state.fit_locked[frame] = True  # fit_source kept; an empty frame stays empty
    app._on_lock_range(frame, frame, locked=True)
    app.history.push(record_edit(state, f"accept of frame {frame + 1}", before))
    what = "accepted and locked" if state.radius[frame] > 0 else "accepted: no bubble (locked)"
    return f"Frame {frame + 1} {what}{UNDO_HINT}"


def accept_segment(app, segment: ReviewSegment) -> str | None:
    """Accept every frame of *segment*: each guess shown becomes the frame's
    result, every listed frame and the span's fitted frames are locked."""
    state = app._get_state()
    locked = state.fit_locked
    if locked is None:
        return None
    before = snapshot(state, range(segment.start, segment.end + 1))
    stored = [f for f in suggestion_actions.guessed_frames(state, segment)
              if edge_support.accept_suggestion(state, f)]
    locked[list(segment.frames)] = True  # with or without a result, fit_source kept
    app._on_lock_range(segment.start, segment.end, locked=True)  # chart, autosave
    app.history.push(record_edit(state, f"accept of {span_text(segment).lower()}", before))
    if state.image_no in stored:  # its guess is now the result on screen
        display_frame(state, app.w, state.image_no, app._set_state, app._image_cache)
    text = f"{span_text(segment)} accepted and locked"
    if stored:
        text += f" ({len(stored)} of Auto's guesses stored)"
    return text + UNDO_HINT


def mark_no_bubble(app, start: int, end: int, what: str) -> str | None:
    """No bubble on frames *start*..*end* ("Frame 5" / "Frames 10–44" as *what*):
    cleared and locked empty; frames locked with a result are kept."""
    state = app._get_state()
    if state.radius is None:
        return None
    before = snapshot(state, range(start, end + 1))
    app._on_mark_no_bubble(start, end)
    edit = record_edit(state, f"no-bubble mark of {what.lower()}", before)
    if edit is None:
        return None  # e.g. every frame was locked with a result (its status says so)
    app.history.push(edit)
    return f"{what} marked as no bubble{UNDO_HINT}"
