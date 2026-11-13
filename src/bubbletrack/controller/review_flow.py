"""After a Review action: move on by itself, say what happened, offer Undo.

* Moving on: the row's next frame that still needs a look; once the row
  is done, the next row down the list, or the top row when worse rows still
  wait above (the list is sorted likely-wrong first).
* Feedback: one line in the panel ("✓ Frame 124 accepted and locked."),
  with an Undo that undoes exactly that action: the history entry it pushed
  (or the skip).  Once anything else changes the history (Ctrl+Z, another
  edit, a re-track), the line goes: its Undo would undo something else.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from bubbletrack.controller.review_queue import next_after_action
from bubbletrack.controller.review_segments import ReviewSegment, next_in_segment, segment_of
from bubbletrack.controller.suggestion_actions import UNDO_HINT
from bubbletrack.ui.review_panel import ReviewPanel

NEXT_ROW, ALL_DONE = "next row", "all done"  # where an action moved on to
_ENDINGS = {NEXT_ROW: ". That row is done.", ALL_DONE: ". All done."}
_OK = "#10b981"


class ReviewFlow:
    """Moving on and the feedback line of one Review dock (see module doc)."""

    def __init__(self, app, panel: ReviewPanel, go: Callable[[int], None],
                 unskip: Callable[[Sequence[int]], None]) -> None:
        self._app = app
        self._panel = panel
        self._go = go  # show a frame
        self._unskip = unskip  # put skipped frames back into the queue
        self._edit: object | None = None  # the history entry the feedback's Undo undoes
        self._skip: tuple[int, ...] = ()  # or the frames it un-skips
        self._frame: int | None = None  # where Undo returns to
        panel.undo_requested.connect(self.undo)

    # -- moving on --------------------------------------------------------- #

    def move_on(self, old: list[int], segment: ReviewSegment | None, frame: int | None) -> str:
        """After an action on *frame* of *segment* (frame None: on the whole
        row); *old* is the queue order before it.  Returns "", NEXT_ROW or ALL_DONE."""
        if segment is None:
            return ""
        panel = self._panel
        new = panel.frames()
        # After a row action, frames it left (e.g. empty ones after Accept
        # all) are next; after a frame action, the row's next frame.
        after = segment.start - 1 if frame is None else frame
        target = next_in_segment(segment, after, set(new))
        moved = ""
        if target is None:
            target = next_after_action(old, segment.end, new)
            if target is None:  # nothing left on show (other kinds may be hidden)
                return ALL_DONE if panel.count() == 0 else ""
            moved = NEXT_ROW
            rows = panel.segments()
            if rows[0].tier != segment_of(rows, target).tier:
                target = rows[0].entry  # worse rows wait above
        if panel.select_frame(target, keep_row=False):
            self._go(target)
        return moved

    # -- feedback ---------------------------------------------------------- #

    def report(self, status: str, moved: str, *, acted_on: int,
               skipped: Sequence[int] = ()) -> None:
        """Say what an action did, in the panel (with Undo) and in the header."""
        self._skip, self._frame = tuple(skipped), acted_on
        self._edit = None if self._skip else self._app.history.peek()
        self._panel.show_feedback(status.replace(UNDO_HINT, "") + _ENDINGS.get(moved, "."),
                                  undo=bool(self._skip) or self._edit is not None)
        self._app.w.header.set_status(status, _OK)

    def hint(self, text: str) -> None:
        """A short note in the feedback line (nothing was done: no Undo)."""
        self.clear()
        self._panel.show_hint(text)

    def check_stale(self) -> None:
        """Drop the line once the history no longer ends with its action."""
        if self._edit is not None and self._app.history.peek() is not self._edit:
            self.clear()

    def clear(self) -> None:
        self._edit, self._skip, self._frame = None, (), None
        self._panel.clear_feedback()

    def undo(self) -> None:
        """The line's Undo: exactly the action it describes, then show its frame.

        An edit is undone like Ctrl+Z (the shared history), which shows the
        restored frame itself (``RESULT_UNDONE``); a skip comes back here.
        """
        frame, skip, edit = self._frame, self._skip, self._edit
        self.clear()
        if skip:
            self._unskip(skip)
            self._app.w.header.set_status("Skip undone", _OK)
            if frame is not None:
                self._panel.select_frame(frame, keep_row=False)
                self._go(frame)
        elif edit is not None and self._app.history.peek() is edit:
            self._app.circle_ctrl.undo()  # the shared history's Ctrl+Z
