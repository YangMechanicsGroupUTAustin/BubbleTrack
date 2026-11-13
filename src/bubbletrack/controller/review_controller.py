"""Review: which frames are likely wrong, and the actions that settle them.

The rules are pure (``review_queue`` lists frames, ``review_segments`` groups
them into rows, ``review_advice`` recommends an action).  ``ReviewController``
wires them to the running app: it refreshes on bus events and edits, keeps
the R(t) chart showing the selected row, performs the actions
(``review_actions``; Fix through ``FixMode``) and then moves on by itself
(``review_flow``).  Nothing in Review deletes a result.

Actions wait while Fit All runs.  A background re-track does not block
them: an edit abandons it and restarts its countdown.  Ctrl+Z and the
feedback line's Undo both show the frame they restored.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence

import numpy as np
from PyQt6.QtCore import pyqtBoundSignal
from PyQt6.QtWidgets import QDockWidget

from bubbletrack.controller import review_actions, suggestion_actions
from bubbletrack.controller.circle_controller import RESULT_UNDONE
from bubbletrack.controller.fix_mode import FixMode
from bubbletrack.controller.review_advice import advise
from bubbletrack.controller.review_background import BackgroundCheck, before_bubble
from bubbletrack.controller.review_flow import ALL_DONE, ReviewFlow
from bubbletrack.controller.review_memory import ReviewMemory
from bubbletrack.controller.review_queue import ReviewItem, build_review_items, in_lifetime
from bubbletrack.controller.review_segments import ReviewSegment, build_segments
from bubbletrack.event_bus import RESULTS_CHANGED
from bubbletrack.model import ml_assist
from bubbletrack.model.result_history import ResultEdit
from bubbletrack.model.state import AppState
from bubbletrack.ui import review_confirm
from bubbletrack.ui.fix_strip import FixStrip
from bubbletrack.ui.review_card import CardAdvice
from bubbletrack.ui.review_panel import ReviewPanel
from bubbletrack.ui.review_text import accept_all_question, span_text
from bubbletrack.ui.shortcuts import FIX_BINDINGS, ReviewShortcuts

__all__ = ["ReviewController", "ReviewItem", "build_review_items", "install_review"]

logger = logging.getLogger(__name__)

_WARN = "#f59e0b"
_FRAME_ACTIONS = frozenset({"accept", "fix", "no_bubble"})
NOTHING_TO_ACCEPT = "Nothing to accept here — F to fix, N if there is no bubble"
NOTHING_TO_ACCEPT_IN_ROW = ("Nothing to accept in this row — F to fix a frame, "
                            "Shift+N if there is no bubble")
REVIEW_HIDDEN = "The Review panel is hidden: View ▸ Review shows it (its keys act there)"
REVIEW_BEHIND = "Review is behind another panel: click its Review tab (its keys act there)"
ML_LEAD_SHARE = 0.4  # above this share of the bubble's frames to check, suggest ML assist


class ReviewController:
    """Keeps the review panel in sync with the app and performs its actions."""

    def __init__(self, window, app, panel: ReviewPanel, dock: QDockWidget,
                 strip: FixStrip) -> None:
        self.w = window
        self._app = app
        self.panel = panel
        self.dock = dock
        self.memory = ReviewMemory()  # extra items, skips, guesses seen
        self.background = BackgroundCheck()  # which empty frames look like the reference
        self._running = False  # a run (Fit All or a re-track) writes results
        self._fit_run = False  # ... and it is Fit All: Review waits
        # (row, queue order) when a fix started inside the selected row
        self._fix_context: tuple[ReviewSegment, list[int]] | None = None
        self.fix = FixMode(app, strip, self._on_fixed)
        self.flow = ReviewFlow(app, panel, self._go, self._unskip)

        panel.set_advisor(self._advice)
        panel.set_row_counter(
            lambda segment: len(review_actions.acceptable(self._state(), segment.frames)))
        panel.frame_activated.connect(self._go)
        panel.action_requested.connect(self.perform)
        panel.segment_changed.connect(self._show_range)
        panel.export_requested.connect(lambda: self.w.export_requested.emit())
        app.bus.subscribe("run_state_changed", self._on_run_state)
        app.bus.subscribe("source_loaded", self._on_source_loaded)
        app.bus.subscribe("frame_changed", self._on_frame_changed)
        app.bus.subscribe("run_finished", self._on_run_finished)
        app.bus.subscribe(RESULTS_CHANGED, lambda *_: self.refresh())
        app.bus.subscribe(RESULT_UNDONE, self._on_undone)

        self.shortcuts = ReviewShortcuts(window, {
            **{name: (lambda n=name: self._key(lambda: self.perform(n)))
               for name in ReviewPanel.ACTIONS},
            "next": lambda: self._key(lambda: panel.step(1)),
            "prev": lambda: self._key(lambda: panel.step(-1)),
        }, is_active=self._shortcuts_active)
        self.fix_keys = ReviewShortcuts(window, {
            "fix_cancel": self.fix.cancel, "fix_done": self.fix.finish_points,
        }, is_active=self._fix_keys_active, bindings=FIX_BINDINGS)
        strip.active_changed.connect(lambda _on: self.fix_keys.sync())
        strip.active_changed.connect(panel.card.set_fixing)
        strip.active_changed.connect(self._on_fix_active)
        dock.visibilityChanged.connect(self._on_dock_visibility)
        self.refresh()

    # -- public API -------------------------------------------------------- #

    @property
    def count_changed(self) -> pyqtBoundSignal:
        """Signal(int): number of frames waiting for review."""
        return self.panel.count_changed

    @property
    def count(self) -> int:
        return self.panel.count()

    @property
    def check_count(self) -> int:
        """Frames in rows to check (red or amber), as Review's header counts them."""
        return len(self.panel.check_frames())

    def set_extra(self, items: Iterable[ReviewItem]) -> None:
        """Merge another review list (e.g. the sequence solver's) and refresh.

        Each item holds for the result it was made from: once its frame gets
        another result, the item is dropped.
        """
        self.memory.set_extra(self._state(), items)
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the queue from the current results (cheap; no-op if unchanged)."""
        state = self._state()
        self.panel.set_segments(build_segments(self._items(state)), state.images,
                                fitted=_has_results(state), skipped=self.memory.skipped(state))
        self.panel.set_fix_label("Fix with a circle" if self._app.circle_ctrl.available()
                                 else "Fix with edge points")
        self.panel.refresh_card()
        self.flow.check_stale()

    def queued_frames(self) -> frozenset[int]:
        """Every frame the queue lists now (all types; skipped ones left out)."""
        return frozenset(item.frame for item in self._items(self._state()))

    def looks_empty(self, frame: int) -> bool | None:
        """Frame *frame* looks like the reference frame (None: cannot tell)."""
        return self.background.for_state(self._state(), frame)

    def _items(self, state: AppState) -> list[ReviewItem]:
        return self.memory.items(state, before_bubble(state, self.looks_empty))

    def perform(self, action: str) -> None:
        """Apply *action* (see ``ReviewPanel.ACTIONS``), then move on by itself.

        Accept, Fix and No bubble act on the frame on screen; Accept all,
        No bubble on all and Skip on the selected row.
        """
        if action not in ReviewPanel.ACTIONS:
            raise ValueError(f"unknown review action {action!r}")
        frame = self._frame_on_screen()
        if self._fit_run or frame is None:
            return
        if action == "fix":
            self.start_fix(frame)
            return
        if not _has_results(self._state()):
            return  # nothing to settle before a fit
        self.panel.select_frame(frame)  # the row of the frame on screen, if any
        segment = self.panel.current_segment()
        if segment is None and action in ReviewPanel.ROW_ACTIONS:
            return
        hint = self._nothing_to_accept(action, segment, frame)
        if hint:
            self.flow.hint(hint)  # Accept never means "no bubble"
            return
        logger.info("Review: %s on frame %d (%s)", action, frame + 1,
                    span_text(segment) if segment is not None else "no row")
        old = self.panel.frames()
        status = self._act(action, segment, frame)
        if status is None:
            return  # nothing happened (e.g. Accept all cancelled): stay
        self.fix.end()  # e.g. N after a refused circle: the user decided
        self.refresh()
        on_frame = action in _FRAME_ACTIONS
        moved = self.flow.move_on(old, segment, frame if on_frame else None)
        self.flow.report(status, moved if on_frame or moved == ALL_DONE else "",
                         acted_on=frame if on_frame else segment.start,
                         skipped=segment.frames if action == "skip" else ())

    def start_fix(self, frame: int) -> bool:
        """Fix *frame* (0-based) with the Review dock's fix tools.

        Shows the frame, opens the dock and starts "Drag a circle" (or
        "Click edge points" outside the Auto engine).  Once a fix is stored,
        Review moves on as after Accept when the frame lies in the selected
        row.  False if no fix can start (no images, Fit All running, out of
        range).  ``install_review`` connects ``RadiusChart.fix_requested``
        ("Fix frame N...") here.
        """
        state = self._state()
        if self._fit_run:
            self.w.header.set_status("Wait for the running fit to finish", _WARN)
            return False
        if not state.images or state.radius is None or not 0 <= frame < len(state.radius):
            return False
        if self.dock.isHidden():
            self.dock.show()
        self.dock.raise_()
        if state.image_no != frame:
            self._go(frame)
        segment = self.panel.current_segment()
        context = (segment, self.panel.frames()) \
            if segment is not None and segment.contains(frame) else None
        if not self.fix.start(frame):
            return False
        self._fix_context = context
        return True

    # -- actions ----------------------------------------------------------- #

    def _act(self, action: str, segment: ReviewSegment | None, frame: int) -> str | None:
        """Perform *action*; the status text, None if nothing happened."""
        app = self._app
        if action == "accept":
            return review_actions.accept_frame(app, frame)
        if action == "no_bubble":
            return review_actions.mark_no_bubble(app, [frame], f"Frame {frame + 1}")
        if action == "no_bubble_all":
            return review_actions.mark_no_bubble(app, segment.frames, span_text(segment))
        if action == "accept_all":
            if not self._confirm_accept_all(segment):
                return None
            return review_actions.accept_segment(app, segment)
        self.memory.dismiss(self._state(), segment.frames)  # skip
        return f"{span_text(segment)} skipped: left as it is"

    def _nothing_to_accept(self, action: str, segment: ReviewSegment | None,
                           frame: int) -> str:
        """The hint when Accept (all) would find nothing to store, else ''."""
        state = self._state()
        if action == "accept" and review_actions.is_empty(state, frame):
            return NOTHING_TO_ACCEPT
        if action == "accept_all" and not review_actions.acceptable(state, segment.frames):
            return NOTHING_TO_ACCEPT_IN_ROW
        return ""

    def _confirm_accept_all(self, segment: ReviewSegment) -> bool:
        """Ask once if Accept all would store guesses the user has not displayed."""
        state = self._state()
        unseen = [f for f in suggestion_actions.guessed_frames(state, segment)
                  if not self.memory.was_seen(state, f)]
        if not unseen:
            return True
        # A re-track must not land under the question (its results would be
        # accepted unseen); the edit restarts its countdown anyway.
        self._app.retrack.cancel()
        frames = len(review_actions.acceptable(state, segment.frames))
        if not review_confirm.ask_accept_all(
                self.w, accept_all_question(frames, len(unseen)),
                empty=segment.count - frames):
            return False
        return not self._fit_run and segment in self.panel.segments()

    def _on_fixed(self, frame: int) -> None:
        """A fix was stored on *frame*: move on as after Accept."""
        context, self._fix_context = self._fix_context, None
        state = self._state()
        status = (f"Frame {frame + 1} fixed: R = {state.radius[frame]:.1f} px"
                  f"{suggestion_actions.UNDO_HINT}")
        self.refresh()
        moved = self.flow.move_on(context[1], context[0], frame) if context is not None else ""
        self.flow.report(status, moved, acted_on=frame)

    def _unskip(self, frames: Sequence[int]) -> None:
        self.memory.undismiss(frames)
        self.refresh()

    # -- helpers ----------------------------------------------------------- #

    def _state(self) -> AppState:
        return self._app._get_state()

    def _frame_on_screen(self) -> int | None:
        state = self._state()
        n = 0 if state.radius is None else len(state.radius)
        return state.image_no if state.images and 0 <= state.image_no < n else None

    def _go(self, frame: int) -> None:
        self._app.file_ctrl.on_frame_changed(frame)

    def _advice(self, frame: int, item: ReviewItem | None) -> CardAdvice:
        advice = advise(self._state(), frame, item, self.looks_empty)
        return CardAdvice(advice.text, advice.action, advice.guess, advice.can_accept)

    def _key(self, act: Callable[[], None]) -> None:
        """A review key: act while the dock shows, else say where it is."""
        if self.dock.isVisible():
            act()
        elif not self.w.isMinimized():
            self.w.header.set_status(REVIEW_HIDDEN if self.dock.isHidden() else REVIEW_BEHIND,
                                     _WARN)

    def _show_range(self, segment: ReviewSegment | None) -> None:
        """Shade the selected run on the R(t) chart while the dock is open.

        A shading of its own, not the chart's range selection: the user's
        brushed range and the point menus inside a run stay as they were.
        """
        shown = segment is not None and not segment.is_single and not self.dock.isHidden()
        self.w.radius_chart.set_review_span((segment.start, segment.end) if shown else None)

    def _shortcuts_active(self) -> bool:
        return not self._fit_run  # a hidden dock answers with a hint (``_key``)

    def _fix_keys_active(self) -> bool:
        return self.fix.active and self.dock.isVisible() and not self._fit_run

    # -- bus events -------------------------------------------------------- #

    def _on_run_state(self, running: bool) -> None:
        was_fit = self._fit_run
        self._running = bool(running)
        self._fit_run = self._running and self._app.auto_ctrl.is_busy()
        self.panel.setEnabled(not self._fit_run)
        self.fix.strip.setEnabled(not self._fit_run)  # FixMode ends a fix on Fit All
        if not self._running:
            self.refresh()
            self.panel.select_frame(self._frame_on_screen())  # a run moves the view
            if was_fit:  # new results: progress counts from here
                self.panel.reset_progress()
                self.flow.clear()
        self.shortcuts.sync()
        self.fix_keys.sync()

    def _on_run_finished(self, info: dict) -> None:
        """After a Fit of every frame, start at the top of the list; when
        much of the bubble needs a check, lead with ML assist."""
        state = self._state()
        if len(info.get("frames", ())) != len(state.images or ()):
            return  # a range refit: stay where the user was
        rows = self.panel.segments()
        if rows and self.panel.select_frame(rows[0].entry, keep_row=False):
            summary = self.w.header.status()  # the run's "Done — ..." stays in view
            self._go(rows[0].entry)
            self.w.header.set_status(*summary)
        if info.get("edge") and not info.get("ml") and not info.get("stopped"):
            self._lead_with_ml(state)

    def _lead_with_ml(self, state: AppState) -> None:
        """Show ML assist's hint (above the rows) when more than
        ``ML_LEAD_SHARE`` of the bubble's frames wait for a check (and at
        least as many frames as ML assist's own hint needs)."""
        ml = getattr(self._app, "ml", None)
        if ml is None or not getattr(ml, "available", False) or state.ml_assist \
                or not ml.hint.isHidden():
            return
        inside, lifetime = in_lifetime(state, self.panel.check_frames())
        if inside >= ml_assist.HINT_MIN_FRAMES and inside > ML_LEAD_SHARE * lifetime:
            ml.hint.show_text(f"{round(100 * inside / lifetime)}% of the bubble's frames need "
                              "a check. ML assist can help: confirm a few frames and teach it.")

    def _on_undone(self, edit: ResultEdit) -> None:
        """Ctrl+Z (or the feedback's Undo) restored *edit*: show its frame."""
        self.flow.clear()
        frames, shown = edit.frames, self._frame_on_screen()
        if not frames or shown in frames:
            self.panel.select_frame(shown)
            return
        self.panel.select_frame(frames[0], keep_row=False)
        self._go(frames[0])

    def _on_fix_active(self, on: bool) -> None:
        if on:
            self.flow.clear()  # the last action's line (and its Undo) steps aside

    def _on_dock_visibility(self, _visible: bool) -> None:
        self.shortcuts.sync()
        self.fix_keys.sync()
        self._show_range(self.panel.current_segment())
        if self.fix.active and not self.dock.isVisible() and not self.w.isMinimized():
            self.fix.cancel()  # its strip is out of sight

    def _on_source_loaded(self, _n: int) -> None:
        self.memory.clear()
        self._fix_context = None
        self.flow.clear()
        self.refresh()
        self.panel.select_frame(self._frame_on_screen())
        self.panel.reset_progress()

    def _on_frame_changed(self, idx: int) -> None:
        self.memory.note_shown(self._state(), idx)
        if not self._running:
            self.refresh()
        self.panel.select_frame(idx)  # keeps the row while its span holds idx


def _has_results(state: AppState) -> bool:
    """Any frame fitted or scored (picks the panel's empty-state message)."""
    if state.radius is None:
        return False
    conf = state.fit_confidence
    return bool(np.any(np.asarray(state.radius) > 0)
                or (conf is not None and np.any(np.isfinite(conf))))


from bubbletrack.controller.review_install import install_review  # noqa: E402  (re-export)
