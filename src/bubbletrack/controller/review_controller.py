"""Review: which frames are likely wrong, and the actions that settle them.

The rules are pure (``review_queue`` lists frames, ``review_segments`` groups
them into rows, ``review_advice`` recommends an action).  ``ReviewController``
wires them to the running app: it refreshes on bus events and edits, keeps
the R(t) chart showing the selected row, performs the actions
(``review_actions``; Fix through ``FixMode``) and then moves on by itself: to
the row's next frame that still needs a look, and once the row is done, to
the next row.  Nothing in Review deletes a result.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDockWidget, QMenu, QVBoxLayout, QWidget

from bubbletrack.controller import review_actions, suggestion_actions
from bubbletrack.controller.fix_mode import FixMode
from bubbletrack.controller.review_advice import advise
from bubbletrack.controller.review_memory import ReviewMemory
from bubbletrack.controller.review_queue import (
    ReviewItem, build_review_items, next_after_action,
)
from bubbletrack.controller.review_segments import ReviewSegment, build_segments, next_in_segment
from bubbletrack.event_bus import RESULTS_CHANGED
from bubbletrack.model.state import AppState
from bubbletrack.ui import review_confirm
from bubbletrack.ui.fix_strip import FixStrip
from bubbletrack.ui.review_card import CardAdvice
from bubbletrack.ui.review_panel import ReviewPanel
from bubbletrack.ui.review_text import accept_all_question, span_text
from bubbletrack.ui.shortcuts import FIX_BINDINGS, ReviewShortcuts

__all__ = ["ReviewController", "ReviewItem", "build_review_items", "install_review"]

logger = logging.getLogger(__name__)

_OK, _WARN = "#10b981", "#f59e0b"
_FRAME_ACTIONS = frozenset({"accept", "fix", "no_bubble"})
NEXT_ROW, ALL_DONE = "next row", "all done"  # where an action moved on to


class ReviewController:
    """Keeps the review panel in sync with the app and performs its actions."""

    def __init__(self, window, app, panel: ReviewPanel, dock: QDockWidget,
                 strip: FixStrip) -> None:
        self.w = window
        self._app = app
        self.panel = panel
        self.dock = dock
        self.memory = ReviewMemory()  # extra items, skips, guesses seen
        self._running = False
        self._fit_run = False  # the active run is Fit All (not a re-track)
        # (row, queue order) when a fix started inside the selected row
        self._fix_context: tuple[ReviewSegment, list[int]] | None = None
        self._last_skip: tuple[int, ...] = ()  # frames the feedback's Undo un-skips
        self._last_frame: int | None = None  # where the feedback's Undo returns to
        self.fix = FixMode(app, strip, self._on_fixed)

        panel.set_advisor(self._advice)
        panel.frame_activated.connect(self._go)
        panel.action_requested.connect(self.perform)
        panel.segment_changed.connect(self._show_range)
        panel.undo_requested.connect(self._undo)
        panel.export_requested.connect(self._export)
        app.bus.subscribe("run_state_changed", self._on_run_state)
        app.bus.subscribe("source_loaded", self._on_source_loaded)
        app.bus.subscribe("frame_changed", self._on_frame_changed)
        app.bus.subscribe(RESULTS_CHANGED, lambda *_: self.refresh())

        self.shortcuts = ReviewShortcuts(window, {
            **{name: (lambda n=name: self.perform(n)) for name in ReviewPanel.ACTIONS},
            "next": lambda: panel.step(1),
            "prev": lambda: panel.step(-1),
        }, is_active=self._shortcuts_active)
        self.fix_keys = ReviewShortcuts(window, {
            "fix_cancel": self.fix.cancel, "fix_done": self.fix.finish_points,
        }, is_active=self._fix_keys_active, bindings=FIX_BINDINGS)
        strip.active_changed.connect(lambda _on: self.fix_keys.sync())
        strip.active_changed.connect(panel.card.set_fixing)
        dock.visibilityChanged.connect(self._on_dock_visibility)
        self.refresh()

    # -- public API -------------------------------------------------------- #

    @property
    def count_changed(self):
        """Signal(int): number of frames waiting for review."""
        return self.panel.count_changed

    @property
    def count(self) -> int:
        return self.panel.count()

    def set_extra(self, items: Iterable[ReviewItem]) -> None:
        """Merge another review list (e.g. the sequence solver's) and refresh.

        Each item holds for the result it was made from: once its frame gets
        another result, the item is dropped.
        """
        self.memory.set_extra(self._state(), items)
        # A (re-)track just changed results: the last action's Undo would now
        # undo the track instead, so its feedback line goes.
        self.panel.clear_feedback()
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the queue from the current results (cheap; no-op if unchanged)."""
        state = self._state()
        self.panel.set_segments(build_segments(self.memory.items(state)), state.images,
                                fitted=_has_results(state))
        self.panel.set_fix_label("Fix with a circle" if self._app.circle_ctrl.available()
                                 else "Fix with edge points")
        self.panel.refresh_card()

    def queued_frames(self) -> frozenset[int]:
        """Every frame the queue lists now (all types; skipped ones left out)."""
        return frozenset(item.frame for item in self.memory.items(self._state()))

    def perform(self, action: str) -> None:
        """Apply *action* (see ``ReviewPanel.ACTIONS``), then move on by itself.

        Accept, Fix and No bubble act on the frame on screen; Accept all,
        No bubble on all and Skip on the selected row.
        """
        if action not in ReviewPanel.ACTIONS:
            raise ValueError(f"unknown review action {action!r}")
        if self._running:
            return
        segment = self.panel.current_segment()
        frame = self.panel.current_frame() if segment is not None else self.panel.shown_frame()
        if action == "fix":
            if frame is not None:
                self.start_fix(frame)
            return
        if frame is None or (segment is None and action in ReviewPanel.ROW_ACTIONS):
            return
        self.fix.end()  # e.g. N after a refused circle: the user decided otherwise
        logger.info("Review: %s on frame %d (%s)", action, frame + 1,
                    span_text(segment) if segment is not None else "no row")
        old = self.panel.frames()
        status = self._act(action, segment, frame)
        if status is None:
            return  # nothing happened (e.g. Accept all cancelled): stay
        on_frame = action in _FRAME_ACTIONS
        moved = self._move_on(old, segment, frame if on_frame else None)
        self._report(status, moved if on_frame or moved == ALL_DONE else "",
                     acted_on=frame if on_frame else segment.start,
                     undo_skip=segment.frames if action == "skip" else ())

    def start_fix(self, frame: int) -> bool:
        """Fix *frame* (0-based) with the Review dock's fix tools.

        Shows the frame, opens the dock and starts "Drag a circle" (or
        "Click edge points" outside the Auto engine).  Once a fix is stored,
        Review moves on as after Accept when the frame lies in the selected
        row.  False if no fix can start (no images, a run active, out of
        range).  ``install_review`` connects ``RadiusChart.fix_requested``
        ("Fix frame N...") here.
        """
        state = self._state()
        if self._running:
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
        """Perform *action*; the feedback text, None if nothing happened."""
        app = self._app
        if action == "accept":
            return review_actions.accept_frame(app, frame)
        if action == "no_bubble":
            return review_actions.mark_no_bubble(app, frame, frame, f"Frame {frame + 1}")
        if action == "no_bubble_all":
            return review_actions.mark_no_bubble(app, segment.start, segment.end,
                                                 span_text(segment))
        if action == "accept_all":
            if not self._confirm_accept_all(segment):
                return None
            return review_actions.accept_segment(app, segment)
        self.memory.dismiss(self._state(), segment.frames)  # skip
        return f"{span_text(segment)} skipped: left as it is"

    def _confirm_accept_all(self, segment: ReviewSegment) -> bool:
        """Ask once if Accept all would store guesses the user has not displayed."""
        state = self._state()
        unseen = [f for f in suggestion_actions.guessed_frames(state, segment)
                  if not self.memory.was_seen(state, f)]
        if not unseen:
            return True
        return review_confirm.ask_accept_all(
            self.w, accept_all_question(segment.count, len(unseen)))

    def _move_on(self, old: list[int], segment: ReviewSegment | None,
                 frame: int | None) -> str:
        """After an action on *frame* (None: on the whole row): show the row's
        next frame that still needs a look, else the next row in queue order.

        Returns "" (same row), NEXT_ROW or ALL_DONE, for the feedback line.
        """
        self.refresh()
        if segment is None:
            return ""
        new = self.panel.frames()
        target = None if frame is None else next_in_segment(segment, frame, set(new))
        moved = ""
        if target is None:
            target = next_after_action(old, segment.end, new)
            moved = ALL_DONE if target is None else NEXT_ROW
        if target is not None and self.panel.select_frame(target, keep_row=False):
            self._go(target)
        return moved

    def _report(self, status: str, moved: str, *, acted_on: int,
                undo_skip: tuple[int, ...] = ()) -> None:
        """Say what the action did: in the panel (with Undo) and in the header
        (over the next frame's own message)."""
        self._last_skip, self._last_frame = tuple(undo_skip), acted_on
        text = status.replace(review_actions.UNDO_HINT, "")
        self.panel.show_feedback(text + {NEXT_ROW: ". That row is done.",
                                         ALL_DONE: ". All done."}.get(moved, "."))
        self.w.header.set_status(status, _OK)

    def _on_fixed(self, frame: int) -> None:
        """A fix was stored on *frame*: move on as after Accept."""
        context, self._fix_context = self._fix_context, None
        state = self._state()
        status = (f"Frame {frame + 1} fixed: R = {state.radius[frame]:.1f} px"
                  f"{review_actions.UNDO_HINT}")
        moved = self._move_on(context[1], context[0], frame) if context is not None else ""
        if context is None:
            self.panel.refresh_card()
        self._report(status, moved, acted_on=frame)

    def _undo(self) -> None:
        """The feedback line's Undo: un-skip a row, else Ctrl+Z (the newest
        edit); then show the frame it was about again."""
        frame, self._last_frame = self._last_frame, None
        if self._last_skip:
            self.memory.undismiss(self._last_skip)
            self._last_skip = ()
            self.w.header.set_status("Skip undone", _OK)
        else:
            self._app._on_undo()
        self.refresh()
        if frame is not None:
            self._return_to(frame)
        self.panel.clear_feedback()

    def _return_to(self, frame: int) -> None:
        """Show *frame* again (after an undo), with its row if it is listed."""
        self.panel.select_frame(frame, keep_row=False)
        self._go(frame)

    def _export(self) -> None:
        """'Export R(t)...' once everything is checked: the window's Export Results."""
        self.w.export_requested.emit()

    # -- helpers ----------------------------------------------------------- #

    def _state(self) -> AppState:
        return self._app._get_state()

    def _go(self, frame: int) -> None:
        self._app.file_ctrl.on_frame_changed(frame)

    def _advice(self, frame: int, item: ReviewItem | None) -> CardAdvice:
        advice = advise(self._state(), frame, item)
        return CardAdvice(advice.text, advice.action, advice.guess, advice.can_accept)

    def _show_range(self, segment: ReviewSegment | None) -> None:
        """Shade the selected run on the R(t) chart while the dock is open.

        A shading of its own, not the chart's range selection: the user's
        brushed range and the point menus inside a run stay as they were.
        """
        shown = segment is not None and not segment.is_single and not self.dock.isHidden()
        self.w.radius_chart.set_review_span((segment.start, segment.end) if shown else None)

    def _shortcuts_active(self) -> bool:
        return self.dock.isVisible() and not self._running

    def _fix_keys_active(self) -> bool:
        return self.fix.active and self.dock.isVisible() and not self._fit_run

    # -- bus events -------------------------------------------------------- #

    def _on_run_state(self, running: bool) -> None:
        was_fit = self._fit_run
        self._running = bool(running)
        self._fit_run = self._running and self._app.auto_ctrl.is_busy()
        self.panel.setEnabled(not self._running)
        # A fix in progress goes on during a re-track (storing it restarts the
        # re-track); Fit All ends it (FixMode).
        self.fix.strip.setEnabled(not self._fit_run)
        if not self._running:
            self.refresh()
            if was_fit:  # new results: progress counts from here
                self.panel.reset_progress()
                self.panel.clear_feedback()
        self.shortcuts.sync()
        self.fix_keys.sync()

    def _on_dock_visibility(self, _visible: bool) -> None:
        self.shortcuts.sync()
        self.fix_keys.sync()
        self._show_range(self.panel.current_segment())
        if self.fix.active and not self.dock.isVisible() and not self.w.isMinimized():
            self.fix.cancel()  # its strip is out of sight

    def _on_source_loaded(self, _n: int) -> None:
        self.memory.clear()
        self._fix_context = None
        self._last_skip = ()
        self.panel.clear_feedback()
        self.refresh()
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


def _view_menu(window) -> QMenu:
    menubar = window.menuBar()
    for action in menubar.actions():
        menu = action.menu()
        if menu is not None and action.text().replace("&", "") == "View":
            return menu
    return menubar.addMenu("&View")


def install_review(window, app_controller) -> ReviewController:
    """Add the Review dock (and a View menu toggle) to *window*.

    Idempotent: the app controller installs it at start-up, and a second
    call returns that instance instead of adding another dock and key set.
    """
    existing = getattr(app_controller, "review", None)
    if isinstance(existing, ReviewController):
        return existing
    panel, strip = ReviewPanel(), FixStrip()
    body = QWidget()
    body.setObjectName("ReviewBody")
    layout = QVBoxLayout(body)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(panel, 1)
    layout.addWidget(strip)  # outside the panel: usable while a re-track locks the list
    dock = QDockWidget("Review", window)
    dock.setObjectName("ReviewDock")
    # Not floatable: the review keys are window shortcuts of the main window.
    dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable
                     | QDockWidget.DockWidgetFeature.DockWidgetMovable)
    dock.setWidget(body)
    window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
    toggle = dock.toggleViewAction()
    toggle.setText("&Review Queue")
    _view_menu(window).addAction(toggle)
    review = ReviewController(window, app_controller, panel, dock, strip)
    # The chart's "Fix frame N..." (a point's right-click menu) opens the same fix.
    window.radius_chart.fix_requested.connect(review.start_fix)
    return review
