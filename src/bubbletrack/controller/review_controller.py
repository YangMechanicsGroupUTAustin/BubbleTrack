"""Review queue: which frames are likely wrong, and one-key actions to fix them.

The rules are pure (``review_queue`` lists frames, ``review_segments`` groups
them into runs).  ``ReviewController`` wires the queue to the running app: it
refreshes on bus events and edits, keeps the R(t) chart showing the selected
run, and routes each action to the existing AppController handlers.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDockWidget, QMenu

from bubbletrack.controller import suggestion_actions
from bubbletrack.controller.review_queue import (
    ReviewItem, build_review_items, next_after_action,
)
from bubbletrack.controller.review_segments import (
    ReviewSegment, anchor_frame, build_segments,
)
from bubbletrack.event_bus import CIRCLE_FIXED, RESULTS_CHANGED
from bubbletrack.model.state import AppState
from bubbletrack.ui.review_panel import ReviewPanel
from bubbletrack.ui.review_text import DASH
from bubbletrack.ui.shortcuts import ReviewShortcuts
from bubbletrack.ui.tab_bar import MANUAL

logger = logging.getLogger(__name__)

MANUAL_TAB = MANUAL  # TabBar name of the Manual tab
_OK = "#10b981"


def span_text(segment: ReviewSegment) -> str:
    """'Frame 5' or 'Frames 10–44' (1-based), for status messages."""
    if segment.is_single:
        return f"Frame {segment.start + 1}"
    return f"Frames {segment.start + 1}{DASH}{segment.end + 1}"


class ReviewController:
    """Keeps the review panel in sync with the app and performs its actions.

    Accept, Refit, Delete and Skip act on the whole selected row (segment);
    Neighbours, Circle and Manual act on the frame on screen.
    """

    def __init__(self, window, app, panel: ReviewPanel, dock: QDockWidget) -> None:
        self.w = window
        self._app = app
        self.panel = panel
        self.dock = dock
        self._extra: tuple[ReviewItem, ...] = ()
        self._extra_keys: dict[int, tuple] = {}  # frame -> result the item was made for
        self._dismissed: dict[int, tuple] = {}  # frame -> result it was skipped with
        self._running = False
        # (frame, its edge points before, queue order, anchor) while a Manual fit runs
        self._manual_pending: tuple[int, object, list[int], int] | None = None
        # (frame, queue order, anchor) while the circle tool runs from Circle (C)
        self._circle_pending: tuple[int, list[int], int] | None = None
        app.bus.subscribe(CIRCLE_FIXED, self._on_circle_fixed)

        panel.frame_activated.connect(self._go)
        panel.action_requested.connect(self.perform)
        panel.segment_changed.connect(self._show_range)
        app.bus.subscribe("run_state_changed", self._on_run_state)
        app.bus.subscribe("source_loaded", self._on_source_loaded)
        app.bus.subscribe("frame_changed", self._on_frame_changed)
        self._refresh_after_edits()

        self.shortcuts = ReviewShortcuts(window, {
            **{name: (lambda n=name: self.perform(n)) for name in ReviewPanel.ACTIONS},
            "next": lambda: panel.step(1),
            "prev": lambda: panel.step(-1),
        }, is_active=self._shortcuts_active)
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

        Each item holds for the result it was made from: once its frame is
        refitted, fixed or deleted, the item is dropped.
        """
        state = self._state()
        n = 0 if state.radius is None else len(state.radius)
        self._extra = tuple(item for item in items if 0 <= item.frame < n)
        self._extra_keys = {item.frame: self._result_key(state, item.frame)
                            for item in self._extra}
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the queue from the current results (cheap; no-op if unchanged)."""
        state = self._state()
        self.panel.set_segments(build_segments(self._current_items()), state.images,
                                fitted=_has_results(state))

    def _current_items(self) -> list[ReviewItem]:
        """The queue's items for the current results (as the dock lists them)."""
        state = self._state()
        extra = [item for item in self._extra if state.radius is not None
                 and item.frame < len(state.radius)
                 and self._extra_keys.get(item.frame) == self._result_key(state, item.frame)]
        return [item for item in build_review_items(state, extra)
                if not self._is_dismissed(state, item.frame)]

    def perform(self, action: str) -> None:
        """Apply *action* (see ``ReviewPanel.ACTIONS``) to the selected row,
        then show the next one (Manual advances after Done instead)."""
        if action not in ReviewPanel.ACTIONS:
            raise ValueError(f"unknown review action {action!r}")
        segment, frame = self.panel.current_segment(), self.panel.current_frame()
        if self._running or segment is None or frame is None:
            return
        logger.info("Review: %s %s (frame %d shown)", action, span_text(segment), frame + 1)
        anchor = anchor_frame(segment, frame)  # the queue continues after it
        if action == "manual":
            self._start_manual(frame, anchor)
            return
        if action == "circle":
            self._start_circle(frame, anchor)
            return
        old = self.panel.frames()
        if action == "refit_neighbors":
            self._refit_neighbors(frame)
        elif getattr(self, f"_{action}")(segment):
            anchor = segment.end  # the whole row is done: continue after it
        else:
            return  # nothing happened (e.g. the range fit was cancelled): stay
        self._advance(old, anchor)

    # -- actions on a row (each returns False if it did nothing) ----------- #

    def _accept(self, segment: ReviewSegment) -> bool:
        state = self._state()
        locked = state.fit_locked
        if locked is None:
            return False
        # A frame showing Auto's guess gets the guess as its result (an
        # undoable edit); every other listed frame is locked as it is.
        promotion = suggestion_actions.promote_guesses(state, segment)
        # Every listed frame, with or without a result ("no bubble here"), as
        # a single-frame Accept does; the range lock adds the span's other
        # fitted frames and updates the chart and the autosave.
        locked[list(segment.frames)] = True  # provenance (fit_source) is left as it was
        self._app._on_lock_range(segment.start, segment.end, locked=True)
        text = f"{span_text(segment)} accepted and locked"
        if promotion is not None:
            text = suggestion_actions.finish_promotion(self._app, promotion, text)
        self.w.header.set_status(text, _OK)
        return True

    def _refit(self, segment: ReviewSegment) -> bool:
        if segment.is_single:
            self._app._on_refit_point(segment.start)
            return True
        # In the background, like the chart's "Refit range"; a started run
        # announces itself (run_state_changed) before fit_range returns.
        self._app.auto_ctrl.fit_range(segment.start, segment.end)
        return self._running

    def _delete(self, segment: ReviewSegment) -> bool:
        if segment.is_single:
            self._app._on_delete_point(segment.start)
        else:
            self._app._on_delete_range(segment.start, segment.end)
        # The user has decided: keep these frames out of the queue until
        # they get a new result.
        for frame in segment.frames:
            self._dismiss(frame)
        return True

    def _skip(self, segment: ReviewSegment) -> bool:
        for frame in segment.frames:
            self._dismiss(frame)
        self.w.header.set_status(f"{span_text(segment)} skipped", _OK)
        return True

    # -- actions on the shown frame ---------------------------------------- #

    def _refit_neighbors(self, frame: int) -> None:
        self._app._on_refit_point_with_neighbors(frame)

    def _open_manual_tab(self, frame: int) -> None:
        """Show *frame* with the Manual tab open."""
        self._go(frame)  # navigate first: a frame change cancels a selection
        lp = self.w.left_panel
        if lp.isHidden():  # the Manual tab's controls live in the sidebar
            toggle = getattr(self.w, "_toggle_sidebar", None)
            if toggle is not None:
                toggle()
        lp.tab_bar.select(MANUAL_TAB)

    def _start_manual(self, frame: int, anchor: int) -> None:
        """Manual tab on *frame* with point selection running."""
        self._open_manual_tab(frame)
        self._app.manual_ctrl.on_manual_select()
        xy = self._state().circle_xy
        self._manual_pending = (frame, xy[frame] if xy is not None else None,
                                self.panel.frames(), anchor)

    def _on_manual_done(self) -> None:
        """Runs after ManualController: advance once the manual fit is stored."""
        pending, self._manual_pending = self._manual_pending, None
        state = self._state()
        if pending is None or state.circle_xy is None or state.image_no != pending[0]:
            self.refresh()
            return
        frame, before, order, anchor = pending
        if state.circle_xy[frame] is before:  # not committed (e.g. < 3 points)
            self._manual_pending = pending
            self.refresh()
            return
        # The queue was already refreshed by RESULTS_CHANGED: advance by the
        # order it had when the Manual fit started.
        self._advance(order, anchor)

    # -- Circle (C): the Manual tab's "Drag a circle" tool ----------------- #

    def _start_circle(self, frame: int, anchor: int) -> None:
        """Manual tab on *frame* with the circle tool; advance once a circle is stored."""
        circle = self._app.circle_ctrl
        if not circle.available():
            circle.explain_unavailable()
            return
        self._open_manual_tab(frame)
        circle.activate()
        self._circle_pending = (frame, self.panel.frames(), anchor)

    def _on_circle_fixed(self, frame: int) -> None:
        pending, self._circle_pending = self._circle_pending, None
        if pending is None or pending[0] != frame:
            return
        self._advance(pending[1], pending[2])

    def queued_frames(self) -> frozenset[int]:
        """Every frame the queue lists now (all types; skipped ones left out)."""
        return frozenset(item.frame for item in self._current_items())

    # -- helpers ----------------------------------------------------------- #

    def _state(self) -> AppState:
        return self._app._get_state()

    def _go(self, frame: int) -> None:
        self._app.file_ctrl.on_frame_changed(frame)

    def _advance(self, old: list[int], anchor: int) -> None:
        self.refresh()
        target = next_after_action(old, anchor, self.panel.frames())
        if target is not None and self.panel.select_frame(target, keep_row=False):
            self._go(target)

    def _show_range(self, segment: ReviewSegment | None) -> None:
        """Shade the selected run on the R(t) chart while the dock is open.

        A shading of its own, not the chart's range selection: the user's
        brushed range and the point menus inside a run stay as they were.
        """
        shown = segment is not None and not segment.is_single and not self.dock.isHidden()
        self.w.radius_chart.set_review_span((segment.start, segment.end) if shown else None)

    @staticmethod
    def _result_key(state: AppState, frame: int) -> tuple:
        conf = state.fit_confidence
        c = float(conf[frame]) if conf is not None else float("nan")
        source = int(state.fit_source[frame]) if state.fit_source is not None else 0
        return float(state.radius[frame]), (c if np.isfinite(c) else None), source

    def _dismiss(self, frame: int) -> None:
        self._dismissed[frame] = self._result_key(self._state(), frame)

    def _is_dismissed(self, state: AppState, frame: int) -> bool:
        """Skipped/deleted frames stay hidden until their result changes."""
        key = self._dismissed.get(frame)
        return key is not None and key == self._result_key(state, frame)

    def _shortcuts_active(self) -> bool:
        return self.dock.isVisible() and not self._running

    def _refresh_after_edits(self) -> None:
        """Edits made anywhere (chart menu, manual tab, Clear, refits)."""
        self._app.bus.subscribe(RESULTS_CHANGED, lambda *_: self.refresh())
        # Runs after ManualController's slot (connected earlier).
        self.w.left_panel.manual_tab.done_clicked.connect(self._on_manual_done)

    # -- bus events -------------------------------------------------------- #

    def _on_run_state(self, running: bool) -> None:
        self._running = bool(running)
        self.panel.setEnabled(not self._running)
        if not self._running:
            self.refresh()
        self.shortcuts.sync()

    def _on_dock_visibility(self, _visible: bool) -> None:
        self.shortcuts.sync()
        self._show_range(self.panel.current_segment())

    def _on_source_loaded(self, _n: int) -> None:
        self._dismissed.clear()
        self._extra = ()
        self._extra_keys = {}
        self._manual_pending = None
        self._circle_pending = None
        self.refresh()

    def _on_frame_changed(self, idx: int) -> None:
        if self._manual_pending is not None and self._manual_pending[0] != idx:
            self._manual_pending = None
        if self._circle_pending is not None and self._circle_pending[0] != idx:
            self._circle_pending = None
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
    panel = ReviewPanel()
    dock = QDockWidget("Review", window)
    dock.setObjectName("ReviewDock")
    # Not floatable: the review keys are window shortcuts of the main window.
    dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable
                     | QDockWidget.DockWidgetFeature.DockWidgetMovable)
    dock.setWidget(panel)
    window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
    toggle = dock.toggleViewAction()
    toggle.setText("&Review Queue")
    _view_menu(window).addAction(toggle)
    return ReviewController(window, app_controller, panel, dock)
