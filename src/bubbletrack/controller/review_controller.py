"""Review queue: which frames are likely wrong, and one-key actions to fix them.

``build_review_items`` is pure (state in, sorted items out).  ``ReviewController``
wires the queue to the running app: it refreshes on bus events and edits, and
routes each action to the existing AppController handlers.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDockWidget, QMenu

from bubbletrack.event_bus import RESULTS_CHANGED
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.sequence import CONTEXT_FLAGS
from bubbletrack.model.state import AppState
from bubbletrack.ui.review_panel import ReviewEntry, ReviewPanel, item_text
from bubbletrack.ui.shortcuts import ReviewShortcuts
from bubbletrack.ui.tab_bar import MANUAL

logger = logging.getLogger(__name__)

LOW_CONFIDENCE = 0.5  # engine confidence below this is worth a look
JUMP_FRACTION = 0.15  # |dR| to both valid neighbours, relative to each neighbour

# Higher = more likely wrong.  Engine and sequence flags share these keys.
REASON_WEIGHTS: dict[str, int] = {
    "lost": 100,
    "rejected": 90,
    "disagree": 85,
    "jump": 80,
    "low_confidence": 60,  # rises to 90 as the confidence falls to 0
    "multiple_edges": 60,
    "non_circular": 55,
    "fragmented": 50,
    "clipped_completed": 45,
    "reacquired": 40,
    "collapse": 35,
    "cloud": 30,
    "shoulder": 25,
}
UNKNOWN_REASON_WEIGHT = 50  # reasons that only arrive through ``extra``
REVIEW_FLAGS = frozenset(REASON_WEIGHTS) - {"rejected"}
# Routine engine flags: they describe a listed frame and add to its rank, but
# never list a frame on their own (see the sequence solver's CONTEXT_FLAGS).
CONTEXT_REASONS = CONTEXT_FLAGS

MANUAL_TAB = MANUAL  # TabBar name of the Manual tab
_OK, _WARN = "#10b981", "#f59e0b"


@dataclass(frozen=True)
class ReviewItem:
    frame: int  # 0-based
    reasons: tuple[str, ...]  # most severe first
    severity: int  # higher = more likely wrong
    confidence: float = float("nan")  # engine confidence, NaN if not scored


# --------------------------------------------------------------------------- #
#  Pure queue logic
# --------------------------------------------------------------------------- #

def build_review_items(state: AppState, extra: Iterable[ReviewItem] | None = None,
                       ) -> list[ReviewItem]:
    """Frames worth a second look, most likely wrong first, then by frame.

    *extra* (e.g. the sequence solver's list) is merged per frame.  Locked
    frames (accepted in review) are never listed.
    """
    if state.radius is None:
        return []
    radius = np.asarray(state.radius, dtype=float)
    n = len(radius)
    conf = _per_frame(state.fit_confidence, n, np.nan, float)
    locked = _per_frame(state.fit_locked, n, False, bool)
    own = _frame_reasons(state, radius, conf)
    added = _merge_extra(extra, n)
    items = [_make_item(f, own.get(f, ()), added.get(f), float(conf[f]))
             for f in sorted(set(own) | set(added)) if not locked[f]]
    items = [item for item in items if not CONTEXT_REASONS.issuperset(item.reasons)]
    return sorted(items, key=lambda item: (-item.severity, item.frame))


def jump_frames(radius: np.ndarray, threshold: float = JUMP_FRACTION) -> np.ndarray:
    """Frames that spike above or dip below BOTH nearest valid neighbours.

    A steady fast change (collapse, rebound) is not a jump: each frame there
    lies between its neighbours.
    """
    radius = np.asarray(radius, dtype=float)
    valid = np.flatnonzero(np.isfinite(radius) & (radius > 0))
    if len(valid) < 3:
        return np.empty(0, dtype=int)
    r = radius[valid]
    mid, prev, nxt = r[1:-1], r[:-2], r[2:]
    up = (mid > prev * (1 + threshold)) & (mid > nxt * (1 + threshold))
    down = (mid < prev * (1 - threshold)) & (mid < nxt * (1 - threshold))
    return valid[1:-1][up | down]


def next_after_action(old: list[int], frame: int, new: list[int]) -> int | None:
    """Frame to show after acting on *frame*: the next old item still listed."""
    remaining = set(new)
    if frame in old:
        pos = old.index(frame)
        candidates = old[pos + 1:] + old[:pos]
    else:
        candidates = old
    for f in candidates:
        if f in remaining and f != frame:
            return f
    return next((f for f in new if f != frame), new[0] if new else None)


def _per_frame(values, n: int, fill, dtype) -> np.ndarray:
    if values is None or len(values) != n:
        return np.full(n, fill, dtype=dtype)
    return np.asarray(values, dtype=dtype)


def _frame_reasons(state: AppState, radius: np.ndarray, conf: np.ndarray,
                   ) -> dict[int, list[str]]:
    n = len(radius)
    source = _per_frame(state.fit_source, n, FitSource.NONE, int)
    manual = source == FitSource.MANUAL
    # A manual fix supersedes the engine's verdict on that frame.
    engine = np.isfinite(conf) & ~manual
    fitted = np.isfinite(radius) & (radius > 0)
    found: dict[int, list[str]] = defaultdict(list)

    def add(i, reason: str) -> None:
        if reason not in found[int(i)]:
            found[int(i)].append(reason)

    for i in np.flatnonzero(engine & ~fitted):
        add(i, "rejected")
    for i in np.flatnonzero(engine & (conf < LOW_CONFIDENCE)):
        add(i, "low_confidence")
    for i, flags in enumerate((state.fit_flags or ())[:n]):
        if flags and not manual[i]:
            for reason in sorted(flags & REVIEW_FLAGS):
                add(i, reason)
    unscored = ~np.isfinite(conf)  # legacy modes produce no flags of their own
    for i in jump_frames(radius):
        if unscored[i]:
            add(i, "jump")
    return found


def _merge_extra(extra: Iterable[ReviewItem] | None, n: int) -> dict[int, ReviewItem]:
    merged: dict[int, ReviewItem] = {}
    for item in extra or ():
        if not 0 <= item.frame < n:
            continue
        prev = merged.get(item.frame)
        if prev is None:
            merged[item.frame] = item
        else:
            reasons = prev.reasons + tuple(r for r in item.reasons if r not in prev.reasons)
            merged[item.frame] = ReviewItem(item.frame, reasons,
                                            max(prev.severity, item.severity))
    return merged


def _weight(reason: str, confidence: float) -> int:
    if reason == "low_confidence" and np.isfinite(confidence):
        shortfall = (LOW_CONFIDENCE - min(max(confidence, 0.0), LOW_CONFIDENCE)) / LOW_CONFIDENCE
        return REASON_WEIGHTS[reason] + round(30 * shortfall)
    return REASON_WEIGHTS.get(reason, UNKNOWN_REASON_WEIGHT)


def _make_item(frame: int, own: Iterable[str], extra: ReviewItem | None,
               confidence: float) -> ReviewItem:
    reasons = list(own)
    if extra is not None:
        reasons += [r for r in extra.reasons if r not in reasons]
    ordered = tuple(sorted(reasons, key=lambda r: -_weight(r, confidence)))
    worst = max((_weight(r, confidence) for r in ordered), default=0)
    severity = worst + max(len(ordered) - 1, 0)  # more reasons rank higher
    if extra is not None:
        severity = max(severity, extra.severity)
    return ReviewItem(frame, ordered, severity, confidence)


# --------------------------------------------------------------------------- #
#  Qt wiring
# --------------------------------------------------------------------------- #

class ReviewController:
    """Keeps the review panel in sync with the app and performs its actions."""

    def __init__(self, window, app, panel: ReviewPanel, dock: QDockWidget) -> None:
        self.w = window
        self._app = app
        self.panel = panel
        self.dock = dock
        self._extra: tuple[ReviewItem, ...] = ()
        self._extra_keys: dict[int, tuple] = {}  # frame -> result the item was made for
        self._dismissed: dict[int, tuple] = {}  # frame -> result it was skipped with
        self._running = False
        # (frame, its edge points before, queue order) while a Manual fit runs
        self._manual_pending: tuple[int, object, list[int]] | None = None

        panel.frame_activated.connect(self._go)
        panel.action_requested.connect(self.perform)
        app.bus.subscribe("run_state_changed", self._on_run_state)
        app.bus.subscribe("source_loaded", self._on_source_loaded)
        app.bus.subscribe("frame_changed", self._on_frame_changed)
        self._refresh_after_edits()

        self.shortcuts = ReviewShortcuts(window, {
            **{name: (lambda n=name: self.perform(n)) for name in ReviewPanel.ACTIONS},
            "next": lambda: panel.step(1),
            "prev": lambda: panel.step(-1),
        }, is_active=self._shortcuts_active)
        dock.visibilityChanged.connect(lambda _visible: self.shortcuts.sync())
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
        extra = [item for item in self._extra if state.radius is not None
                 and item.frame < len(state.radius)
                 and self._extra_keys.get(item.frame) == self._result_key(state, item.frame)]
        items = [item for item in build_review_items(state, extra)
                 if not self._is_dismissed(state, item.frame)]
        self.panel.set_entries([ReviewEntry(item.frame, item_text(item, state.images),
                                            item.severity) for item in items])

    def perform(self, action: str) -> None:
        """Apply *action* (see ``ReviewPanel.ACTIONS``) to the selected item,
        then show the next item (Manual advances after Done instead)."""
        if action not in ReviewPanel.ACTIONS:
            raise ValueError(f"unknown review action {action!r}")
        frame = self.panel.current_frame()
        if self._running or frame is None:
            return
        logger.info("Review: %s frame %d", action, frame + 1)
        if action == "manual":
            self._start_manual(frame)
            return
        old = self.panel.frames()
        getattr(self, f"_{action}")(frame)
        self._advance(old, frame)

    # -- actions ----------------------------------------------------------- #

    def _refit(self, frame: int) -> None:
        self._app._on_refit_point(frame)

    def _refit_neighbors(self, frame: int) -> None:
        self._app._on_refit_point_with_neighbors(frame)

    def _accept(self, frame: int) -> None:
        locked = self._state().fit_locked
        if locked is None:
            return
        locked[frame] = True  # provenance (fit_source) is left as it was
        self.w.header.set_status(f"Frame {frame + 1} accepted and locked", _OK)

    def _delete(self, frame: int) -> None:
        self._app._on_delete_point(frame)
        # The deleted frame keeps its confidence, so it would come back as
        # "rejected"; the user has decided, so keep it out of the queue.
        self._dismiss(frame)

    def _skip(self, frame: int) -> None:
        self._dismiss(frame)
        self.w.header.set_status(f"Frame {frame + 1} skipped", _OK)

    def _start_manual(self, frame: int) -> None:
        """Manual tab on *frame* with point selection running."""
        self._go(frame)  # navigate first: a frame change cancels a selection
        lp = self.w.left_panel
        if lp.isHidden():  # the Done button lives in the sidebar
            toggle = getattr(self.w, "_toggle_sidebar", None)
            if toggle is not None:
                toggle()
        lp.tab_bar.select(MANUAL_TAB)
        self._app.manual_ctrl.on_manual_select()
        xy = self._state().circle_xy
        self._manual_pending = (frame, xy[frame] if xy is not None else None,
                                self.panel.frames())

    def _on_manual_done(self) -> None:
        """Runs after ManualController: advance once the manual fit is stored."""
        pending, self._manual_pending = self._manual_pending, None
        state = self._state()
        if pending is None or state.circle_xy is None or state.image_no != pending[0]:
            self.refresh()
            return
        frame, before, order = pending
        if state.circle_xy[frame] is before:  # not committed (e.g. < 3 points)
            self._manual_pending = pending
            self.refresh()
            return
        # The queue was already refreshed by RESULTS_CHANGED: advance by the
        # order it had when the Manual fit started.
        self._advance(order, frame)

    # -- helpers ----------------------------------------------------------- #

    def _state(self) -> AppState:
        return self._app._get_state()

    def _go(self, frame: int) -> None:
        self._app.file_ctrl.on_frame_changed(frame)

    def _advance(self, old: list[int], frame: int) -> None:
        self.refresh()
        target = next_after_action(old, frame, self.panel.frames())
        if target is not None and self.panel.select_frame(target):
            self._go(target)

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

    def _on_source_loaded(self, _n: int) -> None:
        self._dismissed.clear()
        self._extra = ()
        self._extra_keys = {}
        self._manual_pending = None
        self.refresh()

    def _on_frame_changed(self, idx: int) -> None:
        if self._manual_pending is not None and self._manual_pending[0] != idx:
            self._manual_pending = None
        if not self._running:
            self.refresh()
        self.panel.select_frame(idx)  # clears the selection off the queue


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
